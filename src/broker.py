import math
import os
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

from storage import PLANS_DIR, buy_order_id, exit_order_id, late_exit_order_id, origin_session

# Load API Keys
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the Trading Client (paper=True is critical: it always targets paper-api.alpaca.markets)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Fund size (matches the Alpaca paper account) and the budget per approved stock: 10% of the fund.
# Up to 5 targets a day means at most 50% of the fund is invested at once.
# We buy whole shares, so the real allocation is floor(budget / price) shares, slightly under the budget.
FUND_SIZE_USD = 100_000.00
POSITION_FRACTION = 0.10
POSITION_SIZE_USD = FUND_SIZE_USD * POSITION_FRACTION

# Order states that will not change any more.
FINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}

# How long the exit job waits for its sells to fill before reporting a failure.
FILL_TIMEOUT_S = 60


def status_of(order) -> str:
    return getattr(order.status, "value", str(order.status))


def existing_order(client_order_id: str, client=None):
    """The order already submitted under this client id, or None."""
    try:
        return (client or trading_client).get_order_by_client_id(client_order_id)
    except APIError as e:
        if e.status_code == 404:
            return None
        raise


def execute_morning_trades(plan: dict, client=None, skip_tickers=()) -> None:
    """
    Queue one market buy per approved target, placed before the open.

    A market DAY order sent pre-market executes as soon as regular trading
    starts at 09:30 ET. (Opening-auction OPG orders were tried first, but the
    paper account filled 3 of 5 with zero shares.) Each order uses a
    deterministic client_order_id, so a second morning attempt finds the
    existing order instead of buying again. Raises if any order could not be placed.

    skip_tickers: stocks whose leftover position is being sold this morning.
    Buying them too would be rejected by Alpaca's wash-trade protection.
    """
    client = client or trading_client
    print("--- STARTING ALPACA BROKER EXECUTION ---")
    if not plan["targets"]:
        print("No targets approved today. Staying in cash.")
        return

    failures = []
    for target in plan["targets"]:
        ticker = target["ticker"]
        if ticker in skip_tickers:
            print(f"⏭️  SKIP {ticker}: a leftover position is being sold this morning (wash-trade protection)")
            target["skip_reason"] = "leftover position being closed"
            continue
        qty = math.floor(POSITION_SIZE_USD / target["ref_price"])
        if qty < 1:
            print(f"⏭️  SKIP {ticker}: one share (${target['ref_price']:.2f}) exceeds the ${POSITION_SIZE_USD:.0f} budget")
            target["skip_reason"] = "price above budget"
            continue
        try:
            cid = buy_order_id(plan["session_date"], ticker)
            order = existing_order(cid, client)
            if order is not None:
                print(f"↩️  ALREADY SUBMITTED: {ticker} (Order ID: {order.id})")
            else:
                order = client.submit_order(order_data=MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=cid,
                ))
                print(f"✅ QUEUED FOR THE OPEN: buy {qty} {ticker} (~${qty * target['ref_price']:,.2f}, Order ID: {order.id})")
            target["qty"] = qty
            target["order_id"] = str(order.id)
        except Exception as e:
            print(f"❌ ERROR submitting order for {ticker}: {e}")
            failures.append(ticker)

    if failures:
        raise RuntimeError(f"Orders failed for: {', '.join(failures)}")


def _held_qty(client, ticker: str) -> int:
    try:
        return int(float(client.get_open_position(ticker).qty))
    except APIError as e:
        if e.status_code == 404:
            return 0
        raise


def _settle_buy(client, buy):
    """If the buy is still open (e.g. a partial fill), cancel the rest so a sell isn't a wash trade."""
    if status_of(buy) in FINAL_STATUSES:
        return buy
    client.cancel_order_by_id(buy.id)
    for _ in range(10):
        buy = client.get_order_by_id(buy.id)
        if status_of(buy) in FINAL_STATUSES:
            break
        time.sleep(1)
    return buy


def place_exit_orders(plan: dict, client=None, sleep=time.sleep) -> bool:
    """
    Sell every position bought this session with a market order shortly before the close,
    then wait until each sell has FILLED. (Closing-auction CLS orders were tried first, but
    the paper account let them expire unfilled.)

    Only sells shares the buy actually filled AND that are still held, so it can never open
    a short position (shorting is enabled on the account). Idempotent: a re-run finds the
    existing sell by its client_order_id. Returns True only when every held position is sold.
    """
    client = client or trading_client
    pending, all_ok = [], True
    for target in plan["targets"]:
        if not target.get("order_id"):
            continue
        ticker = target["ticker"]
        cid = exit_order_id(plan["session_date"], ticker)
        try:
            sell = existing_order(cid, client)
            if sell is not None:
                print(f"↩️  {ticker}: exit order already placed ('{status_of(sell)}')")
            else:
                buy = _settle_buy(client, client.get_order_by_id(target["order_id"]))
                bought = int(float(buy.filled_qty or 0))
                if bought == 0:
                    print(f"⏭️  {ticker}: buy ended as '{status_of(buy)}' with no fill, nothing to sell")
                    continue
                qty = min(bought, _held_qty(client, ticker))
                if qty <= 0:
                    print(f"⏭️  {ticker}: no shares held any more, nothing to sell")
                    continue
                sell = client.submit_order(order_data=MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=cid,
                ))
                print(f"📤 SELL SENT: {qty} {ticker} at market (Order ID: {sell.id})")
            pending.append((ticker, sell.id))
        except Exception as e:
            print(f"❌ ERROR placing exit for {ticker}: {e}")
            all_ok = False

    # Accepted is not filled: confirm every sell actually executed.
    waited = 0
    while pending and waited < FILL_TIMEOUT_S:
        still = []
        for ticker, oid in pending:
            o = client.get_order_by_id(oid)
            if status_of(o) == "filled":
                print(f"✅ SOLD: {o.filled_qty} {ticker} at ${float(o.filled_avg_price):,.2f}")
            elif status_of(o) in FINAL_STATUSES:
                print(f"❌ {ticker}: exit order ended as '{status_of(o)}' (filled {o.filled_qty})")
                all_ok = False
            else:
                still.append((ticker, oid))
        pending = still
        if pending:
            sleep(5)
            waited += 5
    for ticker, _ in pending:
        print(f"❌ {ticker}: exit order not filled after {FILL_TIMEOUT_S}s")
        all_ok = False
    return all_ok


def unsold_positions(plan: dict, client=None) -> list:
    """Tickers from this session's plan that are still held (used after the close to raise an alert)."""
    client = client or trading_client
    return [t["ticker"] for t in plan["targets"] if t.get("order_id") and _held_qty(client, t["ticker"]) > 0]


def close_leftover_positions(client=None, plans_dir=PLANS_DIR):
    """
    Safety net, run before the morning buys: sell any position left over from an earlier
    session whose exit failed. The market sell executes at the open.

    The sell is tagged "<session>-<ticker>-late-exit" with the session that bought it, so the
    evaluator can replace that session's estimated exit with this real fill.
    Returns (symbols being closed, symbols that could not be closed).
    """
    client = client or trading_client
    closed, failed = [], []
    for pos in client.get_all_positions():
        try:
            origin = origin_session(pos.symbol, plans_dir)
            qty = abs(float(pos.qty))
            if origin and qty == int(qty) and existing_order(late_exit_order_id(origin, pos.symbol), client) is None:
                client.submit_order(order_data=MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=int(qty),
                    side=OrderSide.SELL if float(pos.qty) > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=late_exit_order_id(origin, pos.symbol),
                ))
                print(f"🧹 Leftover from {origin}: queued a sell of {pos.qty} {pos.symbol} for the open")
            else:
                client.close_position(pos.symbol)  # not the bot's position (or already tagged): plain close
                print(f"🧹 Leftover position: queued a close of {pos.qty} {pos.symbol} for the open")
            closed.append(pos.symbol)
        except Exception as e:
            print(f"❌ Could not close leftover {pos.symbol}: {e}")
            failed.append(pos.symbol)
    return closed, failed
