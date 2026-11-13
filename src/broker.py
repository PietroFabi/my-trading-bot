import math
import os

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

from storage import buy_order_id, exit_order_id

# Load API Keys
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the Trading Client (paper=True is critical: it always targets paper-api.alpaca.markets)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Fund size (matches the Alpaca paper account) and the budget per approved stock: 10% of the fund.
# Up to 5 targets a day means at most 50% of the fund is invested at once.
# We buy whole shares (auction orders don't accept fractions), so the real allocation is
# floor(budget / price) shares, slightly under the budget.
FUND_SIZE_USD = 100_000.00
POSITION_FRACTION = 0.10
POSITION_SIZE_USD = FUND_SIZE_USD * POSITION_FRACTION

# Order states that will not change any more.
FINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


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


def execute_morning_trades(plan: dict, client=None) -> None:
    """
    Queue one market-on-open (OPG) buy per approved target.

    OPG orders execute only in the opening auction, so the fill is the real
    opening price. Each order uses a deterministic client_order_id, so a second
    morning attempt finds the existing order instead of buying again.
    Raises if any order could not be placed.
    """
    client = client or trading_client
    print("--- STARTING ALPACA BROKER EXECUTION ---")
    if not plan["targets"]:
        print("No targets approved today. Staying in cash.")
        return

    failures = []
    for target in plan["targets"]:
        ticker = target["ticker"]
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
                    time_in_force=TimeInForce.OPG,
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


def place_exit_orders(plan: dict, client=None) -> bool:
    """
    Queue a market-on-close (CLS) sell for every position bought this session.

    Only sells shares that the buy actually filled AND that are still held,
    so it can never open a short position (shorting is enabled on the account).
    Returns True when every filled buy has an exit order.
    """
    client = client or trading_client
    all_ok = True
    for target in plan["targets"]:
        if not target.get("order_id"):
            continue
        ticker = target["ticker"]
        cid = exit_order_id(plan["session_date"], ticker)
        try:
            if existing_order(cid, client) is not None:
                print(f"↩️  {ticker}: exit order already queued")
                continue

            buy = client.get_order_by_id(target["order_id"])
            bought = int(float(buy.filled_qty or 0))
            if bought == 0:
                if status_of(buy) in FINAL_STATUSES:
                    print(f"⏭️  {ticker}: buy ended as '{status_of(buy)}' with no fill, nothing to sell")
                else:
                    print(f"⏳ {ticker}: buy not filled yet ('{status_of(buy)}'), a later attempt will retry")
                    all_ok = False
                continue

            try:
                held = int(float(client.get_open_position(ticker).qty))
            except APIError as e:
                if e.status_code != 404:
                    raise
                held = 0
            qty = min(bought, held)
            if qty <= 0:
                print(f"⏭️  {ticker}: no shares held any more, nothing to sell")
                continue

            order = client.submit_order(order_data=MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.CLS,
                client_order_id=cid,
            ))
            print(f"✅ QUEUED FOR THE CLOSE: sell {qty} {ticker} (Order ID: {order.id})")
        except Exception as e:
            print(f"❌ ERROR placing exit for {ticker}: {e}")
            all_ok = False
    return all_ok


def close_leftover_positions(client=None) -> list:
    """
    Safety net, run before the morning buys: close any position left over from an
    earlier session whose closing order was missed. The sell is queued for the open.
    Returns the symbols it could not close.
    """
    client = client or trading_client
    failed = []
    for pos in client.get_all_positions():
        try:
            client.close_position(pos.symbol)
            print(f"🧹 Leftover position: queued a sell of {pos.qty} {pos.symbol} for the open")
        except Exception as e:
            print(f"❌ Could not close leftover {pos.symbol}: {e}")
            failed.append(pos.symbol)
    return failed
