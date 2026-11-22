"""
Orders for the morning (entries), the exit job and the safety net. Light on purpose: the exit job installs only
alpaca-py and python-dotenv, so this module imports nothing else from the project except storage.

v2 targets carry a side ("long" / "short"), a whole-share qty and an optional stop price. v1 targets (no side, no
stop) are treated as longs, so old plans still exit and revise exactly as before.
"""
import os
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrderByIdRequest, GetOrdersRequest, MarketOrderRequest, StopLossRequest
from dotenv import load_dotenv

from storage import PLANS_DIR, buy_order_id, exit_order_id, late_exit_order_id, origin_session

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# paper=True is critical: it always targets paper-api.alpaca.markets
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

FUND_SIZE_USD = 100_000.00  # starting value of the fund shown on the site (v1 results); v2 sizes from account equity

# Order states that will not change any more.
FINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}

# How long the exit job waits for its closing orders to fill before reporting a failure.
FILL_TIMEOUT_S = 60
CANCEL_POLLS = 10


def status_of(order) -> str:
    return getattr(order.status, "value", str(order.status))


def side_sign(target: dict) -> int:
    return -1 if target.get("side") == "short" else 1


def existing_order(client_order_id: str, client=None):
    """The order already submitted under this client id, or None."""
    try:
        return (client or trading_client).get_order_by_client_id(client_order_id)
    except APIError as e:
        if e.status_code == 404:
            return None
        raise


# --- Entries -------------------------------------------------------------------------------------------

def place_entries(plan: dict, client=None, skip_tickers=()) -> None:
    """
    One market DAY order per target, queued before the open (it executes when trading starts at 09:30): a buy for a
    long, a short sale for a short. With a stop price, the order is an OTO: Alpaca keeps the stop-loss leg on hold
    and activates it as soon as the entry fills (long: sell stop below; short: buy stop above).
    Deterministic client ids make a second morning attempt find the order instead of trading twice.
    Raises if any order could not be placed (after trying all of them).

    skip_tickers: stocks whose leftover position is being closed this morning (wash-trade protection).
    """
    client = client or trading_client
    print("--- STARTING ALPACA BROKER EXECUTION ---")
    failures = []
    for target in plan["targets"]:
        ticker, side = target["ticker"], target.get("side", "long")
        if ticker in skip_tickers:
            print(f"⏭️  SKIP {ticker}: a leftover position is being closed this morning (wash-trade protection)")
            target["skip_reason"] = "leftover position being closed"
            continue
        qty = int(target["qty"])
        if qty < 1:
            target["skip_reason"] = "position too small"
            continue
        try:
            cid = buy_order_id(plan["session_date"], ticker)
            order = existing_order(cid, client)
            if order is not None:
                print(f"↩️  ALREADY SUBMITTED: {ticker} (Order ID: {order.id})")
            else:
                stop = target.get("stop_price")
                extra = {"order_class": OrderClass.OTO, "stop_loss": StopLossRequest(stop_price=stop)} if stop else {}
                order = client.submit_order(order_data=MarketOrderRequest(
                    symbol=ticker, qty=qty, side=OrderSide.BUY if side == "long" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY, client_order_id=cid, **extra))
                print(f"✅ QUEUED FOR THE OPEN: {'buy' if side == 'long' else 'short'} {qty} {ticker}"
                      f"{f' with a stop at ${stop:,.2f}' if stop else ''} (Order ID: {order.id})")
            target["order_id"] = str(order.id)
            legs = getattr(order, "legs", None) or []
            if legs:
                target["stop_order_id"] = str(legs[0].id)
        except Exception as e:
            print(f"❌ ERROR submitting order for {ticker}: {e}")
            failures.append(ticker)
    if failures:
        raise RuntimeError(f"Orders failed for: {', '.join(failures)}")


# --- Exits ---------------------------------------------------------------------------------------------

def _held_qty(client, ticker: str) -> int:
    """Signed shares held: positive long, negative short, 0 if no position."""
    try:
        pos = client.get_open_position(ticker)
    except APIError as e:
        if e.status_code == 404:
            return 0
        raise
    qty = abs(int(float(pos.qty)))
    side = getattr(pos.side, "value", pos.side)
    return -qty if str(side) == "short" or float(pos.qty) < 0 else qty


def _cancel_and_wait(client, order_id: str, sleep=time.sleep) -> str:
    """Cancel an order unless it is already final, then poll until it is. Returns its final (or last seen) status."""
    order = client.get_order_by_id(order_id)
    if status_of(order) not in FINAL_STATUSES:
        try:
            client.cancel_order_by_id(order_id)
        except APIError as e:
            print(f"   cancel of {order_id} refused ({e}); re-reading it")
        for _ in range(CANCEL_POLLS):
            order = client.get_order_by_id(order_id)
            if status_of(order) in FINAL_STATUSES:
                break
            sleep(1)
    return status_of(order)


def _settle_entry(client, entry, sleep=time.sleep):
    """If the entry is still open (e.g. a partial fill), cancel the rest so the closing order is not a wash trade."""
    if status_of(entry) in FINAL_STATUSES:
        return entry
    _cancel_and_wait(client, entry.id, sleep)
    return client.get_order_by_id(entry.id)


def _stop_leg_id(client, target: dict):
    """The stop-loss leg's id: stored at entry time, else read from the parent order (legs can be missing from the
    first submit response)."""
    if target.get("stop_order_id"):
        return target["stop_order_id"]
    if not target.get("stop_price"):
        return None
    parent = client.get_order_by_id(target["order_id"], filter=GetOrderByIdRequest(nested=True))
    legs = getattr(parent, "legs", None) or []
    return str(legs[0].id) if legs else None


def place_exit_orders(plan: dict, client=None, sleep=time.sleep) -> bool:
    """
    Close every position opened this session with a market order shortly before the close, then wait until each
    closing order has FILLED.

    Per target: settle the entry remainder -> cancel the stop leg and wait until it is final -> if the stop filled,
    the trade is already closed (exit_reason "stop"), nothing to send -> re-read the position -> close
    min(filled, held on that side) shares: sell a long, buy back a short. Never more than is held on the trade's
    side, so a position can never flip. Idempotent: a re-run finds the closing order by its client id.
    Returns True only when every held position is closed.
    """
    client = client or trading_client
    pending, all_ok = [], True
    for target in plan["targets"]:
        if not target.get("order_id"):
            continue
        ticker, sign = target["ticker"], side_sign(target)
        cid = exit_order_id(plan["session_date"], ticker)
        try:
            closing = existing_order(cid, client)
            if closing is not None:
                print(f"↩️  {ticker}: exit order already placed ('{status_of(closing)}')")
                pending.append((ticker, closing.id))
                continue
            entry = _settle_entry(client, client.get_order_by_id(target["order_id"]), sleep)
            filled = int(float(entry.filled_qty or 0))
            leg = _stop_leg_id(client, target)
            if leg and _cancel_and_wait(client, leg, sleep) == "filled":
                print(f"🛑 {ticker}: the stop-loss filled earlier, the trade is already closed")
                continue
            if filled == 0:
                print(f"⏭️  {ticker}: entry ended as '{status_of(entry)}' with no fill, nothing to close")
                continue
            qty = min(filled, sign * _held_qty(client, ticker))
            if qty <= 0:
                print(f"⏭️  {ticker}: no shares held on the trade's side any more, nothing to close")
                continue
            closing = client.submit_order(order_data=MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL if sign > 0 else OrderSide.BUY,
                time_in_force=TimeInForce.DAY, client_order_id=cid))
            print(f"📤 {'SELL' if sign > 0 else 'BUY TO COVER'} SENT: {qty} {ticker} at market (Order ID: {closing.id})")
            pending.append((ticker, closing.id))
        except Exception as e:
            print(f"❌ ERROR placing exit for {ticker}: {e}")
            all_ok = False

    # Accepted is not filled: confirm every closing order actually executed.
    waited = 0
    while pending and waited < FILL_TIMEOUT_S:
        still = []
        for ticker, oid in pending:
            o = client.get_order_by_id(oid)
            if status_of(o) == "filled":
                print(f"✅ CLOSED: {o.filled_qty} {ticker} at ${float(o.filled_avg_price):,.2f}")
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
    """Tickers from this session's plan still held on their trade's side (used after the close to raise an alert)."""
    client = client or trading_client
    return [t["ticker"] for t in plan["targets"]
            if t.get("order_id") and side_sign(t) * _held_qty(client, t["ticker"]) > 0]


# --- Safety net ----------------------------------------------------------------------------------------

def close_leftover_positions(client=None, plans_dir=PLANS_DIR, sleep=time.sleep):
    """
    Run before the morning entries: close any position left over from an earlier session whose exit failed. Its
    open orders (e.g. a stop leg) are cancelled first and confirmed final, then a market order (sell a long, buy
    back a short) executes at the open, tagged "<session>-<ticker>-late-exit" with the session that opened it, so
    the evaluator can replace that session's estimated exit with this real fill.
    Returns (symbols being closed, symbols that could not be closed).
    """
    client = client or trading_client
    closed, failed = [], []
    for pos in client.get_all_positions():
        try:
            for order in client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[pos.symbol])):
                _cancel_and_wait(client, order.id, sleep)
            origin = origin_session(pos.symbol, plans_dir)
            held = _held_qty(client, pos.symbol)
            if origin and held and existing_order(late_exit_order_id(origin, pos.symbol), client) is None:
                client.submit_order(order_data=MarketOrderRequest(
                    symbol=pos.symbol, qty=abs(held), side=OrderSide.SELL if held > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, client_order_id=late_exit_order_id(origin, pos.symbol)))
                print(f"🧹 Leftover from {origin}: queued a {'sell' if held > 0 else 'buy-back'} of {abs(held)} "
                      f"{pos.symbol} for the open")
            else:
                client.close_position(pos.symbol)  # not the bot's position (or already tagged): plain close
                print(f"🧹 Leftover position: queued a close of {pos.qty} {pos.symbol} for the open")
            closed.append(pos.symbol)
        except Exception as e:
            print(f"❌ Could not close leftover {pos.symbol}: {e}")
            failed.append(pos.symbol)
    return closed, failed
