import os

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the Trading Client (paper=True is critical: it always targets paper-api.alpaca.markets)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Amount of fake dollars to allocate to each approved stock
POSITION_SIZE_USD = 1000.00


def _existing_order(client_order_id: str):
    """The order already submitted under this id, or None."""
    try:
        return trading_client.get_order_by_client_id(client_order_id)
    except APIError as e:
        if e.status_code == 404:
            return None
        raise


def execute_morning_trades(plan: dict) -> None:
    """
    Submit one notional market order per approved target, queued for the open.

    Each order gets a deterministic client_order_id ("<session>-<ticker>"), so
    if the morning job runs twice for the same session the second run finds the
    existing order instead of buying again. The Alpaca order id is written back
    onto each target. Raises if any order could not be placed.
    """
    print("--- STARTING ALPACA BROKER EXECUTION ---")
    targets = plan["targets"]
    if not targets:
        print("No targets approved today. Staying in cash.")
        return

    failures = []
    for target in targets:
        ticker = target["ticker"]
        client_order_id = f"{plan['session_date']}-{ticker}"
        try:
            order = _existing_order(client_order_id)
            if order is not None:
                print(f"↩️  ALREADY SUBMITTED: {ticker} (Order ID: {order.id})")
            else:
                order = trading_client.submit_order(order_data=MarketOrderRequest(
                    symbol=ticker,
                    notional=POSITION_SIZE_USD,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=client_order_id,
                ))
                print(f"✅ ORDER QUEUED: ${POSITION_SIZE_USD} of {ticker} (Order ID: {order.id})")
            target["order_id"] = str(order.id)
        except Exception as e:
            print(f"❌ ERROR submitting order for {ticker}: {e}")
            failures.append(ticker)

    if failures:
        raise RuntimeError(f"Orders failed for: {', '.join(failures)}")
