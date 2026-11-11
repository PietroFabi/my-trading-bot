import os
import json
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS_FILE = os.path.join(DATA_DIR, "targets.json")

# Load API Keys
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the Trading Client (paper=True is critical!)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Amount of fake dollars to allocate to each approved stock
POSITION_SIZE_USD = 1000.00


def execute_morning_trades():
    print("--- STARTING ALPACA BROKER EXECUTION ---")

    if not os.path.exists(TARGETS_FILE):
        print("No targets.json found. Aborting execution.")
        return

    with open(TARGETS_FILE, "r") as f:
        targets = json.load(f)

    if not targets:
        print("No targets approved today. Staying in cash.")
        return

    print(f"Found {len(targets)} targets. Submitting orders...")

    for target in targets:
        ticker = target["ticker"]
        sentiment = target["sentiment_score"]

        try:
            # Create a market order to buy a specific dollar amount
            order_data = MarketOrderRequest(
                symbol=ticker,
                notional=POSITION_SIZE_USD,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            # Submit the order to Alpaca
            order = trading_client.submit_order(order_data=order_data)
            print(f"✅ ORDER QUEUED: Bought ${POSITION_SIZE_USD} of {ticker} (Order ID: {order.id})")

        except Exception as e:
            print(f"❌ ERROR submitting order for {ticker}: {e}")


if __name__ == "__main__":
    execute_morning_trades()