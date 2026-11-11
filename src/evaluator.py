import os
import json
import yfinance as yf
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS_FILE = os.path.join(DATA_DIR, "targets.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")

# Define our granularities in minutes (rows after open)
TIMEFRAMES = {
    "1_min": 1,
    "10_min": 10,
    "30_min": 30,
    "1_hour": 60,
    "5_hour": 300
}


def evaluate_performance():
    print(f"--- STARTING POST-MARKET EVALUATION ({datetime.now().strftime('%Y-%m-%d')}) ---")

    if not os.path.exists(TARGETS_FILE):
        print("No targets.json found. Did the morning run execute?")
        return

    with open(TARGETS_FILE, "r") as f:
        targets = json.load(f)

    if not targets:
        print("No targets were approved this morning. Nothing to evaluate.")
        return

    # We will compile today's results and append them to a master list
    master_results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                master_results = json.load(f)
            except json.JSONDecodeError:
                master_results = []

    today_date = datetime.now().strftime("%Y-%m-%d")
    daily_report = {
        "date": today_date,
        "trades": []
    }

    for target in targets:
        ticker = target["ticker"]
        sentiment = target["sentiment_score"]
        print(f"\nEvaluating {ticker} (Sentiment: {sentiment})...")

        try:
            # Download exactly 1 day of data at 1-minute intervals
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            closes = data['Close'][ticker].dropna()

            if len(closes) < 300:
                print(f"  -> Not enough intraday data for {ticker}. Skipping.")
                continue

            # Row 0 is the market open price
            open_price = closes.iloc[0]

            trade_result = {
                "ticker": ticker,
                "sentiment_score": sentiment,
                "open_price": round(open_price, 2),
                "performance_pct": {}
            }

            # Calculate % change for each specific minute granularity
            for label, minutes in TIMEFRAMES.items():
                if minutes < len(closes):
                    price_at_time = closes.iloc[minutes]
                    pct_change = ((price_at_time - open_price) / open_price) * 100
                    trade_result["performance_pct"][label] = round(pct_change, 2)
                    print(f"  -> {label}: {pct_change:+.2f}% (Price: ${price_at_time:.2f})")

            # EOD (End of Day) is simply the very last row available
            eod_price = closes.iloc[-1]
            eod_pct = ((eod_price - open_price) / open_price) * 100
            trade_result["performance_pct"]["EOD"] = round(eod_pct, 2)
            print(f"  -> EOD: {eod_pct:+.2f}% (Price: ${eod_price:.2f})")

            daily_report["trades"].append(trade_result)

        except Exception as e:
            print(f"  -> Error evaluating {ticker}: {e}")

    # Append today's report to the master results file
    master_results.append(daily_report)

    with open(RESULTS_FILE, "w") as f:
        json.dump(master_results, f, indent=4)

    print(f"\nEvaluation complete! Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_performance()