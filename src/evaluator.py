import os
import json
import yfinance as yf
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS_FILE = os.path.join(DATA_DIR, "targets.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")

INITIAL_CAPITAL = 10000.00
POSITION_SIZE_USD = 1000.00

# Define granularities in minutes (rows after open)
TIMEFRAMES = {
    "1_min": 1,
    "10_min": 10,
    "30_min": 30,
    "1_hour": 60,
    "5_hour": 300
}


def evaluate_performance():
    print(f"--- STARTING POST-MARKET EVALUATION ({datetime.now().strftime('%Y-%m-%d')}) ---")

    # Load existing results to carry forward the previous ending balance
    master_results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                master_results = json.load(f)
            except json.JSONDecodeError:
                master_results = []

    starting_balance = INITIAL_CAPITAL
    if master_results and isinstance(master_results[-1], dict):
        starting_balance = master_results[-1].get("ending_balance", INITIAL_CAPITAL)

    targets = []
    if os.path.exists(TARGETS_FILE):
        with open(TARGETS_FILE, "r") as f:
            try:
                targets = json.load(f)
            except json.JSONDecodeError:
                targets = []

    today_date = datetime.now().strftime("%Y-%m-%d")

    # Initialize portfolio curve at the starting balance for every timeframe
    all_keys = ["Open", "1_min", "10_min", "30_min", "1_hour", "5_hour", "EOD"]
    portfolio_curve = {k: round(starting_balance, 2) for k in all_keys}

    daily_report = {
        "date": today_date,
        "starting_balance": round(starting_balance, 2),
        "ending_balance": round(starting_balance, 2),
        "portfolio_curve": portfolio_curve,
        "trades": []
    }

    if not targets:
        print(f"No targets approved today. Holding cash at ${starting_balance:,.2f}.")
    else:
        for target in targets:
            ticker = target["ticker"]
            sentiment = target["sentiment_score"]
            print(f"\nEvaluating {ticker} (Sentiment: {sentiment})...")

            try:
                data = yf.download(ticker, period="1d", interval="1m", progress=False)
                # Handle both single-level and multi-level column structures from yfinance
                close_col = data['Close'][ticker] if ticker in data['Close'] else data['Close']
                closes = close_col.dropna()

                if len(closes) < 300:
                    print(f"  -> Not enough intraday data for {ticker}. Skipping.")
                    continue

                open_price = float(closes.iloc[0])
                trade_result = {
                    "ticker": ticker,
                    "sentiment_score": sentiment,
                    "allocation_usd": POSITION_SIZE_USD,
                    "open_price": round(open_price, 2),
                    "performance_pct": {}
                }

                # Calculate % change and dollar P&L for each granularity
                for label, minutes in TIMEFRAMES.items():
                    if minutes < len(closes):
                        price_at_time = float(closes.iloc[minutes])
                        pct_change = ((price_at_time - open_price) / open_price) * 100
                        dollar_pnl = POSITION_SIZE_USD * (pct_change / 100.0)

                        trade_result["performance_pct"][label] = round(pct_change, 2)
                        portfolio_curve[label] = round(portfolio_curve[label] + dollar_pnl, 2)
                        print(f"  -> {label}: {pct_change:+.2f}% (${dollar_pnl:+.2f})")

                # End of Day (EOD)
                eod_price = float(closes.iloc[-1])
                eod_pct = ((eod_price - open_price) / open_price) * 100
                eod_pnl = POSITION_SIZE_USD * (eod_pct / 100.0)

                trade_result["performance_pct"]["EOD"] = round(eod_pct, 2)
                portfolio_curve["EOD"] = round(portfolio_curve["EOD"] + eod_pnl, 2)
                print(f"  -> EOD: {eod_pct:+.2f}% (${eod_pnl:+.2f})")

                daily_report["trades"].append(trade_result)

            except Exception as e:
                print(f"  -> Error evaluating {ticker}: {e}")

    daily_report["portfolio_curve"] = portfolio_curve
    daily_report["ending_balance"] = portfolio_curve["EOD"]

    # Overwrite today's entry if re-run on the same day, otherwise append
    if master_results and master_results[-1].get("date") == today_date:
        master_results[-1] = daily_report
    else:
        master_results.append(daily_report)

    with open(RESULTS_FILE, "w") as f:
        json.dump(master_results, f, indent=4)

    print(f"\nEvaluation complete! Ending Balance: ${daily_report['ending_balance']:,.2f}")


if __name__ == "__main__":
    evaluate_performance()