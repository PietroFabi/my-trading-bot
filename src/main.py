import os
import json
from datetime import datetime
from scanner import get_top_gappers, WATCHLIST
from sentiment import get_sentiment_for_tickers

# Dynamically find the root directory to create the data/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def run_morning_strategy():
    print(f"--- STARTING MORNING RESEARCH ({datetime.now().strftime('%Y-%m-%d')}) ---")

    # 1. Scan for the top 5 movers today
    top_movers = get_top_gappers(WATCHLIST, top_n=5)
    tickers = [mover["ticker"] for mover in top_movers]

    # 2. Analyze the news sentiment for those 5 stocks
    sentiments = get_sentiment_for_tickers(tickers)

    # 3. Formulate the trading plan
    targets = []
    print("\n--- FINAL TRADING PLAN ---")
    for mover in top_movers:
        ticker = mover["ticker"]
        score = sentiments.get(ticker, 0.0)

        # Our Strategy Logic: Only buy stocks with a positive AI sentiment
        if score > 0.15:
            print(f"✅ APPROVED: {ticker} (Sentiment: {score:+.2f})")
            targets.append({
                "ticker": ticker,
                "gap_percent": round(mover["gap_percent"], 2),
                "sentiment_score": round(score, 2),
                "date": datetime.now().strftime("%Y-%m-%d")
            })
        else:
            print(f"❌ REJECTED: {ticker} (Sentiment: {score:+.2f} is too low)")

    # 4. Save to targets.json
    targets_file = os.path.join(DATA_DIR, "targets.json")
    with open(targets_file, "w") as f:
        json.dump(targets, f, indent=4)

    print(f"\nMorning planning complete! Saved {len(targets)} targets to data/targets.json")


if __name__ == "__main__":
    run_morning_strategy()