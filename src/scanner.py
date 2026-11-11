import yfinance as yf
import pandas as pd
from datetime import datetime

# A solid universe of highly liquid, news-sensitive stocks
WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META", "GOOGL",
    "NFLX", "COIN", "PLTR", "ROKU", "SNOW", "CRWD", "UBER"
]


def get_top_gappers(tickers: list, top_n: int = 5) -> list:
    """
    Scans the watchlist for the biggest pre-market or open price gaps
    compared to yesterday's close.
    """
    movers = []

    # yfinance allows downloading multiple tickers at once for speed
    print(f"Scanning {len(tickers)} tickers for price gaps...")
    data = yf.download(tickers, period="5d", interval="1d", progress=False)

    for ticker in tickers:
        try:
            # Extract the closing prices for the specific ticker
            closes = data['Close'][ticker].dropna()

            if len(closes) < 2:
                continue

            yesterday_close = closes.iloc[-2]
            today_price = closes.iloc[-1]  # This acts as pre-market/open price today

            # Calculate gap percentage
            gap_percent = ((today_price - yesterday_close) / yesterday_close) * 100

            # We look at absolute gap (magnitude of the move, up or down)
            movers.append({
                "ticker": ticker,
                "yesterday_close": yesterday_close,
                "current_price": today_price,
                "gap_percent": gap_percent,
                "abs_gap": abs(gap_percent)
            })
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    # Sort by the largest absolute gap (biggest movers)
    movers.sort(key=lambda x: x["abs_gap"], reverse=True)

    top_movers = movers[:top_n]

    print(f"\n--- Top {top_n} Movers Today ---")
    for m in top_movers:
        direction = "UP" if m["gap_percent"] > 0 else "DOWN"
        print(f"{m['ticker']}: {direction} {abs(m['gap_percent']):.2f}% (Price: ${m['current_price']:.2f})")

    return top_movers


if __name__ == "__main__":
    # Test the scanner
    top_5 = get_top_gappers(WATCHLIST, top_n=5)