import datetime as dt

import yfinance as yf

# A solid universe of highly liquid, news-sensitive stocks
WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META", "GOOGL",
    "NFLX", "COIN", "PLTR", "ROKU", "SNOW", "CRWD", "UBER"
]


def get_top_gappers(tickers: list, prev_session_date: dt.date, session_date: dt.date, top_n: int = 5) -> list:
    """
    Pre-market gap = latest pre-market trade today vs. the previous session's close.

    Both sides are pinned to explicit dates. Before the open, Yahoo's daily
    bars have no row for today yet, so "last row vs. second-to-last row" would
    silently measure yesterday's move instead of today's gap.
    """
    print(f"Scanning {len(tickers)} tickers for gaps vs. the {prev_session_date} close...")

    # Daily bars up to (not including) today: the row dated prev_session_date is our reference close.
    daily = yf.download(
        tickers,
        start=prev_session_date - dt.timedelta(days=7),
        end=session_date,  # exclusive
        interval="1d",
        progress=False,
    )
    # 1-minute bars including pre/post market; today's pre-market rows start at 04:00 ET.
    intraday = yf.download(tickers, period="1d", interval="1m", prepost=True, progress=False)

    movers = []
    for ticker in tickers:
        try:
            closes = daily["Close"][ticker].dropna()
            if closes.empty or closes.index[-1].date() != prev_session_date:
                print(f"  {ticker}: no close for {prev_session_date}, skipping")
                continue
            prev_close = float(closes.iloc[-1])

            live = intraday["Close"][ticker].dropna()
            if live.empty or live.index[-1].date() != session_date:
                print(f"  {ticker}: no pre-market trade yet today, skipping")
                continue
            current_price = float(live.iloc[-1])

            gap_percent = (current_price - prev_close) / prev_close * 100
            movers.append({
                "ticker": ticker,
                "yesterday_close": prev_close,
                "current_price": current_price,
                "price_time": live.index[-1].isoformat(),
                "gap_percent": gap_percent,
                "abs_gap": abs(gap_percent),
            })
        except Exception as e:
            print(f"  {ticker}: could not fetch data: {e}")

    if not movers:
        # Nothing usable at all is a data outage, not a quiet day: fail loudly.
        raise RuntimeError("Scanner got no usable price data for any ticker")

    # Sort by the largest absolute gap (biggest movers, up or down)
    movers.sort(key=lambda x: x["abs_gap"], reverse=True)
    top_movers = movers[:top_n]

    print(f"\n--- Top {len(top_movers)} Movers Today ---")
    for m in top_movers:
        direction = "UP" if m["gap_percent"] > 0 else "DOWN"
        print(f"{m['ticker']}: {direction} {m['abs_gap']:.2f}% "
              f"(${m['yesterday_close']:.2f} -> ${m['current_price']:.2f} at {m['price_time']})")

    return top_movers
