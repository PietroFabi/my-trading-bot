import datetime as dt
import json
import os

import yfinance as yf

from market_session import now_ny

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS_FILE = os.path.join(DATA_DIR, "targets.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")

INITIAL_CAPITAL = 10000.00
POSITION_SIZE_USD = 1000.00

# Minutes after the open. On half days, points past the close use the closing price.
TIMEFRAMES = {
    "1_min": 1,
    "10_min": 10,
    "30_min": 30,
    "1_hour": 60,
    "5_hour": 300,
}
# Keys the portfolio site reads from portfolio_curve. Do not rename.
CURVE_KEYS = ["Open", "1_min", "10_min", "30_min", "1_hour", "5_hour", "EOD"]

# Wait this long after the close before trusting Yahoo's 1-minute bars to be complete.
SETTLE_AFTER_CLOSE = dt.timedelta(minutes=10)
# The first/last bar must be this close to the open/close, or the data has a hole.
MAX_EDGE_GAP = dt.timedelta(minutes=5)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def _minute_bars(ticker: str, session_date: dt.date, session_open: dt.datetime, session_close: dt.datetime):
    """Regular-hours 1-minute bars for one ticker on one explicit date."""
    data = yf.download(
        [ticker],
        start=session_date,
        end=session_date + dt.timedelta(days=1),
        interval="1m",
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"no 1-minute data for {session_date}")
    bars = data.xs(ticker, axis=1, level="Ticker") if data.columns.nlevels > 1 else data
    bars = bars.dropna(subset=["Open", "Close"])
    # Keep regular hours only. A bar's timestamp is its start: 09:30 covers 09:30-09:31.
    bars = bars[(bars.index >= session_open) & (bars.index < session_close)]
    if bars.empty:
        raise RuntimeError(f"no regular-hours bars for {session_date}")
    if bars.index[0] - session_open > MAX_EDGE_GAP or session_close - bars.index[-1] > MAX_EDGE_GAP + dt.timedelta(minutes=1):
        raise RuntimeError(f"incomplete session data ({bars.index[0]:%H:%M}-{bars.index[-1]:%H:%M})")
    return bars


def _price_at(bars, t: dt.datetime) -> float:
    """Price at clock time t = close of the last bar that started before t."""
    return float(bars[bars.index < t]["Close"].iloc[-1])


def _evaluate_trade(target: dict, session_date, session_open, session_close) -> dict:
    ticker = target["ticker"]
    bars = _minute_bars(ticker, session_date, session_open, session_close)
    open_price = float(bars["Open"].iloc[0])  # market orders queued pre-market fill at the open

    result = {
        "ticker": ticker,
        "sentiment_score": target["sentiment_score"],
        "allocation_usd": POSITION_SIZE_USD,
        "open_price": round(open_price, 2),
        "performance_pct": {},
    }
    points = {label: session_open + dt.timedelta(minutes=m) for label, m in TIMEFRAMES.items()}
    points["EOD"] = session_close
    for label, t in points.items():
        price = _price_at(bars, min(t, session_close))
        result["performance_pct"][label] = round((price - open_price) / open_price * 100, 2)
    return result


def evaluate_performance(now=None, skip_if_done=False, targets_file=TARGETS_FILE, results_file=RESULTS_FILE) -> bool:
    """
    Score the session recorded in targets.json and write it to results.json.

    The session date comes from the morning plan, never from the clock, so a
    run that GitHub starts hours late (even after midnight) still labels and
    prices the right day. Re-running for the same session overwrites its entry.
    Returns True if results.json was written.
    """
    now = now or now_ny()
    plan = load_json(targets_file, default={})
    if not isinstance(plan, dict) or "session_date" not in plan:
        print("No morning plan to evaluate.")
        return False

    session_date = dt.date.fromisoformat(plan["session_date"])
    session_open = dt.datetime.fromisoformat(plan["market_open"])
    session_close = dt.datetime.fromisoformat(plan["market_close"])
    print(f"--- EVALUATING SESSION {session_date} (now {now:%Y-%m-%d %H:%M %Z}) ---")

    if now < session_close + SETTLE_AFTER_CLOSE:
        print(f"Session {session_date} has not finished yet (closes {session_close:%H:%M %Z}). Nothing to do.")
        return False

    results = load_json(results_file, default=[])
    if not isinstance(results, list):
        raise RuntimeError("results.json is not a list; refusing to overwrite it")
    dates = [r.get("date") for r in results if isinstance(r, dict)]
    if plan["session_date"] in dates:
        if skip_if_done:
            print(f"Session {session_date} is already in results.json.")
            return False
        if dates[-1] != plan["session_date"]:
            print(f"Session {session_date} is already recorded and is not the latest entry; leaving it alone.")
            return False
        results = results[:-1]  # re-run: replace the latest entry

    # Carry the balance forward from the previous session (not from this session's old entry).
    starting_balance = INITIAL_CAPITAL
    if results and isinstance(results[-1], dict):
        starting_balance = float(results[-1].get("ending_balance", INITIAL_CAPITAL))

    # Only score positions that were actually ordered.
    ordered = [t for t in plan.get("targets", []) if t.get("order_id")]
    skipped = [t["ticker"] for t in plan.get("targets", []) if not t.get("order_id")]
    if skipped:
        print(f"Not scoring {', '.join(skipped)}: no order was placed.")

    trades = []
    for target in ordered:
        print(f"\nEvaluating {target['ticker']} (Sentiment: {target['sentiment_score']})...")
        trade = _evaluate_trade(target, session_date, session_open, session_close)  # raises on bad data
        for label, pct in trade["performance_pct"].items():
            print(f"  -> {label}: {pct:+.2f}% (${POSITION_SIZE_USD * pct / 100:+.2f})")
        trades.append(trade)

    curve = {"Open": starting_balance}
    for key in CURVE_KEYS[1:]:
        pnl = sum(POSITION_SIZE_USD * t["performance_pct"][key] / 100 for t in trades)
        curve[key] = starting_balance + pnl
    curve = {k: round(v, 2) for k, v in curve.items()}

    results.append({
        "date": plan["session_date"],
        "starting_balance": round(starting_balance, 2),
        "ending_balance": curve["EOD"],
        "portfolio_curve": curve,
        "trades": trades,
    })
    # allow_nan=False: NaN is not valid JSON and would break the site's response.json().
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, allow_nan=False)

    print(f"\nEvaluation complete for {session_date}! Ending Balance: ${curve['EOD']:,.2f}")
    return True


if __name__ == "__main__":
    evaluate_performance()
