import datetime as dt
import json
import sys

from broker import execute_morning_trades, trading_client
from evaluator import TARGETS_FILE, evaluate_performance, load_json
from market_session import now_ny, previous_session, session_on

# Morning research must happen inside this pre-market window (New York time).
# Several cron attempts are scheduled; the first one that lands inside the
# window does the work, later ones see the saved plan and exit.
EARLIEST_BEFORE_OPEN = dt.timedelta(hours=2, minutes=30)  # 07:00 ET on a normal day
CUTOFF_BEFORE_OPEN = dt.timedelta(minutes=5)              # 09:25 ET: orders must be queued before the open

SENTIMENT_THRESHOLD = 0.15


def run_morning_strategy() -> int:
    """Returns a process exit code: 0 = done or nothing to do, 1 = today's session was missed."""
    now = now_ny()
    print(f"--- MORNING JOB woke up at {now:%Y-%m-%d %H:%M %Z} ---")

    # 0. If the previous session was never evaluated (afternoon run delayed or failed), do it now,
    #    before targets.json is overwritten with today's plan.
    evaluate_performance(now=now, skip_if_done=True)

    # 1. Is the market open today at all? (weekends, holidays)
    session = session_on(trading_client, now.date())
    if session is None:
        print("Market is closed today (weekend or holiday). Nothing to do.")
        return 0

    plan = load_json(TARGETS_FILE, default={})
    if isinstance(plan, dict) and plan.get("session_date") == session.date.isoformat():
        print(f"Plan for {session.date} already exists (an earlier attempt ran). Nothing to do.")
        return 0

    # 2. Are we inside the pre-market window?
    if now < session.open - EARLIEST_BEFORE_OPEN:
        print(f"Too early: research starts at {session.open - EARLIEST_BEFORE_OPEN:%H:%M %Z}. A later attempt will run.")
        return 0
    if now > session.open - CUTOFF_BEFORE_OPEN:
        # GitHub started us too late to buy at the open. Trading now would be a different strategy,
        # so skip the day and fail the run so GitHub sends a notification.
        print(f"::warning::Morning run started at {now:%H:%M %Z}, after the {session.open - CUTOFF_BEFORE_OPEN:%H:%M} "
              f"cutoff for the {session.date} session. Skipping today's trades.")
        return 1

    prev = previous_session(trading_client, session.date)
    print(f"Session {session.date}: opens {session.open:%H:%M}, closes {session.close:%H:%M} ET "
          f"(previous session {prev.date})")

    # Imported here so skipped runs don't spend time loading the FinBERT model.
    from scanner import WATCHLIST, get_top_gappers
    from sentiment import get_sentiment_for_tickers

    # 3. Scan for the top 5 pre-market movers
    top_movers = get_top_gappers(WATCHLIST, prev.date, session.date, top_n=5)
    tickers = [mover["ticker"] for mover in top_movers]

    # 4. Analyze the news sentiment for those stocks
    sentiments = get_sentiment_for_tickers(tickers)

    # 5. Formulate the trading plan
    plan = {
        "session_date": session.date.isoformat(),
        "market_open": session.open.isoformat(),
        "market_close": session.close.isoformat(),
        "planned_at": now.isoformat(),
        "targets": [],
    }
    print("\n--- FINAL TRADING PLAN ---")
    for mover in top_movers:
        ticker = mover["ticker"]
        score = sentiments.get(ticker, 0.0)
        if score > SENTIMENT_THRESHOLD:
            print(f"✅ APPROVED: {ticker} (Sentiment: {score:+.2f})")
            plan["targets"].append({
                "ticker": ticker,
                "gap_percent": round(mover["gap_percent"], 2),
                "sentiment_score": round(score, 2),
            })
        else:
            print(f"❌ REJECTED: {ticker} (Sentiment: {score:+.2f} is too low)")

    # 6. Place orders, then save the plan (with order ids) even if some orders failed,
    #    so the evaluator only scores positions that really exist.
    try:
        execute_morning_trades(plan)
    finally:
        with open(TARGETS_FILE, "w") as f:
            json.dump(plan, f, indent=4, allow_nan=False)
        print(f"\nSaved plan with {len(plan['targets'])} targets to data/targets.json")
    return 0


if __name__ == "__main__":
    sys.exit(run_morning_strategy())
