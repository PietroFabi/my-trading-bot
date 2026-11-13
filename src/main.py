import datetime as dt
import os
import sys

from broker import close_leftover_positions, execute_morning_trades, trading_client
from evaluator import evaluate_pending
from market_session import now_ny, previous_session, session_on
from storage import PLANS_DIR, plan_path, save_json

# Morning research must happen inside this pre-market window (New York time).
# Several cron attempts are scheduled; the first one that lands inside the
# window does the work, later ones see the saved plan and exit.
EARLIEST_BEFORE_OPEN = dt.timedelta(hours=2, minutes=30)  # 07:00 ET on a normal day
CUTOFF_BEFORE_OPEN = dt.timedelta(minutes=5)              # 09:25 ET: Alpaca takes market-on-open orders until ~09:28

SENTIMENT_THRESHOLD = 0.15


def run_morning_strategy(now=None, plans_dir=PLANS_DIR) -> int:
    """Returns a process exit code: 0 = done or nothing to do, 1 = something needs attention."""
    now = now or now_ny()
    print(f"--- MORNING JOB woke up at {now:%Y-%m-%d %H:%M %Z} ---")
    exit_code = 0

    # 0. Record any finished session the afternoon job has not recorded yet (it may have run late or failed).
    #    Plans are kept per session, so a failure here loses nothing; the afternoon job will retry.
    try:
        evaluate_pending(now=now, plans_dir=plans_dir)
    except Exception as e:
        print(f"::warning::Catch-up evaluation failed: {e}")
        exit_code = 1

    # 1. Is the market open today at all? (weekends, holidays)
    session = session_on(trading_client, now.date())
    if session is None:
        print("Market is closed today (weekend or holiday). Nothing to do.")
        return exit_code

    today_plan = plan_path(session.date.isoformat(), plans_dir)
    if os.path.exists(today_plan):
        print(f"Plan for {session.date} already exists (an earlier attempt ran). Nothing to do.")
        return exit_code

    # 2. Are we inside the pre-market window?
    if now < session.open - EARLIEST_BEFORE_OPEN:
        print(f"Too early: research starts at {session.open - EARLIEST_BEFORE_OPEN:%H:%M %Z}. A later attempt will run.")
        return exit_code
    if now > session.open - CUTOFF_BEFORE_OPEN:
        # GitHub started us too late to buy at the open. Trading now would be a different strategy,
        # so skip the day and fail the run so GitHub sends a notification.
        print(f"::warning::Morning run started at {now:%H:%M %Z}, after the {session.open - CUTOFF_BEFORE_OPEN:%H:%M} "
              f"cutoff for the {session.date} session. Skipping today's trades.")
        return 1

    prev = previous_session(trading_client, session.date)
    print(f"Session {session.date}: opens {session.open:%H:%M}, closes {session.close:%H:%M} ET "
          f"(previous session {prev.date})")

    # 3. Safety net: sell anything left over from a session whose closing order was missed.
    if close_leftover_positions():
        exit_code = 1

    # Imported here so skipped runs don't spend time loading the FinBERT model.
    from scanner import WATCHLIST, get_top_gappers
    from sentiment import get_sentiment_for_tickers

    # 4. Scan for the top 5 pre-market movers
    top_movers = get_top_gappers(WATCHLIST, prev.date, session.date, top_n=5)
    tickers = [mover["ticker"] for mover in top_movers]

    # 5. Analyze the news sentiment for those stocks
    sentiments = get_sentiment_for_tickers(tickers)

    # 6. Formulate the trading plan
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
                "ref_price": round(mover["current_price"], 2),  # pre-market price, used to size whole shares
            })
        else:
            print(f"❌ REJECTED: {ticker} (Sentiment: {score:+.2f} is too low)")

    # 7. Place orders, then save the plan (with order ids) even if some orders failed,
    #    so the evaluator only scores positions that really exist.
    try:
        execute_morning_trades(plan)
    finally:
        save_json(today_plan, plan)
        print(f"\nSaved plan with {len(plan['targets'])} targets to data/plans/{session.date}.json")
    return exit_code


if __name__ == "__main__":
    sys.exit(run_morning_strategy())
