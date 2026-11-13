"""
Midday job: queue the closing sells for today's positions.

Market-on-close orders can only be placed for shares we already hold (after the
opening fill) and must reach Alpaca before its closing-auction cutoff, so this
runs inside a window between the open and shortly before the close. Several
cron attempts are scheduled; the work is idempotent, so extra runs are harmless.
"""
import datetime as dt
import sys

from broker import existing_order, place_exit_orders
from market_session import now_ny
from storage import PLANS_DIR, exit_order_id, load_json, plan_path

AFTER_OPEN = dt.timedelta(minutes=2)       # give the opening auction time to fill
CUTOFF_BEFORE_CLOSE = dt.timedelta(minutes=12)  # Alpaca rejects CLS orders after ~15:50 ET (12:50 on half days)


def run_midday(now=None, plans_dir=PLANS_DIR) -> int:
    now = now or now_ny()
    print(f"--- MIDDAY JOB woke up at {now:%Y-%m-%d %H:%M %Z} ---")

    plan = load_json(plan_path(now.date().isoformat(), plans_dir), default=None)
    if not plan:
        print("No plan for today (market closed, or the morning job skipped). Nothing to do.")
        return 0
    if not any(t.get("order_id") for t in plan["targets"]):
        print("No orders were placed today. Nothing to do.")
        return 0

    market_open = dt.datetime.fromisoformat(plan["market_open"])
    market_close = dt.datetime.fromisoformat(plan["market_close"])
    if now < market_open + AFTER_OPEN:
        print(f"Too early: exits can be placed after {market_open + AFTER_OPEN:%H:%M %Z}.")
        return 0

    cutoff = market_close - CUTOFF_BEFORE_CLOSE
    if now > cutoff:
        # Too late to queue a market-on-close order. Report whether an earlier attempt already did it.
        missing = [t["ticker"] for t in plan["targets"]
                   if t.get("order_id") and existing_order(exit_order_id(plan["session_date"], t["ticker"])) is None]
        if missing:
            print(f"::warning::Started at {now:%H:%M %Z}, after the {cutoff:%H:%M} cutoff. No closing order for: "
                  f"{', '.join(missing)}. The next morning job will close them at the open.")
            return 1
        print("All exit orders were already placed by an earlier attempt.")
        return 0

    return 0 if place_exit_orders(plan) else 1


if __name__ == "__main__":
    sys.exit(run_midday())
