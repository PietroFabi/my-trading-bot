"""
Exit job: close today's positions (sell longs, buy back shorts) with market orders just before the close,
after cancelling each trade's stop-loss leg and confirming the cancel (broker.place_exit_orders).

The Cloudflare scheduler starts this job at :55 every hour from 16:55 to 20:55 UTC.
Only the run that lands just before today's real close (from the plan, so daylight
saving and half days are handled) does anything; the others exit at once:
  normal day, winter: 20:55 UTC = 15:55 ET    summer: 19:55 UTC = 15:55 ET
  half day (13:00 close), winter: 17:55 UTC   summer: 16:55 UTC
"""
import datetime as dt
import sys

from broker import place_exit_orders, unsold_positions
from market_session import now_ny
from storage import PLANS_DIR, load_json, plan_path

WINDOW_START = dt.timedelta(minutes=10)  # earliest: close - 10 min
WINDOW_END = dt.timedelta(minutes=1)     # latest: close - 1 min, so market orders still fill before 16:00


def run_exits(now=None, plans_dir=PLANS_DIR) -> int:
    now = now or now_ny()
    print(f"--- EXIT JOB woke up at {now:%Y-%m-%d %H:%M %Z} ---")

    plan = load_json(plan_path(now.date().isoformat(), plans_dir), default=None)
    if not plan:
        print("No plan for today (market closed, or the morning job skipped). Nothing to do.")
        return 0
    if not any(t.get("order_id") for t in plan["targets"]):
        print("No orders were placed today. Nothing to do.")
        return 0

    close = dt.datetime.fromisoformat(plan["market_close"])
    if now < close - WINDOW_START:
        print(f"Not yet: exits are placed from {close - WINDOW_START:%H:%M} (close {close:%H:%M} ET).")
        return 0

    if now > close - WINDOW_END:
        # Too late to sell before the close. Alert only if shares bought today are really still held.
        held = unsold_positions(plan)
        if held:
            print(f"::warning::Started at {now:%H:%M %Z}, after the {close - WINDOW_END:%H:%M} cutoff, "
                  f"and still holding: {', '.join(held)}. The next morning job will sell them at the open.")
            return 1
        print("Nothing left to sell for today.")
        return 0

    return 0 if place_exit_orders(plan) else 1


if __name__ == "__main__":
    sys.exit(run_exits())
