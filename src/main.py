"""
The morning job (v2): scan the S&P 500 at open - 35 min, score the gated stocks with the frozen model, place the
day's trades before the open.

  1. catch-up evaluation of any finished session (the afternoon job owns it; a warning only)
  2. calendar checks: market open today, no plan yet, inside the 07:00-09:25 window
  3. frozen model checked against the pre-registered sha256 (a mismatch stops the day)
  4. safety net: close leftovers from a session whose exit failed
  5. score the month of baseline news while waiting, then wait until open - 35 min (08:55 ET)
  6. feature table as of 08:55 (prices to 08:39 because of the free plan's delay, news to 08:55), selection
  7. plan v2 + data/scores/<date>.csv; orders only if still before the 09:25 cutoff

DRY_RUN=1 does everything except orders, the safety net and the evaluation; it writes to data/dryrun/ only.
"""
import datetime as dt
import json
import os
import sys
import time

from market_session import now_ny
from storage import (DRYRUN_DIR, HOLDOUT_FILE, MODEL_FILE, PLANS_DIR, PREREG_FILE, SCORES_DIR, SENTIMENT_CACHE,
                     plan_path, save_json)

EARLIEST_BEFORE_OPEN = dt.timedelta(hours=2, minutes=30)  # 07:00 ET on a normal day
SCAN_BEFORE_OPEN = dt.timedelta(minutes=35)               # 08:55 ET: the time the backtest replayed
CUTOFF_BEFORE_OPEN = dt.timedelta(minutes=5)              # 09:25 ET: orders must be queued before trading starts
BASELINE_SESSIONS_BEFORE = 21                             # features.BASELINE_SESSIONS + 1


HEADLINE_FIELDS = ("headline", "url", "source")


def fresh_headlines(articles: list, symbols, since, until, n: int = 5) -> dict:
    """{symbol: up to n newest articles published in (since, until]}: headline, url, source, time and FinBERT tone.
    Article text and summaries are never copied."""
    wanted, out = set(symbols), {}
    for a in sorted(articles, key=lambda a: a["created_at"], reverse=True):
        if a["symbol"] in wanted and since < a["created_at"] <= until and len(out.setdefault(a["symbol"], [])) < n:
            out[a["symbol"]].append({**{k: a.get(k) for k in HEADLINE_FIELDS},
                                     "created_at": a["created_at"].isoformat(), "score": round(float(a["s"]), 4),
                                     "price_move": bool(a.get("price_move"))})
    return out


class LiveDeps:
    """The real clock, calendar, data and broker. Tests pass a fake with the same methods."""

    def __init__(self):
        import broker
        import sentiment
        self.client, self.broker, self.sentiment = broker.trading_client, broker, sentiment
        self.cache = sentiment.load_cache(SENTIMENT_CACHE)
        self.articles = []
        self._shortable = {}

    def now(self):
        return now_ny()

    def sleep(self, seconds):
        time.sleep(seconds)

    def sessions(self, start, end):
        import market_session
        return market_session.get_sessions(self.client, start, end)

    def evaluate_pending(self, now, plans_dir):
        from evaluator import evaluate_pending
        evaluate_pending(now=now, plans_dir=plans_dir)

    def close_leftovers(self, plans_dir):
        return self.broker.close_leftover_positions(plans_dir=plans_dir)

    def members(self, day):
        import universe
        return universe.load_universe(day)

    def warm_news(self, members, before, now):
        import news
        articles = news.fetch_articles([m["symbol"] for m in members], before[0].close, now)
        self.sentiment.score_articles(articles, cache=self.cache)
        print(f"Scored the baseline news: {len(articles)} article rows, cache now {len(self.cache)} articles")

    def feature_table(self, members, session, before, as_of):
        import features
        self.articles = []
        return features.morning_feature_table(members, session, before, as_of, score_cache=self.cache,
                                              articles_out=self.articles)

    def headlines(self, symbols, since, until):
        return fresh_headlines(self.articles, symbols, since, until)

    def equity(self):
        return float(self.client.get_account().equity)

    def is_shortable(self, symbol):
        if symbol not in self._shortable:
            asset = self.client.get_asset(symbol)
            status = getattr(getattr(asset, "borrow_status", None), "value", getattr(asset, "borrow_status", None))
            easy = status == "easy_to_borrow" if status is not None else bool(getattr(asset, "easy_to_borrow", False))
            self._shortable[symbol] = bool(asset.shortable) and easy
        return self._shortable[symbol]

    def place_entries(self, plan, skip):
        self.broker.place_entries(plan, skip_tickers=skip)

    def save_cache(self, before):
        self.sentiment.save_cache(SENTIMENT_CACHE, self.cache, keep_since=before[0].close)

    def live_strategy(self):
        with open(HOLDOUT_FILE) as f:
            return json.load(f)["goes_live"]


def _write_scores(scored, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scored.rename_axis("symbol").reset_index().to_csv(path, index=False, float_format="%.6g")


def run_morning_strategy(now=None, plans_dir=PLANS_DIR, dry_run=False, deps=None, dryrun_dir=DRYRUN_DIR,
                         scores_dir=SCORES_DIR, model_file=MODEL_FILE, prereg_file=PREREG_FILE) -> int:
    """Returns a process exit code: 0 = done or nothing to do, 1 = something needs attention."""
    deps = deps or LiveDeps()
    now = now or deps.now()
    print(f"--- MORNING JOB (v2{', DRY RUN' if dry_run else ''}) woke up at {now:%Y-%m-%d %H:%M %Z} ---")
    exit_code = 0

    if not dry_run:
        try:
            deps.evaluate_pending(now, plans_dir)
        except Exception as e:
            print(f"::warning::Catch-up evaluation failed (the afternoon job will retry): {e}")

    calendar = deps.sessions(now.date() - dt.timedelta(days=60), now.date())
    if not calendar or calendar[-1].date != now.date():
        print("Market is closed today (weekend or holiday). Nothing to do.")
        return exit_code
    session, before = calendar[-1], calendar[:-1][-BASELINE_SESSIONS_BEFORE:]
    out_dir = dryrun_dir if dry_run else plans_dir
    today_plan = plan_path(session.date.isoformat(), out_dir)
    if not dry_run and os.path.exists(today_plan):
        print(f"Plan for {session.date} already exists (an earlier attempt ran). Nothing to do.")
        return exit_code
    if now < session.open - EARLIEST_BEFORE_OPEN:
        print(f"Too early: research starts at {session.open - EARLIEST_BEFORE_OPEN:%H:%M %Z}. A later attempt will run.")
        return exit_code
    cutoff = session.open - CUTOFF_BEFORE_OPEN
    if now > cutoff:
        print(f"::warning::Morning run started at {now:%H:%M %Z}, after the {cutoff:%H:%M} cutoff for the "
              f"{session.date} session. Skipping today's trades.")
        return 1
    print(f"Session {session.date}: opens {session.open:%H:%M}, closes {session.close:%H:%M} ET "
          f"(previous session {before[-1].date})")

    import strategy
    try:
        frozen = strategy.load_frozen(model_file, prereg_file, live=deps.live_strategy())
    except strategy.FrozenModelMismatch as e:
        print(f"::error::The frozen model does not match the pre-registration, so nothing is traded: {e}")
        return 1
    spec = frozen["spec"]
    print(f"Live strategy {frozen['strategy']} ({spec['kind']}), model sha256 {frozen['sha256'][:12]}")

    leftovers, leftover_failures = [], []
    if not dry_run:
        leftovers, leftover_failures = deps.close_leftovers(plans_dir)
        if leftovers:
            print(f"::notice::Closing leftover positions from an earlier session at the open: {', '.join(leftovers)}. "
                  f"Their results will be corrected with the real price tonight.")
        if leftover_failures:
            print(f"::error::Could not close leftover positions: {', '.join(leftover_failures)}. Check the Alpaca dashboard.")
            exit_code = 1

    members = deps.members(session.date)
    try:
        deps.warm_news(members, before, now)
    except Exception as e:
        print(f"::warning::Could not pre-score the baseline news (it will be scored after the wait): {e}")

    as_of = session.open - SCAN_BEFORE_OPEN
    wait = (as_of - deps.now()).total_seconds()
    if wait > 0:
        print(f"Waiting {wait / 60:.0f} min for the scan at {as_of:%H:%M %Z}")
        deps.sleep(wait)
    table = deps.feature_table(members, session, before, as_of)
    import alerts
    data_coverage = alerts.coverage(table)
    for note in alerts.coverage_notices(table):
        print(f"::notice::{note}")
    equity = deps.equity()
    scored, targets = strategy.choose_trades(table, frozen, equity, deps.is_shortable)
    news = deps.headlines([t["ticker"] for t in targets], before[-1].close, as_of)
    for t in targets:
        t["headlines"] = news.get(t["ticker"], [])
    gated = scored[scored["gate_long"].astype(bool) | scored["gate_short"].astype(bool)]

    plan = {
        "session_date": session.date.isoformat(),
        "market_open": session.open.isoformat(),
        "market_close": session.close.isoformat(),
        "planned_at": deps.now().isoformat(),
        "scan_as_of": as_of.isoformat(),
        "strategy": "v2",
        "strategy_name": frozen["strategy"],
        "model_sha256": frozen["sha256"],
        "entry": spec["entry"],
        "stop_k": spec.get("stop_k"),
        "score_threshold": spec.get("threshold_return"),
        "dry_run": dry_run,
        "equity": equity,
        "coverage": data_coverage,
        "scanned": int(len(scored)),
        "candidates": [{"ticker": s, "gate": "long" if r["gate_long"] else "short",
                        "score": strategy._clean(r["score"]), "decision": r["decision"],
                        "gap_percent": strategy._clean(r.get("gap_pct")),
                        "sentiment_score": strategy._clean(r.get("news_today"))} for s, r in gated.iterrows()],
        "targets": targets,
    }
    _write_scores(scored, os.path.join(dryrun_dir, f"scores-{session.date}.csv") if dry_run
                  else os.path.join(scores_dir, f"{session.date}.csv"))
    print(f"\n--- PLAN {session.date}: {len(scored)} scanned, {len(gated)} gated, {len(targets)} targets ---")
    for t in targets:
        print(f"  {t['side'].upper():5} {t['ticker']:6} score {t['score']:+.4f}  qty {t['qty']}  stop {t['stop_price']}")

    try:
        if dry_run:
            print("Dry run: no orders.")
        elif deps.now() > cutoff:
            plan["skipped"] = "plan finished after the order cutoff"
            print(f"::warning::The plan was ready at {deps.now():%H:%M}, after the {cutoff:%H:%M} cutoff. No orders today.")
            exit_code = 1
        elif targets:
            deps.place_entries(plan, set(leftovers) | set(leftover_failures))
        else:
            print("No targets today. Staying in cash.")
    finally:
        save_json(today_plan, plan)
        print(f"Saved the plan with {len(targets)} targets to {os.path.relpath(today_plan)}")
        try:
            deps.save_cache(before)
        except Exception as e:
            print(f"::warning::Could not save the FinBERT score cache: {e}")
    return exit_code


if __name__ == "__main__":
    sys.exit(run_morning_strategy(dry_run=os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")))
