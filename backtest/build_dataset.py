"""
Step 7: replay every morning since the news history starts, exactly as the live bot would have
seen it, and attach what happened next.

For each session from the first one with a full 20-session news baseline up to `end`:
  - membership = the S&P 500 as listed on Wikipedia that morning (no survivorship bias)
  - features  = features.morning_feature_table(...) as of open - 35 min: prices up to
                open - 51 min (free-plan delay), news up to open - 35 min. Same code as live.
  - outcomes  = 1-minute bars 09:30 -> close for the gated stocks and their sector ETFs,
                plus first-5-minute relative volume (for the opening-range-breakout variant)

Everything heavy is cached in .cache/ (gitignored): bars, Wikipedia revisions, FinBERT scores.
Article text is scored in memory and removed before anything is written to disk.

Run from the repo root:  python -m backtest.build_dataset
"""
import datetime as dt
import json
import os
import pickle
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

import backtest  # noqa: F401  (src/ on the path)
import features
import marketdata
import news
import sentiment
from backtest import history
from storage import json_safe
from universe import SECTOR_ETF

NY = ZoneInfo("America/New_York")
NEWS_START = dt.date(2026, 4, 29)          # first article in Alpaca's news history
SCAN_BEFORE_OPEN = dt.timedelta(minutes=35)
BEFORE = features.BASELINE_SESSIONS + 1     # sessions passed to the feature code
OPENING_MINUTES = 5
OPENING_RVOL_SESSIONS = 14

CACHE = ".cache"
DATASET = os.path.join(CACHE, "dataset.pkl")
SUMMARY = os.path.join("data", "backtest", "dataset_summary.json")


# --- Pure helpers (tested) -------------------------------------------------------------------------

def eval_sessions(sessions: list, news_start: dt.date = NEWS_START, n_before: int = BEFORE) -> list:
    """[(session, the n_before sessions before it)] for every session whose baseline window starts inside the news history."""
    return [(sessions[i], sessions[i - n_before:i]) for i in range(n_before, len(sessions))
            if sessions[i - n_before].date >= news_start]


def scan_time(session) -> dt.datetime:
    return session.open - SCAN_BEFORE_OPEN


def news_in_window(articles: list, symbols, start: dt.datetime, end: dt.datetime) -> list:
    wanted = set(symbols)
    return [a for a in articles if a["symbol"] in wanted and start < a["created_at"] <= end]


def strip_text(articles: list) -> list:
    """Drop article bodies and summaries: only headline, link, source, tags and scores are ever saved."""
    return [{k: v for k, v in a.items() if k not in ("text", "summary")} for a in articles]


def opening_rvol(first5: dict, day: dt.date, symbols, n: int = OPENING_RVOL_SESSIONS) -> pd.Series:
    """First-5-minute volume on `day` vs its mean over the previous n sessions (never later ones)."""
    prior = sorted(d for d in first5 if d < day)[-n:]
    today = first5.get(day, pd.Series(dtype=float))
    out = {}
    for s in symbols:
        history_ = [first5[d].get(s) for d in prior if s in first5[d].index]
        mean = sum(history_) / len(history_) if history_ else 0.0
        out[s] = today[s] / mean if s in today.index and mean > 0 else float("nan")
    return pd.Series(out, dtype=float)


# --- Orchestration (real data) ---------------------------------------------------------------------

def _trading_client():
    from alpaca.trading.client import TradingClient
    from dotenv import load_dotenv
    load_dotenv()
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)


def _scored_news(symbols: list, end: dt.datetime, log) -> list:
    path = os.path.join(CACHE, "news_scored.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            articles = pickle.load(f)
        if articles and max(a["created_at"] for a in articles) >= end - dt.timedelta(days=3):
            log(f"news: {len(articles)} scored articles from cache")
            return articles
    t0 = time.time()
    raw = news.fetch_articles(symbols, dt.datetime.combine(NEWS_START, dt.time(0), NY), end)
    log(f"news: fetched {len(raw)} article rows in {time.time() - t0:.0f}s")
    score_path = os.path.join(CACHE, "finbert_scores.json")
    cache = sentiment.load_cache(score_path)
    todo = len({a["id"] for a in raw if a["id"] not in cache})
    log(f"FinBERT: {todo} articles to score ({len(cache)} cached)")
    keep_since = dt.datetime(2000, 1, 1, tzinfo=NY)
    for i in range(0, len(raw), 2000):  # save progress regularly: a crash loses at most one chunk
        sentiment.score_articles(raw[i:i + 2000], cache=cache)
        sentiment.save_cache(score_path, cache, keep_since=keep_since)
        log(f"  scored {min(i + 2000, len(raw))}/{len(raw)} rows ({time.time() - t0:.0f}s)")
    articles = strip_text(raw)
    with open(path, "wb") as f:
        pickle.dump(articles, f)
    return articles


def build(end: dt.date = dt.date(2026, 11, 20), log=print) -> dict:
    t_start = time.time()
    os.makedirs(CACHE, exist_ok=True)
    bars_cache = os.path.join(CACHE, "bars")
    import market_session as ms
    sessions = ms.get_sessions(_trading_client(), NEWS_START - dt.timedelta(days=45), end)
    pairs = eval_sessions(sessions)
    log(f"{len(pairs)} sessions to replay: {pairs[0][0].date} .. {pairs[-1][0].date}")

    membership = history.membership_history([s.date for s, _ in pairs], os.path.join(CACHE, "membership.json"))
    union = history.union_members(membership)
    symbols = [m["symbol"] for m in union]
    etfs = sorted(set(SECTOR_ETF.values()))
    log(f"membership: {len(union)} distinct stocks over the period")

    daily = marketdata.daily_bars(symbols + etfs, pairs[0][0].date - dt.timedelta(days=400), end,
                                  cache_dir=bars_cache, fallback=False)
    articles = _scored_news(symbols, dt.datetime.combine(end, dt.time(16), NY), log)

    def premarket(day: dt.date, cutoff: dt.time):
        return marketdata.minute_bars(symbols + etfs, dt.datetime.combine(day, features.PREMARKET_START, NY),
                                      dt.datetime.combine(day, cutoff, NY), cache_dir=bars_cache, fallback=False)

    first_idx = sessions.index(pairs[0][0])
    first5 = {}
    for s in sessions[max(0, first_idx - OPENING_RVOL_SESSIONS):]:
        bars = marketdata.minute_bars(symbols, s.open, s.open + dt.timedelta(minutes=OPENING_MINUTES - 1),
                                      cache_dir=bars_cache, fallback=False)
        first5[s.date] = bars.groupby(level=0)["volume"].sum() if not bars.empty else pd.Series(dtype=float)

    tables, outcome_bars = [], {}
    for n, (s, before) in enumerate(pairs, 1):
        as_of = scan_time(s)
        cutoff = marketdata.latest_allowed(as_of).time()
        prior = before[-features.RVOL_SESSIONS:]
        minute = pd.concat([premarket(d.date, cutoff) for d in [s] + prior]).sort_index()

        table = features.morning_feature_table(
            membership[s.date], s, before, as_of,
            load_prices=lambda *a, **k: (daily, minute),
            fetch=lambda syms, start, stop: news_in_window(articles, syms, start, stop),
            score=lambda arts, cache=None: arts)
        gated = table[table.gate_long | table.gate_short]
        table["rvol5"] = opening_rvol(first5, s.date, list(gated.index)).reindex(table.index)
        table.insert(0, "session_date", s.date)
        tables.append(table.reset_index())

        wanted = sorted(set(gated.index) | set(gated["sector_etf"]))
        if wanted:
            outcome_bars[s.date] = marketdata.minute_bars(wanted, s.open, s.close - dt.timedelta(minutes=1),
                                                          cache_dir=bars_cache, fallback=False)
        log(f"[{n}/{len(pairs)}] {s.date}: liquid {int(table.liquid.sum())}, fresh news {int((table.n_fresh > 0).sum())}, "
            f"long {int(table.gate_long.sum())}, short {int(table.gate_short.sum())} ({time.time() - t_start:.0f}s)")

    frame = pd.concat(tables, ignore_index=True)
    dataset = {
        "sessions": [{"date": s.date, "open": s.open, "close": s.close, "before": [b.date for b in before]}
                     for s, before in pairs],
        "features": frame,
        "outcome_bars": outcome_bars,
        "meta": {"built_at": dt.datetime.now(NY).isoformat(), "news_start": NEWS_START.isoformat(),
                 "end": end.isoformat(), "finbert_revision": sentiment.MODEL_REVISION,
                 "scan_before_open_min": SCAN_BEFORE_OPEN.seconds // 60},
    }
    with open(DATASET, "wb") as f:
        pickle.dump(dataset, f)

    per_day = frame.groupby("session_date")
    summary = {
        "sessions": len(pairs), "first_session": pairs[0][0].date.isoformat(), "last_session": pairs[-1][0].date.isoformat(),
        "distinct_stocks": len(union), "articles": len(articles),
        "mean_per_session": {
            "members": round(per_day.size().mean(), 1), "priced": round(per_day.ref_price.count().mean(), 1),
            "liquid": round(per_day.liquid.sum().mean(), 1),
            "fresh_news": round(per_day.n_fresh.apply(lambda x: (x > 0).sum()).mean(), 1),
            "gate_long": round(per_day.gate_long.sum().mean(), 2), "gate_short": round(per_day.gate_short.sum().mean(), 2)},
        "gated_rows_total": int((frame.gate_long | frame.gate_short).sum()),
        "build_seconds": round(time.time() - t_start),
        "finbert_revision": sentiment.MODEL_REVISION,
    }
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    log(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    end = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2026, 11, 20)
    build(end)
