"""
Features: turn bars and articles into one row per stock, "as of" the morning scan.

Pure functions, shared by the live morning job and the backtest, so both compute every number
the same way. Nothing here may look past `as_of`: price bars stop at the free plan's cutoff
(as_of - 16 min) and news at as_of. Tests pin this.

Price features (per stock):
  gap_pct         last pre-market price vs the previous session's close, in %
  beta            1-year daily-return beta vs the stock's sector ETF (clipped 0-3; 1.0 if < 120 obs)
  sector_gap_pct  the sector ETF's own gap
  excess_gap_pct  gap_pct - beta * sector_gap_pct: the stock-specific part of the move
  atr_pct         Wilder ATR(14) as % of the previous close: a normal day's range
  gap_atr         excess_gap_pct / atr_pct: "how many normal days of stock-specific move"
  premarket_usd   dollars traded pre-market so far (data quality: is the gap real?)
  rvol            pre-market volume vs the same clock window on the previous 10 sessions
  trend           price / 50-day moving average - 1
  dist_52w_high   price / 52-week high - 1
  adv_usd         20-day average dollar volume
  liquid          passes every filter (ADV, price, pre-market dollars, sane gap, data present)
"""
import datetime as dt
import math

import numpy as np
import pandas as pd

from marketdata import empty_frame, latest_allowed

ATR_N = 14
MA_N = 50
YEAR = 252
BETA_MIN_OBS = 120
ADV_N = 20
MIN_ADV_USD = 20e6
MIN_PRICE = 5.0
MIN_PREMARKET_USD = 250_000
SUSPECT_GAP_PCT = 40.0   # bigger than this overnight is almost always a data/split problem
PREMARKET_START = dt.time(4, 0)


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ATR_N) -> float:
    """Average True Range with Wilder's smoothing, at the last bar. NaN with fewer than n+1 bars."""
    high, low, close = (pd.Series(x, dtype=float).reset_index(drop=True) for x in (high, low, close))
    if len(close) < n + 1:
        return float("nan")
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.iloc[:n].mean()
    for value in tr.iloc[n:]:
        atr = (atr * (n - 1) + value) / n
    return float(atr)


def _beta(stock_close: pd.Series, etf_close: pd.Series) -> float:
    joined = pd.concat([stock_close, etf_close], axis=1, keys=["s", "e"]).dropna()
    returns = joined.pct_change().dropna().iloc[-YEAR:]
    if len(returns) < BETA_MIN_OBS or returns["e"].var() == 0:
        return 1.0
    beta = returns["s"].cov(returns["e"]) / returns["e"].var()
    return float(np.clip(beta, 0.0, 3.0))


def _by_symbol(frame: pd.DataFrame) -> dict:
    """Split a (symbol, time) frame once into {symbol: frame indexed by time}: far faster than .loc per lookup."""
    if frame is None or len(frame) == 0:
        return {}
    return {sym: part.droplevel(0) for sym, part in frame.groupby(level=0, sort=False)}


def _window(bars, day: dt.date, end_time: dt.time) -> pd.DataFrame:
    """Minute bars on `day` from 04:00 up to and including end_time (New York). bars may be None."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["close", "volume"])
    tz = bars.index.tz
    start, end = dt.datetime.combine(day, PREMARKET_START, tz), dt.datetime.combine(day, end_time, tz)
    return bars[(bars.index >= start) & (bars.index <= end)]


def _gap(minute_by, daily_by, symbol, session, prev, cutoff):
    hist = daily_by.get(symbol)
    prev_close = float(hist["close"].get(prev.date, float("nan"))) if hist is not None else float("nan")
    today = _window(minute_by.get(symbol), session.date, cutoff.time())
    if today.empty or math.isnan(prev_close):
        return prev_close, float("nan"), None, today
    return prev_close, float(today["close"].iloc[-1]), today.index[-1], today


def price_features(daily: pd.DataFrame, minute: pd.DataFrame, members: list, session, prev,
                   prior_sessions: list, as_of: dt.datetime) -> pd.DataFrame:
    """
    One row per member, index = symbol.
    daily:  split-adjusted daily bars (symbol, date) for members AND their sector ETFs, up to prev.date.
    minute: 1-minute bars (symbol, New York timestamp) for members and ETFs: today's pre-market plus
            the same pre-market window on each of prior_sessions (for relative volume).
    """
    cutoff = latest_allowed(as_of)
    minute_by, daily_by = _by_symbol(minute), _by_symbol(daily)
    etf_gaps, rows = {}, []
    for etf in sorted({m["sector_etf"] for m in members}):
        prev_close, ref, _, _ = _gap(minute_by, daily_by, etf, session, prev, cutoff)
        etf_gaps[etf] = (ref / prev_close - 1) * 100 if not math.isnan(ref) else 0.0

    empty_daily = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
    for m in members:
        s = m["symbol"]
        hist = daily_by.get(s, empty_daily)
        hist = hist[hist.index <= prev.date]
        prev_close, ref, ref_time, today = _gap(minute_by, daily_by, s, session, prev, cutoff)
        gap = (ref / prev_close - 1) * 100 if not (math.isnan(ref) or math.isnan(prev_close)) else float("nan")

        etf_hist = daily_by[m["sector_etf"]]["close"] if m["sector_etf"] in daily_by else pd.Series(dtype=float)
        beta = _beta(hist["close"], etf_hist) if len(hist) else 1.0
        atr = wilder_atr(hist["high"], hist["low"], hist["close"]) if len(hist) else float("nan")
        atr_pct = atr / prev_close * 100 if prev_close and not math.isnan(atr) else float("nan")
        sector_gap = etf_gaps.get(m["sector_etf"], 0.0)
        excess = gap - beta * sector_gap
        gap_atr = excess / atr_pct if atr_pct and not math.isnan(atr_pct) else float("nan")

        volume_today = float(today["volume"].sum()) if len(today) else 0.0
        bars = minute_by.get(s)
        prior_volumes = [float(_window(bars, p.date, cutoff.time())["volume"].sum()) for p in prior_sessions]
        rvol = (volume_today + 1) / (float(np.mean(prior_volumes)) + 1) if prior_volumes else float("nan")
        premarket_usd = float((today["close"] * today["volume"]).sum()) if len(today) else 0.0

        closes = hist["close"]
        trend = ref / closes.iloc[-MA_N:].mean() - 1 if len(closes) >= MA_N else float("nan")
        dist_high = ref / hist["high"].iloc[-YEAR:].max() - 1 if len(hist) else float("nan")
        adv = float((hist["close"] * hist["volume"]).iloc[-ADV_N:].mean()) if len(hist) else 0.0

        suspect = bool(not math.isnan(gap) and abs(gap) > SUSPECT_GAP_PCT)
        liquid = bool(not math.isnan(gap) and not math.isnan(atr_pct) and not suspect
                      and adv >= MIN_ADV_USD and ref >= MIN_PRICE and premarket_usd >= MIN_PREMARKET_USD)
        rows.append({
            "symbol": s, "sector": m["sector"], "sector_etf": m["sector_etf"],
            "prev_close": prev_close, "ref_price": ref, "ref_time": ref_time, "gap_pct": gap,
            "beta": beta, "sector_gap_pct": sector_gap, "excess_gap_pct": excess,
            "atr_pct": atr_pct, "gap_atr": gap_atr, "premarket_usd": premarket_usd, "rvol": rvol,
            "trend": trend, "dist_52w_high": dist_high, "adv_usd": adv,
            "suspect_gap": suspect, "liquid": liquid,
        })
    return pd.DataFrame(rows).set_index("symbol")


def load_price_inputs(members: list, session, prev, prior_sessions: list, as_of: dt.datetime,
                      cache_dir: str | None = None, fallback: bool = True, md=None):
    """
    Fetch what price_features needs: ~13 months of daily bars up to prev.date, today's pre-market
    minute bars (up to the free-plan cutoff) and the same clock window on each prior session.
    Returns (daily, minute).
    """
    if md is None:
        import marketdata as md
    symbols = [m["symbol"] for m in members] + sorted({m["sector_etf"] for m in members})
    daily = md.daily_bars(symbols, prev.date - dt.timedelta(days=400), prev.date,
                          cache_dir=cache_dir, fallback=fallback)
    tz = session.open.tzinfo
    cutoff = latest_allowed(as_of)
    frames = [md.minute_bars(symbols, dt.datetime.combine(session.date, PREMARKET_START, tz), cutoff,
                             as_of=as_of, cache_dir=cache_dir, fallback=fallback)]
    for p in prior_sessions:  # history: no delay issue, and Yahoo has no old minute bars, so no fallback
        frames.append(md.minute_bars(symbols, dt.datetime.combine(p.date, PREMARKET_START, tz),
                                     dt.datetime.combine(p.date, cutoff.time(), tz),
                                     cache_dir=cache_dir, fallback=False))
    frames = [f for f in frames if not f.empty]
    minute = pd.concat(frames).sort_index() if frames else empty_frame()
    return daily, minute


# --- News features --------------------------------------------------------------------------------

BASELINE_SESSIONS = 20
HALF_LIFE_SESSIONS = 5
SHRINK_K = 2
NEWS_COLUMNS = ["n_fresh", "news_today", "baseline", "shrunk", "surprise", "news_volume", "n_price_move"]


def _weighted_tone(group: list) -> float | None:
    """w-weighted mean of s; None when the articles carry no weight at all."""
    total = sum(a["w"] for a in group)
    return sum(a["w"] * a["s"] for a in group) / total if total > 1e-9 else None


def news_features(articles: list, session, sessions_before: list, as_of: dt.datetime,
                  exclude_price_move: bool = False) -> pd.DataFrame:
    """
    Per stock, as of the scan. `articles` are scored (s, w); `sessions_before` are the
    BASELINE_SESSIONS + 1 sessions before today, oldest first (the first one only marks where
    the baseline window starts; the last one is the previous session).

    fresh        published after the previous session's close and no later than as_of
                 (on a Monday that includes the weekend)
    news_today   w-weighted mean tone of fresh articles (0 if none)
    baseline     per-session w-weighted tone over the previous 20 sessions, combined with
                 exponential weights (half-life 5 sessions); sessions without news are skipped
    shrunk       baseline + n/(n+k) * (news_today - baseline): one article moves it a little
    surprise     shrunk - baseline
    news_volume  log((fresh count + 1) / (average articles per session + 1))
    """
    prev_close = sessions_before[-1].close
    window_start = sessions_before[0].close
    buckets = [(sessions_before[i - 1].close, sessions_before[i].close, len(sessions_before) - i)
               for i in range(1, len(sessions_before))]  # (start, end, age in sessions)

    by_symbol = {}
    for a in articles:
        if window_start < a["created_at"] <= as_of:
            by_symbol.setdefault(a["symbol"], []).append(a)

    rows = []
    for symbol, arts in by_symbol.items():
        fresh_all = [a for a in arts if a["created_at"] > prev_close]
        usable = [a for a in arts if not (exclude_price_move and a.get("price_move"))]
        fresh = [a for a in usable if a["created_at"] > prev_close]
        news_today = _weighted_tone(fresh) or 0.0

        weighted, weights, history_count = 0.0, 0.0, 0
        for start, end, age in buckets:
            day = [a for a in usable if start < a["created_at"] <= end]
            history_count += len(day)
            tone = _weighted_tone(day) if day else None
            if tone is not None:
                k = 0.5 ** (age / HALF_LIFE_SESSIONS)
                weighted, weights = weighted + k * tone, weights + k
        baseline = weighted / weights if weights else 0.0

        n = len(fresh)
        shrunk = baseline + n / (n + SHRINK_K) * (news_today - baseline) if n else baseline
        rows.append({
            "symbol": symbol, "n_fresh": n, "news_today": news_today, "baseline": baseline, "shrunk": shrunk,
            "surprise": shrunk - baseline,
            "news_volume": math.log((n + 1) / (history_count / len(buckets) + 1)),
            "n_price_move": sum(1 for a in fresh_all if a.get("price_move")),
        })
    frame = pd.DataFrame(rows, columns=["symbol"] + NEWS_COLUMNS)
    return frame.set_index("symbol")


def build_feature_table(price: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """
    Price features joined with news features (stocks without news get zeros), plus the gates:
      gate_long  = liquid and sector-adjusted gap up   and fresh news with positive tone
      gate_short = liquid and sector-adjusted gap down and fresh news with negative tone
    """
    table = price.join(news, how="left")
    table[NEWS_COLUMNS] = table[NEWS_COLUMNS].fillna(0)
    has_news = table["n_fresh"] > 0
    liquid = table["liquid"].astype(bool)
    up, down = table["excess_gap_pct"] > 0, table["excess_gap_pct"] < 0  # sector-adjusted (user decision)
    table["gate_long"] = liquid & has_news & up & (table["news_today"] > 0)
    table["gate_short"] = liquid & has_news & down & (table["news_today"] < 0)
    return table


RVOL_SESSIONS = 10


def morning_feature_table(members: list, session, sessions_before: list, as_of: dt.datetime,
                          score_cache: dict | None = None, cache_dir: str | None = None, fallback: bool = True,
                          exclude_price_move: bool = False,
                          load_prices=None, compute_prices=None, fetch=None, score=None) -> pd.DataFrame:
    """
    The full feature table for one morning in one call (live job and backtest alike):
    prices -> price features; news since the start of the baseline window -> FinBERT -> news
    features; joined, with gates. `sessions_before` = the BASELINE_SESSIONS + 1 previous sessions.
    """
    import news as news_module
    import sentiment
    load_prices = load_prices or load_price_inputs
    compute_prices = compute_prices or price_features
    fetch = fetch or news_module.fetch_articles
    score = score or sentiment.score_articles

    prev, prior = sessions_before[-1], sessions_before[-RVOL_SESSIONS:]
    daily, minute = load_prices(members, session, prev, prior, as_of, cache_dir=cache_dir, fallback=fallback)
    price = compute_prices(daily, minute, members, session, prev, prior, as_of)
    articles = score(fetch([m["symbol"] for m in members], sessions_before[0].close, as_of), cache=score_cache)
    news = news_features(articles, session, sessions_before, as_of, exclude_price_move=exclude_price_move)
    return build_feature_table(price, news)
