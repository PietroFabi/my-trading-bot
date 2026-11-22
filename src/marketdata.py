"""
Market data: daily and 1-minute bars for many symbols at once.

Primary source: Alpaca's full-market (SIP) feed. On the free plan the most recent 15 minutes
are not available, so every request ends at least 16 minutes before the moment it is made
(one minute of margin). The backtest uses the same source, so live and backtest features
are computed from identical data.

Fallback: Yahoo (yfinance), only for the symbols Alpaca returned nothing for, or for
everything if Alpaca fails. Switched off for history (Yahoo keeps ~30 days of minute bars).

Normalised format (both sources):
  minute bars: index (symbol, timestamp in New York time), columns open/high/low/close/volume
  daily bars:  index (symbol, date),                       same columns
Daily bars are split-adjusted, so an overnight split does not look like a crash.
"""
import datetime as dt
import hashlib
import os
import pickle
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
COLUMNS = ["open", "high", "low", "close", "volume"]
FREE_PLAN_DELAY = dt.timedelta(minutes=16)  # 15-minute SIP delay + 1 minute of margin

_client_singleton = None


def _client():
    global _client_singleton
    if _client_singleton is None:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from dotenv import load_dotenv
        load_dotenv()
        _client_singleton = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    return _client_singleton


def latest_allowed(as_of: dt.datetime) -> dt.datetime:
    """The newest bar time the free plan lets us request at moment as_of."""
    return as_of - FREE_PLAN_DELAY


def empty_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
    return pd.DataFrame({c: pd.Series(dtype=float) for c in COLUMNS}, index=index)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """alpaca-py BarSet.df -> our minute format (timestamps converted to New York time)."""
    if df is None or df.empty:
        return empty_frame()
    df = df[COLUMNS].copy()
    symbols = df.index.get_level_values(0)
    stamps = pd.DatetimeIndex(df.index.get_level_values(1)).tz_convert(NY)
    df.index = pd.MultiIndex.from_arrays([symbols, stamps], names=["symbol", "timestamp"])
    return df.sort_index()


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        frame = empty_frame()
        frame.index = frame.index.set_names(["symbol", "date"])
        return frame
    dates = [t.date() for t in df.index.get_level_values("timestamp")]
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays([df.index.get_level_values("symbol"), dates], names=["symbol", "date"])
    return df


def coverage(frame: pd.DataFrame, symbols) -> float:
    """Share of `symbols` with at least one bar in `frame`."""
    symbols = list(symbols)
    if not symbols:
        return 0.0
    have = set(frame.index.get_level_values(0)) if not frame.empty else set()
    return sum(s in have for s in symbols) / len(symbols)


def yahoo_bars(yahoo_symbols, start: dt.datetime, end: dt.datetime, interval: str) -> pd.DataFrame:
    """Yahoo fallback in our minute format, still in Yahoo spelling (the caller maps "BRK-B" -> "BRK.B")."""
    import logging
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # quiet stocks with no pre-market trades are expected
    data = yf.download(list(yahoo_symbols), start=start, end=end, interval=interval, prepost=(interval != "1d"),
                       auto_adjust=False, group_by="ticker", progress=False, threads=True)
    frames = []
    for ysym in yahoo_symbols:
        try:
            part = data[ysym] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        part = part.rename(columns=str.lower)[COLUMNS].dropna(subset=["close"])
        if part.empty:
            continue
        stamps = pd.DatetimeIndex(part.index)
        stamps = stamps.tz_localize(NY) if stamps.tz is None else stamps.tz_convert(NY)
        part.index = pd.MultiIndex.from_arrays([[ysym] * len(part), stamps],
                                               names=["symbol", "timestamp"])
        frames.append(part)
    return pd.concat(frames).sort_index() if frames else empty_frame()


def _cached(cache_dir, key_parts, compute):
    if not cache_dir:
        return compute()
    key = hashlib.sha256(repr(key_parts).encode()).hexdigest()[:24]
    path = os.path.join(cache_dir, f"bars-{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def _fetch(symbols, start, end, timeframe, interval, client, yahoo, fallback):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    symbols = list(dict.fromkeys(symbols))
    try:
        response = (client or _client()).get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end,
            adjustment=Adjustment.SPLIT, feed=DataFeed.SIP))
        bars = normalise(response.df)
    except Exception as e:
        if not fallback:
            raise
        print(f"::notice::Alpaca bars request failed ({e}); using Yahoo for all {len(symbols)} symbols")
        bars = empty_frame()
    missing = [s for s in symbols if s not in set(bars.index.get_level_values(0))]
    if fallback and missing:
        extra = (yahoo or yahoo_bars)([s.replace(".", "-") for s in missing], start, end, interval)
        if not extra.empty:
            # Yahoo spells dotted tickers with "-"; map back to Alpaca's form here, in one place.
            extra.index = extra.index.set_levels(
                [s.replace("-", ".") for s in extra.index.levels[0]], level=0)
            bars = pd.concat([bars, extra]).sort_index()
    return bars


def minute_bars(symbols, start: dt.datetime, end: dt.datetime, as_of: dt.datetime | None = None, client=None,
                yahoo=None, fallback: bool = True, cache_dir: str | None = None, now: dt.datetime | None = None) -> pd.DataFrame:
    """1-minute bars in [start, end], never newer than the free plan allows at `as_of` (and right now)."""
    from alpaca.data.timeframe import TimeFrame
    end = min(end, latest_allowed(now or dt.datetime.now(NY)))
    if as_of is not None:
        end = min(end, latest_allowed(as_of))
    return _cached(cache_dir, ("minute", sorted(symbols), start.isoformat(), end.isoformat(), fallback),
                   lambda: _fetch(symbols, start, end, TimeFrame.Minute, "1m", client, yahoo, fallback))


def daily_bars(symbols, start: dt.date, end: dt.date, client=None, yahoo=None, fallback: bool = True,
               cache_dir: str | None = None, now: dt.datetime | None = None) -> pd.DataFrame:
    """Split-adjusted daily bars for sessions start..end (inclusive), indexed (symbol, New York date)."""
    from alpaca.data.timeframe import TimeFrame
    t0 = dt.datetime.combine(start, dt.time(0), NY)
    t1 = min(dt.datetime.combine(end, dt.time(23, 59), NY), latest_allowed(now or dt.datetime.now(NY)))
    bars = _cached(cache_dir, ("daily", sorted(symbols), t0.isoformat(), t1.isoformat(), fallback),
                   lambda: _fetch(symbols, t0, t1, TimeFrame.Day, "1d", client, yahoo, fallback))
    daily = _to_daily(bars)
    # Look-ahead guard: the bar for date end+1 is stamped exactly at the midnight after `end`,
    # so a time-based request can return it. Filter by date explicitly.
    dates = daily.index.get_level_values("date")
    return daily[(dates >= start) & (dates <= end)]
