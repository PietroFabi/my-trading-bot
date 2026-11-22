import datetime as dt
import math
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import tests  # noqa: F401
import features
from market_session import Session

NY = ZoneInfo("America/New_York")
MEMBERS = [{"symbol": "AAA", "yahoo": "AAA", "sector": "Industrials", "sector_etf": "XLI"}]
SESSION = Session(dt.date(2026, 11, 20), dt.datetime(2026, 11, 20, 9, 30, tzinfo=NY),
                  dt.datetime(2026, 11, 20, 16, 0, tzinfo=NY))
AS_OF = dt.datetime(2026, 11, 20, 8, 55, tzinfo=NY)   # data allowed up to 08:39
DATES = [d.date() for d in pd.bdate_range(end="2026-11-19", periods=270)]
PREV = Session(DATES[-1], dt.datetime.combine(DATES[-1], dt.time(9, 30), NY),
               dt.datetime.combine(DATES[-1], dt.time(16), NY))
PRIOR = [Session(d, dt.datetime.combine(d, dt.time(9, 30), NY), dt.datetime.combine(d, dt.time(16), NY))
         for d in DATES[-10:]]


def daily_frame(stock_close_last=None):
    """ETF follows a random walk; the stock's daily returns are exactly 2x the ETF's (beta = 2)."""
    r = np.random.default_rng(7).normal(0, 0.01, len(DATES))
    r[0] = 0
    etf = 100 * np.cumprod(1 + r)
    stock = 50 * np.cumprod(1 + 2 * r)
    if stock_close_last is not None:
        stock[-1] = stock_close_last
    rows = []
    for sym, closes, vol in (("XLI", etf, 5e6), ("AAA", stock, 1e6)):
        for d, c in zip(DATES, closes):
            rows.append((sym, d, c, c * 1.01, c * 0.99, c, vol))
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    return df.set_index(["symbol", "date"])


def minute_frame(daily, stock_gap=0.02, etf_gap=0.01, stock_volume=50_000, prior_volume=50_000 / 3,
                 extra=()):
    rows = []
    for sym, gap, vol in (("XLI", etf_gap, 1000), ("AAA", stock_gap, stock_volume)):
        prev_close = daily.loc[(sym, DATES[-1]), "close"]
        if gap is not None:
            rows.append((sym, dt.datetime(2026, 11, 20, 8, 0, tzinfo=NY), prev_close * (1 + gap), vol))
        for s in PRIOR:
            if sym == "AAA":
                rows.append((sym, dt.datetime.combine(s.date, dt.time(8, 0), NY), prev_close, prior_volume))
    rows.extend(extra)
    df = pd.DataFrame([(s, t, p, p, p, p, v) for s, t, p, v in rows],
                      columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    return df.set_index(["symbol", "timestamp"]).sort_index()


def compute(daily=None, minute=None):
    daily = daily_frame() if daily is None else daily
    minute = minute_frame(daily) if minute is None else minute
    return features.price_features(daily, minute, MEMBERS, SESSION, PREV, PRIOR, AS_OF)


class AtrTest(unittest.TestCase):
    def test_wilder_atr_on_a_hand_computed_series(self):
        # 15 bars whose true range is 2, then one bar with true range 16: ATR = (13*2 + 16) / 14 = 3
        close = [100.0] * 16
        high = [101.0] * 15 + [108.0]
        low = [99.0] * 15 + [92.0]
        self.assertAlmostEqual(features.wilder_atr(pd.Series(high), pd.Series(low), pd.Series(close)), 3.0)

    def test_too_little_history_gives_nan(self):
        s = pd.Series([1.0] * 10)
        self.assertTrue(math.isnan(features.wilder_atr(s, s, s)))


class PriceFeaturesTest(unittest.TestCase):
    def test_beta_to_the_sector_etf(self):
        self.assertAlmostEqual(compute().loc["AAA", "beta"], 2.0, places=6)

    def test_gap_explained_by_the_sector_leaves_no_excess_gap(self):
        row = compute().loc["AAA"]
        self.assertAlmostEqual(row["gap_pct"], 2.0, places=6)
        self.assertAlmostEqual(row["sector_gap_pct"], 1.0, places=6)
        self.assertAlmostEqual(row["excess_gap_pct"], 0.0, places=6)
        self.assertAlmostEqual(row["gap_atr"], 0.0, places=6)

    def test_excess_gap_is_measured_in_atr_units(self):
        row = compute(minute=minute_frame(daily_frame(), stock_gap=0.06)).loc["AAA"]
        self.assertAlmostEqual(row["excess_gap_pct"], 4.0, places=6)          # 6% - 2 x 1%
        self.assertAlmostEqual(row["gap_atr"], 4.0 / row["atr_pct"], places=6)

    def test_relative_volume_against_the_same_window_on_prior_sessions(self):
        self.assertAlmostEqual(compute().loc["AAA", "rvol"], 3.0, places=2)

    def test_bars_after_the_data_cutoff_are_ignored(self):
        daily = daily_frame()
        late = [("AAA", dt.datetime(2026, 11, 20, 8, 50, tzinfo=NY), 999.0, 1e6)]  # after 08:39
        row = compute(daily, minute_frame(daily, extra=late)).loc["AAA"]
        self.assertAlmostEqual(row["gap_pct"], 2.0, places=6)

    def test_liquid_stock_passes_all_filters(self):
        row = compute().loc["AAA"]
        self.assertGreater(row["adv_usd"], 20e6)
        self.assertGreater(row["premarket_usd"], 250e3)
        self.assertTrue(row["liquid"])

    def test_thin_premarket_trading_fails_the_liquidity_filter(self):
        daily = daily_frame()
        row = compute(daily, minute_frame(daily, stock_volume=100)).loc["AAA"]  # ~$5k traded
        self.assertFalse(row["liquid"])

    def test_unadjusted_split_shows_as_a_suspect_gap_and_is_excluded(self):
        daily = daily_frame()
        prev_close = daily.loc[("AAA", DATES[-1]), "close"]
        row = compute(daily, minute_frame(daily, stock_gap=-0.5)).loc["AAA"]  # 2-for-1 split not adjusted
        self.assertAlmostEqual(row["gap_pct"], -50.0, places=6)
        self.assertTrue(row["suspect_gap"])
        self.assertFalse(row["liquid"])
        self.assertGreater(prev_close, 0)

    def test_no_premarket_trade_means_no_reference_price(self):
        daily = daily_frame()
        row = compute(daily, minute_frame(daily, stock_gap=None)).loc["AAA"]
        self.assertTrue(math.isnan(row["ref_price"]))
        self.assertFalse(row["liquid"])

    def test_trend_and_52_week_high(self):
        row = compute().loc["AAA"]
        closes = daily_frame().loc["AAA", "close"]
        self.assertAlmostEqual(row["trend"], row["ref_price"] / closes.iloc[-50:].mean() - 1, places=6)
        highs = daily_frame().loc["AAA", "high"]
        self.assertAlmostEqual(row["dist_52w_high"], row["ref_price"] / highs.iloc[-252:].max() - 1, places=6)


if __name__ == "__main__":
    unittest.main()


class LoaderTest(unittest.TestCase):
    def test_loader_fetches_today_and_the_same_window_on_each_prior_session(self):
        calls = []

        class FakeMD:
            @staticmethod
            def minute_bars(symbols, start, end, as_of=None, **kw):
                calls.append(("minute", start, end, as_of, tuple(symbols)))
                return minute_frame(daily_frame()).iloc[0:0]

            @staticmethod
            def daily_bars(symbols, start, end, **kw):
                calls.append(("daily", start, end, None, tuple(symbols)))
                return daily_frame()

        features.load_price_inputs(MEMBERS, SESSION, PREV, PRIOR, AS_OF, md=FakeMD)
        daily = [c for c in calls if c[0] == "daily"]
        minute = [c for c in calls if c[0] == "minute"]
        self.assertEqual(daily[0][2], PREV.date)                     # never the session's own daily bar
        self.assertIn("XLI", daily[0][4])                            # sector ETFs included
        self.assertEqual(len(minute), 1 + len(PRIOR))
        today = minute[0]
        self.assertEqual((today[1].hour, today[1].minute), (4, 0))
        self.assertEqual(today[3], AS_OF)                            # today's window obeys the data delay
        for _, start, end, _, _ in minute[1:]:
            self.assertEqual((end.hour, end.minute), (8, 39))        # same clock window as today


class FutureDataTest(unittest.TestCase):
    def test_daily_bars_after_the_previous_session_change_nothing(self):
        # The backtest passes one daily frame covering all of history to every session.
        daily = daily_frame()
        future = []
        for sym, price in (("AAA", 500.0), ("XLI", 10.0)):
            for d in pd.bdate_range("2026-11-20", periods=30):
                future.append((sym, d.date(), price, price * 3, price / 3, price, 9e9))
        extra = pd.DataFrame(future, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        with_future = pd.concat([daily, extra.set_index(["symbol", "date"])]).sort_index()
        base = compute(daily, minute_frame(daily))
        leaked = compute(with_future, minute_frame(daily))
        cols = ["gap_pct", "beta", "atr_pct", "gap_atr", "trend", "dist_52w_high", "adv_usd", "sector_gap_pct"]
        pd.testing.assert_frame_equal(base[cols], leaked[cols])
