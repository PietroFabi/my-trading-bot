import datetime as dt
import math
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
import features
from market_session import Session

NY = ZoneInfo("America/New_York")


def session(d: dt.date) -> Session:
    return Session(d, dt.datetime.combine(d, dt.time(9, 30), NY), dt.datetime.combine(d, dt.time(16), NY))


# Monday 2026-11-23; the 21 sessions before it, oldest first (the last one is Friday 11-20).
TODAY = session(dt.date(2026, 11, 23))
BEFORE = [session(d.date()) for d in pd.bdate_range(end="2026-11-20", periods=21)]
AS_OF = dt.datetime(2026, 11, 23, 8, 55, tzinfo=NY)
_ids = iter(range(1, 10_000))


def art(when: dt.datetime, s: float, w: float = 0.95, symbol="AAA", price_move=False):
    return {"id": next(_ids), "symbol": symbol, "created_at": when, "s": s, "w": w, "price_move": price_move,
            "headline": "h"}


def fresh(minutes_before_scan=60, **kw):
    return art(AS_OF - dt.timedelta(minutes=minutes_before_scan), **kw)


def on_session(sessions_ago: int, **kw):
    """An article published at 12:00 during the session `sessions_ago` sessions before today."""
    return art(dt.datetime.combine(BEFORE[-sessions_ago].date, dt.time(12), NY), **kw)


def compute(articles, **kw):
    return features.news_features(articles, TODAY, BEFORE, AS_OF, **kw)


class NewsFeaturesTest(unittest.TestCase):
    def test_one_fresh_article_on_a_zero_baseline_moves_surprise_by_a_third(self):
        row = compute([fresh(s=0.9)]).loc["AAA"]
        self.assertEqual(row["n_fresh"], 1)
        self.assertAlmostEqual(row["news_today"], 0.9)
        self.assertAlmostEqual(row["baseline"], 0.0)
        self.assertAlmostEqual(row["surprise"], 0.9 / 3)        # n/(n+2) with n = 1

    def test_five_fresh_articles_move_it_by_five_sevenths(self):
        row = compute([fresh(minutes_before_scan=10 * i + 5, s=0.9) for i in range(5)]).loc["AAA"]
        self.assertAlmostEqual(row["surprise"], 0.9 * 5 / 7)

    def test_neutral_filler_barely_dilutes_real_news(self):
        arts = [fresh(s=0.8, w=0.9)] + [fresh(minutes_before_scan=90 + i, s=0.02, w=0.05) for i in range(3)]
        row = compute(arts).loc["AAA"]
        self.assertGreater(row["news_today"], 0.65)             # a plain mean would give 0.215

    def test_baseline_is_exponentially_weighted_with_a_five_session_half_life(self):
        # yesterday's tone +0.5 weighs twice as much as -0.5 from 6 sessions ago -> (2*0.5 - 0.5) / 3
        row = compute([fresh(s=0.0, w=0.01), on_session(1, s=0.5), on_session(6, s=-0.5)]).loc["AAA"]
        self.assertAlmostEqual(row["baseline"], 1 / 6)

    def test_surprise_is_measured_against_the_baseline(self):
        row = compute([fresh(s=0.5), on_session(1, s=0.5)]).loc["AAA"]
        self.assertAlmostEqual(row["surprise"], 0.0)

    def test_news_volume_compares_fresh_count_with_the_normal_daily_count(self):
        normal = [on_session(k, s=0.1) for k in range(1, 21)]   # one article per session
        row = compute(normal + [fresh(minutes_before_scan=5 + i, s=0.1) for i in range(3)]).loc["AAA"]
        self.assertAlmostEqual(row["news_volume"], math.log((3 + 1) / (1 + 1)))

    def test_articles_after_the_scan_are_ignored(self):
        row = compute([fresh(s=0.5), art(AS_OF + dt.timedelta(minutes=1), s=-0.9)]).loc["AAA"]
        self.assertEqual(row["n_fresh"], 1)
        self.assertAlmostEqual(row["news_today"], 0.5)

    def test_monday_news_window_starts_at_fridays_close(self):
        saturday = art(dt.datetime(2026, 11, 21, 10, tzinfo=NY), s=0.6)
        friday_afternoon = art(dt.datetime(2026, 11, 20, 15, tzinfo=NY), s=-0.6)
        row = compute([saturday, friday_afternoon]).loc["AAA"]
        self.assertEqual(row["n_fresh"], 1)
        self.assertAlmostEqual(row["news_today"], 0.6)
        self.assertAlmostEqual(row["baseline"], -0.6)

    def test_price_move_headlines_can_be_excluded_but_are_always_counted(self):
        arts = [fresh(s=0.5), fresh(minutes_before_scan=30, s=-0.9, price_move=True)]
        kept = compute(arts).loc["AAA"]
        dropped = compute(arts, exclude_price_move=True).loc["AAA"]
        self.assertEqual((kept["n_fresh"], dropped["n_fresh"]), (2, 1))
        self.assertAlmostEqual(dropped["news_today"], 0.5)
        self.assertEqual(dropped["n_price_move"], 1)


class FeatureTableTest(unittest.TestCase):
    def price(self, gap, liquid=True, excess=None):
        excess = gap if excess is None else excess
        return pd.DataFrame([{"symbol": s, "gap_pct": g, "excess_gap_pct": e, "liquid": liquid, "sector": "Industrials"}
                             for s, g, e in zip(["AAA", "BBB", "CCC"], gap, excess)]).set_index("symbol")

    def test_gates_need_gap_and_fresh_news_in_the_same_direction(self):
        news = compute([fresh(s=0.7, symbol="AAA"), fresh(s=0.7, symbol="BBB"), fresh(s=-0.7, symbol="CCC")])
        table = features.build_feature_table(self.price([2.0, -2.0, -3.0]), news)
        self.assertEqual(table.loc["AAA", ["gate_long", "gate_short"]].tolist(), [True, False])
        self.assertEqual(table.loc["BBB", ["gate_long", "gate_short"]].tolist(), [False, False])  # good news, gap down
        self.assertEqual(table.loc["CCC", ["gate_long", "gate_short"]].tolist(), [False, True])

    def test_gates_use_the_sector_adjusted_gap_not_the_raw_gap(self):
        # MRVL on 2026-11-20: raw gap -5.7% but +0.2% after removing the chip sector's drop
        news = compute([fresh(s=-0.7, symbol="AAA"), fresh(s=0.7, symbol="BBB")])
        table = features.build_feature_table(self.price([-5.7, -1.0, 0.0], excess=[0.2, 1.5, 0.0]), news)
        self.assertEqual(table.loc["AAA", ["gate_long", "gate_short"]].tolist(), [False, False])
        self.assertEqual(table.loc["BBB", ["gate_long", "gate_short"]].tolist(), [True, False])

    def test_no_fresh_news_means_no_gate_and_neutral_news_features(self):
        table = features.build_feature_table(self.price([5.0, 1.0, 1.0]), compute([on_session(2, s=0.4)]))
        self.assertFalse(table.loc["BBB", "gate_long"])
        self.assertEqual(table.loc["BBB", ["n_fresh", "news_today", "surprise", "news_volume"]].tolist(), [0, 0, 0, 0])
        self.assertFalse(table.loc["AAA", "gate_long"])          # only old news, nothing fresh

    def test_illiquid_stocks_are_never_gated(self):
        table = features.build_feature_table(self.price([2.0, 2.0, 2.0], liquid=False),
                                             compute([fresh(s=0.7, symbol="AAA")]))
        self.assertFalse(table.loc["AAA", "gate_long"])


if __name__ == "__main__":
    unittest.main()


class MorningTableTest(unittest.TestCase):
    def test_one_call_fetches_scores_and_joins_everything_for_one_morning(self):
        calls = {}
        members = [{"symbol": "AAA", "yahoo": "AAA", "sector": "Industrials", "sector_etf": "XLI"}]

        def fake_prices(members_, session_, prev_, prior_, as_of_, cache_dir=None, fallback=True):
            calls["prior"] = prior_
            return "daily", "minute"

        def fake_price_features(daily, minute, members_, session_, prev_, prior_, as_of_):
            return pd.DataFrame([{"symbol": "AAA", "gap_pct": 3.0, "excess_gap_pct": 2.0, "liquid": True,
                                  "sector": "Industrials"}]
                                ).set_index("symbol")

        def fake_fetch(symbols, start, end):
            calls["window"] = (start, end)
            return [fresh(s=0.0)]

        def fake_score(articles, cache=None):
            for a in articles:
                a.update(s=0.8, w=0.9)
            return articles

        table = features.morning_feature_table(
            members, TODAY, BEFORE, AS_OF, load_prices=fake_prices, compute_prices=fake_price_features,
            fetch=fake_fetch, score=fake_score)
        self.assertEqual(calls["window"], (BEFORE[0].close, AS_OF))     # baseline start .. scan time
        self.assertEqual(calls["prior"], BEFORE[-10:])                  # 10 sessions for relative volume
        self.assertTrue(table.loc["AAA", "gate_long"])
