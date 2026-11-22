import datetime as dt
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
from backtest import build_dataset as bd
from market_session import Session

NY = ZoneInfo("America/New_York")


def session(d: dt.date, close=dt.time(16)) -> Session:
    return Session(d, dt.datetime.combine(d, dt.time(9, 30), NY), dt.datetime.combine(d, close, NY))


SESSIONS = [session(d.date()) for d in pd.bdate_range("2026-04-01", "2026-06-30")]


class EvalSessionsTest(unittest.TestCase):
    def test_first_session_has_a_full_baseline_of_news(self):
        pairs = bd.eval_sessions(SESSIONS, news_start=dt.date(2026, 4, 29))
        first, before = pairs[0]
        self.assertEqual(len(before), 21)
        self.assertGreaterEqual(before[0].date, dt.date(2026, 4, 29))   # baseline window starts inside the news history
        self.assertEqual(before[-1], SESSIONS[SESSIONS.index(first) - 1])

    def test_every_later_session_is_included(self):
        pairs = bd.eval_sessions(SESSIONS, news_start=dt.date(2026, 4, 29))
        self.assertEqual(pairs[-1][0], SESSIONS[-1])
        self.assertEqual(len(pairs), len(SESSIONS) - SESSIONS.index(pairs[0][0]))


class ScanTimeTest(unittest.TestCase):
    def test_scan_is_35_minutes_before_the_open_also_on_half_days(self):
        self.assertEqual(bd.scan_time(session(dt.date(2026, 11, 27), close=dt.time(13))),
                         dt.datetime(2026, 11, 27, 8, 55, tzinfo=NY))


def art(minutes_after_open, symbol="AAA"):
    return {"id": minutes_after_open, "symbol": symbol, "headline": "h", "summary": "secret summary",
            "text": "secret body", "created_at": dt.datetime(2026, 6, 1, 9, 30, tzinfo=NY)
            + dt.timedelta(minutes=minutes_after_open), "s": 0.5, "w": 0.9}


class NewsWindowTest(unittest.TestCase):
    def test_window_excludes_its_start_includes_its_end_and_filters_symbols(self):
        arts = [art(-10), art(0), art(5), art(10), art(5, symbol="ZZZ")]
        start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=NY)
        got = bd.news_in_window(arts, ["AAA"], start, start + dt.timedelta(minutes=5))
        self.assertEqual([a["id"] for a in got], [5])

    def test_article_text_is_removed_before_anything_is_saved(self):
        kept = bd.strip_text([art(1)])[0]
        self.assertNotIn("text", kept)
        self.assertNotIn("summary", kept)
        self.assertEqual((kept["headline"], kept["s"]), ("h", 0.5))


class OpeningVolumeTest(unittest.TestCase):
    def test_first_five_minute_volume_relative_to_the_previous_14_sessions(self):
        days = [d.date() for d in pd.bdate_range("2026-06-01", periods=16)]
        first5 = {d: pd.Series({"AAA": 1000.0, "BBB": 50.0}) for d in days[:15]}
        first5[days[15]] = pd.Series({"AAA": 3000.0, "BBB": 50.0})
        rvol5 = bd.opening_rvol(first5, days[15], ["AAA", "BBB", "CCC"])
        self.assertAlmostEqual(rvol5["AAA"], 3.0)
        self.assertAlmostEqual(rvol5["BBB"], 1.0)
        self.assertTrue(pd.isna(rvol5["CCC"]))

    def test_opening_volume_never_uses_the_future(self):
        days = [d.date() for d in pd.bdate_range("2026-06-01", periods=16)]
        first5 = {d: pd.Series({"AAA": 1000.0}) for d in days}
        first5[days[15]] = pd.Series({"AAA": 1e9})      # a later day must not enter day 10's baseline
        self.assertAlmostEqual(bd.opening_rvol(first5, days[10], ["AAA"])["AAA"], 1.0)


if __name__ == "__main__":
    unittest.main()
