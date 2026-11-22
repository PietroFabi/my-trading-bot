import contextlib
import datetime as dt
import io
import os
import tempfile
import unittest

import pandas as pd

import tests  # noqa: F401
import alerts
from tests import test_morning as tm
from tests.test_morning import FakeDeps, at, stock


def table(n=100, daily_missing=0, premarket_missing=0, news=True):
    rows = []
    for i in range(n):
        r = stock(f"S{i}", 1.0 if i % 2 else -1.0, sector=f"sec{i % 11}")
        if i < daily_missing:
            r["prev_close"] = r["atr_pct"] = float("nan")
        if i < premarket_missing:
            r["ref_price"] = float("nan")
        if not news:
            r["n_fresh"] = 0
        rows.append(r)
    return pd.DataFrame(rows).set_index("symbol")


class CoverageTest(unittest.TestCase):
    def test_a_normal_morning_raises_no_notice(self):
        self.assertEqual(alerts.coverage_notices(table(premarket_missing=28)), [])   # 72% priced is normal

    def test_daily_data_below_70_percent_is_a_notice(self):
        notes = alerts.coverage_notices(table(daily_missing=40))
        self.assertEqual(len(notes), 1)
        self.assertIn("60%", notes[0])

    def test_premarket_prices_far_below_normal_is_a_notice(self):
        self.assertEqual(len(alerts.coverage_notices(table(premarket_missing=65))), 1)

    def test_no_fresh_news_at_all_is_a_notice(self):
        self.assertEqual(len(alerts.coverage_notices(table(news=False))), 1)

    def test_coverage_numbers_are_reported(self):
        c = alerts.coverage(table(daily_missing=10, premarket_missing=30))
        self.assertEqual((c["daily"], c["premarket"]), (0.9, 0.7))


class MorningNoticeTest(unittest.TestCase):
    setUp, tearDown = tm.MorningTest.setUp, tm.MorningTest.tearDown
    run_job, plan = tm.MorningTest.run_job, tm.MorningTest.plan

    def test_low_coverage_is_a_notice_and_the_run_still_succeeds(self):
        rows = [dict(stock(f"S{i}", 0.1), prev_close=float("nan")) for i in range(6)] + \
               [stock("AAA", 1.0), stock("BBB", 0.1), stock("CCC", 0.1), stock("DDD", 0.1)]
        deps = FakeDeps(at(8, 0), rows=rows)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = self.run_job(deps)
        self.assertEqual(code, 0)
        self.assertIn("::notice::", out.getvalue())
        self.assertEqual(self.plan()["coverage"]["daily"], 0.4)
        self.assertEqual(len(deps.placed), 1)                        # it still trades what it can

    def test_the_rules_that_need_you_still_fail(self):
        self.assertEqual(self.run_job(FakeDeps(at(9, 26))), 1)       # started after the cutoff
        self.assertEqual(self.run_job(FakeDeps(at(8, 0), leftovers=([], ["ZZZ"]))), 1)


def candidates(day, scores):
    return pd.DataFrame({"session_date": day, "symbol": [f"S{i}" for i in range(len(scores))], "score": scores,
                         "target": 0.0})


class DriftTest(unittest.TestCase):
    BAND = {"window": 5, "mean": [-0.002, 0.002], "std": [0.0005, 0.003]}

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def write(self, days, scores):
        for d in days:
            candidates(d, scores).to_csv(os.path.join(self.dir.name, f"{d}.csv"), index=False)

    DAYS = [dt.date(2026, 12, d) for d in (1, 2, 3, 4, 7)]

    def test_scores_inside_the_hold_out_band_raise_no_notice(self):
        self.write(self.DAYS, [-0.001, 0.0, 0.001, 0.002, -0.002])
        self.assertIsNone(alerts.score_drift_notice(self.dir.name, self.BAND))

    def test_a_shifted_score_distribution_is_a_notice(self):
        self.write(self.DAYS, [0.010, 0.011, 0.012, 0.013, 0.014])
        note = alerts.score_drift_notice(self.dir.name, self.BAND)
        self.assertIn("mean", note)

    def test_a_collapsed_spread_is_a_notice(self):
        self.write(self.DAYS, [0.0001] * 5)
        self.assertIn("spread", alerts.score_drift_notice(self.dir.name, self.BAND))

    def test_fewer_sessions_than_the_window_raise_nothing(self):
        self.write(self.DAYS[:3], [0.010, 0.011, 0.012])
        self.assertIsNone(alerts.score_drift_notice(self.dir.name, self.BAND))

    def test_the_band_is_the_95_percent_range_of_rolling_windows(self):
        days = [d.date() for d in pd.bdate_range("2026-10-09", periods=12)]
        frame = pd.concat([candidates(d, [i / 1000, i / 1000 + 0.002, i / 1000 - 0.002]) for i, d in enumerate(days)])
        band = alerts.score_band(frame, window=5)               # built from replayed sessions
        self.assertEqual(band["window"], 5)
        self.assertEqual(band["windows"], 8)
        self.assertLess(band["mean"][0], band["mean"][1])


if __name__ == "__main__":
    unittest.main()
