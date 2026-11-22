import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
import main
import model
from market_session import Session

NY = ZoneInfo("America/New_York")
FRIDAY = dt.date(2026, 11, 20)


def at(hh, mm, day=FRIDAY):
    return dt.datetime.combine(day, dt.time(hh, mm), NY)


def calendar(half_day=None):
    days = [d.date() for d in pd.bdate_range("2026-09-01", "2026-12-31")]
    return [Session(d, at(9, 30, d), at(13, 0, d) if d == half_day else at(16, 0, d)) for d in days]


def stock(symbol, gap_atr, sector="Tech"):
    up = gap_atr > 0
    return {"symbol": symbol, "sector": sector, "liquid": True, "gate_long": up, "gate_short": not up,
            "ref_price": 100.0, "prev_close": 100.0, "atr_pct": 2.0, "gap_pct": gap_atr * 2, "excess_gap_pct": gap_atr * 2,
            "n_fresh": 1, "gap_atr": gap_atr, "rvol": 1.0, "trend": 0.0, "dist_52w_high": -0.1,
            "news_today": 0.5 if up else -0.5, "baseline": 0.3, "surprise": 0.1, "news_volume": 0.0}


class FakeDeps:
    def __init__(self, start, rows=None, half_day=None, leftovers=((), ()), build_minutes=0):
        self.t, self.cal = start, calendar(half_day)
        self.rows = rows if rows is not None else [stock("AAA", 1.0), stock("BBB", 0.1)]
        self.leftovers, self.build_minutes = leftovers, build_minutes
        self.slept, self.as_of, self.placed, self.calls = [], None, [], []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += dt.timedelta(seconds=seconds)

    def sessions(self, start, end):
        return [s for s in self.cal if start <= s.date <= end]

    def evaluate_pending(self, now, plans_dir):
        self.calls.append("evaluate")

    def close_leftovers(self, plans_dir):
        self.calls.append("leftovers")
        return [list(x) for x in self.leftovers]

    def members(self, day):
        return [{"symbol": r["symbol"]} for r in self.rows]

    def warm_news(self, members, before, now):
        self.calls.append("warm")

    def feature_table(self, members, session, before, as_of):
        self.as_of, self.before = as_of, before
        self.t += dt.timedelta(minutes=self.build_minutes)
        return pd.DataFrame(self.rows).set_index("symbol")

    def equity(self):
        return 100_000.0

    def is_shortable(self, symbol):
        return True

    def place_entries(self, plan, skip):
        self.placed.append((plan, set(skip)))
        for t in plan["targets"]:
            t["order_id"] = f"id-{t['ticker']}"

    def save_cache(self, before):
        self.calls.append("save_cache")

    def live_strategy(self):
        return "A"


class MorningTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        self.plans, self.dry, self.scores = (os.path.join(d, x) for x in ("plans", "dryrun", "scores"))
        self.model, self.prereg = os.path.join(d, "frozen_model.json"), os.path.join(d, "preregistration.json")
        ridge = {"lam": 1.0, "clip_y": None, "clip_z": 4.0, "mean": [0.0] * 8, "std": [1.0] * 8,
                 "coef": [0.01] + [0.0] * 7, "intercept": 0.0}
        body = {"strategies": {"A": {"kind": "ridge", "features": model.FEATURES, "exclude_price_move": False,
                                     "ridge": ridge, "threshold_return": 0.005, "entry": "open", "stop_k": 0.75,
                                     "exit": "close"}}}
        with open(self.model, "w", newline="\n") as f:
            json.dump(body, f)
        with open(self.model, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        with open(self.prereg, "w") as f:
            json.dump({"frozen_model_sha256": digest}, f)

    def tearDown(self):
        self.dir.cleanup()

    def run_job(self, deps, dry_run=False):
        return main.run_morning_strategy(deps.now(), plans_dir=self.plans, dry_run=dry_run, deps=deps,
                                         dryrun_dir=self.dry, scores_dir=self.scores, model_file=self.model,
                                         prereg_file=self.prereg)

    def plan(self, folder=None, day=FRIDAY):
        with open(os.path.join(folder or self.plans, f"{day}.json")) as f:
            return json.load(f)

    # --- unchanged calendar behaviour

    def test_weekend_does_nothing(self):
        deps = FakeDeps(at(8, 0, dt.date(2026, 11, 21)))
        self.assertEqual(self.run_job(deps), 0)
        self.assertIsNone(deps.as_of)

    def test_an_existing_plan_means_an_earlier_attempt_already_ran(self):
        os.makedirs(self.plans)
        with open(os.path.join(self.plans, f"{FRIDAY}.json"), "w") as f:
            f.write("{}")
        deps = FakeDeps(at(7, 45))
        self.assertEqual(self.run_job(deps), 0)
        self.assertEqual(deps.placed, [])

    def test_too_early_exits_quietly(self):
        deps = FakeDeps(at(6, 30))
        self.assertEqual(self.run_job(deps), 0)
        self.assertIsNone(deps.as_of)

    def test_starting_after_the_cutoff_skips_the_day_and_fails(self):
        deps = FakeDeps(at(9, 26))
        self.assertEqual(self.run_job(deps), 1)
        self.assertEqual(deps.placed, [])

    # --- waiting for the scan

    def test_waits_until_35_minutes_before_the_open_then_scans_as_of_then(self):
        deps = FakeDeps(at(7, 45))
        self.assertEqual(self.run_job(deps), 0)
        self.assertEqual(deps.slept, [70 * 60])
        self.assertEqual(deps.as_of, at(8, 55))
        self.assertEqual(deps.calls.index("warm") < deps.calls.index("save_cache"), True)

    def test_a_half_day_uses_the_same_scan_time_from_the_calendar(self):
        deps = FakeDeps(at(8, 0), half_day=FRIDAY)
        self.run_job(deps)
        self.assertEqual(deps.as_of, at(8, 55))
        self.assertEqual(self.plan()["market_close"], at(13, 0).isoformat())

    def test_a_late_start_builds_the_same_08_55_table_without_waiting(self):
        deps = FakeDeps(at(9, 5))
        self.assertEqual(self.run_job(deps), 0)
        self.assertEqual((deps.slept, deps.as_of), ([], at(8, 55)))
        self.assertEqual(len(deps.placed), 1)

    def test_the_baseline_gets_the_21_sessions_before_today(self):
        deps = FakeDeps(at(8, 0))
        self.run_job(deps)
        self.assertEqual(len(deps.before), 21)
        self.assertEqual(deps.before[-1].date, dt.date(2026, 11, 19))

    # --- the plan

    def test_a_normal_morning_saves_a_v2_plan_places_orders_and_logs_every_scanned_stock(self):
        deps = FakeDeps(at(8, 0))
        self.assertEqual(self.run_job(deps), 0)
        p = self.plan()
        self.assertEqual((p["strategy"], p["strategy_name"], p["entry"], p["stop_k"]), ("v2", "A", "open", 0.75))
        self.assertEqual(len(p["model_sha256"]), 64)
        self.assertEqual([(t["ticker"], t["side"], t["qty"], t["order_id"]) for t in p["targets"]],
                         [("AAA", "long", 30, "id-AAA")])
        self.assertEqual({c["ticker"]: c["decision"] for c in p["candidates"]},
                         {"AAA": "selected", "BBB": "below threshold"})
        with open(os.path.join(self.scores, f"{FRIDAY}.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["symbol"] for r in rows], ["AAA", "BBB"])
        self.assertIn("score", rows[0])
        self.assertIn("decision", rows[0])

    def test_a_zero_signal_day_saves_an_empty_plan_and_places_no_orders(self):
        deps = FakeDeps(at(8, 0), rows=[stock("AAA", 0.1), stock("BBB", -0.2)])
        self.assertEqual(self.run_job(deps), 0)
        self.assertEqual(self.plan()["targets"], [])
        self.assertEqual(deps.placed, [])

    def test_leftovers_being_closed_are_not_bought_and_a_failed_close_fails_the_run(self):
        deps = FakeDeps(at(8, 0), leftovers=(["AAA"], ["ZZZ"]))
        self.assertEqual(self.run_job(deps), 1)
        self.assertEqual(deps.placed[0][1], {"AAA", "ZZZ"})

    def test_when_building_runs_past_the_cutoff_no_orders_are_placed(self):
        deps = FakeDeps(at(9, 10), build_minutes=20)               # done at 09:30, after the 09:25 cutoff
        self.assertEqual(self.run_job(deps), 1)
        self.assertEqual(deps.placed, [])
        self.assertEqual(self.plan()["skipped"], "plan finished after the order cutoff")

    def test_a_frozen_model_that_does_not_match_the_preregistration_stops_the_morning(self):
        with open(self.model, "a") as f:
            f.write(" ")
        deps = FakeDeps(at(8, 0))
        self.assertEqual(self.run_job(deps), 1)
        self.assertEqual(deps.placed, [])
        self.assertFalse(os.path.exists(os.path.join(self.plans, f"{FRIDAY}.json")))

    # --- dry run

    def test_a_dry_run_places_no_orders_and_writes_only_to_the_dry_run_folder(self):
        deps = FakeDeps(at(8, 0))
        self.assertEqual(self.run_job(deps, dry_run=True), 0)
        self.assertEqual(deps.placed, [])
        self.assertFalse(os.path.exists(self.plans))
        self.assertNotIn("leftovers", deps.calls)
        self.assertNotIn("evaluate", deps.calls)
        self.assertTrue(self.plan(self.dry)["dry_run"])
        self.assertTrue(os.path.exists(os.path.join(self.dry, f"scores-{FRIDAY}.csv")))


if __name__ == "__main__":
    unittest.main()
