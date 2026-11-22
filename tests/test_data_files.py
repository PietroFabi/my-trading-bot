"""Every data file the site or a reader may use: its documented keys exist in the real file (or in what the code
writes), and the README mentions each of them, so the documentation cannot drift from the data."""
import csv
import json
import os
import tempfile
import unittest

import tests  # noqa: F401
import evaluator_v2 as ev2
from tests.test_evaluator_v2 import FRI, bars_for, target, trade_day, v2_plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTED = {
    "data/preregistration.json": ["frozen_on", "frozen_model_file", "frozen_model_sha256", "hypotheses",
                                  "variants_tried", "holdout", "decision_rule", "live_confirmation", "costs"],
    "data/model/frozen_model.json": ["version", "data_cutoff", "finbert", "scan_before_open_min", "selection", "sizing",
                                     "costs", "strategies"],
    "data/backtest/exploration.json": ["window", "variants_tried", "stage1", "stage1_choice", "stage2", "stage2_choice",
                                       "chosen_rule_by_side_and_horizon", "claimed_candidates", "ic_by_month",
                                       "benchmarks", "final_model", "diagnostics"],
    "data/backtest/holdout.json": ["holdout", "pass_rule", "A", "fade", "goes_live", "benchmarks"],
    "data/backtest/score_band.json": ["window", "windows", "sessions", "mean", "std", "strategy", "model_sha256"],
    "data/backtest/dataset_summary.json": ["sessions", "first_session", "last_session", "distinct_stocks", "articles",
                                           "mean_per_session", "gated_rows_total"],
}
RESULTS_V2_ENTRY = ["strategy", "strategy_name", "model_sha256", "score_thresholds", "costs_bps", "portfolio_curve_net",
                    "ending_balance_net", "by_side"]
RESULTS_V2_TRADE = ["side", "score", "features", "stop_price", "exit_reason", "performance_pct_net", "pnl_usd_net"]
CONFIRMATION = ["sessions", "trades", "required", "progress", "ic", "ic_t", "mean_net_per_trade_pct", "mean_net_t",
                "primary_passed", "secondary_passed", "status", "first_session", "updated_after"]
CANDIDATES_CSV = ["session_date", "symbol", "gate", "sector_etf", "beta", "score", "decision", "target", "traded"]
PLAN_V2 = ["strategy", "strategy_name", "model_sha256", "scan_as_of", "entry", "stop_k", "score_threshold", "equity",
           "coverage", "candidates", "targets"]


def readme() -> str:
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        return f.read()


class CommittedFilesTest(unittest.TestCase):
    def test_every_documented_key_exists_in_the_committed_file(self):
        for path, keys in DOCUMENTED.items():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                data = json.load(f)
            for k in keys:
                self.assertIn(k, data, f"{path}: {k}")

    def test_the_readme_names_every_file_and_key(self):
        text = readme()
        for path, keys in DOCUMENTED.items():
            self.assertIn(path, text)
            for k in keys:
                self.assertIn(f"`{k}`", text, f"README does not mention `{k}` ({path})")


class WrittenByTheBotTest(unittest.TestCase):
    """Files that appear only once v2 trades: checked against what the code actually writes."""

    def setUp(self):
        self.plan = v2_plan(FRI, target("LLL"))
        self.client = trade_day(self.plan, exit_px=101.0)
        self.entry = ev2.evaluate_plan_v2(self.plan, 100_000.0, self.client, bars_for(FRI, {"LLL": 101.0}))

    def test_v2_results_fields_are_written_and_documented(self):
        text = readme()
        for k in RESULTS_V2_ENTRY:
            self.assertIn(k, self.entry)
            self.assertIn(f"`{k}`", text, k)
        for k in RESULTS_V2_TRADE:
            self.assertIn(k, self.entry["trades"][0])
            self.assertIn(f"`{k}`", text, k)

    def test_confirmation_and_candidate_log_are_written_and_documented(self):
        text = readme()
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "scores"))
            with open(os.path.join(d, "scores", f"{FRI}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["symbol", "sector", "sector_etf", "beta", "gate_long", "gate_short", "score", "decision",
                            "gap_atr", "news_today", "surprise"])
                w.writerow(["LLL", "Tech", "XLK", 1.0, True, False, 0.004, "selected", 0.5, 0.3, 0.1])
            path = ev2.log_candidates(self.plan, self.entry, bars_for(FRI, {"LLL": 101.0, "XLK": 50.0}),
                                      os.path.join(d, "scores"), os.path.join(d, "candidates"))
            with open(path) as f:
                header = next(csv.reader(f))
            conf = ev2.update_confirmation([self.entry], os.path.join(d, "candidates"), os.path.join(d, "c.json"),
                                           os.path.join(ROOT, "data", "preregistration.json"))
        for k in CANDIDATES_CSV:
            self.assertIn(k, header)
            self.assertIn(f"`{k}`", text, k)
        for k in CONFIRMATION:
            self.assertIn(k, conf)
            self.assertIn(f"`{k}`", text, k)
        for k in PLAN_V2:
            self.assertIn(f"`{k}`", text, k)
        for name in ("data/scores/", "data/candidates/", "data/confirmation.json", "data/sentiment/finbert_cache.json"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
