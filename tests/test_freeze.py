import hashlib
import json
import os
import subprocess
import tempfile
import unittest

import tests  # noqa: F401
from backtest import freeze, holdout

EXPLORATION = {
    "window": {"first": "2026-05-29", "last": "2026-10-08", "oof_first": "2026-07-01", "oof_sessions": 70,
               "holdout_start": "2026-10-09"},
    "variants_tried": {"total": 32},
    "stage1_choice": {"lam": 1.0, "exclude_price_move": False, "ic": -0.02, "t": -0.85},
    "stage2_choice": {"entry": "open", "stop": 0.75, "threshold": 1.5, "n": 91, "mean": -0.0002, "t": -0.07},
    "final_model": {"features": ["gap_atr"], "exclude_price_move": False, "lam": 1.0, "clip_y": [0.01, 0.99],
                    "clip_z": 4.0, "mean": [0.1], "std": [0.6], "coef": [-0.002], "intercept": -0.0007,
                    "in_sample_pred_std": 0.002, "threshold_return": 0.003, "training_rows": 4058},
}
FADE_IN_SAMPLE = {"ic": [0.059, 2.36], "eod_net": {"n": 689, "mean": 0.0034, "t": 2.92}}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


class FreezeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.repo = self.dir.name
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "Test")
        self.model = os.path.join(self.repo, "data", "model", "frozen_model.json")
        self.prereg = os.path.join(self.repo, "data", "preregistration.json")
        self.out = os.path.join(self.repo, "data", "backtest", "holdout.json")
        freeze.write_freeze(EXPLORATION, FADE_IN_SAMPLE, self.model, self.prereg, frozen_on="2026-11-22")

    def tearDown(self):
        self.dir.cleanup()

    def commit(self):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "pre-registration")

    def guard(self):
        holdout.check_preregistered(self.repo, self.model, self.prereg, self.out)

    def test_preregistration_records_the_model_files_hash_and_both_hypotheses(self):
        with open(self.model, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        with open(self.prereg) as f:
            p = json.load(f)
        self.assertEqual(p["frozen_model_sha256"], digest)
        self.assertEqual([h["name"] for h in p["hypotheses"]], ["A", "fade"])
        self.assertEqual(p["holdout"]["pass"]["t_min"], 2.24)
        with open(self.model) as f:
            m = json.load(f)
        self.assertEqual(m["strategies"]["A"]["threshold_return"], 0.003)
        self.assertEqual(m["strategies"]["fade"]["score"], "-gap_atr")

    def test_hold_out_refuses_while_the_files_are_not_committed(self):
        with self.assertRaises(holdout.NotPreregistered):
            self.guard()

    def test_hold_out_refuses_when_the_model_changed_after_the_commit(self):
        self.commit()
        with open(self.model, "a") as f:
            f.write(" ")
        with self.assertRaises(holdout.NotPreregistered):
            self.guard()

    def test_hold_out_refuses_when_the_model_hash_does_not_match_the_preregistration(self):
        with open(self.model) as f:
            m = json.load(f)
        m["strategies"]["A"]["threshold_return"] = 0.0001         # edited, then committed together
        with open(self.model, "w") as f:
            json.dump(m, f)
        self.commit()
        with self.assertRaises(holdout.NotPreregistered):
            self.guard()

    def test_hold_out_runs_once_then_refuses_a_second_run(self):
        self.commit()
        self.guard()                                              # passes
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        with open(self.out, "w") as f:
            f.write("{}")
        with self.assertRaises(holdout.AlreadyRun):
            self.guard()


class DecisionTest(unittest.TestCase):
    def res(self, ic, t, mean):
        return {"ic": ic, "ic_t": t, "eod_net_mean": mean}

    def test_a_passing_primary_goes_live(self):
        self.assertEqual(holdout.decide(self.res(0.05, 2.5, 0.001), self.res(0.08, 3.0, 0.003)), "A")

    def test_the_fade_goes_live_only_if_it_passes_and_the_primary_does_not(self):
        self.assertEqual(holdout.decide(self.res(0.01, 1.0, 0.001), self.res(0.08, 3.0, 0.003)), "fade")

    def test_passing_needs_both_the_stricter_t_and_a_positive_net_return(self):
        self.assertEqual(holdout.decide(self.res(0.0, 0.0, 0.0), self.res(0.08, 2.2, 0.003)), "A")    # t below 2.24
        self.assertEqual(holdout.decide(self.res(0.0, 0.0, 0.0), self.res(0.08, 3.0, -0.001)), "A")   # net < 0

    def test_if_neither_passes_the_primary_goes_live_as_pre_agreed(self):
        self.assertEqual(holdout.decide(self.res(-0.01, -1, -0.001), self.res(0.0, 0.1, -0.002)), "A")


if __name__ == "__main__":
    unittest.main()
