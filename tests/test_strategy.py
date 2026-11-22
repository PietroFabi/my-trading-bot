import unittest

import pandas as pd

import tests  # noqa: F401
import strategy


def cand(symbol, pred, side="long", sector="Tech", shortable=True):
    return {"symbol": symbol, "pred": pred, "gate_long": side == "long", "gate_short": side == "short",
            "sector": sector, "shortable": shortable}


def pick(rows, threshold=0.001, **kw):
    return strategy.select_candidates(pd.DataFrame(rows), threshold, **kw)


class SelectTest(unittest.TestCase):
    def test_longs_need_a_prediction_above_the_threshold_and_shorts_below_minus_it(self):
        got = pick([cand("A", 0.002), cand("B", 0.0005), cand("C", -0.002, "short"), cand("D", -0.0005, "short")])
        self.assertEqual(list(zip(got.symbol, got.side)), [("A", "long"), ("C", "short")])

    def test_a_long_gated_stock_with_a_negative_prediction_is_skipped_not_shorted(self):
        self.assertTrue(pick([cand("A", -0.01, "long")]).empty)

    def test_strongest_absolute_prediction_first_and_at_most_ten(self):
        rows = [cand(f"S{i}", 0.002 + i * 0.001, sector=f"sec{i}") for i in range(12)]
        got = pick(rows)
        self.assertEqual(len(got), 10)
        self.assertEqual(got.symbol.iloc[0], "S11")
        self.assertNotIn("S0", set(got.symbol))

    def test_at_most_three_per_sector(self):
        rows = [cand(f"T{i}", 0.01 - i * 0.001) for i in range(5)] + [cand("H", 0.002, sector="Health")]
        got = pick(rows)
        self.assertEqual(list(got.symbol), ["T0", "T1", "T2", "H"])

    def test_shorts_need_a_shortable_easy_to_borrow_stock(self):
        self.assertTrue(pick([cand("A", -0.01, "short", shortable=False)]).empty)

    def test_no_candidates_gives_an_empty_selection(self):
        self.assertTrue(pick([cand("A", 0.0)]).empty)


def gapper(symbol, gap_atr, sector="Tech", shortable=True):
    return {"symbol": symbol, "gap_atr": gap_atr, "gate_long": gap_atr > 0, "gate_short": gap_atr < 0,
            "sector": sector, "shortable": shortable}


class FadeTest(unittest.TestCase):
    def test_gap_ups_are_shorted_and_gap_downs_bought_biggest_gap_first(self):
        got = strategy.fade_candidates(pd.DataFrame([gapper("UP", 0.5), gapper("DOWN", -1.5), gapper("BIG", 2.0)]))
        self.assertEqual(list(zip(got.symbol, got.side)), [("BIG", "short"), ("DOWN", "long"), ("UP", "short")])

    def test_fade_keeps_the_daily_and_sector_limits_and_borrow_rule(self):
        rows = [gapper(f"T{i}", 1.0 + i) for i in range(5)] + [gapper("NB", 9.0, sector="Health", shortable=False)]
        got = strategy.fade_candidates(pd.DataFrame(rows))
        self.assertEqual(list(got.symbol), ["T4", "T3", "T2"])

    def test_ungated_stocks_are_never_faded(self):
        row = dict(gapper("X", 3.0), gate_long=False)
        self.assertTrue(strategy.fade_candidates(pd.DataFrame([row])).empty)


# --- Live selection (Step 11) -----------------------------------------------------------------------

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import model  # noqa: E402

def frozen(kind="ridge", stop_k=0.75, threshold=0.005):
    ridge = {"lam": 1.0, "clip_y": None, "clip_z": 4.0, "mean": [0.0] * 8, "std": [1.0] * 8,
             "coef": [0.01] + [0.0] * 7, "intercept": 0.0}          # prediction = 1% per unit of gap_atr
    a = {"kind": "ridge", "features": model.FEATURES, "exclude_price_move": False, "ridge": ridge,
         "threshold_return": threshold, "entry": "open", "stop_k": stop_k, "exit": "close"}
    fade = {"kind": "gap_fade", "score": "-gap_atr", "entry": "open", "stop_k": None, "exit": "close"}
    return {"strategy": "A" if kind == "ridge" else "fade", "spec": a if kind == "ridge" else fade}


def stock(symbol, gap_atr, sector="Tech", atr_pct=2.0, ref=100.0, trend=0.0):
    up = gap_atr > 0
    return {"symbol": symbol, "sector": sector, "gate_long": up, "gate_short": not up, "ref_price": ref,
            "prev_close": ref, "atr_pct": atr_pct, "gap_pct": gap_atr * atr_pct, "excess_gap_pct": gap_atr * atr_pct,
            "n_fresh": 2, "gap_atr": gap_atr, "rvol": 1.0, "trend": trend, "dist_52w_high": -0.1,
            "news_today": 0.5 if up else -0.5, "baseline": 0.3, "surprise": 0.1, "news_volume": 0.0}


def table(rows):
    return pd.DataFrame(rows).set_index("symbol")


class LiveChoiceTest(unittest.TestCase):
    def choose(self, rows, kind="ridge", equity=100_000.0, shortable=lambda s: True, **kw):
        return strategy.choose_trades(table(rows), frozen(kind, **kw), equity, shortable)

    def test_a_long_signal_becomes_a_sized_target_with_a_stop_below_the_reference_price(self):
        scored, targets = self.choose([stock("AAA", 1.0)])
        t = targets[0]
        self.assertEqual((t["ticker"], t["side"], t["qty"]), ("AAA", "long", 30))   # 0.06% / 2% ATR = 3% of $100k
        self.assertEqual(t["stop_price"], 98.5)                                    # 100 x (1 - 0.75 x 2%)
        self.assertAlmostEqual(t["score"], 0.01)
        self.assertEqual(scored.loc["AAA", "decision"], "selected")

    def test_a_short_signal_has_its_stop_above_and_no_stop_when_k_is_none(self):
        _, targets = self.choose([stock("SSS", -1.0)])
        self.assertEqual((targets[0]["side"], targets[0]["stop_price"]), ("short", 101.5))
        _, targets = self.choose([stock("SSS", -1.0)], stop_k=None)
        self.assertIsNone(targets[0]["stop_price"])

    def test_a_hard_to_borrow_short_is_dropped_and_only_short_candidates_are_checked(self):
        asked = []

        def shortable(sym):
            asked.append(sym)
            return False
        scored, targets = self.choose([stock("LLL", 1.0), stock("SSS", -1.0), stock("WEAK", -0.2)], shortable=shortable)
        self.assertEqual([t["ticker"] for t in targets], ["LLL"])
        self.assertEqual(asked, ["SSS"])                                            # WEAK is below the threshold
        self.assertEqual(scored.loc["SSS", "decision"], "hard to borrow")
        self.assertEqual(scored.loc["WEAK", "decision"], "below threshold")

    def test_the_fourth_stock_of_a_sector_and_the_eleventh_trade_are_dropped(self):
        rows = [stock(f"T{i}", 2.0 + i / 10) for i in range(4)] + \
               [stock(f"S{i}", 1.0 + i / 10, sector=f"sec{i}") for i in range(9)]
        scored, targets = self.choose(rows)
        self.assertEqual(len(targets), 10)
        self.assertNotIn("T0", [t["ticker"] for t in targets])                   # weakest of four Tech names
        self.assertNotIn("S0", [t["ticker"] for t in targets])                   # 11th strongest
        self.assertEqual(scored.loc["T0", "decision"], "daily or sector limit")

    def test_a_zero_signal_day_has_no_targets(self):
        scored, targets = self.choose([stock("AAA", 0.2), stock("BBB", -0.3)])
        self.assertEqual(targets, [])
        self.assertTrue((scored.decision == "below threshold").all())

    def test_sizing_caps_each_stock_at_4_percent_and_the_day_at_20_percent(self):
        _, one = self.choose([stock("AAA", 1.0, atr_pct=1.0)])
        self.assertEqual(one[0]["qty"], 40)                                        # 6% wanted, capped at 4%
        rows = [stock(f"S{i}", 1.0 + i / 10, sector=f"sec{i}", atr_pct=1.0) for i in range(10)]
        _, targets = self.choose(rows)
        self.assertEqual([t["qty"] for t in targets], [20] * 10)                 # 10 x 4% scaled to 20%

    def test_the_fade_trades_against_the_gap(self):
        _, targets = self.choose([stock("UP", 0.3), stock("DOWN", -2.0)], kind="fade")
        self.assertEqual([(t["ticker"], t["side"]) for t in targets], [("DOWN", "long"), ("UP", "short")])

    def test_targets_are_json_safe_even_with_a_missing_feature(self):
        _, targets = self.choose([stock("AAA", 1.0, trend=float("nan"))])
        self.assertIsNone(targets[0]["features"]["trend"])
        json.dumps(targets, allow_nan=False)

    def test_an_entry_mode_the_live_bot_does_not_support_is_refused(self):
        f = frozen()
        f["spec"]["entry"] = "orb"
        with self.assertRaises(NotImplementedError):
            strategy.choose_trades(table([stock("AAA", 1.0)]), f, 100_000.0, lambda s: True)


class FrozenModelTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.dir.name, "frozen_model.json")
        self.prereg = os.path.join(self.dir.name, "preregistration.json")
        body = {"strategies": {"A": frozen()["spec"], "fade": frozen("fade")["spec"]}}
        with open(self.model, "w", newline="\n") as f:
            json.dump(body, f)
        with open(self.model, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        with open(self.prereg, "w") as f:
            json.dump({"frozen_model_sha256": digest}, f)

    def tearDown(self):
        self.dir.cleanup()

    def test_loads_the_strategy_that_goes_live(self):
        f = strategy.load_frozen(self.model, self.prereg, live="A")
        self.assertEqual((f["strategy"], f["spec"]["kind"]), ("A", "ridge"))

    def test_refuses_a_model_that_does_not_match_the_preregistration(self):
        with open(self.model, "a") as fh:
            fh.write(" ")
        with self.assertRaises(strategy.FrozenModelMismatch):
            strategy.load_frozen(self.model, self.prereg, live="A")


if __name__ == "__main__":
    unittest.main()
