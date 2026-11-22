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


if __name__ == "__main__":
    unittest.main()
