import datetime as dt
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
from backtest import simulate as sim
from market_session import Session

NY = ZoneInfo("America/New_York")
DAY = dt.date(2026, 6, 10)
FULL = Session(DAY, dt.datetime(2026, 6, 10, 9, 30, tzinfo=NY), dt.datetime(2026, 6, 10, 16, tzinfo=NY))
HALF = Session(DAY, dt.datetime(2026, 6, 10, 9, 30, tzinfo=NY), dt.datetime(2026, 6, 10, 13, tzinfo=NY))


def bars(session, price=100.0, changes=None):
    """Flat 1-minute bars open -> close; changes = {minute: (open, high, low, close)} overrides single bars,
    and from a changed minute on the flat level becomes that bar's close."""
    rows, level = [], price
    minutes = int((session.close - session.open).total_seconds() // 60)
    for m in range(minutes):
        if changes and m in changes:
            o, h, lo, c = changes[m]
            level = c
        else:
            o = h = lo = c = level
        rows.append({"timestamp": session.open + dt.timedelta(minutes=m), "open": o, "high": h, "low": lo,
                     "close": c, "volume": 1000.0})
    return pd.DataFrame(rows).set_index("timestamp")


def row(side="long", atr_pct=2.0, ref_price=100.0, rvol5=2.0):
    return pd.Series({"symbol": "AAA", "gate_long": side == "long", "gate_short": side == "short",
                      "atr_pct": atr_pct, "ref_price": ref_price, "rvol5": rvol5})


def run(r, b, session=FULL, entry_mode="open", stop_k=None, equity=100_000.0):
    return sim.simulate_trade(r, b, session, entry_mode, stop_k, equity)


class ReturnTest(unittest.TestCase):
    def test_long_up_one_percent_at_the_close_nets_one_percent_minus_two_sides_of_costs(self):
        t = run(row(), bars(FULL, changes={389: (100, 101, 100, 101)}))
        self.assertTrue(t["entered"])
        self.assertAlmostEqual(t["ret_gross"]["EOD"], 0.01)
        self.assertAlmostEqual(t["ret_net"]["EOD"], 0.01 - 0.0010)
        self.assertEqual(t["exit_reason"], "close")

    def test_short_whose_price_falls_two_percent_makes_two_percent_gross(self):
        t = run(row("short"), bars(FULL, changes={100: (100, 100, 98, 98)}))
        self.assertAlmostEqual(t["ret_gross"]["EOD"], 0.02)
        self.assertAlmostEqual(t["ret_gross"]["1_hour"], 0.0)          # the drop happens at minute 100

    def test_an_explicit_side_overrides_the_gate(self):
        r = row("long")
        r["side"] = "short"                                              # the fade trades against the gate
        t = run(r, bars(FULL, changes={389: (100, 101, 100, 101)}))
        self.assertEqual(t["side"], "short")
        self.assertAlmostEqual(t["ret_gross"]["EOD"], -0.01)

    def test_pnl_is_quantity_times_entry_times_net_return(self):
        t = run(row(), bars(FULL, changes={389: (100, 101, 100, 101)}))
        self.assertAlmostEqual(t["pnl_net"]["EOD"], t["qty"] * 100 * (0.01 - 0.0010))

    def test_horizon_names_match_the_live_evaluator(self):
        self.assertEqual(list(sim.HORIZONS), ["1_min", "10_min", "30_min", "1_hour", "5_hour", "EOD"])


class StopTest(unittest.TestCase):
    def test_stop_hit_at_10_05_fixes_every_later_horizon_at_the_stop_return(self):
        # ATR 2%, k = 1 -> stop at 98. Bar 35 (10:05) trades down to 97.5, then the price recovers to 103.
        b = bars(FULL, changes={35: (99.5, 99.5, 97.5, 98.5), 36: (103, 103, 103, 103)})
        t = run(row(), b, stop_k=1.0)
        self.assertEqual((t["exit_reason"], t["exit"]), ("stop", 98.0))
        self.assertAlmostEqual(t["ret_gross"]["30_min"], 0.0)            # 10:00, before the stop
        for h in ("1_hour", "5_hour", "EOD"):
            self.assertAlmostEqual(t["ret_gross"][h], -0.02)

    def test_a_bar_that_opens_through_the_stop_fills_at_its_open(self):
        t = run(row(), bars(FULL, changes={50: (96, 96.5, 95, 96)}), stop_k=1.0)
        self.assertEqual(t["exit"], 96.0)
        self.assertAlmostEqual(t["ret_gross"]["EOD"], -0.04)

    def test_short_stop_is_above_the_entry(self):
        t = run(row("short"), bars(FULL, changes={20: (100, 101.6, 100, 100)}), stop_k=0.75)  # stop at 101.5
        self.assertEqual((t["exit_reason"], t["exit"]), ("stop", 101.5))
        self.assertAlmostEqual(t["ret_gross"]["EOD"], -0.015)

    def test_a_stop_inside_the_bar_a_horizon_reads_counts_first(self):
        # The 10:29 bar dips to 97 (stop 98) and closes back at 100; the 1-hour price (10:30) reads that bar.
        t = run(row(), bars(FULL, changes={59: (100, 100, 97, 100)}), stop_k=1.0)
        self.assertAlmostEqual(t["ret_gross"]["30_min"], 0.0)
        self.assertAlmostEqual(t["ret_gross"]["1_hour"], -0.02)

    def test_the_stop_is_live_from_the_entry_bar_itself(self):
        t = run(row(), bars(FULL, changes={0: (100, 100, 97, 99)}), stop_k=1.0)
        self.assertEqual((t["exit_reason"], t["exit"]), ("stop", 98.0))

    def test_no_stop_rides_to_the_close(self):
        t = run(row(), bars(FULL, changes={35: (99.5, 99.5, 90, 95)}), stop_k=None)
        self.assertEqual(t["exit_reason"], "close")
        self.assertAlmostEqual(t["ret_gross"]["EOD"], -0.05)


class OpeningRangeBreakoutTest(unittest.TestCase):
    RANGE = {0: (100, 101, 99, 100), 4: (100, 100.5, 99.5, 100)}       # 09:30-09:34 range = 99 .. 101

    def test_long_enters_at_the_close_of_the_09_35_bar_when_it_closes_above_the_range(self):
        t = run(row(), bars(FULL, changes={**self.RANGE, 5: (100, 101.6, 100, 101.5)}), entry_mode="orb")
        self.assertTrue(t["entered"])
        self.assertEqual((t["entry"], t["entry_time"]), (101.5, FULL.open + dt.timedelta(minutes=6)))
        self.assertIsNone(t["ret_gross"]["1_min"])                     # 09:31 is before the entry
        self.assertAlmostEqual(t["ret_gross"]["EOD"], 0.0)

    def test_no_breakout_at_09_35_means_no_trade_even_if_it_breaks_out_later(self):
        t = run(row(), bars(FULL, changes={**self.RANGE, 40: (100, 103, 100, 103)}), entry_mode="orb")
        self.assertEqual((t["entered"], t["exit_reason"]), (False, "no_breakout"))

    def test_a_breakout_against_the_trade_direction_is_not_an_entry(self):
        t = run(row(), bars(FULL, changes={**self.RANGE, 5: (99, 99, 98, 98.5)}), entry_mode="orb")
        self.assertFalse(t["entered"])

    def test_short_breaks_out_below_the_range(self):
        t = run(row("short"), bars(FULL, changes={**self.RANGE, 5: (99, 99, 98, 98.5)}), entry_mode="orb")
        self.assertEqual((t["entered"], t["entry"]), (True, 98.5))

    def test_quiet_opening_volume_filters_the_breakout_out(self):
        t = run(row(rvol5=0.8), bars(FULL, changes={**self.RANGE, 5: (100, 101.6, 100, 101.5)}), entry_mode="orb")
        self.assertEqual((t["entered"], t["exit_reason"]), (False, "rvol"))


class HalfDayTest(unittest.TestCase):
    def test_horizons_past_a_13_00_close_use_the_closing_price(self):
        t = run(row(), bars(HALF, changes={209: (100, 102, 100, 102)}), session=HALF)
        self.assertAlmostEqual(t["ret_gross"]["5_hour"], 0.02)          # 14:30 is after the 13:00 close
        self.assertAlmostEqual(t["ret_gross"]["EOD"], 0.02)


class SizingTest(unittest.TestCase):
    def test_atr_2_percent_on_100k_buys_3_percent_of_equity(self):
        t = run(row(atr_pct=2.0), bars(FULL))
        self.assertEqual(t["qty"], 30)                                   # $3,000 at the $100 reference price

    def test_atr_1_percent_is_capped_at_4_percent_of_equity(self):
        self.assertEqual(run(row(atr_pct=1.0), bars(FULL))["qty"], 40)

    def test_quantity_comes_from_the_morning_reference_price_not_the_fill(self):
        t = run(row(atr_pct=2.0, ref_price=50.0), bars(FULL))           # fills at 100, sized at 50
        self.assertEqual(t["qty"], 60)

    def test_a_single_share_bigger_than_the_size_is_skipped(self):
        t = run(row(atr_pct=2.0, ref_price=5000.0), bars(FULL, price=5000.0))
        self.assertEqual((t["entered"], t["exit_reason"]), (False, "size"))

    def test_the_day_is_scaled_down_to_20_percent_of_equity(self):
        trades = [{"entered": True, "qty": 40, "entry": 100.0, "ret_net": {"EOD": 0.01}, "pnl_net": {"EOD": 40.0}}
                  for _ in range(10)]                                    # 10 x 4% = 40% of equity
        scaled = sim.apply_day_cap(trades, equity=100_000.0)
        self.assertEqual([t["qty"] for t in scaled], [20] * 10)
        self.assertAlmostEqual(scaled[0]["pnl_net"]["EOD"], 20.0)


class MissingDataTest(unittest.TestCase):
    def test_no_bars_means_no_trade(self):
        t = run(row(), bars(FULL).iloc[0:0])
        self.assertEqual((t["entered"], t["exit_reason"]), (False, "no_bars"))


if __name__ == "__main__":
    unittest.main()
