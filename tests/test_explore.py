import datetime as dt
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
from backtest import explore
from market_session import Session

NY = ZoneInfo("America/New_York")
S = Session(dt.date(2026, 6, 10), dt.datetime(2026, 6, 10, 9, 30, tzinfo=NY), dt.datetime(2026, 6, 10, 16, tzinfo=NY))


def bars(open_, close):
    idx = [S.open, S.open + dt.timedelta(minutes=1), S.close - dt.timedelta(minutes=1)]
    return pd.DataFrame({"open": [open_, open_, close], "high": [close] * 3, "low": [open_] * 3,
                         "close": [open_, open_, close]}, index=idx)


class HoldOutTest(unittest.TestCase):
    def test_exploration_never_sees_a_hold_out_session(self):
        f = pd.DataFrame({"session_date": [dt.date(2026, 10, 8), dt.date(2026, 10, 9), dt.date(2026, 11, 20)]})
        self.assertEqual(list(explore.exploration_rows(f).session_date), [dt.date(2026, 10, 8)])


class TargetTest(unittest.TestCase):
    def test_market_relative_return_removes_beta_times_the_sector_etf(self):
        r = explore.market_relative_return(bars(100, 102), bars(50, 50.5), beta=1.5, session=S)
        self.assertAlmostEqual(r, 0.02 - 1.5 * 0.01)

    def test_missing_opening_bar_gives_no_target(self):
        late = bars(100, 102).iloc[1:]
        self.assertTrue(pd.isna(explore.market_relative_return(late, bars(50, 50.5), beta=1.0, session=S)))


class MetricsTest(unittest.TestCase):
    def test_trade_statistics(self):
        st = explore.trade_stats([0.01, -0.005, 0.02, 0.0])
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["mean"], 0.00625)
        self.assertAlmostEqual(st["hit_rate"], 0.5)                   # zero is not a hit

    def test_drawdown_and_sharpe_of_a_daily_return_series(self):
        st = explore.curve_stats(pd.Series([0.01, -0.02, 0.01, 0.0]))
        self.assertAlmostEqual(st["max_drawdown"], 1 - (1.01 * 0.98) / 1.01)
        self.assertGreater(st["sharpe"], -10)


if __name__ == "__main__":
    unittest.main()
