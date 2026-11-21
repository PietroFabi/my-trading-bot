import datetime as dt
import tempfile
import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

import tests  # noqa: F401
import marketdata

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
AS_OF = dt.datetime(2026, 11, 20, 8, 55, tzinfo=NY)


def alpaca_frame(rows):
    """rows: [(symbol, utc datetime, close, volume)] in the shape alpaca-py's BarSet.df returns."""
    index = pd.MultiIndex.from_tuples([(s, t) for s, t, _, _ in rows], names=["symbol", "timestamp"])
    return pd.DataFrame({"open": [c for *_, c, _ in rows], "high": [c for *_, c, _ in rows],
                         "low": [c for *_, c, _ in rows], "close": [c for *_, c, _ in rows],
                         "volume": [v for *_, v in rows], "trade_count": 1, "vwap": [c for *_, c, _ in rows]},
                        index=index)


class FakeDataClient:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail, self.requests = list(rows), fail, []

    def get_stock_bars(self, request):
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("alpaca down")
        wanted = set(request.symbol_or_symbols)
        return SimpleNamespace(df=alpaca_frame([r for r in self.rows if r[0] in wanted]))


class FakeYahoo:
    def __init__(self, prices):
        self.prices, self.calls = prices, []

    def __call__(self, yahoo_symbols, start, end, interval):
        self.calls.append(list(yahoo_symbols))
        rows = [(s, dt.datetime(2026, 11, 20, 13, 30, tzinfo=UTC), self.prices[s], 1000)
                for s in yahoo_symbols if s in self.prices]
        return marketdata.normalise(alpaca_frame(rows)) if rows else marketdata.empty_frame()


class MinuteBarsTest(unittest.TestCase):
    start = dt.datetime(2026, 11, 20, 4, 0, tzinfo=NY)

    def test_requests_never_end_after_the_free_plan_delay(self):
        client = FakeDataClient()
        marketdata.minute_bars(["AAPL"], self.start, dt.datetime(2026, 11, 20, 9, 25, tzinfo=NY), as_of=AS_OF,
                               client=client, yahoo=FakeYahoo({}))
        # alpaca-py stores request times as naive UTC: 13:39 UTC = 08:39 New York = 08:55 - 16 min
        self.assertEqual(client.requests[0].end, dt.datetime(2026, 11, 20, 13, 39))

    def test_timestamps_come_back_in_new_york_time(self):
        client = FakeDataClient([("AAPL", dt.datetime(2026, 11, 20, 13, 0, tzinfo=UTC), 270.0, 500)])
        bars = marketdata.minute_bars(["AAPL"], self.start, AS_OF, as_of=AS_OF, client=client, yahoo=FakeYahoo({}))
        ts = bars.loc["AAPL"].index[0]
        self.assertEqual((ts.hour, ts.minute, str(ts.tzinfo)), (8, 0, "America/New_York"))

    def test_symbols_missing_from_alpaca_come_from_yahoo_in_alpaca_form(self):
        client = FakeDataClient([("AAPL", dt.datetime(2026, 11, 20, 13, 0, tzinfo=UTC), 270.0, 500)])
        yahoo = FakeYahoo({"BRK-B": 480.0})
        bars = marketdata.minute_bars(["AAPL", "BRK.B"], self.start, AS_OF, as_of=AS_OF, client=client, yahoo=yahoo)
        self.assertEqual(yahoo.calls, [["BRK-B"]])  # only the missing one
        self.assertEqual(sorted(bars.index.get_level_values("symbol").unique()), ["AAPL", "BRK.B"])

    def test_alpaca_outage_falls_back_to_yahoo_for_everything(self):
        yahoo = FakeYahoo({"AAPL": 270.0, "MSFT": 470.0})
        bars = marketdata.minute_bars(["AAPL", "MSFT"], self.start, AS_OF, as_of=AS_OF,
                                      client=FakeDataClient(fail=True), yahoo=yahoo)
        self.assertEqual(marketdata.coverage(bars, ["AAPL", "MSFT"]), 1.0)

    def test_fallback_can_be_switched_off_for_history(self):
        client = FakeDataClient([("AAPL", dt.datetime(2026, 6, 1, 13, 0, tzinfo=UTC), 200.0, 500)])
        yahoo = FakeYahoo({"MSFT": 1.0})
        marketdata.minute_bars(["AAPL", "MSFT"], dt.datetime(2026, 6, 1, 4, tzinfo=NY),
                               dt.datetime(2026, 6, 1, 9, tzinfo=NY), client=client, yahoo=yahoo, fallback=False)
        self.assertEqual(yahoo.calls, [])

    def test_cache_returns_the_same_frame_without_a_second_call(self):
        rows = [("AAPL", dt.datetime(2026, 6, 1, 13, 0, tzinfo=UTC), 200.0, 500)]
        client = FakeDataClient(rows)
        with tempfile.TemporaryDirectory() as cache:
            args = (["AAPL"], dt.datetime(2026, 6, 1, 4, tzinfo=NY), dt.datetime(2026, 6, 1, 9, tzinfo=NY))
            first = marketdata.minute_bars(*args, client=client, fallback=False, cache_dir=cache)
            second = marketdata.minute_bars(*args, client=client, fallback=False, cache_dir=cache)
        self.assertEqual(len(client.requests), 1)
        pd.testing.assert_frame_equal(first, second)


class DailyBarsTest(unittest.TestCase):
    def test_daily_bars_are_indexed_by_new_york_date_and_split_adjusted(self):
        # Alpaca stamps daily bars at 00:00 New York = 05:00 UTC in winter
        client = FakeDataClient([("AAPL", dt.datetime(2026, 11, 19, 5, 0, tzinfo=UTC), 268.0, 9e6)])
        bars = marketdata.daily_bars(["AAPL"], dt.date(2026, 11, 1), dt.date(2026, 11, 20), client=client,
                                     yahoo=FakeYahoo({}))
        self.assertEqual(bars.loc["AAPL"].index[0], dt.date(2026, 11, 19))
        self.assertEqual(str(client.requests[0].adjustment.value), "split")

    def test_daily_bars_never_include_a_date_after_end(self):
        # Look-ahead guard: the bar dated end+1 is stamped exactly at midnight after `end`.
        client = FakeDataClient([("AAPL", dt.datetime(2026, 11, 19, 5, 0, tzinfo=UTC), 268.0, 9e6),
                                 ("AAPL", dt.datetime(2026, 11, 20, 5, 0, tzinfo=UTC), 270.0, 9e6)])
        bars = marketdata.daily_bars(["AAPL"], dt.date(2026, 11, 1), dt.date(2026, 11, 19), client=client,
                                     yahoo=FakeYahoo({}))
        self.assertEqual(list(bars.loc["AAPL"].index), [dt.date(2026, 11, 19)])


class CoverageTest(unittest.TestCase):
    def test_coverage_is_the_share_of_symbols_with_any_bar(self):
        frame = marketdata.normalise(alpaca_frame([("A", dt.datetime(2026, 1, 2, 14, tzinfo=UTC), 1.0, 1)]))
        self.assertEqual(marketdata.coverage(frame, ["A", "B", "C", "D"]), 0.25)
        self.assertEqual(marketdata.coverage(marketdata.empty_frame(), ["A"]), 0.0)


if __name__ == "__main__":
    unittest.main()
