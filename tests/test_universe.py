import datetime as dt
import json
import os
import tempfile
import unittest

import tests  # noqa: F401  (puts src/ on sys.path)
import universe

SECTORS = list(universe.SECTOR_ETF)


def fake_html(n_rows: int) -> str:
    """A page shaped like Wikipedia's: a table with id="constituents" and the same column names."""
    rows = ["<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Insurance</td></tr>",
            "<tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td><td>Distillers</td></tr>"]
    for i in range(n_rows - 2):
        rows.append(f"<tr><td>T{i}</td><td>Company {i}</td><td>{SECTORS[i % len(SECTORS)]}</td><td>x</td></tr>")
    return ("<html><body><table class='wikitable' id='other'><tr><th>Ignore</th></tr><tr><td>me</td></tr></table>"
            "<table class='wikitable' id='constituents'><thead><tr><th>Symbol</th><th>Security</th>"
            "<th>GICS Sector</th><th>GICS Sub-Industry</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>")


def members(n: int) -> list:
    return universe.parse_constituents(fake_html(n))


class ParseTest(unittest.TestCase):
    def test_parses_the_constituents_table_only(self):
        self.assertEqual(len(members(503)), 503)

    def test_dotted_symbols_keep_alpaca_form_and_get_yahoo_form(self):
        brk = next(m for m in members(10) if m["symbol"] == "BRK.B")
        self.assertEqual(brk["yahoo"], "BRK-B")
        self.assertEqual(brk["sector"], "Financials")
        self.assertEqual(brk["sector_etf"], "XLF")

    def test_every_gics_sector_maps_to_a_sector_etf(self):
        self.assertEqual(len(universe.SECTOR_ETF), 11)
        self.assertTrue(all(m["sector_etf"] in universe.SECTOR_ETF.values() for m in members(50)))


class LoadUniverseTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, "universe.json")
        self.today = dt.date(2026, 11, 23)
        self.calls = []

    def tearDown(self):
        self.dir.cleanup()

    def write_cache(self, age_days: int, n: int = 500):
        with open(self.cache, "w") as f:
            json.dump({"fetched_on": (self.today - dt.timedelta(days=age_days)).isoformat(),
                       "source": "wikipedia", "members": members(n)}, f)

    def fetch_ok(self):
        self.calls.append("wikipedia")
        return members(503)

    def fetch_fails(self):
        self.calls.append("wikipedia")
        raise ConnectionError("down")

    def fallback_ok(self):
        self.calls.append("datasets")
        return members(501)

    def fallback_fails(self):
        self.calls.append("datasets")
        raise ConnectionError("down too")

    def load(self, fetch, fallback):
        return universe.load_universe(self.today, cache_path=self.cache, fetch=fetch, fallback=fallback)

    def test_fresh_cache_is_used_without_fetching(self):
        self.write_cache(age_days=3)
        self.assertEqual(len(self.load(self.fetch_ok, self.fallback_ok)), 500)
        self.assertEqual(self.calls, [])

    def test_stale_cache_triggers_a_refresh_that_is_saved(self):
        self.write_cache(age_days=8)
        self.assertEqual(len(self.load(self.fetch_ok, self.fallback_ok)), 503)
        self.assertEqual(self.calls, ["wikipedia"])
        with open(self.cache) as f:
            saved = json.load(f)
        self.assertEqual(saved["fetched_on"], "2026-11-23")
        self.assertEqual(len(saved["members"]), 503)

    def test_failed_refresh_keeps_the_stale_cache(self):
        self.write_cache(age_days=8)
        self.assertEqual(len(self.load(self.fetch_fails, self.fallback_ok)), 500)
        self.assertEqual(self.calls, ["wikipedia"])

    def test_no_cache_and_wikipedia_down_uses_the_fallback_source(self):
        self.assertEqual(len(self.load(self.fetch_fails, self.fallback_ok)), 501)
        self.assertEqual(self.calls, ["wikipedia", "datasets"])

    def test_no_cache_and_both_sources_down_raises(self):
        with self.assertRaises(RuntimeError):
            self.load(self.fetch_fails, self.fallback_fails)

    def test_a_suspiciously_short_list_is_rejected_as_broken(self):
        self.write_cache(age_days=8)
        short = lambda: members(120)  # noqa: E731  e.g. Wikipedia changed the page layout
        self.assertEqual(len(self.load(short, self.fallback_ok)), 500)  # keeps the cache instead


if __name__ == "__main__":
    unittest.main()
