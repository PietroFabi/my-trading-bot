import datetime as dt
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import tests  # noqa: F401
from backtest import history

UTC = dt.timezone.utc


def member(symbol, sector="Industrials"):
    return {"symbol": symbol, "yahoo": symbol, "sector": sector, "sector_etf": "XLI"}


class FakeWiki:
    """Revisions: r1 has OLD; r2 (live from Wednesday 2026-06-10) has NEW instead of OLD."""
    def __init__(self, fail_after=None):
        self.revision_calls, self.member_calls, self.fail_after = [], [], fail_after

    def revision_before(self, when: dt.datetime):
        if self.fail_after is not None and len(self.revision_calls) >= self.fail_after:
            raise ConnectionError("rate limited")
        self.revision_calls.append(when)
        return {"revid": 2} if when >= dt.datetime(2026, 6, 10, 12, tzinfo=UTC) else {"revid": 1}

    def members_at(self, revid: int):
        self.member_calls.append(revid)
        return [member("AAA"), member("OLD")] if revid == 1 else [member("AAA"), member("NEW")]


# Fri 06-05 (week of 06-01), Tue 06-09 and Thu 06-11 (week of 06-08), Tue 06-16 (week of 06-15)
DATES = [dt.date(2026, 6, 5), dt.date(2026, 6, 9), dt.date(2026, 6, 11), dt.date(2026, 6, 16)]


class MembershipHistoryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, "membership.json")

    def tearDown(self):
        self.dir.cleanup()

    def build(self, wiki):
        return history.membership_history(DATES, cache_path=self.cache, revision_before=wiki.revision_before,
                                          members_at=wiki.members_at)

    def test_each_session_uses_the_page_as_it_stood_on_its_weeks_monday_morning(self):
        h = self.build(FakeWiki())
        symbols = {d: [m["symbol"] for m in h[d]] for d in DATES}
        self.assertEqual(symbols[dt.date(2026, 6, 5)], ["AAA", "OLD"])
        self.assertEqual(symbols[dt.date(2026, 6, 11)], ["AAA", "OLD"])   # change on Wed shows from next week
        self.assertEqual(symbols[dt.date(2026, 6, 16)], ["AAA", "NEW"])

    def test_one_lookup_per_week_at_monday_8am_new_york(self):
        wiki = FakeWiki()
        self.build(wiki)
        self.assertEqual(wiki.revision_calls, [dt.datetime(2026, 6, 1, 12, tzinfo=UTC), dt.datetime(2026, 6, 8, 12, tzinfo=UTC),
                                               dt.datetime(2026, 6, 15, 12, tzinfo=UTC)])

    def test_each_revision_is_downloaded_once(self):
        wiki = FakeWiki()
        self.build(wiki)
        self.assertEqual(sorted(wiki.member_calls), [1, 2])

    def test_progress_is_saved_after_every_lookup_so_a_crash_loses_nothing(self):
        with self.assertRaises(ConnectionError):
            self.build(FakeWiki(fail_after=2))
        with open(self.cache) as f:
            self.assertEqual(len(json.load(f)["by_week"]), 2)
        wiki = FakeWiki()
        self.build(wiki)
        self.assertEqual(len(wiki.revision_calls), 1)                      # only the missing week

    def test_second_run_uses_the_cache_without_network(self):
        self.build(FakeWiki())
        wiki = FakeWiki()
        self.assertEqual(len(self.build(wiki)), 4)
        self.assertEqual((wiki.revision_calls, wiki.member_calls), ([], []))

    def test_union_lists_every_symbol_that_was_ever_a_member(self):
        self.assertEqual(history.union_members(self.build(FakeWiki())), [member("AAA"), member("NEW"), member("OLD")])


class PoliteGetTest(unittest.TestCase):
    def test_rate_limited_requests_are_retried_with_backoff(self):
        responses = [SimpleNamespace(status_code=429, headers={"Retry-After": "7"}),
                     SimpleNamespace(status_code=429, headers={}),
                     SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None)]
        sleeps = []
        got = history.polite_get("u", {}, get=lambda *a, **k: responses.pop(0), sleep=sleeps.append)
        self.assertEqual(got.status_code, 200)
        self.assertEqual(sleeps[:2], [7.0, 20.0])          # Retry-After honoured, then backoff

    def test_gives_up_after_several_attempts(self):
        always = lambda *a, **k: SimpleNamespace(status_code=429, headers={})  # noqa: E731
        with self.assertRaises(RuntimeError):
            history.polite_get("u", {}, get=always, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
