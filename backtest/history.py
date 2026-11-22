"""
Historical S&P 500 membership, so the backtest has no survivorship bias.

Wikipedia keeps every revision of "List of S&P 500 companies". The constituents table as it
stood on a session's morning is the membership the bot would have scanned that day, including
stocks that later left the index (e.g. EA, WBD, TTD since April 2026).
Revision ids and tables are cached in .cache/ (gitignored), so this runs once.
"""
import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

import universe

NY = ZoneInfo("America/New_York")
API = "https://en.wikipedia.org/w/api.php"
PAGE = "List of S&P 500 companies"
MORNING = dt.time(8, 0)  # look the page up as it stood at 08:00 New York on the week's Monday


def polite_get(url: str, params: dict, get=None, sleep=time.sleep, attempts: int = 6):
    """GET with Wikipedia etiquette: retry HTTP 429 using Retry-After (else 10 s, 20 s, 40 s...), then a 1 s pause."""
    if get is None:
        import requests
        get = requests.get
    for attempt in range(attempts):
        response = get(url, headers=universe.HEADERS, timeout=30, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            sleep(1.0)
            return response
        sleep(float(response.headers.get("Retry-After", 10 * 2 ** attempt)))
    raise RuntimeError(f"{url}: still rate-limited after {attempts} attempts")


def revision_before(when: dt.datetime) -> dict:
    """The latest page revision at or before `when`: {revid, timestamp}."""
    response = polite_get(API, {
        "action": "query", "prop": "revisions", "titles": PAGE, "rvlimit": 1, "rvdir": "older",
        "rvstart": when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rvprop": "ids|timestamp", "format": "json"})
    page = next(iter(response.json()["query"]["pages"].values()))
    return page["revisions"][0]


def members_at(revid: int) -> list:
    return universe.parse_constituents(polite_get("https://en.wikipedia.org/w/index.php", {"oldid": revid}).text)


def _save(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def membership_history(dates: list, cache_path: str, revision_before=revision_before, members_at=members_at) -> dict:
    """
    {session date: [Member, ...]} using the page as it stood at 08:00 New York on the Monday of
    that session's week: one lookup per week (index changes are announced days ahead and are rare,
    so a change shows up at most a few days late). Progress is saved after every lookup.
    """
    cache = {"by_week": {}, "revisions": {}}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        cache.setdefault("by_week", {})
        cache.setdefault("revisions", {})
    result = {}
    for d in dates:
        monday = d - dt.timedelta(days=d.weekday())
        key = monday.isoformat()
        if key not in cache["by_week"]:
            morning = dt.datetime.combine(monday, MORNING, NY).astimezone(dt.timezone.utc)
            cache["by_week"][key] = revision_before(morning)["revid"]
            _save(cache, cache_path)
        revid = str(cache["by_week"][key])
        if revid not in cache["revisions"]:
            cache["revisions"][revid] = members_at(int(revid))
            _save(cache, cache_path)
        result[d] = cache["revisions"][revid]
    return result


def union_members(history: dict) -> list:
    """Every stock that was a member on any of the dates, once, sorted by symbol."""
    seen = {}
    for members in history.values():
        for m in members:
            seen.setdefault(m["symbol"], m)
    return [seen[s] for s in sorted(seen)]
