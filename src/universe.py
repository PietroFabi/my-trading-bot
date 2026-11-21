"""
Universe: the S&P 500 members the bot scans, with each stock's GICS sector and sector ETF.

Source: Wikipedia's "List of S&P 500 companies" table, cached in data/universe.json and
refreshed when the copy is more than a week old (index changes happen a few times a quarter,
so a week-old list is harmless). If Wikipedia is unreachable the cached copy is used
(a notice, not a failure); with no copy at all, the GitHub "datasets" CSV is tried.

Symbols are kept in Alpaca's form ("BRK.B"); the Yahoo form ("BRK-B") is stored alongside
for the price fallback.
"""
import datetime as dt
import io
import os

from storage import DATA_DIR, load_json, save_json

UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.json")
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATASETS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
# Generic client name: never send personal details to outside services.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trading-bot research script)"}

MAX_AGE_DAYS = 7
MIN_ROWS = 450  # fewer means the page changed shape; don't trade a truncated universe

# GICS sector -> SPDR sector ETF, used to measure each stock's gap relative to its sector.
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def _member(symbol: str, sector: str) -> dict:
    symbol = symbol.strip().replace("-", ".")
    if sector not in SECTOR_ETF:
        raise ValueError(f"{symbol}: unknown GICS sector {sector!r}")
    return {"symbol": symbol, "yahoo": symbol.replace(".", "-"), "sector": sector, "sector_etf": SECTOR_ETF[sector]}


def _from_frame(frame) -> list:
    return [_member(str(row["Symbol"]), str(row["GICS Sector"])) for _, row in frame.iterrows()]


def parse_constituents(html: str) -> list:
    """Members from the Wikipedia page's table with id="constituents"."""
    import pandas as pd
    return _from_frame(pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0])


def fetch_wikipedia() -> list:
    import requests
    response = requests.get(WIKIPEDIA_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return parse_constituents(response.text)


def fetch_datasets_csv() -> list:
    import pandas as pd
    import requests
    response = requests.get(DATASETS_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return _from_frame(pd.read_csv(io.StringIO(response.text)))


def _checked(members: list, source: str) -> list:
    if len(members) < MIN_ROWS:
        raise RuntimeError(f"{source} returned only {len(members)} members (expected ~503)")
    return members


def load_universe(session_date: dt.date, cache_path: str = UNIVERSE_FILE,
                  fetch=fetch_wikipedia, fallback=fetch_datasets_csv) -> list:
    """The S&P 500 members as of session_date: [{symbol, yahoo, sector, sector_etf}, ...]."""
    cache = load_json(cache_path, default=None)
    if cache:
        age = (session_date - dt.date.fromisoformat(cache["fetched_on"])).days
        if age <= MAX_AGE_DAYS:
            return cache["members"]

    for source, get in (("wikipedia", fetch), ("datasets", fallback)):
        if source == "datasets" and cache:
            break  # a week-old Wikipedia copy beats a list of unknown freshness
        try:
            members = _checked(get(), source)
        except Exception as e:
            print(f"::notice::Could not refresh the S&P 500 list from {source}: {e}")
            continue
        save_json(cache_path, {"fetched_on": session_date.isoformat(), "source": source, "members": members})
        print(f"S&P 500 list refreshed from {source}: {len(members)} members")
        return members

    if cache:
        print(f"::notice::Using the cached S&P 500 list from {cache['fetched_on']}.")
        return cache["members"]
    raise RuntimeError("No S&P 500 list available: both sources failed and there is no cached copy")
