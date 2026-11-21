"""
News: every Alpaca article (currently all from fiscal.ai) for the universe in a time window.

- Multi-symbol requests in chunks of 100 symbols; alpaca-py follows the pages itself.
- Full text is requested (include_content) so FinBERT can read it, stripped of HTML. The text
  only ever lives in memory: plans and results keep headline, url, source and scores, never bodies.
- Dedupe: by article id, and by normalised headline per stock (keep the earliest).
- Guard: articles tagging more than 3 symbols are dropped (roundups, not news about one company).
  In the current feed every article tags exactly one symbol, so this is insurance for later.
- `price_move` tags headlines that describe a move that already happened ("Shares of X Decline
  After..."): their tone may echo past returns. The backtest decides whether to exclude them.
"""
import datetime as dt
import os
import re
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
CHUNK = 100
MAX_SYMBOLS_PER_ARTICLE = 3

_MOVE_VERBS = (r"rise|rises|rose|jump|jumps|jumped|surge|surges|surged|gain|gains|gained|climb|climbs|climbed|"
               r"soar|soars|soared|rally|rallies|rallied|advance|advances|advanced|"
               r"fall|falls|fell|drop|drops|dropped|decline|declines|declined|slide|slides|slid|sink|sinks|sank|"
               r"tumble|tumbles|tumbled|plunge|plunges|plunged|slump|slumps|slumped|retreat|retreats|retreated")
PRICE_MOVE_PATTERN = re.compile(rf"\b(shares|stock)\b.{{0,40}}?\b({_MOVE_VERBS})\b", re.IGNORECASE)


def is_price_move(headline: str) -> bool:
    return bool(PRICE_MOVE_PATTERN.search(headline or ""))


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(html: str) -> str:
    parser = _TextOnly()
    parser.feed(html or "")
    parser.close()
    return " ".join(" ".join(parser.parts).split())  # also turns &nbsp; (U+00A0) into a plain space


def _normalised(headline: str) -> str:
    return " ".join(re.sub(r"\W+", " ", headline.lower()).split())


def _client():
    from alpaca.data.historical.news import NewsClient
    from dotenv import load_dotenv
    load_dotenv()
    return NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), raw_data=True)


def fetch_articles(symbols, start: dt.datetime, end: dt.datetime, client=None) -> list:
    """
    Articles published in [start, end] for `symbols`, one row per (article, universe symbol):
    {id, symbol, created_at (New York time), headline, summary, text, url, source, price_move}.
    Sorted by created_at.
    """
    from alpaca.data.requests import NewsRequest
    client = client or _client()
    universe = set(symbols)
    seen_ids, raw = set(), []
    for i in range(0, len(symbols), CHUNK):
        response = client.get_news(NewsRequest(symbols=",".join(symbols[i:i + CHUNK]), start=start, end=end,
                                               include_content=True))
        for a in response.get("news", []):
            if a["id"] not in seen_ids:
                seen_ids.add(a["id"])
                raw.append(a)

    rows, seen_headlines = [], set()
    for a in sorted(raw, key=lambda a: a["created_at"]):
        tagged = a.get("symbols") or []
        if len(tagged) > MAX_SYMBOLS_PER_ARTICLE or not a.get("headline"):
            continue
        created = dt.datetime.fromisoformat(a["created_at"].replace("Z", "+00:00")).astimezone(NY)
        for symbol in tagged:
            key = (symbol, _normalised(a["headline"]))
            if symbol not in universe or key in seen_headlines:
                continue
            seen_headlines.add(key)
            rows.append({
                "id": a["id"], "symbol": symbol, "created_at": created,
                "headline": a["headline"].strip(), "summary": strip_html(a.get("summary") or ""),
                "text": strip_html(a.get("content") or ""), "url": a.get("url"), "source": a.get("source"),
                "price_move": is_price_move(a["headline"]),
            })
    return rows
