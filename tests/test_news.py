import datetime as dt
import unittest
from zoneinfo import ZoneInfo

import tests  # noqa: F401
import news

NY = ZoneInfo("America/New_York")
START = dt.datetime(2026, 11, 19, 16, 0, tzinfo=NY)
END = dt.datetime(2026, 11, 20, 8, 55, tzinfo=NY)


def raw(id_, symbols, headline, created="2026-11-20T11:31:54Z", content="<p>Body</p>", summary="Summary."):
    return {"id": id_, "symbols": symbols, "headline": headline, "summary": summary, "content": content,
            "created_at": created, "updated_at": created, "url": f"https://example.com/{id_}",
            "source": "fiscal.ai", "author": "x", "images": []}


class FakeNewsClient:
    def __init__(self, articles):
        self.articles, self.requests = articles, []

    def get_news(self, request):
        self.requests.append(request)
        wanted = set(request.symbols.split(","))
        return {"news": [a for a in self.articles if wanted & set(a["symbols"])], "next_page_token": None}


def fetch(articles, symbols=("AAPL", "MSFT")):
    return news.fetch_articles(list(symbols), START, END, client=FakeNewsClient(articles))


class FetchTest(unittest.TestCase):
    def test_one_row_per_article_and_universe_symbol_with_clean_fields(self):
        rows = fetch([raw(1, ["AAPL"], "Apple Beats Estimates")])
        self.assertEqual(len(rows), 1)
        a = rows[0]
        self.assertEqual((a["id"], a["symbol"], a["headline"], a["source"]), (1, "AAPL", "Apple Beats Estimates", "fiscal.ai"))
        self.assertEqual(a["created_at"], dt.datetime(2026, 11, 20, 6, 31, 54, tzinfo=NY))
        self.assertEqual(a["text"], "Body")

    def test_html_is_stripped_and_entities_decoded(self):
        rows = fetch([raw(1, ["AAPL"], "H", content="<div><p>a &amp; b</p>\n<p>c&nbsp;d</p></div>")])
        self.assertEqual(rows[0]["text"], "a & b c d")

    def test_duplicate_ids_across_symbol_chunks_are_kept_once(self):
        client = FakeNewsClient([raw(1, ["AAPL"], "Apple Beats Estimates")])
        symbols = ["AAPL"] + [f"S{i}" for i in range(150)]  # forces two chunks of 100
        rows = news.fetch_articles(symbols, START, END, client=client)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(rows), 1)

    def test_same_headline_for_the_same_stock_is_a_duplicate_and_the_earliest_is_kept(self):
        rows = fetch([raw(2, ["AAPL"], "Apple Beats Estimates!", created="2026-11-20T12:00:00Z"),
                      raw(1, ["AAPL"], "apple beats   estimates", created="2026-11-20T11:00:00Z")])
        self.assertEqual([a["id"] for a in rows], [1])

    def test_articles_tagging_more_than_three_symbols_are_dropped(self):
        rows = fetch([raw(1, ["AAPL", "MSFT", "GOOG", "AMZN", "META"], "Big Tech Roundup")])
        self.assertEqual(rows, [])

    def test_symbols_outside_the_universe_are_ignored(self):
        rows = fetch([raw(1, ["AAPL", "SAMSUNG"], "Apple and Samsung")])
        self.assertEqual([a["symbol"] for a in rows], ["AAPL"])

    def test_requests_ask_for_full_text_in_the_window(self):
        client = FakeNewsClient([])
        news.fetch_articles(["AAPL"], START, END, client=client)
        self.assertTrue(client.requests[0].include_content)


class PriceMoveTest(unittest.TestCase):
    def test_headlines_describing_a_price_move_are_tagged(self):
        for h in ["Shares of Nike Decline After Earnings Miss",
                  "Advanced Micro Devices Shares Decline Following Hunterbrook Media Report",
                  "Tesla Stock Jumps on Delivery Numbers",
                  "Boeing shares fell 4% in premarket trading"]:
            self.assertTrue(news.is_price_move(h), h)

    def test_other_headlines_are_not(self):
        for h in ["Nike Declines to Comment on Report",
                  "Apple Raises Dividend",
                  "Analyst Raises Price Target on Shares of Visa to $400",
                  "Stock Split Announced by Walmart",
                  "Consumer Edge Report Highlights Ulta Beauty's Market Share Gains"]:
            self.assertFalse(news.is_price_move(h), h)


if __name__ == "__main__":
    unittest.main()
