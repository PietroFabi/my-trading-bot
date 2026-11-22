"""FinBERT scoring. Uses the real (pinned) model: slow to load once, then a few seconds."""
import datetime as dt
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

import tests  # noqa: F401
import sentiment

NY = ZoneInfo("America/New_York")


def article(id_, headline, summary="", text=""):
    return {"id": id_, "symbol": "X", "created_at": dt.datetime(2026, 11, 20, 7, tzinfo=NY),
            "headline": headline, "summary": summary, "text": text, "url": "u", "source": "s", "price_move": False}


class ScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored = sentiment.score_articles([
            article(1, "Company beats estimates and raises full-year guidance"),
            article(2, "Company misses estimates and cuts its outlook"),
            article(3, "Company to present at investor conference next week"),
        ])

    def test_positive_news_scores_strongly_positive(self):
        self.assertGreater(self.scored[0]["s"], 0.5)

    def test_negative_news_scores_strongly_negative(self):
        self.assertLess(self.scored[1]["s"], -0.5)

    def test_routine_news_gets_little_weight(self):
        self.assertLess(self.scored[2]["w"], 0.3)

    def test_fields_are_consistent_probabilities(self):
        for a in self.scored:
            self.assertAlmostEqual(a["p_pos"] + a["p_neg"] + a["p_neu"], 1.0, places=4)
            self.assertAlmostEqual(a["s"], a["p_pos"] - a["p_neg"], places=6)
            self.assertAlmostEqual(a["w"], 1 - a["p_neu"], places=6)
            self.assertLessEqual(abs(a["s"]), a["w"] + 1e-9)  # tone can never exceed the non-neutral mass

    def test_scoring_is_deterministic_regardless_of_batch(self):
        # Alone vs. inside a batch the padding differs, which moves float results in the ~7th decimal.
        # No randomness is involved; 5 decimals is far finer than any threshold the strategy uses.
        again = sentiment.score_articles([article(1, "Company beats estimates and raises full-year guidance")])
        self.assertAlmostEqual(again[0]["p_pos"], self.scored[0]["p_pos"], places=5)

    def test_full_text_is_part_of_the_input(self):
        text = sentiment.model_input(article(1, "Headline", "Summary.", "Body text."))
        self.assertEqual(text, "Headline. Summary. Body text.")

    def test_pinned_model_revision(self):
        self.assertEqual(sentiment.MODEL_REVISION, "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031")


class CacheTest(unittest.TestCase):
    def test_cached_articles_are_not_rescored(self):
        cache = {7: {"p_pos": 0.1, "p_neg": 0.2, "p_neu": 0.7}}
        out = sentiment.score_articles([article(7, "Company beats estimates")], cache=cache)
        self.assertEqual(out[0]["p_pos"], 0.1)          # the model would have said ~0.9
        self.assertAlmostEqual(out[0]["s"], -0.1)

    def test_new_scores_are_added_to_the_cache(self):
        cache = {}
        sentiment.score_articles([article(8, "Company misses estimates")], cache=cache)
        self.assertIn(8, cache)

    def test_cache_file_round_trip_keeps_recent_entries_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scores.json")
            cache = {1: {"p_pos": 0.5, "p_neg": 0.25, "p_neu": 0.25, "created_at": "2026-11-19T07:00:00-05:00"},
                     2: {"p_pos": 0.5, "p_neg": 0.25, "p_neu": 0.25, "created_at": "2026-08-01T07:00:00-04:00"}}
            sentiment.save_cache(path, cache, keep_since=dt.datetime(2026, 10, 1, tzinfo=NY))
            loaded = sentiment.load_cache(path)
        self.assertEqual(list(loaded), [1])            # ids come back as ints; old entries pruned

    def test_cache_file_has_one_article_per_line_in_id_order_for_small_git_diffs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scores.json")
            row = {"p_pos": 0.5, "p_neg": 0.25, "p_neu": 0.25, "created_at": "2026-11-19T07:00:00-05:00"}
            sentiment.save_cache(path, {30: row, 4: row, 120: row}, keep_since=dt.datetime(2026, 10, 1, tzinfo=NY))
            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual(sorted(sentiment.load_cache(path)), [4, 30, 120])
        article_lines = [ln for ln in lines if ln.lstrip().startswith('"') and '"p_pos"' in ln]
        self.assertEqual([ln.split('"')[1] for ln in article_lines], ["4", "30", "120"])


if __name__ == "__main__":
    unittest.main()
