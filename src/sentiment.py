import os

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv
from transformers import pipeline

# Load the API keys from your .env file
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize Alpaca News Client
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY, raw_data=True)

print("Loading FinBERT AI Model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def fetch_news(ticker: str, limit: int = 3) -> list:
    """
    Latest articles for a ticker from Alpaca's News API (Benzinga).
    Keeps only the headline, timestamp, source and link: enough to show the
    evidence on the site and link back to the publisher, without copying articles.
    """
    try:
        response = news_client.get_news(NewsRequest(symbols=ticker, limit=limit))
        return [
            {
                "headline": a["headline"],
                "created_at": a.get("created_at"),
                "source": a.get("source"),
                "url": a.get("url"),
            }
            for a in response.get("news", [])
            if a.get("headline")
        ]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def analyze_text(text: str) -> dict:
    """
    ADAPTER PATTERN: the only function that knows which model scores text (FinBERT).
    Returns the model's label, its confidence (0-1) and a signed score from
    -1.0 (confidently negative) to +1.0 (confidently positive); neutral scores 0.
    """
    try:
        result = sentiment_analyzer(text)[0]
        label, confidence = result["label"], float(result["score"])
        score = confidence if label == "positive" else -confidence if label == "negative" else 0.0
        return {"label": label, "confidence": round(confidence, 4), "score": round(score, 4)}
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {"label": "error", "confidence": 0.0, "score": 0.0}


def get_sentiment_for_tickers(tickers: list) -> dict:
    """
    For each ticker: {"score": average signed score, "headlines": [article + FinBERT result, ...]}.
    The per-headline detail is saved so the site can show why the model took a position.
    """
    results = {}
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        articles = fetch_news(ticker, limit=3)
        if not articles:
            print(f"  -> No news found for {ticker}")
            results[ticker] = {"score": 0.0, "headlines": []}
            continue

        for article in articles:
            article.update(analyze_text(article["headline"]))
            print(f"  -> [{article['score']:+.2f} {article['label']}] {article['headline']}")

        avg = sum(a["score"] for a in articles) / len(articles)
        results[ticker] = {"score": avg, "headlines": articles}
        print(f"  => Average Sentiment for {ticker}: {avg:+.2f}\n")
    return results
