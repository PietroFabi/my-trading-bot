import os
from dotenv import load_dotenv
from transformers import pipeline
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

# Load the API keys from your .env file
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize Alpaca News Client
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY, raw_data=True)

print("Loading FinBERT AI Model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def fetch_news_headlines(ticker: str, limit: int = 3) -> list:
    """Fetches the latest news headlines using Alpaca's official News API."""
    try:
        req = NewsRequest(symbols=ticker, limit=limit)
        response = news_client.get_news(req)

        articles = response.get("news", [])

        headlines = [article.get("headline") for article in articles if "headline" in article]
        return headlines

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def analyze_text(text: str) -> float:
    """
    ADAPTER PATTERN: Uses FinBERT.
    Returns a float from -1.0 (highly negative) to 1.0 (highly positive).
    """
    try:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        confidence = result['score']

        if label == "positive":
            return confidence
        elif label == "negative":
            return -confidence
        else:
            return 0.0

    except Exception as e:
        print(f"Error analyzing text: {e}")
        return 0.0


def get_sentiment_for_tickers(tickers: list) -> dict:
    """Gets headlines via Alpaca, analyzes them, and averages the sentiment."""
    sentiment_results = {}

    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        headlines = fetch_news_headlines(ticker, limit=3)

        if not headlines:
            print(f"  -> No news found for {ticker}")
            sentiment_results[ticker] = 0.0
            continue

        total_score = 0.0
        for headline in headlines:
            score = analyze_text(headline)
            print(f"  -> [{score:+.2f}] {headline}")
            total_score += score

        avg_score = total_score / len(headlines)
        sentiment_results[ticker] = avg_score
        print(f"  => Average Sentiment for {ticker}: {avg_score:+.2f}\n")

    return sentiment_results


if __name__ == "__main__":
    # Test with the stocks your scanner found earlier!
    test_tickers = ["AMD", "NVDA", "SNOW"]
    results = get_sentiment_for_tickers(test_tickers)
    print("\nFinal Sentiment Scores:", results)