# Quantitative AI Trading Bot & Performance Tracker

An automated, serverless algorithmic trading system that executes paper trades based on real-time price action and Natural Language Processing (NLP) sentiment analysis. 

The bot operates autonomously via GitHub Actions, scanning for high-volatility gaps, analyzing financial news with **FinBERT**, executing trades via the **Alpaca API**, and exporting end-of-day (EOD) performance metrics to a live **Astro / Cloudflare** frontend.

## System Architecture

The system utilizes an event-driven, two-phase daily lifecycle:

### Phase 1: Morning Research & Execution (8:30 AM EST)
1. **Dynamic Scanner:** Fetches a predefined watchlist of highly liquid stocks (e.g., AAPL, NVDA, SNOW) and calculates pre-market gap percentages using `yfinance`.
2. **News Aggregator:** Fetches the latest institutional news headlines for the top 5 gappers via the Alpaca News API (Benzinga).
3. **Sentiment Engine (NLP):** Passes headlines through `ProsusAI/finbert` (a financial-specific LLM) to generate a numeric sentiment score (-1.0 to +1.0).
4. **Broker Execution:** If the sentiment exceeds the risk threshold (>+0.15), it submits a notional market order of $1,000 to the Alpaca Paper Trading API.
5. **Data Artifact:** Saves the morning thesis to `data/targets.json`.

### Phase 2: Post-Market Evaluator (4:15 PM EST)
1. **Data Ingestion:** Reads the morning `targets.json` and downloads the 1-minute interval historical data for the trading session.
2. **Granular Backtesting:** Calculates the exact yield of the morning positions across multiple time horizons: `1-min`, `10-min`, `30-min`, `1-hour`, `5-hour`, and `EOD`.
3. **Data Export:** Appends the results to `data/results.json`, which is consumed live by the portfolio frontend using Chart.js.

## Tech Stack & Design Patterns

* **Language:** Python 3.12
* **APIs:** Alpaca Trading API, Alpaca News API, Yahoo Finance (`yfinance`)
* **AI / Machine Learning:** Hugging Face `transformers`, PyTorch, FinBERT
* **CI/CD & Automation:** GitHub Actions (Cron scheduling)
* **Frontend Integration:** JSON artifacts served via GitHub Raw Content to an Astro (SSG) web app.

### Highlight: The Adapter Pattern
The `sentiment.py` module utilizes the **Adapter Design Pattern**. Currently, the `analyze_text()` function parses text locally using the open-source FinBERT model. The architecture is explicitly designed so the sentiment engine can be swapped to OpenAI's API (e.g., `gpt-4o`) in the future by modifying just one function, without breaking the downstream broker or evaluator logic.

## Local Setup & Installation

To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
   cd YOUR-REPO-NAME