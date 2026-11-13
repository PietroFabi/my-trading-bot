# Quantitative AI Trading Bot & Performance Tracker

An automated, serverless paper-trading system that combines pre-market price action with NLP sentiment analysis of financial news.

Every trading day it scans a watchlist for pre-market gaps, scores the latest headlines with **FinBERT**, buys the approved stocks at the opening auction and sells them at the closing auction on an **Alpaca paper account**. It then records the real fills in `data/results.json`, which a live chart on an **Astro / Cloudflare** portfolio site reads.

> Paper trading only: the Alpaca client is hard-coded to the paper endpoint (`paper=True`). No real money is involved.

## Daily lifecycle

Three GitHub Actions workflows run on weekdays. GitHub's cron is in UTC and runs can start late, so each job checks **Alpaca's market calendar** for the real session times. That covers US daylight saving, market holidays and half days. A job that lands outside its window does nothing.

| Window (New York time) | Workflow | What it does |
|---|---|---|
| 07:00 – 09:25 | **Morning** (`src/main.py`) | Scans, scores sentiment, queues **market-on-open** buys and saves the plan to `data/plans/<date>.json` |
| open + 2 min – close − 12 min | **Midday** (`src/midday.py`) | Queues **market-on-close** sells for the shares actually held |
| after the close | **Afternoon** (`src/evaluator.py`) | Reads the real fills and appends the session to `data/results.json` |

### 1. Morning research & orders
1. **Scanner:** for 15 liquid, news-sensitive stocks, computes the pre-market gap: the latest pre-market trade compared with the previous session's close (`yfinance`, both pinned to explicit dates).
2. **News:** fetches the latest headlines for the top 5 movers from the Alpaca News API (Benzinga).
3. **Sentiment:** scores each headline with [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) on a scale of −1 to +1 and averages the scores per stock.
4. **Orders:** stocks scoring above **+0.15** get a buy order for **10% of the $100,000 fund** ($10,000) in whole shares, as a market-on-open (`OPG`) order. It fills in the opening auction.

### 2. Midday exits
Once the opening buys have filled, the bot queues a market-on-close (`CLS`) sell for exactly the shares it holds. It never sells more than it owns, so it cannot open a short position by accident.

### 3. Afternoon evaluation
For each finished session, the evaluator reads the real entry and exit fill prices from Alpaca. It values the position at Yahoo 1-minute prices for the intraday curve (`Open`, `1_min`, `10_min`, `30_min`, `1_hour`, `5_hour`, `EOD`). If a closing order was missed, the official close is used and the trade is flagged `"exit_status": "missed"`. The next morning closes any leftover position.

## Reliability by design

* **Calendar, not clock:** session dates and times come from Alpaca's calendar, never from the time a job started. A run that GitHub starts hours late still records the right day.
* **Idempotent:** every order has a deterministic `client_order_id` (`2026-11-16-CRWD`, `2026-11-16-CRWD-exit`), so repeated attempts can't place duplicate orders. Several cron attempts per window make a timely run likely.
* **Fails loudly:** missed windows, failed orders and data outages fail the workflow, so GitHub sends a notification instead of the bot quietly recording a flat day.
* **No lost sessions:** each plan is its own file, and the evaluator records sessions oldest first, so balances always carry forward in order.
* **Least privilege:** each workflow only gets the permissions it needs (the midday job cannot write to the repo), and API keys come from GitHub Actions secrets.

## Data files

| File | Written by | Purpose |
|---|---|---|
| `data/plans/<date>.json` | Morning | The day's targets, sentiment scores, share counts and order ids |
| `data/results.json` | Afternoon | One entry per session, oldest first, never trimmed. **Read by the portfolio site, so keep field names stable (add, don't rename).** |

### `results.json` entry

```jsonc
{
  "date": "2026-11-16",                 // trading session (New York date)
  "starting_balance": 100000.0,         // fund value at the open = previous ending_balance
  "ending_balance": 100143.2,           // fund value after the closing fills
  "portfolio_curve": {                  // fund value if every position were sold at that point
    "Open": 100000.0, "1_min": 99980.1, "10_min": 100031.5, "30_min": 100060.0,
    "1_hour": 100012.4, "5_hour": 100120.7, "EOD": 100143.2   // EOD = real closing fills
  },
  "invested_usd": 19842.0,              // money in positions that day (rest stays in cash)
  "sentiment_threshold": 0.15,
  "trades": [{
    "ticker": "CRWD",
    "gap_percent": 2.45,                // pre-market gap vs previous close
    "sentiment_score": 0.55,            // average of the headline scores below
    "headlines": [{
      "headline": "...", "source": "benzinga", "url": "https://...", "created_at": "2026-11-16T11:02:00Z",
      "label": "positive", "confidence": 0.93, "score": 0.93   // FinBERT output; score = ±confidence, neutral = 0
    }],
    "qty": 42, "allocation_usd": 9891.0,
    "open_price": 235.5,                // real opening-auction fill
    "exit_price": 238.9,                // real closing-auction fill (or official close if missed)
    "exit_status": "filled",            // "filled" | "missed"
    "performance_pct": { "1_min": -0.1, "...": 0, "EOD": 1.44 },
    "pnl_usd":         { "1_min": -9.9, "...": 0, "EOD": 142.8 }
  }],
  "rejected": [{                        // stocks the model looked at but did not hold
    "ticker": "MSFT", "gap_percent": -2.45, "sentiment_score": -0.49, "headlines": [ /* as above */ ],
    "reason": "sentiment below threshold"  // or "price above budget", "order could not be placed", "buy order was not filled"
  }]
}
```

Only `EOD` is an executed result. The other `portfolio_curve` points answer "what if every position had been sold N minutes after the open?" for the same entries. Adding up each key's daily gain (`curve[key] − starting_balance`) across sessions compares holding periods.

## Tech stack

* **Language:** Python 3.12
* **APIs:** Alpaca Trading API (paper), Alpaca News API, Yahoo Finance (`yfinance`)
* **AI / ML:** Hugging Face `transformers`, PyTorch, FinBERT
* **Automation:** GitHub Actions (cron schedules)
* **Frontend:** JSON served via GitHub raw content to an Astro site with Chart.js

### Design note: the adapter pattern
`analyze_text()` in `src/sentiment.py` is the only place that knows which model scores the text. Swapping FinBERT for another model, such as an LLM API, means changing that one function. The scanner, broker and evaluator stay the same.

## Project structure

```
src/
  main.py            morning job: calendar checks, scan, sentiment, orders
  scanner.py         pre-market gap scanner
  sentiment.py       Alpaca news + FinBERT scoring
  broker.py          Alpaca orders (open buys, close sells, leftover clean-up)
  midday.py          midday job: queues the closing sells
  evaluator.py       afternoon job: real fills -> results.json
  market_session.py  Alpaca market calendar helpers
  storage.py         file paths and JSON helpers
.github/workflows/   morning.yml, midday.yml, afternoon.yml
data/                plans/ and results.json
```

## Local setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PietroFabi/my-trading-bot.git
   cd my-trading-bot
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
3. **Add your Alpaca paper keys** to a `.env` file in the project root. The file is gitignored; never commit it.
   ```
   ALPACA_API_KEY=your_paper_key
   ALPACA_SECRET_KEY=your_paper_secret
   ```
   For GitHub Actions, add the same two values as repository secrets.
4. **Run a job by hand:** `python src/main.py`, `python src/midday.py` or `python src/evaluator.py`. Each one checks the market calendar and exits if it is outside its window.
