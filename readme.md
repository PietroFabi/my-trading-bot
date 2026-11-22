# Quantitative AI Trading Bot & Performance Tracker

An automated, serverless paper-trading system that trades **S&P 500 stocks long and short** around the open, on
the combination of a **pre-market gap** and a **news surprise** scored by **FinBERT**, with a **ridge regression**
deciding which signals are strong enough.

The strategy was developed the way a research claim should be: replayed on six months of history with the same code
the live bot runs, explored out of sample, **frozen and pre-registered in a public commit**, tested **once** on a
hold-out the exploration never saw, and now traded unchanged until a pre-registered live test decides whether it has an
edge. Every result, including the negative ones, is published in this repository and charted on an
**Astro / Cloudflare** portfolio site that reads `data/results.json`.

> Paper trading only: the Alpaca client is hard-coded to the paper endpoint (`paper=True`). No real money is involved.

## The strategy (v2)

| Step | Rule |
|---|---|
| Universe | The S&P 500 as listed on Wikipedia (refreshed weekly, `data/universe.json`), each stock mapped to its SPDR sector ETF |
| Liquidity | 20-day average dollar volume ≥ $20M, price ≥ $5, pre-market dollar volume ≥ $250k by the scan |
| Scan time | 08:55 New York (open − 35 min, from Alpaca's calendar). Prices up to 08:39 (the free data plan is 15 min delayed), news up to 08:55 |
| News | Every Alpaca news article since the previous close, scored by [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) (pinned revision) on headline + summary + text. Per article: tone `s = P(pos) − P(neg)`, weight `w = 1 − P(neutral)`. `news_today` = weighted tone of the fresh news; `baseline` = the stock's usual tone (20 sessions, half-life 5); `surprise` = today vs usual, shrunk when there is only one article; `news_volume` = more or fewer articles than usual |
| Price | Gap vs the previous close, **sector-adjusted**: `excess gap = gap − β × sector-ETF gap`, in ATR units (`gap_atr`); pre-market relative volume; trend vs the 50-day average; distance from the 52-week high |
| Gates | Long: sector-adjusted gap up **and** positive fresh news. Short: gap down **and** negative news. No fresh news, no trade |
| Model | Ridge regression (numpy, λ = 1) on 8 standardised features, predicting the **market-relative return** (stock − β × sector ETF) from the open to the close. A gated stock trades when the prediction agrees with its gate by more than **0.306%** |
| Selection | At most 10 trades a day, 3 per sector, strongest prediction first; shorts only if Alpaca reports the stock shortable and easy to borrow |
| Sizing | Equal risk: `equity × 0.06% ÷ ATR%`, at most 4% of equity per stock and 20% per day, whole shares |
| Orders | A market order at the open with a **stop-loss at 0.75 × ATR** attached as an Alpaca OTO leg; everything closed just before the close. Costs: 5 bps per side in every "net" figure |

## How it was tested (and what it found)

1. **Replay** (`backtest/build_dataset.py`): 123 mornings, 29 May → 20 Nov 2026, rebuilt exactly as the live bot
   would have seen them (index membership as listed that week, so stocks that later left the S&P 500 are included;
   66,377 scored articles; 5,637 gated stock-days). One replayed morning was checked against the live code path and
   matched to 5×10⁻⁹.
2. **Exploration** (`backtest/explore.py`, report: `docs/backtest/exploration-report.md`): walk-forward by month,
   each month predicted by a model trained only on earlier months; 32 variants, all listed. **No edge**: out-of-fold IC
   −0.02 (t −0.85), the best trading rule −0.02% net per trade. The diagnostics showed something else: news-confirmed
   gaps tended to **fade** during the day.
3. **Freeze** (`backtest/freeze.py`): the model, the rules and the success criteria written to
   `data/model/frozen_model.json` and `data/preregistration.json` and committed publicly (commit `aac07f6`, 22 Nov 2026)
   **before** the hold-out was read. Two hypotheses: **A**, the pre-registered strategy above (frozen even without an edge,
   as agreed before seeing data), and the **gap fade**, labelled as found in exploration, fixed with nothing tuned.
4. **Hold-out** (`backtest/holdout.py`, report: `docs/backtest/holdout-report.md`): 9 Oct → 20 Nov (31 sessions),
   evaluated **once**. The code refuses to run unless the frozen files are committed unchanged and their hash matches
   the pre-registration, and refuses a second run.

   | | IC (t) | trades | net per trade (t) | passes (IC > 0, t > 2.24, net > 0)? |
   |---|---|---|---|---|
   | A | 0.050 (1.24) | 31 | +0.15% (0.36) | no |
   | fade | 0.030 (0.75) | 305 | +0.17% (0.91) | no |

   Neither passed, so by the pre-registered rule **A trades live**.
5. **Live confirmation** (now): the model is frozen (no retraining). Success is judged after the later of 60 sessions
   and 150 trades: primary, the IC over **every gated stock, traded or not**, is > 0 with t > 2; secondary, the mean net
   return per trade is > 0. Progress is in `data/confirmation.json`.

## Daily lifecycle

Three GitHub Actions workflows run on weekdays. A small **Cloudflare Worker** (`scheduler/`) starts them on time
through GitHub's API, because GitHub's own cron started runs 4-7 hours late on this repo. GitHub's schedules are kept
as a backup. Each job checks **Alpaca's market calendar** for the real session times (daylight saving, holidays, half
days). A job that lands outside its window does nothing.

| Window (New York time) | Workflow | What it does |
|---|---|---|
| 07:00 – 09:25 | **Morning** (`src/main.py`) | Checks the frozen model's hash, closes any leftover position, scores the month of news, **waits until 08:55**, builds the feature table, selects and sizes the trades, queues the entries with their stop-loss legs, saves `data/plans/<date>.json` and `data/scores/<date>.csv` |
| close − 10 min – close − 1 min | **Exits** (`src/exits.py`) | For each trade: cancels the stop-loss and confirms the cancel, skips trades the stop already closed, then sells longs / buys back shorts with market orders and confirms the fills |
| after the close | **Afternoon** (`src/evaluator.py`) | Turns the real fills into a `data/results.json` entry, logs the outcome of every gated stock (`data/candidates/<date>.csv`), updates `data/confirmation.json`; on Fridays checks the scores for drift |

The morning job builds the table **as of 08:55** even if it starts later (the summer retry runs at 09:05): prices are
requested up to `as_of − 16 min` and news up to `as_of`, so the table never depends on when the job ran. Orders still
have to be queued by 09:25. A `dry_run` checkbox on the morning workflow's "Run workflow" button does everything except
orders and commits.

## Reliability by design

* **Calendar, not clock:** session dates and times come from Alpaca's calendar, never from the time a job started.
* **Idempotent:** every order has a deterministic `client_order_id` (`2026-11-16-CRWD`, `2026-11-16-CRWD-exit`,
  `...-late-exit`), so repeated attempts can't place duplicate orders.
* **A position can never flip:** the exit job closes at most the shares filled and still held on the trade's side,
  after the stop-loss leg is cancelled and confirmed; a stop that filled meanwhile means nothing is sent.
* **The frozen model cannot change silently:** the morning job checks the model file's sha256 against the
  pre-registration and trades nothing on a mismatch.
* **Fails loudly, but only when you must act:** a missed order window, failed orders or a leftover it could not close
  fail the workflow (GitHub emails). Things the bot handles itself are `::notice::` lines: data coverage below normal
  (daily data < 70%, pre-market prices < 40%, no fresh news) and weekly score drift outside the band in
  `data/backtest/score_band.json`.
* **No lost sessions:** each plan is its own file and the evaluator records sessions oldest first.
* **Least privilege:** each workflow gets only the permissions it needs (the exit job cannot write to the repo); API
  keys come from GitHub Actions secrets.

## Scheduler (Cloudflare Worker)

`scheduler/` holds a Worker whose only job is to start the workflows on time. It has no public URL: Cloudflare Cron
Triggers call it, and it calls GitHub's `workflow_dispatch` API, which starts a run immediately instead of queueing it
like scheduled runs.

| Trigger (UTC) | Winter (EST) | Summer (EDT) | Workflow |
|---|---|---|---|
| 12:45, 13:05 | 07:45, 08:05 | 08:45, 09:05 | morning (main attempt + retry) |
| 16:55 – 20:55, hourly at :55 | | | exits (only the run just before today's close acts: 20:55 winter / 19:55 summer, earlier on half days) |
| 21:30 | 16:30 | 17:30 | afternoon |

One-time setup:
1. On GitHub, create a **fine-grained personal access token** (Settings → Developer settings → Fine-grained tokens):
   repository access *Only select repositories* → this repo; permission **Actions: Read and write**. Nothing else.
2. Deploy from the `scheduler/` folder:
   ```bash
   npm install
   npx wrangler login
   npx wrangler secret put GITHUB_TOKEN
   npm run deploy
   ```
   `secret put` prompts for the token, so it never ends up in a file or in your shell history.
3. Check it under Workers & Pages → trading-bot-scheduler: the four Cron Triggers are listed, and each run's logs show
   `Started <workflow>.yml`.

Test locally: `npm run dev`, then `curl "http://127.0.0.1:8787/cdn-cgi/local/scheduled?cron=0+15+*+*+MON-FRI"`.
This calls the real GitHub API unless you pass `--var GITHUB_API:<fake server>`.

## Data files

| File | Written by | Purpose |
|---|---|---|
| `data/results.json` | Afternoon | One entry per session, oldest first, never trimmed. **Read by the portfolio site: fields are only ever added, never renamed** |
| `data/plans/<date>.json` | Morning | The day's decision (v2 keys below) |
| `data/scores/<date>.csv` | Morning | Every scanned stock (~503 rows): its features, gates, `score` and `decision`. No article text |
| `data/candidates/<date>.csv` | Afternoon | The would-have-traded log: the realised outcome of every gated stock, traded or not (feeds the IC) |
| `data/confirmation.json` | Afternoon | Live confirmation progress against the pre-registered criteria |
| `data/preregistration.json` | Freeze (once) | The public pre-registration |
| `data/model/frozen_model.json` | Freeze (once) | Everything the live bot needs to trade; nothing left to tune |
| `data/backtest/dataset_summary.json` | Replay | Size of the replayed dataset |
| `data/backtest/exploration.json` | Exploration | Every variant tried and the chosen one, out of fold |
| `data/backtest/holdout.json` | Hold-out (once) | The one-time hold-out result and which strategy goes live |
| `data/backtest/score_band.json` | Backtest | The band for the weekly score-drift notice |
| `data/sentiment/finbert_cache.json` | Morning | Rolling FinBERT scores by article id (no text), so a morning only scores new articles |
| `data/universe.json` | Morning | The S&P 500 list with sectors, refreshed weekly |

**Plan v2** (`data/plans/<date>.json`): `strategy` ("v2"), `strategy_name` (the live hypothesis, "A"),
`model_sha256`, `scan_as_of` (08:55), `entry` ("open"), `stop_k` (0.75), `score_threshold`, `equity` (account equity
used for sizing), `coverage` (data coverage shares), `candidates` (every gated stock with its score and decision) and
`targets` (side, score, features, reference price, ATR, qty, stop price, fresh headlines, order ids).

**Would-have-traded log** (`data/candidates/<date>.csv`): `session_date`, `symbol`, `gate` (long/short),
`sector_etf`, `beta`, `score`, `decision` (selected, below threshold, hard to borrow, daily or sector limit),
`target` (realised market-relative return, open → close), `traded`.

**Confirmation** (`data/confirmation.json`): `sessions` and `trades` so far, `required` (60 sessions, 150 trades,
IC t > 2), `progress` (0–1 each), `ic` and `ic_t` (daily rank correlation of score vs outcome over all gated stocks),
`mean_net_per_trade_pct` and `mean_net_t`, `primary_passed` and `secondary_passed` (null until complete), `status`
("in progress" | "passed" | "failed"), `first_session`, `updated_after`.

**Research files:**
- `data/preregistration.json`: `frozen_on`, `frozen_model_file`, `frozen_model_sha256`, `hypotheses` (A and fade, with
  their exploration numbers), `variants_tried`, `holdout` (dates and pass rule), `decision_rule`, `live_confirmation`,
  `costs`.
- `data/model/frozen_model.json`: `version`, `data_cutoff`, `finbert` (model and revision), `scan_before_open_min`,
  `selection`, `sizing`, `costs`, `strategies` (A: ridge weights, scaler, threshold, entry, stop; fade: its rule).
- `data/backtest/exploration.json`: `window`, `variants_tried`, `stage1` / `stage1_choice` (model settings by IC),
  `stage2` / `stage2_choice` (trading rules by net return), `chosen_rule_by_side_and_horizon`, `claimed_candidates`,
  `ic_by_month`, `benchmarks` (SPY), `final_model`, `diagnostics`.
- `data/backtest/holdout.json`: `holdout` (dates, rows), `pass_rule`, `A` and `fade` (IC, trades, returns, `passes`),
  `goes_live`, `benchmarks`.
- `data/backtest/score_band.json`: `window` (5 sessions), `windows`, `sessions`, `mean` and `std` (95% ranges),
  `strategy`, `model_sha256`.
- `data/backtest/dataset_summary.json`: `sessions`, `first_session`, `last_session`, `distinct_stocks`, `articles`,
  `mean_per_session`, `gated_rows_total`.

### `results.json` entry

v1 sessions (until the switch-over) look like this; v2 sessions keep every one of these fields with the same meaning
and add the fields listed after the example.

```jsonc
{
  "date": "2026-11-16",                 // trading session (New York date)
  "starting_balance": 100000.0,         // fund value at the open = previous ending_balance
  "ending_balance": 100143.2,           // fund value after the closing fills
  "portfolio_curve": {                  // fund value if every position were closed at that point (gross)
    "Open": 100000.0, "1_min": 99980.1, "10_min": 100031.5, "30_min": 100060.0,
    "1_hour": 100012.4, "5_hour": 100120.7, "EOD": 100143.2   // EOD = real closing fills
  },
  "invested_usd": 19842.0,              // money in positions that day (rest stays in cash)
  "sentiment_threshold": 0.15,          // v1 only; null for v2
  "trades": [{
    "ticker": "CRWD",
    "gap_percent": 2.45,                // pre-market gap vs previous close
    "sentiment_score": 0.55,            // v1: mean headline score; v2: FinBERT tone of the fresh news (news_today)
    "headlines": [{
      "headline": "...", "source": "benzinga", "url": "https://...", "created_at": "2026-11-16T11:02:00Z",
      "label": "positive", "confidence": 0.93, "score": 0.93   // v1 FinBERT output; v2 headlines carry score = tone and price_move
    }],
    "qty": 42, "allocation_usd": 9891.0,
    "open_price": 235.5,                // real entry fill (market order at the open)
    "exit_price": 238.9,                // real exit fill (or the stop fill, or the official close if missed)
    "exit_status": "filled",            // "filled" | "missed" (valued at the close for now) | "late" (closed next morning; real price)
    "performance_pct": { "1_min": -0.1, "...": 0, "EOD": 1.44 },   // side-aware: a short gains when the price falls
    "pnl_usd":         { "1_min": -9.9, "...": 0, "EOD": 142.8 }
  }],
  "rejected": [{                        // stocks the model looked at but did not hold
    "ticker": "MSFT", "gap_percent": -2.45, "sentiment_score": -0.49, "headlines": [ /* as above */ ],
    "reason": "sentiment below threshold"  // v2: "below threshold", "hard to borrow", "daily or sector limit", ...
  }]
}
```

**v2 additions** (new fields only; nothing above is renamed):
- Entry: `strategy` ("v2"; the fund is continuous: the first v2 session starts from the last v1 `ending_balance`,
  so the field marks where the strategy changed),
  `strategy_name` ("A"), `model_sha256`, `score_thresholds` (`{"long": 0.003, "short": -0.003}`), `costs_bps` (5),
  `portfolio_curve_net` (the curve after 5 bps per side), `ending_balance_net`, `by_side` (per side: trades,
  invested, gross and net P&L per horizon, return %).
- Trade: `side` ("long" | "short"), `score` (the model's predicted market-relative return), `features` (the inputs),
  `stop_price` (planned), `exit_reason` ("close" | "stop": once the stop fills, every later horizon uses the stop
  price), `performance_pct_net`, `pnl_usd_net`, `bars_incomplete` (true if the stock stopped trading during the
  day, e.g. a halt; its horizons then use the last known price).
- `rejected` lists every gated stock that was not traded (often ~40 a day) with its decision as the `reason`.

Only `EOD` is an executed result. The other `portfolio_curve` points answer "what if every position had been closed
N minutes after the open?" for the same entries.

## Tech stack

* **Language:** Python 3.12; numpy/pandas (the ridge regression is the numpy closed form, no ML framework)
* **APIs:** Alpaca Trading API (paper, OTO orders), Alpaca market data (SIP, 15-min delayed free plan) and News API,
  Yahoo Finance (fallback), Wikipedia (S&P 500 membership)
* **NLP:** Hugging Face `transformers`, PyTorch, FinBERT (pinned revision)
* **Automation:** GitHub Actions, triggered on time by a Cloudflare Worker (Cron Triggers)
* **Frontend:** JSON served via GitHub raw content to an Astro site with Chart.js

### Design notes
- **One code path.** The backtest and the live bot share `src/` (universe, market data, news, FinBERT, features,
  model, selection, sizing), so what was tested is exactly what trades.
- **Adapter pattern.** `src/sentiment.py` is the only module that knows which model scores text; everything else
  consumes five numbers per article (`p_pos`, `p_neg`, `p_neu`, `s`, `w`).
- **Text is never stored.** Articles are scored in memory; only headlines, links, sources and scores are kept.

## Project structure

```
src/
  main.py            morning job: calendar, frozen-model check, safety net, 08:55 scan, plan, entries
  strategy.py        frozen model loading, scoring, selection, sizing, stop prices
  model.py           ridge regression, walk-forward, information coefficient, the target definition
  features.py        price features, news features, gates, the morning feature table
  universe.py        S&P 500 membership and sector ETFs
  marketdata.py      Alpaca SIP bars (delay-aware) with Yahoo fallback
  news.py            Alpaca news: clean, dedupe, tag price-move headlines
  sentiment.py       FinBERT scoring and the rolling score cache
  broker.py          orders: OTO entries, stop-aware exits, leftover clean-up
  exits.py           exit job (light: alpaca-py + python-dotenv)
  evaluator.py       afternoon job: v1 sessions, dispatches v2 to evaluator_v2.py
  evaluator_v2.py    v2 results, would-have-traded log, confirmation progress
  alerts.py          coverage and score-drift notices
  market_session.py  Alpaca market calendar helpers
  storage.py         file paths and JSON helpers
backtest/            replay, simulator, exploration, freeze, hold-out, reports (run locally)
tests/               unittest suite (fake broker and synthetic data; no orders, no network)
docs/                design spec, implementation plan, backtest reports
.github/workflows/   morning.yml, exits.yml, afternoon.yml
scheduler/           Cloudflare Worker that starts the workflows on time
data/                see "Data files"
```

## Local setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/PietroFabi/my-trading-bot.git
   cd my-trading-bot
   python -m venv venv
   venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
2. **Add your Alpaca paper keys** to a `.env` file in the project root (gitignored; never commit it):
   ```
   ALPACA_API_KEY=your_paper_key
   ALPACA_SECRET_KEY=your_paper_secret
   ```
   For GitHub Actions, add the same two values as repository secrets.
3. **Run the tests:** `python -m unittest discover -s tests -t .`
4. **Run a job by hand:** `python src/main.py` (set `DRY_RUN=1` for a plan without orders, written to `data/dryrun/`),
   `python src/exits.py` or `python src/evaluator.py`. Each one checks the market calendar and exits outside its window.
5. **Reproduce the research** (slow; caches go to `.cache/`, gitignored): `python -m backtest.build_dataset`, then
   `python -m backtest.explore` and `python -m backtest.report`. The freeze and the hold-out are one-time steps that
   have already been run.
