# Long/Short News-Surprise Strategy: Implementation Plan

> **For agentic workers:** implement step by step with superpowers:executing-plans, as the user asked: one step at a
> time, each step tested and reported before the next starts. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Never commit unless the user asks.**

**Goal:** replace the 15-stock long-only bot with a backtested, pre-registered long/short S&P 500 strategy
(news surprise + market-relative gap, ridge model), and keep the live system and the site working throughout.

**Architecture:** one shared library (`src/universe.py`, `marketdata.py`, `news.py`, `sentiment.py`, `features.py`,
`model.py`, `strategy.py`) is used by both the local backtest (`backtest/`) and the live morning job. That means one
code path, so no train/serve mismatch. The backtest explores, freezes a model plus rules into a committed
pre-registration, evaluates a hold-out once, and the live bot then trades exactly that frozen model.

**Tech stack:** Python 3.12, alpaca-py (orders, SIP data, news), yfinance (fallback), transformers + torch (FinBERT),
numpy/pandas (ridge and statistics, no new dependency), stdlib `unittest`, GitHub Actions, Cloudflare Worker (unchanged).

**Spec:** `docs/superpowers/specs/2026-11-21-long-short-strategy-design.md` (confirmed 2026-11-21).

## Global Constraints

- Paper only: `TradingClient(..., paper=True)`. Tests use a fake broker. Real orders only with the user's explicit OK.
- Plain market DAY entries/exits. No OPG/CLS orders. Deterministic `client_order_id`s; exits confirm fills.
- Times from Alpaca's calendar (`src/market_session.py`), never the wall clock or "today".
- `data/results.json`: existing fields keep their names and meaning; only add fields. `exit_status` stays
  `filled|missed|late`; late-exit revision kept.
- Exit job imports only `alpaca-py` + `python-dotenv` (never yfinance/torch/pandas).
- Alert policy: a run fails only when the user must act; everything self-handled is a `::notice::`.
- Dependencies pinned exactly in `requirements.txt`; no new packages unless a step says so and the user agrees.
- Article text is scored in memory and never written to disk or the repo (headline, url, source and scores are kept).
- Free tiers only; Cloudflare stays at 4 of 5 triggers; `main` and the Worker untouched without the user's OK.
- Fixed values from the spec: scan at open − 35 min; SIP data ends at `as_of − 16 min`; ADV ≥ $20M; price ≥ $5;
  pre-market $ volume ≥ $250k; baseline 20 sessions, half-life 5; shrink k = 2; max 10 trades/day, 3 per sector;
  R = 0.06% of equity per ATR; caps 4%/stock, 20%/day; costs 5 bps per side; stop k ∈ {0.75, 1, 1.5, none};
  history from 2026-04-29; hold-out ≈ the last 6 weeks (≈ 2026-10-09 → 2026-11-20); confirmation = the later of
  60 sessions and 150 trades; success = IC > 0 with t > 2 (primary) and mean net return per trade > 0 (secondary).

## Review Focus

Failure modes the spec implies but no step's main tests cover. Each one gets a pinning test in the step named.
1. **Half-day sessions (13:00 close):** horizons past the close clamp to the close; the exit job and stops still work. → tests in Steps 8 and 13.
2. **Monday / day after a holiday:** "fresh" news starts at the previous *session's* close (includes the weekend); the previous close comes from the calendar. → test in Step 6.
3. **Split or corporate action overnight:** a fake 50% "gap". Daily bars split-adjusted; |gap| > 40% is flagged and excluded. → test in Step 3.
4. **Dotted symbols** (`BRK.B`, `BF.B`): Alpaca/Wikipedia use `.`, Yahoo uses `-`; plans and results always use the Alpaca form. → test in Step 1.
5. **Stop fills while the exit job is cancelling it:** re-read the position after the cancel; never flip. → test in Step 12.

---

## Phase A: shared library (Steps 1–6). Each step is testable alone with real read-only data.

### Step 1: Universe (`src/universe.py`) + test scaffolding

**What:** load the S&P 500 members with their GICS sector and sector ETF. Cache them in `data/universe.json`,
refresh if older than 7 days, and fall back to the cache, then to the GitHub "datasets" CSV. Also create `tests/`
(`tests/__init__.py`, `tests/fakes.py` with a fake Alpaca trading client built from the existing ad-hoc stubs) so
every later step has a home for its tests.

**Interfaces:** produces `Member = dict(symbol: str  # Alpaca form, yahoo: str, sector: str, sector_etf: str)`;
`load_universe(session_date: date, cache_path=UNIVERSE_FILE, fetch=fetch_wikipedia) -> list[Member]`;
`SECTOR_ETF: dict[str, str]` (Information Technology→XLK, Health Care→XLV, Financials→XLF,
Consumer Discretionary→XLY, Communication Services→XLC, Industrials→XLI, Consumer Staples→XLP,
Energy→XLE, Utilities→XLU, Real Estate→XLRE, Materials→XLB).

| Option | Pro | Con |
|---|---|---|
| **`pandas.read_html` on the `constituents` table (recommended)** | Already tested (503 rows in 1 s); lxml is installed | Breaks if Wikipedia renames the table id (caught by the fallback + a row-count check) |
| requests + BeautifulSoup by hand | More control | More code for the same result |

Survivorship bias: for the backtest, Step 7 rebuilds membership per date from the page's "changes" table.

- [ ] Tests (`tests/test_universe.py`): parse a saved HTML fixture → 503 members, `BRK.B`/`yahoo == "BRK-B"`,
  every sector maps to an ETF; a cache 3 days old is used without fetching; a cache 8 days old triggers a fetch;
  a failing fetch with a cache returns the cache plus a notice; a failing fetch without a cache raises; a result
  with < 450 rows is rejected as broken.
- [ ] Implement; run `python -m unittest tests.test_universe -v` → all pass.
- [ ] Real read-only check: `load_universe(date(2026, 11, 23))` → ~503 members, 11 sectors.
- **Done:** tests pass, `data/universe.json` written, real fetch verified and reported.

### Step 2: Market data (`src/marketdata.py`)

**What:** batched Alpaca SIP daily and 1-minute bars (split-adjusted), never newer than `as_of − 16 min`
(the free-plan delay), with Yahoo as a per-symbol fallback. An optional on-disk cache for the backtest goes in
`.cache/` (gitignored).

**Interfaces:** `daily_bars(symbols, start: date, end: date, cache=None) -> DataFrame` indexed (symbol, date) with
open/high/low/close/volume; `minute_bars(symbols, start: datetime, end: datetime, cache=None) -> DataFrame` indexed
(symbol, timestamp in New York time); `latest_allowed(as_of: datetime) -> datetime` (= as_of − 16 min);
`coverage(df, symbols) -> float`.

| Choice | Options | Recommendation |
|---|---|---|
| Client | alpaca-py `StockHistoricalDataClient` vs raw REST | **alpaca-py**: already a dependency, handles pagination |
| Fallback granularity | whole request vs per missing symbol | **per missing symbol**: keeps Alpaca data wherever it exists |
| Cache format | pickle vs CSV vs parquet | **pickle** (local only, fast, no new dependency; parquet would need pyarrow) |

- [ ] Tests (`tests/test_marketdata.py`, fake data client): requests never end after `as_of − 16 min`; symbols
  missing from Alpaca are fetched from Yahoo with `.`→`-` and mapped back; an Alpaca exception → full Yahoo
  fallback plus a notice; the cache returns the same frame without a second call.
- [ ] Real read-only check: Fri 2026-11-20, as_of 08:55 → coverage ≥ 0.8; three spot prices vs Yahoo within 0.5%.
- **Done:** tests pass, real coverage/time reported.

### Step 3: Price features (`src/features.py`, part 1)

**What:** pure functions turning bars into one row per stock "as of" the scan time:
`prev_close`, `ref_price` (last pre-market trade ≤ cutoff), `gap_pct`, `beta` (1 y daily returns vs the sector ETF,
OLS, clipped 0–3, ≥ 120 observations else 1.0), `sector_gap_pct`, `excess_gap_pct`, `atr_pct`, `gap_atr`,
`premarket_usd`, `rvol`, `trend` (price / 50-day MA − 1), `dist_52w_high`, `adv_usd`, `liquid` (all filters), `suspect_gap` (|gap| > 40%).

**Interfaces:** `price_features(daily: DataFrame, minute: DataFrame, members: list[Member], session: Session,
prev: Session, as_of: datetime) -> DataFrame` (index = symbol, columns above).

| Choice | Options | Recommendation |
|---|---|---|
| ATR | Wilder smoothing vs simple 14-day mean of true range | **Wilder** (the standard definition; documented) |
| rvol baseline | same clock window over 5 / 10 / 20 prior sessions | **10**: 5 is noisy, 20 needs more minute data per morning |
| Beta window | 6 months vs 1 year | **1 year** (spec), fallback 1.0 for new listings |

- [ ] Tests (`tests/test_price_features.py`, synthetic bars with known answers): ATR on a hand-computed series;
  beta = 2.0 for a series built as 2 × ETF returns; excess gap = 0 when the stock moves exactly β × ETF; rvol = 3 when
  today's window has 3× the average volume; a bar after the cutoff is ignored (look-ahead guard); **(RF3)** a 2-for-1
  split day gives `suspect_gap = True` and `liquid = False`; a stock with no pre-market trade → `ref_price` NaN and not liquid.
- [ ] Real read-only check on 2026-11-20: AMD's `gap_pct` ≈ −6.9% (matches the live plan); feature summary table reported.
- **Done:** tests pass, real-day table sane (no NaN beyond the unpriced stocks).

### Step 4: News fetch and cleaning (`src/news.py`)

**What:** fetch all Alpaca articles for the universe in a window (100-symbol chunks, full pagination,
`include_content=True`). Strip HTML from the text, dedupe by id and normalised headline, drop articles tagging
> 3 symbols, tag `price_move` headlines. The text exists only in memory.

**Interfaces:** `Article = dict(id, symbol, created_at: datetime, headline, summary, text, url, source, price_move: bool)`
(one row per article × universe symbol); `fetch_articles(symbols, start: datetime, end: datetime, client=None) -> list[Article]`;
`PRICE_MOVE_PATTERN` (regex: shares/stock + rise/rises/jump/surge/gain/climb/fall/drop/decline/slide/sink/tumble/plunge).

| Choice | Options | Recommendation |
|---|---|---|
| HTML stripping | stdlib `html.parser` vs BeautifulSoup (installed) | **stdlib**: one tiny class, no extra dependency for live |
| Price-move detection | regex vs a FinBERT/LLM classifier | **regex** (transparent; precision checked on 50 samples) |

- [ ] Tests (`tests/test_news.py`, fixture JSON): duplicates removed (same id; same normalised headline);
  a 5-symbol article dropped; `<p>a &amp; b</p>` → `a & b`; "Shares of X Decline After…" tagged, "X Declines to Comment" not;
  pagination follows `next_page_token` until it is absent.
- [ ] Real read-only check: 2026-11-19 16:00 → 2026-11-20 08:55 ≈ 180 articles; precision of `price_move` on 50 tagged headlines reported.
- **Done:** tests pass, counts and tag precision reported.

### Step 5: FinBERT scoring (`src/sentiment.py`, rewritten)

**What:** batched FinBERT on `headline + ". " + summary + " " + text` (truncated to 512 tokens), returning
`p_pos, p_neg, p_neu`, `s = p_pos − p_neg`, `w = 1 − p_neu`. Model revision pinned. An optional score cache keyed by
article id (backtest: `.cache/finbert.pkl`) lets ~60k articles be scored once.

**Interfaces:** `score_articles(articles: list[Article], cache: dict | None = None, batch_size=32) -> list[Article]`
(adds the five fields); `MODEL_ID = "ProsusAI/finbert"`, `MODEL_REVISION = <commit hash recorded at implementation>`.

| Choice | Options | Recommendation |
|---|---|---|
| Probabilities | pipeline `top_k=None` vs model + softmax directly | **model + softmax**: batches cleanly, no pipeline quirks |
| Pinning | latest vs pinned revision | **pinned**: backtest and live must use the identical model |

- [ ] Tests (`tests/test_sentiment.py`, real model, small): "Company beats estimates and raises guidance" → s > 0.5;
  "Company misses estimates, cuts outlook" → s < −0.5; "Company to present at conference" → w < 0.3;
  scoring twice gives identical numbers; a cached id is not re-scored.
- [ ] Measure throughput on 200 real articles (full text) and report.
- **Done:** tests pass; throughput reported; the overnight backtest scoring time estimated from it.

### Step 6: News features (`src/features.py`, part 2)

**What:** per stock, as of the scan: `n_fresh`, `news_today` (w-weighted mean of fresh `s`), `baseline`
(exponentially weighted daily means over the previous 20 sessions, half-life 5, 0 if none), `shrunk`
(`baseline + n/(n+2)·(news_today − baseline)`), `surprise = shrunk − baseline`, `news_volume`
(`log((n_fresh+1)/(avg daily count over 20 sessions + 1))`), `n_price_move`. Variant flag: `exclude_price_move: bool`.
Also `build_feature_table(...)`, which merges price and news features and adds `gate_long` / `gate_short`.

**Interfaces:** `news_features(articles: list[Article], session: Session, sessions_before: list[Session], as_of: datetime,
exclude_price_move=False) -> DataFrame`; `build_feature_table(price: DataFrame, news: DataFrame) -> DataFrame`
(columns: all features + `gate_long`, `gate_short`, `sector`).

| Choice | Options | Recommendation |
|---|---|---|
| Baseline unit | per-article EW vs per-session daily mean, then EW | **per-session daily mean**: a 10-article day doesn't swamp the baseline |
| Fresh boundary | previous session close vs fixed 16:00 yesterday | **previous session close from the calendar** (handles weekends, holidays, half days) |

- [ ] Tests (`tests/test_news_features.py`, synthetic articles): one strongly positive fresh article on a zero baseline →
  surprise = 1/3 × s; five such → 5/7 × s; neutral-heavy articles barely move `news_today`; an article at
  `as_of + 1 min` is ignored; **(RF2)** on a Monday, a Saturday article counts as fresh, a Friday 15:00 article counts
  as baseline; gates: positive news + gap down → neither gate.
- **Done:** tests pass. Phase A complete: one function call gives the full feature table for any past or live morning.

---

## Phase B: backtest, exploration, freeze (Steps 7–10). Runs locally.

### Step 7: Historical dataset (`backtest/build_dataset.py`)

**What:** for every session from 2026-04-29 to 2026-11-20: rebuild that day's S&P 500 membership, compute the
feature table as of open − 35 min (prices to open − 51 min, news to open − 35 min), and attach **outcomes**:
1-minute bars 09:30 → close for gated stocks and their sector ETFs (entries, ORB, stops, horizons). FinBERT runs over
all articles once (cached). Output: `.cache/dataset.pkl` plus a committed `data/backtest/dataset_summary.json`
(counts only).

| Choice | Options | Recommendation |
|---|---|---|
| Membership | today's list for all dates vs rebuilt from Wikipedia's "changes" table | **rebuilt**: ~10 changes since April, removes survivorship bias cheaply |
| Outcome bars | all 503 stocks vs gated stocks + ETFs only | **gated only** (~10× less data; non-gated stocks never trade) |

- [ ] Tests (`tests/test_dataset.py`, on a 3-day slice): every news `created_at` < its row's as_of; every feature bar
  ≤ as_of − 16 min; outcome bars start ≥ 09:30; the 2026-11-20 rows for AMD/PLTR/META match the live plan's gaps
  within 0.3 percentage points.
- [ ] Full build run in the background; report sessions, gated rows per day, coverage, and FinBERT time.
- **Done:** dataset built, look-ahead tests pass, summary reported.

### Step 8: Trade simulator (`backtest/simulate.py`)

**What:** pure simulation of one trade from 1-minute bars: entry (`open` = first 09:30 bar open; `orb` = enter at the
first close beyond the 09:30–09:35 range in the trade's direction, otherwise no trade, with an optional
first-5-minute rvol filter), stop at k × ATR (fill at the stop price, or at the bar open if the bar gapped
through it, the conservative choice), horizons measured as in the evaluator, exit at the close,
signed by side, gross and net of 5 bps per side, sizing `R / ATR%` with the caps.

**Interfaces:** `simulate_trade(row: Series, bars: DataFrame, session: Session, entry_mode: str, stop_k: float | None,
equity: float) -> dict(entered: bool, entry, exit, exit_reason, qty, ret_gross: dict[horizon], ret_net: dict[horizon], pnl_net: dict[horizon])`.

| Choice | Options | Recommendation |
|---|---|---|
| Stop fill when a bar spans the stop | stop price vs bar open vs bar low/high | **stop price, or the bar open if it opened beyond** (realistic, not optimistic) |
| Same-bar ambiguity (stop and horizon in one bar) | assume the stop first vs the horizon first | **stop first** (conservative) |

- [ ] Tests (`tests/test_simulate.py`, synthetic bars): a long +1% at the close → net = 1% − 0.10%; a short whose price
  falls 2% → +2% gross; a stop hit at 10:05 → horizons after 10:05 equal the stop return; a gap through the stop → fill
  at the bar open; ORB with no breakout → `entered = False`; **(RF1)** a half day: `5_hour` is clamped to the 13:00 close;
  sizing: ATR 2%, equity 100k → 3% notional; ATR 1% → capped at 4%.
- **Done:** tests pass.

### Step 9: Model and exploration (`src/model.py`, `backtest/explore.py`)

**What:** ridge regression (numpy closed form; standardisation fitted on training folds only) predicting the
market-relative return from entry to close on gated rows. **Walk-forward** by month over the exploration window
(after 20 warm-up sessions, up to 2026-10-08). Selection in two stages, to limit snooping:
(1) model level, chosen by out-of-fold IC: λ ∈ {0.1, 1, 10, 100}, `exclude_price_move` ∈ {no, yes};
(2) trading rules, chosen by out-of-fold mean net return per trade at the close: entry ∈ {open, orb},
stop k ∈ {0.75, 1, 1.5, none}, threshold ∈ {1.5, 2.0, 2.5} × the std of the training-fold predictions (absolute,
history-based, so zero-trade days are possible). The equal-weight z-score composite is reported as a benchmark.
Output: `data/backtest/exploration.json` + `docs/backtest/exploration-report.md`: per side × horizon, the mean net
return per trade with t-stat, hit rate, Sharpe and max drawdown (gross and net); IC by month; benchmarks SPY
open-to-close and SPY buy-and-hold; feature weights; share of trades at the 4% cap; number of variants tried.

**Interfaces:** `RidgeModel(lam).fit(X: ndarray, y: ndarray) -> self`, `.predict(X) -> ndarray`, `.to_dict()/from_dict()`;
`walk_forward(frame, features, target, lam, folds) -> DataFrame(oof predictions)`; `information_coefficient(pred, realised, dates) -> (mean_ic, t_stat)`
(daily Spearman, then t over days); `FEATURES` list (frozen order).

| Choice | Options | Recommendation |
|---|---|---|
| Ridge implementation | numpy closed form vs scikit-learn | **numpy**: 10 lines, no new dependency, easy to explain |
| CV scheme | random K-fold vs walk-forward | **walk-forward**: random folds leak the future in time series |
| Selection order | one big grid vs two stages | **two stages**: 8 + 24 configurations instead of 192 |

- [ ] Tests (`tests/test_model.py`): ridge matches `numpy.linalg.lstsq` on the augmented system; λ → 0 matches OLS;
  walk-forward never trains on a date ≥ the test fold start; standardisation uses training rows only (assert on means);
  IC of perfect predictions = 1, of shuffled ≈ 0.
- [ ] Run exploration; report to the user. **Decision point:** the user reviews the report and approves the frozen choices.
  Pre-agreed rule: if no variant shows an edge, the best candidate is frozen anyway (no extra exploration rounds).
- **Done:** report delivered; the user approves (or asks for changes to) the choice.

### Step 10: Freeze + one-time hold-out (`backtest/freeze.py`, `backtest/holdout.py`)

**What:** `freeze.py` writes `data/model/frozen_model.json` (features, weights, scaler, λ, thresholds, entry mode,
stop k, FinBERT revision, data cut-off) and `data/preregistration.json` (frozen date, success criteria, confirmation
length, cost assumption, hold-out dates, sha256 of the model file). **The user commits both (pre-registration).**
`holdout.py` then evaluates the hold-out once: it refuses to run unless both files are committed and unchanged in git,
and refuses a second run if `data/backtest/holdout.json` exists.

| Choice | Options | Recommendation |
|---|---|---|
| Pre-registration proof | a git commit vs an external timestamp (e.g. a GitHub release) | **git commit on the public repo** (the push date is visible); optionally tag it `prereg-v2` |
| Hold-out guard | honour system vs enforced in code | **enforced** (a portfolio reviewer can read the guard) |

- [ ] Tests (`tests/test_freeze.py`, temp git repo): hold-out refuses when the files are uncommitted, refuses when the
  model hash differs, runs once, then refuses a second run.
- [ ] Ask the user to commit the freeze; run the hold-out; report results honestly whatever they are.
- **Done:** frozen files committed by the user; `holdout.json` + report written.

---

## Phase C: live system (Steps 11–15). Steps 12–13 can be built while FinBERT runs in Step 7.

### Step 11: Live strategy + morning job (`src/strategy.py`, `src/main.py`, `morning.yml`)

**What:** `strategy.py`: filters → gates → frozen-model score → threshold → rank by |score| → sector cap 3 → max 10 →
shortable/easy-to-borrow check (Alpaca `get_asset`, shorts only) → sizing (`R/ATR%`, caps, whole shares, equity from
the account). `main.py`: keep the calendar checks, catch-up evaluation and safety net; then **wait until open − 35 min**,
build the feature table with the Phase A code, select trades, write plan v2 (`strategy: "v2"`, `model_sha`, thresholds,
targets with side, score, features, ref price, ATR, qty, stop price), write `data/scores/<date>.csv`, place orders.
If the frozen entry mode is `orb`: stay alive, read the 09:30–09:35 bars at 09:36 (live SIP minute data is 15 min
delayed, so the breakout check uses Yahoo 1-minute data with Alpaca's latest trade as a cross-check), then submit.
A `dry_run` input/env does everything except orders, writes to `data/dryrun/`, and commits nothing.

| Choice | Options | Recommendation |
|---|---|---|
| Waiting for 08:55 | sleep inside the job vs change the Worker crons | **sleep in the job** (free, DST-safe from the calendar, no Worker redeploy); `timeout-minutes: 150` |
| Old vs new code | keep v1 modules side by side vs replace | **replace** on the branch (`main` keeps v1 until the merge); v1 plans and results stay readable |
| ORB live data | Yahoo 1-min vs Alpaca IEX real-time | **Yahoo + Alpaca latest-trade check** (SIP is delayed; IEX is too thin pre-open but fine after 09:30) |

- [ ] Tests (`tests/test_strategy.py`, `tests/test_morning.py`, fake broker + fake feature table): a hard-to-borrow
  short is dropped; the 4th stock of a sector is dropped; the 11th trade is dropped; a zero-signal day → plan with 0
  targets and no orders; sizing caps; the wait is computed from the calendar (a half day included); dry run places no
  orders and writes no `data/plans` file; existing calendar/cutoff/catch-up behaviour unchanged.
- [ ] Real read-only replay: run the morning pipeline in dry-run "as of" 2026-11-20 08:55 → plan + scores CSV reported, with the run time.
- **Done:** tests pass; replay output and timing reported.

### Step 12: Side-aware orders, OTO stops, exits (`src/broker.py`, `src/exits.py`)

**What:** entries as market DAY orders with `order_class=OTO` and a stop-loss leg (long: sell stop at ref − k·ATR;
short: buy stop at ref + k·ATR; rounded to cents; no stop leg if k = none). The exit job per target: settle the entry
remainder → cancel the stop leg and poll until final → re-read the position → if the stop filled, record
`exit_reason="stop"` and stop; else sell (long) or buy-to-cover (short) `min(filled, held on that side)` → confirm fills.
The safety net cancels a leftover stock's open orders first, then closes it (BUY for shorts), with `-late-exit` tags.
Client ids: `<date>-<T>` entry, `<date>-<T>-exit`, `<date>-<T>-late-exit` (unchanged).

| Choice | Options | Recommendation |
|---|---|---|
| Finding the stop leg | id stored from the submit response vs re-query the parent with `nested=True` | **store it, fall back to the nested query** (legs can be missing from the first response) |
| Cancelling | cancel the leg by id vs `close_position` (cancels orders implicitly) | **cancel by id + poll**: explicit and testable; close_position hides what happened |

- [ ] Tests (`tests/test_broker_v2.py`, fake broker): a long and a short entry produce the right sides and stop prices;
  stop untouched → cancelled → exit sent; stop already filled → no exit order; **(RF5)** the stop fills between the
  cancel request and the poll → the position re-read shows 0 → no exit (no flip); a partial entry fill → exit only the
  filled qty; a leftover short → orders cancelled, then BUY with a `-late-exit` id; a re-run is idempotent; importing
  `exits` loads no yfinance/torch/pandas (checked in a subprocess via `sys.modules`).
- [ ] **With the user's OK only:** submit one 1-share OTO order on the paper account (weekend or pre-market), confirm
  it is accepted with its leg, cancel immediately, verify no position or order is left.
- **Done:** tests pass; the paper OTO check reported (or explicitly deferred).

### Step 13: Evaluator v2 (`src/evaluator.py`)

**What:** side-aware P&L (`sign = −1` for shorts) for `performance_pct`/`pnl_usd`; stop-aware horizons (after the
stop fill time every horizon uses the stop price); net-of-cost figures as new fields (`performance_pct_net`,
`pnl_usd_net`); new trade fields `side`, `score`, `features`, `stop_price`, `exit_reason`; entry fields `strategy`,
`by_side` (trades, invested, pnl and return per horizon), `score_thresholds`, `costs_bps`. After each session: the
realised market-relative return of **every gated candidate** → `data/candidates/<date>.csv` (the would-have-traded
log, which feeds the IC); update `data/confirmation.json` (sessions, trades, running IC and t, mean net return vs the
pre-registered criteria). Late-exit revision works for shorts (BUY late exits). v1 entries and plans are evaluated
exactly as before.

| Choice | Options | Recommendation |
|---|---|---|
| Portfolio curve with costs | gross (as now) vs net | **keep the curve gross** (the existing meaning for the site), net curve as a new field `portfolio_curve_net` |
| Candidate outcomes source | Alpaca SIP bars (after the close, no delay issue) vs Yahoo | **Alpaca SIP** (same source as the backtest) |

- [ ] Tests (`tests/test_evaluator_v2.py`, fake broker + fake bars): re-evaluating the stored 2026-11-20 v1 plan gives
  the same entry as `results.json` today (regression); a short gaining 2% → +pnl; a stop at 10:05 → horizons after
  10:05 equal the stop return; `by_side` sums equal the totals; **(RF1)** half day: the curve is clamped; a short late
  exit is revised with the right sign; JSON is written with `allow_nan=False`.
- **Done:** tests pass; a sample v2 entry is shown to the user before anything else uses it.

### Step 14: Alerts and workflows

**What:** notices for data coverage < 70% and for score drift (weekly: live predicted-score mean/std outside the
hold-out's 95% band). `morning.yml`: `workflow_dispatch` input `dry_run`, `timeout-minutes: 150`, commit step skipped
on a dry run. `.gitignore`: `.cache/`. No new Worker trigger.

- [ ] Tests (`tests/test_alerts.py`): coverage 0.6 → a notice and exit code 0; drift outside the band → a notice; the
  rules that should fail still fail.
- [ ] Workflow syntax check (`actionlint` if available, else a careful review) and a dry-run plan of what each job runs.
- **Done:** tests pass; the workflow diff is shown to the user.

### Step 15: Site data files + README

**What:** document everything for the site session and for portfolio readers: README sections on the strategy,
the research method (explore → freeze → hold-out → confirm), the `results.json` additions (field by field), and the new
files (`data/backtest/exploration.json`, `holdout.json`, `data/preregistration.json`, `data/confirmation.json`,
`data/scores/`, `data/candidates/`). A hand-over note for the site session in the shared notes folder.

- [ ] Check every new JSON file against its README description (a small test that loads each file and checks the
  documented keys exist).
- **Done:** README updated; schema test passes; note written.

---

## Phase D: switch-over (Step 16). Every action here needs the user's OK.

### Step 16: Dry run, PR, merge, reset

1. With OK: one GitHub dry run of `morning.yml` from the branch on a weekday morning. Report real timing, coverage, Yahoo throttling.
2. PR with a clear description (backtest + hold-out numbers, pre-registration commit, schema additions, risks). Report the link. **Do not merge.**
3. On the user's go: after the last v1 session is fully evaluated (no open positions, no `missed` exits), merge to `main`.
4. The user resets the paper account in Alpaca's dashboard (updates keys/secrets if new ones are issued).
5. The first v2 morning runs; `results.json` gets `strategy: "v2"` entries.

**Open decision (pending your answer):** after the reset, (a) v2 starts a new segment at $100,000 (recommended) or
(b) continues v1's balance. Assumed (a) until you say otherwise; it only affects Steps 13 and 15.

- **Done:** the PR is open and reviewed; the merge and reset happen only on your instruction.
