# Long/short news-surprise strategy: agreed design

Status: **summary of the interview, awaiting your confirmation** (2026-11-21). Not committed.
Branch `long-short-strategy`. Nothing reaches `main` and the Cloudflare Worker is not changed
without your explicit OK. "You" = decided by you in the interview; "proposal" = my default, open to correction.

## 1. Goal and how success is judged

- **Goal:** a portfolio showcase that demonstrates a *real, honestly tested* edge, running reliably.
- **Success** = (1) the frozen strategy passes its pre-registered test (§12) and (2) the bot runs with no manual fixes.
- **Method (you):** explore → **freeze** (a dated git commit = public pre-registration) → evaluate the untouched
  **hold-out** once → **live confirmation** with the model completely frozen.
- **What is claimed:** only "side × entry mode, exit at the close" (2 × 2 = 4 candidates). Other horizons are
  measured and shown as exploratory, with no claim.

## 2. Universe and filters (you)

- S&P 500 from the Wikipedia constituents table, refreshed weekly, copy cached in the repo (GitHub "datasets" CSV as
  fallback). Sector (GICS) taken from the same table.
- Filters: 20-day average dollar volume ≥ $20M; price ≥ $5; pre-market dollar volume ≥ $250k by scan time
  (data quality: is the gap real?); no ATR filter. Relative pre-market volume is a model feature, not a filter.

## 3. Data sources (you)

| Data | Source | Notes |
|---|---|---|
| Prices, volumes (live + backtest) | **Alpaca SIP, free plan = 15 min delayed** | 425/503 stocks priced by 08:55 (data to 08:40), ~1 s; minute bars back to 2019 |
| Price fallback | Yahoo (`yfinance`) | used only if Alpaca fails or misses a stock |
| News | **Alpaca only** (all fiscal.ai) | history starts **2026-04-29**; ~440 articles/day; always 1 symbol/article |
| Sentiment | FinBERT `ProsusAI/finbert` | headline + summary + full text (HTML stripped), scored in memory, **text never stored** |
| Borrowability | Alpaca assets | `shortable` and `easy_to_borrow` |

## 4. News features (you)

- Per article: `s = P(positive) − P(negative)`, weight `w = 1 − P(neutral)`. Dedupe by id and normalised headline;
  drop articles tagging > 3 symbols; tag "price-move" headlines ("shares rise/fall/decline…"), and the backtest
  decides whether to exclude them (they may echo past returns).
- **Fresh** = published after the previous session's 16:00 close, up to scan time.
- `news_today` = w-weighted mean of fresh `s`. `baseline` = exponentially weighted mean over the last 20 sessions,
  half-life 5 sessions (0 if no articles).
- `surprise` = shrunk today − baseline: `shrunk = baseline + n/(n+k) · (news_today − baseline)`, k ≈ 2.
- `news_volume` = `log((fresh count + 1) / (normal daily count + 1))`.

## 5. Price and market features (you)

- `gap_atr` = (stock gap − β × sector-ETF gap) ÷ ATR(14)%, β from 1 year of daily returns vs the stock's SPDR
  sector ETF (11 ETFs). Gap = latest pre-market price vs previous close.
- `rvol` = pre-market volume today ÷ normal pre-market volume for the same clock window.
- `trend` = price vs 50-day MA; `dist_52w_high`. RSI and moving-average crossovers are excluded.

## 6. Model and score (you)

- **Gates:** a long needs a **sector-adjusted** gap (`excess_gap_pct`) > 0 and `news_today` > 0; a short needs it < 0
  and `news_today` < 0; no fresh news = no trade. (Changed from the raw gap after the 2026-11-20 check, you.)
- **Strategy model:** ridge regression on gated stocks predicting the **market-relative return** (stock − β × sector ETF)
  from entry to close; features standardised using training data only; penalty chosen by walk-forward CV.
- **Benchmark only** (proposal): an equal-weight z-score composite, computed in the backtest so we can see whether
  the fitted model adds anything. It never trades.
- **Thresholds:** absolute, picked from a coarse grid during exploration, symmetric unless the data says otherwise,
  then frozen. Zero-trade days are expected.

## 7. Selection (you)

At most 10 trades a day, strongest |predicted return| first, any long/short mix, **at most 3 per GICS sector**,
shorts only if shortable and easy to borrow. One side per stock per day; skip a stock whose leftover is being
closed that morning (wash-trade protection). Intraday only.

## 8. Timing (you)

- **Scan** at about open − 35 min (08:55 ET, from Alpaca's calendar) inside the morning job.
- **Entry mode chosen by the backtest:** (a) market order queued pre-market and filled at the open, or (b) a 09:35
  opening-range breakout filtered by first-5-minute relative volume. If (b) wins, the morning job stays alive until
  ~09:36 (free on a public repo), so no new Cloudflare trigger is needed.
- **Exit:** always just before the close (the existing exit job, market orders, fill confirmation).
- **Measured horizons:** 1 min, 10 min, 30 min, 1 h, 5 h, close, all from 1-minute bars, as today.

## 9. Risk (you)

- **Stop-loss** = k × ATR from the entry reference, k picked in exploration from {0.75, 1, 1.5, none}; attached to the
  entry as an Alpaca **OTO** order (live as soon as the entry fills).
- The exit job **cancels the stop and confirms the cancel** before closing; quantity = min(filled, held on that side),
  so a position can never flip. The safety net cancels open orders for a leftover stock before closing it.

## 10. Sizing (you: equal risk; numbers are a proposal)

- `size = R / ATR%`, where R = the loss if the stock moves 1 ATR against us. Proposal: **R = 0.06% of equity**
  (≈ $60 on $100k). Caps: ≤ 4% of equity per stock, ≤ 20% per day; whole shares; skip if one share exceeds the size.
- Why 0.06% and not 0.25%: with a 4% cap, any R ≥ 0.25% makes the cap bind for almost every stock (ATR ≤ 6%), so
  "equal risk" would quietly become "equal dollars". At 0.06%, a stock with a 2% ATR gets 3% of equity and one
  with a 1.2% ATR hits the 4% cap.
- R is defined per ATR, so sizing does not depend on the stop multiple k (including "no stop").
- Equity = Alpaca account equity at plan time (compounding).

## 11. Costs (you)

5 bps per side (entry and exit) deducted in every reported "net" figure; gross is shown as well. Borrow fee: 0 for
intraday shorts (Alpaca charges borrow on positions held overnight); stated as an assumption.

## 12. Evaluation and confirmation (you)

- **Per side × horizon:** mean net return per trade (t-stat), hit rate, Sharpe, maximum drawdown; the
  **information coefficient** (rank correlation of predicted vs realised market-relative return over all gated
  stocks, traded or not); benchmarks SPY open-to-close and SPY buy-and-hold.
- **Split:** history 2026-04-29 → about 2026-10-08 = exploration (the first 20 sessions are baseline warm-up);
  about 2026-10-09 → 2026-11-20 (~6 weeks) = **hold-out**, evaluated once, after the freeze.
- **Confirmation length:** whichever comes later, 60 sessions or 150 trades.
- **Success:** primary = IC > 0 with t > 2 over the confirmation period; secondary = the frozen strategy's mean net
  return per trade > 0. Beating SPY is reported but not required.
- Model frozen during confirmation (no retraining).
- **If exploration shows no edge (you, decided before seeing data):** freeze the best candidate anyway and run the
  hold-out and live confirmation as planned; the pre-registered criteria decide. No extra exploration rounds.

## 13. Logging and alerts (you)

- `data/scores/<date>.csv`: every scanned stock's features, gates and score (no text).
- After the close: the realised outcome of every gated candidate (traded or not) = the "would-have-traded" log,
  feeding the IC. Rejected stocks keep a reason.
- Alert policy unchanged (fail only when you must act). New **notices**: data coverage < 70%; live score
  distribution drifting from the backtest.

## 14. Portfolio site (you: option b)

Data files produced here; site changes are made later in the site repo/session.
- `results.json`: **additive only** (`side`, `score`, `features`, `stop_price`, `exit_reason`, `by_side`, cost-adjusted
  figures, `strategy` version, …); every existing field keeps its meaning.
- New: backtest summary per variant, a pre-registration file (freeze date, commit, criteria), live confirmation
  progress vs criteria.

## 15. Constraints

- All free: GitHub Actions (public repo), Cloudflare (still 4 of 5 triggers), Alpaca free data, FinBERT on CPU.
- Backtest runs **locally**; caches (bars, FinBERT scores) gitignored; code and summary committed.
- Exit job stays light (`alpaca-py` + `python-dotenv` only). Pinned dependencies. Paper only. Tests use a fake broker.

## 16. Meanwhile and switch-over (you: a)

The old strategy keeps trading from `main` until the freeze, then the new one replaces it. The user resets the paper
account at the switch; `results.json` entries get a `strategy` field and v2 starts a new segment at $100,000
(assumed recommendation (a); the user may still choose to continue v1's balance).

## 17. Plan-level calls (mine, accepted by the user)

- If the backtest picks the 09:35 entry, the live breakout check reads Yahoo 1-minute bars (Alpaca SIP is 15 min
  delayed), cross-checked with Alpaca's latest trade: a small source mismatch on that one decision.
- `portfolio_curve` stays gross (its current meaning); a cost-adjusted `portfolio_curve_net` is added.
