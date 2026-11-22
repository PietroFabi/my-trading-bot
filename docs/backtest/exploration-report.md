# Exploration report (Step 9)

Exploration window 2026-05-29 → 2026-10-08 (92 sessions). Out-of-fold predictions for 2026-07-01 onward (70 sessions), walk-forward by month, expanding window: each month is predicted by a model trained only on earlier months. The hold-out (from 2026-10-09) was not read.

Variants tried: 8 model settings + 24 trading rules = 32. Every one of them is listed below, so the best can be judged against how many were tried.

## Stage 1: model level (chosen by out-of-fold information coefficient)

| Price-move headlines | λ | IC (mean daily Spearman) | t | rows |
|---|---|---|---|---|
| included | 0.1 | -0.0205 | -0.86 | 3062 |
| included | 1.0 | -0.0203 | -0.85 | 3062 |
| included | 10.0 | -0.0215 | -0.90 | 3062 |
| included | 100.0 | -0.0298 | -1.25 | 3062 |
| included | composite benchmark | -0.0308 | -1.29 | 3062 |
| excluded | 0.1 | -0.0246 | -1.04 | 3046 |
| excluded | 1.0 | -0.0241 | -1.02 | 3046 |
| excluded | 10.0 | -0.0248 | -1.05 | 3046 |
| excluded | 100.0 | -0.0345 | -1.45 | 3046 |
| excluded | composite benchmark | -0.0270 | -1.12 | 3046 |

**Chosen:** λ = 1.0, price-move headlines included (IC -0.0203, t -0.85).

### IC by month (chosen model vs composite benchmark)

| Month | model IC | t | composite IC | t |
|---|---|---|---|---|
| 2026-07 | -0.1106 | -2.37 | -0.0607 | -1.51 |
| 2026-08 | 0.0228 | 0.50 | 0.0165 | 0.34 |
| 2026-09 | 0.0020 | 0.06 | -0.0123 | -0.29 |
| 2026-10 | 0.0815 | 1.22 | -0.1515 | -2.44 |

## Stage 2: trading rules (chosen by out-of-fold mean net return per trade at the close)

Rules with fewer than 30 out-of-fold trades cannot be chosen.

| Entry | Stop (× ATR) | Threshold (× pred. std) | trades | mean net / trade | t | hit rate | zero-trade days |
|---|---|---|---|---|---|---|---|
| open | 0.75 | 1.5 | 91 | -0.022% | -0.07 | 44.0% | 20 |
| open | 0.75 | 2.0 | 29 | 0.519% | 0.90 | 51.7% | 49 |
| open | 0.75 | 2.5 | 12 | 0.925% | 0.89 | 50.0% | 59 |
| open | 1.0 | 1.5 | 91 | -0.092% | -0.30 | 44.0% | 20 |
| open | 1.0 | 2.0 | 29 | 0.392% | 0.62 | 51.7% | 49 |
| open | 1.0 | 2.5 | 12 | 0.925% | 0.89 | 50.0% | 59 |
| open | 1.5 | 1.5 | 91 | -0.176% | -0.52 | 44.0% | 20 |
| open | 1.5 | 2.0 | 29 | 0.231% | 0.31 | 51.7% | 49 |
| open | 1.5 | 2.5 | 12 | 0.925% | 0.89 | 50.0% | 59 |
| open | none | 1.5 | 91 | -0.146% | -0.44 | 44.0% | 20 |
| open | none | 2.0 | 29 | 0.231% | 0.31 | 51.7% | 49 |
| open | none | 2.5 | 12 | 0.925% | 0.89 | 50.0% | 59 |
| orb | 0.75 | 1.5 | 5 | 1.350% | 1.37 | 60.0% | 65 |
| orb | 0.75 | 2.0 | 1 | 2.439% | n/a | 100.0% | 69 |
| orb | 0.75 | 2.5 | 0 | n/a | n/a | n/a | 70 |
| orb | 1.0 | 1.5 | 5 | 1.350% | 1.37 | 60.0% | 65 |
| orb | 1.0 | 2.0 | 1 | 2.439% | n/a | 100.0% | 69 |
| orb | 1.0 | 2.5 | 0 | n/a | n/a | n/a | 70 |
| orb | 1.5 | 1.5 | 5 | 1.350% | 1.37 | 60.0% | 65 |
| orb | 1.5 | 2.0 | 1 | 2.439% | n/a | 100.0% | 69 |
| orb | 1.5 | 2.5 | 0 | n/a | n/a | n/a | 70 |
| orb | none | 1.5 | 5 | 1.350% | 1.37 | 60.0% | 65 |
| orb | none | 2.0 | 1 | 2.439% | n/a | 100.0% | 69 |
| orb | none | 2.5 | 0 | n/a | n/a | n/a | 70 |

**Chosen:** entry open, stop 0.75, threshold 1.5 × the training predictions' std (0.306% predicted market-relative return for the frozen model). 1.30 trades per day; 0.0% of trades at the 4% cap.

## The four claimed candidates (side × entry mode, exit at the close; chosen stop and threshold)

| Candidate | trades | mean net / trade | t | hit rate | Sharpe | max drawdown | total return |
|---|---|---|---|---|---|---|---|
| long_open | 24 | -0.798% | -1.09 | 29.2% | -3.15 | 0.31% | -0.29% |
| short_open | 67 | 0.256% | 0.87 | 49.3% | 0.48 | 0.24% | 0.06% |
| long_orb | 1 | -0.210% | n/a | 0.0% | -1.90 | 0.00% | -0.00% |
| short_orb | 4 | 1.740% | 1.49 | 75.0% | 2.16 | 0.02% | 0.06% |

## Chosen rule by side and horizon (other horizons are exploratory: no claim)

### long

| Horizon | trades | mean net | t | hit rate | mean gross | Sharpe net | max DD net | Sharpe gross | max DD gross |
|---|---|---|---|---|---|---|---|---|---|
| 1_min | 24 | -0.055% | -0.25 | 37.5% | 0.045% | -1.59 | 0.04% | -0.20 | 0.03% |
| 10_min | 24 | -0.370% | -0.84 | 29.2% | -0.270% | -2.77 | 0.11% | -2.17 | 0.10% |
| 30_min | 24 | -0.148% | -0.23 | 20.8% | -0.048% | -2.32 | 0.15% | -1.84 | 0.12% |
| 1_hour | 24 | -0.082% | -0.09 | 25.0% | 0.018% | -1.62 | 0.18% | -1.26 | 0.17% |
| 5_hour | 24 | -0.786% | -1.20 | 29.2% | -0.686% | -2.84 | 0.29% | -2.56 | 0.27% |
| EOD | 24 | -0.798% | -1.09 | 29.2% | -0.698% | -3.15 | 0.31% | -2.87 | 0.28% |

### short

| Horizon | trades | mean net | t | hit rate | mean gross | Sharpe net | max DD net | Sharpe gross | max DD gross |
|---|---|---|---|---|---|---|---|---|---|
| 1_min | 67 | -0.189% | -1.97 | 43.3% | -0.089% | -4.50 | 0.18% | -1.93 | 0.09% |
| 10_min | 67 | -0.082% | -0.43 | 44.8% | 0.018% | -2.01 | 0.19% | -0.44 | 0.15% |
| 30_min | 67 | 0.065% | 0.26 | 52.2% | 0.165% | -0.69 | 0.26% | 0.57 | 0.22% |
| 1_hour | 67 | 0.105% | 0.39 | 53.7% | 0.205% | -0.32 | 0.28% | 0.79 | 0.24% |
| 5_hour | 67 | 0.073% | 0.27 | 44.8% | 0.173% | -0.92 | 0.27% | 0.13 | 0.24% |
| EOD | 67 | 0.256% | 0.87 | 49.3% | 0.356% | 0.48 | 0.24% | 1.48 | 0.21% |

### all

| Horizon | trades | mean net | t | hit rate | mean gross | Sharpe net | max DD net | Sharpe gross | max DD gross |
|---|---|---|---|---|---|---|---|---|---|
| 1_min | 91 | -0.154% | -1.67 | 41.8% | -0.054% | -4.59 | 0.20% | -1.75 | 0.11% |
| 10_min | 91 | -0.158% | -0.87 | 40.7% | -0.058% | -2.97 | 0.23% | -1.48 | 0.19% |
| 30_min | 91 | 0.009% | 0.03 | 44.0% | 0.109% | -1.84 | 0.32% | -0.55 | 0.27% |
| 1_hour | 91 | 0.056% | 0.18 | 46.2% | 0.156% | -1.24 | 0.36% | -0.16 | 0.31% |
| 5_hour | 91 | -0.153% | -0.58 | 40.7% | -0.053% | -2.50 | 0.43% | -1.52 | 0.39% |
| EOD | 91 | -0.022% | -0.07 | 44.0% | 0.078% | -1.53 | 0.39% | -0.59 | 0.36% |

## Diagnostics (descriptive; nothing above was chosen with them)

Mean market-relative return (open → close) of the gated stocks, out of fold:

| Gate | rows | mean | t | median |
|---|---|---|---|---|
| gate_long | 2398 | -0.143% | -3.27 | -0.109% |
| gate_short | 664 | 0.162% | 1.75 | -0.088% |

Each feature's own IC against the target (out-of-fold gated rows):

| Feature | IC | t |
|---|---|---|
| gap_atr | -0.0591 | -2.36 |
| log_rvol | 0.0061 | 0.32 |
| trend | -0.0222 | -0.84 |
| dist_52w_high | -0.0311 | -1.29 |
| news_today | -0.0347 | -1.48 |
| baseline | -0.0098 | -0.44 |
| surprise | -0.0146 | -0.67 |
| news_volume | -0.0097 | -0.51 |

## Benchmarks (same out-of-fold sessions)

- SPY open-to-close: mean -0.016% per day, Sharpe -0.54, total -1.15%.
- SPY buy-and-hold: 3.64%.

## Frozen-model weights (fit on every exploration row; standardised features)

| Feature | weight (return per 1 std) |
|---|---|
| gap_atr | -0.2269% |
| log_rvol | 0.0793% |
| trend | 0.1630% |
| dist_52w_high | -0.1195% |
| news_today | -0.2171% |
| baseline | 0.1010% |
| surprise | 0.2621% |
| news_volume | -0.0161% |

Intercept -0.0703%; 4058 training rows; λ = 1.0.

Costs: 5 bps per side in every net figure; borrow fee 0 for intraday shorts (assumption). Shortability uses today's Alpaca easy-to-borrow flags (no history is available).
