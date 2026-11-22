# Hold-out report (Step 10, evaluated once)

Sessions 2026-10-09 → 2026-11-20 (31), read for the first time after the pre-registration commit. 1557 gated rows (1548 with a measurable outcome).

Pass rule (pre-registered): IC > 0 with t > 2.24 AND mean net return per trade at the close > 0.

| Hypothesis | IC | t | trades | trades/day | mean net / trade | t | hit rate | mean gross | Sharpe (net) | max DD | total | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0495 | 1.24 | 31 | 1.00 | 0.153% | 0.36 | 48.4% | 0.253% | -1.21 | 0.34% | -0.09% | no |
| fade | 0.0304 | 0.75 | 305 | 9.84 | 0.166% | 0.91 | 51.8% | 0.266% | 2.19 | 0.37% | 0.65% | no |

**Goes live (pre-registered decision rule): A.**

## A by side and horizon (only the close is claimed)

| Side | Horizon | trades | mean net | t | hit rate |
|---|---|---|---|---|---|
| long | 1_min | 6 | 0.346% | 0.70 | 66.7% |
| long | 10_min | 6 | 1.225% | 0.87 | 50.0% |
| long | 30_min | 6 | 0.555% | 1.02 | 66.7% |
| long | 1_hour | 6 | 0.842% | 0.85 | 50.0% |
| long | 5_hour | 6 | 0.422% | 0.37 | 33.3% |
| long | EOD | 6 | 0.039% | 0.04 | 50.0% |
| short | 1_min | 25 | -0.042% | -0.24 | 36.0% |
| short | 10_min | 25 | -0.169% | -0.52 | 36.0% |
| short | 30_min | 25 | -0.456% | -1.16 | 36.0% |
| short | 1_hour | 25 | -0.151% | -0.45 | 44.0% |
| short | 5_hour | 25 | 0.316% | 0.73 | 56.0% |
| short | EOD | 25 | 0.181% | 0.38 | 48.0% |

## fade by side and horizon (only the close is claimed)

| Side | Horizon | trades | mean net | t | hit rate |
|---|---|---|---|---|---|
| long | 1_min | 115 | 0.151% | 1.61 | 50.4% |
| long | 10_min | 115 | 0.373% | 2.44 | 62.6% |
| long | 30_min | 115 | 0.605% | 3.01 | 65.2% |
| long | 1_hour | 115 | 0.560% | 2.34 | 57.4% |
| long | 5_hour | 115 | -0.051% | -0.16 | 48.7% |
| long | EOD | 115 | 0.134% | 0.40 | 51.3% |
| short | 1_min | 190 | -0.050% | -0.71 | 47.9% |
| short | 10_min | 190 | -0.172% | -1.34 | 46.8% |
| short | 30_min | 190 | -0.121% | -0.84 | 49.5% |
| short | 1_hour | 190 | -0.021% | -0.14 | 51.1% |
| short | 5_hour | 190 | 0.142% | 0.71 | 52.6% |
| short | EOD | 190 | 0.185% | 0.88 | 52.1% |

## Benchmarks (same sessions)

- SPY open-to-close: mean -0.386% per day, total -11.51%.
- SPY buy-and-hold: -10.87%.

Per-trade means weight every trade equally; the Sharpe, drawdown and total use the dollar P&L of equal-risk sizing (larger positions in calmer stocks), so the two can differ in sign.
