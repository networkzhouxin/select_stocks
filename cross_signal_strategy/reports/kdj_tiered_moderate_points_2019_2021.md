# KDJ tiered moderate-points candidate: 2019-2021 local A/B

## Hypothesis and frozen rule

This candidate tested the previously proposed moderate asymmetric point set
after the stronger 50%/25% threshold-weight candidate failed.

- Current causal T-1 K<=20: unified buy score +20.
- Current causal T-1 20<K<=30: unified buy score +10.
- Current causal T-1 70<=K<80: unified sell score +5.
- Current causal T-1 K>=80: unified sell score +10.
- No state retention and no KDJ cross requirement.
- Current `downside_continuation` blocks the buy bonus.
- Formal buy/sell thresholds remain 60/30.
- A sell at or above 30 still requires the official price confirmation and
  must not be protected by the official ADX uptrend guard.
- The official five-session minimum hold, ATR exits, pool, ranking, position
  sizing, fees, and every other execution rule remain unchanged.

Only this point set was tested in this experiment. No third point set, nearby
point search, or parameter grid was run.

## Data and causality

- Read-only daily warm-up: `cross_signal_warmup_2018`.
- Read-only training data: `cross_signal_train_2019_2021`.
- Signal calculation uses only cached official T-1 snapshots.
- Execution remains 09:35 in the local replay.
- No reserved validation, recent/live outcome, or full-period data was read.
- Historical point-in-time IOPV is unavailable and was not simulated.
- JoinQuant remains the authority for performance; this local replay is a
  training-only structural screen.

## Results

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total return | 125.00% | 119.48% |
| Annualized return | 31.13% | 30.05% |
| Maximum drawdown | 6.03% | 6.12% |
| Sharpe | 2.262 | 2.180 |
| Sortino | 3.581 | 3.439 |
| Win rate | 56.18% | 54.35% |
| Profit/loss ratio | 4.878 | 4.366 |
| Buys / sells | 92 / 89 | 95 / 92 |

Annual returns:

| Year | Baseline | Candidate |
|---|---:|---:|
| 2019 | 35.84% | 33.34% |
| 2020 | 52.68% | 52.06% |
| 2021 | 8.49% | 8.25% |

Changed filled-order days: 15 (2019: 10, 2020: 3, 2021: 2).

Double-friction results:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total return | 108.15% | 101.22% |
| Annualized return | 27.77% | 26.33% |
| Maximum drawdown | 6.39% | 6.30% |
| Sharpe | 2.039 | 1.940 |
| Sortino | 3.186 | 3.016 |
| Win rate | 51.69% | 48.91% |
| Profit/loss ratio | 3.966 | 3.527 |
| Buys / sells | 92 / 89 | 95 / 92 |

## Interpretation and decision

Rejected. The point set was materially less destructive than the 50%/25%
candidate, but it still lost 5.52 percentage points of total return, reduced
win rate by 1.83 percentage points, reduced the profit/loss ratio by about
10.5%, and underperformed the baseline in every training year. Double
friction widened the return shortfall to 6.93 percentage points.

The experiment gate rejected the candidate because win rate did not improve
and the profit/loss ratio worsened by more than 5%. The result also carries
higher multiple-testing risk because it followed inspection of the stronger
candidate's failure. It must not be treated as if it were the only tested
point set.

No JoinQuant or PTrade candidate was generated. Both formal strategy files
remain unchanged. The isolated candidate and this failed result are retained
as research evidence only, and no further KDJ point search is authorized by
this experiment.
