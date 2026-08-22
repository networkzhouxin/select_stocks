# KDJ tiered 50%/25% threshold candidate: 2019-2021 local A/B

## Hypothesis and frozen rule

The hypothesis was that current T-1 KDJ location could receive enough weight
to improve early entries and protected exits without becoming an independent
trade trigger.

- Current causal T-1 K<=20: unified buy score +30 (50% of buy threshold).
- Current causal T-1 20<K<=30: unified buy score +15 (25% of buy threshold).
- Current causal T-1 70<=K<80: unified sell score +7.5 (25% of sell threshold).
- Current causal T-1 K>=80: unified sell score +15 (50% of sell threshold).
- No state retention and no KDJ cross requirement.
- Current `downside_continuation` blocks the buy bonus.
- Formal buy/sell thresholds remain 60/30.
- A sell at or above 30 still requires the official price confirmation and
  must not be protected by the official ADX uptrend guard.
- The official five-session minimum hold, ATR exits, pool, ranking, position
  sizing, fees, and every other execution rule remain unchanged.

This was the only point set tested in this experiment. No nearby point search
or parameter grid was run.

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
| Total return | 125.00% | 87.81% |
| Annualized return | 31.13% | 23.45% |
| Maximum drawdown | 6.03% | 10.74% |
| Sharpe | 2.262 | 1.709 |
| Sortino | 3.581 | 2.609 |
| Win rate | 56.18% | 48.96% |
| Profit/loss ratio | 4.878 | 2.545 |
| Buys / sells | 92 / 89 | 99 / 96 |

Annual returns:

| Year | Baseline | Candidate |
|---|---:|---:|
| 2019 | 35.84% | 40.30% |
| 2020 | 52.68% | 37.44% |
| 2021 | 8.49% | -2.60% |

Changed filled-order days: 67 (2019: 24, 2020: 22, 2021: 21).

Double-friction results:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total return | 108.15% | 71.56% |
| Annualized return | 27.77% | 19.77% |
| Maximum drawdown | 6.39% | 11.57% |
| Sharpe | 2.039 | 1.467 |
| Sortino | 3.186 | 2.210 |
| Win rate | 51.69% | 43.75% |
| Profit/loss ratio | 3.966 | 2.178 |
| Buys / sells | 92 / 89 | 99 / 96 |

## Interpretation and decision

Rejected. The candidate lost 37.19 percentage points of total return, reduced
win rate by 7.22 percentage points, increased maximum drawdown by 4.71
percentage points, and turned 2021 negative. The lower profit/loss ratio and
worse double-friction result show that the changed path was not compensated by
better trade quality.

The combination is proven unsuitable on the training screen. This experiment
changed both buy and sell bonuses, so the available result does not prove which
side caused how much of the damage. A causal split would require separately
pre-declared buy-only and sell-only ablations; none were run, because that
would expand this frozen one-candidate test into a new search family.

No JoinQuant or PTrade candidate was generated. Both formal strategy files
remain unchanged. The isolated candidate and this failed result are retained
as research evidence only.
