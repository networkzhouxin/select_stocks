# KDJ tiered current-state direct-exit candidate: 2019-2021 local A/B

## Frozen rule

- Current causal T-1 K<=20: unified buy score +10.
- Current causal T-1 20<K<=30: unified buy score +5.
- Current causal T-1 70<=K<80: unified sell score +5.
- Current causal T-1 K>=80: unified sell score +10.
- No state retention and no KDJ cross requirement.
- Current `downside_continuation` blocks the buy bonus.
- After the official five-session minimum hold, a positive extreme sell bonus
  plus final sell score >=30 sells directly without price confirmation or ADX
  protection.
- ATR stops and every other official execution/risk rule remain unchanged.

## Data and causality

- Read-only daily warm-up: `cross_signal_warmup_2018`.
- Read-only training data: `cross_signal_train_2019_2021`.
- Signal calculation uses only cached official T-1 snapshots.
- Execution remains 09:35 in the local replay.
- No reserved validation, recent/live outcome, or full-period data was read.
- Historical point-in-time IOPV is unavailable and was not simulated.

## Results

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total return | 125.00% | 101.84% |
| Annualized return | 31.13% | 26.46% |
| Maximum drawdown | 6.03% | 6.33% |
| Sharpe | 2.262 | 2.261 |
| Sortino | 3.581 | 3.791 |
| Win rate | 56.18% | 58.21% |
| Profit/loss ratio | 4.878 | 2.954 |
| Buys / sells | 92 / 89 | 136 / 134 |

Annual returns:

| Year | Baseline | Candidate |
|---|---:|---:|
| 2019 | 35.84% | 20.50% |
| 2020 | 52.68% | 56.11% |
| 2021 | 8.49% | 7.30% |

Changed filled-order days: 155 (2019: 51, 2020: 44, 2021: 60).

Double-friction results:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total return | 108.15% | 77.35% |
| Maximum drawdown | 6.39% | 7.08% |
| Win rate | 51.69% | 54.48% |
| Profit/loss ratio | 3.966 | 2.359 |

## Frozen-gate decision

Rejected. The candidate improved the closed-trade win rate by 2.03 percentage
points, but retained only 81.5% of baseline total return and reduced the
profit/loss ratio by 39.4%. It also added 44 buys and 45 sells and was much
more sensitive to doubled friction. The result is consistent with direct
KDJ-overbought exits clipping payoff size and creating extra recycling.

No JoinQuant or PTrade candidate was generated. Both formal strategy files
remain unchanged, and reserved validation periods were not inspected.
