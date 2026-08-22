# KDJ tiered current-state candidate: 2019-2021 local A/B

## Frozen rule

- Current causal T-1 K<=20: unified buy score +10.
- Current causal T-1 20<K<=30: unified buy score +5.
- Current causal T-1 70<=K<80: unified sell score +5.
- Current causal T-1 K>=80: unified sell score +10.
- No state retention and no KDJ cross requirement.
- Current `downside_continuation` blocks the buy bonus.
- Formal buy/sell thresholds remain 60/30; all other formal rules remain unchanged.

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
| Total return | 125.00% | 125.00% |
| Annualized return | 31.13% | 31.13% |
| Maximum drawdown | 6.03% | 6.03% |
| Sharpe | 2.262 | 2.262 |
| Sortino | 3.581 | 3.581 |
| Win rate | 56.18% | 56.18% |
| Profit/loss ratio | 4.878 | 4.878 |
| Buys / sells | 92 / 89 | 92 / 89 |

Annual returns were identical at 2019 +35.8390%, 2020 +52.6785%, and
2021 +8.4888%. Changed filled-order days: 0.

Double-friction results were also identical: return +108.15%, drawdown 6.39%,
win rate 51.69%, and profit/loss ratio 3.966.

## Frozen-gate decision

Rejected. It produced no changed filled-order day in 2019, 2020, or 2021 and
did not improve win rate. The local candidate is retained only as failed
research evidence. No JoinQuant or PTrade candidate was generated, and both
formal strategies remain unchanged.
