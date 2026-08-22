# KDJ tiered three-session state candidate: 2019-2021 local A/B

## Frozen rule

- T-1 K<=20: unified buy score +10.
- T-1 20<K<=30: unified buy score +5.
- T-1 70<=K<80: unified sell score +5.
- T-1 K>=80: unified sell score +10.
- Keep the state for exactly three decision sessions including the event session.
- Same-direction points take the maximum tier and never accumulate.
- The most recent direction wins when opposite states occur in the three-session window.
- Current `downside_continuation` blocks a retained buy bonus.
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
| Total return | 125.00% | 118.33% |
| Annualized return | 31.13% | 29.82% |
| Maximum drawdown | 6.03% | 6.06% |
| Sharpe | 2.262 | 2.142 |
| Sortino | 3.581 | 3.324 |
| Win rate | 56.18% | 55.43% |
| Profit/loss ratio | 4.878 | 4.384 |
| Buys / sells | 92 / 89 | 95 / 92 |

Annual returns:

- Baseline: 2019 +35.8390%, 2020 +52.6785%, 2021 +8.4888%.
- Candidate: 2019 +32.6140%, 2020 +51.8908%, 2021 +8.3891%.

Changed filled-order days: 22 (`2019=20`, `2020=2`, `2021=0`).

Double-friction results:

- Baseline: return +108.15%, drawdown 6.39%, win rate 51.69%, profit/loss ratio 3.966.
- Candidate: return +100.56%, drawdown 6.41%, win rate 48.91%, profit/loss ratio 3.560.

## Frozen-gate decision

Rejected. It failed because:

- 2021 had no changed filled-order day.
- Win rate did not improve.
- Total return retained less than 95% of baseline.
- Sharpe, Sortino, and profit/loss ratio each worsened by more than 5%.

The local candidate is retained only as failed research evidence. No JoinQuant
or PTrade candidate was generated, and both formal strategies remain unchanged.
