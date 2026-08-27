# Cross-signal bullish-cross age-decay experiment

- Scope: approved 2019-2021 training replay; 2018 is warm-up only.
- Hypothesis: an age 2 bullish cross is less timely than age 0/1.
- Frozen change: keep age 0/1 at full official weight and multiply only contributing age 2 bullish RSI12/RSI24/MACD/KDJ-K/KDJ-J weights by 0.5.
- Sell rules and every other strategy rule remain unchanged.

## Performance

| Arm | Total return | Annualized | Max drawdown | Sharpe | Sortino | Win rate | P/L ratio | Buys | Sells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 125.00% | 31.13% | 6.03% | 2.262 | 3.581 | 56.18% | 4.878 | 92 | 89 |
| Candidate | 87.35% | 23.35% | 8.79% | 1.832 | 2.801 | 57.47% | 3.405 | 90 | 87 |

Annual returns:

- Baseline: 2019: 35.84%, 2020: 52.68%, 2021: 8.49%
- Candidate: 2019: 33.26%, 2020: 32.07%, 2021: 6.46%

## Filled-order path

- Changed filled-order days: 64
- By year: 2019: 15, 2020: 27, 2021: 22

## Frozen gate

- Decision: REJECT
- Failure: candidate total return does not improve
- Failure: candidate annualized return does not improve
- Failure: candidate maximum drawdown worsens
- Failure: candidate Sharpe ratio worsens
- Failure: candidate Sortino ratio worsens
- Failure: candidate profit/loss ratio worsens
- Failure: 2019 candidate annual return worsens
- Failure: 2020 candidate annual return worsens
- Failure: 2021 candidate annual return worsens

## Interpretation and next action

The local gate failed. Reject this candidate, record the failed experiment, and do not generate a JoinQuant candidate or tune a replacement rule.
