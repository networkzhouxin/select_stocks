# Fresh-Unextended Entry Candidate: Local Screen

Status: `BLOCKED_PENDING_JOINQUANT`

This is a training-only engineering screen. Local execution data is not the
performance authority and these figures cannot approve or reject the candidate.

## Frozen candidate

- Keep the official `buy_score >= 60` primary queue first and unchanged.
- Fill only slots left by the primary queue.
- Candidate buy score: 50-59 inclusive.
- Minimum reversal score: 35.
- Earliest contributing bullish cross age: 0 or 1 only.
- T-1 close extension from that cross close: at most 1.0 ATR14.
- Keep RSI overheat, position, blocked-combination, sell-score, ATR cooldown,
  sizing, sell, and risk rules unchanged.
- Data scope: approved read-only 2018 warm-up plus 2019-2021 training only.
- Candidate variants: exactly one; no neighboring thresholds or ETF exceptions.

## Local screen result

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Total return | +125.00% | +98.11% |
| Maximum drawdown | 6.03% | 6.03% |
| Closed-trade win rate | 56.18% | 49.48% |
| Profit/loss ratio | 4.878 | 3.197 |
| Double-friction return | +108.15% | +81.01% |
| Double-friction maximum drawdown | 6.39% | 7.78% |

- Filled fresh-entry buys: 19.
- Coverage years: 2019, 2020, and 2021.
- The candidate is not a sparse/no-op implementation.
- Directional warning: local accuracy and friction sensitivity worsened.

## Decision boundary

Proceed to exactly one official JoinQuant 2019-2021 run because the local
minute data cannot authoritatively simulate the changed order path. Do not use
this local result to alter 50/35/age-1/1-ATR, add a hard anti-chase veto, remove
an ETF, or change a sell rule. The formal `cross-v0.3.3` strategy remains
unchanged.

The official nominal run must satisfy all of these predeclared gates against
the official `cross-v0.3.3` baseline: win rate improves by at least 3 percentage
points; positive-to-negative round trips fall by at least 3; maximum drawdown
does not worsen; total return retains at least 95%; profit/loss ratio remains at
least 3.0; and each of 2019/2020/2021 remains positive. Only if nominal gates
pass will the identical rule be checked under doubled friction; no signal
threshold may change between the two runs.
