# KDJ extreme-zone score candidate: 2019-2021 local A/B

- Date: 2026-08-23
- Scope: approved read-only 2019-2021 training data; 2018 is indicator warm-up only.
- Execution: unchanged 09:35 local execution model using T-1 signals.
- Frozen candidate: when `K <= 20`, add 5 to the unified buy score unless
  `downside_continuation` is active; when `K >= 80`, add 5 to the unified sell
  score. No KDJ cross is required. The formal 60/30 score thresholds and every
  other rule remain unchanged.
- Authority warning: this is a local path screen, not an authoritative
  JoinQuant performance result.

## Portfolio result

| Arm | Total return | Annualized | Max drawdown | Sharpe | Sortino | Win rate | P/L ratio | Buys | Sells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 125.00% | 31.13% | 6.03% | 2.262 | 3.581 | 56.18% | 4.878 | 92 | 89 |
| Candidate | 125.00% | 31.13% | 6.03% | 2.262 | 3.581 | 56.18% | 4.878 | 92 | 89 |
| Baseline, doubled friction | 108.15% | 27.77% | 6.39% | 2.039 | 3.186 | 51.69% | 3.966 | 92 | 89 |
| Candidate, doubled friction | 108.15% | 27.77% | 6.39% | 2.039 | 3.186 | 51.69% | 3.966 | 92 | 89 |

Annual returns were identical in both normal-cost arms:

- 2019: 35.84%
- 2020: 52.68%
- 2021: 8.49%

Changed filled-order days: **0**.

## Binding diagnosis

The full scan covered 6,111 valid code-date score snapshots.

- Oversold buy bonuses: 93.
- Oversold bonuses that moved the buy score from below 60 to at least 60: 0.
- Overbought sell bonuses: 1,382.
- Overbought bonuses that moved the sell score from below 30 to at least 30: 11.
- Sell threshold crossings by ETF: `159915=2`, `159928=3`, `512100=3`,
  `513100=1`, `518880=2`.

All 11 sell-score crossings remained non-actionable in the ordinary sell path:
four lacked official price-structure confirmation, and every event was protected
by the official strong-ADX uptrend rule. Therefore none changed an order.

## IOPV limitation

The approved historical training data has no point-in-time historical IOPV.
Consequently this replay cannot test the PTrade-only branch where unified sell
score at least 30 plus live premium at least 8% bypasses price confirmation and
ADX protection. The formal PTrade file was not changed, and the absence of local
order changes must not be interpreted as proof that the bonus would be harmless
inside that live-only override.

## Decision

**REJECT before JoinQuant.** The fixed five-point rule did not improve accuracy
or change any filled order, so it failed the pre-registered materiality and
win-rate gates. No JoinQuant candidate and no PTrade candidate were generated.
Do not search neighboring K levels, point values, RSI duplicates, ETF exceptions,
or protection bypasses from this result.

