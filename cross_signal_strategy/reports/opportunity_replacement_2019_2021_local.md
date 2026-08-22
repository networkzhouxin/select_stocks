# Opportunity-Cost Replacement Candidate — Local Training Result

Date: 2026-08-22

Version: `cross-v0.3.3-opportunity-replacement-candidate` (local research only)

## Frozen hypothesis

When all three slots are occupied, every holding has completed the official
five-trading-day minimum hold, and a new ETF passes the complete official buy
filter, replace one holding only if its sell score is already at least 30 but
the official price-confirmation or ADX protection prevents an ordinary sell.
Choose the highest sell score, break ties with the lower current buy score, and
buy the highest-ranked official candidate. Permit at most one replacement per
day. Keep all indicators, thresholds, normal sells, ATR stops, sizing, pool,
fees, and 09:35/T-1 timing unchanged.

## Data and causal boundary

- Read-only warm-up: approved 2018 daily bars.
- Training only: approved 2019-01-01 through 2021-12-31 daily and 09:35 bars.
- Reserved validation, recent-market, live-outcome, and full-period data were
  not read or used.
- Decisions used T-1 score snapshots. T-day 09:35 prices were used only for
  execution and portfolio valuation.

## Local A/B result

| Metric | Official local baseline | Candidate |
| --- | ---: | ---: |
| Total return | +125.00% | +89.87% |
| Annualized return | +31.13% | +23.90% |
| Maximum drawdown | 6.03% | 6.18% |
| Sharpe ratio | 2.262 | 1.947 |
| Sortino ratio | 3.581 | 3.042 |
| Closed-trade win rate | 56.18% | 55.05% |
| Profit/loss ratio | 4.878 | 3.244 |
| Buys / sells | 92 / 89 | 112 / 109 |
| Closed trades | 89 | 109 |

Annual returns:

| Year | Official local baseline | Candidate |
| --- | ---: | ---: |
| 2019 | +35.84% | +32.68% |
| 2020 | +52.68% | +32.97% |
| 2021 | +8.49% | +7.62% |

The candidate completed 19 opportunity-replacement sells and 19 matching
replacement buys. The changed portfolio path affected 168 trading days.

## Decision

Reject before JoinQuant. The target metric, win rate, worsened instead of
improving, while return, drawdown, Sharpe, Sortino, profit/loss ratio, and every
training-year return also worsened. The rule increases recycling but the new
entry signal is not a reliable proof that a protected existing trend has lower
forward value. No standalone JoinQuant or PTrade candidate is generated.

Do not search nearby sell scores, stronger buy thresholds, ETF exceptions,
score spreads, hold periods, tie-breakers, or cooldowns from this result. The
formal JoinQuant and PTrade files remain unchanged.
