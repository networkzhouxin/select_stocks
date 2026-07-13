# Horizontal Price Structure Diagnostic Design

## Objective

Test one fixed, training-only hypothesis: whether official `cross-v0.3.2`
mild-uptrend buys made within one ATR below a pre-existing 20-trading-day
resistance level are consistently weaker than the remaining mild-uptrend buys.

The experiment is observation-only. It must not change the JoinQuant or PTrade
mainline, score, ranking, position sizing, order, or risk logic.

## Locked Definition

- Training window: `2019-01-01` through `2021-12-31` only.
- Warm-up: approved read-only 2018 daily bars may supply lookback history only.
- Signal date: T-1 for a T-day 09:35 decision.
- Level window: the 20 valid daily bars strictly before the signal date, so the
  latest level input is T-2.
- Resistance: maximum adjusted `high` in that fixed window.
- Support: minimum adjusted `low` in that fixed window.
- Normalization: official T-1 ATR(14) from the base score snapshot.
- Resistance distance: `(resistance - T-1 close) / T-1 ATR`.
- Support distance: `(T-1 close - support) / T-1 ATR`.
- Pressure bucket:
  - `breakout`: resistance distance is below zero.
  - `near_resistance`: resistance distance is from zero through one ATR.
  - `room_to_resistance`: resistance distance is above one ATR.
- Support bucket:
  - `breakdown`: support distance is below zero.
  - `near_support`: support distance is from zero through one ATR.
  - `away_from_support`: support distance is above one ATR.

No alternative periods, ATR thresholds, pivot algorithms, Fibonacci levels,
or volume-profile levels may be tested after seeing this result.

## Pre-Registered Hypothesis And Gate

The only actionable hypothesis is that `near_resistance` mild-uptrend entries
(`0 < trend_score < 20`) are weaker than all other mild-uptrend entries.

The gate passes only if:

1. Both the near-resistance and comparison subsets contain at least 15 closed
   trades overall.
2. Each subset contains at least 3 closed trades in each of 2019, 2020, and
   2021.
3. Near-resistance entries have both lower average trade return and lower win
   rate in every training year.

Profit/loss ratio, PnL, support buckets, and all-entry summaries are reported
for interpretation but cannot override this gate or generate a different rule.

## Safety Boundaries

- Reject any input frame containing a row after its base signal date.
- Require exactly 20 valid T-2-or-earlier bars; otherwise report `no_data`.
- Reject attribution containing a buy or sell outside 2019-2021.
- Use defensive score copies and attach diagnostic fields only.
- Do not inspect or run any reserved validation period.
- Close the one-item research budget immediately after recording the result.

## Deliverables

- Structured research-budget registration.
- Unit tests proving T-2 exclusion, fixed buckets, defensive copies, data-date
  enforcement, annual gate behavior, and validation-date rejection.
- Observation-only diagnostic module and a 2019-2021 training report.
- Decision, backtest note, failed-experiment or adopted-decision record, and a
  closed research budget.

