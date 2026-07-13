# Controlled Breakout Anti-Chase Experiment Design

## Objective

Test one fixed, training-only question raised by the user: whether an existing
cross-signal buy should be rejected when it has already broken resistance in a
technically overextended state. The strategy remains a reversal/cross strategy;
horizontal breakout must never create a buy, add score, or replace RSI, MACD,
and KDJ crosses as the primary entry mechanism.

This is a narrowly reopened, explicitly authorized experiment after the prior
horizontal-structure observation. It counts as another attempted strategy rule.
It does not reopen nearby channel periods, thresholds, breakout rewards, pivot
algorithms, Fibonacci levels, or volume-profile searches.

## Locked Data And Timing

- Training performance window: `2019-01-01` through `2021-12-31` only.
- Warm-up: approved read-only 2018 daily bars may supply indicator lookback only.
- Decision time: T-day 09:35.
- Base signal and extension state: T-1 daily data only.
- Resistance inputs: exactly 20 valid daily bars strictly before T-1, ending no
  later than T-2.
- Resistance: maximum adjusted `high` in that fixed window.
- Breakout: T-1 adjusted close is strictly above resistance.
- No reserved validation period may be read, run, summarized, or used to change
  this experiment.

## Locked Overextension Definition

An entry is an `extended_breakout` only when it is a breakout and at least one
of these already-established diagnostic states is true on T-1:

1. `RSI6 >= 75`.
2. `T-1 close / T-1 MA20 - 1 >= 10%`.

A breakout satisfying neither condition is a `controlled_breakout`. An entry
that does not break resistance is `no_breakout`.

These values reuse the project's existing coarse attribution boundaries. They
are not selected by a new sweep. The official strategy already rejects
`RSI6 >= 85` and entries more than 12% above MA20; this experiment asks only
whether the narrower condition is useful specifically after a resistance
breakout. Do not test RSI 70/80/85, MA20 distances 5%/8%/12%, different channel
lengths, or AND-versus-OR variants after seeing the result.

For interpretation only, report each breakout entry's T-1 RSI6, percentage
distance from MA20, trailing five-, ten-, and twenty-trading-day close return
ending on T-1, and the percentage rise from the same T-2-safe 20-day minimum
low to the T-1 close. These continuous diagnostics cannot create a different
rule or threshold.

## Stage One: Observation Gate

Classify the official local replay's closed buys without changing any order.
The anti-chase candidate may exist only if all conditions hold:

1. `controlled_breakout` and `extended_breakout` each contain at least six
   closed trades overall.
2. Both groups contain at least two closed trades in each of 2019, 2020, and
   2021.
3. `extended_breakout` has both lower average trade return and lower win rate
   than `controlled_breakout` in every training year.

If this gate fails, stop. Record the observation as non-adopted, create no
order-changing candidate, and leave both official platform strategies intact.

## Stage Two: Single Candidate

If and only if Stage One passes, run exactly one local candidate:

- Start from all official `cross-v0.3.2` score, eligibility, ranking, sizing,
  sell, ATR, holding-period, ETF-pool, and execution rules.
- Preserve the existing buy threshold and all cross-signal calculations.
- Reject an otherwise eligible new buy only when its label is
  `extended_breakout`.
- Do not reward or prioritize `controlled_breakout`.
- Do not alter existing holdings or any sell decision.

The candidate is eligible for a JoinQuant training candidate file only if:

1. The local order path changes, proving the rule is active.
2. Total return is strictly higher than the frozen local baseline.
3. Maximum drawdown is no higher than baseline.
4. Sharpe and Sortino are both no lower than baseline.
5. Candidate calendar-year return is no lower than baseline in 2019, 2020, or
   2021.

If any condition fails, reject the candidate and do not search a replacement.
If all pass, create an isolated JoinQuant candidate for an official 2019-2021
training confirmation; do not modify the formal JoinQuant or PTrade mainline.

## Safety And Implementation Boundaries

- Assert the approved training and warm-up roots through existing loaders.
- Reject frames containing data after the declared T-1 signal date.
- Use defensive score copies; observation must not mutate base snapshots.
- Write tests before implementation and verify the expected RED state.
- Keep source market-data folders read-only and write no derived files there.
- Do not touch the production multi-factor strategy.
- Record this attempt in the research ledger and increment the multiple-testing
  lower bound whether the rule passes or fails.
- A strong aggregate result cannot override annual or sample gates.

## Expected Files

- Create `cross_signal_strategy/breakout_extension_diagnostics.py` for pure
  classification, reporting, gates, and the isolated local candidate wrapper.
- Create `tests/test_cross_signal_breakout_extension_diagnostics.py` first.
- Update the research budget, backtest notes, decisions, failed/adopted record,
  README, and multiple-testing audit after the result is known.
- Create a JoinQuant candidate only if every local gate passes.
