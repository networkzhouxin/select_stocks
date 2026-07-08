# Cross-Signal Validation Protocol

Status: frozen before first reserved-period result is inspected.

## Purpose

Validate whether the current cross-signal training mainline and the ATR-stress candidate survive unseen market periods without using validation results to tune parameters, add indicators, remove ETFs, or redesign rules.

## Frozen Strategies To Run

Run these files exactly as they are before inspecting validation results:

1. Official mainline:
   - `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
   - Version: `cross-v0.3.1`
   - Role: current official training-period safety point.

2. Risk-control candidate:
   - `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
   - Version: `cross-v0.3.1-atr-stress-candidate`
   - Role: candidate only; not adopted.

## Validation Windows

Use JoinQuant as the performance authority.

First validation window:
- 2022-01-01 to 2023-12-31.
- Purpose: weak/sideways and difficult market validation.

Do not run or inspect later windows until the first validation result is recorded:
- 2024-01-01 to latest available date.
- 2015-01-01 to 2018-12-31.
- 2010-01-01 to 2014-12-31.
- 2015-latest final summary.

## Required Exports Per Run

For each strategy run, collect:

- Screenshot of JoinQuant performance summary.
- Full log export.
- Transaction detail CSV export.

Minimum log checks:
- Version line matches the intended strategy.
- `ERROR=0`.
- `Traceback=0`.
- `Exception=0`.
- Warnings are classified instead of ignored.
- Removed training-pool symbols `510300`, `510880`, and `159920` do not appear in trades.

ATR-stress candidate extra checks:
- Count all `stress=0.50` buy logs.
- List triggered dates, symbols, target values, and matched transaction values.
- Confirm the rule is not silently inactive unless that is the natural validation-period result.

## Fixed Evaluation Metrics

Record, at minimum:

- Strategy return.
- Annualized return.
- Excess return.
- Benchmark return.
- Max drawdown.
- Max drawdown interval.
- Sharpe ratio.
- Sortino ratio.
- Win rate.
- Profit/loss ratio.
- Alpha.
- Beta.
- Information ratio.
- Buy count.
- Sell count.
- Filled/canceled/rejected transaction counts.

## Decision Rules

### Official `v0.3.1`

The mainline passes first validation if:

- It does not produce runtime errors.
- It does not trade removed symbols.
- It does not suffer a catastrophic drawdown inconsistent with the user's capital tolerance.
- It remains meaningfully competitive versus the benchmark and does not collapse in the weak/sideways validation window.

If it fails:

- Do not tune immediately from the failed validation period.
- Record the failure and diagnose whether the issue is data/execution, code, or strategy structure.
- Only consider changes if the weakness has a clear market-structure explanation and can later be checked against another reserved period.

### ATR-Stress Candidate

The ATR-stress candidate can be considered for adoption only if it improves or preserves risk-adjusted behavior without materially damaging return.

Prefer adoption only if most of these are true versus official `v0.3.1`:

- Max drawdown is lower or not materially worse.
- Sharpe and Sortino are higher or not materially worse.
- Total/annualized return is not materially lower.
- Trade count and cancellations do not increase in a concerning way.
- `stress=0.50` triggers are understandable and not concentrated in a single validation oddity.

Reject or keep as candidate if:

- It improves only one metric while harming several others.
- It materially reduces return without a clear drawdown benefit.
- It triggers only once or twice and the improvement depends on one lucky historical event.
- It is inactive in validation, unless official `v0.3.1` already passes and the rule remains harmless.

## Anti-Overfitting Rules

- Do not change `15 trading days`, `3 ATR stops`, or `0.50 scale` after seeing validation.
- Do not add a new filter because of one validation chart or one validation trade.
- Do not remove more ETFs from the pool based on validation attribution.
- Do not use 2022-2023 to choose between multiple new variants.
- If both strategies disappoint, stop and record the result instead of immediately searching for a better validation fit.

## Reporting Template

For each validation run:

- Strategy/version:
- Period:
- Headline metrics:
- Transaction/log health:
- Removed-symbol check:
- Warning classification:
- ATR-stress trigger audit, if applicable:
- Comparison to official mainline, if applicable:
- Pass/fail/hold judgment:
- What this result proves:
- What this result does not prove:
- Next allowed action:
