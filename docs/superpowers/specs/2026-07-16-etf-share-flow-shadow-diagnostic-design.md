# ETF Share-Flow Shadow Diagnostic Design

## Objective

Test one independent, observation-only question using the frozen
`cross-v0.3.2` training replay: do existing upward-cross entries behave
differently when the ETF's primary-market shares outstanding increased over
the preceding five trading observations?

This experiment does not create a buy, block a buy, change a score, rank a
candidate, size a position, or affect any sell or ATR decision. A positive
result may justify a separately pre-registered order-changing experiment; it
cannot change either formal platform strategy by itself.

## Locked Data Scope

- Performance and signal window: `2019-01-01` through `2021-12-31` only.
- Share-flow warm-up: approved read-only 2018 share rows may supply lookback
  only and must never enter performance statistics.
- Approved share-flow root:
  `G:\financial\history_data\cross_signal_flow_train_2018_2021`.
- The share-flow root and the existing price-data roots are immutable.
- Formal JoinQuant and PTrade `cross-v0.3.2` files remain untouched.
- Production multi-factor files remain untouched.
- Reserved validation dates must not be read, run, summarized, or used to
  change this diagnostic.

## Eligible Universe And Availability

The fixed domestic-ETF research universe is:

`159915`, `512100`, `159928`, `518880`, and `159985`.

The four QDII ETFs `513100`, `513500`, `513880`, and `513050` are always
labelled `blocked_qdii`. Their exact historical publication time is not proven,
so their share records cannot participate in this experiment.

At T-day 09:35, the diagnostic may use only the share row whose
`trade_date` exactly equals the official score's T-1 `signal_date`. Any row on
or after T, any mismatch between the price signal date and share-flow date, or
any frame containing a future row is an error rather than a fallback.

## Locked Feature

For an eligible ETF and T-1 signal date, take the last six valid share
observations ending exactly on T-1 and calculate:

```text
share_flow_5 = log(total_share_wan[T-1] / total_share_wan[T-6])
```

No alternate periods, smoothing, winsorization, magnitude thresholds, z-scores,
fund-size fields, NAV fields, price interactions, or parameter sweeps are
permitted after observing the result.

The raw state labels are:

- `net_creation`: `share_flow_5 > 0`.
- `net_redemption`: `share_flow_5 < 0`.
- `flat`: `share_flow_5 == 0` within exact floating-point comparison.
- `corporate_action`: the interval crosses a registered share split.
- `insufficient_history`: fewer than six valid observations or no exact T-1 row.
- `blocked_qdii`: code is in the fixed QDII set.

The pre-registered statistical comparison combines `net_redemption` and
`flat` into `non_positive`; it compares that group with `positive`
(`net_creation`). Raw states remain visible in the report.

## Corporate-Action Reset

`159928` had a four-for-one share split on `2021-06-25`. If a five-observation
comparison interval contains a registered corporate-action date strictly after
the baseline row and on or before the T-1 endpoint, label the observation
`corporate_action` and expose no numeric flow value.

The first window whose baseline is on or after the split is valid again. The
diagnostic must not infer or optimize split factors from price movements.

## Shadow Integration

Wrap the official local training signal adapter and add diagnostic fields to a
defensive copy of each score. The wrapper must preserve all original fields and
all order decisions. Run the ordinary `DiagnosticOrderPlanner` and local replay,
then attach the entry-time flow metadata to closed trades.

The report must include:

- coverage across all closed buys and across eligible domestic closed buys;
- raw-state counts;
- `positive` versus `non_positive` closed-trade count, wins, losses, realized
  PnL, average return, win rate, and gross profit/loss ratio;
- the same two-group statistics separately for 2019, 2020, and 2021;
- an explicit observation-gate decision and reasons.

## Observation Gate

The observation can be considered stable enough for a future, separately
designed candidate only when all conditions hold:

1. `positive` and `non_positive` each contain at least six closed trades.
2. Each group contains at least two closed trades in each of 2019, 2020, and
   2021.
3. One group has both higher average return and higher win rate in the same
   direction in every training year.

The gate does not adopt a strategy rule. If it fails, close this research
family and prohibit nearby flow periods or thresholds. If it passes, record
only that a separate one-shot candidate may be designed; do not implement that
candidate in this experiment.

## Safety And Evidence Requirements

- Tests precede every implementation change and the expected RED state must be
  observed.
- The loader asserts the exact approved root, schema, date ranges, positive
  shares, unique dates, and defensive copies.
- The score wrapper rejects future rows and signal-date disagreement.
- The report rejects any trade outside 2019-2021.
- Observation must leave the source score and official order path unchanged.
- Register one research attempt before execution, then close it and increment
  the multiple-testing lower bound after the result.
- Record the hypothesis, exact change, result, interpretation, and next step,
  including a failed or sparse result rather than silently discarding it.

## Expected Files

- Create `cross_signal_strategy/share_flow_diagnostics.py`.
- Create `tests/test_cross_signal_share_flow_diagnostics.py` first.
- Update research-budget tests and records before running the experiment.
- Update the research ledger, backtest notes, decisions, README, and
  multiple-testing audit after the training-only result is known.
- Do not create a platform strategy candidate in this experiment.
