# Intraday-High Trailing Anchor Design

Status: pre-registered 2026-08-16, user-authorized. One fixed variant; no search.
Outcome (2026-08-16): REJECTED after Step 1 local A/B. The candidate bound
strongly (1604 binding days, 38 extra triggers, 175 changed fills) but failed
the gates: +125.00%→+119.40% return, 6.03%→6.06% drawdown, 2019 +35.84%→+30.55%,
because the high anchor clipped the 2019-02-11 159928 winner (2.232 vs 2.666).
The family is exhausted. Full evidence is recorded in `docs/backtest_notes.md`
and `docs/failed_experiments.md`.

## Family Key

`intraday_high_anchor_user_authorized` (registered in `research_budget.json`).

## Reopen Justification

The official stop anchors on closing prices by an explicit design rule
(noise-resistant trailing high, multiplier calibrated to closes, live-verifiable
finalized bars). The user asked to test the alternative empirically:
anchor on intraday highs instead. This is a tightening direction (the stop
sits closer to the current price by the peak-day upper-wick size), and the
prior from the gold-stop and giveback experiments predicts it clips winners,
but no data has directly measured this swap with all other parameters frozen.

## Hypothesis

Replacing the trailing-high anchor with the intraday high, keeping the 2.5×
multiplier and the 5%/15% clamps unchanged, improves the stop's responsiveness
without damaging trend-following — improving total return while not worsening
maximum drawdown or any annual return.

## Frozen Variant

- New state `highest_high_since_buy`: the maximum completed daily HIGH bar
  from the buy date onward (buy day included), initialized at the buy fill
  price and updated after each session close with that session's final high.
- Stop formula unchanged except for the anchor:
  `stop = highest_high × (1 − clamp(2.5 × entry_atr / highest_high, 0.05, 0.15))`.
- Entry ATR, multiplier 2.5, floor 5%, cap 15%, same-day-buy exemption, and
  every other rule (signals, scoring, ranking, sizing, minimum hold,
  ATR-stress rule, ETF pool) are untouched.

## Step 0: Binding Observation (read-only, no candidate)

On the official `cross-v0.3.3` training replay, record every stop-check day
where the high-anchored stop differs from the close-anchored stop (binding)
and every day where the high-anchored stop would trigger while the
close-anchored stop would not (same-day extra trigger).

Gates: at least 10 binding days AND at least 3 extra-trigger days required to
proceed to Step 1. Otherwise the family closes without a candidate; no anchor
blends, multiplier re-calibrations, or threshold searches are allowed.

## Step 1: Local A/B

- Baseline: official `cross-v0.3.3` local training replay
  (+125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581;
  annual +35.84% / +52.68% / +8.49%; 92 buys / 89 sells).
- Candidate: isolated file under `archive/candidates/` (version
  `cross-v0.3.3-high-anchor-candidate`); mainline files stay untouched.

Pass gates (all required):
- Total return ≥ baseline.
- Max drawdown ≤ baseline; Sharpe and Sortino ≥ baseline.
- Every annual return (2019/2020/2021) ≥ baseline.
- At least 3 filled orders change.

## Step 2-4

- Step 2: JoinQuant 2019-2021 training confirmation with the same gates.
- Step 3: the four reserved validation windows, frozen protocol, record only.
- Step 4: only after all pass — adopt as `cross-v0.3.4`, sync PTrade
  (including high-bar confirmation in the live close-confirmation machinery),
  restore full-parity tests, and update the docs.

## Prohibited

No anchor blends (e.g., max(close, high×0.995)), no multiplier
re-calibration, no floor/cap changes, no per-ETF anchor exceptions, no
validation-period influence, and no post-hoc variant after seeing any result.
