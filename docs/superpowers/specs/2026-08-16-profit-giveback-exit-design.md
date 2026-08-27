# Profit-Giveback Direct Exit Design

Status: pre-registered 2026-08-16, user-authorized. One fixed variant; no search.
Outcome (2026-08-16): REJECTED at Step 0. The trade-level counterfactual fired
79 times across 21 affected closed trades with a negative total per-share delta
(-0.352; annual -0.380/-0.101/+0.129) because it clipped two major winners
(2019-02-11 159928, 2020-04-17 513050) while saving small amounts elsewhere.
The family is exhausted; no candidate was created. Full evidence is recorded in
`docs/backtest_notes.md` and `docs/failed_experiments.md`.

## Family Key

`profit_giveback_exit_user_authorized` (registered in `research_budget.json`).

## Reopen Justification

The profit-protection family has two prior failures in this repository: the
price-anchored break-even floor (cross-signal, -6.3pp, 2026-08-12) and the
profit-giveback protection overlay (multi-factor V2.10, +422%→+382%, rejected).
This reopen is authorized for one mechanism that was not tested in
cross-signal: a **profit-anchored direct exit** (sell when profit gives back a
fixed amount from its peak), motivated by:

1. The measured 28.4% conditional round-trip rate: 67 training trades reached
   at least one entry ATR of closing-price profit and 19 finished non-profitable.
2. The live 159985 case (2026-07): bought 2.155, peaked +6.2%, and the official
   5%-from-peak price stop sold it at about break-even, giving the whole profit
   back.
3. The rejected mechanisms were different: a cost-floor stop and multiplier
   tiering. A giveback exit anchors on profit itself, which is looser for small
   winners (where the current price stop locks almost nothing) and tighter for
   large winners (where it risks clipping trends).

## Hypothesis

For positions whose peak closing-price profit reaches at least 5%, exiting
immediately once current profit falls 3 percentage points below that peak
locks in more of the small-to-medium winners without damaging the large trends
the official stop already protects, improving total return while not worsening
maximum drawdown or any annual return.

## Frozen Variant

- Track `peak_profit = highest_since_buy / entry_cost - 1` (the strategy's own
  trailing-high state, closing-price based).
- At the daily 09:35 stop check, compute
  `current_profit = execution_price / entry_cost - 1`.
- If `peak_profit >= 0.05` AND `current_profit <= peak_profit - 0.03`:
  submit an immediate full exit (same unconditional path as an ATR stop,
  reason `giveback_stop`). Positions bought on the same trading day are exempt.
- Everything else (signals, scoring, ranking, sizing, minimum hold, ATR-stress
  rule, ETF pool) is untouched.

## Step 0: Trade-Level Counterfactual Observation (read-only, no candidate)

On the official `cross-v0.3.3` training replay, record every day where the
giveback rule would fire while the official path still holds the position, then
map each firing to its closed trade. Per trade, the delta is the difference
between exiting at the first rule-fire price and the official exit price (same
entry, same shares).

Gates (all required to proceed to Step 1):
- At least 5 affected closed trades.
- Positive total delta across the training window.
- Positive delta in each of 2019, 2020, and 2021 (annual consistency).

If any gate fails, the family closes without a candidate; no nearby thresholds
or mechanism variants are searched.

## Step 1: Local A/B

- Baseline: official `cross-v0.3.3` local training replay
  (+125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581;
  annual +35.84% / +52.68% / +8.49%; 92 buys / 89 sells).
- Candidate: isolated file under `archive/candidates/` (version
  `cross-v0.3.3-giveback-candidate`); mainline files stay untouched.

Pass gates (all required):
- Total return ≥ baseline.
- Max drawdown ≤ baseline; Sharpe and Sortino ≥ baseline.
- Every annual return (2019/2020/2021) ≥ baseline.
- At least 3 filled orders change.

## Step 2-4

- Step 2: JoinQuant 2019-2021 training confirmation with the same gates.
- Step 3: the four reserved validation windows, frozen protocol, record only.
- Step 4: only after all pass — adopt as `cross-v0.3.4`, sync PTrade, restore
  full-parity tests, and update the docs.

## Prohibited

No activation or giveback threshold search (5% / 3pp fixed), no relative
giveback fractions, no peak-profit floor hybrids, no tiered variants, no
per-ETF exceptions, no validation-period influence, and no post-hoc variant
after seeing any result.
