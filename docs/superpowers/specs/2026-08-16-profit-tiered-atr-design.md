# Profit-Tiered ATR Tightening Design

Status: user-authorized, pre-registered 2026-08-16. One fixed variant; no search.
Outcome (2026-08-16): REJECTED after Step 1 local A/B. The candidate changed 0
filled orders on the official training path (exact no-op), so the pre-registered
"at least 3 filled orders change" gate failed and the family is exhausted. Full
evidence is recorded in `docs/backtest_notes.md` and
`docs/failed_experiments.md`.

## Family Key

`profit_tiered_atr_user_authorized` (registered in `research_budget.json`).

## Reopen Justification

`exit_and_atr_control` was exhausted, but three external reasons justify this one
user-authorized reopen:

1. Proven strategy change: the ATR-stress rule was adopted as `cross-v0.3.3`
   (2026-08-16), changing the portfolio risk stack and its failure modes.
2. Independent framework evidence: the multi-factor strategy's V2.6 adopted the
   same mechanism (profit > 5% → multiplier ×0.8; profit > 15% → ×0.6) as its
   largest single improvement (+28pp on JoinQuant).
3. Cross-signal structural weakness: 67 training trades reached at least one
   entry ATR of closing-price profit, and 19 (28.36%) later finished at or below
   zero. The previously tested break-even floor failed; multiplier tiering is a
   different mechanism never tested in this framework.

## Hypothesis

For profitable high-volatility holdings, tightening the trailing ATR multiplier
by profit tier (2.5× → 2.0× above 5% profit → 1.5× above 15% profit) reduces
profit giveback without damaging trend-following, improving total return while
not worsening maximum drawdown.

## Frozen Variant

- `profit_pct = current_price / entry_cost - 1`, computed at the daily stop
  check from the execution-time price (same measurement as multi-factor V2.6:
  current profit, not peak profit).
- If `profit_pct > 0.15`: multiplier factor ×0.6 (2.5× → 1.5×).
- Else if `profit_pct > 0.05`: multiplier factor ×0.8 (2.5× → 2.0×).
- Else: multiplier unchanged.
- Stop formula unchanged otherwise:
  `stop = highest_close × (1 − clamp(mult × entry_atr / highest_close, 0.05, 0.15))`.
- The 5% stop floor and 15% stop cap are unchanged. The floor dominates for
  low-volatility entries, so the variant can only bind where the unfloored
  stop exceeds 5% (entry ATR / price above roughly 2%).

Changed call sites only: `calc_stop_price` gains a `profit_pct` parameter;
`check_atr_stops` and the after-close stop log pass the current profit. Buy
signals, sell signals, scoring, ranking, sizing, minimum hold, ETF pool, and
the ATR-stress rule are untouched.

## Step 0: Binding Observation (read-only, no candidate)

On the official `cross-v0.3.3` training replay, count stop-check events where
`profit_pct > 0.05` AND the unfloored baseline stop exceeds the 5% floor
(tightening would change the effective stop). Also count events where the
tightened stop would have triggered on that day while the baseline stop would
not.

Gate: at least 10 binding events required to proceed to Step 1. If the gate
fails, the family closes without a candidate; no wider tiers, multipliers, or
profit measurements are searched.

## Step 1: Local A/B

- Baseline: official `cross-v0.3.3` local training replay
  (+125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581;
  annual +35.84% / +52.68% / +8.49%; 92 buys / 89 sells).
- Candidate: isolated file under `archive/candidates/` (version
  `cross-v0.3.3-profit-tier-candidate`); mainline files stay untouched.

Pass gates (all required):
- Total return ≥ baseline.
- Max drawdown ≤ baseline; Sharpe and Sortino ≥ baseline.
- Every annual return (2019/2020/2021) ≥ baseline.
- The candidate changes at least 3 filled orders (a no-op variant fails).

Separately reported (no gate): the change in ATR-stop counts and any
interaction with the ATR-stress buy scale.

## Step 2-4

- Step 2: JoinQuant 2019-2021 training confirmation with the same gates.
- Step 3: the four reserved validation windows, frozen protocol, record only.
- Step 4: only after all pass — adopt as `cross-v0.3.4`, sync PTrade
  (including `profit_pct` at live stop checks), restore full-parity tests, and
  update the docs.

## Prohibited

No tier threshold search (5%/15% fixed), no factor search (0.8/0.6 fixed), no
peak-profit alternative measurement, no profit floor, no per-ETF overrides
(gold is a separate pre-registered direction), no validation-period influence,
and no post-hoc variant after seeing any result.
