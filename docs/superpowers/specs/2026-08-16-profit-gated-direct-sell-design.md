# Profit-Gated Direct-Sell Matrix Design

Status: pre-registered 2026-08-16, user-authorized. One fixed 4x3 matrix; no
search. The selection rule is declared before any data is seen.
Outcome (2026-08-16): REJECTED at Step 0. All 12 variants failed the gates:
the 38/40 score thresholds never fired, and the 32/35 thresholds produced
negative total per-share deltas because the 513050 +34% winner's mid-hold
pullback satisfies the trigger and would be exited early. The family is
exhausted; no candidate was created. Full evidence is recorded in
`docs/backtest_notes.md` and `docs/failed_experiments.md`.

## Family Key

`profit_gated_direct_sell_user_authorized` (registered in `research_budget.json`).

## Reopen Justification

The sell side has prior failures (sell35 threshold, weak-replacement
protection, profit-giveback exit), but they share one cause: they fired
regardless of how much profit was at stake, so they clipped large winners. A
strong reversal signal (sell score ≥ 32) inside a small profit band (2-6%) is
different: at such profits the position is not yet a big winner, so selling
before the price-structure confirmation arrives caps the cost of being early.
The user's live case (159985 peaked +6.2% with sell score 35 and no structure
confirmation, then round-tripped to break-even) is the motivating sample.

## Frozen Matrix (12 variants, one family, one replay)

| Sell-score threshold \ Profit band | 2% ~ 4% | 3% ~ 5% | 4% ~ 6% |
|---|---|---|---|
| 32 | A1 | A2 | A3 |
| 35 | B1 | B2 | B3 |
| 38 | C1 | C2 | C3 |
| 40 | D1 | D2 | D3 |

Per-variant trigger at the daily 09:35 sell evaluation, for held positions the
official path is NOT selling that day:

- `sell_score >= threshold` (T-1 score) AND `current_profit ∈ [low, high)` where
  `current_profit = 09:35 price / entry cost - 1`;
- exempt: positions bought on the same trading day, positions still inside the
  5-trading-day minimum hold, and positions protected by the ADX strong-uptrend
  exemption (ADX ≥ 25, +DI > -DI, MA20 slope non-negative, no severe break);
- the trigger bypasses the price-structure confirmation of the official
  channel. The official sell channel, buys, ATR stops, and the ATR-stress rule
  are unchanged.

## Step 0: Trade-Level Counterfactual Observation (one replay, 12 variants)

On the official `cross-v0.3.3` training replay, record every firing event per
variant, map each firing to its closed trade, and compare the first firing
price with the official exit price (same entry, same shares).

Per-variant gates (all required to pass):
- At least 5 affected closed trades.
- Positive total per-share delta.
- Positive per-share delta in each of 2019, 2020, and 2021.

Selection rule (pre-registered): if any variants pass, select ONLY the one
with the highest total per-share delta; all other variants are recorded as
failed. If none pass, the family closes and no variant may be searched again.

## Step 1: Local A/B

- Baseline: official `cross-v0.3.3` local training replay
  (+125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581;
  annual +35.84% / +52.68% / +8.49%; 92 buys / 89 sells).
- Candidate: isolated file under `archive/candidates/` (version
  `cross-v0.3.3-profit-gated-candidate`) implementing only the selected
  variant; mainline files stay untouched.

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

No nearby thresholds (31/33/34/36/37/39), no nearby band edges (1.5%/2.5%/...),
no relative profit fractions, no re-selection after seeing Step 1 results, no
per-ETF exceptions, no validation-period influence, and no post-hoc variants.
