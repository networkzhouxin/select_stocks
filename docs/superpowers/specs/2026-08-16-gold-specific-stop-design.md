# Gold-Specific Stop Design (Direction ②)

Status: pre-registered 2026-08-16. One fixed variant; no search.
Outcome (2026-08-16): REJECTED after Step 1 local A/B. The candidate bound
strongly (223 binding days, 6 extra triggers, gold ATR stops 2→5) but clipped
winners: return +125.00%→+120.96%, drawdown 6.03%→6.08%, 2019/2021 worse. The
family is exhausted. Full evidence is recorded in `docs/backtest_notes.md` and
`docs/failed_experiments.md`.

## Family Key

`gold_specific_stop_user_authorized` (to be registered in `research_budget.json`
upon authorization).

## Reopen Justification

`exit_and_atr_control` and `threshold_and_period_search` are exhausted, but this
one user-authorized reopen rests on:

1. Independent Walk-Forward evidence: the multi-factor V2.8 adopted the
   identical mechanism (gold 518880 `stop_floor=0.03`, `trailing_atr_mult=2.0`)
   after an 8-window walk-forward where it won 7/8 out-of-sample windows with
   monotonically accumulating annual improvement (+16.9pp full period, Sharpe
   1.19→1.22).
2. Mechanism fit: gold is the only mean-reverting asset in the cross-signal
   pool; its training attribution shows 72.7%-80% closed-trade win rate driven
   by reversals, not trend persistence, so the trend-style wide stop is
   structurally mismatched.
3. Framework-specific lever: the rejected profit-tier experiment (2026-08-16)
   proved that the 5% stop floor dominates this framework's stop behavior.
   Direction ② changes the floor itself (0.05→0.03) plus the multiplier
   (2.5→2.0) for gold only, which is exactly the lever that can bind here.
4. Proven strategy change: ATR-stress adoption (`cross-v0.3.3`) changed the
   portfolio risk stack.

## Hypothesis

For 518880 only, a tighter trailing stop (floor 3%, multiplier 2.0×) preserves
more of gold's mean-reversion profits by exiting failed bounces earlier,
improving total return while not worsening maximum drawdown or any annual
return.

## Frozen Variant

- Two new frozen parameters: `gold_stop_floor=0.03`,
  `gold_trailing_atr_mult=2.0` (multi-factor V2.8 values, copied verbatim).
- `calc_stop_price` gains a `code` argument; for code prefix `518880` it uses
  the gold values, otherwise the official 5%/2.5×. Stop cap stays 0.15 for all
  ETFs. Formula otherwise unchanged:
  `stop = highest_close × (1 − clamp(mult × entry_atr / highest_close, floor, cap))`.
- Call sites pass the code (`check_atr_stops`, after-close stop log).
- Buy signals, sell signals, scoring, ranking, sizing, minimum hold, ETF pool,
  and the ATR-stress rule are untouched.

## Step 0: Binding Observation (read-only, no candidate)

On the official `cross-v0.3.3` training replay, count 518880 stop-check days
where the gold-specific stop differs from the baseline stop, and count days
where the gold stop would have triggered while the baseline stop would not
(extra triggers).

Gates: at least 10 binding check-days AND at least 3 extra-trigger days
required to proceed to Step 1. Otherwise the family closes without a candidate;
no nearby floor or multiplier values are searched.

## Step 1: Local A/B

- Baseline: official `cross-v0.3.3` local training replay
  (+125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581;
  annual +35.84% / +52.68% / +8.49%; 92 buys / 89 sells).
- Candidate: isolated file under `archive/candidates/` (version
  `cross-v0.3.3-gold-stop-candidate`); mainline files stay untouched.

Pass gates (all required):
- Total return ≥ baseline.
- Max drawdown ≤ baseline; Sharpe and Sortino ≥ baseline.
- Every annual return (2019/2020/2021) ≥ baseline.
- At least 3 filled orders change (a no-op variant fails).

Separately reported (no gate): the change in gold ATR-stop counts, any
interaction with the ATR-stress buy scale, and the per-trade attribution of
changed gold exits (avoided loss versus clipped winner).

## Step 2-4

- Step 2: JoinQuant 2019-2021 training confirmation with the same gates.
- Step 3: the four reserved validation windows, frozen protocol, record only.
- Step 4: only after all pass — adopt as `cross-v0.3.4`, sync PTrade
  (gold stop params in the frozen config and live stop checks), restore
  full-parity tests, and update the docs.

## Prohibited

No floor/multiplier value search for gold (0.03/2.0 fixed), no extension to
other ETFs (soymeal, QDII, and A-share tightenings all failed in multi-factor
and are closed), no profit floors, no tiered multipliers, no validation-period
influence, and no post-hoc variant after seeing any result.
