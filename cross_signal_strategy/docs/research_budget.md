# Cross-Signal Research Budget

This document is the human-readable map for the structured source of truth in
`research_budget.json`. It governs training-only research for `cross-v0.3.2`;
it does not change any score, order, position, or risk rule.

## Current Accounting

- Training window: 2019-01-01 through 2021-12-31.
- Recorded failed or non-adopted experiments: 48 real ledger entries.
- The empty `Date:` line in the ledger template is not an experiment.
- Mainline: `cross-v0.3.2` remains unchanged.
- Validation tuning: strictly forbidden. 不得查看或利用验证期结果选择指标、阈值、参数、ETF 或规则。
- New open budget: none. The horizontal-price-structure observation has been consumed and closed.
- The completed fixed `MACD(6,13,5)` comparison remains closed.

## Frozen Families

| Family key | Status | Decision |
| --- | --- | --- |
| `cross_signal_definition` | adopted | Positional cross alignment, T-1 timing, and the three-day window remain frozen. |
| `indicator_enumeration` | exhausted | Traditional indicator coverage is already broad; adding more now creates multiple-testing bias. |
| `threshold_and_period_search` | exhausted | Do not fine-tune indicator periods, score thresholds, ATR multipliers, or hold days. |
| `entry_confirmation` | exhausted | Volume, trend, weak-entry, gap, BOLL width, sequence, and backup-fill variants are closed. |
| `exit_and_atr_control` | exhausted | Signal-sell, ATR, cooldown, hold, and replacement-protection variants are closed. |
| `position_sizing_and_capacity` | exhausted | Exposure scales, idle-slot fills, and strong-trend extra capacity are closed. |
| `training_period_pool_selection` | exhausted | Do not delete ETFs based on 2019-2021 attribution. |
| `candidate_ranking` | exhausted | Existing-score ranking combinations are closed after the endpoint-artifact result. |
| `execution_gap_and_timing` | exhausted | Do not search smaller post-hoc gap thresholds. |
| `portfolio_dependence` | exhausted | The fixed 20-day/0.80 observation had only 9 high-dependence trades, and high dependence strongly outperformed in 2020. |
| `market_breadth` | exhausted | The fixed MA20/50% observation reversed in 2021; low breadth was then the better early-reversal environment. |
| `etf_microstructure` | exhausted | Only two above-5% buys existed, both in 2020 and `513100`; do not lower the threshold or widen the QDII scope post hoc. |
| `macd_half_cycle_user_authorized` | exhausted | The fixed `(6,13,5)` variant reduced return and most quality metrics; do not search neighboring MACD periods. |
| `horizontal_price_structure` | exhausted | The fixed 20-day T-2-safe/one-ATR observation failed annual consistency; do not search alternate windows, thresholds, pivots, breakout rules, or volume profiles. |

An exhausted family can reopen only after a new external market-structure reason
or a proven strategy change creates a genuinely different failure mode. A better
number from another nearby variant is not enough.

## Open Families

None. The fixed `horizontal_price_structure` observation found that mild
near-resistance entries did not consistently underperform: they were better on
both locked metrics in 2019 and had higher win rate in 2021. All 89 closed-buy
snapshots were more than one ATR above prior support, so the support premise had
no actionable sample. No neighboring periods, thresholds, pivot definitions,
Fibonacci levels, volume profiles, breakout rules, or post-hoc support rule may
be tested. Validation periods remain unavailable for tuning.

## Mandatory Sequence

1. Call the research-budget gate before writing an experiment.
2. Write tests first and verify the red state.
3. Implement observation-only attribution without changing orders.
4. Use only the approved warm-up and 2019-2021 training sources.
5. Apply annual consistency, sample-size, and concentration gates before any candidate exists.
6. Record the result in `failed_experiments.md` or the adopted decision log.
7. Update `expected_failed_experiment_count` when a new non-adopted ledger entry is appended.
8. Never use validation periods to reopen, tune, or select a research family.

The gate is intentionally strict: every open family permits exactly one
pre-registered variant. A request for multiple variants is rejected before a
backtest can turn into winner selection.
