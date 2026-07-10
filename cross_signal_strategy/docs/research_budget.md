# Cross-Signal Research Budget

This document is the human-readable map for the structured source of truth in
`research_budget.json`. It governs training-only research for `cross-v0.3.2`;
it does not change any score, order, position, or risk rule.

## Current Accounting

- Training window: 2019-01-01 through 2021-12-31.
- Recorded failed or non-adopted experiments: 43 real ledger entries.
- The empty `Date:` line in the ledger template is not an experiment.
- Mainline: `cross-v0.3.2` remains unchanged.
- Validation tuning: strictly forbidden. 不得查看或利用验证期结果选择指标、阈值、参数、ETF 或规则。
- New open budget: two independent families, one pre-registered observation per family.

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
| `etf_microstructure` | blocked | Point-in-time premium, NAV, quota, FX, and tracking data are not in the approved source. |

An exhausted family can reopen only after a new external market-structure reason
or a proven strategy change creates a genuinely different failure mode. A better
number from another nearby variant is not enough.

## Open Families

### `portfolio_dependence`

One observation-only representative is allowed. At each official buy decision,
measure standard 20-day return correlation using only adjusted closes available
through T-1. Use the pre-registered `0.80` split between the candidate and current
holdings. Do not search correlation windows or thresholds. A strategy candidate
is allowed only if high-dependence entries have adequate samples and consistently
worse return and drawdown contribution in 2019, 2020, and 2021.

### `market_breadth`

One observation-only representative is allowed. On T-1, calculate the share of
eligible pool ETFs above their standard MA20 and use the pre-registered majority
split of below `50%` versus at least `50%`. Do not search MA periods or breadth
thresholds. A strategy candidate is allowed only if the below-majority state
consistently identifies worse mild-trend entries in all three training years with
adequate samples.

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

