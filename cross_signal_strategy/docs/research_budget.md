# Cross-Signal Research Budget

This document is the human-readable map for the structured source of truth in
`research_budget.json`. It governs training-only research for `cross-v0.3.2`;
it does not change any score, order, position, or risk rule.

## Current Accounting

- Training window: 2019-01-01 through 2021-12-31.
- Recorded failed or non-adopted experiments: 56 real ledger entries.
- The empty `Date:` line in the ledger template is not an experiment.
- Mainline: `cross-v0.3.2` remains unchanged.
- Validation tuning: strictly forbidden. 不得查看或利用验证期结果选择指标、阈值、参数、ETF 或规则。
- New open budget: zero. The one fixed `etf_share_flow_shadow`
  `positive_vs_non_positive` observation is complete and exhausted.
- The completed fixed `MACD(6,13,5)` comparison remains closed.
- The completed fixed `cross_window=1/2/3/4` comparison remains closed; window 3 is retained.
- The completed fixed `09:35/10:00` execution-time comparison remains closed; `09:35` is retained.
- The completed fixed ordinary-buy minute execution overlay remains closed; its
  aggregate fill improvement did not remain positive in 2020.
- The completed fixed one-entry-ATR break-even candidate remains closed; it
  worsened return, drawdown, risk-adjusted metrics, win rate, and 2020/2021.
- The completed fixed MACD-free/KDJ-only candidate remains closed; it nearly
  doubled trading activity and worsened every aggregate gate and every training year.
- One independent QDII underlying-index direction observation is pre-registered.
  Four raw 2018-2021 value series are staged and hashed, but the observation
  remains blocked because `SPX` and `H30533` lack approved historical
  final-value availability policies. The formal source root does not exist.
  A blocked observation is not an open experiment budget.
- The 2026-07-18 publisher-evidence audit is closed without an unlock: S&P DJI
  permits EOD recalculation and reposting, while CSI documents daily H30533
  publication without an exact historical clock, timezone, or finality cutoff.
- Prospective log collection does not reopen a research family. Future PTrade
  exports may be archived from 2026-07-18 onward, but their outcomes cannot be
  evaluated until a hypothesis and a later independent confirmation sample are
  frozen in advance.

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
| `underlying_market_direction` | blocked | The fixed four-index sign-only observation cannot run until official final values, source calendars, and timezone-aware historical availability timestamps pass the isolated data contract. |
| `macd_half_cycle_user_authorized` | exhausted | The fixed `(6,13,5)` variant reduced return and most quality metrics; do not search neighboring MACD periods. |
| `cross_window_user_authorized` | exhausted | The fixed `1/2/3/4` matrix retained window 3; do not search wider, per-indicator, or age-weighted windows. |
| `execution_time_user_authorized` | exhausted | The fixed `09:35/10:00` comparison failed its annual, trade-quality, and non-QDII execution gates. Retain `09:35`; do not search nearby or ETF-specific times. |
| `horizontal_price_structure` | exhausted | The fixed 20-day T-2-safe/one-ATR observation failed annual consistency; do not search alternate windows, thresholds, pivots, breakout rules, or volume profiles. |
| `controlled_breakout_anti_chase` | exhausted | The fixed 20-day T-2-safe observation found only 2 extended breakouts, failed total and annual sample gates, and did not permit a candidate. Do not search another `RSI6 >= 75`, MA20 10%, window, AND rule, breakout reward, or sell change. |
| `etf_share_flow_shadow` | exhausted | The fixed five-observation sign-only attribution covered all 52 eligible domestic buys, but non-positive flow led average return in 2019/2020 while positive flow led both metrics in 2021. No candidate was permitted. |
| `intraday_execution_overlay_v1` | exhausted | The fixed 09:35 arrival-price limit with six five-minute cycles and a 10:05 fallback matched 92/92 ordinary buys, but average execution worsened in 2020. No full portfolio candidate was permitted. |
| `entry_atr_breakeven_user_authorized` | exhausted | The fixed 1-entry-ATR activation and cost floor reduced return and trade quality. Do not search nearby ATR activations, profit floors, staged stops, or ETF/year exceptions. |
| `macd_free_kdj_exit_user_authorized` | exhausted | The fixed MACD-observation-only buy score plus K/D-only ordinary exit reduced return from 120.61% to 41.87%, worsened every major metric and every training year, and increased buys from 92 to 170. Do not decompose or search nearby KDJ/MACD/hold/threshold variants post hoc. |
| `macd_fast_exit_user_authorized` | exhausted | The fixed recent MACD death-cross OR exit after the five-day hold reduced return from 120.61% to 81.75%, worsened every major metric and every training year, and increased buys from 92 to 128. Keep the official sell-score, price-structure, and ADX protections; do not search nearby MACD/hold/window variants post hoc. |
| `kdj_only_exit_user_authorized` | exhausted | The fixed K/D-death-cross-only ordinary exit retained official MACD buy scoring but reduced return from 120.61% to 42.64%, worsened every major metric and every training year, and increased buys from 92 to 170. Keep the official sell-score, price-structure, and ADX protections; do not search nearby KDJ/hold/window variants post hoc. |

An exhausted family can reopen only after a new external market-structure reason
or a proven strategy change creates a genuinely different failure mode. A better
number from another nearby variant is not enough.

## Open Families

None. `underlying_market_direction` is blocked rather than open. Raw values in
`G:\financial\history_data\cross_signal_underlying_staging_2018_2021` are not
formal point-in-time data and cannot run the observation. The approved root
must remain absent until all four publication-time policies pass. Its exact
schema and frozen gate are documented in `underlying_market_direction.md` and
the acquisition evidence is documented in `underlying_source_acquisition.md`.
The publisher-evidence audit may reopen only for new primary evidence that
proves 2018-2021 point-in-time final availability; another search result or an
assumed market-close timestamp is not sufficient.

`etf_share_flow_shadow` consumed its only fixed observation. It used the
isolated read-only root
`G:\financial\history_data\cross_signal_flow_train_2018_2021`, exactly five
prior share observations, and the fixed `positive_vs_non_positive` grouping.
All 52 eligible domestic entries were comparable: 24 positive and 28
non-positive. Non-positive flow led average return in 2019 and 2020, but
positive flow led both average return and win rate in 2021. The annual gate
therefore failed. Do not search another period, threshold, magnitude bucket,
fund-size/NAV interaction, QDII extension, or sell rule from this result.

`intraday_execution_overlay_v1` also consumed its only fixed experiment. It
froze the formal 09:35 ordinary-buy code and quantity, prohibited same-minute
and touch-only fills, allowed six five-minute passive-limit cycles, and used the
first executable minute at or after 10:05 as the market fallback. All 92
eligible buys were comparable: 75 filled passively and 17 used the fallback.
Average signed execution improvement was +2.63 basis points overall, but the
annual averages were +1.02/-0.78/+6.73 basis points in 2019/2020/2021. The
pre-registered every-year gate therefore failed before a full portfolio
candidate could exist. Do not search nearby times, cycle counts, limit prices,
fallbacks, ETF exceptions, or sell-side overlays from this result.

`macd_fast_exit_user_authorized` consumed its one fixed candidate. The recent
MACD death-cross OR exit was activated only after the official five-day hold,
but it still changed 156 filled-order days and worsened every aggregate metric
and all three annual returns. Do not search MACD periods, cross windows, hold
days, delayed confirmations, profit conditions, or ETF/year exceptions from
this result.

`kdj_only_exit_user_authorized` consumed its one fixed candidate. It retained
the complete official buy path, including MACD scoring, and changed only the
ordinary exit to a recent K/D death cross after five trading days. It worsened
every aggregate metric and every annual return while increasing buys from 92
to 170. Do not search K/D versus J/D, KDJ periods, hold days, cross windows,
profit conditions, delayed confirmations, or ETF/year exceptions from this
result.

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

The PTrade forward-log archive is an evidence-preservation tool, not an
experiment. It records immutable raw-log hashes and non-performance event
counts only. Logs already collected when a future question is formulated are
discovery material; only continuously archived logs generated after that
question is pre-registered can serve as its independent confirmation sample.
