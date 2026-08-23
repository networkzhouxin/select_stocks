# Cross-Signal Research Budget

This document is the human-readable map for the structured source of truth in
`research_budget.json`. It governs training-only research for `cross-v0.3.2`;
it does not change any score, order, position, or risk rule.

## Current Accounting

- Training window: 2019-01-01 through 2021-12-31.
- Recorded failed or non-adopted experiments: 79 real ledger entries.
- The empty `Date:` line in the ledger template is not an experiment.
- Mainline: `cross-v0.3.3` remains unchanged.
- Validation tuning: strictly forbidden. 不得查看或利用验证期结果选择指标、阈值、参数、ETF 或规则。
- New open budget: zero. The sole implementation-correction replay of the
  approved v0.4 dimension-capped rule has been consumed and failed seven frozen
  local gates. The first local run remains classified and preserved separately
  as `invalid_implementation`; the corrected canonical report is the only
  evidence used to reject the approved rule. No JoinQuant/PTrade candidate or
  neighboring variant is open. The separate late-veto/early-pre-MACD family
  remains blocked, not open.
- The user-authorized `fresh_unextended_entry_user_authorized` family is
  exhausted. Its official JoinQuant result reduced return from 129.25% to
  111.14%, win rate from 55.8% to 49.0%, and profit/loss ratio from 5.297 to
  3.904 while positive-to-negative round trips increased from 31 to 39.
- The fixed `etf_share_flow_shadow` `positive_vs_non_positive` observation is
  complete and exhausted.
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
| `dimension_capped_score_v04_user_authorized` | exhausted | The corrected rule changed 196 filled-order days, but win rate fell 56.18%→51.76%, return fell +125.00%→+78.13%, and every frozen payoff/robustness gate listed in the canonical report failed. Doubled-friction return fell +108.15%→+63.32% and win rate fell 51.69%→45.88%. No neighboring variant, JoinQuant/PTrade candidate, or validation access is allowed. |
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
| `profit_tiered_atr_user_authorized` | exhausted | The fixed profit-tiered ATR tightening (×0.8 above 5% profit, ×0.6 above 15% profit) found 36 binding stop checks but 0 days where the tightened stop triggered while the baseline stop did not. The local A/B changed 0 filled orders and every metric matched the cross-v0.3.3 baseline, so the candidate was rejected before JoinQuant or validation. The 5% stop floor plus frozen entry ATR makes the multi-factor V2.6 mechanism a no-op in cross-signal; no tier/multiplier/measurement/floor/per-ETF search is allowed. |
| `gold_specific_stop_user_authorized` | exhausted | The fixed gold-only stop (518880: floor 0.03, multiplier 2.0×) bound on 223 check days with 6 extra triggers, but the local A/B failed the gates: return +125.00%→+120.96%, drawdown 6.03%→6.08%, Sharpe/Sortino worse, 2019 and 2021 annual returns worse. The first extra stop clipped a 2019 winner (+4.7% exit versus +9.0% baseline exit) and cascaded into 162 changed fills. The multi-factor V2.8 gold stop does not transfer to this reversal-entry framework; no nearby gold floor/multiplier values and no per-ETF extension are allowed. |
| `profit_giveback_exit_user_authorized` | exhausted | The fixed profit-giveback direct exit (peak profit ≥5%, giveback 3pp → sell at the 09:35 stop check) fired 79 times across 21 affected closed trades, but the total per-share delta was -0.352 with annual deltas -0.380/-0.101/+0.129, failing the total and annual gates. It would have clipped two major winners (2019-02-11 159928 and 2020-04-17 513050) while saving small amounts elsewhere. Large trend winners routinely give back more than 3pp mid-hold, so the mechanism kills this framework's payoff source. No candidate was created; no threshold or mechanism search is allowed. |
| `intraday_high_anchor_user_authorized` | exhausted | The fixed intraday-high trailing anchor (multiplier 2.5, floor 5%, cap 15% unchanged) bound on 1604 check days with 38 extra triggers, but the local A/B failed the gates: return +125.00%→+119.40%, drawdown 6.03%→6.06%, 2019 annual +35.84%→+30.55%. The dominant clip was 2019-02-11 159928 (exit 2.232 versus official 2.666 on an +18.8% winner); seven small saves could not offset it. The peak-day upper wick raises the anchor into the winner's normal pullback band, turning the noise the close anchor was designed to filter back into stop triggers. No anchor blends or re-calibrations are allowed. |
| `profit_gated_direct_sell_user_authorized` | exhausted | The fixed 4×3 direct-sell matrix (sell scores 32/35/38/40 crossed with profit bands 2-4%/3-5%/4-6%) was consumed with 0 of 12 variants passing the Step 0 gates. The 38/40 thresholds never fired (scores that high arrive only after profit leaves the band), and the 32/35 thresholds produced negative total per-share deltas because the 513050 +34% winner's mid-hold pullback satisfies the trigger and would be exited early. No candidate was created; no nearby thresholds, bands, or mechanism variants may be searched. |
| `bullish_cross_age2_half_decay_user_authorized` | exhausted | The fixed buy-side age-2 multiplier 0.5 changed 64 filled-order days across all three years but cut return +125.00%→+87.35%, worsened drawdown 6.03%→8.79%, Sharpe, Sortino, profit/loss ratio, and every annual return. Keep full official weights for ages 0/1/2; no other decay coefficient, per-indicator age weight, exception, or compensating rule may be searched. |
| `intraday_signal_clock_1445_user_authorized` | exhausted | The fixed causal 09:35+14:45 full signal candidate reduced return from +125.00% to +85.00%, raised drawdown from 6.03% to 7.49%, lowered win rate from 56.18% to 47.66%, lowered profit/loss ratio to 2.813, and increased positive-to-negative round trips from 31 to 40. Doubled friction also worsened return and drawdown. Retain the official single 09:35 path; no nearby-time, per-ETF, side-only, indicator-subset, threshold, hold, or cooldown search is allowed. |
| `fresh_unextended_entry_user_authorized` | exhausted | The official JoinQuant candidate reduced return 129.25%→111.14%, win rate 55.8%→49.0%, and profit/loss ratio 5.297→3.904, while positive-to-negative round trips rose 31→39. The fresh channel closed 4 winners and 15 losers. Reject and archive it; keep official score≥60 and do not search neighboring score, age, ATR, ETF, queue, or sell-compensation variants. |
| `late_macd_boll_upper_filter_user_authorized` | exhausted | The exact standalone veto emitted 2 events but left win rate unchanged at 55.8% and reduced return 129.25%→124.09%, annual return 32.86%→31.83%, profit/loss ratio 5.297→5.208, Sharpe 2.275→2.185, and information ratio 0.839→0.790. It is rejected and retained only as a controlled comparison base; no nearby veto rule is allowed. |
| `late_veto_early_pre_macd_user_authorized` | blocked | One user-authorized stacked candidate keeps the failed late veto, preserves the full ≥60 primary queue first, and lets only 50-59 entries with fresh RSI/KDJ crosses plus a negative but narrowing pre-cross MACD spread fill leftover slots. BOLL upper and RSI6≥85 remain hard exclusions; sells are unchanged. It is frozen pending one official 2019-2021 JoinQuant run; no alternatives are allowed. |
| `opportunity_replacement_user_authorized` | exhausted | The fixed full-capacity rule replaced only sell-score-at-least-30 holdings blocked by price/ADX when all three holdings had completed five sessions and a formal score-at-least-60 candidate existed. Nineteen replacements cut return +125.00%→+89.87%, win rate 56.18%→55.05%, worsened drawdown and every risk-adjusted/annual metric, and increased buys 92→112. Reject before JoinQuant; do not search thresholds, score spreads, ETF exceptions, rankings, hold periods, or cooldowns. |
| `kdj_extreme_zone_score_user_authorized` | exhausted | The fixed K≤20 buy +5 / K≥80 sell +5 unified-score candidate changed zero filled orders. Of 93 oversold bonuses none crossed 60; 11 of 1,382 overbought bonuses crossed 30, but every event remained blocked by formal price/ADX protection. All normal and doubled-friction metrics were identical, so stop before JoinQuant. Historical point-in-time IOPV was unavailable; the live-only override remains untested and undeployed. |
| `kdj_tiered_persistence_user_authorized` | exhausted | The fixed 10/5-point, three-session KDJ state rule changed 22 filled-order days but none in 2021. Win rate fell 56.18%→55.43%, return 125.00%→118.33%, profit/loss ratio 4.878→4.384, and doubled-friction win rate 51.69%→48.91%. Reject before JoinQuant/PTrade; do not search nearby points, zones, retention lengths, direction precedence, or protection overrides. |
| `kdj_tiered_current_state_user_authorized` | exhausted | The fixed 10/5-point current-T-1-only KDJ state rule changed zero filled orders. All normal and doubled-friction metrics exactly matched the baseline, so accuracy did not improve. Reject before JoinQuant/PTrade; do not search nearby points, zones, state definitions, or protection overrides. |
| `kdj_tiered_direct_exit_user_authorized` | exhausted | The fixed current-T-1 10/5-point tiers plus direct extreme exits changed 155 filled-order days and raised win rate 56.18%→58.21%, but cut return 125.00%→101.84%, profit/loss ratio 4.878→2.954, and doubled-friction return 108.15%→77.35%. Reject before JoinQuant/PTrade; do not search nearby tiers, thresholds, retention, protection, or hold variants. |

An exhausted family can reopen only after a new external market-structure reason
or a proven strategy change creates a genuinely different failure mode. A better
number from another nearby variant is not enough.

## Open Families

No research family is open. The sole corrected replay for
`dimension_capped_score_v04_user_authorized` is complete and the family is
exhausted. The invalid first run remains preserved at
`cross_signal_strategy/reports/dimension_capped_score_v04_invalid_implementation_2019_2021.md`;
the corrected approved-rule result is preserved at
`cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md` with
rule fingerprint `0493e7fbeb80cdaa6d8ab0fe9c47d3fa8ca8b680e6556ca805de4d6e742f7f63`.
The corrected replay retained 85/89 closed trades and changed 196 filled-order
days, but failed seven gates: nominal win rate, return, Sharpe, Sortino, and
profit/loss quality, plus doubled-friction return and win rate. Terminal action
is `STOP`; no JoinQuant/PTrade candidate exists and validation influence remains
none. KDJ tier variants, direct extreme exits, further MACD changes, score
rebalance, indicator deletion, nearby thresholds, score floors, ADX changes,
rankings, and ETF/year exceptions remain exhausted and prohibited.

The fixed
`late_macd_boll_upper_filter_user_authorized` candidate completed its official
2019-2021 JoinQuant run and was rejected because win rate stayed at 55.8% while
return and all listed payoff/risk-adjusted metrics worsened. The separately
authorized `late_veto_early_pre_macd_user_authorized` candidate is now blocked
pending one official 2019-2021 JoinQuant run. Its exact 50-59 early-entry rule
does not reopen threshold, age, MACD, BOLL, RSI, ETF, queue, or sell searches.

For historical context, the user-authorized
`bullish_cross_age2_half_decay_user_authorized`
family was consumed on 2026-08-17: the single frozen 0.5 multiplier changed
64 filled-order days across 2019/2020/2021 but reduced return from +125.00% to
+87.35%, worsened drawdown from 6.03% to 8.79%, and worsened every annual
return. The candidate was rejected before JoinQuant or validation; no nearby
coefficient or per-indicator age weighting may be searched. Its design is
documented in
`docs/superpowers/specs/2026-08-17-cross-signal-age2-half-decay-design.md`.

The user-authorized `intraday_signal_clock_1445_user_authorized` family was
consumed on 2026-08-21: the exact 09:35+14:45 candidate failed nine frozen
quality and friction gates (+125.00% to +85.00% return, 6.03% to 7.49%
drawdown, 56.18% to 47.66% win rate, and 31 to 40 positive-to-negative round
trips). It closed without a JoinQuant or PTrade candidate and does not reopen
MACD, KDJ, ADX, direct-sell, ATR, execution-wait, neighboring-time, or
ETF-specific families.

The user-authorized `profit_gated_direct_sell_user_authorized` family
was consumed on 2026-08-16: the Step 0 matrix counterfactual found 0 of 12
variants passing the gates (38/40 score thresholds never fired; 32/35
thresholds clipped the 513050 +34% winner's mid-hold pullback), so the family
closed without a candidate. Its design is documented in
`docs/superpowers/specs/2026-08-16-profit-gated-direct-sell-design.md`.

The user-authorized `intraday_high_anchor_user_authorized` family was
consumed on 2026-08-16: the Step 0 binding observation passed (1604 binding
days, 38 extra triggers) but the Step 1 local A/B failed the gates
(+125.00%→+119.40% return, 6.03%→6.06% drawdown, worse 2019), because the
high anchor clipped the 2019-02-11 159928 winner (2.232 vs 2.666), so the
family closed without adoption. Its design is documented in
`docs/superpowers/specs/2026-08-16-intraday-high-anchor-design.md`.

The user-authorized `profit_giveback_exit_user_authorized` family was
consumed on 2026-08-16: the Step 0 trade-level counterfactual fired 79 times
across 21 affected closed trades but the total per-share delta was negative
(-0.352, annual -0.380/-0.101/+0.129), because it clipped two major winners
(2019-02-11 159928, 2020-04-17 513050) while saving small amounts elsewhere,
so the family closed without a candidate. Its design is documented in
`docs/superpowers/specs/2026-08-16-profit-giveback-exit-design.md`.

The user-authorized `gold_specific_stop_user_authorized` family was
consumed on 2026-08-16: the Step 0 binding observation passed (223 binding
days, 6 extra triggers) but the Step 1 local A/B failed every quality gate
(+125.00%→+120.96% return, 6.03%→6.08% drawdown, worse 2019/2021), so the
candidate was rejected and the family is exhausted. Its design is documented
in `docs/superpowers/specs/2026-08-16-gold-specific-stop-design.md`.

The user-authorized `profit_tiered_atr_user_authorized` family was consumed on
2026-08-16: the Step 0 binding observation passed its 10-event gate (36 events)
but recorded zero same-day extra triggers, and the Step 1 local A/B confirmed
an exact no-op (0 changed filled orders, all metrics identical to
`cross-v0.3.3`), so the candidate was rejected and the family is exhausted. Its
design is documented in
`docs/superpowers/specs/2026-08-16-profit-tiered-atr-design.md`.

`underlying_market_direction` is blocked rather than open. Raw values in
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
