# Strategy Decisions

Use this file to record design decisions before and after each experiment.

## Decision Template

```text
Date:
Decision:
Reason:
Evidence:
Affected files:
Allowed validation influence: training only / multiple validation periods / none
Status: proposed / adopted / rejected / reverted
```

## Initial Decisions

### Isolate The New Strategy

Date: 2026-07-02
Decision: Build the cross-signal strategy in `cross_signal_strategy/` instead of modifying the current production multi-factor strategy.
Reason: The new strategy has a different trading philosophy. Mixing it into the production strategy would make results hard to interpret.
Evidence: User specifically requested a new file and independent folder.
Affected files: `cross_signal_strategy/*`
Allowed validation influence: none
Status: adopted

### Use 2019-2021 As Development Window

Date: 2026-07-02
Decision: Use 2019-01-01 to 2021-12-31 as the development/training period.
Reason: This period includes rising, crash/rebound, and structural-divergence behavior, but keeps other periods reserved for out-of-sample validation.
Evidence: Research protocol agreed in conversation.
Affected files: `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Freeze v0.1 As Cross-Signal Timing, Not Momentum Rotation

Date: 2026-07-02
Decision: First implementation will use RSI/MACD/KDJ cross resonance as the primary buy/sell driver, with MA/BOLL location, light trend context, volume confirmation, and ATR risk control.
Reason: The experiment is meant to test earlier swing turning points, not to retune the existing momentum-led multi-factor strategy.
Evidence: User described repeated live concern that momentum confirmation may enter after a short swing has already risen for several days.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Disable Profit Floor In v0.1

Date: 2026-07-02
Decision: v0.1 will disable profit-floor protection while keeping ATR stop logic.
Reason: The first test should isolate whether cross-signal sell timing has value before adding the current production strategy's profit-protection layer.
Evidence: Profit floor is already validated in production strategy; adding it immediately would obscure whether new sell signals work.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Use Equal-Weight Sizing In v0.1

Date: 2026-07-02
Decision: v0.1 will use equal-weight sizing across selected holdings rather than volatility-inverse sizing.
Reason: The first experiment should evaluate signal quality before adding sizing complexity.
Evidence: User's recent concern includes whether volatility-based sizing can misallocate capital; v0.1 should not inherit that complexity before signal validation.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Implement JoinQuant v0.1 From Frozen Spec

Date: 2026-07-02
Decision: Create `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py` as the first runnable JoinQuant version.
Reason: The strategy needs an isolated file for training-period validation without modifying the production multi-factor strategy.
Evidence: Implementation follows the frozen v0.1 spec: RSI/MACD/KDJ cross resonance, MA/BOLL location, light trend context, volume confirmation, equal-weight sizing, ATR stop, no profit floor.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: none
Status: adopted

### Add Score Skip Diagnostics Before Tuning

Date: 2026-07-02
Decision: Add per-ETF score skip reason diagnostics when v0.1 produces no valid scores.
Reason: The first JoinQuant training log showed no errors but also no valid scores or trades, so the next step must identify the filter/data root cause before changing any strategy thresholds.
Evidence: `cross_signal_strategy/logs/test.log` had 295 rebalance days, 295 `no valid scores`, zero candidates, zero buys, and zero sells.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: training only for debugging, not for parameter tuning
Status: adopted

### Add Full-Pool Cross-Signal Diagnostics

Date: 2026-07-02
Decision: Log full-pool `reversal_score > 0` cross-signal candidates on every rebalance day, even when they are not in the top buy-score candidates.
Reason: The training log showed valid scoring but zero trades because top candidates had `reversal_score=0` and buy scores never reached 60. The next diagnostic must determine whether cross signals are absent from the full ETF pool or merely ranked below non-cross candidates.
Evidence: Latest training log had 295 top-candidate sections, max buy score 41, zero buys, and top displayed candidates with `rev=0`.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: training only for debugging, not for parameter tuning
Status: adopted

### Add Loose Turn-Up Diagnostics

Date: 2026-07-02
Decision: Add observation-only loose reversal diagnostics for RSI6, MACD DIF, and KDJ K/J turning upward.
Reason: The training log showed strict RSI/MACD/KDJ upward crosses were absent across rebalance checks. Before lowering buy thresholds or changing strategy rules, the next step is to see whether indicators are at least turning upward but not crossing by the strict definition.
Evidence: Latest training log had 295 `[cross signals] none in full pool`, zero buys, max buy score 41, and no strict reversal candidates.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: training only for debugging, not for parameter tuning
Status: adopted

### Align Cross Detection With Logged Diff

Date: 2026-07-02
Decision: Rewrite strict cross detection to use the same difference semantics as the diagnostic log: previous difference <= 0 and current difference > 0 for upward crosses, and the inverse for downward crosses.
Reason: The diff diagnostics showed many true crosses, while `reversal_score` stayed at zero and `[cross signals]` reported none. The strategy needs one consistent definition for scoring and diagnostics before any performance judgment.
Evidence: Training log showed examples such as RSI_DIFF and KDJ_DIFF moving from negative to positive while `rev=0`.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: bug fix only; rerun training before any strategy tuning
Status: adopted

### Add Versioned Startup Self Check

Date: 2026-07-02
Decision: Print a startup self-check line with version `cross-v0.1.1`, the diff-cross fix flag, and a minimal reversal-score check.
Reason: JoinQuant logs still showed `rev=0` after the local diff-cross fix, which likely means the platform was running an older pasted strategy. A visible startup marker makes copy/version mismatch immediately detectable.
Evidence: Local pipeline test returns `diff_cross_self_check=True` and `self_rev=12`, while the uploaded JoinQuant log had no version marker and still showed zero reversal scores.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`
Allowed validation influence: diagnostic only; no trading rule change
Status: adopted

### Exclude Buy Candidates With Force-Sell Conflict

Date: 2026-07-02
Decision: New buy candidates must have `sell_score < sell_threshold`; an ETF with simultaneous force-sell resonance is not eligible for fresh buying.
Reason: A single signal snapshot can contain both high buy score and force-sell score. Buying such a conflicted candidate violates the sell rule and makes the training result hard to interpret.
Evidence: Added a failing unit test before implementation: `test_buy_candidates_exclude_force_sell_conflicts`.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for correctness; no validation-period influence
Status: adopted

### Guard Cross Detection Against JoinQuant Global Name Pollution

Date: 2026-07-03
Decision: Cross detection must explicitly use Python built-in `any`, not the module-global `any` name.
Reason: `from jqdata import *` can pollute module globals in the JoinQuant runtime. The training log showed the startup self-check returning `diff_cross_self_check=False expected=True`, and every rebalance day reported no strict cross signals despite logged indicator differences crossing from negative to positive.
Evidence: 2019-2021 training log: `[cross-v0.1.2] ... diff_cross_self_check=False expected=True | self_rev=0`, followed by repeated `[cross signals] none in full pool`. Added failing unit test `test_cross_detection_ignores_jqdata_any_global_pollution` before implementation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for correctness; no validation-period influence
Status: adopted

### Use Latest Cross Direction Within Recent Window

Date: 2026-07-03
Decision: Recent cross detection must use the latest cross direction inside the lookback window. If an upward cross is followed by a downward cross inside the same window, only the downward cross remains active, and vice versa.
Reason: The 2019-2021 training log after the v0.1.3 fix showed many candidates with simultaneous `RSI_UP=True` and `RSI_DOWN=True`, or simultaneous `KDJ_UP=True` and `KDJ_DOWN=True`. This made buy and sell reversal scores both react to stale crosses inside the same recent window.
Evidence: Training-log diagnostic count: 2632 cross detail rows, 611 with at least one same-indicator directional conflict. Added failing unit test `test_recent_cross_detection_uses_latest_cross_direction` before implementation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for correctness; no validation-period influence
Status: adopted

### Expand Cross Flag Logging Detail

Date: 2026-07-03
Decision: Cross-signal detail logs should print each component flag separately: RSI12/RSI24 up and down, MACD up and down, and KDJ K/J up and down.
Reason: The v0.1.4 training log still showed aggregate `RSI_UP=True` and `RSI_DOWN=True` in 54 cross rows. Those remaining cases may come from RSI6 crossing RSI12 and RSI24 in different directions, so aggregate logs are not precise enough to diagnose whether a scoring change is needed.
Evidence: Added failing unit test `test_format_cross_flags_shows_rsi_and_kdj_detail` before implementation. This is observation-only logging and does not change scores, thresholds, ranking, or order execution.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for diagnostics; no validation-period influence
Status: adopted

### Use Net RSI Group Direction For Scoring

Date: 2026-07-03
Decision: RSI scoring first resolves the RSI12/RSI24 group direction. If active RSI crosses contain both up and down directions, RSI contributes zero points to both buy and sell reversal scores. If active RSI crosses agree, the existing per-pair scoring is preserved.
Reason: Mixed RSI group direction is contradictory evidence, not simultaneous buy and sell confirmation. This keeps MACD/KDJ scoring unchanged while preventing RSI from raising both sides during short-term oscillation.
Evidence: The v0.1.5 2019-2021 training log showed 54 RSI group mixed rows, with `rsi12_both=0`, `rsi24_both=0`, `kdj_mixed_group=0`, and `macd_both=0`. Added failing tests `test_buy_score_ignores_mixed_rsi_group_direction` and `test_sell_score_ignores_mixed_rsi_group_direction` before implementation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for correctness; no validation-period influence
Status: adopted

### Add Buy-Side Widening Confirmation Signals

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.0` by adding buy-side confirmation points when a fast line is already above its slow line and the positive difference is widening. Confirmation points use half of the strict-cross weight: RSI pair +6, MACD +5, KDJ K +3, and KDJ J +2. Strict crosses keep their original full weight and do not double-count confirmation points.
Reason: The v0.1.6 training result showed correct cross detection but modest return. A strict-cross-only entry can miss ETF trends after the crossing day. Widening positive differences capture continued strengthening without lowering thresholds or using validation-period feedback.
Evidence: Added failing tests `test_buy_score_adds_half_weight_for_widening_positive_confirmations` and `test_buy_score_does_not_double_count_confirmations_after_strict_cross` before implementation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only for structural research; no validation-period influence
Status: reverted by `cross-v0.2.1`

### Revert Widening Confirmation And Use Daily Event Evaluation

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.1` by reverting the failed widening-confirmation buy-score booster and evaluating signals on every trading weekday instead of only Tuesday and Thursday.
Reason: The cross-signal strategy is event-driven: crosses can occur on any trading day. Fixed weekday rotation belongs to ranking/rotation strategies and can delay valid cross-signal exits or entries. The widening-confirmation booster failed in training and should not remain in the baseline.
Evidence: `cross-v0.2.0` 2019-2021 training result fell to +19.99% versus `cross-v0.1.6` +32.39%, with higher drawdown. Added failing tests before implementation: `test_buy_score_does_not_add_widening_positive_confirmations_without_cross`, `test_default_params_evaluate_signals_every_trading_weekday`, and the version self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/failed_experiments.md`
Allowed validation influence: training only; no validation-period influence
Status: adopted

### Require Low-Position Eligibility For New Buys

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.2` by requiring new buy candidates to have a reasonable price-position setup: BOLL lower-to-middle region, a recent cross back above BOLL middle, or price near MA20. Candidates clearly far above MA20 are not eligible for new entry even if their cross score reaches the buy threshold.
Reason: `cross-v0.2.1` confirmed daily event evaluation works mechanically, but naked daily response amplified noise. Training-period trades rose from `cross-v0.1.6` 145 buys / 145 sells to 222 buys / 214 sells, while return fell from +32.39% to +23.18%. The strategy goal is low-position reversal buying, not chasing every daily cross.
Evidence: Added failing tests before implementation: `test_buy_candidates_require_low_position_for_new_entries`, `test_buy_candidates_accept_ma20_repair_position_for_new_entries`, and the `cross-v0.2.2` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: adopted

### Confirm Signal Sells With Price Structure

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.3` by requiring signal-based sells to have both `sell_score >= sell_threshold` and price-structure confirmation. Confirmation can come from price below MA20, price below BOLL middle, close below a falling MA10, downside continuation, or a high-position RSI turn-down. ATR stops remain unconditional.
Reason: The `cross-v0.2.2` training log showed all 212 sells were `sell_score` sells, while buy count barely fell versus `cross-v0.2.1`. This suggests daily down-cross signals are still acting as short-term noise exits. Mature versions in this repository use noise-control ideas such as hold protection and structure checks; the suitable idea to borrow here is not the old rotation framework, but requiring a real price-structure break before a normal signal sell.
Evidence: Added failing tests before implementation: `test_signal_sell_requires_structure_confirmation`, `test_signal_sell_confirmed_by_ma20_break`, `test_atr_stop_sells_without_signal_confirmation`, and the `cross-v0.2.3` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: adopted

### Convert Risk-Tighten Warning Into Tighter ATR Stop

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.4` by making `risk-tighten` set a per-position tightened stop state instead of only logging. Normal ATR stop remains `2.5x` with a 5% floor; risk-tightened positions use `1.5x` ATR with a 3% floor. Signal sells still require structure confirmation, and ATR stops remain the only unconditional exit.
Reason: The `cross-v0.2.3` training log showed 362 `risk-tighten` warnings. This is useful information that should improve protection after down-cross risk appears, but turning warnings directly into sells would reintroduce the noise problem v0.2.3 fixed. Tightening the stop borrows the suitable risk-management idea from mature strategy versions while preserving the cross-signal framework.
Evidence: Added failing tests before implementation: `test_risk_tightened_stop_price_is_higher_than_normal_stop`, `test_check_atr_stops_uses_risk_tightened_state`, `test_risk_tightened_state_is_created_when_missing`, and the `cross-v0.2.4` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: archived by `cross-v0.2.5`; keep as a future combination factor, not a mainline rule

### Archive Risk-Tighten ATR And Restore Mainline Stop

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.5` by functionally restoring the `cross-v0.2.3` mainline: `risk-tighten` remains an observation-only warning, and ATR stops use the normal `2.5x` multiplier with the 5% floor. The failed `cross-v0.2.4` risk-tighten ATR idea is retained in research notes as a future combination factor.
Reason: The user correctly noted that a single factor can fail alone but work in combination. The right handling is to keep the production/research mainline clean while preserving the factor for later controlled combination tests, especially with ADX/DMI trend-vs-chop detection.
Evidence: `cross-v0.2.4` 2019-2021 training fell to +39.44% from `cross-v0.2.3` +44.15%, while max drawdown improved only from 6.43% to 6.24%. Added failing tests before rollback: `test_risk_warning_does_not_change_mainline_stop_price`, `test_check_atr_stops_ignores_archived_risk_tightened_state`, and the `cross-v0.2.5` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/failed_experiments.md`
Allowed validation influence: training only; no validation-period influence
Status: adopted

### Add ADX/DMI Strong-Trend Sell Protection

Date: 2026-07-03
Decision: Upgrade to `cross-v0.2.6` by adding standard DMI/ADX(14). ADX/DMI is used only to protect strong upward trends from non-severe signal sells: if ADX >= 25, +DI > -DI, and MA20 slope is non-negative, a sell signal caused only by softer structure such as BOLL-middle weakness is blocked. Severe structure breaks such as close below MA20, close below falling MA10, or downside continuation still sell normally.
Reason: The strategy's core problem is noisy daily down-crosses during otherwise healthy trends. ADX/DMI is a standard trend-strength tool and fits this specific role better than blindly tightening ATR. This keeps the cross-signal framework while borrowing a mature trend/noise distinction.
Evidence: Added failing tests before implementation: `test_dmi_adx_identifies_directional_uptrend`, `test_strong_adx_uptrend_blocks_nonsevere_signal_sell`, `test_strong_adx_uptrend_does_not_block_severe_structure_sell`, the DMI log assertion, and the `cross-v0.2.6` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: adopted

### Use ADX/DMI Regime For Buy Eligibility

Date: 2026-07-04
Decision: Upgrade to `cross-v0.2.7` by applying ADX/DMI to new-buy eligibility without changing buy scores. In strong ADX uptrends, candidates above MA20 but not far above MA20 are eligible even if they are not in the original low-position bucket. In strong ADX downtrends, MA20-only repair entries are rejected; candidates must be in BOLL lower-to-middle position or have crossed back above the BOLL middle.
Reason: `cross-v0.2.6` improved training-period return by using ADX/DMI on sells. The next structurally consistent step is to let ADX/DMI distinguish trend-continuation entries from weak-trend repairs on the buy side, while avoiding a broad score boost that could become parameter fitting.
Evidence: Added failing tests before implementation: `test_buy_candidates_accept_nonextended_strong_adx_uptrend_entry`, `test_buy_candidates_reject_ma20_only_entry_in_strong_adx_downtrend`, and the `cross-v0.2.7` self-check expectation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: archived; no material improvement versus `cross-v0.2.6`

### Add Low-Position Weak-Buy Supplement

Date: 2026-07-04
Decision: Upgrade to `cross-v0.2.8` by allowing weak buy candidates with `buy_score >= 55` only when they have low-position reversal quality. The supplement requires BOLL lower-to-middle position or a cross back above BOLL middle, plus `reversal_score >= 35`, no overheat block, no sell-score conflict, and no far-above-MA20 condition. MA20-only repair entries still require the normal `buy_threshold` and do not qualify for weak-buy supplementation.
Reason: The `cross-v0.2.6` training log showed 249 no-buy days and high cash near the end of 2021, suggesting the next improvement should target missed low-position entries rather than further sell protection. The rule is deliberately narrow so it does not become a broad threshold cut or trend-chasing rule.
Evidence: Added failing tests before implementation: `test_weak_buy_candidate_accepts_low_position_reversal_quality` and `test_weak_buy_candidate_rejects_high_position_low_reversal_and_sell_conflict`, then implemented the minimal weak-buy gate and updated the startup self-check to `cross-v0.2.8`.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; no validation-period influence
Status: archived; training result worse than `cross-v0.2.6`

### Isolate Local Training Data Root

Date: 2026-07-04
Decision: Local `cross_signal_strategy` training, debugging, and structural tuning must use only `G:\financial\history_data\cross_signal_train_2019_2021`, and that folder is read-only. Scripts must not modify, overwrite, clean in place, delete, or generate derived files inside it. Delete/remove commands must never target this folder or any file below it.
Reason: The source directory `G:\financial\history_data\按年份合并` contains many years of minute and daily market data. Keeping training work pointed at a separate 2019-2021-only copy reduces accidental validation-period leakage and keeps the research protocol enforceable.
Evidence: Created `G:\financial\history_data\cross_signal_train_2019_2021` with only 12 target ETFs for 2019, 2020, and 2021. Integrity check found 12 one-minute files and 12 daily files for each year, no missing `09:35` rows, no non-positive minute prices, and no basic OHLC errors.
Affected files: `AGENTS.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none
Status: adopted

### Enforce No Future Functions And No Overfitting

Date: 2026-07-04
Decision: Cross-signal research must strictly prevent look-ahead bias and overfitting. Daily signals may use only T-1 and earlier data; T-day 09:35 minute data may be used only for execution pricing or explicitly documented intraday execution filters. Independent pre-2019 warm-up daily bars may be used only as an indicator lookback buffer for 2019-2021 training signals. Validation, full-period, or final-summary results must not be used to tune parameters, choose indicators, change thresholds, or select rules.
Reason: The strategy goal is a robust low-position cross-signal framework, not a historical fit. Local minute data makes it easier to accidentally mix decision-time and execution-time information, so the rule must be explicit before building the local backtester.
Evidence: User explicitly requested hard rules to prevent future functions and overfitting before continuing local backtest implementation.
Affected files: `AGENTS.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none
Status: adopted

### Allow Independent Pre-Training Warm-Up Daily Bars

Date: 2026-07-04
Decision: Add `G:\financial\history_data\cross_signal_warmup_2018` as an independent read-only daily-bar warm-up source. It may only be used to compute rolling indicators for early 2019 signals. It must not be used for performance statistics, parameter tuning, rule selection, validation, or execution prices.
Reason: A professional daily strategy needs historical lookback before the first evaluated trading day. JoinQuant `get_price(..., count=120, end_date=T-1)` naturally pulls pre-2019 bars for early-2019 signals. Without warm-up, local replay creates artificial `short_data` gaps and diverges mechanically from JoinQuant.
Evidence: User asked whether January 2019 backtests should query part of 2018 history; analysis concluded this is correct as a warm-up buffer, not as training/evaluation data.
Affected files: `AGENTS.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none
Status: adopted

### Summarize And Commit Milestones

Date: 2026-07-04
Decision: After a clear implementation or research milestone passes its relevant tests/checks, summarize the current version's scope, verification result, risks, and next step, then create a commit as a rollback point. If analysis shows the milestone is safe to commit, the agent may commit without separate user confirmation, but must report the commit summary, verification, and remaining risks.
Reason: The cross-signal project is iterative and can involve many small experiments. Milestone commits prevent unrelated changes from accumulating and make it easy to return to a known-good state if later experiments fail. Allowing autonomous milestone commits avoids unnecessary interruptions once the safety criteria are met.
Evidence: User requested a rule that after comprehensive analysis reaches a milestone, the agent can commit without asking again, as long as it summarizes the result.
Affected files: `AGENTS.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none
Status: adopted

### Require Price Decline For Falling-MA10 Sell Confirmation

Date: 2026-07-05
Decision: Refine `close_below_falling_ma10` so it requires the latest close to be at or below the previous close, below MA10, and MA10 below its previous value. A price that rebounds but remains under a gently falling MA10 is a risk-tighten warning, not a force-sell structure break; a flat weak close under a falling MA10 remains a valid structure break.
Reason: The JoinQuant 2019-2021 training log showed `159928` on 2019-11-13 as `sell_score 24` and risk-tighten only, while local replay scored it as 34 and sold early because it counted `close_below_falling_ma10` even though the latest close rose from 3.044 to 3.057. The later 2019-11-18 sell still triggers when price actually declines and breaks weaker structure.
Evidence: Added failing tests `test_below_falling_ma10_requires_price_to_decline` and `test_below_falling_ma10_accepts_flat_weak_close` before implementation. Local signal check now matches the log pattern: 2019-11-13 `159928` is `sell=24 force=False`; 2019-11-18 `159928` is `sell=32 force=True`; 2019-09-30 `513500` is `sell=44 force=True`.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: superseded by "Use JoinQuant Exact Falling-MA10 Confirmation"

### Use JoinQuant Exact Falling-MA10 Confirmation

Date: 2026-07-05
Decision: `close_below_falling_ma10` should match the JoinQuant platform code exactly: latest close below MA10 and current MA10 below prior MA10 (`ma10[-1] < ma10[-2]`). It should not require the latest close to be below or equal to the prior close, and it should not add a local-only floating tolerance.
Reason: Full JoinQuant/local score alignment showed that the price-decline requirement was too strict for `cross-v0.2.6`. In the JoinQuant training log, 2020-10-27 `513050` sold with `sell_score 45`; local scored only 35 because the close rose slightly from 2.078 to 2.080 while still below a clearly falling MA10. The earlier 2019-11-13 `159928` case is better explained by a nearly flat MA10 floating-point artifact, not by a general price-decline requirement.
Evidence: Added failing tests before implementation. The final test `test_below_falling_ma10_matches_joinquant_exact_less_than_comparison` proves that even a tiny MA10 decrease is counted, matching the uploaded JoinQuant platform code. Full source diff against the uploaded JoinQuant platform code returned zero lines. Prior alignment evidence remains: JoinQuant/local rich-row sell-score mismatches fell from 294 to 12 rows; 2020-10-27 `513050` and 2019-11-13 `159928` both match. In filled-order path comparison, JoinQuant has 262 events and local has 260; after filtering the previously diagnosed 2020-09-22/2020-09-23 `512100` KDJ boundary pair, the remaining 260 local filled events match the 260 filtered JoinQuant events exactly.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Suppress Local Sub-Float Falling-MA10 Artifact

Date: 2026-07-06
Decision: Keep `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py` byte-aligned with the uploaded JoinQuant platform source, but let `LocalSignalAdapter` suppress `close_below_falling_ma10` only when the local MA10 decrease is a sub-float artifact (`0 < ma10[-2] - ma10[-1] < 1e-12`).
Reason: After exact source alignment, local replay treated 2019-11-13 `159928` as `close_below_falling_ma10=True` because local Pandas computed `MA10_prev - MA10_latest = 4.44e-16`. The JoinQuant log for the same platform source shows only `risk-tighten sell_score 24`, not a sell, so this is a local replay precision artifact rather than a strategy rule.
Evidence: Added failing test `test_signal_score_suppresses_sub_float_falling_ma10_artifact` before implementation. After the adapter fix, 2019-11-13 `159928` scores `sell_score=24` and does not sell locally. Full local replay against the 2019-2021 JoinQuant log returns to the expected state: JoinQuant 262 filled events, local 260; first unfiltered divergence is 2020-09-22 BUY `512100`; after filtering the known 2020-09-22/2020-09-23 `512100` boundary pair, the remaining 260 events match exactly.
Affected files: `cross_signal_strategy/local_signal_adapter.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Local Adjustment Factor Alignment

Date: 2026-07-05
Decision: Apply 2019-2021 target-ETF adjustment factors inside the local cross-signal training replay so T-1 signal OHLC matches JoinQuant's adjusted historical price口径 on ex-dividend/split dates.
Reason: The remaining JoinQuant/local close outliers were not bad raw data. They occurred exactly on ETF ex-dates: `510880` on 2020-01-17 and `510300` on 2021-01-18. Dividing the previous signal close by the same-day `ex_factor` matches JoinQuant's logged close to rounding precision.
Evidence: `510880` local 2020-01-16 close `2.947 / 1.0513740030198886 = 2.8029987`, matching JoinQuant `2.803`; `510300` local 2021-01-15 close `5.526 / 1.0132002506617996 = 5.4540058`, matching JoinQuant `5.454`. Tests added before implementation and passed with `uvx --with pandas pytest`; full local training replay test passed.
Protocol guard: The replay uses a small 2019-2021 target-ETF factor table and does not read the full `G:\financial\history_data\按年份合并` market-data directory during normal training replay. Only events on or before the current decision date are applied; future factor events are not applied early.
Affected files: `cross_signal_strategy/local_adjustment.py`, `cross_signal_strategy/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Use ETF Tick Precision For Local ATR Stop Comparison

Date: 2026-07-05
Decision: In local replay, compare ATR stop trigger prices at 0.001 ETF quote precision. Execution price and signal scoring remain unchanged.
Reason: JoinQuant logs ATR stop comparisons at ETF quote precision. On 2020-03-02, local computed `518880` stop as 3.53875 and used a 09:35 price of 3.539, missing the stop by a sub-tick float difference, while JoinQuant triggered the stop as `3.538<=3.539`.
Evidence: Added failing test `test_planner_atr_stop_uses_etf_tick_precision_for_trigger` before implementation. After the fix, local replay sells `518880` on 2020-03-02, matching JoinQuant order timing.
Affected files: `cross_signal_strategy/local_order_planner.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Allow Core-Indicator-Ready New Listings Before MA60

Date: 2026-07-05
Decision: Local replay no longer requires 60 daily bars before scoring a newly listed ETF. It requires enough history for the core required indicators (RSI24, MACD, BOLL20, ATR14, ADX14) and allows MA60 to be NaN, which naturally gives no MA60 trend contribution.
Reason: JoinQuant scored and bought `159985` on 2020-03-03 with 56 available bars and `MA60=nan`, producing `buy=70 trend=0`. Local replay incorrectly skipped it as `short_data:56<60`.
Evidence: Added failing test `test_signal_score_allows_listing_before_ma60_when_core_indicators_are_valid` before implementation. Local replay now scores `159985` on 2020-03-03 as `buy=70 trend=0 sell=0`.
Affected files: `cross_signal_strategy/local_signal_adapter.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Correct Confirmed Local Daily Bar Defect

Date: 2026-07-06
Decision: Apply confirmed local daily-bar defects through an external read-time correction layer, without modifying the read-only training source data. The first adopted correction is `512100` on 2020-09-02: close `1.000` in the local daily CSV is overridden to `1.001` for local signal replay.
Reason: JoinQuant diagnostic logs, local 1-minute aggregation, and the user's trading software all showed `512100` 2020-09-02 close `1.001`; only the local daily CSV showed `1.000`. This bad close shifted KDJ's zero-cross boundary and caused the last JoinQuant/local path divergence on 2020-09-22/2020-09-23.
Evidence: Added failing tests before implementation. After the correction layer, `512100` on 2020-09-22 scores `buy=65 rev=35` with both KDJ up-cross flags true, matching JoinQuant. Full local replay against the latest JoinQuant log has 262 local filled events versus 262 JoinQuant filled events and no order-path divergence. Full cross-signal test suite passed with 85 tests.
Affected files: `cross_signal_strategy/local_adjustment.py`, `cross_signal_strategy/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_local_signal_adapter.py`, `tests/test_cross_signal_local_training_run.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log and confirmed source-data correction only; no validation-period influence
Status: adopted

### Use ETF Tick Precision For Local Execution Prices

Date: 2026-07-06
Decision: Local replay should round slippage-adjusted execution prices to ETF tick precision (`0.001`) before applying cash, commission, and position updates.
Reason: JoinQuant strategy uses `PriceRelatedSlippage(0.001)`, but ETF market orders still fill at quoted ETF price precision. The prior local broker kept sub-tick prices such as `3.06306`, which is not a tradable ETF price and caused extra cash/position drift.
Evidence: Added failing tests before implementation and updated local broker expectations from sub-tick prices to tick-rounded prices. Full order path remains aligned with JoinQuant at 262 events versus 262 events, with no first divergence. Full cross-signal test suite passed with 88 tests. The remaining return gap is still mostly JoinQuant internal market-order matching price and rolling share-quantity drift, so no further strategy-rule change is justified.
Affected files: `cross_signal_strategy/local_backtester.py`, `cross_signal_strategy/order_path_diagnostics.py`, `tests/test_cross_signal_local_backtester.py`, `tests/test_cross_signal_order_path_diagnostics.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log execution diagnostics only; no validation-period influence
Status: adopted

### Raise Cross-Signal Base Ratio To 0.90

Date: 2026-07-07
Decision: Set the cross-signal default `base_ratio` to 0.90 instead of the inherited initial value 0.75.
Reason: After local/JoinQuant order-path alignment, the frozen training baseline showed average exposure of only 59.74%. Position-count diagnostics showed this was not mostly caused by missing candidates: the strategy held the full 3 positions on 443 of 730 training days, but a 0.75 base ratio means even a full book uses only about 75% of account value. Raising base usage is a simple capital-utilization policy, not a new indicator or narrow signal threshold.
Evidence: Training-only sweep with unchanged signals and unchanged trade path: base_ratio 0.75 returned +45.45% local replay, annualized +13.34%, max drawdown 7.67%; 0.80 returned +49.27%, annualized +14.33%, max drawdown 8.07%; 0.85 returned +53.19%, annualized +15.32%, max drawdown 8.44%; 0.90 returned +57.87%, annualized +16.49%, max drawdown 8.85%; 0.95 returned +62.30%, annualized +17.57%, max drawdown 9.20%. The 0.95 candidate was not adopted because it is near full exposure and more likely to reflect training-period aggressiveness. The adopted 0.90 keeps a 10% cash buffer while reaching the initial 16%-17% annualized target in local training replay.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; validation periods were not inspected
Status: adopted

### Add One-Week Minimum Hold For Normal Signal Sells

Date: 2026-07-07
Decision: Add `min_signal_hold_days=5` for normal signal-based sells. ATR stops remain unconditional and can sell before the minimum hold period.
Reason: After the base-ratio improvement, trade-reason diagnostics showed ATR exits were strong while normal signal sells were noisy. On the 2019-2021 training replay with `base_ratio=0.90`, ATR exits produced +9670.9 realized PnL with P/L ratio 5.766, while normal signal sells produced only +1853.6 with P/L ratio 1.186. A one-week minimum hold is a standard anti-noise rule with clear market meaning; it filters short-term cross reversals without weakening hard risk control.
Evidence: Training-only coarse sweep, with unchanged buy rules and ATR stop logic: no minimum hold returned +57.87%, annualized +16.49%, max drawdown 8.85%; 3 trading days returned +70.73%, annualized +19.58%, max drawdown 9.08%; 5 trading days returned +98.34%, annualized +25.72%, max drawdown 8.94%; 7 trading days returned +98.94%, annualized +25.85%, max drawdown 9.71%. The 5-day rule was adopted instead of 7 days because it has clearer one-week market meaning and similar return with lower drawdown.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local_order_planner.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_strategy.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; validation periods were not inspected
Status: adopted

### Raise Cross-Signal Base Ratio To 0.95 After Sell-Noise Filter

Date: 2026-07-07
Decision: After adopting the one-week normal-signal minimum hold, raise the cross-signal default `base_ratio` from 0.90 to 0.95.
Reason: The sell-noise filter materially improved trade quality and lowered churn, so the sizing policy was rechecked at coarse, market-meaningful levels only. `0.95` is a common maximum-capital-use policy with a 5% cash buffer; no fine-grained ratios above 0.95 were tested.
Evidence: Training-only sweep with unchanged signals and `min_signal_hold_days=5`: `base_ratio=0.85` returned +90.13%, annualized +23.96%, max drawdown 8.58%; `base_ratio=0.90` returned +98.34%, annualized +25.72%, max drawdown 8.94%; `base_ratio=0.95` returned +106.17%, annualized +27.36%, max drawdown 9.35%. Trade count and signal path were unchanged at 103 buys and 101 sells.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training only; validation periods were not inspected
Status: adopted

### Add Local Risk Metrics To Baseline Report

Date: 2026-07-07
Decision: Extend `BaselineReport` with daily win rate, annualized volatility, Sharpe ratio, and Sortino ratio.
Reason: The user asked for metrics commonly shown in JoinQuant, and future experiments need a richer comparison set than return and max drawdown alone.
Evidence: Added failing tests before implementation. Current training mainline reports: +106.17% return, +27.36% annualized return, 9.35% max drawdown, 53.47% closed-trade win rate, 53.70% daily win rate, 13.29% annualized volatility, Sharpe 1.8866, Sortino 2.9313.
Affected files: `cross_signal_strategy/baseline_report.py`, `tests/test_cross_signal_baseline_report.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; reporting only
Status: adopted

### Add Entry-Score Trade Diagnostics

Date: 2026-07-07
Decision: Add a dedicated trade-diagnostics module that captures buy-entry score snapshots at order-planning time and uses those snapshots for closed-trade attribution.
Reason: A temporary attribution script had read `planner.last_scores` after later days had already refreshed it, which could mislabel historical buys. The issue did not affect local replay returns, positions, or daily equity, but it could mislead factor diagnostics.
Evidence: Added failing-style tests for score snapshot capture and closed-trade attribution before implementation. Current 2021Q3 diagnostic using the formal tool confirms the same PnL conclusion: sell-date Q3 trades PnL -1579.5; buy-date Q3 trades PnL -1445.1; no-volume-confirmation Q3 buys were 5/5 losers with PnL -2000.9.
Affected files: `cross_signal_strategy/trade_diagnostics.py`, `tests/test_cross_signal_trade_diagnostics.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: none; reporting only
Status: adopted

### Sync JoinQuant State Only After Actual Position Change

Date: 2026-07-08
Decision: In the JoinQuant cross-signal strategy, update buy/sell internal state only after the portfolio position actually reflects the order result. A sell order keeps ATR, buy date, and last-score state if the position still exists; a buy order writes ATR and buy-date state only if a position exists after the order call.
Reason: The 2019-2021 JoinQuant transaction export contained one canceled order: `513880.XSHG` on 2019-12-12. The strategy emitted a sell signal, but JoinQuant reported zero current volume and canceled the market sell. The old code cleared `highest_since_buy`, `entry_atr`, `buy_date`, and `last_scores` immediately after `order_target(code, 0)`, treating a submitted order as a completed fill. That is incorrect under JoinQuant's actual matching behavior and could weaken later ATR risk control for still-held positions.
Evidence: Added failing tests before implementation for three cases: canceled sell must keep state, completed sell must clear state, and unfilled buy must not create buy state. A follow-up regression test verifies that `has_position` iterates actual holdings instead of calling `positions.get`, avoiding JoinQuant missing-position compatibility warnings after completed sells. `python -m pytest tests/test_cross_signal_strategy.py tests/test_cross_signal_order_path_diagnostics.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py -q` passes, and `py_compile` passes for the strategy and test file.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training-period JoinQuant order/transaction diagnostics only; no validation-period influence
Status: adopted

### Treat 513880 Zero-Volume Cancellation As Execution Risk

Date: 2026-07-08
Decision: Do not change cross-signal buy/sell rules based on the 2019-12-12 `513880.XSHG` canceled sell. Record it as an execution-liquidity risk: JoinQuant did not mark the ETF as paused, but the exact order minute had zero tradable volume, so the market sell could not be matched. Keep the state-sync protection for unfilled orders and avoid adding a broad `volume == 0` trading rule without separate evidence.
Reason: A zero-volume market-order cancellation can be caused by sparse minute liquidity rather than a halted security or a wrong signal. Using current-minute volume as a strategy gate would mix execution-time microstructure into a daily signal framework and could change the training path for reasons unrelated to signal quality.
Evidence: A temporary JoinQuant probe file (`cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_probe_513880.py`) printed `paused=False` at `2019-12-12 09:35`, `10:35`, and `14:50`. The same minutes had `volume=0` and `money=0`. The full-day minute summary for `2019-12-12` showed `total_minutes=240`, `nonzero_minutes=26`, `total_volume=1405700.0`, `total_money=1539142.0`, `first_nonzero=2019-12-12 09:38:00`, and `last_nonzero=2019-12-12 14:57:00`. This proves the ETF was not considered paused by JoinQuant and was not zero-volume all day; it traded sparsely, and the 09:35 sell minute was one of the zero-volume minutes.
Affected files: `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: training-period execution diagnostics only; no signal or parameter tuning
Status: adopted

### Half-Size A-Share Buys Without Volume Confirmation

Date: 2026-07-08
Decision: When a new-buy candidate is an A-share ETF (`510300`, `159915`, `512100`, `159928`, `510880`) and its `volume_score` is `0`, size the new buy at 50% of the normal per-slot target. Do not apply this rule to cross-market or cross-asset ETFs.
Reason: In A-share ETFs, a reversal signal without volume confirmation is more likely to be a low-quality repair attempt. For QDII/cross-market and commodity ETFs, local volume has different microstructure and earlier global volume rules damaged returns. A 50% scale is a broad risk-control rule; it deliberately avoids the training-best 25% scale to reduce overfitting risk.
Evidence: Training-only 2019-2021 local replay: baseline returned +106.17%, annualized +27.36%, max drawdown 9.35%, Sharpe 1.887, Sortino 2.931, 2021Q3 -5.32%. A-share-only zero-volume scale `0.50` returned +109.19%, annualized +27.98%, max drawdown 7.86%, Sharpe 1.995, Sortino 3.113, 2021Q3 -4.25%, with the same 103/101 buy/sell event count. Cross-market and cross-asset scaling reduced return and was rejected. JoinQuant 2019-2021 training confirmation returned +115.41%, annualized +30.06%, max drawdown 6.93%, Sharpe 2.870, with 102 buy logs, 101 sell logs, and transaction export status 202 fully filled plus one known `513880.XSHG` zero-volume cancellation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local_order_planner.py`, `tests/test_cross_signal_strategy.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: none; validation periods were not inspected
Status: adopted and JoinQuant-training-confirmed

### Adopt JoinQuant-Confirmed 9-ETF Pool For Cross-Signal Mainline

Date: 2026-07-08
Decision: Promote the training-confirmed ETF-pool candidate into the official cross-signal JoinQuant mainline as `cross-v0.3.1`. Remove `510300.XSHG`, `510880.XSHG`, and `159920.XSHE` from `get_default_etf_pool()`. Keep `159915.XSHE`, `512100.XSHG`, `159928.XSHE`, `513100.XSHG`, `513500.XSHG`, `513880.XSHG`, `513050.XSHG`, `518880.XSHG`, and `159985.XSHE`.
Reason: Training attribution showed `510880` and `159920` were the clearest drag symbols in 2019-2021 and `510300` was a weak contributor under this cross-signal framework. A temporary JoinQuant candidate file confirmed that the 9-ETF pool improved the current official training result across return, annualized return, drawdown, Sharpe, Sortino, win rate, and profit/loss ratio.
Evidence: Current official `cross-v0.3.0` JoinQuant training result after the A-share zero-volume half-size rule: +115.41% return, +30.06% annualized return, 6.93% max drawdown, Sharpe 2.022, Sortino 2.870, win rate 0.530, profit/loss ratio 3.597. Candidate `cross-v0.3.0-pool-candidate`: +120.42% return, +31.08% annualized return, 6.82% max drawdown, Sharpe 2.097, Sortino 2.960, win rate 0.552, profit/loss ratio 4.263. Log/transaction checks found no removed-symbol trades, no runtime errors, and only the known `2019-12-12 513880.XSHG` sparse-liquidity zero-volume cancellation.
Risk: ETF-pool deletion is more exposed to training-window selection bias than a pure execution or risk-control fix. This decision is allowed only as a training-mainline promotion; it is not validation approval and must be checked on reserved periods after the rule set is frozen.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; validation periods were not inspected
Status: adopted and JoinQuant-training-confirmed

### Promote Entry-Combo Filter Candidate To Cross-Signal Mainline

Date: 2026-07-10
Decision: Promote the frozen `cross-v0.3.1-combo-candidate` entry filter into the official cross-signal JoinQuant mainline as `cross-v0.3.2`. The rule blocks new buys where RSI crosses up and MACD crosses up, KDJ does not cross up, volume confirmation is positive, and trend score is positive but below strong-trend level (`0 < trend_score < 20`).
Reason: Training attribution identified this combination as a weak repair-bounce entry pattern: it has some reversal and volume confirmation, but lacks KDJ timing confirmation and is not a strong trend continuation. The rule is intentionally narrow and uses existing signal fields rather than introducing new indicators or fine-grained thresholds.
Evidence: JoinQuant training 2019-2021 improved from official `cross-v0.3.1` +122.47% return, +31.50% annualized, 6.38% max drawdown, Sharpe 3.057, Sortino 0.759, win rate 0.552, profit/loss ratio 4.466 to combo candidate +125.82% return, +32.18% annualized, 6.70% max drawdown, Sharpe 3.109, Sortino 0.799, win rate 0.558, profit/loss ratio 4.845. Reserved validation supported the candidate in 2022-2023 (+17.36% vs +15.49%, max drawdown 11.63% vs 13.38%), 2024-2026 (+58.17% vs +56.99%, max drawdown 9.98% vs 10.65%), and 2010-2014 (+1.20% vs -0.61%, limited-pool supplement). The 2015-2018 stress window was mixed but not failed: +23.21% vs +23.58%, with slightly better max drawdown, win rate, profit/loss ratio, and fewer trades.
Risk: This is a learned entry-quality filter, so it carries more overfitting risk than execution bug fixes. Mitigations: the candidate was frozen before reserved validation, was checked across multiple distinct windows, and is narrow enough to avoid reshaping the whole strategy.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `tests/test_cross_signal_strategy.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/validation_summary.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: validation periods were used only for adoption judgment after the rule was frozen; no retuning or expansion from validation results
Status: adopted after frozen training and reserved validation checks

### Prepare Low-Bounce Entry Filter Candidate

Date: 2026-07-10
Decision: Create a JoinQuant candidate file `cross-v0.3.2-low-bounce-candidate` that keeps the official `cross-v0.3.2` logic unchanged except for one additional entry filter. The candidate blocks new buys where RSI and KDJ cross up, MACD does not cross up, the price is in the BOLL lower-to-middle/near-MA20 repair zone, volume confirmation is positive, and trend score is positive but below strong-trend level (`0 < trend_score < 20`).
Reason: Training-only buy attribution on `cross-v0.3.2` showed the weakest repeatable entry combo was `kdj_up+low_location+rsi_up+trend_support+volume_confirmed`, with 8 closed trades, -431.60 local realized PnL, 25% win rate, and 0.71 profit/loss ratio. This pattern has a plausible market interpretation: a low-position volume bounce with RSI/KDJ timing, but without MACD confirmation and without strong trend support, can be a false rebound.
Evidence: Local 2019-2021 same-mouth replay improved from official mainline +118.75% return, 6.81% max drawdown, 99 buys, 96 sells, end value 43749.40 to candidate +122.49% return, 7.20% max drawdown, 95 buys, 92 sells, end value 44498.60.
Risk: This is an entry-quality filter learned from training attribution and therefore has overfitting risk. The local improvement comes with higher drawdown and fewer trades, so it must be treated as a candidate only. JoinQuant training confirmation is required before any reserved validation or adoption discussion.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py`, `tests/test_cross_signal_low_bounce_candidate.py`, `cross_signal_strategy/attribution_diagnostics.py`, `tests/test_cross_signal_attribution_diagnostics.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; validation periods must not be inspected unless the candidate is first frozen after JoinQuant training confirmation
Status: rejected after JoinQuant 2019-2021 training confirmation. The candidate improved win rate, profit/loss ratio, and Sharpe slightly, but reduced return and annualized return while worsening max drawdown and Sortino. It must not proceed to reserved validation.

### Add Training-Window Stability Diagnostics

Date: 2026-07-10
Decision: Add a standalone `training_stability.py` diagnostic module for official `cross-v0.3.2`. It reports annual performance, profit concentration, exit-reason quality, trading-day holding periods, T-1 entry trend/volatility groups, average exposure, and a complete doubled-friction replay. It does not change the strategy or production files.
Reason: Repeated entry and exit candidates showed that a higher local win rate can hide lost payoff opportunities. Before searching for another indicator or threshold, the training return must be checked for year, trade, ETF, exit, and market-state concentration.
Evidence: The local 2019-2021 baseline returned +118.75% with 6.81% max drawdown and 0.732 average exposure. Calendar-year returns were +35.68% in 2019, +48.45% in 2020, and +8.60% in 2021. The largest winning trade contributed 8.32% of gross profit, the top three contributed 23.51%, and the largest ETF contribution was 22.00%, so the result was not supported by one trade or one ETF. A complete 2x commission/minimum-commission/slippage replay still returned +100.33%, but the 18.42 percentage-point reduction shows meaningful friction sensitivity.
Interpretation: Strong-trend entries produced +14847.60 realized PnL from 26 trades, while mild-trend entries produced +9256.40 from 69 trades. ATR exits produced +14831.10 and signal sells +9058.30; both mechanisms were profitable. This supports preserving trend participation and ATR trailing exits. It does not justify promoting the descriptive volatility median or any precise diagnostic bucket into a trading rule.
Risk: This is local attribution, not authoritative performance validation. The volatility split uses the median normalized ATR of training entries only to create balanced descriptive groups; it is an ex-post diagnostic boundary and must never be copied into the strategy without a separately specified, test-first hypothesis.
Affected files: `cross_signal_strategy/training_stability.py`, `tests/test_cross_signal_training_stability.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only diagnostic infrastructure; no strategy rule changed

### Add Training Friction Decomposition

Date: 2026-07-10
Decision: Add `friction_diagnostics.py` to replay the official `cross-v0.3.2` local training path under five locked broker configurations: baseline, commission rate doubled, minimum commission doubled, slippage doubled, and all three doubled. Precompute each date/code T-1 signal once and return defensive copies to every scenario.
Reason: The stability stress test showed an 18.42 percentage-point return reduction when all friction assumptions doubled. A component decomposition is required before considering any strategy-level turnover change, because execution-price loss, minimum-ticket cost, and percentage commission imply different remedies.
Evidence: Baseline local training return was +118.75%. Commission-rate-only doubling returned +117.34% (-1.41pp); minimum-commission-only doubling returned +112.67% (-6.07pp); slippage-only doubling returned +106.61% (-12.13pp); all friction doubled returned +100.33% (-18.42pp). Standalone component deltas sum to -19.62pp and the combined path has +1.20pp nonlinear interaction. All scenarios reported 99 buys and 96 sells.
Interpretation: Slippage is the dominant modeled friction source, followed by minimum commission; the percentage commission rate is much less important. The remedies should therefore focus on execution timing/order style and verification of the broker's actual ETF minimum-commission terms, not on adding indicators or broadly suppressing signal sells. Identical event counts do not prove identical code/date paths, so no stronger path-equivalence claim is made.
Risk: This is a local execution stress model, not a prediction of live slippage or a JoinQuant performance result. The broker's actual ETF fee schedule is not established by this experiment and must be confirmed before translating minimum-commission sensitivity into live decisions.
Affected files: `cross_signal_strategy/friction_diagnostics.py`, `tests/test_cross_signal_friction_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only execution diagnostic infrastructure; no strategy rule changed

### Add Capital-Utilization And Shadow-Candidate Diagnostics

Date: 2026-07-10
Decision: Add `capital_utilization_diagnostics.py` to measure occupied slots, vacant-slot causes, and T-1 rejected-candidate forward returns for official `cross-v0.3.2`. Consecutive appearances of the same ETF/rejection reason are collapsed into one signal episode, and score bands are kept broad (`50-59`, `40-49`, `30-39`, `20-29`, below 20).
Reason: The local stability report showed only 0.732 value-weighted exposure. Before changing signals, the research needed to distinguish intentional cash caused by no reversal opportunities from potentially usable candidates rejected by the current gate.
Evidence: Across 730 training days, the portfolio held 0/1/2/3 ETFs on 43/86/172/429 days. There were 301 days with at least one vacant slot and 473 vacant slot-days. Causes were 326 below-threshold candidate slots, 136 slots with no reversal candidate, 6 blocked-combo candidates, 4 overheated candidates, and 1 location-filter candidate. After collapsing consecutive duplicates, the 50-59 below-threshold band contained 51 episodes with average 5/10/20-day shadow returns of +0.93%/+0.44%/+1.38%.
Interpretation: Empty capacity is mostly associated with sub-threshold reversal candidates, but fixed-horizon shadow returns are not portfolio returns. They ignore exits, capital competition, path dependence, and later primary signals. They may justify one isolated local candidate, never a direct threshold change.
Risk: Forward returns use future training prices only as ex-post diagnostics. They are not available to the strategy at decision time and must not be called from production strategy code. No validation-period data was read.
Affected files: `cross_signal_strategy/capital_utilization_diagnostics.py`, `tests/test_cross_signal_capital_utilization.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only diagnostic infrastructure; no strategy rule changed

### Reject 50-59 Backup Cross-Signal Slot Fill

Date: 2026-07-10
Decision: Reject the local candidate that retained the official 60-point primary gate but filled remaining portfolio slots with 50-59 point reversal candidates that passed every other mainline filter.
Reason: This was the narrowest structural test supported by the capital-utilization report. It did not lower the global threshold, did not displace primary candidates, and kept overheat, sell-conflict, location, blocked-combo, cooldown, and same-day ATR-stop protections.
Evidence: Local 2019-2021 baseline returned +118.75% with 6.81% max drawdown, 99 buys, 96 sells, and 0.732 value-weighted exposure. The backup-fill candidate returned +86.39% with 9.17% max drawdown, 110 buys, 107 sells, and 0.810 exposure. It filled 50 backup buys, reduced return by 32.35 percentage points, and increased drawdown by 2.36 percentage points.
Interpretation: Positive 5/10/20-day shadow returns did not survive actual portfolio path simulation. Backup positions occupied capital needed for later stronger signals and interacted poorly with the existing exit system. Idle slots are therefore a feature of signal selectivity, not an error that should be filled mechanically.
Risk: This is a local direction check rather than JoinQuant authority performance. The failure is sufficiently large and directionally adverse that a JoinQuant run would waste a candidate test and increase multiple-testing risk.
Affected files: `cross_signal_strategy/backup_fill_candidate.py`, `tests/test_cross_signal_backup_fill_candidate.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; validation periods were not inspected
Status: rejected locally; do not run on JoinQuant or reserved validation

### Keep CMF(20) Observation-Only

Date: 2026-07-10
Decision: Add a T-1-safe CMF(20) diagnostic, but do not create the pre-specified candidate that would require `CMF > 0` for mild-trend entries. Official `cross-v0.3.2` remains unchanged.
Reason: CMF was selected as one independent price-volume dimension because the existing volume score measures volume expansion but not where the close occurs inside the daily range. The rule was locked before attribution: strong-trend entries would remain unchanged, while mild-trend entries would require CMF above the standard zero line.
Evidence: Among mild-trend entries, `CMF <= 0` produced 17 closed trades, +3412.60 realized PnL, 64.71% win rate, and 3.924 profit/loss ratio. `CMF > 0` produced 52 trades, +5843.80 PnL, 46.15% win rate, and 2.218 profit/loss ratio. The sign relationship was not consistent by year: non-positive CMF was especially strong in 2019, positive CMF dominated in 2020, and both were weaker in 2021. Overall positive-CMF PnL was concentrated in strong-trend entries, where 20 trades produced +14777.70 and a 15.931 profit/loss ratio.
Interpretation: Requiring positive CMF in mild trends would remove many profitable low-position reversals, exactly the behavior the cross-signal strategy is designed to capture. The strong-trend positive-CMF result is descriptive but cannot be converted into a new rule in the same experiment; the non-positive strong-trend sample contains only six trades and was not the pre-registered hypothesis.
Risk: CMF is computed from corrected/adjusted daily bars ending at the frozen signal date and is never called from production order logic. Overall CMF attribution is confounded by trend regime, so using the aggregate result alone would be a selection error.
Affected files: `cross_signal_strategy/cmf_diagnostics.py`, `tests/test_cross_signal_cmf_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; CMF strategy candidate rejected before implementation
