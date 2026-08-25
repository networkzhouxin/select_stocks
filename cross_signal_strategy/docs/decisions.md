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
Affected files: `cross_signal_strategy/local/local_signal_adapter.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Local Adjustment Factor Alignment

Date: 2026-07-05
Decision: Apply 2019-2021 target-ETF adjustment factors inside the local cross-signal training replay so T-1 signal OHLC matches JoinQuant's adjusted historical price口径 on ex-dividend/split dates.
Reason: The remaining JoinQuant/local close outliers were not bad raw data. They occurred exactly on ETF ex-dates: `510880` on 2020-01-17 and `510300` on 2021-01-18. Dividing the previous signal close by the same-day `ex_factor` matches JoinQuant's logged close to rounding precision.
Evidence: `510880` local 2020-01-16 close `2.947 / 1.0513740030198886 = 2.8029987`, matching JoinQuant `2.803`; `510300` local 2021-01-15 close `5.526 / 1.0132002506617996 = 5.4540058`, matching JoinQuant `5.454`. Tests added before implementation and passed with `uvx --with pandas pytest`; full local training replay test passed.
Protocol guard: The replay uses a small 2019-2021 target-ETF factor table and does not read the full `G:\financial\history_data\按年份合并` market-data directory during normal training replay. Only events on or before the current decision date are applied; future factor events are not applied early.
Affected files: `cross_signal_strategy/local/local_adjustment.py`, `cross_signal_strategy/local/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Use ETF Tick Precision For Local ATR Stop Comparison

Date: 2026-07-05
Decision: In local replay, compare ATR stop trigger prices at 0.001 ETF quote precision. Execution price and signal scoring remain unchanged.
Reason: JoinQuant logs ATR stop comparisons at ETF quote precision. On 2020-03-02, local computed `518880` stop as 3.53875 and used a 09:35 price of 3.539, missing the stop by a sub-tick float difference, while JoinQuant triggered the stop as `3.538<=3.539`.
Evidence: Added failing test `test_planner_atr_stop_uses_etf_tick_precision_for_trigger` before implementation. After the fix, local replay sells `518880` on 2020-03-02, matching JoinQuant order timing.
Affected files: `cross_signal_strategy/local/local_order_planner.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Allow Core-Indicator-Ready New Listings Before MA60

Date: 2026-07-05
Decision: Local replay no longer requires 60 daily bars before scoring a newly listed ETF. It requires enough history for the core required indicators (RSI24, MACD, BOLL20, ATR14, ADX14) and allows MA60 to be NaN, which naturally gives no MA60 trend contribution.
Reason: JoinQuant scored and bought `159985` on 2020-03-03 with 56 available bars and `MA60=nan`, producing `buy=70 trend=0`. Local replay incorrectly skipped it as `short_data:56<60`.
Evidence: Added failing test `test_signal_score_allows_listing_before_ma60_when_core_indicators_are_valid` before implementation. Local replay now scores `159985` on 2020-03-03 as `buy=70 trend=0 sell=0`.
Affected files: `cross_signal_strategy/local/local_signal_adapter.py`, `tests/test_cross_signal_local_signal_adapter.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log alignment only; no validation-period influence
Status: adopted

### Correct Confirmed Local Daily Bar Defect

Date: 2026-07-06
Decision: Apply confirmed local daily-bar defects through an external read-time correction layer, without modifying the read-only training source data. The first adopted correction is `512100` on 2020-09-02: close `1.000` in the local daily CSV is overridden to `1.001` for local signal replay.
Reason: JoinQuant diagnostic logs, local 1-minute aggregation, and the user's trading software all showed `512100` 2020-09-02 close `1.001`; only the local daily CSV showed `1.000`. This bad close shifted KDJ's zero-cross boundary and caused the last JoinQuant/local path divergence on 2020-09-22/2020-09-23.
Evidence: Added failing tests before implementation. After the correction layer, `512100` on 2020-09-22 scores `buy=65 rev=35` with both KDJ up-cross flags true, matching JoinQuant. Full local replay against the latest JoinQuant log has 262 local filled events versus 262 JoinQuant filled events and no order-path divergence. Full cross-signal test suite passed with 85 tests.
Affected files: `cross_signal_strategy/local/local_adjustment.py`, `cross_signal_strategy/local/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_local_signal_adapter.py`, `tests/test_cross_signal_local_training_run.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: training log and confirmed source-data correction only; no validation-period influence
Status: adopted

### Use ETF Tick Precision For Local Execution Prices

Date: 2026-07-06
Decision: Local replay should round slippage-adjusted execution prices to ETF tick precision (`0.001`) before applying cash, commission, and position updates.
Reason: JoinQuant strategy uses `PriceRelatedSlippage(0.001)`, but ETF market orders still fill at quoted ETF price precision. The prior local broker kept sub-tick prices such as `3.06306`, which is not a tradable ETF price and caused extra cash/position drift.
Evidence: Added failing tests before implementation and updated local broker expectations from sub-tick prices to tick-rounded prices. Full order path remains aligned with JoinQuant at 262 events versus 262 events, with no first divergence. Full cross-signal test suite passed with 88 tests. The remaining return gap is still mostly JoinQuant internal market-order matching price and rolling share-quantity drift, so no further strategy-rule change is justified.
Affected files: `cross_signal_strategy/local/local_backtester.py`, `cross_signal_strategy/research/order_path_diagnostics.py`, `tests/test_cross_signal_local_backtester.py`, `tests/test_cross_signal_order_path_diagnostics.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
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
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local/local_order_planner.py`, `cross_signal_strategy/local_training_run.py`, `tests/test_cross_signal_strategy.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/strategy_spec.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
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
Affected files: `cross_signal_strategy/research/baseline_report.py`, `tests/test_cross_signal_baseline_report.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; reporting only
Status: adopted

### Add Entry-Score Trade Diagnostics

Date: 2026-07-07
Decision: Add a dedicated trade-diagnostics module that captures buy-entry score snapshots at order-planning time and uses those snapshots for closed-trade attribution.
Reason: A temporary attribution script had read `planner.last_scores` after later days had already refreshed it, which could mislabel historical buys. The issue did not affect local replay returns, positions, or daily equity, but it could mislead factor diagnostics.
Evidence: Added failing-style tests for score snapshot capture and closed-trade attribution before implementation. Current 2021Q3 diagnostic using the formal tool confirms the same PnL conclusion: sell-date Q3 trades PnL -1579.5; buy-date Q3 trades PnL -1445.1; no-volume-confirmation Q3 buys were 5/5 losers with PnL -2000.9.
Affected files: `cross_signal_strategy/research/trade_diagnostics.py`, `tests/test_cross_signal_trade_diagnostics.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
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
Evidence: A temporary JoinQuant probe file (`cross_signal_strategy/archive/probes/smart_trade_joinquant_cross_signal_etf_probe_513880.py`) printed `paused=False` at `2019-12-12 09:35`, `10:35`, and `14:50`. The same minutes had `volume=0` and `money=0`. The full-day minute summary for `2019-12-12` showed `total_minutes=240`, `nonzero_minutes=26`, `total_volume=1405700.0`, `total_money=1539142.0`, `first_nonzero=2019-12-12 09:38:00`, and `last_nonzero=2019-12-12 14:57:00`. This proves the ETF was not considered paused by JoinQuant and was not zero-volume all day; it traded sparsely, and the 09:35 sell minute was one of the zero-volume minutes.
Affected files: `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
Allowed validation influence: training-period execution diagnostics only; no signal or parameter tuning
Status: adopted

### Half-Size A-Share Buys Without Volume Confirmation

Date: 2026-07-08
Decision: When a new-buy candidate is an A-share ETF (`510300`, `159915`, `512100`, `159928`, `510880`) and its `volume_score` is `0`, size the new buy at 50% of the normal per-slot target. Do not apply this rule to cross-market or cross-asset ETFs.
Reason: In A-share ETFs, a reversal signal without volume confirmation is more likely to be a low-quality repair attempt. For QDII/cross-market and commodity ETFs, local volume has different microstructure and earlier global volume rules damaged returns. A 50% scale is a broad risk-control rule; it deliberately avoids the training-best 25% scale to reduce overfitting risk.
Evidence: Training-only 2019-2021 local replay: baseline returned +106.17%, annualized +27.36%, max drawdown 9.35%, Sharpe 1.887, Sortino 2.931, 2021Q3 -5.32%. A-share-only zero-volume scale `0.50` returned +109.19%, annualized +27.98%, max drawdown 7.86%, Sharpe 1.995, Sortino 3.113, 2021Q3 -4.25%, with the same 103/101 buy/sell event count. Cross-market and cross-asset scaling reduced return and was rejected. JoinQuant 2019-2021 training confirmation returned +115.41%, annualized +30.06%, max drawdown 6.93%, Sharpe 2.870, with 102 buy logs, 101 sell logs, and transaction export status 202 fully filled plus one known `513880.XSHG` zero-volume cancellation.
Affected files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local/local_order_planner.py`, `tests/test_cross_signal_strategy.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/docs/backtest_notes.md`
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
Affected files: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py`, `tests/test_cross_signal_low_bounce_candidate.py`, `cross_signal_strategy/research/attribution_diagnostics.py`, `tests/test_cross_signal_attribution_diagnostics.py`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; validation periods must not be inspected unless the candidate is first frozen after JoinQuant training confirmation
Status: rejected after JoinQuant 2019-2021 training confirmation. The candidate improved win rate, profit/loss ratio, and Sharpe slightly, but reduced return and annualized return while worsening max drawdown and Sortino. It must not proceed to reserved validation.

### Add Training-Window Stability Diagnostics

Date: 2026-07-10
Decision: Add a standalone `research/training_stability.py` diagnostic module for official `cross-v0.3.2`. It reports annual performance, profit concentration, exit-reason quality, trading-day holding periods, T-1 entry trend/volatility groups, average exposure, and a complete doubled-friction replay. It does not change the strategy or production files.
Reason: Repeated entry and exit candidates showed that a higher local win rate can hide lost payoff opportunities. Before searching for another indicator or threshold, the training return must be checked for year, trade, ETF, exit, and market-state concentration.
Evidence: The local 2019-2021 baseline returned +118.75% with 6.81% max drawdown and 0.732 average exposure. Calendar-year returns were +35.68% in 2019, +48.45% in 2020, and +8.60% in 2021. The largest winning trade contributed 8.32% of gross profit, the top three contributed 23.51%, and the largest ETF contribution was 22.00%, so the result was not supported by one trade or one ETF. A complete 2x commission/minimum-commission/slippage replay still returned +100.33%, but the 18.42 percentage-point reduction shows meaningful friction sensitivity.
Interpretation: Strong-trend entries produced +14847.60 realized PnL from 26 trades, while mild-trend entries produced +9256.40 from 69 trades. ATR exits produced +14831.10 and signal sells +9058.30; both mechanisms were profitable. This supports preserving trend participation and ATR trailing exits. It does not justify promoting the descriptive volatility median or any precise diagnostic bucket into a trading rule.
Risk: This is local attribution, not authoritative performance validation. The volatility split uses the median normalized ATR of training entries only to create balanced descriptive groups; it is an ex-post diagnostic boundary and must never be copied into the strategy without a separately specified, test-first hypothesis.
Affected files: `cross_signal_strategy/research/training_stability.py`, `tests/test_cross_signal_training_stability.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only diagnostic infrastructure; no strategy rule changed

### Add Training Friction Decomposition

Date: 2026-07-10
Decision: Add `research/friction_diagnostics.py` to replay the official `cross-v0.3.2` local training path under five locked broker configurations: baseline, commission rate doubled, minimum commission doubled, slippage doubled, and all three doubled. Precompute each date/code T-1 signal once and return defensive copies to every scenario.
Reason: The stability stress test showed an 18.42 percentage-point return reduction when all friction assumptions doubled. A component decomposition is required before considering any strategy-level turnover change, because execution-price loss, minimum-ticket cost, and percentage commission imply different remedies.
Evidence: Baseline local training return was +118.75%. Commission-rate-only doubling returned +117.34% (-1.41pp); minimum-commission-only doubling returned +112.67% (-6.07pp); slippage-only doubling returned +106.61% (-12.13pp); all friction doubled returned +100.33% (-18.42pp). Standalone component deltas sum to -19.62pp and the combined path has +1.20pp nonlinear interaction. All scenarios reported 99 buys and 96 sells.
Interpretation: Slippage is the dominant modeled friction source, followed by minimum commission; the percentage commission rate is much less important. The remedies should therefore focus on execution timing/order style and verification of the broker's actual ETF minimum-commission terms, not on adding indicators or broadly suppressing signal sells. Identical event counts do not prove identical code/date paths, so no stronger path-equivalence claim is made.
Risk: This is a local execution stress model, not a prediction of live slippage or a JoinQuant performance result. The broker's actual ETF fee schedule is not established by this experiment and must be confirmed before translating minimum-commission sensitivity into live decisions.
Affected files: `cross_signal_strategy/research/friction_diagnostics.py`, `tests/test_cross_signal_friction_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only execution diagnostic infrastructure; no strategy rule changed

### Add Capital-Utilization And Shadow-Candidate Diagnostics

Date: 2026-07-10
Decision: Add `research/capital_utilization_diagnostics.py` to measure occupied slots, vacant-slot causes, and T-1 rejected-candidate forward returns for official `cross-v0.3.2`. Consecutive appearances of the same ETF/rejection reason are collapsed into one signal episode, and score bands are kept broad (`50-59`, `40-49`, `30-39`, `20-29`, below 20).
Reason: The local stability report showed only 0.732 value-weighted exposure. Before changing signals, the research needed to distinguish intentional cash caused by no reversal opportunities from potentially usable candidates rejected by the current gate.
Evidence: Across 730 training days, the portfolio held 0/1/2/3 ETFs on 43/86/172/429 days. There were 301 days with at least one vacant slot and 473 vacant slot-days. Causes were 326 below-threshold candidate slots, 136 slots with no reversal candidate, 6 blocked-combo candidates, 4 overheated candidates, and 1 location-filter candidate. After collapsing consecutive duplicates, the 50-59 below-threshold band contained 51 episodes with average 5/10/20-day shadow returns of +0.93%/+0.44%/+1.38%.
Interpretation: Empty capacity is mostly associated with sub-threshold reversal candidates, but fixed-horizon shadow returns are not portfolio returns. They ignore exits, capital competition, path dependence, and later primary signals. They may justify one isolated local candidate, never a direct threshold change.
Risk: Forward returns use future training prices only as ex-post diagnostics. They are not available to the strategy at decision time and must not be called from production strategy code. No validation-period data was read.
Affected files: `cross_signal_strategy/research/capital_utilization_diagnostics.py`, `tests/test_cross_signal_capital_utilization.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: adopted as training-only diagnostic infrastructure; no strategy rule changed

### Reject 50-59 Backup Cross-Signal Slot Fill

Date: 2026-07-10
Decision: Reject the local candidate that retained the official 60-point primary gate but filled remaining portfolio slots with 50-59 point reversal candidates that passed every other mainline filter.
Reason: This was the narrowest structural test supported by the capital-utilization report. It did not lower the global threshold, did not displace primary candidates, and kept overheat, sell-conflict, location, blocked-combo, cooldown, and same-day ATR-stop protections.
Evidence: Local 2019-2021 baseline returned +118.75% with 6.81% max drawdown, 99 buys, 96 sells, and 0.732 value-weighted exposure. The backup-fill candidate returned +86.39% with 9.17% max drawdown, 110 buys, 107 sells, and 0.810 exposure. It filled 50 backup buys, reduced return by 32.35 percentage points, and increased drawdown by 2.36 percentage points.
Interpretation: Positive 5/10/20-day shadow returns did not survive actual portfolio path simulation. Backup positions occupied capital needed for later stronger signals and interacted poorly with the existing exit system. Idle slots are therefore a feature of signal selectivity, not an error that should be filled mechanically.
Risk: This is a local direction check rather than JoinQuant authority performance. The failure is sufficiently large and directionally adverse that a JoinQuant run would waste a candidate test and increase multiple-testing risk.
Affected files: `cross_signal_strategy/archive/candidates/backup_fill_candidate.py`, `tests/test_cross_signal_backup_fill_candidate.py`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; validation periods were not inspected
Status: rejected locally; do not run on JoinQuant or reserved validation

### Keep CMF(20) Observation-Only

Date: 2026-07-10
Decision: Add a T-1-safe CMF(20) diagnostic, but do not create the pre-specified candidate that would require `CMF > 0` for mild-trend entries. Official `cross-v0.3.2` remains unchanged.
Reason: CMF was selected as one independent price-volume dimension because the existing volume score measures volume expansion but not where the close occurs inside the daily range. The rule was locked before attribution: strong-trend entries would remain unchanged, while mild-trend entries would require CMF above the standard zero line.
Evidence: Among mild-trend entries, `CMF <= 0` produced 17 closed trades, +3412.60 realized PnL, 64.71% win rate, and 3.924 profit/loss ratio. `CMF > 0` produced 52 trades, +5843.80 PnL, 46.15% win rate, and 2.218 profit/loss ratio. The sign relationship was not consistent by year: non-positive CMF was especially strong in 2019, positive CMF dominated in 2020, and both were weaker in 2021. Overall positive-CMF PnL was concentrated in strong-trend entries, where 20 trades produced +14777.70 and a 15.931 profit/loss ratio.
Interpretation: Requiring positive CMF in mild trends would remove many profitable low-position reversals, exactly the behavior the cross-signal strategy is designed to capture. The strong-trend positive-CMF result is descriptive but cannot be converted into a new rule in the same experiment; the non-positive strong-trend sample contains only six trades and was not the pre-registered hypothesis.
Risk: CMF is computed from corrected/adjusted daily bars ending at the frozen signal date and is never called from production order logic. Overall CMF attribution is confounded by trend regime, so using the aggregate result alone would be a selection error.
Affected files: `cross_signal_strategy/research/cmf_diagnostics.py`, `tests/test_cross_signal_cmf_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; CMF strategy candidate rejected before implementation

### Reject Strong-Trend Idle-Slot Sizing Before Candidate Creation

Date: 2026-07-10
Decision: Add a training-only strong-trend capacity diagnostic, but do not create the proposed candidate that would let the highest-ranked strong-trend buy consume one otherwise unused slot. Official `cross-v0.3.2` remains unchanged.
Reason: Strong-trend entries were highly profitable in aggregate, but additional sizing is relevant only when all official same-day candidates have been allocated and a complete extra slot remains fundable without using the official cash reserve. This narrower executable subset must be stable on its own.
Evidence: The official local path contained 27 filled strong-trend buys and 26 closed trades with +14847.60 PnL. Only 5 trades were capacity eligible. They produced +1371.00 PnL and a 4.249 profit/loss ratio, but the yearly counts were only 2/2/1 for 2019/2020/2021, the 2021 trade lost 44.80, and one ETF contributed 60.31% of capacity-subset gross profit. The pre-registered minimum sample, yearly profitability, and concentration gates all failed.
Interpretation: Strong-trend quality does not imply that concentration should be increased. Most strong entries have no full unused slot after other valid candidates are allocated, and the few remaining opportunities are too sparse and concentrated to support a robust sizing rule.
Risk: MFE/MAE and realized trade outcomes use future training prices only for ex-post diagnostics. They never enter strategy code. Turning five selected trades into a sizing rule would be a high-risk form of small-sample overfitting.
Affected files: `cross_signal_strategy/research/strong_trend_capacity_diagnostics.py`, `tests/test_cross_signal_strong_trend_capacity.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; sizing candidate rejected before implementation

### Reject The Greater-Than-One-ATR 09:35 Gap Filter

Date: 2026-07-10
Decision: Add a T-1-safe execution-gap diagnostic, but do not create the pre-registered candidate that would skip new buys when the T-day 09:35 price is more than one T-1 ATR above the T-1 close. Official `cross-v0.3.2` remains unchanged.
Reason: A T-1 low-position signal can theoretically become an unattractive chase after a large next-morning gap. The test therefore separated signal quality from the actual 09:35 entry price without changing the original trading path.
Evidence: The `>1 ATR` group contained only 5 closed trades but produced +3309.80 PnL, +8.15% average return, 60.00% win rate, and a 9.120 profit/loss ratio. It was profitable in 2019 and 2020; four of the five trades were strong-trend entries and produced +3340.20. The pre-registered sample-size and annual-underperformance gates failed.
Interpretation: Large positive gaps are not automatically chase-risk in this strategy. In the training path they usually represent strong-trend continuation, so a broad high-gap block would remove valuable opportunities. The weak post-hoc mild-trend `0.5-1 ATR` subgroup is not eligible for rule creation from this experiment.
Risk: T-day 09:35 is permissible only as an explicit execution-time filter. Later prices used for MFE/MAE are ex-post diagnostics. Searching smaller gap cutoffs or trend interactions after seeing these groups would create multiple-testing bias.
Affected files: `cross_signal_strategy/research/gap_execution_diagnostics.py`, `tests/test_cross_signal_gap_execution_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; gap-filter candidate rejected before implementation

### Keep BOLL BandWidth Observation-Only

Date: 2026-07-10
Decision: Add a T-1-safe standard BOLL(20,2) BandWidth diagnostic, but do not create the pre-registered candidate that would require rising BandWidth for mild-trend entries. Official `cross-v0.3.2` remains unchanged.
Reason: The strategy already uses BOLL location and middle/upper-band events, but not volatility expansion or contraction. BandWidth direction was tested as one independent extension without changing the established BOLL parameters.
Evidence: Mild-trend rising width strongly outperformed declining width in 2019 and 2020. In 2021 the relationship reversed: rising width had 17 trades, +0.36% average return, and 41.18% win rate, versus 14 declining-width trades with +0.72% average return and 64.29% win rate. The pre-registered annual consistency gate failed.
Interpretation: BandWidth contains descriptive information, but its direction is regime-dependent. Adding it as a universal mild-trend confirmation would fit 2019-2020 behavior and damage the distinct 2021 regime.
Risk: Searching an absolute width, alternate BOLL parameters, multi-day slope, or a special 2021 exception after seeing the annual split would be post-hoc parameter mining. BandWidth must remain diagnostic-only in this experiment.
Affected files: `cross_signal_strategy/research/boll_width_diagnostics.py`, `tests/test_cross_signal_boll_width_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; BandWidth strategy candidate rejected before implementation

### Reject Cross-Sequence Filtering Before Candidate Creation

Date: 2026-07-10
Decision: Add T-1-safe cross-event timing diagnostics, but do not create the pre-registered candidate that would block mild-trend entries where MACD crosses up before all active RSI/KDJ upward crosses. Official `cross-v0.3.2` remains unchanged.
Reason: The existing three-day window records resonance but not event order. A clean fast-oscillator-then-MACD sequence could theoretically distinguish early reversal plus later confirmation from delayed oscillator chasing.
Evidence: Among 96 closed trades, no `macd_leads_oscillators` trade occurred. Only 11 trades had oscillators lead MACD, including 7 mild-trend trades with yearly counts 2/3/2. The two 2021 mild oscillator-leading trades both lost. Most trades, 70, had no currently active MACD upward confirmation and still generated +16316.10 PnL.
Interpretation: There is no executable training sample for the proposed MACD-leading block. The strategy's edge mainly comes from early oscillator reversal signals rather than a stable MACD confirmation sequence, so forcing a sequence state machine would misdescribe the actual strategy.
Risk: Reframing the post-hoc same-day or mixed groups as candidates would be a new experiment selected after seeing outcomes. The current samples are also too small for sequence-specific rules.
Affected files: `cross_signal_strategy/research/sequence_diagnostics.py`, `tests/test_cross_signal_sequence_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; sequence candidate rejected before implementation

### Reject Reversal-First Candidate Ranking

Date: 2026-07-11
Decision: Reject the isolated local candidate that sorts already-eligible buys by reversal score before total buy score. Keep the official `buy_score -> reversal_score -> code` ordering in `cross-v0.3.2`.
Reason: The cross-signal philosophy gives reversal signals a central role, so reversal-first ranking was the one structurally motivated alternative worth testing without weight search. Both paths shared identical cached T-1 signals and differed only in candidate ordering.
Evidence: Reversal-first improved local return from +118.75% to +121.69% with unchanged 6.81% drawdown, but it changed only one buy day, 2021-12-27. Official selected `159928` (buy 70, reversal 35); the candidate selected `513500` (buy 69, reversal 45). Through the 2021-12-31 boundary, the former moved -3.29% and the latter +0.95%.
Interpretation: The apparent improvement is a single terminal mark-to-market event, not a repeatable ranking edge. The activity gate failed, and 2019/2020 had no changed decisions at all.
Risk: Promoting a rule from one late-boundary event would be severe small-sample and endpoint overfitting. No JoinQuant or reserved validation run is justified.
Affected files: `cross_signal_strategy/archive/candidates/ranking_candidate.py`, `tests/test_cross_signal_ranking_candidate.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: rejected locally; official ranking unchanged

### Keep Kaufman ER(10) Observation-Only

Date: 2026-07-11
Decision: Add a T-1-safe standard Kaufman ER(10) diagnostic, but do not create the pre-registered candidate that would require rising ER for mild-trend entries. Official `cross-v0.3.2` remains unchanged.
Reason: ER measures directional path efficiency rather than oscillator level, volume, or volatility, making it a genuinely independent market-state dimension worth one standard-parameter test.
Evidence: Mild rising ER produced 30 trades, +2812.30 PnL, +1.12% average return, 50.00% win rate, and 2.059 profit/loss ratio, versus declining ER with 38 trades, +6597.10 PnL, +1.89% average return, 52.63% win rate, and 3.092 profit/loss ratio. Annual comparisons failed in 2019, 2020, and 2021 on at least one locked metric.
Interpretation: ER direction adds descriptive context but does not robustly improve mild-trend entries. Its apparent strength in strong-trend trades overlaps a state already captured by the strategy and was not the pre-registered target.
Risk: Searching ER levels, periods, multi-day slopes, or a strong-trend exception after seeing these results would be post-hoc parameter mining.
Affected files: `cross_signal_strategy/research/efficiency_ratio_diagnostics.py`, `tests/test_cross_signal_efficiency_ratio_diagnostics.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic adopted; ER strategy candidate rejected before implementation

### Make Local 09:35 Fills Liquidity-Aware And State-Atomic

Date: 2026-07-11
Decision: Reject local 09:35 fills when the execution minute is missing or both `volume` and `num_trades` are zero. Preserve buy date, entry ATR, and highest-close state until a sell is actually filled. Before each new buy, enforce `max_hold` against the broker's post-execution holdings so an unfilled sell cannot create a phantom slot.
Reason: The source minute files retain stale-price bars during no-trade intervals. The previous engine consumed only `close`, so it could fill an order at a price with no supporting trade. The planner also cleared position state when merely planning a sell and counted the planned slot as free before knowing whether the sell filled.
Evidence: Approved training data contains a full morning suspension/resumption pattern for `159915` on 2020-12-16 and 2021-02-09: volume and trade count remain zero through 10:30, then the first positive minute is 10:31. A direct regression reproduced an erroneous 2021-02-09 local buy at the stale 09:35 price before the fix. The first corrected full replay exposed a four-holding state when an unfilled sell was followed by a buy; the state-atomic fix restored the maximum to three. Final corrected local replay ended at 44122.30 (+120.61%) with 7.47% max drawdown, 92 filled buys, 89 filled sells, 15 no-trade rejections, and 2 buy rejections because no slot was actually released.
Interpretation: Minute-bar availability is execution evidence, not a new alpha filter. A zero-volume minute does not prove a formal exchange suspension, but it does mean the local replay has no transaction with which to justify a fill. State transitions must follow actual fills rather than planned orders.
Risk: The local data has no explicit `paused` field or order-book quotes. A liquid order might be executable despite no transaction in one recorded minute, while a positive-volume minute does not guarantee the strategy's full order size could fill. JoinQuant remains the performance authority. No 10:35 retry is implemented in this milestone.
Affected files: `cross_signal_strategy/local/local_backtester.py`, `cross_signal_strategy/local/local_order_planner.py`, `tests/test_cross_signal_local_backtester.py`, `tests/test_cross_signal_local_order_planner.py`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2019-2021 training minute/daily data and approved 2018 warm-up data were read
Status: adopted as local execution-correctness infrastructure; official `cross-v0.3.2` JoinQuant strategy logic unchanged

### Keep Historical IOPV Diagnostic-Only

Date: 2026-07-11
Decision: Add a read-only training-data IOPV quality audit, but do not create a premium-filter strategy candidate or modify official `cross-v0.3.2`.
Reason: QDII secondary-market prices can carry economically meaningful premium risk, but the local IOPV field must be complete, point-in-time safe, and reproducible on JoinQuant before it can influence a 09:35 order.
Evidence: The audit inspected 2,029,422 minute rows across the frozen 12-ETF pool. There were no duplicate minute keys, non-positive IOPV values, or infinite values. Valid IOPV covered 89.06% of all rows and 89.24% of represented 09:35 observations. The 09:35 coverage was 78.45% in 2019, 88.37% in 2020, and 99.69% in 2021. Ten full-year 2019 ETFs shared the same 50 missing dates, and all twelve 2020 ETFs shared the same 27 missing dates, proving material cross-sectional source-level gaps. Local `513100` close/IOPV pairs on 2020-02-07 and 2020-09-21 reproduced official 8.10% and 22.5% premium announcements to rounding. Of 216,447 no-trade minutes, 20,576 still had changing IOPV, confirming that a moving IOPV does not make a stale secondary-market price executable.
Interpretation: Local IOPV is correctly scaled and captures genuine historical premium episodes, but incomplete 2019-2020 coverage prevents unbiased threshold research. The source metadata also does not prove whether a row labelled 09:35 was available at 09:35:00, so using it as a decision input could create same-minute look-ahead.
Risk: Daily NAV cannot backfill the missing morning IOPV without leakage. QDII IOPV remains indicative rather than guaranteed realizable NAV. A future implementation requires a point-in-time source/platform probe and must define conservative behavior for missing IOPV without selecting that behavior from validation results.
Affected files: `cross_signal_strategy/research/iopv_quality_diagnostics.py`, `tests/test_cross_signal_iopv_quality.py`, `cross_signal_strategy/docs/iopv_data_quality.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2019-2021 training minute data and contemporaneous public fund-manager announcements were used
Status: audit infrastructure adopted; premium factor not opened for strategy experimentation

### Reject The US-QDII Previous-NAV Premium Filter

Date: 2026-07-11
Decision: Keep the JoinQuant capability probe and training-only attribution, but reject a `513100/513500` premium-filter candidate and leave official `cross-v0.3.2` unchanged.
Reason: JoinQuant proved T-1 unit NAV is available at 09:35 without future leakage, which justified one market-structure observation. The candidate still required enough elevated-premium buys across years and both ETFs before any order rule could exist.
Evidence: Of 28 closed target trades, 27 had usable reference data. Only two exceeded the fixed 5% elevated-premium boundary; both were `513100` trades in 2020, averaged 8.16% premium, and together earned +388.80 with +2.55% average return. `513500` had no elevated-premium trade, no trade exceeded 10%, and the normal at-or-below-2% group remained strongly profitable with 24 trades and +6509.90 PnL.
Interpretation: The strategy's existing signal and ranking path already avoided nearly all severe premium episodes. A new veto would add platform-specific complexity for two profitable historical entries and has no cross-year or cross-code support.
Risk: Lowering the boundary after seeing only two observations would be direct parameter mining. Applying T-1 NAV to `159920` or `513880` would also misrepresent their demonstrably dynamic intraday IOPV.
Affected files: `cross_signal_strategy/research/us_qdii_premium_diagnostics.py`, `tests/test_cross_signal_us_qdii_premium_diagnostics.py`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only approved 2019-2021 training data and no-order JoinQuant capability probes on two training dates were used
Status: diagnostic adopted; candidate rejected before implementation; microstructure budget exhausted

### Harden PTrade Risk-State Recovery Without Guessing

Date: 2026-07-11
Decision: Keep the external atomic checkpoint as the primary cross-day state source, use broker portfolio quantity and cost as live facts, use strategy-only `get_trades()` for current-day restart recovery, and allow account delivery replay only when quantity, entry signal, ATR date, fill price, and price history all reconcile. Any unresolved position remains unverified and cannot be sold automatically.
Reason: PTrade may restore or reset `g` during restart and strategy update. A valid open position still needs its buy date, entry ATR, and highest close for minimum-hold and ATR-stop behavior. The existing multi-factor adapter recovers some missing values with guesses (`cost * 2%`, `prev_date - 10 days`, or arbitrary lookback highs), which is not acceptable for the cross-signal strategy's no-invention rule.
Evidence: Tests were written and observed failing before implementation for delivery net-position reconstruction, broker-quantity mismatch, proven T-1 entry reconstruction, ineligible/manual entry rejection, stale closed-position cleanup, invalid broker cost, current-day `get_trades()` recovery, PTrade API phase restrictions, and fill-price-versus-cost-basis precision. The PTrade adapter suite passes 63 tests after implementation, including an AST guard for 40 pure business functions shared with the frozen JoinQuant mainline.
Interpretation: Normal restarts and code updates recover exact state from the account/trade-specific checkpoint. If that file is absent, same-day strategy fills can still be attributed exactly through `get_trades()`. Older `get_deliver()` rows are account-wide, so they are strong transaction evidence but not an unconditional strategy-ownership proof; mixed/manual accounts require operator review.
Risk: No platform API can recreate deleted strategy ownership metadata from an account-wide historical statement with absolute certainty. Corporate share changes or manual trades cause quantity mismatch and fail closed. Pre-adjusted historical closes are used only for a disaster-recovery trailing peak and do not enter signal calculation or strategy research.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; this is platform reliability work and reads no reserved market period
Status: adopted for the PTrade adapter; JoinQuant business rules and production multi-factor files unchanged

### Reject The Fixed MACD(6,13,5) Candidate

Date: 2026-07-13
Decision: Reject the isolated MACD(6,13,5) candidate and retain official MACD(12,26,9) in `cross-v0.3.2`.
Reason: A widely discussed faster MACD setting was a concrete external claim worth one explicitly authorized, fixed, single-variable training comparison. It did not justify a broader parameter search.
Evidence: The candidate changed 89 filled-order days across 2019/2020/2021, so the experiment was behaviorally active. Local total return fell from +120.61% to +84.69%, annualized return from 30.27% to 22.76%, Sharpe from 2.172 to 1.766, Sortino from 3.415 to 2.670, win rate from 56.18% to 50.00%, and profit/loss ratio from 4.440 to 2.834. Maximum drawdown improved from 7.47% to 7.00%, and 2020 improved from 49.74% to 51.94%, but 2019 fell from 35.84% to 17.02% and 2021 from 8.46% to 3.87%.
Interpretation: Faster MACD responds earlier but also confirms more short-lived noise in this daily cross-signal ensemble. The result is not an endpoint artifact; it is a broad three-year path change with worse aggregate and cross-year quality.
Risk: Trying 5/12/4, 7/14/5, other nearby periods, or compensating score thresholds after seeing this result would be direct parameter mining. The 2018 data used here was warm-up only, and no validation period was inspected.
Affected files: `cross_signal_strategy/archive/candidates/macd_parameter_candidate.py`, `tests/test_cross_signal_macd_parameter_candidate.py`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: rejected locally; official strategy unchanged; one-shot MACD budget exhausted

### Record The Multiple-Testing Audit Without Changing Strategy

Date: 2026-07-13
Decision: Adopt a reproducible training-only multiple-testing audit and keep official `cross-v0.3.2` frozen.
Reason: Repeated 2019-2021 experiments create strategy-selection bias even when every individual experiment obeys the no-validation rule. After the 2026-07-16 share-flow observation, the retained ledger can prove a minimum of 50 failed or non-adopted experiments plus the selected mainline, so the apparent training significance must be judged against at least 51 trials.
Evidence: The frozen local path returned +120.61% with 30.27% annualized return and 2.172 annualized Sharpe. Its single-trial PSR p-value is 0.000123988; the minimum-51 Bonferroni value is 0.00632340. A Newey-West/HAC mean-return test with automatic lag 6 gives a single-trial p-value of 0.0000622008 and minimum-51 Bonferroni value of 0.00317224. The PSR approximation remains below 5% through 403 trials.
Interpretation: The retained evidence is stronger than a marginal best-of-51 result, but the corrected confidence is an optimistic upper bound because 51 is only a provable minimum. This remains in-sample evidence and is not out-of-sample validation.
Risk: Canonical DSR is unavailable because the complete candidate Sharpe distribution was not retained. PBO is unavailable because aligned daily return curves for all candidates were not retained. Reconstructing either from prose would create false precision.
Affected files: `cross_signal_strategy/research/multiple_testing_audit.py`, `tests/test_cross_signal_multiple_testing_audit.py`, `cross_signal_strategy/docs/multiple_testing_audit.md`, `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: audit infrastructure adopted; strategy logic, ETF pool, and parameters unchanged; the retained trial-count lower bound is updated whenever a new experiment closes

### Add An Isolated PTrade IOPV Capability Probe

Date: 2026-07-13
Decision: Add a separate, no-order PTrade probe for QDII IOPV availability and keep it outside official `cross-v0.3.2`.
Reason: The local official PTrade documentation identifies a genuinely new point-in-time mechanism: trading-module `get_snapshot()` documents real-time `iopv` and `hsTimeStamp`, while `get_etf_info()` documents whether IOPV should be published plus T-1 NAV metadata. This can establish live data feasibility but cannot retroactively improve the training backtest.
Evidence: The documentation states that `get_snapshot()` is available only in the trading module and returns real-time snapshots. It also states that `get_etf_info()` is PTrade-client and counter dependent. The probe checks the four QDII ETFs in the frozen PTrade pool at 09:34, 09:35, and 09:36, uses three of the five allowed callbacks, logs only capability fields, and contains no order call.
Interpretation: PTrade may provide a better live fair-value input than JoinQuant, but operational availability remains unproved until Guojin simulation/live logs show positive IOPV with fresh timestamps for ETFs marked `publish=1`.
Risk: Current-market output is not training or validation evidence and must not be used to choose a premium threshold. The previously rejected `513100/513500` premium candidate remains rejected, the microstructure research family remains exhausted, and no production rule changes in this milestone.
Affected files: `cross_signal_strategy/archive/probes/smart_trade_ptrade_cross_signal_iopv_probe.py`, `tests/test_cross_signal_ptrade_iopv_probe.py`, `cross_signal_strategy/docs/ptrade_iopv_probe.md`, `cross_signal_strategy/docs/iopv_data_quality.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only local official PTrade documentation and existing training-governance records were read
Status: capability probe ready; operational result pending a separate one-session Guojin PTrade run

### Observe PTrade IOPV Without Changing Frozen Orders

Date: 2026-07-14
Decision: Add failure-open IOPV logging to actual QDII buy submissions in the formal PTrade adapter while retaining `cross-v0.3.2` and every frozen business rule.
Reason: The user accepted the official PTrade API contract as a provisional operational assumption and chose to validate actual Guojin field quality through release logs because a separate intraday probe is inconvenient. Reusing the snapshot already fetched for the buy price gives the required evidence without an extra API call or scheduled task.
Evidence: Tests first failed because no IOPV observation builder or buy-time log existed. The implementation records price, positive IOPV, descriptive premium, raw `hsTimeStamp`, and snapshot age before QDII order submission. Runtime tests prove that valid, zero, and missing IOPV produce the same order quantity and path, while A-share buys emit no IOPV line. The frozen JoinQuant/PTrade pure-business AST parity guard remains unchanged.
Interpretation: These logs can establish whether the Guojin connection supplies usable real-time IOPV at actual exposure decisions. They are operational telemetry, not a signal and not evidence that a premium filter improves returns.
Risk: PTrade may return zero, missing, or stale IOPV despite the documented field. Logging therefore catches its own errors and cannot block or resize an order. Early live observations must not be used to tune a threshold or override the exhausted microstructure research budget.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; no market period was read and no strategy result informed this platform-only change
Status: adopted as observation-only PTrade release telemetry; JoinQuant strategy and production multi-factor files unchanged

### Reject The Fixed Horizontal Support/Resistance Filter

Date: 2026-07-14
Decision: Keep the T-2-safe horizontal-structure diagnostic, reject the pre-registered near-resistance filter before candidate creation, and leave official `cross-v0.3.2` unchanged.
Reason: Prior horizontal highs and lows are economically distinct enough from MA/BOLL location to justify one fixed observation, but they must improve the actual cross-signal path consistently rather than merely sound intuitive.
Evidence: The diagnostic used exactly 20 valid bars ending T-2 and normalized T-1 distance with official ATR(14). The locked mild-trend near-resistance group materially outperformed the comparison in 2019 (+6.25% average return and 50.00% win rate versus about +0.35% and 42.86%), underperformed both metrics in 2020 (+3.56% and 40.00% versus +3.97% and 66.67%), and had lower average return but higher win rate in 2021 (+0.61% and 61.54% versus about +0.75% and 50.00%). All 89 closed buys were more than one ATR above prior support.
Interpretation: Near resistance is not a universal false-signal condition; it can also describe an entry already participating in emerging strength. The support side had no actionable sample under the existing location filters. The strong breakout group's descriptive result cannot justify a breakout rule because that candidate was not pre-registered.
Risk: Searching alternate channel periods, ATR distances, pivots, Fibonacci levels, support exceptions, breakout rewards, or volume profiles after seeing these groups would be post-hoc winner selection. No validation period was inspected.
Affected files: `cross_signal_strategy/research/horizontal_structure_diagnostics.py`, `tests/test_cross_signal_horizontal_structure_diagnostics.py`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/multiple_testing_audit.md`, `tests/test_cross_signal_multiple_testing_audit.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: diagnostic retained; candidate rejected; research family exhausted; official strategy and production multi-factor files unchanged

### Reject The Controlled-Breakout Anti-Chase Filter

Date: 2026-07-14
Decision: Retain the T-2-safe breakout-extension diagnostic, reject the candidate before implementation, exhaust the reopened research family, and leave official `cross-v0.3.2` unchanged.
Reason: The user-authorized question correctly distinguished a controlled breakout from an already extended one without allowing breakout to replace the core cross signals. The rule still required repeated cross-year evidence before it could veto a buy.
Evidence: Controlled breakouts had 15 closed trades, +7.83% average return, 73.33% win rate, and 11.365 profit/loss ratio. Extended breakouts had only 2 trades, +1.32% average return, and 50.00% win rate. Extended annual counts were 1/0/1, below every sample gate; the single 2019 extended trade won, so extended win rate was 100% versus 80% for controlled breakouts that year.
Interpretation: The descriptive gap is interesting but not decision-grade. Fifteen of seventeen breakouts were controlled, averaging only 3.29% above MA20 and +6.93% over the trailing 20 closes, which supports the idea that many breakouts in this strategy are early-strength entries rather than late momentum chasing. The two extended cases cannot support a stable veto.
Risk: Searching a lower RSI threshold, smaller MA20 distance, another resistance window, an AND condition, or a controlled-breakout bonus after seeing the result would be post-hoc parameter mining. No validation period was inspected.
Affected files: `cross_signal_strategy/research/breakout_extension_diagnostics.py`, `tests/test_cross_signal_breakout_extension_diagnostics.py`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/multiple_testing_audit.md`, `tests/test_cross_signal_multiple_testing_audit.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only approved 2018 warm-up and 2019-2021 training data were read
Status: observation diagnostic retained; candidate not created; research budget returned to zero; official strategy and production multi-factor files unchanged

### Reject ETF Share-Flow Sign As A Universal Entry Filter

Date: 2026-07-16
Decision: Retain the exact-root, observation-only share-flow diagnostic, reject any order-changing candidate, exhaust the one-shot family, and leave official `cross-v0.3.2` unchanged.
Reason: Shares outstanding are an independent primary-market dimension worth one controlled observation, but publication timing and corporate actions require stricter handling than exchange volume. The pre-registered question used only five domestic ETFs, exact T-1 endpoints, a fixed five-observation sign, and a cross-year consistency gate.
Evidence: All 52 eligible domestic closed buys were comparable. Positive flow had 24 trades, +1.39% average return, 54.17% win rate, and 3.398 profit/loss ratio. Non-positive flow had 28 trades, +3.70% average return, 50.00% win rate, and 3.624 profit/loss ratio. Non-positive average return led in 2019 (+8.10% versus +1.35%) and 2020 (+5.92% versus +1.24%), but positive flow led both average return and win rate in 2021 (+1.59%/62.50% versus -0.20%/46.15%).
Interpretation: The factor contains descriptive information, but its sign is not a stable universal confirmation. ETF creations can follow price demand, arbitrage, or delayed allocation rather than precede a durable rise; redemptions can also occur during profitable mean-reversion entries. The 2021 reversal is large and based on 21 trades, so it cannot be dismissed as one endpoint event.
Risk: Turning the aggregate PnL difference into a non-positive-flow preference would fit 2019-2020 and fail the annual gate. Searching another horizon, threshold, magnitude bucket, fund-size/NAV interaction, QDII subset, code exception, or sell rule would be post-hoc mining. Exact historical QDII share-publication timing remains unproved.
Affected files: `cross_signal_strategy/research/share_flow_diagnostics.py`, `tests/test_cross_signal_share_flow_diagnostics.py`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/backtest_notes.md`, `cross_signal_strategy/docs/multiple_testing_audit.md`, `tests/test_cross_signal_multiple_testing_audit.py`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`
Allowed validation influence: none; only approved 2018 warm-up, 2019-2021 training prices, and the isolated 2018-2021 share-flow dataset were read
Status: diagnostic retained; candidate not created; research budget returned to zero; official strategy and production multi-factor files unchanged

### Harden PTrade Checkpoints And Recovery Gating

Date: 2026-07-17
Decision: Replace the formal cross-signal PTrade adapter's vulnerable single writable checkpoint with checksummed A/B generations, retain the old file only as a read-only migration source, block every new buy while any held position is unverified, and add a source-attributed startup recovery report. Keep `cross-v0.3.2` business logic, parameters, ETF pool, and JoinQuant/local implementations unchanged.
Reason: PTrade forbids `os`, so a rename-based atomic replacement is unavailable. Directly overwriting the only pickle can destroy the last valid state during a process or disk interruption. Compatible strategy redeployments should also be governed by a stable state schema rather than an exact strategy-version string, and an incompletely recovered portfolio must not add exposure.
Evidence: Tests were written and observed failing before each implementation stage. They cover alternating `.a`/`.b` generations, protocol-4 payloads, SHA256 rejection, truncated-newest fallback, compatible producer versions, unknown-schema rejection without partial restore, legacy migration, a portfolio-wide new-buy gate, continuity of verified-position exits, broker evidence-source tracking, and one startup audit line per holding. The complete formal PTrade adapter suite passes 97 tests after implementation.
Interpretation: A single damaged write no longer removes the previous valid generation. PTrade-restored `g`, strategy-only current-day fills, and account delivery records remain separate evidence layers whose source is visible at startup. The recovery gate is intentionally asymmetric: uncertainty blocks exposure expansion, while independently verified holdings can still reduce risk normally.
Risk: Local tests cannot prove the Guojin client filesystem's behavior during an actual terminal or machine crash. A simulation restart and one-slot-corruption drill remain mandatory. `get_deliver()` is account-wide, so a mixed manual/multi-strategy account may still fail ownership proof and remain `UNVERIFIED`; the strategy does not guess missing entry facts.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`, `docs/superpowers/specs/2026-07-17-ptrade-resilient-state-checkpoint-design.md`, `docs/superpowers/plans/2026-07-17-ptrade-resilient-state-checkpoint.md`
Allowed validation influence: none; this is platform reliability work and no market-period data or validation result was read
Status: adopted for the formal cross-signal PTrade adapter; production multi-factor files and all strategy rules remain unchanged

### Adopt Existing PTrade Account Positions On Strategy Handover

Date: 2026-07-17
Decision: Treat all broker positions inside the frozen cross-signal ETF pool as owned by the active strategy under an explicit one-account/one-active-strategy operating contract. Resolve the account/trade checkpoint path in `before_trading_start`, use documented `get_trading_day(0/-1)` as the primary current T-1 calendar source, and allow exact account-delivery reconstruction without requiring the historical buy to have been generated by a cross-signal entry. Keep exact quantity, fill price, buy date, pre-buy ATR, and post-entry close-history proof mandatory.
Reason: Guojin PTrade rejects `get_trade_name()` during `initialize`, so the checkpoint path was disabled before recovery began. The deployed account also contained three in-pool positions bought by the now-stopped multi-factor strategy. Requiring those historical buys to reproduce a cross-signal entry contradicted the user's operating model: the account never runs two strategies concurrently and receives no manual trades, so the newly active strategy must take over the account's existing positions.
Evidence: Tests were written and observed failing before implementation for deferred path resolution, `get_user_name(False)`, initialization-phase isolation, the documented `get_trading_day` path, unusable-calendar payload diagnostics, and adoption of an exact broker position whose historical buy score was ineligible for cross-signal. The complete formal PTrade adapter suite passes 100 tests after implementation.
Interpretation: Existing in-pool positions can now resume ATR and signal-based risk management after their delivery quantity exactly reconciles with the broker position. Their ATR is rebuilt from the trading day before the actual buy and their trailing peak from actual entry price and subsequent T-1 closes, so takeover introduces no future data. Same-day fills remain attributed through strategy-only `get_trades()`.
Risk: `get_deliver()` is account-wide. The rule is safe only while the operational contract remains true: one active strategy, the old strategy stopped before handover, and no manual account trades. Out-of-pool holdings, quantity mismatch, incomplete delivery history, missing price history, invalid broker cost, or unprovable calendar dates remain `UNVERIFIED` and block new exposure. The next Guojin simulation restart must confirm that `get_trade_name()` is accepted in `before_trading_start` and that the three inherited positions report `account-takeover:get-deliver`.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; this is PTrade lifecycle and account-state recovery work, and no market validation period or strategy-performance result was read
Status: adopted for the formal cross-signal PTrade adapter; JoinQuant/local business logic and production multi-factor files remain unchanged

### Instrument PTrade Account-Takeover Recovery Failures

Date: 2026-07-17
Decision: Add stage-specific, broker-safe diagnostics to the formal cross-signal PTrade recovery path while leaving every recovery gate and trading decision unchanged. When the existing historical calendar APIs cannot prove the day before an inherited buy, call documented `get_trading_day_by_date(query_date, -1)` only as a `non_binding=True` observation probe.
Reason: The first Guojin restart after account-takeover support successfully resolved the lifecycle phase, current T-1 date, and 140 account delivery records, but all three inherited positions still collapsed into the same generic `UNVERIFIED` message. That message could not distinguish delivery-field/quantity mismatch from historical-calendar, ATR, fill-price, or close-history failure.
Evidence: Tests were written first and observed failing because no replay summary, historical API probe, or stage-specific ATR rejection existed. The implementation logs expected and replayed quantity, record counts, buy/sell counts, date range, available ETF codes, direction labels, field names, and at most six date/side/quantity/price samples. Tests prove that account and shareholder-account values are not emitted, a valid historical probe is not adopted, score calculation does not continue after an unresolved calendar, and the holding remains `UNVERIFIED`. The complete formal PTrade adapter suite passes 103 tests.
Interpretation: The next Guojin restart can identify the exact failing boundary without weakening fail-closed behavior or guessing historical state. `delivery-replay` points to actual cabinet field/quantity parsing; `historical-calendar` distinguishes a usable documented by-date result from the currently unusable calendar path; later stages isolate entry ATR, fill price, and trailing-close reconstruction.
Risk: Diagnostics can establish where recovery fails but do not themselves prove that any alternate field or API should become authoritative. A subsequent behavior change still requires evidence from the new runtime log and a separate test-first fix. The bounded samples contain only security-level transaction facts, but deployment logs should still be handled as account-operational records.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; this is PTrade recovery observability work and no market-period data or validation result was read
Status: adopted as diagnostics only; JoinQuant/local business logic, frozen parameters and ETF pool, and production multi-factor files remain unchanged

### Normalize PTrade NumPy Calendar Strings

Date: 2026-07-17
Decision: Convert `numpy.str_` calendar scalars to native Python `str` at the shared PTrade date-normalization boundary before pandas parsing. Keep the documented calendar API order, historical-date comparison, recovery evidence gates, and non-binding probe policy unchanged.
Reason: Guojin runtime logs proved that both `get_trade_days` and `get_all_trades_days` returned correct ISO dates inside NumPy Unicode arrays, yet every element became unparseable. The independent `get_trading_day_by_date` probe returned the same preceding dates as native strings and parsed successfully.
Evidence: Local reproduction showed that the installed pandas raises `TypeError: Expected str, got numpy.str_` for `pd.Timestamp(np.str_('2026-07-15'))`. Tests were written first and failed for both `YYYY-MM-DD` and `YYYYMMDD` NumPy string scalars and for the exact `<U10 ndarray` payload observed in PTrade. The one-line scalar normalization makes all three cases pass and proves that the first documented calendar API succeeds without invoking either fallback API or the non-binding probe. The complete formal PTrade adapter suite passes 106 tests.
Interpretation: The three inherited holdings had already passed delivery replay; their false `historical-calendar` rejection was a platform return-type compatibility defect, not missing market data or an unproved buy date. Recovery can now advance to the existing ATR, fill-price, and trailing-close evidence gates using the correct T-1 date.
Risk: This fix removes only the demonstrated calendar parsing blocker. The next PTrade restart may expose a later independent recovery failure, which must be diagnosed from its stage-specific log rather than bypassed. No claim is made that all three positions are verified until runtime reports `status=VERIFIED`.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; this is PTrade platform type compatibility work and no market-period data, strategy return, or validation result was read
Status: adopted for the formal cross-signal PTrade adapter; JoinQuant/local business logic, frozen parameters and ETF pool, and production multi-factor files remain unchanged

### Localize Formal PTrade Logs To Chinese

Date: 2026-07-17
Decision: Translate every strategy-authored log template in the formal cross-signal PTrade adapter into Chinese, including initialization, calendar, scoring, order submission, halt recovery, checkpoint recovery, broker takeover diagnostics, IOPV observation, and order/trade callbacks. Preserve official PTrade API names, ETF/indicator abbreviations, raw broker exception text, internal enum values, and all business logic.
Reason: The deployed strategy is operated and diagnosed in a Chinese PTrade client. Mixed English log tags and field names made live recovery review slower and increased the chance of searching for the wrong message after a restart.
Evidence: Tests were written first and observed failing against 115 English direct log templates and missing Chinese formatter behavior. The implementation adds output-only translation helpers, a static AST log contract, dynamic formatter assertions, and updated runtime-log assertions. The complete formal PTrade adapter suite passes 108 tests after localization.
Interpretation: Strategy-generated operational logs now use Chinese labels while documented API identifiers such as `get_deliver`, `get_trades`, and `get_trade_days` remain searchable against Guojin documentation. External exception and broker rejection text is displayed verbatim because altering it would discard diagnostic evidence.
Risk: This change does not translate messages produced by the PTrade platform itself or arbitrary text returned by broker APIs. Those values are external evidence rather than strategy-authored log templates. Chinese localization must remain an output-layer concern and must not be applied to persisted keys, order statuses, recovery sources, or strategy decisions.
Affected files: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`, `tests/test_cross_signal_ptrade_strategy.py`, `cross_signal_strategy/docs/ptrade_deployment.md`, `cross_signal_strategy/docs/decisions.md`
Allowed validation influence: none; this is PTrade observability work and no market-period data, strategy return, or validation result was read
Status: adopted for the formal cross-signal PTrade adapter; JoinQuant/local strategy logic, frozen parameters and ETF pool, and production multi-factor files remain unchanged

### Archive Cross-Signal Files By Operational Role

Date: 2026-07-17
Decision: Keep only the three formal Python entry files in the `cross_signal_strategy` root: JoinQuant, PTrade, and local training replay. Move formal local support into `local/`, training-only diagnostics and governance tools into `research/`, rejected candidates into `archive/candidates/`, no-order probes into `archive/probes/`, and chart generation into `tools/`. Keep all automated tests in the repository-level `tests/` directory.
Reason: The root had accumulated formal deployment files, reusable local infrastructure, rejected experiments, diagnostics, and temporary probes. Similar filenames made it too easy to deploy or edit a candidate instead of the frozen mainline.
Evidence: A test-first directory contract requires exactly three root Python files, explicit role directories, importable package markers, representative archived files, and README documentation. All internal imports, test imports, executable paths, and current documentation references were updated to the new locations. File contents were moved with Git history preserved; formal JoinQuant and PTrade strategy files were not edited.
Interpretation: Deployment choices are now visually unambiguous while research evidence remains available and testable. Archiving means “not a formal entry point”; it does not erase failed experiments or their historical evidence.
Risk: External scripts or personal commands that imported the old flat module paths must switch to the new package paths. The repository tests cover maintained imports, but unknown out-of-repository callers cannot be discovered locally.
Affected files: `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/README.md`, `cross_signal_strategy/docs/decisions.md`, moved modules under `cross_signal_strategy/local/`, `cross_signal_strategy/research/`, `cross_signal_strategy/archive/`, and `cross_signal_strategy/tools/`, plus corresponding repository tests and path references
Allowed validation influence: none; this is repository organization work and no market data, strategy return, or validation-period result was read
Status: adopted as a repository-layout milestone; strategy logic, parameters, ETF pool, market data, production multi-factor files, and generated reports remain unchanged
### Retain The Three-Trading-Day Cross Window

- Decision: Keep the formal `cross_window=3` unchanged after the locked local training comparison of windows `1/2/3/4`.
- Evidence: On 2019-2021 only, window 3 produced the highest total and annualized return, the lowest maximum drawdown, and the strongest Sharpe, Sortino, and profit/loss ratio. Windows 1, 2, and 4 all failed the pre-registered strict non-degradation gate.
- Data boundary: 2018 was used only as read-only indicator warm-up. No reserved validation-period data was read or used for this decision.
- Platform boundary: The alternatives failed locally, so no JoinQuant candidate was run and no formal JoinQuant, PTrade, or local-mainline strategy file was changed.
- Overfitting control: Treat the `1/2/3/4` comparison as the complete neighboring integer-window budget. Do not continue searching wider or per-indicator windows from these outcomes.

### Retain 09:35 After The Fixed 10:00 Execution Comparison

- Decision: Keep formal execution at `09:35`; reject the single pre-registered `10:00` candidate and close the execution-time family.
- Evidence: On 2019-2021 local training, `10:00` improved total return from +120.61% to +127.65% and reduced drawdown from 7.47% to 7.15%, but it lowered 2021 return from +8.46% to +7.85% and profit/loss ratio from 4.440 to 4.413. Across 135 matched orders, side-adjusted execution was slightly worse overall, worse in 2019 and 2021, and worse for non-QDII ETFs.
- Interpretation: The aggregate gain came from 78 changed downstream order days, not a stable opening-noise execution advantage. Choosing the higher total-return clock would violate the pre-registered mechanism and annual-consistency gate.
- Data boundary: Only approved 2018 warm-up and 2019-2021 training data were read. No reserved validation period was inspected.
- Overfitting control: Do not search nearby times, VWAP windows, ETF-specific clocks, QDII-only clocks, or timing interactions. Reopening requires prospective evidence with a separately reserved confirmation sample.
- Platform boundary: No JoinQuant candidate was run and no formal JoinQuant, PTrade, or local-mainline strategy file was changed.

### Add A Causal Training Trade-Quality Ledger And Block Unproved Underlying Filters

- Date: 2026-07-18.
- Decision: Add an observation-only ledger for every closed 2019-2021 local-mainline trade, using actual 09:35 fills, causal holding-path boundaries, fixed 5/10-day excursion labels, a fixed 10-day one-ATR first-barrier label, and post-sell labels. Keep all formal strategy files frozen.
- Evidence: Tests were written first and observed failing for the missing module, the post-09:35 sell-close boundary, an incomplete ATR-label window, training-period leakage, signal-date ordering, cross-year data assembly, and QDII classification. Eight ledger tests, seven adjacent diagnostic tests, and all 536 repository tests pass. The final training replay produced 89/89 ledger rows: 57.30% win rate, +3.11% average trade return, 77.53% profitable within the first three closing observations, 62.92% `up_first`, and 30.34% `down_first`.
- Interpretation: QDII trades had 62.16% win rate and +3.77% average return versus 53.85% and +2.63% for non-QDII, but also a materially higher `down_first` rate (40.54% versus 23.08%). This motivates an independent source-market consistency check; it does not justify a price threshold or a QDII penalty from the ETF series itself.
- Data audit: The approved training root contains only the ETFs' own daily/minute histories and metadata. It does not contain Nasdaq-100, S&P 500, CSI Overseas China Internet 50, Nikkei 225, source-market calendars, point-in-time FX, or publication timestamps. No underlying-market experiment was run and no candidate was created.
- Future-function control: Buy-day closes and later prices exist only as ex-post ledger labels. Sell-day close is excluded from holding MFE/MAE because the position exits at 09:35. Japan/Hong Kong T-day closes are explicitly forbidden at the China 09:35 decision time; source sessions must be aligned by their own calendars.
- Overfitting control: Do not mine ledger horizons, ATR multipliers, quantiles, or QDII-specific thresholds. A single pre-registered underlying observation may open only after an explicitly authorized, independent, read-only 2019-2021 source dataset proves dates, time zones, publication times, and required FX.
- Affected files: `cross_signal_strategy/research/trade_quality_ledger.py`, `tests/test_cross_signal_trade_quality_ledger.py`, `cross_signal_strategy/docs/trade_quality_ledger.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`.
- Allowed validation influence: none; only approved 2019-2021 ETF training data, 2018 indicator warm-up inside the unchanged replay, and public fund/exchange documentation were read.
- Status: ledger retained; underlying-market observation blocked on missing independent data; formal JoinQuant, PTrade, local-mainline, and production multi-factor files unchanged.

### Pre-register QDII Underlying Direction And Keep It Data-blocked

- Date: 2026-07-18.
- Decision: Pre-register exactly one observation for the four formal QDII ETFs: at China 09:35, compare the two most recent final official underlying-index closes whose historical `available_at` is no later than the decision time. Group strictly by positive versus non-positive one-session return. Do not create a strategy candidate yet.
- Hypothesis: A positive move in the independently traded underlying market may confirm that an ETF oscillator cross reflects external price discovery rather than only local ETF noise or premium movement.
- Data contract: `513100/NDX`, `513500/SPX`, `513050/H30533`, and `513880/N225`; source sessions may span different calendars and time zones. The exact read-only root is `G:\financial\history_data\cross_signal_underlying_train_2018_2021`. Only 2018 warm-up and 2019-2021 training sessions are accepted, and every row requires a final value plus timezone-aware historical availability.
- Frozen gate: coverage at least 90% and 30 trades; both groups at least 10; both groups at least 3 in each training year with confirmed average return and win rate higher every year; at least three ETF-level comparisons with two trades per group and no dual-metric contradiction; aggregate return and win rate must also be higher.
- Evidence: Tests were written first and observed failing for the absent source-contract and observation modules. The completed modules pass 22 focused tests covering exact-root isolation, source mapping, finality, session duplicates, timezone awareness, Japan same-day and malformed future-session exclusion, 2018 warm-up, training boundaries, sign grouping, missing coverage, and frozen candidate gates. The approved source root audit returned `exists=False`.
- Interpretation: This is a genuinely independent market-structure question, not another indicator or premium threshold search. It remains blocked because the required point-in-time official-index evidence is absent; ETF prices cannot stand in for the underlying market.
- Overfitting and future-function control: No observation result exists, no validation period was read, and no threshold may be changed after data arrives. `available_at` must reflect historical availability, not today's download timestamp or an assumed market close. Japan/Hong Kong/US calendar alignment is selected by timestamps rather than natural-date shifting.
- Affected files: `cross_signal_strategy/research/underlying_market_data.py`, `cross_signal_strategy/research/underlying_consistency.py`, `tests/test_cross_signal_underlying_market_data.py`, `tests/test_cross_signal_underlying_consistency.py`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/underlying_market_direction.md`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `cross_signal_strategy/docs/decisions.md`, `cross_signal_strategy/README.md`.
- Allowed validation influence: none.
- Status: pre-registered and blocked on missing independent data; formal JoinQuant, PTrade, local-mainline, and production multi-factor files unchanged.

### Stage Underlying Raw Values But Keep Point-In-Time Observation Blocked

- Date: 2026-07-18.
- Decision: Add a locked acquisition layer for the four pre-registered underlying indices. Download only 2018-2021 raw final values to a separate staging root, hash every normalized CSV, and refuse to publish the formal bundle unless all four source-specific historical availability policies are approved.
- Evidence: Tests were written first and failed for the absent source registry, parsers, locked request ranges, availability policies, all-or-nothing bundle gate, staging guard, hashes, and acquisition runner. Eleven focused acquisition tests, 48 adjacent contract/governance tests, and all 571 repository tests pass. The real staging audit contains `513050=1044`, `513100=1009`, `513500=1008`, and `513880=974` valid rows, with zero duplicate dates, invalid dates, missing/non-positive closes, or dates outside 2018-2021.
- Point-in-time status: `NDX` uses the Nasdaq end-of-day correction cutoff at 17:15 New York time; `N225` uses the official Nikkei daily update around 16:00 Tokyo time. `SPX` and `H30533` remain blocked because their exact historical final-value availability has not been proved. Download time is metadata only and is never copied into `available_at`.
- Interpretation: Historical value coverage is solved, but the causal availability contract is not. The formal approved root remains absent, no consistency report was run, and no strategy candidate exists.
- Overfitting and future-function control: Requests are hard-limited to 2018 warm-up plus 2019-2021 training dates. Missing source-market sessions are dropped rather than filled. No validation-period data or result was read, and no score, indicator, threshold, ETF pool, order, or risk rule changed.
- Affected files: `cross_signal_strategy/research/underlying_source_acquisition.py`, `cross_signal_strategy/tools/fetch_underlying_sources.py`, `tests/test_cross_signal_underlying_source_acquisition.py`, `tests/test_cross_signal_research_budget.py`, `cross_signal_strategy/docs/underlying_source_acquisition.md`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, `cross_signal_strategy/docs/decisions.md`, and `cross_signal_strategy/README.md`.
- Status: raw values staged and audited; point-in-time observation remains blocked; formal JoinQuant, PTrade, local-mainline, and production multi-factor files unchanged.

### Probe JoinQuant Underlying-Index Readability Without Publishing Availability

- Date: 2026-07-18.
- Decision: Add one isolated, no-order JoinQuant probe for `513500` and `513050`. On four fixed 2019-2021 dates at 09:35, it discovers each ETF's tracked index through `finance.FUND_INVEST_TARGET`, attempts to read two daily closes ending at China T-1, and attempts a same-day close as a future-data negative control.
- First-run evidence: On all four dates, `513500` metadata identified the S&P 500 Net Total Return Index but returned an empty `traced_index_code`, so no price request was possible. `513050` returned `H30533.CSI` and the T-1 API calls completed, but all returned `close` values were `NaN`; call success therefore did not establish usable data. Every same-day close request raised `FutureDataError`, confirming that `avoid_future_data=True` blocked the negative control at 09:35.
- Follow-up decision: Strengthen the same isolated probe with a second-stage diagnostic. It now reports call success separately from finite-value usability, checks `get_security_info` and historical index registration, queries an explicit 14-calendar-day OHLC range, and cross-checks `attribute_history`. Finite rows from a different source-market calendar remain usable evidence, while requested-end-date presence is logged separately rather than required.
- Second-run evidence: `SPTR500N` has no supported code or name match in the historical JoinQuant index registry on any of the four dates. `H30533.CSI` is metadata-visible but has no finite OHLC: it is typed as `csi`, is absent from `get_all_securities(types=["index"])`, has no name match, and returns zero finite closes through count-based `get_price`, explicit-range `get_price`, and `attribute_history`. The same-day negative control remained blocked by `FutureDataError` on every date.
- Diagnostic correction: `from jqdata import *` shadows Python's built-in `any` in JoinQuant, so the first second-stage run rendered `requested_end_present` as a generator object. The probe now uses an explicit list-length check. This affected one descriptive boolean only; it did not affect `finite_close_count=0`, `data_usable=False`, index registration, or the platform-capability conclusion.
- Platform-capability conclusion: JoinQuant cannot supply usable historical `SPTR500N` or `H30533` values through the tested metadata and行情 interfaces. Further JoinQuant probing is closed unless a new documented API or supported identifier appears. This result still cannot prove publisher-level historical availability, so no formal data bundle or strategy candidate may be released from it.
- Evidence boundary: A successful T-1 read proves only that the JoinQuant historical backtest exposes that row at the simulated decision time with `avoid_future_data=True`. It does not expose the index publisher's first-release timestamp, JoinQuant's historical ingestion timestamp, or a point-in-time unrevised snapshot.
- Failure interpretation: An empty `traced_index_code`, unsupported index identifier, metadata error, or price-query error is a platform-capability result. The probe must not replace the underlying index with the ETF price or another proxy.
- Overfitting and future-function control: The callback is hard-limited to four fixed training dates, places no trades, reads no validation-period data, and changes no signal, threshold, pool, ranking, position, stop, or execution rule.
- Status: second-stage diagnostic complete; 513050 and 513500 remain blocked from the formal `available_at` bundle until publisher-level evidence or an explicitly approved contract change exists. No formal JoinQuant, PTrade, local-mainline, or production multi-factor strategy changed.

### Add A Reproducible Cross-Signal Release Identity And Gate

- Date: 2026-07-18.
- Decision: Keep every `cross-v0.3.2` business rule frozen while adding deployment build `20260718.1`, a 12-character SHA256 business-configuration fingerprint, dynamic formal-version log labels, and a read-only release verifier. The fingerprint covers the strategy version, sorted frozen parameters, and the normalized nine-ETF pool; platform code suffixes are deliberately excluded.
- Test-first evidence: New tests first failed on the stale `v0.1` source/log labels, missing build and fingerprint, absent verifier, and outdated Tuesday/Thursday documentation. After implementation, the focused suite passed 175 tests. The final `python cross_signal_strategy/tools/verify_release.py --run-tests` check passed with business fingerprint `1506a0e834fe`, 529 tests passed, and 78 unrelated tests deselected.
- Release checks: three formal entries exist and parse; JoinQuant/PTrade version, build, parameters and normalized ETF pool match; all frozen pure-business functions remain AST-identical; PTrade does not use the forbidden `os` module; live-state schema version is positive; formal source contains no stale `[cross-v0.1]` label.
- Interpretation: This change makes copied PTrade deployments independently identifiable and turns the existing parity contracts into one repeatable command. The fingerprint and verifier are observation-only and are never read by signal scoring, ranking, position sizing, stop calculation, scheduling, or order submission.
- Affected files: formal JoinQuant and PTrade adapters, `cross_signal_strategy/tools/verify_release.py`, release/deployment documentation, and repository tests.
- Allowed validation influence: none; no market data, return series, training result, or reserved validation result was read or used.
- Status: adopted as a release-engineering milestone; strategy logic, parameters, ETF pool, execution time, risk rules, local replay, and production multi-factor files remain unchanged.

### Reject The Pre-Registered Ordinary-Buy Minute Execution Overlay

- Date: 2026-07-18.
- Decision: Consume the one user-authorized `intraday_execution_overlay_v1` budget without creating a formal strategy candidate. Freeze each official 09:35 ordinary-buy code and quantity, test one arrival-price passive limit for six five-minute cycles with a first-executable-minute fallback at or after 10:05, and stop at the counterfactual gate.
- Test-first evidence: Execution semantics and budget constraints were written as failing tests before implementation. The final focused suite passed 30 tests, and the release verifier passed 538 tests with 78 unrelated tests deselected. The complete training run matched all 92 eligible ordinary buys, with 75 passive fills and 17 fallbacks.
- Training evidence: Average signed execution improvement was +2.63 basis points overall, split +1.02/-0.78/+6.73 basis points in 2019/2020/2021 and +4.12/+0.40 basis points for non-QDII/QDII. The locked positive-in-every-year gate failed in 2020.
- Interpretation: The aggregate improvement is too small and regime-dependent to justify a new order path. Running a full portfolio candidate or searching a nearby execution schedule after this result would be post-hoc selection.
- Allowed validation influence: none; only approved 2019-2021 minute data and the 2018 read-only warm-up were used. Reserved validation periods were not read.
- Affected files: isolated local execution helper, isolated research report, tests, and research-governance documents only.
- Status: rejected and exhausted; formal JoinQuant, PTrade, local-mainline, and production multi-factor strategy files remain unchanged.

### Record The Failure-Year Fragility Atlas Without Reopening Research

- Date: 2026-07-18.
- Decision: Add a training-governance atlas that parses all 53 retained failed or non-adopted experiments and manually annotates only the 13 records that preserve explicit annual gate contradictions. Missing annual evidence remains unreported rather than inferred. This does not authorize a strategy change.
- Test-first evidence: Seven focused tests first failed for the absent parser, annotations, report, documentation index, and decision record. The parser excludes the empty ledger template, verifies 53 unique real entries, rejects unknown IDs and non-training years, and requires non-empty evidence for every annotation.
- Training evidence: Explicit contradiction counts are 7/6/10 for 2019/2020/2021. The counts are not performance scores and one experiment may contribute more than one year. The formal local mainline annual returns remain +35.84%/+49.74%/+8.46%, so 2020 is not a mainline weak year.
- Interpretation: The most common annotated cause is market-regime reversal. The 2020 minute-overlay failure is separately classified as execution-tail risk: 23 of 29 buys improved and the median was +9.52 basis points, but one `513500` fallback on 2020-02-03 worsened by 119.45 basis points and pulled the unweighted annual mean below zero. The more frequent 2021 contradictions show that many confirmation variables reverse meaning across regimes; they do not justify 2021-specific tuning.
- Research boundary: Keep `cross-v0.3.2` frozen. Do not reopen indicator periods, thresholds, cross windows, execution clocks, limit offsets, or ETF exceptions from this atlas. The only pre-registered independent direction remains QDII underlying-index direction, which stays blocked until official final-value and historical `available_at` evidence is complete.
- Data boundary: The atlas reads only `failed_experiments.md` and its curated annotation file. It opens no market-data directory and reads no validation-period result.
- Affected files: `cross_signal_strategy/research/failure_year_atlas.py`, `tests/test_cross_signal_failure_year_atlas.py`, `cross_signal_strategy/docs/failure_year_fragility_annotations.json`, `cross_signal_strategy/docs/failure_year_fragility_atlas.md`, `cross_signal_strategy/docs/README.md`, and `cross_signal_strategy/docs/decisions.md`.
- Status: adopted as research-governance documentation; formal JoinQuant, PTrade, local-mainline, ETF pool, parameters, execution rules, and production multi-factor files remain unchanged.

### Close The Publisher Evidence Audit Without Unlocking QDII

- Date: 2026-07-18.
- Decision: Keep `513500/SPX` and `513050/H30533` absent from `APPROVED_AVAILABILITY_POLICIES`. Do not create the formal underlying-index root and do not run the pre-registered direction observation.
- S&P DJI evidence: The official Equity Indices Policies & Practices states that official EOD prices are validated before index-file distribution, but incorrect closes, missed corporate actions, and calculation/data errors may trigger recalculation. Events found within two trading days are generally recalculated and reposted; later events can still be decided by the Index Committee. The document provides neither an immutable finality cutoff nor row-level historical distribution timestamps for 2018-2021.
- CSI evidence: The official H30533 methodology confirms a cross-market Hong Kong/other-overseas constituent set. CSI's stock-index calculation rules state only that closing indices are published every index trading day; the official documents do not specify H30533's exact publication clock, timezone, immutable finality cutoff, or a historical 2018-2021 publication SLA.
- Root-cause conclusion: Raw close coverage is not the blocker. The missing fact is historical point-in-time final availability. Ordinary exchange closes, current download timestamps, FRED observation dates, natural-date shifts, or later-visible official histories cannot prove that fact.
- Reopen condition: New primary publisher evidence must supply historical point-in-time distribution records or an explicit final-value cutoff applicable to 2018-2021. Repeating web searches or adopting a conservative guessed delay does not qualify.
- Test-first evidence: A focused documentation test first failed because the official evidence and exact blockers were absent. The acquisition gate itself remained closed throughout.
- Allowed validation influence: none; this audit used official methodology/governance documents only and read no new market series or validation-period result.
- Affected files: `tests/test_cross_signal_underlying_source_acquisition.py`, `cross_signal_strategy/docs/underlying_source_acquisition.md`, `cross_signal_strategy/docs/underlying_market_direction.md`, `cross_signal_strategy/docs/research_budget.md`, and `cross_signal_strategy/docs/decisions.md`.
- Status: evidence audit closed without unlock; formal JoinQuant, PTrade, local-mainline, ETF pool, parameters, orders, risk rules, and production multi-factor files remain unchanged.

### Adopt A Prospective PTrade Log Archive Without Opening Research

- Date: 2026-07-18.
- Decision: Add an external, content-addressed archive for future exported PTrade logs. Bind every accepted file to formal `cross-v0.3.2`, deployment build `20260718.1`, and business fingerprint `1506a0e834fe`; reject pre-protocol, unidentified, or mixed-release input.
- Test-first evidence: The parser, fail-closed identity checks, immutable source behavior, idempotent SHA256 archive, manifest boundary, CLI release lookup, and documentation contract were written as failing tests before implementation. The focused suite passes after implementation.
- Interpretation: Existing PTrade logs already contain release identity, 09:35 execution boundaries, orders, fills, IOPV observations, and 10:35 recovery events. Preserving those exports provides a prospective evidence chain without adding a scheduler, market-data request, or strategy-side log call.
- Research boundary: The manifest stores timestamps, dates, hashes, and event counts only; it does not calculate prices, returns, win rates, or signal quality. Prospective collection does not reopen any exhausted family. Before future analysis, first freeze a hypothesis; logs already seen are discovery material, and only later continuous logs may be an independent confirmation sample.
- Data and validation boundary: No market-data root or validation-period result is read. The archive cannot be used to tune against validation periods or to create ETF-specific exceptions after seeing outcomes.
- Affected files: `cross_signal_strategy/research/prospective_log_archive.py`, `cross_signal_strategy/tools/archive_ptrade_forward_logs.py`, `tests/test_cross_signal_prospective_log_archive.py`, `cross_signal_strategy/docs/prospective_live_log_protocol.md`, `cross_signal_strategy/docs/README.md`, `cross_signal_strategy/docs/research_budget.md`, and `cross_signal_strategy/docs/decisions.md`.
- Status: adopted as evidence preservation only; formal JoinQuant, PTrade, local-mainline, parameters, ETF pool, schedules, orders, risk rules, and production multi-factor files remain unchanged.

### Harden Cross-Signal Live Engineering Without Changing Business Rules

- Date: 2026-07-18.
- Decision: Release deployment build `20260718.2` with the `cross-v0.3.2` business configuration frozen. Require same-day timestamps for executable live snapshots, perform a read-only `get_open_orders()` audit in `after_trading_end`, bind A/B checkpoints to business fingerprint `1506a0e834fe`, reject malformed callback records, and log returned broker order IDs.
- Test-first evidence: Each production change was preceded by a failing focused test. Added contracts cover stale/current-session snapshots, after-close order auditing without guard mutation, fingerprint mismatch rejection, malformed callback handling, broker order-ID logs, and a same-synthetic-day JoinQuant/PTrade buy-selection comparison.
- State migration: PTrade state schema advances from `1` to `2`. Old incompatible A/B envelopes are not partially restored. Existing holdings must be reconstructed through the already fail-closed broker delivery/current-trade evidence path before automatic exits or new exposure are allowed.
- Test-environment decision: Windows ACL inspection proved the inaccessible pytest directories are owned by the current user; the denial is caused by the restricted execution token used for this review. The release verifier now disables pytest's repository cache and labels `PermissionError`/`WinError 5` as a test-environment failure instead of a strategy assertion failure.
- Business boundary: No signal, indicator, threshold, ETF pool, ranking, position size, stop formula, rebalance weekday, or execution time changed. No market data or validation-period result was read. The production multi-factor strategy was not modified.
- Verification: The static release gate passed for all three formal entry points, release identity, syntax, business fingerprint, JoinQuant/PTrade pure-function parity, PTrade `os` prohibition, and state schema. The complete cross-signal suite passed `567` tests with `78` unrelated tests deselected.
- Status: adopted as a release-engineering milestone; automated verification is complete, while deployment-log confirmation for build `20260718.2` remains pending.

### Replace A/B Checkpoints With Broker-First State Journal

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.1` without changing `cross-v0.3.2` business rules. Replace per-instance A/B checkpoint files with one account/trade-scoped append-only state journal. Bind each record to the current broker position snapshot and business fingerprint. On every process start, reconstruct risk state from current-strategy fills, delivery records, and broker positions first; use the matching journal only for intraday continuity and old holdings that broker history cannot prove.
- Reason: PTrade manual stop/start does not reliably preserve the platform `g` object, and a generated instance UUID changed the external path on restart. Delivery history is strong broker evidence but the official API does not promise unlimited retention or completeness. The journal therefore remains a necessary fallback, while broker-first reconstruction prevents it from overriding stronger current facts.
- State contract: PTrade state schema advances from `2` to `3`. The journal now persists sell-retry reasons and a normalized code/quantity/cost snapshot. Any position-set, quantity, cost, schema, fingerprint, checksum, or required-field mismatch rejects recovery. A truncated tail leaves prior records readable; the next save removes only incomplete tail bytes before appending.
- Safety boundary: A journal fallback may fill only an incomplete held position and may not overwrite a position already reconstructed from broker evidence. If neither source proves buy date, entry ATR, and highest close, the position remains unverified and new exposure stays blocked. No inferred dates, ATR percentages, peaks, or validation-period data are introduced.
- Business boundary: Signal calculations, indicators, thresholds, ETF pool, ranking, position sizing, 09:35/10:35 schedule, stop rules, and JoinQuant performance behavior are unchanged. The production multi-factor strategy is untouched.
- Test-first evidence: Failing tests were added for restart-stable path identity, one-file journal generations, broker-snapshot accept/reject behavior, broker-first ordering, journal-only old-position fallback, sell-retry persistence, and truncated-tail repair before each implementation change.
- Status: adopted as a live-engineering milestone pending one PTrade simulation restart log for build `20260720.1`.

### Prefer Broker-Validated PTrade G State On Restart

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.2` with the `cross-v0.3.2` business configuration unchanged. On a restart of the same PTrade strategy, first consider the framework-restored ordinary `g` state before querying delivery history.
- Acceptance contract: `g` is usable only when its state schema, business fingerprint, positive generation, complete buy-date/ATR/highest-close fields, and recorded normalized code/quantity/cost snapshot all match current broker holdings. Current broker positions remain the source of truth; `g` is only a broker-bound cache of derived strategy state.
- Freshness contract: The explicit journal and `g` share monotonically increasing generations. When a matching journal has a higher generation, reject the older `g` state and continue through broker reconstruction plus journal fallback. This prevents a stale highest close, ATR, sell retry, or deferred state from replacing a fresher record.
- Fallback contract: Missing, malformed, mismatched, incomplete, or future-dated `g` state does not block the existing recovery chain. The strategy clears untrusted risk fields, queries current trades and delivery records, uses a matching journal only for remaining gaps, and leaves unproved holdings `UNVERIFIED`.
- Deployment boundary: PTrade persists ordinary `g` fields only for the same strategy record. A newly created independent strategy does not inherit the stopped strategy's `g`; account delivery evidence and an identity-compatible journal remain necessary for that handover case.
- Test-first evidence: Focused tests failed before implementation for missing `g` validation, missing broker-bound metadata, unwanted delivery-history queries despite valid `g`, changed-position rejection, and newer-journal precedence. The PTrade test file passed after the minimal implementation.
- Business boundary: No indicator, signal, threshold, ETF pool, ranking, position size, execution time, stop rule, or JoinQuant performance behavior changed. No market data or validation-period result was read, and the production multi-factor strategy was not modified.
- Status: adopted as a live-engineering reliability milestone pending one same-strategy PTrade restart log for build `20260720.2`.

### Report The Actual PTrade Recovery Source

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.3` without changing the frozen `cross-v0.3.2` business configuration. Derive the recovery-summary source from each current holding's verified source instead of defaulting an absent overall source to `ptrade-g`.
- Reason: A newly created PTrade strategy correctly reconstructed all holdings from delivery records, but the overall summary incorrectly displayed `PTrade持久状态` with no generation. Per-position logs were correct; the aggregate label was misleading during a high-risk recovery audit.
- Source contract: If all held positions share one source, report that source. If held positions use different sources, report `混合恢复`. If there are no positions and no restored state, report `无持仓`. A missing or invalid persistent-state generation is rendered as `不适用`.
- Test-first evidence: Failing tests first reproduced both the all-delivery mislabel and a mixed-source case. Existing journal-source coverage remains in force.
- Business boundary: No indicator, signal, threshold, ETF pool, ranking, position size, execution time, order behavior, stop rule, recovery calculation, or JoinQuant performance behavior changed. Business fingerprint remains `1506a0e834fe`; the production multi-factor strategy is untouched.

### Separate PTrade Framework, Continuity, And Position Recovery Diagnostics

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.4` without changing the frozen `cross-v0.3.2` business configuration. Split startup recovery reporting into `[PTrade框架g]`, `[连续状态恢复]`, and `[持仓风险恢复]` so one line no longer combines unrelated evidence dimensions.
- Reason: A same-strategy restart can have no framework-restored ordinary `g`, recover continuity from the append-only journal, and reconstruct every held position from delivery records. Reporting only `账户接管:交割单 代次=3` was factually incomplete because the delivery record proved position risk facts while generation 3 belonged to the journal.
- Diagnostic contract: Ordinary `g` reports `未提供`, `已接受`, `已拒绝`, or `已接受但未采用`, with a Chinese reason and its own generation. Continuity reports its source and generation independently. Position recovery aggregates the actual per-position evidence source and continues to list every holding's verification details.
- Audit contract: The read-only PTrade runtime-log auditor now requires all three diagnostic dimensions and still fails closed on any `未验证` holding. A rejected ordinary `g` is not itself a trading failure when journal or broker evidence completes recovery.
- Test-first evidence: Seven focused tests first failed for missing ordinary-`g` diagnostics and the old combined summary; a separate newer-journal test failed before recording the superseded state. The runtime-log auditor tests then failed against the new format before its parser was updated.
- Business boundary: Recovery priority, broker reconstruction, delivery replay, journal fallback, signals, indicators, thresholds, ETF pool, orders, stops, and schedules are unchanged. Business fingerprint remains `1506a0e834fe`; the production multi-factor strategy is untouched.

### Treat Successful Broker-Fact Takeover As Information

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.5` without changing the frozen `cross-v0.3.2` business configuration. Log a fully verified delivery-record account takeover at `INFO` instead of `WARNING`.
- Reason: The runtime-log auditor correctly treats every warning as requiring review. Build `20260720.4` restored and verified all three holdings, but its three successful takeover messages alone forced the daily audit to `需复核`, obscuring the absence of any real warning or error.
- Severity contract: Only the final successful takeover message changes severity. Delivery-query failures, malformed records, incomplete calendars, invalid ATR/high/cost facts, unverified holdings, and blocked trading retain their existing warning or error levels.
- Test-first evidence: The existing broker-fact takeover test was extended first to require the success message in `INFO` and prohibit it in `WARNING`; it failed against build `20260720.4` before the one-line implementation change.
- Business boundary: Recovery calculations, source priority, journal behavior, signals, indicators, thresholds, ETF pool, orders, stops, and schedules are unchanged. Business fingerprint remains `1506a0e834fe`; the production multi-factor strategy is untouched.

### Complete PTrade Recovery And Log Semantics Closeout

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.6` without changing the frozen `cross-v0.3.2` business configuration. A complete state journal that still matches the current broker code, quantity, and cost snapshot may restore all held-position risk fields directly; incomplete or future-dated journal state continues through current fills, delivery records, broker reconstruction, and journal gap fallback.
- Reason: Re-reading the complete delivery history on every same-strategy restart adds latency and noisy startup logs even when the append-only journal has already been checksum-, schema-, fingerprint-, generation-, and broker-snapshot-validated. Direct use is safe only after independently proving buy date, positive entry ATR, positive highest close, and no unverified held code for every current broker holding.
- Log contract: The observation-only sell score message is named `卖出风险观察` in PTrade and `sell-risk-observation` in JoinQuant, and explicitly states that it does not tighten the stop. Known internal journal validation errors are rendered as Chinese diagnostics; platform exception details remain attached when they are not generated by this strategy.
- Test-first evidence: Focused tests first failed for complete-journal delivery bypass, newer-journal precedence, incomplete-journal broker fallback, Chinese state-error formatting, the observation-only log name, and the new build identity. The implementation then passed the complete PTrade test file.
- Business boundary: No indicator, signal calculation, threshold, ETF pool, ranking, position size, execution time, order condition, stop formula, or performance behavior changed. Business fingerprint remains `1506a0e834fe`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Harden PTrade Fills, Journal Writes, And Snapshot Freshness

- Date: 2026-07-20.
- Decision: Release deployment build `20260720.7` without changing the frozen `cross-v0.3.2` business configuration. Deduplicate broker trade pushes by non-empty `business_id`, reuse a verified state-journal tail, suppress identical journal snapshots, and reject live snapshots that cannot be proved current to the second or are more than 300 seconds old.
- Fill contract: Matching still requires the current pending `order_id`. Within that pending order, a repeated non-empty成交编号 is ignored before quantity or value accumulation. Missing成交编号 keeps the previous conservative behavior because inventing an identity would risk dropping distinct fills.
- Journal contract: The first access validates the complete append-only journal. The process then caches path, file length, latest generation, and payload digest. An unchanged state plus broker snapshot keeps the existing generation; changed state appends without a full rescan. Any file-length mismatch or damaged tail invalidates the cache and preserves the existing full scan, truncation repair, checksum, schema, fingerprint, and broker-snapshot checks.
- Snapshot contract: Live `hsTimeStamp` must contain a full second-level timestamp, must not be in the future, and must be at most five minutes old. Rejection remains fail-closed and affects only the use of abnormal execution data; T-1 signals and all scoring rules are unchanged.
- Test-first evidence: Duplicate buy/sell fill tests first failed by counting 200 shares twice. Journal tests first failed because an identical save doubled the file and a changed second save rescanned it. Snapshot tests first failed because date-only, ten-minute-old, and future timestamps were accepted. Each focused group passed after its minimal implementation.
- Business boundary: No indicator, signal, threshold, ETF pool, ranking, position size, execution schedule, minimum hold, ATR formula, or JoinQuant performance behavior changed. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Bound The PTrade State Journal To Two Valid Generations

- Date: 2026-07-21.
- Decision: Release deployment build `20260720.8` without changing the frozen `cross-v0.3.2` business configuration. Keep one state journal, but after a third complete generation is appended, compact it to the latest two checksum-, schema-, fingerprint-, and broker-snapshot-valid generations.
- Interruption contract: Compaction writes and fully decodes a same-directory temporary journal before atomically replacing the original. An interruption before replacement leaves the original journal intact; an interruption after replacement leaves two already-validated generations. A failed replacement is non-fatal because the newly appended original journal remains recoverable and the next changed-state save retries compaction.
- Recovery contract: A retained previous generation is useful only after its recorded broker code, quantity, and cost snapshot still matches the current account. A mismatch, incomplete risk state, or damaged journal continues through current fills and delivery-record recovery; unresolved holdings remain unverified and block new exposure.
- Test-first evidence: The retention test first failed with generations `[1, 2, 3]` instead of `[2, 3]`. A separate replacement-failure test proves the original three-generation journal still restores generation `3`. Existing truncated-tail recovery tests remain green.
- Business boundary: Signals, indicators, thresholds, ETF pool, ranking, position sizing, execution schedule, minimum hold, ATR formula, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Persist A Bounded Full PTrade Strategy Audit Log

- Date: 2026-07-22.
- Decision: Release deployment build `20260722.1` without changing the frozen `cross-v0.3.2` business configuration. In live mode, mirror every strategy-originated platform log call to `cross_signal_logs/cross_signal_v032_audit.log` under the PTrade research root.
- Retention contract: Preserve the full UTF-8 message, timestamp, and level. Cap the single file at `20 MB`; before an append would exceed the cap, retain the newest complete lines at an approximately `16 MB` target and atomically replace the original. A compaction failure leaves the original file intact and does not block platform logging or trading.
- Evidence boundary: The file captures only records emitted by this strategy through its `log` object. PTrade engine, scheduler, gateway, and server messages created outside strategy code remain available only through the platform's own log facilities.
- Separation contract: The audit file is operational evidence and is not a recovery source. The bounded two-generation state journal continues to hold broker-bound continuity and position-risk state; neither file replaces the other.
- Test-first evidence: Focused tests first failed for the absent dedicated directory, complete-message mirroring, bounded line-safe compaction, interruption-safe replacement, and deployment-document contract. The implementation adds no market-data access and does not alter any signal or order condition.
- Business boundary: Indicators, crosses, scores, thresholds, ETF pool, ranking, position sizing, execution schedule, minimum hold, ATR formula, orders, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; the production multi-factor strategy is untouched.

### Reconcile Missing PTrade Sell-Fill Callbacks At 09:36

- Date: 2026-07-23.
- Decision: Release deployment build `20260723.1` without changing the frozen `cross-v0.3.2` business configuration. Keep `on_trade_response` as the fast path, and add one 09:36 `get_trades()` reconciliation for sell orders submitted at 09:35 whose callbacks have not cleared the pending-sell guard.
- Root cause: A live sell completed shortly after 09:35, but no strategy trade callback was observed. The pending-sell guard therefore blocked replacement buys until the next active 10:35 recovery task. The broker position later proved the sell had completed, so the one-hour delay was an order-state observation gap rather than a strategy signal decision.
- Matching contract: Only current-day sell fills whose official order ID matches a current pending sell are routed through the existing callback state machine. The official non-empty `business_id` is preserved so a late callback and the active query cannot count the same fill twice.
- Resume contract: Once every pending sell is confirmed, the adapter immediately resumes the candidate list frozen at 09:35. It does not recalculate indicators, crosses, scores, ranking, or T-1 signals at 09:36. If any sell remains pending, the existing 10:35 halt/reject/partial-fill recovery remains authoritative.
- Platform contract: PTrade live mode now registers three tasks, still below the documented combined `run_daily`/`run_interval` limit of five. The next-minute query avoids the documented same-minute `get_trades()` first-query cache.
- Test-first evidence: Focused tests first failed for the missing 09:36 task, missing active fill reconciliation, missing `business_id` preservation, and stale deployment identity. The implementation then passed the complete PTrade test file.
- Business boundary: No indicator, signal, threshold, ETF pool, ranking, position size, minimum hold, ATR formula, or JoinQuant performance behavior changed. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Audit Raw PTrade Trade Pushes Before Filtering

- Date: 2026-07-24.
- Decision: Release deployment build `20260724.1` without changing the frozen `cross-v0.3.2` business configuration. Record every `on_trade_response` entry and one bounded raw-detail line per callback item before the live-mode guard, cancellation filter, data validation, or pending-order match.
- Diagnostic contract: Each dictionary detail includes code, direction, quantity, price, balance, raw `order_id`, entrust number, business ID, entrust status, callback type, trade status, original entrust number for cancellation, rejection reason, and business time. A missing `order_id` is printed as `<空>` instead of being hidden by an early return. Non-dictionary records retain their raw type. Logging failures are contained and cannot interrupt the original callback state machine.
- Root-cause boundary: The new evidence distinguishes no PTrade callback invocation from a callback that entered strategy code but was ignored because of runtime mode, cancellation type, malformed payload, missing quantity, unmatched pending order, or order-ID mismatch. It does not assume which cause occurred.
- Test-first evidence: The new callback-entry test first failed because `g.__is_live=False` returned before any log and therefore hid both non-empty and empty `order_id` values. It passed after the audit layer was placed ahead of that guard.
- Business boundary: The callback matching rules, fill accumulation, duplicate protection, 09:36 reconciliation, indicators, signals, thresholds, ETF pool, ranking, position sizing, execution schedule, minimum hold, and ATR formula are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Resume Deferred Buys Immediately After Complete Sell Pushes

- Date: 2026-07-25.
- Decision: Release deployment build `20260725.1` without changing the frozen `cross-v0.3.2` business configuration. A matching `on_trade_response` batch that confirms every pending sell now resumes the 09:35-frozen buy candidates immediately; 09:36 remains the deterministic fallback for missing or delayed pushes and for an immediate attempt that submits no order.
- Root cause: The callback state machine cleared a fully filled pending sell but only the separately scheduled 09:36 reconciliation contained the deferred-buy resume call. A valid complete push therefore could not release the buy workflow itself.
- Snapshot-lag contract: At the first deferral, record broker cash before sell proceeds are reflected. Sum only deduplicated confirmed sell fills. When all pending sells are terminal, use the greater of current broker cash and base cash plus confirmed proceeds, and remove only fully sold codes from the possibly stale holding snapshot. Partial fills retain their holding slot and risk state.
- Idempotency contract: Clear the deferred flag before callback-triggered order submission, process a callback batch completely before resuming, and preserve the existing order-ID match plus non-empty `business_id` deduplication. Duplicate full-fill pushes cannot submit a second replacement buy.
- Test-first evidence: New tests first failed because complete sell pushes submitted no replacement order, while the partial-fill guard already passed. After the implementation, complete-fill, partial-fill, duplicate-fill, 09:36 fallback, and the full PTrade strategy test file passed.
- Business boundary: No indicator, cross, score, threshold, ETF pool, ranking, target-value formula, minimum hold, ATR formula, T-1 signal, or JoinQuant performance behavior changed. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Explain Every Rejected PTrade Buy Candidate

- Date: 2026-07-25.
- Decision: Release deployment build `20260725.2` without changing the frozen `cross-v0.3.2` business configuration. When the existing buy filter returns no candidate, emit a source-tagged summary and one complete rejection line per scored ETF.
- Diagnostic contract: Reuse the exact frozen predicates for buy permission, buy threshold, sell threshold, fresh low-position evidence, blocked entry combinations, and current/pending holdings. The source distinguishes the 09:35 main pass, complete sell push, 09:36 fill fallback, and 10:35 halt/sell compensation.
- Test-first evidence: Focused tests first failed because the buy function accepted no diagnostic source and only emitted a generic no-candidate line. Source labels and full rejection reasons passed after the isolated logging layer was added.
- Business boundary: Candidate filtering remains in the original function and order. The new helper only explains a result after that filter has already returned an empty list. No indicator, score, threshold, ETF pool, ranking, position size, schedule, order, minimum hold, ATR formula, T-1 signal, or JoinQuant performance behavior changed. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Add PTrade Order Lifecycle Timing Diagnostics

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.1` without changing the frozen `cross-v0.3.2` business configuration. Add one normalized order-lifecycle record for submission, order callbacks, trade callbacks, 09:36 reconciliation, recovered open orders, and the after-close closure summary.
- Diagnostic contract: Each lifecycle line records source, side, code, order ID, requested quantity, cumulative fill, remaining quantity, elapsed seconds, and status. Submission timestamps are transient execution evidence. If a restart loses that timestamp, recovered open orders report unknown elapsed time instead of inventing an age.
- Reconciliation contract: The existing 09:36 task emits pending, matched-fill, and unresolved counts. The existing after-trading callback emits pending-buy, pending-sell, deferred-buy, and unknown-order-state counts. No new scheduled task is added.
- Test-first evidence: Focused tests first failed for missing submission timestamps, lifecycle records, 09:36 summary, after-close summary, documentation, and release identity. Submission, order-response, trade-response, active-query, and closing-summary tests passed after the isolated diagnostics were added.
- Business boundary: Callback matching, fill accumulation, duplicate protection, order submission, retry behavior, cash, positions, indicators, scores, thresholds, ETF pool, ranking, schedule, minimum hold, ATR formula, T-1 signals, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Reconcile PTrade Buy Fills And Isolate Log Failures

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.2` without changing the frozen `cross-v0.3.2` business configuration. Extend the existing 09:36 `get_trades()` task from pending sells to both pending buys and pending sells, and isolate the platform-log and persistent-audit-file sinks from each other.
- Buy-recovery contract: Only a current pending buy whose side and non-empty official order ID exactly match a `get_trades()` record may enter the existing trade-callback state machine. Confirmed quantity, value, buy date, entry ATR, and highest-close baseline are recovered through that same state machine. Wrong, empty, stale, or unmatched order IDs remain pending and cannot create inferred position state.
- Logging contract: A platform-log exception cannot prevent the persistent audit sink from receiving the original message. An audit-file failure cannot interrupt platform logging or trading and is reported through the underlying platform logger at most once. After the broker accepts an order, the duplicate-order guard is registered before post-submission diagnostics.
- Scheduling contract: The live task count remains three. The 09:36 pass still performs one official trade query, does not recalculate indicators or scores, does not change candidate ranking, and does not create a second trading decision.
- Test-first evidence: Eight focused test functions (ten cases) first failed for absent buy reconciliation, absent wrong-order rejection summary, unisolated platform logger exceptions, silent audit-file failures, stale documentation, stale release identity, and buy/sell guards being registered after fallible diagnostics. Implementation followed only after those failures were recorded.
- Business boundary: Signals, indicators, crosses, thresholds, ETF pool, ranking, position sizing, order quantities, execution schedule, minimum hold, ATR formula, cash rules, sell-before-buy behavior, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Reconcile Missing PTrade Recovery-Order Fills At 10:36

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.3` without changing the frozen `cross-v0.3.2` business configuration. Register one 10:36 `run_daily` task that reuses the strict 09:36 buy/sell fill reconciliation for orders still pending after the 10:35 halt, rejection, and partial-fill recovery pass.
- Trigger contract: `on_trade_response` remains the fast path. The 10:36 task calls `get_trades()` only when an in-memory pending buy or sell has a non-empty official order ID; otherwise it returns without a broker trade query.
- Matching contract: Only current pending orders whose side and exact order ID match an official current-strategy fill enter the existing callback state machine. Lifecycle and reconciliation summaries identify `10:36主动核对` separately from `09:36主动核对`.
- Scheduling contract: PTrade live mode now registers four tasks, below the documented combined `run_daily`/`run_interval` limit of five. Sleeping inside the 10:35 callback and a broader interval timer were rejected because they add blocking or duplicate-trigger complexity.
- Test-first evidence: Focused tests first failed for the missing 10:36 task, absent source parameters, absent wrapper, stale three-task documentation, and stale release identity. The implementation was added only after those expected failures were recorded.
- Business boundary: The 10:36 pass does not read daily bars, recalculate indicators, crosses, scores, or ranking, or create a new signal decision. ETF pool, thresholds, position sizing, order quantities, minimum hold, ATR formula, T-1 signals, and JoinQuant performance behavior remain unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Layer PTrade Daily Logs Without Reducing Audit Detail

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.4` without changing the frozen `cross-v0.3.2` business configuration. Add explicit daily start/end boundaries, five 09:35 stage markers, ranked short summaries, and one daily execution summary.
- Level contract: Keep decisions, orders, fills, lifecycle state, risk events, and aggregate counts at `INFO` or above. Move repeated full indicator payloads and per-ETF buy-rejection details to `DEBUG`, using stable `[指标明细]` labels. The existing bounded audit proxy continues to mirror `DEBUG`, so no diagnostic field is removed from the 20 MB audit file.
- Cost contract: The summary reuses values already produced by the 09:35 path. It adds no scheduled task and no market, position, order, fill, delivery-record, or IOPV API call.
- Test-first evidence: Focused tests first failed for absent daily/stage markers, full indicator payloads remaining at `INFO`, buy-rejection details remaining at `INFO`, and stale release documentation. Implementation followed only after those expected failures were recorded.
- Business boundary: Indicators, crosses, scores, thresholds, ETF pool, ranking, sell-before-buy ordering, position sizing, ATR stops, order quantities, task schedule, T-1 data boundary, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Make PTrade Fill Reconciliation Cumulative And Idempotent

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.5` without changing the frozen `cross-v0.3.2` business configuration. Treat each `get_trades()` result as the current day's cumulative fill history for an order instead of adding every returned row on top of the cumulative `Order.filled` quantity restored by `get_open_orders()`.
- Failure mode: After a restart or the 10:35 open-order rebuild, an order already reported as partially filled could return the same historical fill again at 09:36 or 10:36. The old incremental callback path counted both values and could falsely complete an order, clear sell-risk state, or release a deferred replacement buy while the broker still held the remaining position.
- Reconciliation contract: Group queried fills by side and exact official order ID, deduplicate business IDs inside the batch, and apply the queried quantity and value as one cumulative broker fact using the greater proven cumulative quantity. Preserve already-seen business IDs when the same open order is rebuilt. Repeating an unchanged query is idempotent; a later unique fill increases the cumulative fact and completes the order normally.
- Test-first evidence: Four focused tests first failed because recovered partial buys and sells were cleared by the same historical fill, seen business IDs were discarded during rebuild, and the query helper mislabeled its output as state recovery. All four passed after the isolated PTrade accounting repair.
- Business boundary: Real-time trade callbacks remain the fast incremental path. Signals, indicators, crosses, scores, thresholds, ETF pool, ranking, position sizing, sell-before-buy ordering, order quantities, task schedule, minimum hold, ATR formula, T-1 data boundary, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Close PTrade Open-Order Disappearance And Same-Day Rebuy Gaps

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.6` without changing the frozen `cross-v0.3.2` business configuration. At 10:35, reconcile exact pending order IDs against current fills before refreshing open orders. Preserve a locally submitted pending order that disappears from `get_open_orders()` until an exact fill or terminal callback proves its outcome.
- Failure mode: A completed order can leave the open-order list before its trade callback or `get_trades()` record reaches the strategy. Clearing the local pending ID at that instant loses the only strict matching key and can delay or incorrectly release replacement buying. Separately, excluding a just-sold holding from the stale portfolio snapshot could allow the same frozen T-1 candidate to buy that code back on the same day.
- Execution contract: A disappeared pending order retains its order ID, cumulative fill, submission timestamp, and seen business IDs and keeps the adapter fail-safe. A code confirmed sold today is excluded from new buys without consuming a portfolio slot; all other frozen candidates keep their original ranking and sizing. The existing four scheduled tasks are unchanged.
- Test-first evidence: Three focused tests first failed for missing 10:35 pre-query, discarded disappeared order IDs, and same-day rebuy. The new tests and 19 related order, deferred-buy, open-order, and halt-recovery regressions passed after the isolated PTrade adapter repair.
- Business boundary: Signals, indicators, crosses, scores, thresholds, ETF pool, ranking, position sizing, order quantities, minimum hold, ATR formula, T-1 data boundary, and JoinQuant performance behavior are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Align JoinQuant Same-Day Sold-Code Exclusion With PTrade

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.7` without changing the frozen `cross-v0.3.2` business configuration. JoinQuant now records a code in the daily sold set only after the post-order portfolio proves that its position is flat, then excludes that code from the remaining same-day buy queue.
- Execution contract: An unfilled or cancelled sell does not create the guard. A confirmed same-day sell remains excluded without occupying a portfolio slot, so the next qualified ETF in the existing frozen ranking can fill the vacancy. The guard resets on the next trade date. PTrade already applies the same sold-code exclusion to its frozen candidate queue.
- Test-first evidence: The new tests covered unfilled-sell preservation, confirmed-flat registration, ATR-sell same-cycle exclusion, ranked backup filling, and next-day reset. Before the release update, the three behavior tests passed while the two new immutable-build assertions failed on the old `.6` identity, providing the expected red phase.
- Verification: The focused release and behavior checks passed (`7 passed`), the complete cross-signal suite passed (`641 passed, 78 deselected`), both formal files passed `py_compile`, and `tools/verify_release.py` confirmed matching version, build, parameters, ETF pool, business fingerprint, and core pure functions.
- Historical evidence boundary: User-run JoinQuant comparisons for 2010-2014, 2015-2018, 2019-2021, 2022-2023, 2024-latest, and the continuous 2015-2026 window were unchanged. These results are evidence that the safety invariant did not alter those historical order paths; they were not used to select indicators, thresholds, ETF membership, or parameters.
- Business boundary: Signals, indicators, crosses, scores, thresholds, ETF pool, ranking, position sizing, order quantities, minimum hold, ATR formula, execution times, and T-1 data boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; the production multi-factor strategy is untouched.

### Backfill Failed Buy Orders Without Consuming A Slot

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.8` without changing the frozen `cross-v0.3.2` business configuration. A buy that cannot create a position no longer consumes a portfolio slot; execution continues with the next qualified ETF in the already frozen ranking.
- JoinQuant contract: After `order_target_value`, count the slot only when the synchronous post-order portfolio proves that the position exists. Otherwise log the execution failure and continue down the original candidate list.
- PTrade contract: A submission exception, missing order ID, or terminal `5`/`6`/`9` response with zero cumulative fill marks the code failed for the current trade day. The adapter excludes that code, immediately attempts the next frozen candidate, and reuses only the existing 09:36/10:36 reconciliation task when broker cash has not synchronized. Partial fills and normal fills retain the prior lifecycle.
- Boundedness contract: Failed-code guards and the pending-backfill flag reset on the next trade date. No indicator, score, rank, or position size is recalculated; no new scheduled task is registered.
- Test-first evidence: Five execution tests first failed for a consumed JoinQuant slot, missing PTrade zero-fill exclusion, absent immediate and delayed backup orders, submission-failure retry leakage, and missing next-day reset. A sixth diagnostic test then failed because the buy-rejection summary did not distinguish an otherwise valid code excluded by the failed-order guard. Implementation followed only after those expected failures were recorded.
- Verification: The focused release contract passed (`5 passed`), both formal strategy test files passed (`254 passed`), the complete cross-signal gate passed (`647 passed, 78 deselected`), all three formal Python entries passed `py_compile`, and `tools/verify_release.py` confirmed matching version, build, parameters, ETF pool, business fingerprint, and core pure functions.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and T-1 data boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; PTrade still registers four scheduled tasks; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Reconcile PTrade Zero-Fill Terminal Orders Proactively

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.9` without changing the frozen `cross-v0.3.2` business configuration. Reuse the existing 09:36 and 10:36 reconciliation tasks to query the exact IDs of unresolved orders with `get_order(order_id)`.
- Failure mode: A terminal zero-fill order can miss or delay its `on_order_response` callback and is absent from `get_trades()`. The adapter then keeps the order pending, which can block a valid replacement buy or prevent a rejected sell from becoming retryable.
- Execution contract: Only official terminal statuses `5`, `6`, and `9` are applied. Returned order ID, code, side, requested quantity, and cumulative fill are validated before state changes. A zero-fill failed buy releases the frozen backup queue; a zero-fill failed sell releases the duplicate-sell guard while retaining holding risk state. Query failures, malformed responses, partial fills, and non-terminal states remain pending and fail closed. No scheduled task is added.
- Cross-platform contract: Flow tests drive both formal strategies through ATR-stop and signal-sell paths. JoinQuant buys the next eligible ETF synchronously; PTrade waits for the exact full sell fill and then buys the same replacement. The just-sold code remains excluded for the day without consuming the vacancy.
- Test-first evidence: Active-query tests first failed because `get_order` was never called and pending zero-fill terminal orders remained blocked. The release-identity tests then failed on the prior `.8` build before the formal identifiers and deployment notes were updated.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and T-1 data boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Validate Signed PTrade Terminal Order Quantities

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.10` without changing the frozen `cross-v0.3.2` business configuration. Interpret PTrade `Order.amount` and `Order.filled` with their documented direction signs, then normalize a proven fill to a positive quantity before passing it into the existing terminal-order state machine.
- Failure mode: PTrade reports sell `amount` and `filled` as negative values. The prior active-query path rejected a valid partially filled sell, while also accepting a response whose symbol was missing or whose amount had the wrong side or quantity. Either case could leave deferred buying blocked or apply a terminal state to an unproved order object.
- Adapter contract: An active `get_order(order_id)` response must prove the exact order ID, normalized security code, signed side, and absolute requested quantity. Buy quantities must be positive and sell quantities negative. A malformed, missing, wrong-side, or wrong-size response retains the pending state and fails closed. Only after these checks is signed cumulative fill converted to an absolute quantity for the existing callback-compatible terminal handler.
- Release-gate contract: A failed full test run now reports up to five failed pytest node IDs together with the summary, so a stale release identity or behavioral regression remains directly diagnosable instead of being reduced to a count.
- Test-first evidence: Five focused PTrade tests first failed for rejecting a valid negative sell fill and accepting missing-code, wrong-side, wrong-size, or wrong-sign responses. Three release tests first failed for hidden node IDs and the prior `.9` identity before implementation began.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and T-1 data boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Use Completed PTrade Daily Bars For Closing Risk State

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.11` without changing the frozen `cross-v0.3.2` business configuration. PTrade live closing-risk maintenance now reads the completed current-session daily bar instead of asking the realtime-snapshot helper for a price at 15:30.
- Failure mode: The realtime helper correctly rejects snapshots older than five minutes. PTrade commonly reports a 15:00 snapshot during `after_trading_end` around 15:30, so the prior path could skip the daily `highest_since_buy` update and leave the ATR trailing baseline stale.
- Closing-state contract: `after_trading_end` requests current-session `close` and `volume` with `get_history(..., include=True)`. It updates the highest closing price only when the returned bar date exactly equals the context trade date and both values are positive finite numbers. Missing, stale-date, suspended, zero-volume, malformed, or failed history remains fail-closed and never falls back to a stale realtime snapshot. Backtest behavior is unchanged.
- Order-state contract: An exact active-order query returning status `8` is logged as fully filled but still waiting for trade details. It remains pending until a real trade callback or `get_trades()` record proves fill price and value; the adapter does not manufacture those facts from the order object.
- Test-first evidence: The closing-state test first failed because the stale realtime snapshot left `highest_since_buy` unchanged. The status-8 test first failed because no diagnostic distinguished a fully filled order awaiting trade details. Both passed only after the isolated adapter changes, and release-identity tests first failed on build `.10` before the formal build metadata was advanced.
- Verification: The affected release suite passed (`284 passed`), the complete cross-signal gate passed (`660 passed, 78 deselected`), all three formal Python entries passed `py_compile`, and `tools/verify_release.py` confirmed matching version, build, parameters, ETF pool, business fingerprint, and core pure functions. The initial in-sandbox runs produced only Windows temporary-directory access errors; the identical tests passed outside that restricted environment.
- Explicit exclusions: The user does not restart during the trading session, so no intraday deferred-order persistence was added. Historical holdings are whole-entry and whole-exit under one active strategy, so mixed-lot reconstruction was not changed.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and T-1 data boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `3`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Confirm PTrade Trailing High From The Final T-1 Daily Bar

- Date: 2026-07-26.
- Decision: Supersede the `.11` same-session closing assumption with deployment build `20260726.12`, without changing the frozen `cross-v0.3.2` business configuration. A live 15:30 daily value is diagnostic observation only. The next `before_trading_start` reads the exact finalized T-1 daily bar and only then may raise `highest_since_buy`.
- Root cause: PTrade documents `include=True` as including the current period, but that does not prove that a current-session daily period returned during `after_trading_end` is already the final official close. Using it as an ATR trailing baseline could therefore persist an unfinalized value. The multi-factor adapter also used a same-session snapshot/history path in its normal close flow, so it was useful historical context but not a correctness proof and was not copied.
- Confirmation contract: Restore holding risk state first; determine T-1 through the proven trading calendar; request the exact T-1 `close` and `volume`; require exact-date and finite values. A positive-volume session may update the high, a zero-volume suspended session keeps the prior confirmed high, and missing/stale/malformed data marks the holding unverified so automatic exits and new buys fail closed.
- Migration contract: Advance the PTrade state schema from `3` to `4` and use a new bounded v4 journal path. This intentionally prevents a potentially contaminated schema-3 high from being reused. The first start after deployment rebuilds held-position facts from broker delivery records and historical daily bars; later restarts use the normal verified schema-4 state.
- Test-first evidence: Eight focused tests first failed on the `.11` identity, schema-3 acceptance, same-session high mutation, absent T-1 confirmation, and missing fail-closed behavior. The integration test also proved confirmation runs after state recovery and before summary/persistence. The implementation followed only after those failures.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. JoinQuant changed only its deployment build identity. Business fingerprint remains `1506a0e834fe`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Reconcile The 15:30 Provisional High With The Final T-1 Close

- Date: 2026-07-26.
- Decision: Supersede the `.12` observation-only close flow with deployment build `20260726.13`, without changing the frozen `cross-v0.3.2` business configuration. PTrade still queries the completed current-session daily value at 15:30 and immediately applies it as a provisional trailing-high update; the next morning then reconciles that provisional state against the exact finalized T-1 daily bar.
- Root cause: Treating the 15:30 value as authoritative can preserve a value that PTrade later revises, while treating it only as a log skips the end-of-day risk-state maintenance the strategy expects. The required behavior is two-stage confirmation, not either extreme.
- Reconciliation contract: The 15:30 path requires exact current-session date, finite positive close, and positive volume. It persists the session date, the prior confirmed high, and the observed close together with the provisional high. On the next `before_trading_start`, an exact finalized T-1 bar replaces that provisional high with `max(prior confirmed high, final close)`, so an overstated provisional value can be corrected downward without losing an earlier confirmed peak. A zero-volume final bar restores the prior confirmed high. Missing, mismatched, or malformed final evidence retains the pending state, marks the holding unverified, and fails closed.
- Migration contract: Advance the PTrade state schema from `4` to `5` and move to a bounded v5 journal. Schema 4 is rejected because it cannot prove whether a stored high is final or provisional and does not preserve the pre-observation baseline needed for a safe downward correction. The first start rebuilds held-position facts from broker evidence; later cross-day restarts preserve the pending confirmation metadata.
- Test-first evidence: Focused release and behavior tests failed first for the old build identity, absent pending-state persistence and validation, schema-4 acceptance, missing provisional mutation, inability to lower an overstated value, failure to restore the prior high on zero volume, and unsafe pending-date handling. Implementation followed only after those failures.
- Verification: The focused PTrade suite passed (`218 passed`), the complete cross-signal gate passed (`672 passed, 78 deselected`), all three formal Python entries passed `py_compile`, and `tools/verify_release.py` confirmed matching version, build, parameters, nine-ETF pool, business fingerprint, core pure functions, and state schema 5.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. JoinQuant changed only its deployment build identity. Business fingerprint remains `1506a0e834fe`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Catch Up Every Unconfirmed PTrade Closing Session

- Date: 2026-07-26.
- Decision: Release deployment build `20260726.14` without changing the frozen `cross-v0.3.2` business configuration. Preserve the earliest unconfirmed closing session even when the 15:30 observation is unavailable, and catch up every exact finalized daily bar through current T-1 before releasing the risk-state guard.
- Root cause: When a finalized T-1 bar could not be proved, the old flow marked the holding unverified but did not always retain a durable cursor for that missing session. On the following trading day it could query only the newer T-1 date, leaving the older gap unproved and the holding blocked until a full broker reconstruction or restart.
- Confirmation contract: Use the official PTrade `get_trade_days(start_date, end_date)` calendar to prove the complete session interval. Read every exact close and volume from the earliest pending session through current T-1, treat zero-volume sessions as suspended, and commit the new highest close only after the whole interval succeeds. Any missing date preserves the original confirmed high and earliest cursor atomically. A fully verified holding also repairs a stale `unverified` provenance label to the actual restored-state source.
- Migration contract: Advance the PTrade state schema from `5` to `6` and move to the bounded v6 journal. Schema 6 permits a pending confirmation with no provisional observation while still requiring a valid session date and prior confirmed high. The first start after deployment rebuilds held-position risk facts from broker evidence; subsequent starts retain the continuous confirmation cursor.
- Test-first evidence: Five focused tests first failed for stale-session rejection, non-atomic interval handling, absent missing-observation cursor support, and stale recovery provenance. After implementation all five passed, and the complete PTrade strategy test file passed (`223 passed`). The first sandboxed full run failed only because Windows denied pytest temporary-directory cleanup; the identical test command passed outside the restricted sandbox.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, order quantities, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. JoinQuant changed only its deployment build identity. Business fingerprint remains `1506a0e834fe`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Use Sell-Five Buy Limits Without Promoting Backup ETFs

- Date: 2026-07-27.
- Decision: Release deployment build `20260727.1` without changing the frozen `cross-v0.3.2` business configuration. PTrade live buys use the sell-five quote from the same fresh snapshot already accepted for current-price execution checks.
- Quote contract: Sell-five must prove a finite positive price, positive displayed quantity, and no breach of the instrument upper limit. Missing or malformed depth fails closed and never falls back to the latest price or upper-limit price. Share sizing and cash reservation use the submitted sell-five limit. PTrade backtests retain the current-price smoke-test path because they cannot prove live five-level depth.
- Candidate contract: The first actual buy evaluation freezes the intended candidates for that trade day. A submission failure, cancellation, rejection, or terminal zero fill does not promote a lower-ranked ETF. Other candidates already selected for distinct open slots continue normally. A code proved paused at 09:35 may be added in original T-1 score order after the 10:35 resume check proves it tradable.
- Test-first evidence: Twelve focused tests first failed for the absent sell-five parser, current-price sizing, and old backup behavior. A separate cross-time test then failed because a lower-ranked ETF was still promoted during the 10:35 compensation pass. Implementation followed only after both expected red phases.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing formula, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `6`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Restrict Ranked Buy Backup To Confirmed Suspensions

- Date: 2026-07-27.
- Decision: Release deployment build `20260727.2` without changing the frozen `cross-v0.3.2` business configuration. Only a candidate proved suspended before order submission may release its intended slot to the next candidate that already satisfies every existing buy condition.
- Execution contract: Invalid or missing price/depth, sell-five failure, submission exception, missing order ID, cancellation, rejection, zero fill, and partial fill all consume the candidate's intended slot and never promote a lower-ranked ETF. An unproved trading status remains fail-closed for order submission but is not classified as a confirmed suspension and therefore does not release the slot. Limit-up is not introduced as a separate backup reason. If no later qualified candidate exists after a suspension, the slot remains cash.
- Recovery contract: PTrade records the suspended code for the existing 10:35 recovery pass. A resumed ETF may buy only if a slot is still open; it cannot displace a replacement already filled or pending from 09:35. JoinQuant uses the same pause-only slot-consumption rule in its synchronous order loop.
- Test-first evidence: The new JoinQuant contract first failed because an ordinary unfilled order still promoted the next ETF. The new PTrade contract first failed because a suspended top candidate remained frozen and blocked a qualified successor. A threshold regression test also proves a sub-threshold successor is never promoted.
- Verification boundary: No market data or validation-period result was read. Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `6`; the production multi-factor strategy is untouched.

### Harden Live Trade-Status Proof Across Platforms

- Date: 2026-07-27.
- Decision: Release deployment build `20260727.3` without changing the frozen `cross-v0.3.2` business configuration. Both adapters now distinguish confirmed suspension, confirmed continuous trading, and unknown execution status.
- PTrade contract: A cached snapshot may support status fallback only when its timestamp proves the current session and an age no greater than 300 seconds. `HALT`, `SUSP`, and `STOPT` prove suspension; only `TRADE` proves continuous matching. Call auction, break, post-trading, end-trading, delisted, missing, malformed, stale, and future-dated states remain unknown. Live sell-five orders require the same fresh snapshot to prove `TRADE`.
- JoinQuant contract: A failed or malformed current-data pause lookup is unknown and fails closed for order execution. It does not count as a confirmed suspension, so it consumes the intended ranked slot and cannot promote a lower candidate. Only an explicit paused flag releases the slot.
- Release gate: Five named execution contracts are now mandatory: same-day sold exclusion, confirmed-pause-only promotion, non-pause failure slot consumption, pending-sell buy blocking, and full-sell-fill buy resumption.
- Test-first evidence: Fourteen focused PTrade assertions, three JoinQuant assertions, the release-contract check, and two release-identity assertions all failed before their corresponding implementation changes and passed afterward.
- Business boundary: Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `6`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Require Continuous Matching Before PTrade Live Sells

- Date: 2026-07-27.
- Decision: Release deployment build `20260727.4` without changing the frozen `cross-v0.3.2` business configuration. PTrade live sell submission now requires the same fresh execution snapshot to prove `trade_status=TRADE`.

## 2026-07-27：PTrade DEBUG 明细日志预渲染

- Evidence: PTrade 模拟回测把 `_log_debug_detail()` 的 `%d`、`%s` 等占位符原样输出，说明平台日志器不保证兼容 Python logging 的延迟格式化参数。
- Decision: `_log_debug_detail()` 在调用平台 `debug/info` 前统一复用 `_render_log_message()` 完成渲染，保证平台界面与审计文件中的完整指标明细一致可读。
- Scope: 仅修复日志输出兼容性；策略版本、业务参数、ETF 池、信号、排序、风控和业务配置指纹均不变。部署构建更新为 `20260727.5`。
- Root cause: `get_stock_status(code, "HALT") == False` proves only that the ETF is not suspended. It does not prove continuous matching, so call auction, midday break, post-trading, or end-trading states could previously reach `order_target`.
- Execution contract: After obtaining a fresh current price, the sell path checks the cached snapshot produced by that quote request. Any state other than fresh `TRADE` fails closed before order submission, leaves `sold_today` and pending-order guards untouched, and preserves the original ATR or signal-sell reason for the existing bounded 10:35 re-evaluation. If the second attempt is still not orderable, the retry reason remains and logs do not falsely claim that the risk condition disappeared.
- Test-first evidence: Five non-continuous or missing-status cases and one unavailable-price case first proved that the old path either submitted an order or lost the retry reason. A separate 10:35 test first proved the old log incorrectly reported that the risk had cleared. The implementation followed those red tests; the complete PTrade test file then passed (`259 passed`).
- Business boundary: JoinQuant changed only its deployment build identity. Signals, indicators, crosses, parameters, thresholds, ETF pool, ranking, target sizing, minimum hold, ATR formula, execution times, and the T-1 signal boundary are unchanged. Business fingerprint remains `1506a0e834fe`; state schema remains `6`; no market or validation-period data was read, and the production multi-factor strategy is untouched.

### Reject Fixed One-Entry-ATR Break-Even Floor

- Date: 2026-08-12.
- Decision: Preserve official `cross-v0.3.2` unchanged. Retain the new profit-giveback module as observation-only research infrastructure and archive the isolated candidate as rejected evidence.
- Reason: Profit giveback exists in training, but the fixed candidate reduced total and annualized return, worsened maximum drawdown, Sharpe, Sortino, win rate, profit/loss ratio, and both 2020 and 2021 annual returns.
- Causal boundary: Activation uses only the stored highest close from completed sessions and entry ATR frozen at the buy signal. The T-day current price is used only for execution, so the candidate contains no future function.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read. No validation-period result was inspected. Nearby ATR activations, floor offsets, staged stops, and ETF/year exceptions are closed as post-hoc searches.
- Business boundary: No JoinQuant, PTrade, or local formal strategy file changed. The production multi-factor strategy is untouched.

### Prepare Frozen ATR-Stress Rule On cross-v0.3.3 Mainline

- Date: 2026-08-16.
- Decision: Raise the JoinQuant mainline to `cross-v0.3.3` / build `20260816.1` with the frozen portfolio ATR-stress rule (15 trading days, 3 ATR stops, 0.50 buy scale) added on top of the `cross-v0.3.2` rules. The PTrade adapter stays frozen at `cross-v0.3.2` until the JoinQuant training confirmation and all four reserved validation windows pass; the release verifier now correctly rejects this transitional parity state, and its tests encode that expectation.
- Local pre-check evidence: A read-only 2019-2021 local replay (identical data and execution model, stress keys injected at planner level only) reproduced the recorded baseline exactly (44122.30, +120.61%, 7.47% max drawdown, 92 buys/89 sells) and returned +125.00% total, 6.03% max drawdown, Sharpe 2.262, Sortino 3.581 for the stress version. The trade path was unchanged (92/89); six half-size buys were filled (2020-03-03/03-05/03-06/03-23, 2020-09-15, 2020-09-22), improving the 2020 annual return from +49.74% to +52.68%.
- Reason: The frozen candidate had already improved training and the 2022-2023 validation window on `cross-v0.3.1`; the local pre-check confirms the rule also helps on the current `cross-v0.3.2` path without changing any entry or exit decision. JoinQuant remains the performance authority, so the adoption is staged: JoinQuant confirmation first, then reserved validation windows, then PTrade sync.
- Research boundary: No validation-period data was read locally. The local replay consumed only the approved 2018 warm-up and 2019-2021 training roots. The three stress parameters are the frozen candidate values; no parameter search was performed.
- Business boundary: JoinQuant file changed (version, build, three frozen parameters, stress sizing, stop-history recording, `stress=` buy log field). PTrade file, ETF pool, signal scoring, ranking, sell rules, ATR stop math, and execution times are unchanged. The production multi-factor strategy is untouched.

### Reject Fixed Profit-Tiered ATR Tightening

- Date: 2026-08-16.
- Decision: Reject the user-authorized `cross-v0.3.3-profit-tier-candidate` before JoinQuant and validation, and close the `profit_tiered_atr_user_authorized` family as exhausted. Keep official `cross-v0.3.3` unchanged.
- Reason: The Step 0 binding observation found 36 binding stop-check events (4/24/8 across 2019/2020/2021) but zero same-day extra triggers, and the Step 1 local A/B confirmed an exact no-op: 0 changed filled orders and every headline metric identical to the baseline (+125.00% return, 6.03% max drawdown, 2.262 Sharpe, 3.581 Sortino, 92/89 orders, 25 ATR stops). The pre-registered "at least 3 filled orders change" gate failed.
- Causal boundary: The cross-signal stop uses a frozen entry ATR and a 5% floor; the median entry ATR is about 1.4% of price, so the floor dominates. On the 36 binding days the tightened stop only moved from 5.02%-7.74% to 5.00%-6.19% below the peak, and no 09:35 price ever fell into that gap. The multi-factor V2.6 profit-tier mechanism therefore does not transfer to this framework.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read; no validation-period result was inspected. No tier threshold, multiplier factor, peak-profit measurement, profit floor, or per-ETF override search is allowed from this result. The failed-experiment ledger count and the research budget were updated.
- Business boundary: No formal JoinQuant, PTrade, or local strategy file changed. The isolated candidate file is archived under `archive/candidates/`. The production multi-factor strategy is untouched.

### Reject Fixed Gold-Specific Stop Tightening

- Date: 2026-08-16.
- Decision: Reject the user-authorized `cross-v0.3.3-gold-stop-candidate` before JoinQuant and validation, and close the `gold_specific_stop_user_authorized` family as exhausted. Keep official `cross-v0.3.3` unchanged.
- Reason: Step 0 passed (223 binding gold stop-check days, 6 same-day extra triggers), but the Step 1 local A/B failed the gates: total return +125.00%→+120.96%, max drawdown 6.03%→6.08%, Sharpe 2.262→2.210, Sortino 3.581→3.492, annual 2019 -1.50pp and 2021 -1.16pp. Gold ATR stops rose from 2 to 5 and the path diverged on 162 filled-order positions.
- Causal boundary: Per-trade attribution shows the first extra stop (2019-07-01 at +4.7% profit) clipped a winner that the baseline exited at +9.0% on 2019-08-02; gold's winning reversal trades tolerate 3-4% pullbacks below the peak while the bounce develops, which the 5% band absorbs and the 3% floor does not. The multi-factor V2.8 gold-stop result does not transfer to this framework's reversal-entry path.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read; no validation-period result was inspected. No nearby gold floor/multiplier values and no per-ETF stop extension are allowed from this result. The failed-experiment ledger count and the research budget were updated.
- Business boundary: No formal JoinQuant, PTrade, or local strategy file changed. The isolated candidate file is archived under `archive/candidates/`. The production multi-factor strategy is untouched.

### Reject Fixed Profit-Giveback Direct Exit

- Date: 2026-08-16.
- Decision: Reject the user-authorized profit-giveback direct exit at the Step 0 observation stage, and close the `profit_giveback_exit_user_authorized` family as exhausted. Keep official `cross-v0.3.3` unchanged.
- Reason: The read-only trade-level counterfactual (peak profit ≥5%, giveback 3pp → sell at the 09:35 stop check) fired 79 times across 21 affected closed trades, with a negative total per-share delta (-0.352) and negative 2019/2020 annual deltas. It clipped two major winners (2019-02-11 159928: 2.245 vs 2.666; 2020-04-17 513050: 1.523 vs 1.927) while salvaging only small amounts on other trades.
- Causal boundary: Large trend winners routinely give back more than 3pp of profit mid-hold before resuming; the same giveback that the user observed in the live 159985 case is also the cost of the +18.8%/+34% winners. A profit-anchored giveback exit kills the payoff source of this framework, consistent with the earlier break-even floor and gold-stop failures.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read; no validation-period result was inspected. No activation or giveback threshold search is allowed from this result. The failed-experiment ledger count and the research budget were updated.
- Business boundary: No formal JoinQuant, PTrade, or local strategy file changed. The production multi-factor strategy is untouched.

### Reject Fixed Intraday-High Trailing Anchor

- Date: 2026-08-16.
- Decision: Reject the user-authorized `cross-v0.3.3-high-anchor-candidate` after the Step 1 local A/B, and close the `intraday_high_anchor_user_authorized` family as exhausted. Keep official `cross-v0.3.3` unchanged.
- Reason: Step 0 passed (1604 binding stop-check days, 38 same-day extra triggers across all nine ETFs), but the local A/B failed the gates: total return +125.00%→+119.40%, max drawdown 6.03%→6.06%, 2019 annual +35.84%→+30.55% (2020/2021 slightly improved). The dominant clip was 2019-02-11 159928: the high anchor stopped it on 2019-02-26 at 2.232 versus the official close anchor's exit at 2.666 on 2019-04-12 (-0.434 per share on an +18.8% winner); seven small saves (+0.001 to +0.090 per share) could not offset it (total per-share delta -0.228).
- Causal boundary: The peak-day upper wick raises the high anchor into the winner's normal pullback band. The 5% close-anchored band exists precisely to absorb intraday spikes; the swap turns the noise the close anchor was designed to filter back into stop triggers. This validates the original close-anchor design rule with data.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read; no validation-period result was inspected. No anchor blends, multiplier re-calibrations, or threshold changes are allowed from this result. The failed-experiment ledger count and the research budget were updated.
- Business boundary: No formal JoinQuant, PTrade, or local strategy file changed. The isolated candidate file is archived under `archive/candidates/`. The production multi-factor strategy is untouched.

### Reject Fixed Profit-Gated Direct-Sell Matrix

- Date: 2026-08-16.
- Decision: Reject the user-authorized 4×3 profit-gated direct-sell matrix at the Step 0 observation stage, and close the `profit_gated_direct_sell_user_authorized` family as exhausted. Keep official `cross-v0.3.3` unchanged.
- Reason: All 12 variants failed the gates. The 38/40 score thresholds never fired (sell scores that high arrive only after profit leaves the 2-6% band), and the 32/35 thresholds produced negative total per-share deltas, dominated by the 513050 +34% winner's mid-hold pullback that satisfies the trigger and would be exited early (1.523 versus the official 1.927).
- Causal boundary: The profit band cannot distinguish a small winner that will keep winning from one that will fail; the framework's large winners pass through the 2-6% profit zone repeatedly with elevated sell scores on pullbacks. This is the same structural cause as the profit-giveback exit, break-even floor, and gold-stop failures.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read; no validation-period result was inspected. No nearby thresholds, bands, or mechanism variants are allowed from this result. The failed-experiment ledger count and the research budget were updated.
- Business boundary: No formal JoinQuant, PTrade, or local strategy file changed. The production multi-factor strategy is untouched.

### Stabilize PTrade State Identity And Broker-First Entry Recovery

- Date: 2026-08-17.
- Decision: Release build `20260817.1` with a version-independent PTrade state journal filename and a broker-first delivery recovery rule. The stable file is `cross_signal_live_state_<identity>.journal`; the current compatible schema-7 versioned journal is validated and atomically copied on first use, while the old file remains untouched.
- Recovery contract: Broker positions are authoritative for current quantity and cost. Delivery history only proves the current holding episode: locate the last actual sell and use the first actual buy afterward; if no sell exists, use the first valid buy. Historical delivery quantities are not replayed or required to equal the current broker amount. A last sell without a later buy remains unverified and fails closed.
- Test-first evidence: The new contracts first failed because the adapter still embedded version/schema in the filename, had no migration helper, and rejected broker holdings whose historical delivery quantity differed. Nine focused state-path and delivery-recovery tests passed after implementation.
- Business boundary: No signal, indicator, cross, parameter, score, threshold, ETF pool, ranking, position-sizing rule, sell rule, ATR formula, execution time, or T-1 boundary changed. The JoinQuant build marker changed only to preserve release identity parity. No market or validation-period data was read, and the production multi-factor strategy is untouched.

### Rotate The PTrade Audit Log Only At Its Size Boundary

- Date: 2026-08-18.
- Decision: Release build `20260818.1` with one version-independent active audit file, `cross_signal_audit.log`. Existing `cross_signal_v032_audit.log` and `cross_signal_v033_audit.log` files remain untouched. Before an append would exceed `20 MiB`, atomically rename the complete active file to `cross_signal_audit_YYYYMMDD_HHMMSS.log` and establish a fresh active file containing the new record. Same-second collisions add milliseconds and, if necessary, a non-overwriting sequence.
- Failure contract: Prepare and verify the next active file before renaming the old one. A failed rotation returns a mirror-write failure, attempts to restore the old active path, never overwrites an existing timestamped archive, and does not interrupt platform logging or trading.
- Test-first evidence: The focused tests first failed because the adapter still used `cross_signal_v033_audit.log`, compacted old lines in place, and appended to the legacy v033 file. The implementation then passed the active-name, timestamp rotation, collision, rollback, UTF-8, and legacy-file preservation cases.
- Business boundary: No signal, indicator, cross, parameter, score, threshold, ETF pool, ranking, position sizing, sell rule, ATR formula, execution time, T-1 boundary, or multi-factor strategy changed. No market, training, validation, or live performance data was used.

### Correct PTrade Pre-Sell Cash Capture And IOPV Probe Time

- Date: 2026-08-20.
- Decision: Release build `20260820.1`. Before `order_target()` can synchronously update the broker cash snapshot, each submitted sell records the actual pre-submit cash. Deferred replacement buys use the earliest recorded pre-submit cash plus confirmed sell proceeds, preventing the same proceeds from being counted twice. The isolated no-order IOPV probe uses the PTrade server wall clock because live `context.current_dt` remained at 09:10 during the 09:34-09:36 callbacks.
- Runtime evidence: On 2026-08-20 the 513100 sell callback reported current cash `17497.12`, confirmed proceeds `888.80`, and an impossible synthetic available cash `18385.92` against total assets near `17497`. The IOPV probe simultaneously emitted negative quote ages because its callback context stayed at 09:10 while snapshot timestamps were current. No replacement buy qualified that day, so the cash defect did not submit an oversized order.
- Test-first evidence: The cash regression first reproduced a synchronous broker update and observed `17497.12` frozen instead of the true pre-submit `16608.32`; the probe regression first observed 09:10 instead of the fixed 09:35 wall clock. Both tests passed only after the minimal corrections, followed by the related PTrade and IOPV suites.
- Business boundary: The strategy remains `cross-v0.3.3` with fingerprint `77e44d93d255`. No signal, indicator, parameter, threshold, ETF pool, ranking, position sizing, sell rule, ATR formula, execution schedule, or T-1 boundary changed. No training or reserved-period market data was read, and the production multi-factor strategy is untouched.

### Reject Full-Capacity Opportunity-Cost Replacement

- Date: 2026-08-22.
- Decision: Reject the frozen opportunity-replacement candidate before JoinQuant and close the user-authorized family as exhausted. Keep formal `cross-v0.3.3` unchanged.
- Reason: Nineteen completed replacements reduced local training return from +125.00% to +89.87%, reduced win rate from 56.18% to 55.05%, worsened drawdown from 6.03% to 6.18%, and lowered Sharpe, Sortino, profit/loss ratio, and every annual return.
- Causal boundary: The rule already required all three holdings to finish five sessions, preserved the official score-60 buy filter, and replaced only sell-score-at-least-30 holdings blocked by price/ADX protection. Its failure therefore rejects the opportunity-cost inference itself: a fresh reversal entry does not prove that a protected existing trend is the weaker asset.
- Research boundary: Only approved 2018 warm-up and 2019-2021 training data were read. No validation, recent-market, live-outcome, or full-period result was inspected, and no threshold or ETF subgroup was searched after the failure.
- Business boundary: No formal JoinQuant, formal PTrade, or production multi-factor file changed. The research-only selector, local planner, tests, and failed-result report remain as reproducible evidence; no upload candidate was generated.

### Add Non-Binding PTrade IOPV Buy And Sell Shadows

- Date: 2026-08-22.
- Decision: Release build `20260822.1` with prospective PTrade-only IOPV shadow evidence. A fixed 5% label observes whether a 09:35 qualified QDII would have been deferred and rechecked at 10:35, and whether a sell-score-at-least-30 QDII blocked by price confirmation or ADX would have been accelerated. The real strategy ignores every shadow result.
- Execution-price contract: Buy observations use the same sell-five limit already selected for the real order. The 10:35 recheck uses the then-current sell-five quote. Blocked-sell observations use bid-one as the conservative immediate execution reference. Missing quote depth or positive IOPV is recorded as unavailable without a latest-price substitution.
- Runtime contract: The existing 10:35 callback performs the recheck before halt/retry logic can return early. No new scheduled task is registered. The shadow queue is double-underscore same-day runtime state, is not persisted, is cleared after one recheck, and cannot submit, cancel, resize, defer, or accelerate a real order.
- Test-first evidence: Seven behavior tests first failed on latest-price precedence, absent 09:35 state, absent 10:35 recheck/wrapper hook, and absent blocked-sell observations. Three additional failure-open tests then failed because an observation exception could still interrupt the buy path, blocked-sell evaluation, or 10:35 recovery. One timestamp test finally reproduced the known 09:10 stale callback context in the formal IOPV log. All eleven passed after the minimal isolated implementation. The directly affected PTrade, JoinQuant identity, forward-log archive, and release-verifier suites passed together (`377 passed`).
- Business boundary: The strategy remains `cross-v0.3.3` with fingerprint `77e44d93d255` and state schema `7`. JoinQuant changed only its deployment build identity. No signal, indicator, formal threshold, ETF pool, ranking, position sizing, sell rule, ATR formula, schedule, or T-1 boundary changed. No market, training, validation, or live performance data was read, and the production multi-factor strategy is untouched.

### Activate An 8% PTrade IOPV Sell Override

- Date: 2026-08-22.
- Decision: Release build `20260822.2` with one explicit PTrade-live execution overlay for `513050.SS`, `513100.SS`, `513500.SS`, and `513880.SS`. After the existing five-trading-day minimum hold and T-1 sell score of at least 30 have passed, a bid-one/IOPV premium of at least 8% bypasses only price confirmation and ADX protection and submits the existing full-position sell. The user selected 8% directly; it was not chosen from a validation-period or nearby-threshold search.
- Safety contract: The override is live-only and requires a tradable snapshot, positive bid-one with positive depth, positive finite IOPV, non-negative quote age no greater than 10 seconds, and premium computed as `bid1 / IOPV - 1`. Missing, stale, halted, malformed, or exception-producing inputs fail open to the original blocked-sell outcome. Original eligible sells and ATR stops remain independent of IOPV. The buy-side 5% shadow remains observation-only.
- Test-first evidence: The first behavior test failed because both price-confirmation and ADX blockers still prevented an 8% premium sell. Boundary and safety tests then failed below 8%, in non-live mode, with a halted snapshot, and with an 11-second snapshot until the strict gates were implemented. An exception-injection test failed until override errors were isolated from the original signal-sell evaluation.
- Evidence boundary: No market, training, validation, or live outcome data was read for this release. JoinQuant lacks the required point-in-time IOPV, while PTrade daily backtests do not reproduce live callback timing, so this release has code-path verification but no claim of improved return or accuracy.
- Business boundary: Indicators, T-1 scoring, sell score 30, minimum hold 5 trading days, ATR stops, buy logic, pool, position sizing, schedule, and state schema remain unchanged. The PTrade live sell path now intentionally differs from JoinQuant only at the documented IOPV execution overlay; the JoinQuant build marker changed solely to preserve release identity parity. The production multi-factor strategy is untouched.

### Prepare Independent KRBA RSI-Low-Turn Scheme-A JoinQuant Candidate

- Date: 2026-08-26.
- User authorization: After reviewing the closed `krba-v0.1-candidate` and the separate prospective RSI observer, the user explicitly confirmed scheme A as a new independent JoinQuant historical candidate. This authorization does not reopen neighboring KRBA thresholds or change the order-free prospective observer.
- Frozen hypothesis: Preserve the exact original KDJ entry channel and all BOLL/ATR exits, while adding the exact RSI6 low-turn event (`r2 > r1`, `r0 > r1`, `r1 <= 30`, and `close[T-1] > close[T-2]`) as an OR entry channel without a KDJ or BOLL-entry requirement. KDJ-qualified candidates rank first; RSI-only candidates use fixed pool order without a new score.
- Causal execution boundary: Every indicator and state transition uses completed T-1-or-earlier daily bars. The 09:35 callback performs ATR, ordinary BOLL exits, and new entries; the 14:50 callback reads no indicators and only rechecks the frozen-entry-ATR stop against the then-available price. A missing exact T-1 daily bar fails closed instead of reusing an older cross.
- Files: Add `smart_trade_joinquant_kdj_rsi_boll_atr_scheme_a_candidate.py` and `tests/test_cross_signal_krba_rsi_turn_joinquant_candidate.py`; update only research-governance documentation. Formal JoinQuant/PTrade `cross-v0.3.3` files remain unchanged.
- Test-first evidence: Each behavior first failed before implementation, including both entry channels, exact T-1 loading and stale-bar rejection, deterministic ranking, exit priority, 14:50 ATR-only isolation, upload compatibility, missing-snapshot ATR protection, and rejected/pending/partial/full sell lifecycle. An independent review found and prompted regression fixes for Python-runtime compatibility, governance status validation, missing-snapshot ATR suppression, sell-order reconciliation, and `jqdata` shadowing of builtin `any`.
- Evidence status: Local behavior, governance, syntax, and formal-release isolation checks are required before handoff. No market, training, validation, PTrade, simulation, or live performance result was used to implement the candidate; an actual JoinQuant upload/import and 2019-2021 run remain external steps.
- Interpretation boundary: A 2019-2021 JoinQuant run is development-period historical diagnosis because that window already influenced the hypothesis. It is not independent out-of-sample evidence and cannot authorize validation, PTrade, simulation, or real funds. No neighboring variant is authorized after the run.
