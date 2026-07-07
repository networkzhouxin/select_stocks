# Failed Experiments

Record failed ideas so they are not rediscovered and retuned later.

## Template

```text
Date:
Version:
Experiment:
Hypothesis:
Training result:
Validation result:
Why it failed:
Can it be revisited? yes/no
Conditions for revisiting:
```

Date: 2026-07-03
Version: cross-v0.2.0
Experiment: Add buy-side widening confirmation points when fast lines are already above slow lines and positive differences are widening.
Hypothesis: Strict-cross-only entries may miss ETF trends after the crossing day; half-weight confirmation points could improve capture without lowering thresholds.
Training result: 2019-2021 return fell from cross-v0.1.6 +32.39% to +19.99%; max drawdown rose from 7.58% to 8.74%; buys rose from 145 to 150 and sells from 145 to 150.
Validation result: Not run. Per protocol, validation periods were not inspected for this failed training-period experiment.
Why it failed: The confirmation points increased signal density but bought earlier and occupied slots before stronger strict-cross setups. It treated continued positive differences as actionable entries even when the strategy goal is low-position turning points.
Can it be revisited? yes
Conditions for revisiting: Only as a gated confirmation after low-position and anti-chase filters exist; do not use it as a standalone buy-score booster.

Date: 2026-07-03
Version: cross-v0.2.4
Experiment: Convert `risk-tighten` warnings into a tighter ATR stop using 1.5x ATR and a 3% stop floor.
Hypothesis: Down-cross risk warnings that do not yet have sell structure confirmation may still justify tighter protection, improving drawdown without reintroducing immediate noise exits.
Training result: 2019-2021 return fell from cross-v0.2.3 +44.15% to +39.44%; annualized return fell from about 12.98% to 11.72%; max drawdown improved only slightly from 6.43% to 6.24%; sells increased from 117 to 146, including 42 ATR stops.
Validation result: Not run. Per protocol, validation periods were not inspected for this failed training-period experiment.
Why it failed: One-size-fits-all risk tightening clipped positions too early. The small drawdown improvement did not compensate for lost upside and higher exit count, so it partially reintroduced the noise-exit problem that cross-v0.2.3 fixed.
Can it be revisited? yes
Conditions for revisiting: Only as a conditional combination factor, for example enabled in choppy or weak-trend regimes and disabled in strong trends after ADX/DMI or equivalent regime detection is tested.

Date: 2026-07-04
Version: cross-v0.2.7
Experiment: Apply ADX/DMI regime to new-buy eligibility while keeping buy scores unchanged.
Hypothesis: Strong ADX uptrends might justify non-extended trend-continuation entries, while strong ADX downtrends might reject weak MA20-only repair entries.
Training result: 2019-2021 return was essentially unchanged versus cross-v0.2.6 (+50.074% versus +50.069%); max drawdown was slightly worse (6.891% versus 6.886%); buys and sells each increased by 1.
Validation result: Not run. Per protocol, validation periods were not inspected for this no-improvement training-period experiment.
Why it failed: The buy-side ADX rule changed only a tiny number of early trades and did not materially improve return, drawdown, or trade quality. It added complexity without a meaningful behavioral benefit.
Can it be revisited? yes
Conditions for revisiting: Only with a different buy-side formulation or after local minute-level backtesting is available; do not keep this rule in the mainline unless it shows clear benefit across reserved periods.

Date: 2026-07-04
Version: cross-v0.2.8
Experiment: Allow low-position weak-buy candidates with `buy_score >= 55` when BOLL location and reversal quality are strong.
Hypothesis: The cross-v0.2.6 log had many no-buy days and high cash near late 2021; narrowly allowing low-position, high-reversal candidates just below the normal 60-point threshold might improve capital utilization without broad threshold cutting.
Training result: 2019-2021 final assets fell from cross-v0.2.6 30013.80 (+50.07%) to 28710.90 (+43.55%). Buys rose from 132 to 142, sells rose from 131 to 141, sell_score exits rose from 112 to 122, and no-buy days fell from 249 to 220.
Validation result: Not run. Per protocol, validation periods were not inspected for this failed training-period experiment.
Why it failed: The added 55-59 point entries reduced cash but mostly added short-term false reversals. A typical early example bought 159928.XSHE at buy=57 on 2019-01-02 and sold it the next day on sell_score=34, showing that lower-threshold low-position entries increased churn rather than trade quality.
Can it be revisited? yes
Conditions for revisiting: Only as a delayed-confirmation or volume-confirmed entry rule. Do not lower the main buy threshold or keep same-day weak-buy supplementation as a mainline rule.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.90` and `min_signal_hold_days=5`
Experiment: Remove `510880` dividend ETF from the cross-signal ETF pool.
Hypothesis: `510880` had the weakest realized trade contribution and its low-volatility dividend style may be unsuitable for a cross-signal reversal/trend framework.
Training result: Return fell from +98.34% to +95.07%; annualized return fell from +25.72% to +25.02%; max drawdown worsened from 8.94% to 9.24%.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Single-ETF realized PnL did not capture portfolio path effects. Even a weak individual ETF may diversify, occupy a slot during otherwise worse opportunities, or change later cash deployment.
Can it be revisited? yes
Conditions for revisiting: Only as part of a broader ETF-pool design rule tested after freeze across validation periods; do not remove solely because training-period standalone PnL is negative.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.90` and `min_signal_hold_days=5`
Experiment: Require `volume_score > 0` for new buys.
Hypothesis: Up-cross buy signals should be more reliable when confirmed by volume expansion.
Training result: Return fell from +98.34% to +67.36%; annualized return fell from +25.72% to +18.78%; max drawdown worsened from 8.94% to 9.69%; average exposure fell from 74.46% to 69.60%.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: A hard volume confirmation filter removed too many valid ETF signals, especially for cross-market/QDII-style ETFs where local turnover patterns are not always a clean confirmation signal.
Can it be revisited? yes
Conditions for revisiting: Only as a soft score or ETF-type-specific diagnostic, not as a hard universal buy gate.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.90` and `min_signal_hold_days=5`
Experiment: Remove `159920` Hang Seng ETF from the cross-signal ETF pool.
Hypothesis: `159920` had weak realized contribution in the 2019-2021 training replay and may drag the cross-signal pool.
Training result: Return improved only modestly from +98.34% to +99.70%; annualized return improved from +25.72% to +26.01%; max drawdown improved from 8.94% to 8.50%.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it was not adopted: The improvement is small and highly likely to be period/market-regime-specific. Removing a broad cross-market ETF because Hong Kong was weak in 2019-2021 risks overfitting the pool to the training window.
Can it be revisited? yes
Conditions for revisiting: Only if reserved validation periods also show persistent weakness or if a broader ETF-pool construction rule justifies excluding it.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.90` and `min_signal_hold_days=5`
Experiment: Add profit-segmented ATR tightening: 2.0x ATR after 5% profit and 1.5x ATR after 15% profit, with lower stop floors.
Hypothesis: Once a position has a profit cushion, a tighter trailing stop may preserve gains without hurting entry quality.
Training result: Return fell from +98.34% to +89.59%; annualized return fell from +25.72% to +23.84%; max drawdown was 9.04%; ATR exits rose from 26 to 33.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The current cross-signal framework benefits from giving profitable ETF moves room to breathe. Profit tightening clipped some larger trends and did not materially reduce drawdown.
Can it be revisited? yes
Conditions for revisiting: Only with regime-specific logic or after validation shows profit giveback is a repeated out-of-sample weakness.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Coarse buy-threshold sweep at 55, 60, 65, and 70.
Hypothesis: After the sell-noise filter, the buy gate may need to be looser or stricter than the original 60-point threshold.
Training result: `55` returned +76.40% with 11.27% max drawdown; `60` returned +106.17% with 9.35% max drawdown; `65` returned +62.65% with 9.50% max drawdown; `70` returned +83.80% with 12.77% max drawdown.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The current 60-point threshold is already the best broad gate in training. Lowering it adds noisy entries; raising it misses too many valid reversal/trend entries. No parameter change is justified.
Can it be revisited? yes
Conditions for revisiting: Only after a structural indicator change alters the meaning of `buy_score`; do not fine-tune thresholds around 60.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Coarse sell-threshold sweep at 25, 30, 35, and 40.
Hypothesis: After adding the one-week minimum hold, normal signal sells may need a different force-sell threshold.
Training result: `25` and `30` produced identical paths: +106.17% return, 9.35% max drawdown, 103 buys, 101 sells. `35` returned +85.31% with 7.64% max drawdown. `40` returned +74.35% with 8.99% max drawdown.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The active sell-score clusters already make 25 and 30 equivalent, while higher thresholds delay exits too much and materially reduce return. No threshold change is justified.
Can it be revisited? yes
Conditions for revisiting: Only if sell-score components change; do not fine-tune the current 30 threshold.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Coarse max-hold sweep at 2, 3, 4, and 5 holdings.
Hypothesis: Higher return might come from either more concentrated winners or broader diversification.
Training result: `max_hold=2` returned +90.22% with 10.08% max drawdown and Sharpe 1.4942; `max_hold=3` returned +106.17% with 9.35% max drawdown and Sharpe 1.8866; `max_hold=4` returned +66.02% with 8.36% max drawdown and Sharpe 1.5005; `max_hold=5` returned +50.62% with 8.69% max drawdown and Sharpe 1.3131.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The current 3-holding structure is already the best broad concentration/diversification balance in training. Two holdings concentrate risk too much; four or five holdings dilute signal quality and admit weaker candidates.
Can it be revisited? yes
Conditions for revisiting: Only after ETF pool or signal scoring changes materially; do not fine-tune holding count around the current framework.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Candidate sorting variants: reversal-first, location-first, risk-adjusted buy-minus-sell, and reversal-plus-location.
Hypothesis: The strategy might improve by prioritizing purer reversal or lower-location candidates instead of total buy score.
Training result: Baseline, reversal-first, risk-adjusted, and reversal-plus-location produced identical paths: +106.17% return, 9.35% max drawdown, Sharpe 1.8866. Location-first returned +102.08% with 8.86% max drawdown and Sharpe 1.8724.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Candidate conflicts are rare under the current filters, so most alternate sorting rules do not change the path. Emphasizing location first slightly worsens return without a meaningful risk-adjusted improvement.
Can it be revisited? yes
Conditions for revisiting: Only if buy filters are loosened or new indicators create more competing candidates.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Profit-position signal-sell protection: skip normal signal sell when profitable and still strong-buy, when profitable and buy-score remains above threshold, or when profitable without severe structure break.
Hypothesis: Letting profitable positions run longer might improve return while ATR stops keep risk bounded.
Training result: All three variants produced the same path as baseline: +106.17% return, 9.35% max drawdown, Sharpe 1.8866, 103 buys and 101 sells.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Current signal sells do not materially occur in the targeted state. When sells happen, positions usually lack the profitable/strong-buy condition, or the existing sell rules already handle the case.
Can it be revisited? yes
Conditions for revisiting: Only after sell-score components or ATR logic change materially.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Component-gate diagnostics for KDJ, location score, volume/trend confirmation, and KDJ+high-location rejection.
Hypothesis: Entry attribution suggested KDJ-tagged trades and high-location-score trades were weaker, so filtering them might reduce noisy entries.
Training result: Baseline returned +106.17% with 9.35% max drawdown and Sharpe 1.8866. Rejecting KDJ-up candidates collapsed return to +3.65% with 14.24% max drawdown. Rejecting location_score >= 15 returned +41.72% with 12.64% max drawdown. Requiring volume confirmation or strong trend returned +81.21% with 10.71% max drawdown. Rejecting both KDJ and high-location candidates returned -9.74%.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Single-label trade attribution was misleading because indicators affect the entire portfolio path. KDJ is a load-bearing trigger even though standalone KDJ-tagged closed trades looked mediocre.
Can it be revisited? yes
Conditions for revisiting: Only as a reweighting experiment after signal scoring is redesigned; do not hard-filter KDJ or location-score candidates in the current framework.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Use volume as a soft preference: volume-first sorting, +5 volume boost, and +10 volume boost.
Hypothesis: Volume-confirmed entries had better standalone trade quality, so soft preference might improve ranking without the damage caused by a hard volume gate.
Training result: Baseline returned +106.17% with 9.35% max drawdown and Sharpe 1.8866. Volume-first returned +100.61% with 8.84% max drawdown and Sharpe 1.8702. +5 volume boost returned +72.71% with 12.19% max drawdown. +10 volume boost returned +91.32% with 11.92% max drawdown.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Volume confirmation is correlated with some good trades but is not a stable primary selection signal across ETF types. Boosting it changes path quality for the worse.
Can it be revisited? yes
Conditions for revisiting: Only as an ETF-type-specific diagnostic or after QDII/cross-market volume behavior is modeled separately.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Cap new buys by ADX at 35, 30, 25, and 20 to avoid chasing overly mature trends.
Hypothesis: Entry attribution showed very high ADX entries had lower win rate and P/L ratio; filtering extreme trend strength might reduce chase risk.
Training result: Baseline returned +106.17% with 9.35% max drawdown and Sharpe 1.8866. ADX<=35 returned +80.04%; ADX<=30 returned +83.32%; ADX<=25 returned +80.22%; ADX<=20 returned +79.57%. Drawdowns improved, but returns and Sharpe were lower than baseline.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Strong ADX can mark mature trends, but it also captures some high-payoff momentum continuation. A hard ADX cap sacrifices too much upside.
Can it be revisited? yes
Conditions for revisiting: Only as a position-sizing modifier or if validation later shows high-ADX chase risk is a repeated weakness.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Coarse overheat RSI sweep at 75, 80, 85, 90, and 95.
Hypothesis: A lower RSI overheat threshold might prevent chase entries.
Training result: RSI 75 returned +100.43%; RSI 80 returned +106.88% with 9.04% max drawdown and Sharpe 1.8962; RSI 85/90/95 produced the current baseline path at +106.17%, 9.35% max drawdown, Sharpe 1.8866.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it was not adopted: RSI 80 is only marginally better than 85 in the training window (+0.71pp return). This is too small for a pure threshold change and risks parameter overfitting.
Can it be revisited? yes
Conditions for revisiting: Only if reserved validation shows repeated high-RSI chase entries, or if a broader anti-overheat rule is adopted for market-structure reasons.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Signal-sell structure variants: severe-only confirmation, remove BOLL-mid confirmation, and MA20/falling-MA10-only confirmation.
Hypothesis: Normal signal sells may still be too sensitive if soft confirmations such as BOLL-mid weakness trigger exits.
Training result: All variants produced the identical path as baseline: +106.17% return, 9.35% max drawdown, Sharpe 1.8866, 103 buys and 101 sells.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The current signal sells that actually execute already satisfy the harder structure conditions. Soft confirmation terms are not changing the realized training path.
Can it be revisited? yes
Conditions for revisiting: Only after sell-score components change or after validation reveals soft-confirmation exits not visible in training.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Conditional volume confirmation for MA20-repair / non-BOLL-low entries.
Hypothesis: 2021Q3 diagnostics showed no-volume-confirmation entries were especially weak, while global volume confirmation was too strict. A narrower rule might filter Q3 noise without killing full-period winners.
Training result: Baseline returned +106.17%, max drawdown 9.35%, Sharpe 1.8866. Conditional volume variants all returned +84.03%, max drawdown 8.24%, Sharpe 1.6476, with 106 buys and 104 sells.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The Q3-specific volume weakness does not generalize cleanly across the full training window. Even conditional volume gates remove too many valid recovery entries.
Can it be revisited? yes
Conditions for revisiting: Only with a broader regime detector that independently identifies the weak environment before applying volume confirmation.

Date: 2026-07-07
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Market-state guards using `510300` T-1 state: block A-share buys during downside continuation, block A-share buys below MA60, and halve all new-buy targets below MA60.
Hypothesis: 2021Q3 weakness may be reduced by avoiding A-share exposure or reducing risk when the broad A-share market is weak.
Training result: Baseline returned +106.17%, max drawdown 9.35%, Sharpe 1.8866. Blocking A-share buys during downside continuation returned +101.56%, max drawdown 9.17%, Sharpe 1.8526. Blocking A-share buys below MA60 returned +86.36%, max drawdown 15.31%, Sharpe 1.6573. Halving all new-buy targets below MA60 returned +83.19%, max drawdown 8.12%, Sharpe 1.7725.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Simple market-state guards cut too many profitable opportunities. The cross-signal framework already uses cross-market and cross-asset switching; broad market filters reduce upside more than they reduce risk.
Can it be revisited? yes
Conditions for revisiting: Only with a more nuanced regime rule that preserves cross-market opportunity and is validated after the training rule set is frozen.

Date: 2026-07-08
Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Experiment: Scale new-buy target value when `volume_score == 0`, using coarse position scales `1.00`, `0.75`, `0.50`, `0.25`, and `0.00`.
Hypothesis: 2021Q3 diagnostics showed no-volume-confirmation entries were especially weak. A soft position-size reduction might reduce false-reversal damage without the path destruction caused by hard volume filters.
Training result: Baseline scale `1.00` returned +106.17%, annualized +27.36%, max drawdown 9.35%, Sharpe 1.887, Sortino 2.931, Q3 return -5.32%. Scale `0.75` returned +98.02%, max drawdown 7.82%, Sharpe 1.865, Q3 -4.19%. Scale `0.50` returned +90.06%, max drawdown 7.07%, Sharpe 1.826, Q3 -2.99%. Scale `0.25` returned +81.93%, max drawdown 7.03%, Sharpe 1.760, Q3 -1.83%. Scale `0.00` returned +72.92%, max drawdown 10.13%, Sharpe 1.456, Q3 -4.37%.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The rule does reduce the 2021Q3 weak spot, but the full training-window opportunity cost is too large. The best drawdown variants lose 16-24pp total return and reduce Sharpe/Sortino, while hard blocking also worsens max drawdown. No adoption is justified.
Can it be revisited? yes
Conditions for revisiting: Only as part of a broader independently defined regime/ETF-type sizing model. Do not use `volume_score == 0` alone as a global position-sizing rule.
