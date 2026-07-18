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

Date: 2026-07-08
Version: cross-signal after `a_share_zero_volume_buy_scale=0.50`
Experiment: Sell-side structure variants: remove all normal signal sells, require severe break only, require MA20-or-severe break, raise sell threshold to 35, and raise sell threshold to 40.
Hypothesis: Post-sell diagnostics showed normal `signal_sell` exits are often followed by rebounds, so weakening signal sells might reduce sell-fly damage.
Training result: Baseline returned +109.19%, annualized +27.98%, max drawdown 7.86%, Sharpe 1.995, Sortino 3.113, 103 buys and 101 sells. Removing signal sells returned +104.35%, annualized +26.98%, max drawdown 7.35%, Sharpe 1.904, with only 40 buys and 37 sells. Severe-only signal sells returned +80.03%, max drawdown 9.30%, Sharpe 1.583. MA20-or-severe produced the identical path as baseline. Sell threshold 35 returned +84.32%, max drawdown 6.59%, Sharpe 1.701. Sell threshold 40 returned +76.65%, max drawdown 7.83%, Sharpe 1.588.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Sell-fly exists, but normal signal sells also free capital for better opportunities. Weakening or removing them reduces churn and sometimes drawdown, but the opportunity cost is larger than the saved sell-fly damage. The current sell structure remains the best full-path training result.
Can it be revisited? yes
Conditions for revisiting: Only with a targeted rule that identifies sell-fly states while preserving capital recycling. Do not globally remove signal sells or raise the sell threshold in the current framework.

Date: 2026-07-08
Version: cross-signal after `a_share_zero_volume_buy_scale=0.50`
Experiment: ETF-pool deletion based on training attribution: remove `510880`, remove `159920`, remove both, or remove `510300/510880/159920`.
Hypothesis: ETF-level attribution showed `510880` and `159920` as drag symbols and `510300` as weak contributor; removing weak symbols might improve capital allocation.
Training result: Baseline returned +109.19%, max drawdown 7.86%, Sharpe 1.995. Removing `510880` returned +105.88%, max drawdown 8.56%, Sharpe 1.939. Removing `159920` returned +111.13%, max drawdown 7.40%, Sharpe 2.029. Removing `510880` and `159920` returned +108.13%, max drawdown 8.03%, Sharpe 1.978. Removing `510300/510880/159920` returned +113.44%, max drawdown 6.94%, Sharpe 2.049.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it was not adopted yet: The best local result comes from deleting symbols after observing training attribution, which has high selection-bias risk. It is a candidate for JoinQuant training confirmation, not an adopted strategy rule.
Can it be revisited? yes
Conditions for revisiting: Run the candidate in JoinQuant over the 2019-2021 training window first. If JoinQuant confirms improvement, document it as a training-confirmed candidate before any reserved validation.

Date: 2026-07-08
Version: `cross-v0.3.1`
Experiment: ATR-stop cooldown after a position is stopped out, tested with cooldown windows of 1, 2, 3, and 5 trading days before the same ETF can be bought again.
Hypothesis: The 2020 max-drawdown interval showed clustered ATR stops followed by new reversal entries. A short post-stop cooldown might avoid re-entering the same ETF too early during crash/noisy regimes without weakening the core signal framework.
Training result: Baseline cooldown 0 returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. Cooldown 1 and 2 produced the identical path as baseline. Cooldown 3 returned +113.25%, annualized +28.80%, max drawdown 6.94%, Sharpe 2.037. Cooldown 5 returned +95.84%, annualized +25.19%, max drawdown 6.94%, Sharpe 1.822, with 99 buys and 96 sells.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Short cooldowns do not touch the realized path, while a one-week cooldown blocks useful re-entries and materially reduces return without improving drawdown. The max-drawdown problem is not solved by a simple per-ETF post-ATR cooldown.
Can it be revisited? yes
Conditions for revisiting: Only as part of a broader independently defined crash/regime model. Do not adopt a standalone ATR-stop cooldown in the current framework.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Buy-entry confirmation sizing: weak confirmation half-size, weak confirmation 40% size, ultra-weak confirmation skip, and three-confirmation half-size.
Hypothesis: Training attribution showed better standalone quality when entries had stronger trend or volume confirmation. Scaling down entries without trend/volume confirmation might keep the cross-signal reversal core while reducing false-reversal damage.
Training result: Baseline returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. Weak confirmation half-size returned +107.95%, max drawdown 6.96%, Sharpe 2.035. Weak confirmation 40% size returned +107.13%, max drawdown 6.98%, Sharpe 2.029. Ultra-weak skip returned +90.19%, max drawdown 6.98%, Sharpe 1.827. Three-confirmation half-size returned +78.67%, max drawdown 5.62%, Sharpe 1.963.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Weak-confirmation entries include enough large winners that simple sizing cuts reduce return more than they reduce drawdown. The drawdown improvement is too small or absent, while opportunity cost is large.
Can it be revisited? yes
Conditions for revisiting: Only inside a broader regime-aware model that can identify when weak confirmation is genuinely dangerous. Do not use trend/volume weakness alone as a global position-size cut.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Buy-candidate ranking by confirmation strength: confirmation-first ranking, trend-first ranking, volume/trend-first ranking, and buy+trend+volume quality-sum ranking.
Hypothesis: If trend and volume confirmation improve entry quality, they might be better used to choose among simultaneous candidates rather than to cut position size after selection.
Training result: Baseline returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. Confirmation-first returned +96.19%, max drawdown 6.99%, Sharpe 1.877. Trend-first returned +94.31%, max drawdown 6.98%, Sharpe 1.813. Volume/trend-first returned +89.15%, max drawdown 6.98%, Sharpe 1.782. Quality-sum returned +98.92%, max drawdown 6.93%, Sharpe 1.920.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Confirmation strength is useful context but not a better primary ranking key than the current cross-signal score. Ranking by confirmation displaces high-payoff reversal opportunities and lowers both return and risk-adjusted quality.
Can it be revisited? yes
Conditions for revisiting: Only after the buy-score formula is redesigned from first principles. Do not reorder candidates by trend or volume confirmation in the current framework.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Extend normal signal-sell minimum hold from 5 trading days to 7, 10, and 15 trading days. ATR stops remain unconditional.
Hypothesis: Post-sell diagnostics showed normal `signal_sell` exits were often followed by positive 3/5/10-day returns, so some signal sells may be too early. A longer minimum hold might reduce sell-fly behavior.
Training result: Baseline 5-day minimum hold returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. 7 days returned +108.92%, max drawdown 8.14%, Sharpe 1.956. 10 days returned +92.13%, max drawdown 8.87%, Sharpe 1.708. 15 days returned +105.70%, max drawdown 7.95%, Sharpe 1.926.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Sell-fly exists, but delaying all normal signal sells traps capital in mediocre positions and weakens capital recycling. Higher win rate under longer holds did not compensate for lower total return and worse drawdown.
Can it be revisited? yes
Conditions for revisiting: Only with a targeted sell-fly detector that preserves capital recycling. Do not globally extend the normal signal-sell minimum hold in the current framework.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Raise normal signal-sell threshold from 30 to 35, 40, and 45.
Hypothesis: If normal signal sells are too sensitive, a coarser sell threshold might reduce premature exits while ATR stops remain as hard risk control.
Training result: Baseline sell threshold 30 returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. Threshold 35 returned +80.75%, max drawdown 6.18%, Sharpe 1.661. Threshold 40 returned +69.88%, max drawdown 6.47%, Sharpe 1.520. Threshold 45 returned +87.23%, max drawdown 7.11%, Sharpe 1.699.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The current sell threshold is load-bearing. Raising it reduces churn, but the lost capital recycling is much larger than the saved premature-exit damage.
Can it be revisited? yes
Conditions for revisiting: Only if sell-score components are redesigned. Do not raise `sell_threshold` as a standalone optimization.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Targeted sell-fly protection using sell-time volume/trend confirmation. Variants protected normal `signal_sell` when exit-time `volume_score >= 4` and `trend_score >= 14`, `volume_score >= 4` and `buy_score >= 35`, `volume_score >= 4` and `sell_risk_score >= 10`, `trend_score >= 14` and `buy_score >= 35`, all three confirmations, or high sell score with confirmation. ATR stops remained unconditional.
Hypothesis: Sell-fly diagnostics showed that some normal signal sells were followed by positive forward returns, especially when exit-time volume/trend context remained constructive. Protecting only those states might reduce premature exits without removing signal sells globally.
Training result: Baseline returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201. Tested variants returned between +80.95% and +106.52%; all had lower Sharpe than baseline and most had worse drawdown. The best variant, `trend_score >= 14 and buy_score >= 35`, returned +106.52% with 7.24% max drawdown and Sharpe 1.939.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: Sell-fly exists, but exit-time trend/volume confirmation is too broad. It blocks too many necessary capital-recycling sells and keeps positions that later become mediocre.
Can it be revisited? yes
Conditions for revisiting: Only with a more precise replacement-aware or opportunity-cost model. Do not protect signal sells globally just because volume/trend remains positive.

Date: 2026-07-09
Version: `cross-v0.3.1`
Experiment: Replacement-aware signal-sell protection. Variants skipped a normal `signal_sell` only when no eligible replacement buy candidate existed, optionally requiring current buy score >=30, trend score >=14, or no downside-continuation state. ATR stops remained unconditional.
Hypothesis: Normal signal sells are valuable mostly when they recycle capital into a better candidate. If no replacement exists, selling may only create idle cash and increase sell-fly risk.
Training result: Baseline returned +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201. Skipping signal sells when no replacement existed returned +113.93%, annualized +28.94%, max drawdown 6.94%, Sharpe 2.026, Sortino 3.190, with 62 buys and 59 sells. Other variants returned +102.11% to +107.10% with lower Sharpe.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it was not adopted: The all-no-replacement variant was a near tie with slightly higher return but lower Sharpe and materially fewer trades. The improvement is too small for a new rule and may be path noise.
Can it be revisited? yes
Conditions for revisiting: If later training-only diagnostics show turnover/friction is a larger problem than currently measured, this can be reconsidered as a simplicity/turnover rule rather than a return enhancer.

Date: 2026-07-09
Version: `cross-v0.3.1-atr2-candidate`
Experiment: Tighten ATR trailing stop multiplier from 2.5 to 2.0 while keeping `stop_floor=0.05`, `stop_cap=0.15`, buy/sell signals, position sizing, and minimum signal hold unchanged.
Hypothesis: A slightly tighter trailing stop might preserve gains after cross-signal entries without changing the signal model. This is a broad risk-control idea, not a precise threshold fit.
Local training result: Local replay over 2019-2021 returned +115.87%, annualized +29.33%, max drawdown 6.97%, Sharpe 2.076, Sortino 3.249, versus local baseline +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201.
JoinQuant training result: JoinQuant over 2019-2021 returned +121.37%, annualized +31.28%, max drawdown 6.85%, Sharpe 2.990, Sortino 0.746, win rate 0.563, profit/loss ratio 4.298, versus official mainline +122.47%, annualized +31.50%, max drawdown 6.38%, Sharpe 3.057, Sortino 0.759, win rate 0.552, profit/loss ratio 4.466.
Validation result: Not run. Per protocol, validation periods were not used to tune or decide this candidate.
Why it failed: The local improvement did not survive the JoinQuant training authority check. JoinQuant showed lower return, worse drawdown, lower Sharpe/Sortino, and lower profit/loss ratio; the slightly higher win rate was not worth the risk-adjusted deterioration.
Can it be revisited? no
Conditions for revisiting: Do not revisit simple ATR multiplier tightening unless a later, independent structural change creates a new stop-loss failure mode. Avoid further fine-grained ATR multiplier searching because it would become parameter mining.

Date: 2026-07-09
Version: `cross-v0.3.1-no-512100-candidate`
Experiment: Remove `512100.XSHG` from the official `cross-v0.3.1` ETF pool while keeping every signal, risk, sizing, and execution rule unchanged.
Hypothesis: `512100` was the weakest local realized contributor and overlaps with A-share broad-base exposure. Removing it might simplify the pool and reduce weak capital allocation without changing the strategy family.
Local training result: Local replay over 2019-2021 improved slightly from +113.44% to +114.57%, max drawdown moved from 6.94% to 6.99%, Sharpe improved from 2.049 to 2.094, and Sortino improved from 3.201 to 3.285.
JoinQuant training result: JoinQuant over 2019-2021 returned +119.31%, annualized +30.86%, max drawdown 6.82%, Sharpe 2.977, Sortino 0.711, win rate 0.548, profit/loss ratio 4.399, versus official mainline +122.47%, annualized +31.50%, max drawdown 6.38%, Sharpe 3.057, Sortino 0.759, win rate 0.552, profit/loss ratio 4.466.
Validation result: Not run. Per protocol, validation periods were not used to tune or decide this candidate.
Why it failed: The small local improvement did not survive the JoinQuant training authority check. The candidate was worse on return, annualized return, max drawdown, Sharpe, Sortino, win rate, and profit/loss ratio.
Can it be revisited? no
Conditions for revisiting: Do not remove `512100` based on standalone attribution or local replay alone. Pool deletion has high selection-bias risk and must be justified by broader pool-design logic, not one training-window contributor ranking.

Date: 2026-07-10
Version: `cross-v0.3.2-sell35-candidate`
Experiment: Raise the normal signal-sell threshold from 30 to 35 on top of the adopted `cross-v0.3.2` mainline, while keeping ATR stops unconditional and all buy logic unchanged.
Hypothesis: Fresh post-sell diagnostics on the `v0.3.2`/combo training log showed some `sell_score 32-34` signal sells were followed by positive 10-day forward returns, so a higher forced-sell threshold might reduce premature exits.
Local training result: Mainline-equivalent local replay with `sell_threshold=30` returned +118.75%, annualized +29.90%, max drawdown 6.81%, Sharpe 2.117, Sortino 3.327, 99 buys, 96 sells, closed-trade win rate 0.552, profit/loss ratio 4.034. Candidate `sell_threshold=35` returned +86.08%, annualized +23.07%, max drawdown 5.97%, Sharpe 1.757, Sortino 2.713, 92 buys, 89 sells, closed-trade win rate 0.528, profit/loss ratio 3.303.
JoinQuant training result: Not run because the local direction check materially failed.
Validation result: Not run. Per protocol, validation periods were not inspected.
Why it failed: The broad threshold increase reduces drawdown, but it damages return and risk-adjusted quality too much. The low-threshold signal sell remains load-bearing for capital recycling even though some individual weak-score exits sell early.
Can it be revisited? no as a standalone threshold raise
Conditions for revisiting: Only study narrower sell-quality rules that preserve capital recycling; do not raise global `sell_threshold` again without a new structural reason.

Date: 2026-07-10
Version: `cross-v0.3.2-weak-replacement-candidate`
Experiment: Protect weak normal signal sells on top of `cross-v0.3.2`: when `sell_score` is between the normal threshold and 35, current `buy_score >= 35`, and selling would leave no eligible replacement buy candidate, skip the signal sell. ATR stops remain unconditional.
Hypothesis: A replacement-aware rule might reduce sell-fly and idle-cash damage without weakening high-conviction risk exits or broadening the global sell threshold.
Local training result: Mainline-equivalent local replay with `sell_threshold=30` returned +118.75%, annualized +29.90%, max drawdown 6.81%, Sharpe 2.117, Sortino 3.327, 99 buys, 96 sells, closed-trade win rate 0.552, profit/loss ratio 4.034. The adopted local candidate variant returned +119.82%, annualized +30.12%, max drawdown 6.86%, Sharpe 2.120, Sortino 3.335, 99 buys, 96 sells, closed-trade win rate 0.552, profit/loss ratio 4.107, and protected only 2 sells.
JoinQuant training result: JoinQuant 2019-2021 was run with the candidate version and produced the same headline result as official `cross-v0.3.2`: +125.82% return, +32.18% annualized, 6.70% max drawdown, Sharpe 3.109, Sortino 0.799, win rate 0.558, profit/loss ratio 4.845. Log audit confirmed the candidate initialized, but the candidate-specific protection log `[hold] ... weak sell_score ... no replacement, skip signal sell` appeared 0 times.
Validation result: Not run. Per protocol, validation periods were not inspected for this no-effect training candidate.
Why it failed: The local improvement was too small and did not translate into an actual JoinQuant path change. Under the JoinQuant authority path, the rule never triggered, so it adds complexity without measurable effect.
Can it be revisited? no as currently defined
Conditions for revisiting: Only revisit replacement-aware selling if future training-only diagnostics show a materially larger set of weak signal sells with no replacement. Any revised rule must include explicit trigger-count logging before JoinQuant testing.

Date: 2026-07-10
Version: `cross-v0.3.2-low-bounce-candidate`
Experiment: Block new buys where RSI and KDJ cross up, MACD does not cross up, price is in the BOLL lower-to-middle/near-MA20 repair zone, volume confirmation is positive, and trend score is positive but below strong-trend level (`0 < trend_score < 20`).
Hypothesis: A low-position volume bounce with RSI/KDJ timing but no MACD confirmation and no strong trend support may be a false rebound. Filtering this pattern might remove weak entries without changing the core cross-signal strategy.
Local training result: Local 2019-2021 replay improved from official mainline +118.75% return, 6.81% max drawdown, 99 buys, 96 sells, and end value 43749.40 to candidate +122.49% return, 7.20% max drawdown, 95 buys, 92 sells, and end value 44498.60.
JoinQuant training result: JoinQuant 2019-2021 returned +124.73%, annualized +31.96%, max drawdown 7.00%, Sharpe 3.127, Sortino 0.778, win rate 0.582, profit/loss ratio 5.117, 53 profitable trades, and 38 losing trades. Official `cross-v0.3.2` returned +125.82%, annualized +32.18%, max drawdown 6.70%, Sharpe 3.109, Sortino 0.799, win rate 0.558, profit/loss ratio 4.845, 53 profitable trades, and 42 losing trades.
Operational result: The candidate log contained 94 buy events and 92 sell events, versus the mainline-equivalent local path of 99 buys and 96 sells, so the rule changed the trade path. There were 0 ERROR-level logs. The two WARNING lines were both the already-understood 2019-12-12 `513880.XSHG` zero-volume market-order matching event, not a strategy exception.
Validation result: Not run. Per protocol, reserved validation periods were not inspected for this failed training candidate.
Why it failed: The filter removed four losing trades and improved win rate, profit/loss ratio, and Sharpe slightly, but it also removed enough profitable opportunity to lower total and annualized return. More importantly, max drawdown worsened from 6.70% to 7.00% and Sortino fell from 0.799 to 0.778. The local gain did not survive the JoinQuant authority check on the primary return-and-drawdown objective.
Can it be revisited? no as currently defined
Conditions for revisiting: Do not widen or tune this pattern with extra thresholds. A future entry filter must come from a new training-only structural hypothesis and must beat the official mainline on JoinQuant without trading lower return for worse downside risk.

Date: 2026-07-10
Version: `cross-v0.3.2-backup-fill-local-candidate`
Experiment: Keep the official 60-point primary buy threshold, then fill only remaining slots with 50-59 point candidates that have a reversal cross and pass all other official entry filters. Primary candidates always rank first and cannot be displaced.
Hypothesis: The capital-utilization diagnostic found 51 independent 50-59 point rejected-signal episodes with positive average 5/10/20-day shadow returns. A backup-only rule might use otherwise idle slots without weakening the main entry gate.
Local training result: Official local baseline returned +118.75%, max drawdown 6.81%, 99 buys, 96 sells, and 0.732 average exposure. The candidate returned +86.39%, max drawdown 9.17%, 110 buys, 107 sells, and 0.810 average exposure. It executed 50 backup buys, reduced return by 32.35 percentage points, and worsened drawdown by 2.36 percentage points.
JoinQuant training result: Not run because the local direction check materially failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Fixed-horizon shadow returns did not model portfolio opportunity cost. Backup positions consumed slots and cash that later stronger primary signals needed, while the existing ATR and signal exits realized a much worse path than the isolated 5/10/20-day snapshots suggested.
Can it be revisited? no as a score-only backup fill
Conditions for revisiting: Do not lower the buy threshold or mechanically fill idle slots based on fixed-horizon shadow returns. Revisit only if a new independent signal dimension can identify backup candidates without relying on the existing buy score alone.

Date: 2026-07-10
Version: `cross-v0.3.2` observation-only CMF gate
Experiment: Pre-register a standard CMF(20) zero-line confirmation for mild-trend entries (`0 < trend_score < 20`) while leaving strong-trend entries unchanged. Run attribution first; implement a candidate only if the mild-trend relationship is consistent across training years.
Hypothesis: Positive CMF might distinguish genuine accumulation from false low-position reversal crosses in mild trends without interfering with proven strong-trend entries.
Training diagnostic result: Mild-trend `CMF <= 0` had 17 trades, +3412.60 PnL, 64.71% win rate, and 3.924 profit/loss ratio. Mild-trend `CMF > 0` had 52 trades, +5843.80 PnL, 46.15% win rate, and 2.218 profit/loss ratio. Year-level CMF direction was inconsistent. The aggregate positive-CMF advantage came mainly from strong-trend trades, not the pre-specified mild-trend target.
Candidate result: Not implemented because the observation gate failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: CMF sign does not provide the intended mild-trend quality separation. Negative/non-positive money flow can be natural near a low-position reversal, and filtering it would remove profitable early entries.
Can it be revisited? no as a mild-trend zero-line gate
Conditions for revisiting: Do not tune CMF periods or thresholds. A future CMF experiment would require a new independent market-structure rationale and must not be inferred from the six non-positive strong-trend trades in this same training attribution.

Date: 2026-07-10
Version: `cross-v0.3.2` observation-only strong-trend capacity gate
Experiment: Before changing position sizing, identify filled `trend_score >= 20` buys where all same-day official candidates were already allocated, at least one slot remained unused, and cash above the official reserve could fund one additional copy of the planned target. Only the highest-ranked strong buy could qualify per day.
Hypothesis: Strong-trend entries might use otherwise idle capital without loosening the buy gate or displacing another valid signal.
Training diagnostic result: All strong-trend entries produced 26 closed trades and +14847.60 PnL. The executable capacity subset contained only 5 closed trades and +1371.00 PnL. Year counts were 2/2/1, the single 2021 trade lost 44.80, and one ETF contributed 60.31% of capacity gross profit.
Candidate result: Not implemented because the pre-registered sample-size, yearly-profitability, and concentration gates failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The profitable strong-trend population and the actually scalable idle-capacity population are different. Most strong buys did not leave a complete unused slot after other official candidates, and the remaining five trades are too sparse and concentrated for a defensible concentration rule.
Can it be revisited? no as a one-extra-slot strong-trend rule
Conditions for revisiting: Do not search smaller multipliers or relax the cash/slot definition. Revisit sizing only after an independent strategy change creates a materially larger executable strong-signal capacity sample.

Date: 2026-07-10
Version: `cross-v0.3.2` observation-only 09:35 gap gate
Experiment: Group filled buys by `(T-day 09:35 raw price - T-1 close) / T-1 ATR` using fixed boundaries `<=0`, `(0, 0.5]`, `(0.5, 1]`, and `>1 ATR`. Create a candidate only if `>1 ATR` high gaps consistently underperform in all three training years with adequate samples.
Hypothesis: A valid low-position T-1 cross signal may become an expensive chase when the next 09:35 execution price gaps more than one ATR above the prior close.
Training diagnostic result: The `>1 ATR` group had only 5 trades, but it earned +3309.80 PnL, +8.15% average return, 60.00% win rate, and a 9.120 profit/loss ratio. The 2019 and 2020 groups were strongly profitable. Four of the five trades were strong-trend entries and produced +3340.20.
Candidate result: Not implemented because the minimum sample and annual-underperformance gates failed, and the observed direction contradicted the hypothesis.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Large positive gaps were rare strong-trend continuation events rather than a stable source of chase losses. A broad `>1 ATR` block would remove some of the training path's highest-quality entries.
Can it be revisited? no as a broad `>1 ATR` entry filter
Conditions for revisiting: Do not search smaller ATR thresholds or add the post-hoc mild-trend interaction seen in this attribution. A future execution filter requires a new independent market-structure hypothesis.

Date: 2026-07-10
Version: `cross-v0.3.2` observation-only BOLL BandWidth gate
Experiment: Keep standard BOLL(20,2), calculate T-1 `BandWidth = (upper-lower)/middle`, and compare rising versus non-rising direction for mild-trend entries. Create a candidate only if rising width improves average return and win rate in every training year with adequate samples.
Hypothesis: Rising BandWidth might distinguish a mild-trend reversal that is expanding into a genuine move from one that remains compressed and noisy.
Training diagnostic result: Across mild-trend trades, rising width produced 31 trades, +7939.40 PnL, +3.05% average return, 51.61% win rate, and 4.040 profit/loss ratio, versus 38 declining-width trades with +1317.00 PnL, +0.19% average return, 50.00% win rate, and 1.393 profit/loss ratio. However, 2021 reversed the relationship: rising width returned +0.36% with 41.18% win rate, while declining width returned +0.72% with 64.29% win rate.
Candidate result: Not implemented because the annual consistency gate failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: BandWidth direction was useful in 2019-2020 but unstable in the 2021 regime. The aggregate improvement conceals a meaningful annual reversal, so it is not a robust universal confirmation.
Can it be revisited? no as a one-day rising-width mild-trend gate
Conditions for revisiting: Do not tune BOLL periods, width thresholds, or slope windows from these results. A future volatility-regime experiment must use a separately motivated dimension rather than repackaging this failed split.

Date: 2026-07-10
Version: `cross-v0.3.2` observation-only cross-sequence gate
Experiment: Preserve the official three-day cross window, record the latest active RSI/KDJ/MACD upward-cross offsets, and compare mild-trend oscillator-leading with MACD-leading sequences. Create a candidate only if both clean sequence groups are adequately sampled and oscillator-leading is better in every training year.
Hypothesis: Fast RSI/KDJ upward crosses followed by slower MACD confirmation may be higher quality than a MACD cross followed by late oscillator confirmation.
Training diagnostic result: No `macd_leads_oscillators` closed trade occurred. Oscillator-leading had 11 trades overall and only 7 mild-trend trades, with annual mild counts 2/3/2. The 2021 mild oscillator-leading group lost 218.00. Seventy trades had no active MACD upward confirmation and produced +16316.10 PnL.
Candidate result: Not implemented because the comparison group was empty and the oscillator-leading group failed sample-size and annual-stability gates.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The strategy's actual closed-trade path does not contain the proposed MACD-first sequence and is dominated by early oscillator entries without active MACD confirmation. There is no robust sample for the intended filter.
Can it be revisited? no as a MACD-leading mild-trend filter
Conditions for revisiting: Do not widen the cross window, change indicator periods, or promote the post-hoc same-day/mixed groups from this result. A future timing experiment needs a new independently specified mechanism.

Date: 2026-07-11
Version: `cross-v0.3.2-reversal-first-local-candidate`
Experiment: Keep every official signal, filter, risk, sizing, and execution rule unchanged, but rank eligible candidates by `reversal_score`, then `buy_score`, then code instead of official total-buy-score-first ordering. Baseline and candidate used identical precomputed T-1 scores.
Hypothesis: Prioritizing the core cross-reversal dimension might select earlier low-position opportunities when portfolio slots are limited.
Local training result: Official returned +118.75% with 6.81% drawdown, Sharpe 2.117, and Sortino 3.327. Reversal-first returned +121.69% with 6.81% drawdown, Sharpe 2.157, and Sortino 3.403. Both had 99 buys and 96 sells.
Path audit: Only 2021-12-27 changed. Official bought `159928` (buy 70/reversal 35); candidate bought `513500` (buy 69/reversal 45). Through the 2021-12-31 training endpoint, `159928` returned -3.29% and `513500` +0.95% from 09:35.
JoinQuant training result: Not run because the local activity gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The entire improvement came from one selection four trading days before the training boundary. There were no changed decisions in 2019 or 2020 and no repeated ranking evidence.
Can it be revisited? no as a global reversal-first ordering
Conditions for revisiting: Do not search weighted ranking combinations from this event. A future ranking change requires a new independent information dimension that changes a materially larger set of decisions.

Date: 2026-07-11
Version: `cross-v0.3.2` observation-only Kaufman ER gate
Experiment: Calculate standard T-1 Kaufman ER(10), compare one-day rising versus non-rising direction for mild-trend entries, and create a candidate only if rising ER improves average return and win rate in every training year with adequate samples.
Hypothesis: Increasing directional efficiency might distinguish mild-trend reversals developing into clean moves from noisy back-and-forth price paths.
Training diagnostic result: Mild rising ER had 30 trades, +2812.30 PnL, +1.12% average return, 50.00% win rate, and 2.059 profit/loss ratio. Mild declining ER had 38 trades, +6597.10 PnL, +1.89% average return, 52.63% win rate, and 3.092 profit/loss ratio. Rising ER failed 2019 average-return/win-rate comparisons, 2020 win rate, and 2021 average return.
Candidate result: Not implemented because the annual consistency gate failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Rising ER is not a stable mild-trend confirmation. Its aggregate descriptive advantage came from strong-trend entries, which were outside the locked hypothesis and overlap existing trend context.
Can it be revisited? no as a mild-trend rising-ER gate
Conditions for revisiting: Do not search ER thresholds, alternate periods, or slope windows from this result. A future regime experiment must use a separately specified mechanism.

Date: 2026-07-11
Version: `cross-v0.3.2` observation-only portfolio-dependence gate
Experiment: At each official buy decision, measure the maximum standard 20-day return correlation between the candidate and current or earlier same-day planned holdings using only adjusted closes available through T-1. Use one fixed `0.80` high-dependence split and leave all orders unchanged.
Hypothesis: Entries that add a highly correlated ETF may crowd the portfolio and contribute lower returns and deeper adverse excursions than low-dependence entries.
Training diagnostic result: The high-dependence group had 9 closed trades, +3365.50 PnL, +3.92% average return, 66.67% win rate, 5.197 profit/loss ratio, and -1.85% average MAE. The low-dependence group had 79 trades, +17803.90 PnL, +2.50% average return, 54.43% win rate, 3.889 profit/loss ratio, and -1.08% average MAE. In 2019, high dependence underperformed average return (1.05% versus 2.83%) and had slightly worse MAE (-1.06% versus -0.82%). In 2020, however, the 2 high-dependence trades averaged +15.31% versus +4.46% for low dependence. In 2021, high dependence averaged +0.37% versus +0.41%, with worse MAE (-2.66% versus -1.02%).
Candidate result: Not implemented because the pre-registered sample-size and annual-return gates failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Only 9 high-dependence trades existed, including just 2 in 2020, and high dependence was not consistently harmful. In a broad market move, correlation can identify simultaneous strong trends rather than avoidable crowding, so a universal correlation block would remove valid opportunities.
Can it be revisited? no as a 20-day/0.80 correlation gate
Conditions for revisiting: Do not search nearby correlation thresholds, return windows, or ETF-pair exceptions. Revisit only if a future strategy structure explicitly optimizes portfolio covariance with a separately justified allocation objective.

Date: 2026-07-11
Version: `cross-v0.3.2` observation-only market-breadth gate
Experiment: On T-1, calculate the share of eligible pool ETFs above their standard MA20, excluding ETFs with fewer than 20 valid adjusted closes. Compare the single pre-registered states below `50%` versus at least `50%`, and evaluate only official mild-trend entries while leaving all orders unchanged.
Hypothesis: A mild-trend reversal may be less reliable when fewer than half of the eligible ETF pool is above MA20, because the signal is isolated rather than broadly supported.
Training diagnostic result: Across all entries, below-majority breadth had 31 closed trades, +4016.80 PnL, +1.37% average return, 54.84% win rate, and 2.817 profit/loss ratio; majority breadth had 65 trades, +19872.60 PnL, +3.54% average return, 55.38% win rate, and 4.509 profit/loss ratio. For the pre-registered mild-trend subset, 2019 below-majority had 6 trades, +0.61% average return, and 33.33% win rate versus majority's 13 trades, +1.94%, and 46.15%. In 2020, below-majority had 5 trades, +0.78%, and 60.00% win rate versus majority's 14 trades, +3.76%, and 57.14%. In 2021, the relationship reversed materially: below-majority had 14 trades, +1.45%, and 64.29% win rate versus majority's 17 trades, -0.24%, and 41.18%.
Candidate result: Not implemented because the pre-registered annual return-and-win-rate consistency gate failed.
JoinQuant training result: Not run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Breadth below 50% is not uniformly hostile to low-position reversal entries. In 2021 it identified the narrower early leaders that this strategy is designed to catch, so a broad market confirmation would delay or block valid reversals.
Can it be revisited? no as an MA20/50% mild-trend gate
Conditions for revisiting: Do not search MA10/MA60, alternate breadth thresholds, smoothing windows, or A-share-only subsets from this result. Revisit only after a new independently justified breadth objective is pre-registered outside the exhausted indicator-search family.

Date: 2026-07-11
Version: `cross-v0.3.2` observation-only US-QDII previous-NAV premium gate
Experiment: For actual closed `513100/513500` mainline buys, calculate the T-day 09:35 market-price premium over the point-in-time reference proxy, preserve fixed economic groups `<=2%`, `2-5%`, `5-10%`, and `>10%`, and consider a candidate only if the above-5% subset passes coverage, sample-size, cross-year, and cross-code gates.
Hypothesis: A technically valid cross-signal entry in a US-market QDII ETF may still be structurally unattractive when quota or secondary-market demand pushes its A-share price materially above the T-1 reference value.
Training diagnostic result: There were 28 closed `513100/513500` trades, of which 27 had usable 09:35 reference data (96.43% coverage). Twenty-four trades were at or below 2% premium and produced +6509.90 PnL, +2.79% average return, 62.50% win rate, and 5.884 profit/loss ratio. One 2-5% trade produced +1105.10 and +12.88%. Only two trades exceeded 5%; both were `513100` trades in 2020, averaged 8.16% premium and +2.55% return, and together produced +388.80 with 50.00% win rate. No `513500` above-5% trade existed and no trade exceeded 10%.
Candidate result: Not implemented because the above-5% subset had only two trades, appeared in only one year, and came from only one ETF. All pre-registered sample, annual, and cross-code gates failed.
JoinQuant training result: No strategy candidate was run. Two no-order capability probes only established that T-1 unit NAV is available at 09:35, same-day NAV is blocked until after 15:00, and standard JoinQuant market APIs do not expose IOPV.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The official strategy almost never bought these ETFs during elevated-premium episodes. There is no repeated cross-year or cross-code evidence that a premium veto would improve the trading path, while the two observed elevated trades were profitable in aggregate.
Can it be revisited? no as a `513100/513500` previous-NAV premium entry filter
Conditions for revisiting: Do not lower 5%, search nearby bands, add `513050`, or extend the stale previous-NAV proxy to dynamic-IOPV products such as `159920` or `513880`. A future reopening requires a genuinely new point-in-time market mechanism and a new governance approval, not a re-slice of these outcomes.

Date: 2026-07-13
Version: `cross-v0.3.2-macd-6-13-5-candidate`
Experiment: Compare official MACD(12,26,9) with MACD(6,13,5) as the only changed parameter group on 2019-2021; use 2018 only as a read-only indicator warm-up and keep every other signal, threshold, ETF, sizing, risk, and execution rule identical.
Hypothesis: The approximately half-cycle MACD might recognize early reversals sooner and improve the cross-signal strategy without changing its core logic.
Training diagnostic result: The candidate changed 89 filled-order days across all three years (38/11/40). Official local replay returned +120.61% annualized at 30.27%, with 7.47% max drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% closed-trade win rate, and 4.440 profit/loss ratio. MACD(6,13,5) returned +84.69% annualized at 22.76%, with 7.00% max drawdown, 1.766 Sharpe, 2.670 Sortino, 50.00% win rate, and 2.834 profit/loss ratio. Annual returns were 35.84%/49.74%/8.46% versus 17.02%/51.94%/3.87% in 2019/2020/2021.
Candidate result: Rejected by the pre-registered gate. Only maximum drawdown and 2020 return improved; total and annualized return, Sharpe, Sortino, win rate, profit/loss ratio, 2019, and 2021 all worsened.
JoinQuant training result: Not run because the local gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The faster MACD materially increased path sensitivity and trading activity but admitted more short-lived crosses. Its small 2020 improvement did not compensate for large degradation in 2019 and 2021 or the broad fall in risk-adjusted and trade-quality metrics.
Can it be revisited? no as a MACD period search
Conditions for revisiting: Do not test neighboring fast/slow/signal periods, optimize a MACD grid, or combine 6/13/5 with new thresholds after seeing this result. Reopening requires a genuinely new externally justified mechanism and a new explicit research authorization.

Date: 2026-07-14
Version: `cross-v0.3.2` observation-only horizontal price-structure gate
Experiment: Use the prior 20 valid daily bars ending T-2 to define horizontal resistance and support, normalize T-1 close distance by official ATR(14), and test the single fixed hypothesis that mild-uptrend buys within one ATR below resistance underperform all other mild-uptrend buys in every training year.
Hypothesis: An oscillator cross that occurs immediately below established horizontal resistance may have poor upside room and may be a false reversal unless it has already broken out.
Training diagnostic result: Across all 89 closed buys, breakouts had 17 trades, +9963.50 PnL, +7.06% average return, 70.59% win rate, and 10.468 profit/loss ratio; near-resistance had 37 trades, +8070.40 PnL, +2.63% average return, 54.05% win rate, and 3.294 profit/loss ratio; room-to-resistance had 35 trades, +6230.20 PnL, +1.69% average return, 51.43% win rate, and 3.508 profit/loss ratio. For the locked mild-trend comparison, 2019 near-resistance had 4 trades, +6.25% average return, and 50.00% win rate versus 14 comparison trades at +0.35% and 42.86%; 2020 near-resistance had 5 trades at +3.56% and 40.00% versus 12 at +3.97% and 66.67%; 2021 near-resistance had 13 trades at +0.61% and 61.54% versus 14 at +0.75% and 50.00%. Every one of the 89 closed-buy snapshots was more than one ATR above prior support, leaving no near-support or support-breakdown sample.
Candidate result: Not implemented because the annual return-and-win-rate gate failed in 2019 and the win-rate gate failed in 2021.
JoinQuant training result: Not run because the local observation gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Near resistance was not a stable source of weak entries. In 2019 it identified materially stronger trades, and in 2021 it had higher win rate despite slightly lower average return. The existing BOLL/MA location filters also left no closed entry near the fixed prior support level, so support could not supply an independent actionable rule.
Can it be revisited? no as a prior-20-day/one-ATR entry filter
Conditions for revisiting: Do not search 10/30/60-day windows, smaller or larger ATR distances, pivot/fractal definitions, breakout rewards, support exceptions, Fibonacci levels, or volume-profile variants from this result. Reopening requires a genuinely new external market mechanism and explicit new research authorization.

Date: 2026-07-14
Version: `cross-v0.3.2` observation-only controlled-breakout anti-chase gate
Experiment: Classify an existing eligible cross-signal buy as a breakout only when T-1 close exceeds the highest adjusted high of the prior 20 valid bars ending T-2. Label that breakout extended when `RSI6 >= 75` or T-1 close is at least 10% above MA20; otherwise label it controlled. Consider one candidate only if both groups have at least 6 closed trades overall and at least 2 per training year, and extended breakouts have lower average return and win rate in every year.
Hypothesis: A cross-signal breakout can still be an early-strength entry, but one already extended by oscillator or MA20 distance may be avoidable chasing.
Training diagnostic result: Controlled breakouts had 15 closed trades, +9823.80 PnL, +7.83% average return, 73.33% win rate, and 11.365 profit/loss ratio. Extended breakouts had only 2 trades, +139.70 PnL, +1.32% average return, 50.00% win rate, and 2.337 profit/loss ratio. Controlled counts were 5/6/4 in 2019/2020/2021; extended counts were 1/0/1. The 2019 extended trade returned +3.33% with 100% win rate versus controlled breakouts at +9.81% and 80%; the 2021 extended trade returned -0.69% with 0% win rate versus controlled at +0.64% and 75%.
Candidate result: Not implemented because the total sample gate, every annual extended-sample gate, and the 2019 win-rate underperformance gate failed.
JoinQuant training result: Not run because the local observation gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Only two realized extended breakouts existed, they were split across two years with none in 2020, and their outcomes were mixed. This is insufficient to prove a stable anti-chase rule; rejecting them would be a decision based on two isolated trades rather than a repeated mechanism.
Can it be revisited? no as a prior-20-day breakout with `RSI6 >= 75` or MA20-distance-at-least-10% veto
Conditions for revisiting: Do not search neighboring RSI or MA20 thresholds, 10/30/60-day resistance windows, replace OR with AND, reward controlled breakouts, or alter sells. Reopening requires a genuinely new external market mechanism and explicit new research authorization.

Date: 2026-07-16
Version: `cross-v0.3.2` observation-only ETF share-flow shadow diagnostic
Experiment: For the fixed five eligible domestic ETFs, calculate `log(shares[T-1] / shares[T-6])` over exactly five share observations and compare official closed upward-cross entries after positive versus non-positive shares-outstanding flow. Block QDII, neutralize registered split crossings, and leave every score and order unchanged.
Hypothesis: Net primary-market creation may independently confirm that an oscillator cross is attracting durable demand, while flat or declining shares may identify weaker reversals.
Training diagnostic result: All 52 eligible domestic closed buys had usable observations (100% eligible coverage); 37 QDII buys were correctly excluded. Positive flow had 24 trades, +3795.30 PnL, +1.39% average return, 54.17% win rate, and 3.398 profit/loss ratio. Non-positive flow had 28 trades, +7422.60 PnL, +3.70% average return, 50.00% win rate, and 3.624 profit/loss ratio. In 2019, positive versus non-positive averaged +1.35%/42.86% win rate versus +8.10%/50.00%. In 2020 the comparison was +1.24%/55.56% versus +5.92%/57.14%. In 2021 it reversed to +1.59%/62.50% versus -0.20%/46.15%.
Candidate result: Not implemented because neither group had both higher average return and higher win rate in every training year. This experiment was observation-only and did not authorize an order-changing branch.
JoinQuant training result: Not run because the local annual-consistency gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Shares-outstanding direction is regime-dependent in this strategy path. Non-positive flow accompanied the stronger 2019-2020 trades, while positive flow was clearly better in 2021; using either sign as a universal confirmation or veto would remove valid entries in a non-trivial year.
Can it be revisited? no as a five-observation shares-outstanding sign rule
Conditions for revisiting: Do not search 3/10/20-day windows, non-zero magnitude thresholds, z-scores, fund-size or NAV interactions, QDII publication assumptions, code exceptions, or sell-side flow rules from this result. Reopening requires a new independent primary-market mechanism and explicit new research authorization.
Date: 2026-07-17
Version: `cross-v0.3.2-cross-window-1-2-3-4-training-comparison`
Experiment: Compare `cross_window=1/2/3/4` on the isolated 2019-2021 local training replay, with official window 3 as the baseline. Keep every other indicator period, score, threshold, ETF, sizing, risk, execution, adjustment, and correction rule identical. Use 2018 only as a read-only indicator warm-up.
Hypothesis: A shorter window might remove stale crosses, while a four-day window might retain early reversals that need more time to satisfy the remaining buy conditions. A non-baseline window could be considered only if it improved total and annualized return, did not worsen drawdown, Sharpe, Sortino, win rate, or profit/loss ratio, did not worsen any training year's return, and changed filled-order paths in all three years.
Local training result: Window 1 returned +50.92% annualized at 14.75%, with 13.53% max drawdown, 1.181 Sharpe, 1.697 Sortino, 57.38% win rate, 2.194 profit/loss ratio, 62 buys, and 61 sells. Window 2 returned +72.95% annualized at 20.09%, with 11.55% drawdown, 1.563 Sharpe, 2.327 Sortino, 58.33% win rate, 2.901 profit/loss ratio, 86 buys, and 84 sells. Official window 3 returned +120.61% annualized at 30.27%, with 7.47% drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio, 92 buys, and 89 sells. Window 4 returned +102.17% annualized at 26.53%, with 8.90% drawdown, 1.883 Sharpe, 2.883 Sortino, 55.56% win rate, 2.863 profit/loss ratio, 102 buys, and 99 sells.
Annual result: Window 1 returned +25.68%/+15.45%/+4.01%; window 2 returned +31.29%/+26.05%/+4.51%; window 3 returned +35.84%/+49.74%/+8.46%; window 4 returned +40.39%/+37.30%/+4.89% in 2019/2020/2021 respectively.
Path audit: Against official window 3, window 1 changed 160 filled-order days across 2019/2020/2021 (45/54/61), window 2 changed 92 (25/37/30), and window 4 changed 97 (29/32/36). All alternatives changed behavior in every training year, so their rejection is not caused by an inactive parameter.
Candidate result: Windows 1, 2, and 4 all failed the pre-registered gate. Window 1 and 2 worsened every annual return and all major risk-adjusted metrics except closed-trade win rate. Window 4 improved 2019 return but worsened 2020 and 2021, total return, drawdown, Sharpe, Sortino, win rate, and profit/loss ratio.
JoinQuant training result: Not run because no alternative passed the local training gate. JoinQuant remains the authority for absolute strategy performance; the local replay was used only for the locked structural comparison.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: One day is too strict and discards many useful crosses before the remaining confirmation conditions align. Two days still expires too many profitable signals. Four days increases activity but retains stale crosses; its isolated 2019 gain does not survive the other two training years and comes with weaker trade quality and risk-adjusted performance. The standard three-trading-day window is the only robust member of this fixed comparison.
Can it be revisited? no as a neighboring integer cross-window search
Conditions for revisiting: Retain `cross_window=3`. Do not search wider windows, per-indicator windows, fractional weighting by cross age, or combinations with threshold changes from this result. Reopening requires a new independently justified market-timing mechanism and explicit authorization.

Date: 2026-07-18
Version: `cross-v0.3.2-execution-time-0935-vs-1000-training-comparison`
Experiment: Compare fixed T-day execution at `09:35` and `10:00` on the isolated 2019-2021 local training replay. Keep the T-1 signal frame, indicators, cross window, scores, ranking, ETF pool, sizing, ATR rules, fees, slippage, and close marking identical. Use 2018 only as a read-only indicator warm-up.
Hypothesis: Waiting until 10:00 may avoid part of the opening price-discovery noise and improve fills without materially delaying daily cross signals. The candidate may pass only if aggregate return improves, no risk/trade-quality metric worsens, no training-year return worsens, and matched side-adjusted execution prices improve in every year and in both QDII and non-QDII groups.
Local training result: The `09:35` baseline returned +120.61% annualized at 30.27%, with 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio, 92 buys, and 89 sells. The `10:00` candidate returned +127.65% annualized at 31.65%, with 7.15% drawdown, 2.280 Sharpe, 3.670 Sortino, 59.09% win rate, 4.413 profit/loss ratio, 91 buys, and 88 sells.
Annual result: Baseline `09:35` returned +35.84%/+49.74%/+8.46% in 2019/2020/2021. Candidate `10:00` returned +39.92%/+50.86%/+7.85%. The candidate therefore improved 2019 and 2020 but worsened 2021.
Execution audit: There were 135 matched filled orders. Average side-adjusted execution improvement was approximately -0.012%, so the candidate was slightly worse overall. Annual averages were -0.0308%/+0.0014%/-0.0095% in 2019/2020/2021. QDII matched orders improved by +0.0307% on average, while non-QDII matched orders worsened by -0.0425%. Filled-order paths differed on 78 days, split 22/29/27 across the three years.
Candidate result: Rejected. The pre-registered gate failed because profit/loss ratio worsened, 2021 annual return worsened, matched execution did not improve in 2019 or 2021, and non-QDII execution worsened.
JoinQuant training result: Not run because the local gate failed. JoinQuant remains the authority for absolute performance, but no platform candidate is justified by a failed structural gate.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The aggregate improvement came from a materially different downstream order path rather than a stable execution-price advantage. Waiting until 10:00 helped QDII fills on average but hurt non-QDII fills, and the annual direction was inconsistent. Selecting 10:00 from total return alone would fit the strongest training years while ignoring the stated mechanism and the weaker 2021 result.
Can it be revisited? no as an intraday execution-time search
Conditions for revisiting: Retain `09:35`. Do not test 09:36, 09:40, 09:45, 10:15, 10:30, 10:35, VWAP windows, per-ETF clocks, QDII-only clocks, regime-dependent clocks, or timing interactions from this result. Reopening requires new prospective execution evidence and a separately reserved confirmation sample, not another pass over 2019-2021.

Date: 2026-07-18
Version: `cross-v0.3.2-intraday-execution-overlay-v1`
Experiment: Freeze every formal 09:35 ordinary-buy code and share amount, submit one passive limit at the raw 09:35 arrival price for six five-minute decision cycles, forbid same-minute and touch-only fills, and use the first executable minute at or after 10:05 as a market fallback. Keep T-1 signals, ranking, pool, sizing, all sells, ATR exits, fees, and the formal strategy unchanged.
Hypothesis: A short passive execution window may capture ordinary intraday mean reversion after a valid daily cross signal without delaying the strategy enough to damage fills across market regimes.
Training diagnostic result: All 92 eligible 2019-2021 ordinary buys were matched. Seventy-five filled through the passive limit path and 17 used the market fallback. Average side-adjusted execution improvement was +0.0263% overall. Annual averages were +0.0102% in 2019, -0.0078% in 2020, and +0.0673% in 2021. Non-QDII averaged +0.0412%; QDII averaged +0.0040%.
Candidate result: Not implemented because the pre-registered gate required positive average execution improvement in every training year, and 2020 was negative. The full portfolio candidate and JoinQuant candidate were therefore not permitted.
JoinQuant training result: Not run because the local counterfactual gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The small aggregate benefit was not regime-stable. The fixed waiting mechanism marginally worsened 2020 ordinary-buy execution, while the QDII benefit was only 0.40 basis point. Advancing to a portfolio backtest or selecting a nearby time after seeing these outcomes would turn one execution hypothesis into post-hoc parameter mining.
Can it be revisited? no as an arrival-price passive-limit window with a fixed morning fallback
Conditions for revisiting: Retain formal 09:35 execution. Do not search nearby arrival times, cycle lengths, cycle counts, limit offsets, fallback times, QDII/domestic exceptions, or sell-side overlays from this result. Reopening requires a new independent market-microstructure mechanism and explicit authorization.
