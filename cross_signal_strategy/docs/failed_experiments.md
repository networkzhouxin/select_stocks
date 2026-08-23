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

Date: 2026-07-27
Version: `cross-v0.3.2-same-side-reexpansion-observation`
Experiment: Test one symmetric three-point shape for RSI6/RSI12, RSI6/RSI24, DIF/DEA, K/D, and J/D. The fast/slow gap must remain on one side, contract for one observation, then expand while the fast line moves in the same direction. Collapse consecutive episodes, exclude novel events that overlap an active same-direction true cross, enter at the next 09:35 executable price, and compare 5-trading-day outcomes with true-cross events. Use 1/3/10-day outcomes only descriptively.
Hypothesis: A same-side contraction followed by renewed divergence may represent a second strengthening or weakening event that adds directional information when no mathematical cross occurs.
Training diagnostic result: The fixed scan produced 2,000 executable event episodes. Bullish novel re-expansions had 318 five-day observations with +0.47% average return and 57.23% directional win rate, below 730 active true-cross observations at +0.62% and 59.32%. Bullish novel annual results were +1.25%/67.74% in 2019, +0.64%/54.39% in 2020, and -0.36%/51.35% in 2021, versus true crosses at +0.77%/60.98%, +0.82%/59.04%, and +0.28%/58.27%. Bearish novel re-expansions had 236 observations followed by +0.73% average return and only 40.68% directional win rate, versus 700 true-down-cross observations followed by +0.53% and 41.43%. MACD bullish novel events averaged +0.28% with 53.85% win rate, below MACD true crosses at +0.92% and 62.71%. KDJ K/D and J/D results were identical because `J-D = 3*(K-D)` under the standard KDJ formula.
Candidate result: Not implemented. The fixed aggregate and annual five-day gate failed in both directions, so no local portfolio candidate and no JoinQuant candidate were permitted.
JoinQuant training result: Not run because the observation gate failed before any trading-rule change.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The shape was not an incremental substitute for a true cross. Its bullish aggregate edge was weaker, it deteriorated across the three training years and turned negative in 2021, while the bearish shape did not reliably precede declines. Selecting only the favorable 2019 bullish subset or the favorable 10-day aggregate would be post-hoc horizon or regime selection.
Can it be revisited? no as a three-point same-side contraction/re-expansion score or filter
Conditions for revisiting: Keep the formal cross definition unchanged. Do not search gap-distance thresholds, extra contraction days, slope cutoffs, indicator-specific variants, voting counts, nearby horizons, or year/ETF exceptions from this result. Reopening requires a genuinely independent market mechanism, prospective evidence, and explicit authorization.

Date: 2026-08-12
Version: `cross-v0.3.2-entry-atr-breakeven-candidate`
Experiment: On the isolated 2019-2021 local replay, preserve the official ATR trailing stop until the stored highest closing price reaches entry cost plus exactly one entry ATR. From the next decision onward, floor the stop at entry cost. Keep every signal, score, threshold, ETF, rank, size, minimum hold, execution time, fee, and official strategy file unchanged.
Hypothesis: Recovering initial risk after a position has earned one entry ATR may prevent a material subset of profitable trades from returning to a loss without clipping the larger trends retained by the official ATR stop.
Training diagnostic result: Of 89 closed baseline trades, 67 reached at least one entry ATR of closing-price profit and 19 of those finished non-profitable, a 28.36% round-trip rate. The rates were 10.53%/33.33%/37.50% in 2019/2020/2021. These are ex-post diagnostic labels and were not used to select another threshold.
Candidate result: The baseline returned +120.61% annualized at +30.27%, with 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, and 4.440 profit/loss ratio. The candidate returned +114.28% annualized at +29.01%, with 7.57% drawdown, 2.096 Sharpe, 3.292 Sortino, 48.94% win rate, and 4.148 profit/loss ratio. Candidate annual returns were +36.55%/+45.91%/+7.55%, versus baseline +35.84%/+49.74%/+8.46%. Filled-order decisions changed on 47 days, split 2/27/18 across 2019/2020/2021.
JoinQuant training result: Not run because the local strict gate failed.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The fixed cost floor reduced some round trips but caused more early exits and subsequent portfolio-path changes. Its small 2019 gain did not survive 2020 or 2021, and every aggregate risk/trade-quality gate worsened. The mechanism therefore trades away too much trend persistence for cosmetic loss avoidance.
Can it be revisited? no as an entry-ATR-activated break-even or profit-floor rule
Conditions for revisiting: Keep the official no-profit-floor ATR exit. Do not search 0.5/1.5/2 ATR activations, positive or negative floor offsets, staged tightening, per-ETF exceptions, or year/regime interactions from this result. Reopening requires a genuinely independent exit mechanism, prospective evidence, and explicit authorization.

Date: 2026-08-13
Version: `cross-v0.3.2-macd-free-kdj-exit-candidate`
Experiment: On the isolated 2019-2021 local replay, remove MACD cross points from both buy and sell scores while retaining MACD as an observation field. Replace the ordinary signal-sell decision with the standard recent K/D death cross alone, without MA20, BOLL-mid, falling-MA10, downside-continuation, sell-score, or ADX protection. Preserve the official five-trading-day minimum hold, ATR stop, ETF pool, cross window, remaining indicators, thresholds, sizing, fees, and 09:35 execution. Use 2018 only as read-only indicator warm-up.
Hypothesis: RSI/KDJ may identify reversals earlier than MACD, while selling directly on a K/D death cross after five trading days may preserve more accrued profit before the slower MACD and price-structure confirmations arrive.
Training diagnostic result: The official local replay returned +120.61% annualized at +30.27%, with 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% closed-trade win rate, 4.440 profit/loss ratio, 92 buys, and 89 sells. The candidate returned +41.87% annualized at +12.40%, with 8.65% drawdown, 1.276 Sharpe, 1.924 Sortino, 55.95% win rate, 1.664 profit/loss ratio, 170 buys, and 168 sells. Candidate annual returns were +10.21%/+33.63%/-3.67% versus baseline +35.84%/+49.74%/+8.46% in 2019/2020/2021. Filled-order decisions changed on 256 days, split 79/88/89 across the three years.
Candidate result: Rejected by every pre-registered aggregate and annual gate. Total and annualized return, drawdown, Sharpe, Sortino, win rate, profit/loss ratio, and all three annual returns worsened.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: Removing MACD's score contribution made entry qualification less coherent, while a K/D death cross alone turned ordinary exits into a much faster oscillator response. The combined rule nearly doubled trading activity, cut the profit/loss ratio by more than half, and exited too many positions before trend profits could develop. Because this was a locked combined hypothesis, the result does not identify how much damage came from each component separately; splitting the losing combination after observing its result would be a new post-hoc search.
Can it be revisited? no as MACD-free buy scoring, KDJ-only ordinary selling, or a nearby decomposition of this combined candidate
Conditions for revisiting: Retain MACD's official score contribution and the official sell-score, price-confirmation, and ADX-protection structure. Do not search K/D versus J/D, nearby hold days, KDJ thresholds, sell-score hybrids, selected ETF/year exceptions, or MACD point redistribution from this result. Reopening requires a genuinely independent mechanism, prospective evidence, and explicit new authorization.

Date: 2026-08-13
Version: `cross-v0.3.2-macd-fast-exit-candidate`
Experiment: Keep all official buy and exit rules, but after the five-trading-day minimum hold add an OR exit when the recent-window MACD DIF/DEA death-cross flag is true, bypassing sell-score, price-structure, and ADX protection.
Hypothesis: MACD death cross may mark the start of a decline early enough to preserve open profit before slower price-structure confirmations trigger.
Training diagnostic result: The candidate changed 156 filled-order days across 2019/2020/2021, split 51/43/62. The baseline returned +120.61% annualized at +30.27%, with 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio, 92 buys, and 89 sells. The candidate returned +81.75% annualized at +22.10%, with 8.53% drawdown, 1.732 Sharpe, 2.641 Sortino, 55.20% win rate, 2.576 profit/loss ratio, 128 buys, and 125 sells. Baseline annual returns were +35.84%/+49.74%/+8.46%, while candidate annual returns were +23.68%/+39.88%/+5.05% in 2019/2020/2021.
Candidate result: Rejected by the pre-registered gate. Every aggregate quality/risk metric and every annual return worsened.
JoinQuant training result: Not run because the local gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: A recent-window MACD death cross by itself is too noisy as a mandatory exit. It exits recoverable pullbacks, increases portfolio recycling, and truncates profitable trends; the official price-structure, sell-score, and ADX filters prevent these premature exits.
Can it be revisited? no as a MACD-only fast-exit OR channel
Conditions for revisiting: Do not search MACD periods, cross windows, hold days, delayed confirmations, ETF/year exceptions, profit conditions, or threshold combinations from this result. Reopening requires a genuinely new externally justified exit mechanism and explicit authorization.

Date: 2026-08-14
Version: `cross-v0.3.2-kdj-only-exit-candidate`
Experiment: Keep the complete official buy path, including MACD cross scoring, unchanged. After the frozen five-trading-day minimum hold, replace the ordinary signal-sell path with the recent KDJ K/D death cross alone, bypassing sell-score, price-structure confirmation, and ADX protection. Preserve the official ATR stop, ETF pool, cross window, sizing, fees, and 09:35 execution.
Hypothesis: K/D death cross may preserve accrued profit earlier than the slower multi-condition ordinary sell path while the unchanged official buy logic maintains entry quality.
Training diagnostic result: The candidate changed 256 filled-order days across 2019/2020/2021, split 80/84/92. The baseline returned +120.61% annualized at +30.27%, with 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio, 92 buys, and 89 sells. The candidate returned +42.64% annualized at +12.60%, with 8.61% drawdown, 1.309 Sharpe, 1.969 Sortino, 54.44% win rate, 1.691 profit/loss ratio, 170 buys, and 169 sells. Baseline annual returns were +35.84%/+49.74%/+8.46%, while candidate annual returns were +10.06%/+30.30%/-0.53% in 2019/2020/2021.
Candidate result: Rejected by the pre-registered gate. Every aggregate quality/risk metric and every annual return worsened.
JoinQuant training result: Not run because the local gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The K/D oscillator is faster but too noisy as the sole ordinary exit. It repeatedly treats recoverable pullbacks as trend endings, nearly doubles portfolio turnover, truncates profitable trends, and leaves the strategy worse in all three training regimes. Preserving MACD on the buy side does not repair the premature-exit mechanism.
Can it be revisited? no as a K/D-death-cross-only ordinary exit
Conditions for revisiting: Retain the official sell-score, price-structure, and ADX protections. Do not search K/D versus J/D, nearby hold days, cross windows, KDJ periods, profit conditions, delayed confirmations, ETF/year exceptions, or threshold hybrids from this result. Reopening requires a genuinely independent exit mechanism and explicit authorization.

Date: 2026-08-16
Version: `cross-v0.3.3-profit-tier-candidate`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, tighten the trailing ATR multiplier by current profit: ×0.8 above 5% profit and ×0.6 above 15% profit (frozen multi-factor V2.6 mechanism). Change only `calc_stop_price` (new `profit_pct` argument) and its two call sites; keep signals, scoring, ranking, sizing, minimum hold, ETF pool, ATR-stress rule, stop floor/cap, fees, and 09:35 execution unchanged.
Hypothesis: Tiered multiplier tightening for profitable high-volatility holdings reduces profit giveback without damaging trend-following, improving total return while not worsening maximum drawdown.
Training diagnostic result: Step 0 binding observation on the official replay found 36 binding stop-check events (profit > 5% and unfloored stop above the 5% floor; 4/24/8 in 2019/2020/2021, ETFs 159915/159928/513100/518880). The baseline stop distance on binding days was 5.02%-7.74% versus 5.00%-6.19% tightened, but zero days had a 09:35 price inside the gap between the tightened and baseline stops.
Candidate result: Local A/B with the isolated candidate file changed 0 filled orders: +125.00% total return, 6.03% maximum drawdown, 2.262 Sharpe, 3.581 Sortino, annual +35.84%/+52.68%/+8.49%, 92 buys/89 sells, 25 ATR stops — all identical to the baseline. The pre-registered "at least 3 filled orders change" gate failed; the candidate is an exact no-op on this path.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The cross-signal stop uses a frozen entry ATR and a 5% floor, and the median entry ATR is about 1.4% of price, so the floor dominates most stops. On the 36 days where tightening could bind, the tightened stop only moved from 5.02%-7.74% to 5.00%-6.19% below the peak, and no 09:35 price ever fell into that gap. The multi-factor V2.6 mechanism does not transfer to this framework because its giveback protection there operates through a different stop construction and profit-floor stack.
Can it be revisited? no as profit-tiered ATR multiplier tightening in cross-signal
Conditions for revisiting: Keep the official ATR stop and its 5%/15% clamps. Do not search other tier thresholds, multiplier factors, peak-profit measurements, profit floors, per-ETF overrides, or interactions from this result. Reopening requires a genuinely independent risk mechanism and explicit authorization (the separate pre-registered gold-specific stop direction remains governed by its own budget entry when authorized).

Date: 2026-08-16
Version: `cross-v0.3.3-gold-stop-candidate`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, apply gold-only stop parameters (`518880`: `stop_floor` 0.03 and `trailing_atr_mult` 2.0 instead of 5%/2.5×, copied from the multi-factor V2.8 walk-forward adoption). Change only `calc_stop_price` (new `code` argument) and its two call sites; keep signals, scoring, ranking, sizing, minimum hold, ETF pool, ATR-stress rule, other ETFs' stops, fees, and 09:35 execution unchanged.
Hypothesis: Gold is the pool's only mean-reverting asset, so a tighter stop exits failed bounces earlier and preserves more of its reversal profits without worsening drawdown or any annual return.
Training diagnostic result: Step 0 binding observation found 223 binding gold stop-check days (73/92/58 in 2019/2020/2021) and 6 same-day extra-trigger events (2019-07-01/02, 2019-09-09, 2021-08-09/10, 2021-11-24), passing the 10/3 gates.
Candidate result: The local A/B changed 162 filled-order positions (92→94 buys, 89→91 sells) and failed the gates: total return +125.00%→+120.96%, max drawdown 6.03%→6.08%, Sharpe 2.262→2.210, Sortino 3.581→3.492, annual 2019 +35.84%→+34.34%, 2020 +52.68%→+53.25%, 2021 +8.49%→+7.33%. Gold ATR stops rose from 2 to 5. Per-trade attribution: the first extra stop (2019-07-01, at +4.7% profit) clipped a winner that the baseline exited at +9.0% on 2019-08-02, and the clipped exit cascaded into the 162-fill path divergence; the 2021-08-09 stop exited two days earlier than the baseline ATR stop for +0.3pp of avoided loss; the 2021-11-24 event only swapped the sell reason at the same price.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: In this framework gold enters on reversal crosses and its winning trades tolerate pullbacks of 3-4% below the peak while the bounce develops (the 2019 winner dipped 3.4% below peak mid-hold before reaching +9%). The 3% floor exits those pullbacks, clipping winners; the multi-factor V2.8 result does not transfer because that framework enters gold through a different rotation path with different exit semantics.
Can it be revisited? no as gold-specific floor/multiplier tightening in cross-signal
Conditions for revisiting: Keep the official uniform 5%/2.5× stop. Do not search other gold floor or multiplier values, and do not extend per-ETF stops to soymeal, QDII, or A-share ETFs (all failed in multi-factor). Reopening requires a genuinely independent mechanism and explicit authorization.

Date: 2026-08-16
Version: `cross-v0.3.3-giveback-observation`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, measure a read-only trade-level counterfactual for a fixed profit-giveback direct exit: at the daily 09:35 stop check, if peak closing-price profit reaches at least 5% and current 09:35 profit falls to peak profit minus 3 percentage points, the rule would exit immediately (same-day buys exempt). Compare the first rule-exit price with the official exit price for the same entry and shares. No order was changed.
Hypothesis: Exiting once profit gives back a fixed amount from its peak locks in more of the small-to-medium winners without damaging the large trends the official stop already protects.
Training diagnostic result: The rule fired 79 times across 21 affected closed trades. The per-share delta was negative overall (-0.352) and negative in 2019 (-0.380) and 2020 (-0.101), positive only in 2021 (+0.129). The two dominant clips were 2019-02-11 159928 (rule exit 2.245 versus official exit 2.666, a +18.8% winner cut by 0.421/share) and 2020-04-17 513050 (rule exit 1.523 versus official exit 1.927, a +34% winner cut by 0.404/share). The saved amounts on other trades were small (+0.004 to +0.157 per share).
Candidate result: Not created. The Step 0 gates (at least 5 affected trades, positive total delta, positive delta in every year) failed on the total and annual deltas.
JoinQuant training result: Not run because the observation gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: This framework's payoff comes from a small number of large trend winners, and those winners routinely give back more than 3 percentage points of profit mid-hold before resuming (159928 +18.8% and 513050 +34% both dipped through the giveback band). A profit-giveback exit therefore clips exactly the trades that pay for everything, while only salvaging small amounts on the losers. This is the same reason the break-even floor and the gold stop failed: trend-following needs the giveback to happen.
Can it be revisited? no as a profit-giveback direct exit
Conditions for revisiting: Keep the official ATR stop as the only profit protection. Do not search other activation levels, giveback thresholds, relative fractions, peak-profit floors, tiered variants, per-ETF exceptions, or sell-side hybrids from this result. Reopening requires a genuinely independent exit mechanism and explicit authorization.

Date: 2026-08-16
Version: `cross-v0.3.3-high-anchor-candidate`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, replace the trailing-high anchor with the intraday-high anchor: track the maximum completed daily HIGH from the buy date onward and keep the stop formula, 2.5 multiplier, 5% floor, 15% cap, and frozen entry ATR unchanged. Change only the anchor update in the after-close path.
Hypothesis: Anchoring on intraday highs makes the stop more responsive to actual price extremes without damaging trend-following.
Training diagnostic result: Step 0 binding observation found 1604 binding stop-check days (499/580/525 in 2019/2020/2021) and 38 same-day extra-trigger events across all nine ETFs, passing the 10/3 gates.
Candidate result: The local A/B changed 175 filled-order positions (92→94 buys, 89→91 sells, ATR stops 25→29) and failed the gates: total return +125.00%→+119.40%, max drawdown 6.03%→6.06%, annual 2019 +35.84%→+30.55% (2020 and 2021 slightly improved). Per-trade attribution across the nine changed exits: seven small saves (+0.001 to +0.090 per share) and two clips, dominated by 2019-02-11 159928 (candidate exit 2019-02-26 at 2.232 versus official 2019-04-12 at 2.666, -0.434 per share on an +18.8% winner); total per-share delta -0.228.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The peak-day upper wick of a major winner raises the high anchor enough to trigger the stop inside the winner's normal pullback band. The 5% close-anchored band exists precisely to absorb those intraday spikes; replacing it with the high anchor turns the noise the close anchor was designed to filter back into stop triggers. This confirms the original close-anchor design rule with data.
Can it be revisited? no as an intraday-high trailing anchor
Conditions for revisiting: Keep the official close-anchored trailing high. Do not search anchor blends (max(close, high×f)), multiplier re-calibrations, floor/cap changes, per-ETF anchor exceptions, or hybrid anchors from this result. Reopening requires a genuinely independent mechanism and explicit authorization.

Date: 2026-08-21
Version: `cross-v0.3.3-dual-timepoint-1445-candidate`
Experiment: On the isolated 2019-2021 local replay, preserve the complete official 09:35 buy/sell path and add one full 14:45 buy/sell pass. Recompute the unchanged RSI, KDJ, MACD, ADX, BOLL, MA, and ATR rules from completed T-day one-minute bars through 14:44; execute at the 14:45 minute open; share broker, portfolio, minimum-hold, sold-today, entry-ATR, trailing-close, score, ranking, sizing, and risk state across both batches. Keep every indicator, parameter, score threshold, ETF, fee, and official 09:35 rule unchanged.
Hypothesis: A second causal decision near the close may enter reversals earlier and exit deteriorating positions before the next morning, increasing closed-trade accuracy and reducing positive-to-negative profit giveback without sacrificing more than 20% of baseline return or worsening drawdown.
Training diagnostic result: The fixed baseline returned +125.0025%, with 6.0316% maximum drawdown, 56.18% closed-trade win rate, 4.440 historical baseline profit/loss ratio, 92 buys, 89 sells, 31 positive-to-negative round trips, and a maximum losing streak of 5. The candidate returned +84.9970%, with 7.4919% drawdown, 47.66% win rate, 2.8131 profit/loss ratio, 109 buys, 107 sells, 40 positive-to-negative round trips, and the same maximum losing streak of 5. Annual win rates fell from 56.00%/58.62%/54.29% to 53.57%/48.65%/42.86% in 2019/2020/2021. Under doubled commission and slippage, return fell from +112.7772% to +73.1887% and drawdown rose from 6.2540% to 8.2763%. Usable 14:45 code-date coverage was 1765/2134/2132, with 431/53/55 missing observations fully disclosed.
Candidate result: Rejected by nine frozen gates: return retained less than 80% of baseline; nominal and doubled-friction drawdowns worsened; profit/loss ratio fell below 3.0; overall and all three annual win rates worsened; positive-to-negative round trips increased instead of falling by at least three; improvement was confined to only 512100; and doubled-friction return retained less than 80% of its baseline. Buy/sell counts stayed within the 30% ceiling and the maximum losing streak did not worsen, but those two passes cannot override the failed quality gates.
Engineering audit: The first CLI attempt stopped before producing any report because 512100 minute rows on 2019-01-02 had missing `prev_close`, which the strict causal frame correctly rejected. A failing regression test was added and the adapter was corrected to record an invalid code-date as missing coverage while leaving the frame validator strict. A full read-only scan then proved nonzero usable coverage in every year; the same fixed candidate was run to completion without changing any strategy rule or parameter. The aborted implementation run is not a second candidate variant and produced no performance result.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant or PTrade candidate was generated; formal platform strategies remain unchanged.
Validation result: Not run. Reserved validation, pressure, recent-market, full-period, and 2026 price data were not inspected.
Why it failed: The extra same-day pass reacts to partial-day oscillator and trend changes that are not reliable enough to distinguish durable reversals from normal intraday noise. It increased portfolio recycling and bought/sold more frequently, but accuracy fell by 8.52 percentage points, round trips from positive excursion to realized loss increased by nine, and both nominal and stressed drawdowns worsened. The additional timing opportunity therefore amplifies noise rather than solving late entry or delayed exit.
Can it be revisited? no as a second full intraday signal pass at 14:45 or a nearby time
Conditions for revisiting: Retain the official single 09:35 decision path. Do not search 14:30/14:40/14:50, other minute cutoffs, ETF-specific afternoon passes, partial indicator subsets, afternoon-only buy/sell variants, threshold offsets, or cooldown/hold interactions from this result. Reopening requires a genuinely independent mechanism, prospective evidence, and explicit new authorization.

Date: 2026-08-16
Version: `cross-v0.3.3-profit-gated-matrix-observation`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, measure a read-only trade-level counterfactual for a fixed 4×3 matrix of direct-sell channels: sell-score thresholds 32/35/38/40 crossed with profit bands 2-4%/3-5%/4-6%. A channel fires when the T-1 sell score reaches the threshold AND the current 09:35 profit falls inside the band, bypassing the price-structure confirmation while keeping the 5-day minimum hold and the ADX strong-uptrend exemption. No order was changed.
Hypothesis: A strong reversal signal inside a small profit band can exit before the structure confirmation arrives while capping the cost of selling too early, improving total return without worsening drawdown or any annual return.
Training diagnostic result: The 38/40 score thresholds never fired (0 events): sell scores that high arrive only after profit has already left the 2-6% band. The 32/35 thresholds fired 19 and 18 times respectively, but every variant's total per-share delta was negative (A1 -0.051, A2/B2 -0.423, A3/B3 -0.313, B1 -0.077), driven by the 513050 +34% winner whose mid-hold pullback satisfies high-score-plus-small-profit and would be exited at about 1.523 versus the official 1.927. All 12 variants failed the gates (at least 5 affected trades, positive total delta, positive delta in every year).
Candidate result: Not created. The pre-registered selection rule found no passing variant.
JoinQuant training result: Not run because the observation gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The profit band does not distinguish a small winner that will keep winning from a small winner that will fail; the framework's large winners pass through the 2-6% profit zone multiple times with elevated sell scores on their pullbacks. Bypassing the price-structure confirmation inside that zone therefore clips the same winners that pay for everything, the same structural cause as the profit-giveback exit, the break-even floor, and the gold stop.
Can it be revisited? no as a profit-gated direct-sell channel
Conditions for revisiting: Keep the official sell channel (score ≥ 30 with price-structure confirmation) unchanged. Do not search nearby score thresholds, band edges, relative fractions, exemptions, or hybrid sell channels from this result. Reopening requires a genuinely independent exit mechanism and explicit authorization.

Date: 2026-08-17
Version: `cross-v0.3.3-age2-half-decay-candidate`
Experiment: On the isolated 2019-2021 local replay of the official `cross-v0.3.3` mainline, keep age-0 and age-1 bullish RSI12/RSI24/MACD/KDJ-K/KDJ-J cross contributions at their full official weights and multiply only contributing age-2 bullish cross weights by exactly 0.5. The sell side, three-day cross window, thresholds, filters, ranking, sizing, ETF pool, ATR rules, costs, and 09:35 execution remained unchanged.
Hypothesis: A bullish cross that first occurred two completed bars ago is less timely than an age-0 or age-1 cross, so halving only its buy-side contribution should avoid late entries without damaging fresher reversal entries.
Training diagnostic result: The candidate changed 64 filled-order days, with changes in every training year (2019: 15, 2020: 27, 2021: 22), so it materially changed the execution path and passed the path-coverage gate.
Candidate result: The local A/B failed the strict gate: total return +125.00% to +87.35%, annualized return 31.13% to 23.35%, and maximum drawdown 6.03% to 8.79%. Sharpe fell from 2.262 to 1.832, Sortino from 3.581 to 2.801, and profit/loss ratio from 4.878 to 3.405. Win rate rose from 56.18% to 57.47%, but all annual returns worsened: 2019 +35.84% to +33.26%, 2020 +52.68% to +32.07%, and 2021 +8.49% to +6.46%. Buys/sells changed from 92/89 to 90/87.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant candidate was generated and the official JoinQuant/PTrade strategies remain unchanged.
Validation result: Not run. Reserved periods and live cases were not inspected or used.
Why it failed: The uniform age-2 discount removed or reordered useful buy-side contributions in all three training years. The evidence proves that age-2 contributions cannot be treated as uniformly stale in this framework; the small win-rate increase came with much weaker payoff size, risk-adjusted return, drawdown, and annual consistency. The aggregate A/B does not prove one single trade-level cause, so no narrower causal claim is made and no post-hoc subgroup is selected.
Can it be revisited? no as bullish-cross age weighting in the current cross-v0.3.3 framework
Conditions for revisiting: Keep the official full weight for ages 0/1/2. Do not search other decay coefficients, per-indicator age weights, age-specific thresholds, ETF/year exceptions, wider windows, or compensating entry/exit rules from this result. Reopening requires a genuinely independent market-structure reason or a proven mainline strategy change plus explicit authorization.

Date: 2026-08-22
Version: `cross-v0.3.3-fresh-unextended-entry-candidate`
Experiment: Append one fixed fresh-entry queue after the complete official score-at-least-60 primary queue. Admit only score 50-59 observations with reversal score at least 35, every contributing bullish cross age 0/1, and T-1 extension no more than one ATR from the earliest contributing cross close. Keep all official filters, ranking priority, sells, risk rules, pool, costs, and 09:35 execution unchanged.
Hypothesis: A narrowly defined fresh, unextended 50-59 reversal may enter valid reversals earlier without displacing stronger official entries, improving accuracy and reducing late-entry giveback.
Training diagnostic result: The official JoinQuant candidate produced 105 filled buys and 102 filled sells. The fresh channel had 19 closed trades plus one open at period end; closed results were 4 winners and 15 losers (21.05% win rate), net PnL CNY +1,853.80. Two winners, 159928 and 513050, supplied about 88.5% of fresh-channel gross profit. Primary-channel closed trades were 46 wins and 37 losses. Candidate calendar returns remained positive at +25.28%/+59.54%/+5.64% in 2019/2020/2021.
Candidate result: Rejected by the pre-registered official gates. Against the official baseline, total return fell from +129.25% to +111.14%, maximum drawdown worsened from 6.28% to 6.29%, closed-trade win rate fell from 55.8% to 49.0%, profit/loss ratio fell from 5.297 to 3.904, and positive-to-negative round trips increased from 31 to 39. Only the profit/loss-ratio floor and positive annual-return condition passed.
JoinQuant training result: Completed once on the frozen 2019-2021 training window with CNY 20,000 and daily frequency. Build `20260822.1-candidate`, business fingerprint `25783cc30ba4`, source-log SHA-256 `D2E42BCB0293A692B4EE2ED402C44713E31D54554FEAFAB08F5338CD843C90C1`.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The new buy channel had very low directional accuracy. A few large winners hid 15 losing entries and did not compensate for the resulting recycling and eight additional positive-to-negative round trips. Because the candidate changed only the appended buy path and left all sells unchanged, this result attributes the degradation to broadening entry eligibility, not to delayed exits.
Can it be revisited? no as a 50-59 fresh-unextended fast-entry channel
Conditions for revisiting: Keep the official score-at-least-60 entry threshold. Do not search nearby score bands, reversal thresholds, cross-age limits, ATR-extension limits, ETF/year exceptions, queue order, or sell-side compensation from this result. Reopening requires a genuinely independent entry mechanism and explicit authorization.

Date: 2026-08-22
Version: `cross-v0.3.3-late-macd-boll-upper-observation`
Experiment: On official JoinQuant `cross-v0.3.3` filled buys only, count the exact T-1 shape where MACD bullish-cross age is 0, an active RSI bullish cross is age 1/2, an active KDJ bullish cross is age 1/2, and close is at or above the upper Bollinger band. Permit one new-buy veto candidate only if at least 3 matches span at least 2 training years. Keep every strategy rule unchanged during Step 0.
Hypothesis: When RSI and KDJ have already crossed one or two completed bars earlier but MACD crosses only after price reaches the BOLL upper band, MACD confirmation is late and the resulting buy is an avoidable overheat entry.
Training diagnostic result: The official baseline log contained 98 filled buys. Only 2 matched: 513100 bought 2019-03-15 from the 2019-03-14 T-1 snapshot (close 2.535, BOLL upper 2.524726, RSI age 2, KDJ age 1, MACD age 0) and 159928 bought 2019-12-31 from the 2019-12-30 snapshot (close 3.000, BOLL upper 2.997819, RSI age 1, KDJ age 1, MACD age 0). Both matches were in 2019.
Candidate result: Not created. The pre-registered Step 0 gate required at least 3 events across at least 2 training years; observed counts were 2 events across 1 year.
JoinQuant training result: Observation used the official filled-buy path from `cross-v0.3.3` build `20260820.1`, business fingerprint `77e44d93d255`; no order-changing JoinQuant candidate was run.
Validation result: Not run. Reserved periods were not inspected.
Why it failed: The exact mechanism is too sparse and concentrated in one training year to justify a general veto. Blocking buys from two isolated cases would fit anecdotes rather than establish a repeated cross-year failure mode.
Can it be revisited? no as this late-MACD/BOLL-upper combination in the current framework
Conditions for revisiting: Keep the formal buy path unchanged. Do not relax BOLL upper to near-upper, widen RSI/KDJ ages, drop either prior-cross requirement, use per-ETF/year exceptions, change MACD periods, add an overheat threshold, or compensate through sells. Reopening requires a genuinely independent mechanism and explicit authorization.

Date: 2026-08-22
Version: `cross-v0.3.3-late-macd-boll-filter-candidate`
Experiment: Veto only a new buy whose T-1 snapshot has a current MACD bullish cross (age 0), an earlier active RSI bullish cross (age 1/2), an earlier active KDJ bullish cross (age 1/2), and close at or above the BOLL upper band. Keep all other buy filters, every sell, position sizing, pool, costs, and 09:35 execution unchanged.
Hypothesis: Removing the two known late-confirmation upper-band entries may improve entry accuracy without sacrificing the strategy's return distribution.
Training diagnostic result: The official JoinQuant run emitted exactly 2 veto events, both in 2019: 513100 on 2019-03-15 and 159928 on 2019-12-31. The candidate still made 98 filled buys and 95 filled sells because later or substitute candidates occupied the released slots. The 513100 veto delayed its next buy to 2019-04-02 at 2.619 rather than eliminating the late path; the 159928 veto promoted another candidate on the same day.
Candidate result: Total return fell from +129.25% to +124.09%, annualized return from +32.86% to +31.83%, profit/loss ratio from 5.297 to 5.208, Sharpe from 2.275 to 2.185, Sortino from 3.245 to 3.028, and information ratio from 0.839 to 0.790. Win rate remained exactly 55.8% and maximum drawdown remained 6.28%. The frozen requirement that win rate improve therefore failed.
JoinQuant training result: Completed once on the frozen 2019-2021 window with CNY 20,000 and daily frequency. Build `20260822.2-candidate`, business fingerprint `a46fff884685`, source-log SHA-256 `AF7E254A7F6C21F9AFA07778375A1922E948E2A6F9A00CBD17494C9989DB8A4E`.
Validation result: Not run. Reserved periods and live cases were not inspected.
Why it failed: A sparse veto does not guarantee removal of the unwanted exposure. The normal queue can buy the same ETF later at a worse price or promote another ETF, so both vetoes changed portfolio sequencing without improving directional accuracy. The exact rule reduced payoff while leaving win rate unchanged.
Can it be revisited? no as a standalone late-MACD/BOLL-upper veto
Conditions for revisiting: Retain this file only for controlled A/B comparison with the separately user-authorized stacked early-entry candidate. Do not search nearby BOLL distance, cross ages, RSI cutoffs, ETF exceptions, or sell compensation from this failed result.

Date: 2026-08-22
Version: `cross-v0.3.3-opportunity-replacement-candidate` (local research only)
Experiment: With all three slots occupied and all three holdings past the official five-trading-day minimum hold, permit one opportunity-cost replacement only when a new ETF passes the complete official buy filter and an existing holding has sell score at least 30 but remains held because ordinary price confirmation or ADX protection blocks the sell. Sell the highest sell-score holding, break ties with lower current buy score, and buy the highest-ranked official candidate. Keep all indicators, thresholds, normal sells, ATR stops, sizing, pool, costs, and 09:35/T-1 timing unchanged.
Hypothesis: A qualified new entry can supply the missing opportunity-cost evidence for recycling a protected sell-risk holding, improving closed-trade accuracy without converting healthy sub-30 holdings into a rotation strategy.
Training diagnostic result: The isolated 2019-2021 local A/B completed 19 opportunity-replacement sells and 19 matching buys and changed 168 trading days. Baseline versus candidate was: total return +125.00% to +89.87%, annualized return 31.13% to 23.90%, maximum drawdown 6.03% to 6.18%, Sharpe 2.262 to 1.947, Sortino 3.581 to 3.042, win rate 56.18% to 55.05%, profit/loss ratio 4.878 to 3.244, and buys/sells 92/89 to 112/109. Annual returns worsened in every year: 2019 +35.84% to +32.68%, 2020 +52.68% to +32.97%, and 2021 +8.49% to +7.62%.
Candidate result: Rejected by the local gate. The target accuracy metric worsened and every payoff, downside, risk-adjusted, and annual-consistency metric also worsened.
JoinQuant training result: Not run because the local gate failed. No standalone JoinQuant or PTrade candidate was generated; both formal strategies remain unchanged.
Validation result: Not run. Reserved validation, pressure, recent-market, live-outcome, and full-period data were not inspected.
Why it failed: A new reversal entry is not reliable evidence that a protected existing trend has lower forward value. The conditional rule avoided selling healthy sub-30 holdings, but the 19 replacements still increased recycling and path dependence enough to reduce accuracy and clip payoff quality.
Can it be revisited? no as score-based full-capacity opportunity replacement in the current framework
Conditions for revisiting: Do not search nearby sell thresholds, stronger buy thresholds, score spreads, ETF/year exceptions, alternate weakest-holding rankings, hold periods, or cooldowns from this result. Reopening requires a genuinely independent cross-sectional expected-return measure, prospective evidence, and explicit authorization.

Date: 2026-08-23
Version: `cross-v0.3.3-kdj-extreme-zone-score-candidate` (local research only)
Experiment: On the isolated 2019-2021 local replay, add exactly 5 points to the unified buy score whenever T-1 KDJ K is at or below 20 unless the existing `downside_continuation` state is active, and add exactly 5 points to the unified sell score whenever T-1 K is at or above 80. Require no KDJ cross, keep the formal buy/sell thresholds at 60/30, and keep every price, ADX, hold, ATR, ranking, sizing, pool, cost, and 09:35 rule unchanged. Count the points identically regardless of source; no hidden base-score branch was introduced.
Hypothesis: A small extreme-zone contribution can admit an otherwise well-supported low-position entry before the lagging KDJ gold cross and can raise early sell awareness in an overbought position, while remaining too small to act alone.
Training diagnostic result: Across 6,111 valid score snapshots, 93 received the oversold buy bonus and 1,382 received the overbought sell bonus. Zero buy observations crossed from below 60 to at least 60. Eleven sell observations crossed from below 30 to at least 30 (`159915=2`, `159928=3`, `512100=3`, `513100=1`, `518880=2`), but four lacked official price confirmation and all eleven were protected by the official strong-ADX rule. The completed local A/B therefore changed zero filled-order days.
Candidate result: Baseline and candidate were identical: +125.00% total return, +31.13% annualized, 6.03% maximum drawdown, 2.262 Sharpe, 3.581 Sortino, 56.18% win rate, 4.878 profit/loss ratio, and 92/89 buys/sells. Annual returns were identical at +35.84%/+52.68%/+8.49%. Under doubled commission, minimum commission, and slippage, both arms again matched at +108.15% return, 6.39% drawdown, 51.69% win rate, and 3.966 profit/loss ratio.
JoinQuant training result: Not run because the local materiality and accuracy gates failed. No JoinQuant candidate was generated.
Validation result: Not run. Reserved periods, 2026 prices, and live outcomes were not inspected or used.
Why it failed: Five points frequently labeled extreme states but never changed a buy threshold decision, while the eleven sell threshold crossings were all neutralized by the unchanged price/ADX protections. The candidate was therefore an exact no-op on the historical ordinary-order path and could not improve accuracy. Historical point-in-time IOPV is absent from the approved training data, so the separate PTrade live override (`sell_score >= 30` and premium at least 8%) cannot be backtested from this result and was not activated in formal code.
Can it be revisited? no as KDJ K 20/80 extreme-zone score points in the current framework
Conditions for revisiting: Keep the formal scores unchanged. Do not search neighboring K thresholds, 3/4/6/8-point bonuses, RSI/J/D duplicates, ETF/year exceptions, or removal of price/ADX protections. A live IOPV interaction requires independently collected point-in-time evidence and separate explicit authorization; this no-op local result is not permission to deploy it.

Date: 2026-08-23
Version: `cross-v0.3.3-kdj-tiered-persistence-candidate` (local research only)
Experiment: Add a unified KDJ state score without requiring a cross. K<=20 contributes 10 buy points, 20<K<=30 contributes 5 buy points, 70<=K<80 contributes 5 sell points, and K>=80 contributes 10 sell points. Retain the state for the current and prior two decision sessions, take the maximum same-direction tier without accumulation, and let the most recent direction replace an older opposite direction. The current official `downside_continuation` state blocks any retained buy bonus. Keep the formal 60/30 thresholds and every price, ADX, hold, ATR, ranking, sizing, pool, friction, and 09:35/T-1 rule unchanged.
Hypothesis: A score sized to the existing MACD/BOLL components, with a short non-cumulative memory, can affect decisions that the earlier five-point no-op could not while still requiring substantial confirmation from the formal score.
Training diagnostic result: The one fixed 2019-2021 local A/B changed 22 filled-order days: 20 in 2019, 2 in 2020, and 0 in 2021. Buys/sells increased from 92/89 to 95/92. The lack of any 2021 order change failed the pre-registered annual materiality requirement.
Candidate result: Total return fell from +125.00% to +118.33%, annualized return from 31.13% to 29.82%, maximum drawdown rose from 6.03% to 6.06%, Sharpe fell from 2.262 to 2.142, Sortino from 3.581 to 3.324, win rate from 56.18% to 55.43%, and profit/loss ratio from 4.878 to 4.384. Annual returns were +35.84%/+52.68%/+8.49% for the baseline and +32.61%/+51.89%/+8.39% for the candidate. Under doubled commission, minimum commission, and slippage, return fell from +108.15% to +100.56%, win rate from 51.69% to 48.91%, and profit/loss ratio from 3.966 to 3.560.
JoinQuant training result: Not run because the local candidate failed the frozen materiality and accuracy gates. No JoinQuant or PTrade candidate was generated; both formal files remain unchanged.
Validation result: Not run. Reserved periods, 2026 trades, and live outcomes were not inspected or used.
Why it failed: The larger and retained state score became operational, but the additional decisions reduced rather than improved closed-trade accuracy and weakened payoff quality. Persisting an oscillator state for three sessions adds stale reversal votes; tiering makes those votes large enough to alter the portfolio, but does not prove that price has actually reversed. The effect was also concentrated in 2019 and absent in 2021.
Can it be revisited? no as tiered or retained KDJ state scoring in the current framework
Conditions for revisiting: Do not search 8/4, 12/6, another 20/30/70/80 boundary, 2/4/5-day retention, cumulative scoring, per-ETF/year rules, or price/ADX bypasses. Historical point-in-time IOPV remains unavailable, so the PTrade-only premium interaction cannot be inferred from this local test.

Date: 2026-08-23
Version: `cross-v0.3.3-kdj-tiered-current-state-candidate` (local research only)
Experiment: Keep the fixed KDJ tiers from the prior user-authorized candidate but remove all state retention. On the current causal T-1 snapshot only, K<=20 contributes 10 buy points, 20<K<=30 contributes 5 buy points, 70<=K<80 contributes 5 sell points, and K>=80 contributes 10 sell points. No KDJ cross is required; the current official `downside_continuation` state blocks the buy bonus. Keep the formal 60/30 thresholds and every price, ADX, hold, ATR, ranking, sizing, pool, friction, and 09:35 rule unchanged.
Hypothesis: Removing stale state memory may preserve the intended low/high-location contribution while avoiding the inaccurate additional trades created by three-session retention.
Training diagnostic result: The one fixed 2019-2021 local A/B changed zero filled-order days in every training year. Both arms retained 92 buys and 89 sells.
Candidate result: Baseline and candidate were exactly identical: +125.00% total return, 31.13% annualized return, 6.03% maximum drawdown, 2.262 Sharpe, 3.581 Sortino, 56.18% win rate, 4.878 profit/loss ratio, and annual returns of +35.84%/+52.68%/+8.49%. Under doubled commission, minimum commission, and slippage, both arms again matched at +108.15% return, 6.39% drawdown, 51.69% win rate, and 3.966 profit/loss ratio.
JoinQuant training result: Not run because the local materiality and accuracy gates failed. No JoinQuant or PTrade candidate was generated; both formal files remain unchanged.
Validation result: Not run. Reserved periods, 2026 trades, and live outcomes were not inspected or used.
Why it failed: Removing persistence eliminated the harmful extra trades, but also removed every operational effect. The current-session tier points never changed the completed order path after the formal threshold, ranking, price confirmation, ADX, and portfolio constraints were applied. The candidate therefore cannot improve accuracy or return.
Can it be revisited? no as current-session-only KDJ tier scoring in the current framework
Conditions for revisiting: Do not search 8/4, 12/6, another K boundary, RSI/J/D duplicates, per-ETF/year rules, or removal of price/ADX protections. Historical point-in-time IOPV remains unavailable, so the live premium interaction was not tested.

Date: 2026-08-23
Version: `cross-v0.3.3-kdj-tiered-direct-exit-candidate` (local research only)
Experiment: Keep the fixed current-T-1 KDJ tiers: K<=20 adds 10 buy points, 20<K<=30 adds 5 buy points, 70<=K<80 adds 5 sell points, and K>=80 adds 10 sell points. Keep the current `downside_continuation` buy block and no state retention. After the official five-session minimum hold, whenever a positive KDJ extreme sell bonus is present and the final unified sell score is at least 30, sell directly without price confirmation or ADX protection. Keep ATR stops, pool, sizing, costs, ranking, and 09:35/T-1 timing unchanged.
Hypothesis: The extreme KDJ state supplies sufficient high-location evidence to convert an otherwise protected sell score into an immediate exit, reducing profit giveback while the unchanged five-session hold prevents very early churn.
Training diagnostic result: The isolated 2019-2021 local A/B changed 155 filled-order days across every training year (2019: 51, 2020: 44, 2021: 60). Buys/sells increased from 92/89 to 136/134. Win rate rose from 56.18% to 58.21%, but the much higher recycling materially increased path dependence and friction exposure.
Candidate result: Rejected by the frozen local gate. Total return fell from +125.00% to +101.84%, annualized return from 31.13% to 26.46%, maximum drawdown rose from 6.03% to 6.33%, and profit/loss ratio fell from 4.878 to 2.954. Annual returns changed from +35.84%/+52.68%/+8.49% to +20.50%/+56.11%/+7.30%. Under doubled friction, return fell from +108.15% to +77.35%, drawdown rose from 6.39% to 7.08%, and profit/loss ratio fell from 3.966 to 2.359.
JoinQuant training result: Not run because the local strict gate failed. No JoinQuant or PTrade candidate was generated; both formal strategy files remain unchanged.
Validation result: Not run. Reserved periods, 2026 prices, and live outcomes were not inspected or used.
Why it failed: An overbought KDJ state is common during profitable trends and does not by itself prove that price has begun a durable reversal. Removing both price confirmation and ADX protection raised the fraction of winning trades but shortened payoff enough to collapse the profit/loss ratio; additional exits and re-entries also made the result substantially more friction-sensitive. The one improved 2020 annual return did not compensate for the large 2019 loss of payoff or the weaker 2021 result.
Can it be revisited? no as a KDJ-extreme direct-sell override in the current framework
Conditions for revisiting: Keep the official price-confirmation and ADX protections for KDJ-based sells. Do not search nearby K tiers, point values, final-score thresholds, retention windows, partial bypasses, ETF/year exceptions, minimum holds, cooldowns, or IOPV combinations from this result. Reopening requires a genuinely independent exit mechanism and explicit authorization.

Date: 2026-08-23
Version: `cross-v0.3.3-sell-score-rebalance-candidate` (local research only)
Experiment: On top of the frozen current-T-1 KDJ moderate tiers (buy +20/+10 and sell +10/+5), leave the buy path unchanged and replace only the sell-score composition. Score one/two RSI down crosses as 12/20, MACD down as 6, one/two KDJ down crosses as 5/10, and one non-cumulative price-weakness bucket at 6/8/10/12 for BOLL-mid/fell-inside, high-location RSI decline, MA20 break, or falling-MA10/downside-continuation respectively. Add the current KDJ sell-tier bonus, retain the sell threshold at 30, and preserve price confirmation, ADX protection, five-session minimum hold, ATR stop, pool, ranking, sizing, costs, and causal 09:35/T-1 timing.
Hypothesis: Reallocating part of lagging MACD weight to current price damage, while capping correlated RSI/KDJ/price-family votes, may recognize genuine reversals earlier without recreating the rejected KDJ-only or MACD-fast exit families.
Training diagnostic result: The frozen local three-path comparison returned +125.00% for the official path, +119.48% for the current KDJ moderate path, and +118.64% for the rebalanced candidate. Win rates were 56.18%/54.35%/53.26%, profit/loss ratios 4.878/4.366/4.301, and maximum drawdowns 6.03%/6.12%/6.14%. Candidate annual returns were +33.51%/+51.29%/+8.24%. Against the current KDJ path, only 6 filled-order days changed (2019=4, 2020=2, 2021=0). Under doubled friction, candidate return was +100.44%, win rate 47.83%, and profit/loss ratio 3.474 versus official +108.15%/51.69%/3.966.
Candidate result: Rejected by seven frozen gates: no 2021 materiality; win rate failed to improve either comparator; nominal return and profit/loss ratio retained less than 95% of official; and doubled-friction return and win rate failed their guards.
Target attribution: The two predeclared profit-giveback entries were unchanged. For 512100 bought 2019-09-30, the rebalanced score reached 30 on 2019-10-17 and 2019-10-18, but both 09:35 minute bars had volume=0 and num_trades=0, so the local broker could not fill; it still sold 2019-10-21 at -1.49%. For 513880 bought 2021-03-04, the high-area score remained below 30 and it still sold 2021-03-23 at -0.48%.
JoinQuant training result: Not run because the local gate failed. No standalone JoinQuant or PTrade candidate was generated; both formal strategies remain unchanged.
Validation result: Not run. Reserved validation, pressure, recent/live, and full-period data were not read or used.
Why it failed: Price-structure scoring is internally more gradual, but earlier threshold crossings did not improve aggregate trade accuracy or payoff. The triggering 512100 example also showed that a causal score at threshold does not guarantee an executable 09:35 fill. Redistributing weights cannot solve both sparse reversal evidence and execution liquidity, while broader early selling again weakens payoff quality.
Can it be revisited? no as a nearby sell-score redistribution in the current framework
Conditions for revisiting: Keep the official sell weights and threshold. Do not search adjacent 18/22 RSI caps, 4/8 MACD weights, 4/8/10/14 price buckets, lower thresholds, additive price flags, ETF/year exceptions, or execution-time exceptions from this training result. Reopening requires a genuinely independent exit mechanism plus prospective evidence and explicit authorization.

Date: 2026-08-23
Version: `cross-v0.3.3-kdj-ranking-only-buy-candidate` (local research only)
Experiment: Preserve the current causal T-1 KDJ moderate tiers, but remove the buy tier from formal eligibility and position sizing. Keep the official buy score and threshold 60 for both purposes; use K<=20 +20 and 20<K<=30 +10 only as a ranking score among already qualified official candidates. Keep the KDJ sell tier, price confirmation, ADX protection, five-session hold, ATR stop, pool, costs, and all other rules unchanged.
Hypothesis: KDJ extreme location may be useful as a relative tie-break after independent evidence already establishes eligibility, while preventing a weak official setup from entering solely because K is low.
Training diagnostic result: The frozen 2019-2021 three-path replay produced official/current-KDJ/ranking-only returns of +125.00%/+119.48%/+125.00%, win rates of 56.18%/54.35%/56.18%, profit/loss ratios of 4.878/4.366/4.878, and maximum drawdowns of 6.03%/6.12%/6.03%. The ranking-only path changed 15 filled-order days versus the current KDJ path (2019=10, 2020=3, 2021=2), but zero filled-order days versus the official path. Under doubled friction it again exactly matched official at +108.15% return, 51.69% win rate, and 3.966 profit/loss ratio.
Candidate result: Rejected by the pre-registered material-effect gate. It removed all operational damage from allowing the KDJ state bonus to cross the buy threshold, but added no independent trade-path benefit versus the simpler official strategy.
Target attribution: The prior changed-day attribution already proved that all four direct KDJ buys were official 41-point setups raised to 61 solely by K<=20 +20; all four later closed at losses (-5.41%, -1.49%, -3.46%, and -0.48%). The ranking-only rule made those entries ineligible. Its exact match to official also proves the ranking bonus did not alter any completed official order path in this training replay.
JoinQuant training result: Not run because the local material-effect gate failed. No standalone JoinQuant or PTrade candidate was generated; both formal strategies remain unchanged.
Validation result: Not run. Reserved validation, recent/live, pressure-period, and full-period data were not read or used.
Why it failed: Qualification and ranking separation is structurally safer than unified KDJ points, but a field that never changes a completed official order adds complexity without measurable benefit. KDJ low state alone is not a reversal confirmation, and among already qualified candidates its relative ranking signal was non-binding in this window.
Can it be revisited? no as a standalone KDJ current-state ranking bonus
Conditions for revisiting: Keep KDJ current-state points out of formal buy eligibility and sizing. Do not search nearby rank points, alternate K bands, threshold changes, ETF/year exceptions, or sell compensation from this result. Reopening early-entry research requires a genuinely independent stop-falling/reversal structure and a separately pre-registered experiment.

Date: 2026-08-23
Version: `cross-v0.3.3-late-veto-early-pre-macd-candidate` (JoinQuant training candidate only)
Experiment: Stack the previously tested late-MACD/BOLL-upper new-buy veto with one early pre-MACD leftover-slot channel. Keep official score-at-least-60 candidates first; then allow score 50-59 only when RSI and KDJ bullish crosses are each no older than one session, MACD remains negative but is narrowing, close is below the BOLL upper band, and RSI6 is at most 85. Preserve the sell path and all other formal rules.
Hypothesis: The late veto may block delayed overheated entries while the early channel captures the same reversal before lagging MACD confirmation, jointly improving entry accuracy without lowering the official primary threshold.
Training diagnostic result: The user-provided completed 2019-2021 JoinQuant log identifies build `20260822.3-candidate`, fingerprint `f6b08195dd3d`, and 20 filled buys tagged `channel=early_pre_macd`, spanning 2019, 2020, and 2021. The corresponding JoinQuant summary reported +97.65% total return, +26.28% annualized return, 6.75% maximum drawdown, 51.5% win rate, 3.700 profit/loss ratio, 52 profitable trades, and 49 losing trades.
Candidate result: Rejected. The formal screenshot comparator was +129.25% return, 6.28% maximum drawdown, 55.8% win rate, and 5.297 profit/loss ratio. The standalone late-veto candidate was +124.09%, 6.28%, 55.8%, and 5.208 respectively. The stacked candidate therefore lost 31.60 percentage points of return versus formal, reduced win rate by 4.3 percentage points, worsened drawdown by 0.47 percentage points, and materially weakened payoff quality.
JoinQuant training result: Completed and rejected by the frozen accuracy, return-retention, drawdown, and payoff gates. Because the tested file stacked the late veto and early channel, the aggregate damage cannot be attributed exclusively to the early channel; nevertheless, the exact stacked rule is disproven.
Validation result: Not run. Reserved validation, recent/live outcomes, and full-period data were not inspected or used.
Why it failed: Fresh oscillator crosses plus negative-but-narrowing MACD did not provide enough independent evidence that price had actually stopped falling. The 50-59 score band admitted many setups before price confirmation, while stacking the late veto also changed later portfolio sequencing. Twenty direct fills established materiality but the completed trades reduced both hit rate and payoff.
Can it be revisited? no as this stacked 50-59, age-at-most-one, narrowing-MACD rule
Conditions for revisiting: Do not search nearby score bands, cross ages, MACD gaps, RSI caps, BOLL distances, ETF/year exceptions, queue order, or sell compensation. A new early-entry idea must use an independently pre-registered causal price-reversal confirmation, retain the official primary queue, and receive separate authorization.

Date: 2026-08-23
Version: `cross-v0.3.3-t1-price-reversal-pre-macd-candidate` (local research only)
Experiment: Preserve the official score-at-least-60 buy queue unchanged and ranked first. Only for a genuinely vacant slot, allow an official score below 60 when the active three-session official snapshot contains at least one RSI bullish cross and at least one KDJ bullish cross, contains no MACD bullish cross, and the completed T-1 bar has a low at least equal to the T-2 low and a close strictly above the T-2 high. Keep `buy_allowed`, location eligibility, entry-combo block, sell score below 30, downside-continuation block, ATR cooldown, sizing, sells, five-session hold, pool, friction, and 09:35/T-1 timing unchanged. Add no score points and search no adjacent thresholds.
Hypothesis: A completed higher/equal-low and prior-high breakout can supply the missing stop-falling evidence that oscillator crosses alone lacked, permitting accurate pre-MACD entries without weakening the official primary path.
Training diagnostic result: The isolated 2018 warm-up plus 2019-2021 training replay produced 26 direct alternative fills across 2019, 2020, and 2021. Twenty-five closed: 8 wins and 17 losses, a 32% direct hit rate. Their combined realized PnL was +1444.10 yuan only because 513050 bought 2020-04-16 and sold 2020-07-16 returned +35.04% and contributed about +2690 yuan; most other direct trades were repeated losses of roughly 1% to 6%.
Candidate result: Rejected by the frozen gate. Official versus candidate was +125.00% versus +70.42% total return, 31.13% versus 19.51% annualized, 6.03% versus 6.28% maximum drawdown, 2.262 versus 1.589 Sharpe, 3.581 versus 2.416 Sortino, 56.18% versus 47.47% win rate, 4.878 versus 2.595 profit/loss ratio, and 92/89 versus 102/99 buys/sells. Annual returns fell in every year: +35.84%/+52.68%/+8.49% to +17.71%/+37.57%/+5.24%.
JoinQuant training result: Not run because the local accuracy, return-retention, Sharpe, Sortino, profit/loss, and doubled-friction gates failed. No standalone JoinQuant or PTrade candidate was generated; both formal strategies remain unchanged.
Validation result: Not run. Reserved validation, recent/live outcomes, 2026 prices, and full-period data were not read or used.
Why it failed: One bullish reversal bar after RSI and KDJ crosses is not durable trend confirmation. It frequently identifies a short bounce inside a weak or unfinished decline, yielding many small losses and relying on one large trend winner. Even though alternatives never displaced a same-day official candidate, they occupied slots on later dates and changed portfolio sequencing, clipping subsequent official opportunities. Under doubled friction return fell from +108.15% to +54.62% and win rate from 51.69% to 42.42%.
Can it be revisited? no as this one-bar T-1/T-2 pre-MACD leftover entry
Conditions for revisiting: Keep the official score threshold and current formal strategy. Do not search a score floor, trend-score floor, strict higher low, close buffer, two-bar variant, volume condition, ETF/year exception, cross-age narrowing, or sell compensation from this failed training result. Reopening requires a genuinely independent prospective feature, a new pre-registration, and explicit authorization.

Date: 2026-08-23
Version: `cross-v0.4.0-dimension-capped-candidate` (local research only)
Experiment: Replace the official additive score aggregation only inside one isolated research candidate with three capped buy dimensions (reversal, location, trend) and two capped sell dimensions (weakness, damage). Require buy score at least 40 with independent floors of reversal at least 12, location at least 7, and trend at least 6; require ordinary signal selling at weakness at least 10, damage at least 8, and total at least 24, while severe damage at least 18 with weakness at least 6 bypasses only ADX protection. Keep the official nine-ETF pool, five-session hold, ATR path, maximum three holdings, 0.95 base ratio, equal-weight ATR stress scale, 09:35 execution, T-1 signal boundary, and all nominal and doubled-friction settings frozen.
Hypothesis: Capping correlated indicator families and requiring independent reversal, price-location, and trend evidence can improve closed-trade accuracy without materially weakening return, drawdown, payoff ratios, trade count, or friction robustness.
Training diagnostic result: The single frozen 2019-2021 local A/B changed 196 filled-order days across every training year (2019: 62, 2020: 64, 2021: 70). Closed trades retained 85 of the baseline 89 (95.51%), so the candidate was operational and passed the materiality and trade-retention gates.
Candidate result: Rejected. Nominal total return fell from +125.00% to +78.13%, annualized return from 31.13% to 21.29%, maximum drawdown rose from 6.03% to 6.37%, win rate fell from 56.18% to 51.76%, Sharpe fell from 2.262 to 1.672, Sortino from 3.581 to 2.533, and profit/loss ratio from 4.878 to 2.831. Annual returns fell from +35.84%/+52.68%/+8.49% to +21.45%/+43.89%/+1.93%. Under doubled friction, return fell from +108.15% to +63.32%, win rate from 51.69% to 45.88%, and profit/loss ratio from 3.966 to 2.347; the candidate's 2021 doubled-friction return was -1.26%.
JoinQuant training result: Not run because the frozen local gate failed. No JoinQuant or PTrade candidate was generated; both formal strategy files remain unchanged.
Validation result: Not run. The 2022-2023, 2024-current, 2015-2018, recent/live, and full-period reserved windows were not read or used.
Why it failed: All seven frozen quality guards failed: candidate win rate does not strictly improve; candidate retains less than 95% of baseline return; candidate Sharpe ratio retains less than 95%; candidate Sortino ratio retains less than 95%; candidate profit/loss ratio retains less than 95%; doubled-friction return retains less than 95%; and doubled-friction win rate is below baseline. The architecture generated broad order-path changes but reduced both hit rate and payoff quality, so the added structural complexity is not justified.
Can it be revisited? no as the frozen v0.4 dimension-capped score family
Conditions for revisiting: Keep formal `cross-v0.3.3` unchanged. Do not search nearby dimension points, caps, floors, buy/sell thresholds, severe-damage rules, ADX behavior, filters, rankings, indicator deletions, ETF/year exceptions, friction settings, hold periods, or portfolio settings from this result. A future proposal must be genuinely independent, prospectively justified, explicitly authorized, and separately pre-registered.
