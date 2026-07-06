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
