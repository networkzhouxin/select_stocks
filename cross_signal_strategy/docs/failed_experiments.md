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
