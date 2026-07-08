# Cross-Signal Frozen Validation Summary

Date: 2026-07-09

This file summarizes the frozen validation evidence for the official cross-signal mainline and the ATR-stress candidate. It is a decision record, not a parameter-search notebook. Do not use these reserved-period results to tune thresholds, add indicators, remove ETFs, or choose a new validation-fitting variant.

## Strategy Family

The cross-signal strategy is a daily ETF strategy built around reversal/cross signals. Its goal is to avoid the "buy high, sell low" behavior that can happen in pure momentum rotation during choppy markets, while still letting profitable trends run through ATR-based risk control.

Official mainline:
- File: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
- Version: `cross-v0.3.1`
- Role: official training-confirmed mainline.

ATR-stress candidate:
- File: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
- Version: `cross-v0.3.1-atr-stress-candidate`
- Rule: if the portfolio has at least 3 ATR stops in the recent 15 trading days, new buys are scaled to 50% target size.
- Role: low-frequency crash-regime risk-control candidate.

## Cross-Period Results

| Period | Role | Official Return | Official Max DD | Official Sharpe | ATR-Stress Return | ATR-Stress Max DD | ATR-Stress Sharpe | ATR-Stress Trigger |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2019-2021 | training | +120.42% | 6.82% | 2.097 | +122.47% | 6.38% | 2.160 | 4 half-size buys |
| 2022-2023 | first validation | +15.49% | 13.38% | 0.346 | +16.01% | 12.94% | 0.373 | 3 half-size buys |
| 2024-2026 | recent validation | +56.99% | 10.65% | 1.276 | +56.99% | 10.65% | 1.276 | 0 half-size buys |
| 2015-2018 | stress validation | +23.58% | 7.49% | 0.192 | +23.58% | 7.49% | 0.192 | 0 half-size buys |
| 2010-2014 | early supplement | -0.61% | 5.36% | -0.822 | -0.61% | 5.36% | -0.822 | assumed inactive from identical summary |

Notes:
- JoinQuant is the performance authority for these results.
- 2010-2014 is an early out-of-sample supplement with an incomplete ETF pool. It is useful for operational sanity, but it should not be weighted like the complete-pool periods.
- The 2010-2014 ATR-stress summary matched the official summary exactly. Because ATR-stress only changes new-buy sizing when the stress rule triggers, this is treated as the same trading path unless a later transaction export proves otherwise.

## Operational Checks

Official `cross-v0.3.1`:
- Passed all recorded windows without runtime errors.
- Removed symbols `510300`, `510880`, and `159920` did not appear in recorded trade logs or transaction exports where checked.
- Known warning/cancellation cases were execution-liquidity facts, not strategy-code errors:
  - 2019-12-12 `513880.XSHG` zero-volume market-order cancellation.
  - 2016-08-03 and 2017-03-09 `159928.XSHE` zero-share sell cancellations, followed by normal sells on the next trading day.

ATR-stress candidate:
- Did not create extra warnings or errors in recorded windows.
- Improved training and 2022-2023 slightly.
- Was inactive in 2024-2026 and 2015-2018.
- Appears inactive in 2010-2014 because the headline summary is identical to the mainline and the early ETF pool was sparse.

## Interpretation

What the evidence supports:
- The official `cross-v0.3.1` mainline is not just a 2019-2021 fit. It remained positive in 2022-2023, 2024-2026, and 2015-2018.
- The strategy's weakest recorded complete-pool validation window was 2022-2023: positive return but weaker Sharpe and higher drawdown.
- The strategy handled 2015-2018 better than the benchmark with controlled drawdown, despite weaker Sharpe.
- The ATR-stress rule behaves like a narrow insurance rule: it helped in crash-like clusters and stayed out of the way elsewhere.

What the evidence does not prove:
- It does not prove the cross-signal strategy is superior to the existing production momentum/multifactor strategies.
- It does not prove the current ETF pool is globally optimal.
- It does not justify tuning `15 days`, `3 stops`, or `0.50 scale`.
- It does not justify adding indicators or deleting ETFs from validation-period attribution.
- The 2010-2014 result does not prove failure of the strategy family, because most current pool ETFs were unavailable or not fully usable then.

## Adoption Recommendation

Official mainline:
- Keep `cross-v0.3.1` as the official cross-signal baseline.
- It has passed the training window and multiple reserved windows without collapse.
- It is still a research strategy, not yet a production replacement for the existing deployed strategies.

ATR-stress candidate:
- Keep as a valid candidate, but do not automatically merge it solely from the current evidence.
- Professional rationale: the rule is broad and understandable, and it did not harm validation windows where inactive.
- Overfitting concern: the benefit comes from a small number of clustered stress events, so the evidence is not yet strong enough to treat it as mandatory mainline logic.
- If the user's priority is maximum drawdown insurance and accepts one extra low-frequency rule, it is reasonable to promote it after a dedicated code-review and PTrade/live-operability review. If the user's priority is simplicity, keep it as an experimental candidate.

Current recommendation:
- Do not tune or add indicators from validation results.
- Finish the frozen evidence record, then start a new training-only research cycle for the next structural improvement.

## Next Training-Only Research Directions

These are allowed only on the 2019-2021 training protocol before any new validation inspection:

- Entry quality: distinguish strong reversal crosses from weak repair bounces without narrow thresholds.
- Trend continuation: review whether profitable positions are still sold too early by normal signal exits.
- Volume confirmation: keep A-share-only volume logic disciplined; do not generalize it to QDII/cross-asset ETFs without training evidence.
- Bollinger/ADX usage: test only as broad structure filters or confirmations, not as many small tuned parameters.
- Cash handling: study whether idle cash needs a conservative fallback, but do not copy momentum-rotation bond logic blindly.
- ETF availability/liquidity: make early-period and sparse-liquidity behavior explicit for live deployment.

## Future-Function And Overfitting Check

- All strategy decisions were developed on the 2019-2021 training workflow before reserved-period inspection.
- Reserved windows were used to validate and summarize, not to tune rules.
- No rule should be changed from this summary without starting a new training-only experiment and recording the hypothesis before implementation.
