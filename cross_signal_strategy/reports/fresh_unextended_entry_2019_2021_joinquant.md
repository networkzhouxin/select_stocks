# Fresh-Unextended Entry Candidate: Official JoinQuant Result

Date: 2026-08-22  
Version: `cross-v0.3.3-fresh-unextended-entry-candidate`  
Build: `20260822.1-candidate`  
Business fingerprint: `25783cc30ba4`  
Window: 2019-01-01 through 2021-12-31, CNY 20,000, daily  
Authority: official JoinQuant training run; no validation data used

## Result

| Metric | Official baseline | Candidate | Gate result |
| --- | ---: | ---: | --- |
| Total return | 129.25% | 111.14% | Fail: below 95% retention |
| Maximum drawdown | 6.28% | 6.29% | Fail: worse |
| Closed-trade win rate | 55.8% | 49.0% | Fail: not +3pp |
| Profit/loss ratio | 5.297 | 3.904 | Pass: at least 3.0 |
| Positive-to-negative round trips | 31 | 39 | Fail: increased by 8 |

All three candidate calendar years remained positive: 2019 +25.28%, 2020
+59.54%, and 2021 +5.64%. That one pass cannot override the failed return,
drawdown, accuracy, and giveback gates.

The official log contained 105 filled buys and 102 filled sells. The added
fresh channel had 19 closed trades and one still-open trade at the end of the
run. Its closed trades were 4 winners and 15 losers (21.05% win rate), with
net PnL of CNY 1,853.80. Two winners (`159928` and `513050`) supplied about
88.5% of the channel's gross profit, masking the large majority of losing
entries. Primary-channel closed trades remained much stronger at 46 winners
and 37 losers.

## Decision

`REJECT`. Archive the standalone candidate and keep formal `cross-v0.3.3`
unchanged. Do not search neighboring score bands, reversal thresholds, cross
ages, ATR-extension limits, ETF exceptions, or sell-side compensation. The
official result attributes the failure to the new buy channel, not to the
unchanged sell rules.

Source log SHA-256:
`D2E42BCB0293A692B4EE2ED402C44713E31D54554FEAFAB08F5338CD843C90C1`.
