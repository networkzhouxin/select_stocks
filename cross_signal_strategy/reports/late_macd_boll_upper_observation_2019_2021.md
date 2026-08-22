# Late-MACD / BOLL-Upper Official-Buy Observation

Date: 2026-08-22  
Baseline: `cross-v0.3.3`, build `20260820.1`, fingerprint `77e44d93d255`  
Baseline log SHA-256: `8247648835F2856AFB730DF9332DE6CF39A235632929FD14DCFE2F62E35B1434`  
Data: read-only 2018 warm-up plus 2019-2021 training only  
Validation: not inspected

## Frozen rule and gate

Inspect official filled buys only. A match requires the complete T-1 snapshot
to have MACD bullish-cross age 0, an active RSI bullish cross at age 1/2, an
active KDJ bullish cross at age 1/2, and close at or above BOLL upper. A single
new-buy-only veto candidate is permitted only with at least 3 matches across
at least 2 training years.

## Result

- Official filled buys: 98.
- Matches: 2.
- Distinct years: 1 (2019 only).
- Gate: `STOP`.

| Buy date | Code | T-1 signal date | Close | BOLL upper | RSI age | KDJ age | MACD age |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2019-03-15 | 513100 | 2019-03-14 | 2.535000 | 2.524726 | 2 | 1 | 0 |
| 2019-12-31 | 159928 | 2019-12-30 | 3.000000 | 2.997819 | 1 | 1 | 0 |

## Decision

No JoinQuant candidate is created. Formal `cross-v0.3.3` remains unchanged.
The event is too sparse and confined to one year; relaxing any age, price
location, indicator, ETF, or year condition after seeing this result is
prohibited.
