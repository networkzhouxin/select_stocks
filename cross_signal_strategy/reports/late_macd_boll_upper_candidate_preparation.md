# Late-MACD / BOLL-Upper JoinQuant Candidate Preparation

Date: 2026-08-22  
Candidate: `cross-v0.3.3-late-macd-boll-filter-candidate`  
Build: `20260822.2-candidate`  
Business fingerprint: `a46fff884685`  
Backtest window: 2019-01-01 through 2021-12-31  
Initial cash: CNY 20,000; daily frequency

## Authorization and scope

The prior Step 0 observation failed its 3-event/2-year gate with only two
2019 matches. Their official closed outcomes were one win and one loss; a
direct veto would have avoided about CNY 91.20 of loss but missed about CNY
446.40 of profit. After those facts were disclosed, the user explicitly
requested creation of the exact JoinQuant candidate for an authoritative
portfolio-path backtest.

This is a single isolated candidate, not a production change. Formal
`cross-v0.3.3` remains unchanged.

## Frozen change

Reject a new buy only when its T-1 snapshot has all of:

- MACD bullish-cross age 0;
- active RSI bullish-cross age 1 or 2;
- active KDJ bullish-cross age 1 or 2;
- close at or above BOLL upper.

All scores, thresholds, periods, ranking, existing entry guards, sell logic,
ADX protection, ATR stop, sizing, pool, costs, and 09:35/T-1 execution remain
unchanged. The log marker for a rejected buy is `[late-macd-boll-veto]`.

## Frozen official decision gates

- Closed-trade win rate must improve.
- Positive-to-negative round trips must not increase.
- Total return must retain at least 95% of the official baseline.
- Maximum drawdown must not worsen.
- Profit/loss ratio must remain at least 3.0.
- Every training year must remain positive.

Do not run doubled friction or validation unless all nominal gates pass. Do
not alter the rule after seeing the result.
