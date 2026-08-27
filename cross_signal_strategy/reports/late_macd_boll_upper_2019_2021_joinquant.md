# Late-MACD/BOLL-upper candidate — JoinQuant training result

Date: 2026-08-22  
Window: 2019-01-01 through 2021-12-31  
Initial capital: CNY 20,000  
Frequency: daily  
Version: `cross-v0.3.3-late-macd-boll-filter-candidate`  
Build: `20260822.2-candidate`  
Fingerprint: `a46fff884685`  
Source-log SHA-256: `AF7E254A7F6C21F9AFA07778375A1922E948E2A6F9A00CBD17494C9989DB8A4E`

## Outcome

The candidate is rejected. It did not improve accuracy: closed-trade win rate
was unchanged at 55.8%. Total and annualized return, profit/loss ratio, Sharpe,
Sortino, and information ratio all worsened; maximum drawdown was unchanged.

| Metric | Formal cross-v0.3.3 | Candidate | Decision |
|---|---:|---:|---|
| Total return | 129.25% | 124.09% | worse |
| Annualized return | 32.86% | 31.83% | worse |
| Maximum drawdown | 6.28% | 6.28% | equal |
| Win rate | 55.8% | 55.8% | no improvement |
| Profit/loss ratio | 5.297 | 5.208 | worse |
| Sharpe | 2.275 | 2.185 | worse |
| Sortino | 3.245 | 3.028 | worse |
| Information ratio | 0.839 | 0.790 | worse |

## Path evidence

- The log contains exactly two `[late-macd-boll-veto]` events: 513100 on
  2019-03-15 and 159928 on 2019-12-31.
- The candidate still recorded 98 buys and 95 sells. Released slots were later
  or immediately occupied, so a veto did not simply remove one bad round trip.
- 513100 was bought again on 2019-04-02 after the veto, at a later and higher
  execution price. The 159928 veto promoted 159915 on the same date.

No validation period was inspected. The standalone candidate remains isolated
for comparison and is not promoted to the formal strategy.
