# Cross-Signal ETF Strategy Spec

Status: v0.1 rules drafted, not implemented.

## Hypothesis

Short-term cross signals can detect swing turning points earlier than a momentum-led rotation strategy. A disciplined combination of reversal, location, trend filter, volume confirmation, and ATR risk control may reduce "buy after several days of rising, then immediately pull back" behavior.

## Development Boundary

- Training period: 2019-01-01 to 2021-12-31.
- Reserved validation periods must not influence first-version rule design.
- Full-period result must be checked only after rules are frozen.

## Strategy Family

Cross-signal swing timing strategy.

This is not a pure momentum rotation strategy. Momentum can be used as context, but should not dominate first-version buy decisions.

## ETF Pool

Use the same 12-ETF pool as the current multi-factor strategy for the first version unless a liquidity or data issue requires exclusion.

Rationale: isolate signal logic first; do not mix signal redesign with ETF-pool redesign.

## Candidate Signal Groups

### Primary Reversal Signals

- RSI6 crosses above RSI12 or RSI24.
- MACD DIF crosses above DEA.
- KDJ K or J crosses above D.

Signals may be counted within a short recent window rather than requiring same-day alignment.

### Location Context

- Price near MA20 or recovering from BOLL middle/lower region is preferred.
- Price far above MA20 is treated as higher chase risk.

### Trend Filter

- Avoid buying obvious downside continuation.
- A light MA20/MA60 context filter is allowed, but it must not turn the strategy back into a late momentum strategy.

### Volume Confirmation

- Volume expansion can confirm reversal quality.
- Volume should not dominate because ETF volume, especially cross-market ETF volume, can be noisy.

### Risk Control

- Use ATR-based stop logic.
- Consider profit protection only if it follows a broad, pre-declared rule.
- Do not add highly specific profit or loss thresholds from one period.

## Buy Logic Draft

Buy candidates with the strongest cross-signal score after risk and location filters. Version v0.1 uses a 100-point score.

### v0.1 Buy Score

- Reversal score, max 45:
  - RSI6 crosses above RSI12 within the last 3 trading days: +12.
  - RSI6 crosses above RSI24 within the last 3 trading days: +12.
  - MACD DIF crosses above DEA within the last 3 trading days: +10.
  - KDJ K crosses above D within the last 3 trading days: +6.
  - KDJ J crosses above D within the last 3 trading days: +5.

- Location score, max 25:
  - Close is between BOLL lower band and BOLL middle band: +10.
  - Close crosses back above BOLL middle band within the last 3 trading days: +8.
  - Close is within +/-5% of MA20: +7.
  - If close is more than 12% above MA20, subtract 10 as chase-risk penalty.

- Trend-context score, max 20:
  - MA5 > MA10: +6.
  - MA10 > MA20: +6.
  - MA20 slope over 5 trading days is non-negative: +5.
  - Close > MA60: +3.
  - If close < MA60 and MA20 slope is negative, subtract 15 as downside-continuation penalty.

- Volume-confirmation score, max 10:
  - Latest volume is above VOL20 and close is up on the day: +6.
  - VOL5 > VOL20: +4.

### v0.1 Buy Threshold

- Candidate threshold: buy_score >= 60.
- Strong candidate threshold: buy_score >= 70.
- If no candidate reaches 60, hold cash instead of forcing a trade.
- Rank by buy_score descending, then by reversal score descending, then by code for deterministic ordering.

### v0.1 Buy Constraints

- Use T-1 daily bars for all signal calculation.
- Execute planned trades at the scheduled JoinQuant order time using current tradable price.
- Do not buy paused securities or securities with invalid price/volume data.
- Do not buy if latest RSI6 >= 85; this is an overheat guard, not an optimized value.

## Sell Logic Draft

Sell when short-term strength rolls over or risk control triggers:

- RSI6 crosses below RSI12 or RSI24.
- MACD DIF crosses below DEA.
- KDJ K or J crosses below D.
- ATR stop triggers.
- Profit protection triggers if present in the frozen design.

A single weak signal may tighten risk, while multi-indicator downside resonance can force exit.

### v0.1 Sell Score

- Downside reversal score:
  - RSI6 crosses below RSI12 within the last 3 trading days: +12.
  - RSI6 crosses below RSI24 within the last 3 trading days: +12.
  - MACD DIF crosses below DEA within the last 3 trading days: +10.
  - KDJ K crosses below D within the last 3 trading days: +6.
  - KDJ J crosses below D within the last 3 trading days: +5.

- Location/risk score:
  - Close is more than 10% above MA20 and RSI6 turns down: +8.
  - Close falls below MA10 while MA10 slope is negative: +10.
  - Close falls from above BOLL upper band back inside the band: +6.

### v0.1 Sell Rules

- Force sell when ATR stop triggers.
- Force sell when sell_score >= 30.
- Normal signal sells require at least 5 trading days since entry. ATR stops are not subject to this minimum hold rule.
- If sell_score is 18 to 29, keep the position but mark it as risk-tightened in the log.
- Do not sell a position bought on the same trading day.
- If a held ETF still has buy_score >= 70 and sell_score < 30, keep it to avoid selling strong trends too early.

## Portfolio And Schedule

- Initial capital assumption: 20,000 CNY.
- Max holdings: 3.
- Base capital usage: 0.90 of account value. The original 0.75 setting was only the inherited initial test baseline; the 2019-2021 training-only exposure diagnostic showed capital utilization, not signal scarcity, was the first structural bottleneck.
- Position sizing: equal-weight among selected holdings for v0.1. Do not add volatility-inverse sizing yet, so signal quality can be evaluated before sizing complexity.
- Rebalance days: Tuesday and Thursday, same as the current production strategy.
- Daily stop check: 09:35.
- Rotation buy/sell: 09:35 on rebalance days.
- After-close logging: 15:30.

## Risk Management

- ATR period: 14.
- Initial trailing stop: 2.5 * ATR from highest close since buy.
- Stop floor/cap: use broad current-production defaults for the first version unless code reuse proves inappropriate.
- Profit floor: disabled in v0.1. The first version should test cross-signal sell timing before adding profit-floor protection.
- Minimum hold: normal signal-based sells require 5 trading days after entry; ATR stops remain unconditional risk control.

## Logging Requirements

Each rebalance-day log should show:

- Data date used for signals.
- Top candidates with buy_score, reversal score, location score, trend-context score, volume score.
- RSI6/12/24 values and whether an up/down cross was detected.
- MACD DIF/DEA and cross state.
- KDJ K/D/J and cross state.
- Close position versus MA20 and BOLL.
- Buy, sell, hold, and risk-tighten reasons.
- ATR stop price and highest close since buy for held positions.

## Success Criteria

The strategy is worth further validation only if:

- Training-period results are reasonable and explainable.
- It does not collapse in reserved validation periods.
- It improves at least one target weakness versus the production strategy, such as bad entries, drawdown, or sell responsiveness.
- Added complexity has measurable benefit.

## Safety Checks

- Signals at 09:35 must use T-1 daily data, not today's full-day data.
- Do not use validation-period performance to tune first-version thresholds.
- Record every material experiment in `decisions.md` or `failed_experiments.md`.
