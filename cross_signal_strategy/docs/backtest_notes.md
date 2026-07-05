# Backtest Notes

Use one section per backtest. Record results before deciding whether to change rules.

## Template

```text
Version:
Code file:
Backtest period:
Protocol role: training / validation / stress / early-oos / final-summary
Initial capital:

Strategy return:
Annualized return:
Excess return:
Benchmark return:
Alpha:
Beta:
Sharpe:
Sortino:
Win rate:
Daily win rate:
Profit/loss ratio:
Max drawdown:
Max drawdown period:
Trade count:
Average holding days:

Main observations:
Bad entries observed:
Sell timing observations:
Abnormal logs/errors:

Can this result be used to change rules? yes/no
Reason:
```

## Results

### Local Replay Baseline: cross-v0.2.6 Mechanics Draft

Version: local replay foundation, using `cross-v0.2.6` scoring rules
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training infrastructure check
Initial capital: 20000

Strategy return: +15.40%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 10.40%
Max drawdown period: not calculated
Trade count: 216 filled orders; 109 buys, 107 sells
Average holding days: not calculated

Main observations:
- Local replay completed all 730 training trading days using only `G:\financial\history_data\cross_signal_train_2019_2021`.
- Maximum holdings stayed within `max_hold=3`.
- Final value was 23080.56 with final holdings `159985,518880`.
- Result is far below the JoinQuant training-period reference (+50.07%), so local mechanics are not yet aligned enough for performance conclusions.

Bad entries observed: not reviewed
Sell timing observations: not reviewed
Abnormal logs/errors: none during the local replay smoke run

Can this result be used to change rules? no
Reason: This is a local infrastructure/mechanics check, not an aligned authoritative backtest. Differences versus JoinQuant must be diagnosed as execution/data/mechanics issues before using local results for strategy decisions.

### Local Replay With 2018 Daily Warm-Up

Version: local replay foundation with 2018 daily warm-up
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training infrastructure/mechanics check
Initial capital: 20000

Strategy return: +34.67%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 10.14%
Max drawdown period: not calculated
Trade count: 236 filled orders; 119 buys, 117 sells
Average holding days: not calculated

Main observations:
- Local replay completed all 730 training trading days.
- 2018 daily warm-up is used only as an indicator lookback buffer. Performance statistics still start on 2019-01-02.
- Return improved from the no-warm-up local baseline (+15.40%) to +34.67%, confirming that early-2019 lookback truncation was a material local-mechanics gap.
- Result remains below the JoinQuant training-period reference (+50.07%), so more mechanics alignment is required before using local replay for strategy decisions.

Bad entries observed: not reviewed
Sell timing observations: not reviewed
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is still a local mechanics-alignment checkpoint. It confirms the need for warm-up data but does not justify parameter, indicator, or rule changes.

### Local Replay With Warm-Up And ATR Stop State

Version: local replay foundation with 2018 daily warm-up and ATR stop state
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training infrastructure/mechanics check
Initial capital: 20000

Strategy return: +41.42%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 9.56%
Max drawdown period: not calculated
Trade count: 254 filled orders; 128 buys, 126 sells
Average holding days: not calculated

Main observations:
- Local replay now records `entry_atr`, `highest_since_buy`, buy dates, and runs ATR stop checks before signal sells/buys.
- Return improved from the warm-up-only local baseline (+34.67%) to +41.42%.
- Result is closer to, but still below, the JoinQuant training-period reference (+50.07%).
- Remaining likely mechanics gaps include target-value total portfolio valuation at 09:35, exact JoinQuant `order_target_value` sizing/fill behavior, and data/fq alignment.

Bad entries observed: not reviewed
Sell timing observations: not reviewed
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is still a local mechanics-alignment checkpoint, not an authoritative strategy result.

### Local Replay With 09:35 Portfolio Mark For Target Sizing

Version: local replay foundation with 2018 daily warm-up, ATR stop state, and 09:35 position marks for buy target sizing
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training infrastructure/mechanics check
Initial capital: 20000

Strategy return: +42.99%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 9.69%
Max drawdown period: not calculated
Trade count: 254 filled orders; 128 buys, 126 sells
Average holding days: not calculated

Main observations:
- New buy target value now uses cash plus existing positions marked at 09:35 when those prices are available.
- Return improved from the ATR-state local baseline (+41.42%) to +42.99% with unchanged trade count.
- The result remains below the JoinQuant training-period reference (+50.07%), so exact order sizing/fill behavior and data/fq alignment still need review.

Bad entries observed: not reviewed
Sell timing observations: not reviewed
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is a local execution-mechanics alignment checkpoint. It does not justify strategy parameter, indicator, or rule changes.

### Local Replay With New-Listing Indicator-Ready Scoring

Version: local replay foundation with 2018 daily warm-up, ATR stop state, 09:35 position marks, and local scoring that allows new listings once required indicators are valid
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: JoinQuant mechanics/log alignment check
Initial capital: 20000

Strategy return: +42.75%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 9.68%
Max drawdown period: not calculated
Trade count: 256 filled orders; 129 buys, 127 sells
Average holding days: not calculated

Main observations:
- JoinQuant training log has 262 filled orders; local replay previously had 254.
- First order-path divergence was 2019-10-18: JoinQuant bought `513880`, while local replay skipped it as `short_data:77<110`.
- Local adapter now allows new listings when the required indicator fields are valid, using the longest structural indicator window as the local minimum history gate.
- Local replay now also buys `513880` on 2019-10-18, reducing the order-path gap from 8 orders to 6 orders.
- Next observed divergence is 2019-11-13/2019-11-18 on `159928`: local adds a 10-point `close_below_falling_ma10` risk score and sells earlier, while JoinQuant only logs risk-tightening on 2019-11-13 and sells on 2019-11-18.

Bad entries observed: not reviewed
Sell timing observations: 2019-11-13 local sell timing differs from JoinQuant for `159928`; likely MA10/risk-score data or calculation alignment issue.
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is a local mechanics/log-alignment checkpoint. It fixes local replay parity for new-listed ETFs but does not justify strategy parameter, indicator, or rule changes.

### Local Replay With Falling-MA10 Sell Confirmation Fix

Version: local replay foundation with new-listing scoring and falling-MA10 sell confirmation requiring price decline
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: JoinQuant training-log alignment and sell-structure correctness check
Initial capital: 20000

Strategy return: +47.50%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 9.57%
Max drawdown period: not calculated
Trade count: 248 filled orders; 125 buys, 123 sells
Average holding days: not calculated

Main observations:
- The 2019-11-13 `159928` divergence is fixed at the signal level: local score is now `sell=24 force=False`, matching JoinQuant's risk-tighten-only behavior.
- The later 2019-11-18 `159928` sell remains active: local score is `sell=32 force=True`, matching JoinQuant's sell date.
- Return moved from +42.75% to +47.50%, closer to the JoinQuant reference +50.07%, but the trade count differs from JoinQuant's 262 orders and still needs order-path review.

Bad entries observed: not reviewed
Sell timing observations: 2019-11-13/2019-11-18 `159928` sell timing aligned at the signal level after the falling-MA10 confirmation fix.
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is a correctness/log-alignment checkpoint for a sell-confirmation condition, not an optimization pass. No validation-period results were inspected or used.

### Local Replay With Tick-Precision ATR And Core-Indicator New Listings

Version: local replay with falling-MA10 flat-close handling, 0.001 ATR stop comparison, and core-indicator-ready new-listing scoring
Code file: `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: JoinQuant training-log alignment check
Initial capital: 20000

Strategy return: +44.12%
Annualized return: not calculated
Excess return: not calculated
Benchmark return: not calculated
Alpha: not calculated
Beta: not calculated
Sharpe: not calculated
Sortino: not calculated
Win rate: not calculated
Daily win rate: not calculated
Profit/loss ratio: not calculated
Max drawdown: 9.70%
Max drawdown period: not calculated
Trade count: 254 filled orders; 128 buys, 126 sells
Average holding days: not calculated

Main observations:
- 2019-09-30 `513500` now matches JoinQuant at the signal level: `sell=44 force=True`.
- 2020-03-02 `518880` now matches JoinQuant's ATR stop timing by comparing stop triggers at ETF 0.001 quote precision.
- 2020-03-03 `159985` now scores as `buy=70 trend=0 sell=0`, matching JoinQuant's ability to score new listings before MA60 is available.
- Current first remaining order-path divergence is 2020-09-22: JoinQuant buys `512100` with `buy=65 rev=35`; local scores it `buy=54 rev=24` because KDJ up-cross falls just outside the local 3-bar recent-cross window. This is not yet changed because it may reflect a broader cross-window definition issue.

Bad entries observed: not reviewed
Sell timing observations: 2020-03-02 `518880` ATR sell aligned; 2020-09-22 `512100` buy divergence remains open.
Abnormal logs/errors: none during the local replay run

Can this result be used to change rules? no
Reason: This is a local mechanics/log-alignment checkpoint. Remaining KDJ window divergence must be investigated separately before any rule change.

### KDJ Window Divergence Diagnostic

Version: no strategy change; diagnostic only
Code file: none
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: data-vs-algorithm diagnosis for remaining JoinQuant/local divergence
Initial capital: not applicable

Main observations:
- The 2020-09-22 `512100` divergence is not likely caused by bad OHLC data. Local RSI, MACD, KDJ values are nearly identical to the JoinQuant log.
- JoinQuant logged `512100` as `buy=65 rev=35` with `KDJ_K_UP=True` and `KDJ_J_UP=True`.
- Local replay scores the same date as `buy=54 rev=24` because KDJ up-cross occurred just outside the local 3-bar recent-cross window.
- If the local KDJ check uses a 4-bar recent-cross window, the 2020-09-22 KDJ flags match JoinQuant.
- Batch comparison found only 4 scored samples where a `+11` reversal-score gap is explained by KDJ window=4. This does not justify changing the global `cross_window` by itself because it would affect RSI, MACD, KDJ, buy and sell logic together.

Bad entries observed: not reviewed
Sell timing observations: not reviewed
Abnormal logs/errors: none

Can this result be used to change rules? no
Reason: This is diagnostic evidence only. A global cross-window change would be a strategy-rule change and needs broader evidence than one remaining order-path divergence.

### JoinQuant Versus Local Close Data Audit

Version: data-quality diagnostic only
Code file: `cross_signal_strategy/local_data_quality.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: determine whether remaining local/JoinQuant divergences are caused by bad local data
Initial capital: not applicable

Main observations:
- Parsed 3650 rich indicator rows from the JoinQuant training log.
- Only 2 rows had `abs(JoinQuant close - local close) > 0.002`.
- Outliers:
  - 2020-01-17 `510880`: JoinQuant close 2.803 vs local signal close 2.947, diff 0.144.
  - 2021-01-18 `510300`: JoinQuant close 5.454 vs local signal close 5.526, diff 0.072.
- The 2020-09-22 `512100` KDJ-window divergence is not caused by close-data mismatch: both JoinQuant and local close are 0.963, and local KDJ/MACD values match the JoinQuant log within rounding.
- The two close outliers occur around execution days where local minute `prev_close` differs sharply from the previous local daily close, which is consistent with ex-dividend/adjustment handling rather than random data corruption.

Bad entries observed: none identified as raw-data corruption
Sell timing observations: not reviewed
Abnormal logs/errors: none

Can this result be used to change rules? no
Reason: This is a data-quality and platform-data-parity audit. It supports further work on local adjustment/复权 alignment, not strategy parameter or signal tuning.
