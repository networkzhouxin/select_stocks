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
