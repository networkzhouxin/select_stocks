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

### Local ETF Adjustment Factor Audit

Version: local replay adjustment-factor alignment
Code files: `cross_signal_strategy/local_adjustment.py`, `cross_signal_strategy/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: fix local/JoinQuant data口径 parity only; no strategy rule or parameter tuning
Initial capital: 20000

Files inspected:
- `G:\financial\history_data\按年份合并\etf_adjust_factor\ETF复权因子.csv`
- `G:\financial\history_data\按年份合并\数据合并工具最新版本下载.txt`
- `G:\financial\history_data\按年份合并\分钟级\全部复权因子\涨跌幅\全部复权因子.zip`
- `G:\financial\history_data\按年份合并\分钟级\全部份额\全部份额.zip`
- `G:\financial\history_data\按年份合并\分钟级\etf.csv`

Main observations:
- `ETF复权因子.csv` directly explains the two close outliers from the JoinQuant/local audit.
- `510880` has an ex-date on 2020-01-17 with `ex_factor=1.0513740030198886`; local 2020-01-16 close `2.947 / 1.0513740030198886 = 2.8029987`, matching JoinQuant close `2.803`.
- `510300` has an ex-date on 2021-01-18 with `ex_factor=1.0132002506617996`; local 2021-01-15 close `5.526 / 1.0132002506617996 = 5.4540058`, matching JoinQuant close `5.454`.
- The minute-level `全部复权因子.zip` independently confirms factor jumps on those dates: `510880` factor `1.318 -> 1.386` on 2020-01-17; `510300` factor `1.131 -> 1.146` on 2021-01-18.
- `全部份额.zip` shows same-day NAV/close drops consistent with ex-dividend handling, useful as a sanity check but not the primary OHLC adjustment source.
- `etf.csv` is ETF metadata, useful for identity/listing/type checks, not for fixing price口径.
- `数据合并工具最新版本下载.txt` only contains a Baidu Netdisk tool link and is not directly useful for this anomaly.

Implementation:
- Added a small 2019-2021 target-ETF adjustment-factor table inside `local_adjustment.py` instead of reading the full `按年份合并` source during replay.
- `LocalSignalAdapter` can apply known adjustment events on or before the decision date. It divides historical OHLC rows before the ex-date by the product of known later ex-factors and does not adjust volume.
- `run_training_replay` enables these training-period adjustment factors by default.
- Future events are not applied before their ex-date.

Verification:
- `uvx --with pandas pytest tests/test_cross_signal_local_signal_adapter.py::test_signal_frame_applies_current_day_adjustment_without_future_events tests/test_cross_signal_local_signal_adapter.py::test_local_signal_adapter_can_align_ex_dividend_signal_close -q` -> 2 passed.
- `uvx --with pandas pytest tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_data_quality.py tests/test_cross_signal_local_data_loader.py -q` -> 17 passed.
- `uvx --with pandas pytest tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py -q` -> 13 passed.
- `uvx --with pandas pytest tests/test_cross_signal_local_training_run.py -q` -> 2 passed in 394.07s.
- Adjusted anomaly check: `510880` on 2020-01-17 scores close `2.8029987`; `510300` on 2021-01-18 scores close `5.4540058`.
- Local training replay after adjustment: start `2019-01-02`, end `2021-12-31`, total return `47.54%`, max drawdown `7.72%`, buys `130`, sells `128`, final holdings `159985` and `518880`.

Can this result be used to change strategy rules? no
Reason: This is a data parity fix. It reduces local/JoinQuant replay differences without looking at validation-period performance or changing trading logic.

### Filled Order Path Diagnostic

Version: local/JoinQuant filled-order path diagnostic
Code file: `cross_signal_strategy/order_path_diagnostics.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: identify remaining platform/local replay divergence using filled order events only
Initial capital: 20000

Main observations:
- A first-pass comparison against strategy intent logs (`[buy]`/`[sell]`) falsely reported 2019-12-12 `513880` as the first divergence.
- The JoinQuant log shows that 2019-12-12 `513880` sell was not filled: volume was 0, the market order was canceled, and the position remained in the 15:30 close log.
- The diagnostic now parses actual JoinQuant fill lines (`order StockOrder ... trade price ... amount ...`) separately from strategy intent logs.
- Against attachment `97b63eb6-be21-46a8-9e26-1acabe2cca7e`, JoinQuant filled events are 262 total: 132 buys and 130 sells.
- Local filled events after adjustment-factor alignment are 258 total: 130 buys and 128 sells.
- The first real filled-order path divergence is order index 127:
  - JoinQuant: 2020-09-22 BUY `512100`, amount 7700, price 0.954.
  - Local: next event is 2020-09-29 BUY `513880`, amount 6700, price 1.08108.
- The surrounding path is aligned through 2020-09-22 SELL `513880`; the missing `512100` buy remains the first actionable divergence.

Interpretation:
- This confirms that the remaining earliest divergence is the previously documented 2020-09-22 `512100` buy.
- The cause is still likely signal-window scoring, not data corruption: JoinQuant logs `512100` as `buy=65 rev=35`, while local previously scored it lower because the KDJ cross fell just outside the local 3-bar recent-cross window.
- This is diagnostic evidence only. Changing the cross-window would be a strategy-rule change and still needs a separate test-first rule decision.

Verification:
- `uvx --with pandas pytest tests/test_cross_signal_order_path_diagnostics.py -q` -> 6 passed.
- Full local replay comparison script completed and produced the filled-event counts and first divergence above.

Can this result be used to change strategy rules? no
Reason: This milestone adds repeatable diagnosis and filters out unfilled JoinQuant intent logs. It does not change scoring, thresholds, ranking, or execution behavior.

### Cross Flag Window Alignment Diagnostic

Version: JoinQuant/local cross-flag alignment diagnostic
Code file: `cross_signal_strategy/local_data_quality.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: determine whether the remaining 2020-09-22 `512100` buy divergence is caused by a broad cross-window mismatch
Initial capital: not applicable

Main observations:
- Parsed 2589 JoinQuant cross-flag rows from attachment `97b63eb6-be21-46a8-9e26-1acabe2cca7e`.
- With local `cross_window=3`, 2580 rows could be scored; only 32 rows had any cross-flag mismatch, with 40 flag mismatches total.
- With local `cross_window=4`, mismatch count worsened sharply to 478 rows and 931 flag mismatches.
- With local `cross_window=5`, mismatch count worsened further to 750 rows and 1559 flag mismatches.
- Therefore, the existing global `cross_window=3` is much closer to JoinQuant overall than `4` or `5`.

Flag mismatch breakdown for `cross_window=3`:
- `rsi6_cross_rsi24_up`: 22
- `macd_cross_up`: 4
- `rsi6_cross_rsi12_up`: 3
- `rsi6_cross_rsi24_down`: 3
- `rsi6_cross_rsi12_down`: 2
- `kdj_k_cross_up`: 2
- `kdj_j_cross_up`: 2
- `macd_cross_down`: 2
- KDJ down-cross flags had 0 mismatches.

Focus case:
- JoinQuant 2020-09-22 `512100`: `buy=65 rev=35`, `KDJ_K_UP=True`, `KDJ_J_UP=True`.
- Local `cross_window=3`: `buy=54 rev=24`, `KDJ_K_UP=False`, `KDJ_J_UP=False`.
- Local `cross_window=4` or `5`: matches the focus-case KDJ flags and score (`buy=65 rev=35`).

Interpretation:
- The focus case is a local KDJ recent-cross boundary mismatch, not a复权/data issue.
- A global `cross_window` change from 3 to 4 would fix this one case but break many more rows. It should not be adopted as a broad rule.
- The next step should be a narrower investigation of why JoinQuant logs only a small number of boundary differences while the code default remains `cross_window=3`: possible causes include exact diff-equality handling, duplicated/filled data rows, indicator warm-up length, or platform rounding/precision around zero-cross boundaries.

Can this result be used to change strategy rules? no
Reason: This is alignment evidence against a broad window change. It does not justify changing the strategy window or thresholds.

### Cross Flag Boundary Narrow Diagnostic

Version: KDJ/RSI/MACD recent-cross boundary diagnosis
Code file: `cross_signal_strategy/local_data_quality.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: determine whether a narrower window/precision fix can explain the remaining 2020-09-22 `512100` buy divergence
Initial capital: not applicable

Focus-case trace:
- For `512100` on 2020-09-22, the local signal date is 2020-09-21.
- Local K-D and J-D crossed above on 2020-09-16.
- Relative to the 2020-09-21 signal row, that is `offset=4`.
- Local `cross_window=3` therefore excludes the KDJ cross; local `cross_window=4` includes it.
- Local RSI6-RSI12 and RSI6-RSI24 crossed above at `offset=2`, so RSI flags already match JoinQuant under `cross_window=3`.
- MACD had no recent up-cross in the inspected 6-day trace.

JoinQuant log evidence:
- JoinQuant logs `512100` KDJ_UP as true on 2020-09-18, 2020-09-21, and 2020-09-22.
- On 2020-09-22, JoinQuant logs `KDJ_DIFF[K-D/J-D]=13.6/40.8(prev 9.8/29.4)`, so the current row itself is not the cross; the cross happened earlier.

Selective-window test:
- All flags with window 3: 32 mismatched rows, 40 flag mismatches.
- All flags with window 4: 478 mismatched rows, 931 flag mismatches.
- KDJ-only window 4 with all other flags window 3: 254 mismatched rows, 490 flag mismatches.

Interpretation:
- The 2020-09-22 `512100` divergence is a real boundary mismatch, but changing KDJ to a 4-day window is not a valid JoinQuant-alignment fix because it makes the full training log much less aligned.
- The evidence points away from a broad parameter/window change and toward a small set of platform/version/boundary quirks. Potential remaining causes include exact historical code version drift, data-row inclusion edge cases, or platform precision/rounding around a prior cross date.
- No strategy rule should be changed from this evidence alone.

Can this result be used to change strategy rules? no
Reason: The narrower KDJ-only change fails the global alignment check. This diagnostic supports leaving `cross_window=3` unchanged for now.

### Cross Flag Version And State Diagnostic

Version: cross-flag version/state diagnosis
Code file: `cross_signal_strategy/local_data_quality.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: check whether the 2020-09-22 `512100` divergence is caused by indicator value mismatch, an alternate cross-state interpretation, or version drift
Initial capital: not applicable

Focus-case indicator parity:
- For `512100` on 2020-09-21 and 2020-09-22, local RSI/MACD/KDJ values match JoinQuant logs within normal rounding.
- Example 2020-09-22:
  - JoinQuant close `0.963`; local close `0.963`.
  - JoinQuant K/D/J `60.1/46.5/87.3`; local `60.0783/46.4763/87.2824`.
  - JoinQuant KDJ_DIFF `13.6/40.8(prev 9.8/29.4)`; local `13.6020/40.8061(prev 9.8172/29.4517)`.
- Because the previous KDJ diff is already positive on 2020-09-22, the current row itself is not an up-cross under the current code. The JoinQuant flag therefore reflects a recent/persistent cross decision rather than an indicator value discrepancy.

Alternate cross-state checks:
- KDJ-only "any positive in recent window" fixes the 2020-09-22 focus flag but worsens full-log alignment to 2053 mismatched rows and 4151 flag mismatches.
- KDJ-only "latest positive state" fixes the focus flag but worsens full-log alignment to 807 mismatched rows and 1633 flag mismatches.
- Applying these state-style interpretations to all indicators is much worse.
- Therefore the focus-case behavior is not explained by a simple global "state instead of cross" implementation.

Path after filtering focus pair:
- If the JoinQuant 2020-09-22 BUY `512100` and 2020-09-23 SELL `512100` are removed from the comparison, the next filled-order divergence is:
  - JoinQuant: 2020-10-27 SELL `513050`, amount 3500, price 2.081.
  - Local: no 2020-10-27 `513050` sell; next event is 2020-10-29 BUY `159928`.

Interpretation:
- The first divergence remains the 2020-09-22 `512100` KDJ boundary case.
- The mismatch is not caused by local OHLC/indicator values and not fixed safely by broader cross-state rules.
- Remaining divergences should be investigated one by one as platform/version/mechanics alignment issues before changing any strategy rule.

Can this result be used to change strategy rules? no
Reason: It rules out several tempting broad changes. It does not provide a safe strategy-rule change.
