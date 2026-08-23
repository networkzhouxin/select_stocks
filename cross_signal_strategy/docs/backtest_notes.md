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

### Frozen Training Baseline After JoinQuant Path Alignment

Version: cross-v0.2.6 local replay baseline after data/path/execution alignment
Code files: `cross_signal_strategy/research/baseline_report.py`, `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: frozen training baseline for future structure experiments
Initial capital: 20000

Strategy return: +45.45% local replay / +50.07% JoinQuant reference
Annualized return: +13.34% local replay
Excess return: not calculated locally
Benchmark return: not calculated locally
Alpha: not calculated locally
Beta: not calculated locally
Sharpe: not calculated locally
Sortino: not calculated locally
Win rate: 42.31% by closed local trades
Daily win rate: not calculated locally
Profit/loss ratio: 1.9597
Max drawdown: 7.67%
Max drawdown period: not calculated locally
Trade count: 262 filled local orders; 132 buys, 130 sells; 130 closed trades
Average holding days: not calculated locally
Average exposure: 59.74%

ETF realized PnL by closed local trades:
- Negative contributors: `510880` -466.5, `510300` -377.9, `159920` -199.8, `513880` -31.3.
- Positive contributors: `159915` +4106.0, `513050` +2038.5, `518880` +1020.6, `513100` +899.9, `513500` +800.3, `159928` +481.5, `512100` +481.1, `159985` +302.3.

Main observations:
- The low-ish win rate is offset by a nearly 2:1 profit/loss ratio, so the strategy is not failing because every trade is random noise.
- Average exposure is only 59.74%, which means capital utilization is a major candidate for future improvement.
- Drag is concentrated in a small group of ETFs, especially `510880`, `510300`, and `159920`, while the largest positive contribution came from `159915`.
- The strongest contributors include cross-market/cross-asset ETFs (`513050`, `518880`, `513100`, `513500`), supporting the value of diversified ETF exposure even in this cross-signal framework.

Bad entries observed:
- Not yet classified; next diagnostic should inspect worst closed trades and whether they were false reversal entries, late entries, or stop/exit problems.

Sell timing observations:
- Not yet classified; next diagnostic should separate signal sells, ATR stops, and risk-tighten cases if possible.

Abnormal logs/errors:
- None in the aligned local replay. Known local-vs-JoinQuant performance gap remains attributable mainly to market-order execution-price modeling, not signal path.

Can this result be used to change rules? yes, as a training baseline only
Reason: This freezes the aligned 2019-2021 training baseline for future experiments. It can guide training-period structure diagnostics, but validation periods must remain unseen until rules are frozen.

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
Code file: `cross_signal_strategy/local/local_data_quality.py`
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
Code files: `cross_signal_strategy/local/local_adjustment.py`, `cross_signal_strategy/local/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`
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
- Added a small 2019-2021 target-ETF adjustment-factor table inside `local/local_adjustment.py` instead of reading the full `按年份合并` source during replay.
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
Code file: `cross_signal_strategy/research/order_path_diagnostics.py`
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
Code file: `cross_signal_strategy/local/local_data_quality.py`
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
Code file: `cross_signal_strategy/local/local_data_quality.py`
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
Code file: `cross_signal_strategy/local/local_data_quality.py`
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

### Falling-MA10 Sell-Structure Alignment

Version: local/JoinQuant falling-MA10 sell-score alignment
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: align local replay to the JoinQuant `cross-v0.2.6` training log without using validation-period results
Initial capital: 20000

Finding:
- The local price-decline requirement inside `close_below_falling_ma10` was too strict for the JoinQuant log.
- Before the fix, JoinQuant/local rich-row sell-score comparison had 294 mismatched rows; 282 were exactly `+10` where JoinQuant counted falling-MA10 risk and local did not.
- The next filled-order divergence after filtering the known 2020-09-22/2020-09-23 `512100` KDJ boundary pair was 2020-10-27 SELL `513050`.
- On that date, `513050` closed at 2.080 after 2.078, but remained below a clearly falling MA10 (`2.0959 -> 2.0945`). JoinQuant sold with `sell_score 45`; local scored 35 because the small rebound blocked `close_below_falling_ma10`.

Implementation:
- `close_below_falling_ma10` now matches the JoinQuant platform code exactly: `close < MA10` and current `MA10 < previous MA10`.
- It no longer requires the latest close to be below or equal to the prior close.
- No local-only floating tolerance is applied.

Verification:
- `uvx --with pandas pytest tests/test_cross_signal_strategy.py -q -k "below_falling_ma10"` -> 3 passed.
- `uvx --with pandas pytest tests/test_cross_signal_strategy.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_data_quality.py tests/test_cross_signal_order_path_diagnostics.py -q` -> 74 passed.
- Rich-row sell-score mismatches fell from 294 to 12.
- Filled-order path: JoinQuant 262 events, local 260 events. The unfiltered first mismatch remains 2020-09-22 BUY `512100` versus local 2020-09-29 BUY `513880`.
- After filtering only the known 2020-09-22 BUY `512100` and 2020-09-23 SELL `512100` boundary pair, the remaining JoinQuant 260 filled events match the local 260 filled events exactly.
- Local replay final value after this alignment check: 29207.02, total return +46.04%, 260 filled orders.
- Full source diff between local `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py` and the uploaded JoinQuant platform code returned zero diff lines after removing the local-only `1e-9` tolerance.

Interpretation:
- This is an alignment correction, not a validation-period optimization.
- The remaining material divergence is concentrated in the previously diagnosed 2020-09-22 `512100` KDJ boundary pair.

Can this result be used to change strategy rules? yes, narrowly
Reason: It corrects local replay to match the JoinQuant training log's falling-MA10 sell-structure口径 and removes a broad systematic local scoring mismatch. It does not justify parameter tuning or validation-period changes.
### Local Sub-Float Falling-MA10 Adapter Alignment

Version: local replay precision alignment for falling-MA10 structure
Code file: `cross_signal_strategy/local/local_signal_adapter.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: keep local replay aligned to JoinQuant platform behavior while leaving the JoinQuant strategy source byte-identical to the uploaded platform code
Initial capital: 20000

Finding:
- After removing the local-only `1e-9` tolerance from the JoinQuant strategy file, local replay again diverged at 2019-11-13.
- JoinQuant log: 2019-11-13 `159928` is `[risk-tighten] sell_score 24`, then buys `513100`.
- Local replay before this adapter fix: 2019-11-13 `159928` scored `sell_score 34` and sold.
- The cause was a local Pandas floating artifact: local `MA10_prev - MA10_latest = 0.000000000000000444`, which made exact `<` treat MA10 as falling even though the platform log did not.

Implementation:
- `smart_trade_joinquant_cross_signal_etf.py` remains exactly aligned to the uploaded JoinQuant platform code.
- `LocalSignalAdapter` suppresses `close_below_falling_ma10` only when the local MA10 decrease is a sub-float artifact: `0 < ma10[-2] - ma10[-1] < 1e-12`.
- This is a local replay adapter correction, not a strategy rule change.

Verification:
- Added failing test `test_signal_score_suppresses_sub_float_falling_ma10_artifact` before implementation.
- `uvx --with pandas pytest tests/test_cross_signal_local_signal_adapter.py -q` -> 10 passed.
- `uvx --with pandas pytest tests/test_cross_signal_strategy.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_data_quality.py tests/test_cross_signal_order_path_diagnostics.py -q` -> 75 passed.
- Full local replay against the latest JoinQuant log: JoinQuant 262 filled events, local 260; first unfiltered mismatch is again 2020-09-22 BUY `512100` vs local 2020-09-29 BUY `513880`.
- After filtering only the known 2020-09-22 BUY `512100` and 2020-09-23 SELL `512100` boundary pair, the remaining JoinQuant 260 events match the local 260 events exactly.
- Local replay final value: 29207.02, total return +46.04%, 260 filled orders.

Interpretation:
- The remaining open issue is concentrated in the already diagnosed 2020-09-22 `512100` KDJ boundary case.
- This adapter fix prevents local floating-point noise from creating a fake earlier path divergence.

Can this result be used to change strategy rules? no
Reason: It only corrects local replay precision. It does not change JoinQuant source strategy logic, parameters, thresholds, or validation behavior.

### 512100 KDJ Boundary Diagnosis

Version: 2020-09-22 `512100` KDJ boundary investigation
Code file: diagnostic only
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: determine whether the last JoinQuant/local path divergence can be fixed by a general local replay correction
Initial capital: not applicable

Finding:
- The remaining unfiltered path divergence is still 2020-09-22 BUY `512100` and 2020-09-23 SELL `512100`.
- Local and JoinQuant indicator values on 2020-09-21/2020-09-22 match normal displayed precision, so this is not a broad OHLC or formula mismatch.
- The boundary is the KDJ cross date. Local K-D/J-D crossed above on signal date 2020-09-16, which is offset 4 relative to the 2020-09-21 signal row used for 2020-09-22 trading, so `cross_window=3` excludes it.
- JoinQuant logs imply the prior KDJ diff on 2020-09-16 was effectively `-0.0/-0.0`, so its cross is treated as occurring on 2020-09-17, which is offset 3 and remains inside the 3-day window.
- Local detailed KDJ around the boundary:
  - 2020-09-15 K-D/J-D: `-1.0179 / -3.0538`
  - 2020-09-16 K-D/J-D: `+0.0349 / +0.1046`
  - 2020-09-17 K-D/J-D: `+1.7909 / +5.3727`
- JoinQuant 2020-09-18 log shows current KDJ_DIFF `1.8/5.3(prev -0.0/-0.0)`, consistent with a tiny sign difference around the 2020-09-16 boundary.

Rejected fixes:
- Truncating local indicator input to `lookback=120` did not fix the `512100` divergence and introduced a later path mismatch after filtering the focus pair. Rich-row mismatch count worsened to 33 rows.
- A KDJ deadband threshold also worsened global flag alignment. Example sweep:
  - `eps=0`: 37 mismatched rows, 53 flag mismatches, focus remains `buy=54 rev=24`.
  - `eps=0.05`: 50 mismatched rows, 73 flag mismatches, focus only partially fixes to `buy=60 rev=30`.
  - `eps=0.2`: 69 mismatched rows, 114 flag mismatches, focus reaches `buy=65 rev=35` but damages many more rows.
- Therefore neither a broad `lookback=120` adapter change nor a general KDJ deadband should be adopted.

Interpretation:
- This appears to be a narrow JoinQuant/local data precision boundary around one KDJ zero-cross, not a structural strategy or replay bug.
- The safest current stance is to keep the local replay adapter as-is and treat the 2020-09-22/2020-09-23 `512100` pair as a documented residual platform/data precision discrepancy.

Can this result be used to change strategy rules? no
Reason: All broad fixes tested so far worsen full training-log alignment. The evidence supports documenting the boundary case, not changing strategy logic or parameters.

### 512100 Local Daily Close Correction

Version: confirmed local daily-bar correction for `512100` on 2020-09-02
Code files: `cross_signal_strategy/local/local_adjustment.py`, `cross_signal_strategy/local/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: align local training replay data to JoinQuant and verified minute/software evidence without modifying read-only source CSVs
Initial capital: 20000

Finding:
- The final 2020-09-22/2020-09-23 `512100` divergence was caused by a bad local daily close, not by a strategy-rule issue.
- JoinQuant diagnostic log showed `512100` 2020-09-02 close `1.001`.
- Local 1-minute data for 2020-09-02 also aggregates to close `1.001` at 15:00.
- The isolated local daily CSV has 2020-09-02 close `1.000`.
- User independently checked trading software and confirmed the close was `1.001`.
- This 0.001 close difference propagated through KDJ, making local `K-D/J-D` on 2020-09-16 slightly positive while JoinQuant remained slightly negative. That shifted the KDJ up-cross one day earlier locally and excluded it from the 3-day cross window on 2020-09-22.

Implementation:
- Added `LocalDailyCorrections` and `default_training_daily_corrections()` with the confirmed correction: `512100`, `2020-09-02`, `close=1.001`.
- `LocalSignalAdapter` applies daily corrections to the visible T-1 signal frame before adjustment factors.
- `run_training_replay` now builds its adapter through `build_training_signal_adapter()`, which applies both confirmed daily corrections and training adjustment factors.
- The read-only source training data folder remains unchanged.
- The JoinQuant strategy source remains unchanged.

Verification:
- Added failing tests before implementation:
  - `test_signal_frame_applies_confirmed_daily_bar_correction_without_mutating_raw_data`
  - `test_training_signal_adapter_applies_confirmed_daily_corrections`
- Targeted tests passed after implementation.
- Full cross-signal test suite: `uvx --with pandas pytest tests/test_cross_signal_strategy.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_data_quality.py tests/test_cross_signal_order_path_diagnostics.py tests/test_cross_signal_local_data_loader.py tests/test_cross_signal_local_training_run.py -q` -> 85 passed.
- Full local replay versus the latest JoinQuant log:
  - JoinQuant filled events: 262
  - Local filled events: 262
  - First order-path divergence: none
  - Local replay final value: 29074.94, total return +45.37%
- The `512100` pair now appears in local order path on the same dates as JoinQuant: BUY 2020-09-22 and SELL 2020-09-23.

Interpretation:
- This is a local data-quality correction, not a strategy rule change, not parameter tuning, and not validation-period influence.
- The correct fix is an external read-time correction layer because project rules prohibit modifying or deleting the source training data.

Can this result be used to change strategy rules? no
Reason: The evidence identifies a local data defect and fixes replay alignment only. It does not support changing indicators, thresholds, windows, or execution rules.

### Local Execution Price Diagnostics

Version: transaction/log execution-field comparison
Code files: `cross_signal_strategy/research/order_path_diagnostics.py`, `cross_signal_strategy/local/local_backtester.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: explain remaining local/JoinQuant return gap after order path alignment
Initial capital: 20000

Finding:
- The JoinQuant transaction CSV parser is now available for exported transaction details, including filled amount, price, signed trade value, commission, and status.
- The original temporary CSV attachment was no longer available when the full diagnostic was rerun, so the execution-field comparison used the latest JoinQuant log's filled-order records.
- Order path remains fully aligned: JoinQuant filled events 262, local filled events 262, first divergence none.
- JoinQuant and local commissions are identical in the log-based comparison: `1310.0` versus `1310.0`.
- The remaining local/JoinQuant return gap is mainly execution-price and rolling share-quantity drift, not signal timing or commission.
- With the prior unrounded 0.1% local slippage model, local final value was `29074.94` (+45.37%).
- A no-slippage local diagnostic raised final value to `31030.00` (+55.15%), overshooting JoinQuant, so removing slippage is not the right alignment fix.
- Applying ETF tick precision to local slippage execution prices raises local final value slightly to `29090.70` (+45.45%) while preserving the aligned 262-event path.

Implementation:
- `parse_joinquant_transaction_csv()` parses JoinQuant exported transaction CSV files and ignores cancelled/unfilled rows by default.
- `parse_joinquant_filled_order_events()` now captures commission and signed trade value from JoinQuant logs.
- `compare_order_execution_fields()` reports per-order amount, price, commission, and signed-trade-value differences after path alignment.
- `LocalBroker` rounds slippage-adjusted execution prices to ETF tick precision (`0.001`).

Verification:
- Added failing tests before implementation for transaction CSV parsing, cancelled-row filtering, execution-field diffing, and tick-rounded local broker prices.
- Full cross-signal test suite: `uvx --with pandas pytest tests/test_cross_signal_strategy.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_data_quality.py tests/test_cross_signal_order_path_diagnostics.py tests/test_cross_signal_local_data_loader.py tests/test_cross_signal_local_training_run.py -q` -> 88 passed.

Interpretation:
- The local replay is now suitable for signal-path debugging and approximate execution diagnostics.
- It should not be treated as a penny-perfect substitute for JoinQuant performance, because JoinQuant's internal market-order matching price is not fully reproduced from the local 09:35 minute bar alone.
- Further forcing local prices to match JoinQuant transaction prices would turn the local broker into a replay of JoinQuant fills rather than an independent backtest model.

Can this result be used to change strategy rules? no
Reason: This is an execution-simulator alignment and diagnostic improvement only. It does not justify changing strategy indicators, thresholds, or parameters.

### Base Ratio Exposure Sweep

Version: cross-v0.2.6 local replay with unchanged signal rules and varied `base_ratio`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only structure experiment
Initial capital: 20000

Hypothesis:
- The frozen baseline's average exposure of 59.74% may be a capital-utilization bottleneck rather than a candidate-generation bottleneck.
- A broad increase in base portfolio usage should improve return without changing entry/exit signal timing, while drawdown should rise in a proportional and understandable way.

Position-count diagnostic at baseline `base_ratio=0.75`:
- Empty days: 36.
- 1-position days: 107.
- 2-position days: 144.
- Full 3-position days: 443.
- Interpretation: the strategy is full 3 holdings on 60.7% of training days, so low exposure mainly comes from the base-ratio cap.

Training-only sweep:
- `base_ratio=0.75`: return +45.45%, annualized +13.34%, max drawdown 7.67%, win rate 42.31%, P/L ratio 1.9597, average exposure 59.74%.
- `base_ratio=0.80`: return +49.27%, annualized +14.33%, max drawdown 8.07%, win rate 43.08%, P/L ratio 1.9579, average exposure 63.69%.
- `base_ratio=0.85`: return +53.19%, annualized +15.32%, max drawdown 8.44%, win rate 43.08%, P/L ratio 1.9567, average exposure 67.75%.
- `base_ratio=0.90`: return +57.87%, annualized +16.49%, max drawdown 8.85%, win rate 43.08%, P/L ratio 1.9622, average exposure 71.87%.
- `base_ratio=0.95`: return +62.30%, annualized +17.57%, max drawdown 9.20%, win rate 43.08%, P/L ratio 1.9615, average exposure 75.85%.

Main observations:
- Trade count and signal path were unchanged across the sweep, so this is a sizing-policy experiment, not an indicator or threshold fit.
- `0.90` reaches the initial training target of roughly 16%-17% annualized while leaving a 10% cash buffer.
- `0.95` has the best training-period return, but it is rejected for now because near-full exposure is more fragile and lacks enough out-of-sample evidence.

Can this result be used to change rules? yes, training-only sizing policy
Reason: The change is broad, explainable, and affects only capital usage after already selected signals. It still requires reserved-period validation after the rule set is frozen.

### Normal Signal Sell Minimum-Hold Sweep

Version: cross-v0.2.6 local replay with `base_ratio=0.90` and varied normal-signal minimum hold
Code files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local/local_order_planner.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only sell-noise structure experiment
Initial capital: 20000

Hypothesis:
- Daily cross-signal exits may react too quickly to short-term noise.
- A standard one-week minimum hold should block early normal `signal_sell` exits while keeping ATR stop-loss fully active.

Pre-experiment trade-reason diagnostic at `base_ratio=0.90`:
- ATR stop exits: 19 trades, win rate 68.42%, realized PnL +9670.9, P/L ratio 5.766, average hold 21.58 trading days.
- Normal signal exits: 111 trades, win rate 38.74%, realized PnL +1853.6, P/L ratio 1.186, average hold 11.66 trading days.
- Interpretation: ATR stop is not the weak link; normal signal exits are the noisy component.

Training-only coarse sweep:
- `min_signal_hold_days=0`: return +57.87%, annualized +16.49%, max drawdown 8.85%, win rate 43.08%, P/L ratio 1.9622, buys 132, sells 130, sell reasons `signal_sell=111`, `atr_stop=19`.
- `min_signal_hold_days=3`: return +70.73%, annualized +19.58%, max drawdown 9.08%, win rate 44.54%, P/L ratio 2.289, buys 122, sells 119, sell reasons `signal_sell=98`, `atr_stop=21`.
- `min_signal_hold_days=5`: return +98.34%, annualized +25.72%, max drawdown 8.94%, win rate 53.47%, P/L ratio 2.8625, buys 103, sells 101, sell reasons `signal_sell=75`, `atr_stop=26`.
- `min_signal_hold_days=7`: return +98.94%, annualized +25.85%, max drawdown 9.71%, win rate 57.89%, P/L ratio 2.882, buys 98, sells 95, sell reasons `signal_sell=65`, `atr_stop=30`.

Adopted:
- `min_signal_hold_days=5`.

Reason for choosing 5 instead of the highest-return 7:
- 5 trading days maps to a normal one-week minimum hold and is easier to justify before validation.
- 7 trading days adds only +0.60pp return in training but increases max drawdown by about +0.76pp versus 5 days.
- Choosing 5 avoids fitting to the single best training number while preserving the clear sell-noise improvement.

Can this result be used to change rules? yes, training-only sell structure
Reason: The rule is broad, standard, and supported by trade-reason diagnostics. It still requires reserved-period validation after the training rule set is frozen.

### Base Ratio Recheck After Sell-Noise Filter

Version: cross-signal local replay after `min_signal_hold_days=5`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only sizing-policy recheck
Initial capital: 20000

Training-only sweep:
- `base_ratio=0.85`: return +90.13%, annualized +23.96%, max drawdown 8.58%, win rate 53.47%, P/L ratio 2.8640, average exposure 70.32%.
- `base_ratio=0.90`: return +98.34%, annualized +25.72%, max drawdown 8.94%, win rate 53.47%, P/L ratio 2.8625, average exposure 74.46%.
- `base_ratio=0.95`: return +106.17%, annualized +27.36%, max drawdown 9.35%, win rate 53.47%, P/L ratio 2.8530, average exposure 78.56%.

Interpretation:
- The signal path was unchanged across the sweep: 103 buys and 101 sells.
- `0.95` is adopted as a broad practical cap with a 5% cash buffer.
- No fine-grained ratios above `0.95` were tested, to avoid fitting the training window.

Can this result be used to change rules? yes, training-only sizing policy
Reason: The change is broad and affects only capital usage after signals are selected. It still requires reserved-period validation after the rule set is frozen.

### Current Training Metrics With Local Risk Ratios

Version: cross-signal current local training mainline
Code file: `cross_signal_strategy/research/baseline_report.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only reporting checkpoint
Initial capital: 20000

Metric口径:
- Return, annualized return, max drawdown, exposure, trade win rate, and profit/loss ratio are computed from local replay results.
- Daily returns use each daily total value versus the previous daily total value; the first day uses initial cash as the prior value.
- Annualized volatility, Sharpe ratio, and Sortino ratio use `244` periods per year and zero risk-free return.
- Sharpe uses population standard deviation of daily returns.
- Sortino uses downside deviation from zero daily return.
- No alpha, beta, information ratio, or benchmark-relative excess metrics are reported locally until a benchmark equity curve is explicitly added.

Current training result:
- Strategy return: +106.17%.
- Annualized return: +27.36%.
- Max drawdown: 9.35%.
- Closed-trade win rate: 53.47%.
- Daily win rate: 53.70%.
- Profit/loss ratio: 2.8530.
- Annualized volatility: 13.29%.
- Sharpe ratio: 1.8866.
- Sortino ratio: 2.9313.
- Buy/sell count: 103 / 101.
- Average exposure: 78.56%.

Can this result be used to change rules? no
Reason: This section adds reporting metrics only. It should help compare future experiments, but it does not itself justify strategy changes.

### Training Segment Diagnostics

Version: cross-signal current local training mainline
Code file: diagnostic only
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only robustness diagnosis
Initial capital: 20000

Annual segments:
- 2019: +38.46%, max drawdown 4.61%, daily win rate 54.51%, Sharpe 2.664, Sortino 4.614.
- 2020: +47.40%, max drawdown 7.70%, daily win rate 54.32%, Sharpe 2.546, Sortino 3.940.
- 2021: +1.02%, max drawdown 9.35%, daily win rate 52.26%, Sharpe 0.148, Sortino 0.204.

Quarter segments:
- Best quarters: 2019Q1 +20.70%, 2020Q2 +16.50%, 2020Q3 +10.57%, 2020Q4 +9.45%.
- Weak quarters: 2021Q1 -0.03%, 2021Q3 -6.04%.
- Main weakness is concentrated in 2021Q3, not spread evenly across all years.

2021Q3 closed-trade diagnosis:
- 13 closed trades, win rate 38.46%, realized PnL -1579.5, P/L ratio 0.304.
- Main negative contributors by sell date: `518880` -659.4, `510300` -492.5, `513500` -386.4, `159915` -278.8.
- Trades bought during 2021Q3: 12 closed trades, win rate 33.33%, realized PnL -1445.1.
- In Q3 buys, no-volume-confirmation entries were especially weak: 5 trades, all losing, realized PnL -2000.9. However, global and conditional volume filters failed in full-period training, so this observation alone does not justify a rule.

Interpretation:
- Training performance is not purely one lucky month, but 2021 is a clear weak regime.
- Simple full-period filters aimed at the 2021Q3 weakness must be rejected unless they improve overall training robustness without sacrificing too much upside.

Can this result be used to change rules? diagnostic only
Reason: It identifies the weak training regime and informs experiments, but does not itself justify a strategy rule.

### Formal Entry-Score Trade Diagnostics

Version: cross-signal current local training mainline
Code file: `cross_signal_strategy/research/trade_diagnostics.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only attribution-tooling checkpoint
Initial capital: 20000

Purpose:
- Capture each buy's score snapshot at order-planning time.
- Attribute closed-trade PnL to the actual entry-day tags, not to a later refreshed `last_scores` map.

Verification:
- Added unit coverage proving score snapshots are frozen before later score mutation.
- Added unit coverage proving closed-trade diagnostics use the entry snapshot.
- Full cross-signal test suite passed: 98 tests.

Current 2021Q3 diagnostic using formal entry snapshots:
- Trades sold in 2021Q3: 13, win rate 38.46%, realized PnL -1579.5, P/L ratio 0.304.
- Trades bought in 2021Q3: 12, win rate 33.33%, realized PnL -1445.1, P/L ratio 0.363.
- Q3 buys with `volume_score=0`: 5 trades, all losing, realized PnL -2000.9.
- Q3 buys with `volume_score>0`: 7 trades, win rate 57.14%, realized PnL +555.8.

Can this result be used to change rules? diagnostic only
Reason: Formal attribution confirms the weak area, but previous volume-filter experiments failed at the full training-window level. The diagnostic should guide future hypotheses, not directly change rules.

### JoinQuant 513880 Sparse-Liquidity Cancellation Probe

Version: temporary JoinQuant probe `cross_signal_strategy/archive/probes/smart_trade_joinquant_cross_signal_etf_probe_513880.py`
Backtest period: 2019-01-01 to 2021-12-31
Protocol role: execution-liquidity diagnosis only

Purpose:
- Explain the two retained JoinQuant warnings on `2019-12-12` for `513880.XSHG`.
- Determine whether the canceled sell came from a paused ETF, a zero-volume day, a data defect, or an exact-minute matching issue.
- Keep the result as execution diagnostics only; do not use it to tune signal rules or thresholds.

Findings:
- JoinQuant reported `paused=False` at `2019-12-12 09:35`, `10:35`, and `14:50`.
- The sampled 1-minute bars at those three exact times all had `volume=0` and `money=0`.
- The full-day 1-minute summary showed `total_minutes=240`, `nonzero_minutes=26`, `total_volume=1405700.0`, `total_money=1539142.0`, `first_nonzero=2019-12-12 09:38:00`, and `last_nonzero=2019-12-12 14:57:00`.

Conclusion:
- The ETF was not considered paused by JoinQuant.
- The ETF was not zero-volume for the whole day.
- The `09:35` market sell was canceled because the exact matching minute had no tradable volume under JoinQuant's market-order model.

Policy:
- Treat this as an execution-liquidity risk, not a strategy signal bug.
- Keep the formal strategy logic unchanged.
- Keep state-sync protection for unfilled orders.
- Do not add a broad `volume == 0` rule without separate evidence, because that would mix execution-time microstructure into a daily signal framework.

Can this result be used to change rules? no
Reason: This probe only explains a known execution warning. It supports risk documentation and order-state safeguards, but it does not justify changing daily signal structure or training parameters.

### A-Share Zero-Volume Buy Half-Size Rule

Version: cross-signal after `base_ratio=0.95` and `min_signal_hold_days=5`
Code files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local/local_order_planner.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only structure experiment
Initial capital: 20000

Hypothesis:
- `volume_score=0` should not be used as a global filter, because QDII/cross-market and commodity ETF volume behaves differently.
- For A-share ETFs only, a buy signal without volume confirmation may deserve smaller initial risk because the reversal quality is weaker.

Training-only group test:
- Baseline: return +106.17%, annualized +27.36%, max drawdown 9.35%, Sharpe 1.887, Sortino 2.931, P/L ratio 2.853, average exposure 78.56%, 2021Q3 -5.32%.
- A-share scale `0.75`: return +107.86%, max drawdown 8.64%, Sharpe 1.944, Sortino 3.028, 2021Q3 -4.80%.
- A-share scale `0.50`: return +109.19%, annualized +27.98%, max drawdown 7.86%, Sharpe 1.995, Sortino 3.113, P/L ratio 3.134, average exposure 75.72%, 2021Q3 -4.25%.
- A-share scale `0.25`: return +110.60%, max drawdown 7.16%, Sharpe 2.042, Sortino 3.188, 2021Q3 -3.73%.
- Cross-market scales `0.75/0.50/0.25`: returns +98.69% / +91.02% / +84.07%, all below baseline.
- Cross-asset scales `0.75/0.50/0.25`: returns +104.22% / +101.53% / +98.69%, all below baseline.

Adopted:
- `a_share_zero_volume_buy_scale=0.50`.

Reason for choosing 0.50 instead of the highest-training 0.25:
- 0.50 is a standard half-risk rule and easier to justify as risk management.
- 0.25 is the best training number, but it is a stronger sizing cut and more likely to be a training-window fit.
- The goal is to reduce weak A-share no-volume reversal risk while preserving enough upside for validation.

Can this result be used to change rules? yes, training-only sizing structure
Reason: The rule is ETF-type-specific, uses a broad half-size control, improves return and drawdown in training, and rejects the global volume rule that previously failed. It still requires JoinQuant training confirmation and reserved-period validation after the rule set is frozen.

### JoinQuant Training Confirmation For A-Share Zero-Volume Half-Size Rule

Version: `cross-v0.3.0` after adopting `a_share_zero_volume_buy_scale=0.50`
Platform: JoinQuant
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: training-window authority confirmation

JoinQuant headline result:
- Strategy return: +115.41%.
- Annualized return: +30.06%.
- Excess return: +31.27%.
- Max drawdown: 6.93%.
- Sharpe ratio: 2.022.
- Sortino ratio: 2.870.
- Win rate: 0.530.
- Profit/loss ratio: 3.597.
- Alpha: 0.208.
- Beta: 0.362.
- Information ratio: 0.690.

Log and transaction checks:
- Strategy log contained 102 `[buy]` lines and 101 `[sell]` lines.
- Transaction export contained 203 rows: 102 buys and 101 sells.
- Transaction status: 202 fully filled rows, 1 canceled row.
- The only canceled row was the known `2019-12-12 513880.XSHG` zero-volume market sell; this is the previously documented sparse-liquidity execution risk, not a new strategy issue.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume cancellation.

Half-size rule evidence:
- A-share `volume_score=0` buys were sized at about half of the normal per-slot target. Examples: `2019-07-17 159915.XSHE target=4086` with filled value about `4020.3`; `2020-03-05 159928.XSHE target=4692` with filled value about `4671`; `2021-07-14 159915.XSHE target=6863` with filled value about `6770`; `2021-07-20 510300.XSHG target=6783` with filled value about `6718.4`.
- Non-A-share `volume_score=0` buys were not half-sized, matching the intended boundary for QDII/cross-market and cross-asset ETFs.

Comparison to prior JoinQuant training result:
- Previous JoinQuant training result before this rule: +112.22% return, +29.39% annualized return, 8.19% max drawdown.
- Current JoinQuant training result after this rule: +115.41% return, +30.06% annualized return, 6.93% max drawdown.

Can this result be used to change rules? yes, training-window authority confirmation
Reason: This confirms that the local training improvement also appears in JoinQuant, which remains the performance authority. Reserved validation periods were still not inspected; this is not validation approval.

### Post-Sell Follow-Through Diagnostics

Version: cross-signal after `a_share_zero_volume_buy_scale=0.50`
Code file: `cross_signal_strategy/research/sell_diagnostics.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only sell-side diagnosis
Initial capital: 20000

Purpose:
- Measure whether sells were followed by further weakness or rebound over 3, 5, 10, and 20 training trading days.
- Separate `atr_stop` from normal `signal_sell`, because they serve different jobs.
- Diagnose sell-fly risk before changing sell rules.

Training diagnostic summary:
- `atr_stop`: 26 available sells. Forward mean returns after sell were +0.59% at 3 days, +0.12% at 5 days, -2.78% at 10 days, and -2.63% at 20 days.
- `signal_sell`: 73 available 3/5-day observations, 69 available 10-day observations, and 66 available 20-day observations. Forward mean returns after sell were +0.45% at 3 days, +0.68% at 5 days, +1.06% at 10 days, and +1.24% at 20 days.
- Signal-sell positive follow-through rates were high: 68.5% at 3 days, 57.5% at 5 days, 65.2% at 10 days, and 68.2% at 20 days.

Interpretation:
- ATR stops are mostly doing risk-control work; the 10/20-day post-sell averages are negative.
- Normal signal sells do have sell-fly risk; sold ETFs often rebound afterward.
- However, sell-fly alone is not enough to justify weakening signal sells because signal sells also recycle capital into other candidates.

Can this result be used to change rules? diagnostic only
Reason: The diagnostic reveals a real weakness but does not include the portfolio opportunity cost of holding instead of rotating. Rule changes require full-path experiments.

### ETF Attribution Diagnostics

Version: cross-signal after `a_share_zero_volume_buy_scale=0.50`
Code file: `cross_signal_strategy/research/attribution_diagnostics.py`
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only attribution diagnosis
Initial capital: 20000

Purpose:
- Identify which ETFs are return engines and which ETFs are drag sources under the current rule set.
- Measure closed-trade PnL, win rate, profit/loss ratio, average holding days, ATR-stop rate, and signal-sell rate by ETF.
- Use this as evidence for cautious ETF-pool experiments, not as standalone proof to delete symbols.

Training attribution by realized PnL:
- `159915`: 9 trades, PnL +5935.9, win rate 66.7%, P/L 13.563, average hold 21.9 trading days.
- `513100`: 11 trades, PnL +5280.6, win rate 54.5%, P/L 7.232, average hold 19.5.
- `513050`: 5 trades, PnL +2942.9, win rate 60.0%, P/L 8.256, average hold 27.0.
- `513500`: 13 trades, PnL +2487.6, win rate 69.2%, P/L 4.435, average hold 19.8.
- `159928`: 15 trades, PnL +1666.4, win rate 46.7%, P/L 2.148, average hold 13.5.
- `513880`: 5 trades, PnL +1488.4, win rate 60.0%, P/L 9.409, average hold 23.4.
- `518880`: 11 trades, PnL +1252.5, win rate 72.7%, P/L 2.234, average hold 20.2.
- `159985`: 9 trades, PnL +883.6, win rate 33.3%, P/L 1.485, average hold 13.9.
- `512100`: 7 trades, PnL +573.8, win rate 57.1%, P/L 1.589, average hold 15.4.
- `510300`: 6 trades, PnL +265.1, win rate 50.0%, P/L 1.318, average hold 11.2.
- `159920`: 4 trades, PnL -190.4, win rate 25.0%, P/L 0.694, average hold 13.2.
- `510880`: 6 trades, PnL -822.0, win rate 16.7%, P/L 0.038, average hold 10.0.

Interpretation:
- Main engines are `159915`, `513100`, `513050`, and `513500`.
- `510880` and `159920` are the clearest drag symbols in the 2019-2021 training window.
- `510300` is weak but not strongly negative; removing it is a larger structural choice because it is the benchmark-like A-share core.

ETF-pool training experiments:
- Baseline pool: return +109.19%, annualized +27.98%, max drawdown 7.86%, Sharpe 1.995, Sortino 3.113.
- Remove `510880`: return +105.88%, max drawdown 8.56%, Sharpe 1.939. Rejected.
- Remove `159920`: return +111.13%, max drawdown 7.40%, Sharpe 2.029. Candidate but small improvement.
- Remove `510880` and `159920`: return +108.13%, max drawdown 8.03%, Sharpe 1.978. Rejected.
- Remove `510300`, `510880`, and `159920`: return +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201, 100 buys and 97 sells. Candidate only.

JoinQuant candidate file:
- `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_pool_candidate.py` was created as a temporary training-confirmation file.
- It only changes `STRATEGY_VERSION` and removes `510300.XSHG`, `510880.XSHG`, and `159920.XSHE` from `get_default_etf_pool()`.
- It is not the official adopted strategy until JoinQuant 2019-2021 training confirms the improvement and logs/transactions are reviewed.

Can this result be used to change rules? candidate only
Reason: ETF-pool deletion is highly exposed to training-window selection bias. The `510300/510880/159920` removal candidate improves local training return and drawdown, but it should be confirmed in JoinQuant training before adoption and must later face reserved validation.

### JoinQuant Training Confirmation For ETF-Pool Candidate

Version: `cross-v0.3.0-pool-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_pool_candidate.py`
Platform: JoinQuant
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: training-window authority confirmation

Candidate pool change:
- Remove `510300.XSHG`, `510880.XSHG`, and `159920.XSHE` from the current cross-signal pool.
- Keep `159915.XSHE`, `512100.XSHG`, `159928.XSHE`, `513100.XSHG`, `513500.XSHG`, `513880.XSHG`, `513050.XSHG`, `518880.XSHG`, and `159985.XSHE`.

JoinQuant headline result:
- Strategy return: +120.42%.
- Annualized return: +31.08%.
- Excess return: +34.32%.
- Benchmark return: +64.10%.
- Max drawdown: 6.82%.
- Sharpe ratio: 2.097.
- Sortino ratio: 2.960.
- Win rate: 0.552.
- Profit/loss ratio: 4.263.
- Alpha: 0.220.
- Beta: 0.348.
- Information ratio: 0.736.

Log and transaction checks:
- Strategy log contained 99 `[buy]` lines and 97 `[sell]` lines.
- Transaction export contained 196 rows: 99 buys and 97 sells.
- Transaction status: 195 fully filled rows, 1 canceled row.
- The only canceled row was `2019-12-12 513880.XSHG` with a `-0` share sell at `09:35`, matching the known sparse-liquidity execution issue.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume market-order cancellation.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.

Comparison to current official JoinQuant training result:
- Current official `cross-v0.3.0` after A-share zero-volume half-size rule: +115.41% return, +30.06% annualized return, 6.93% max drawdown, Sharpe 2.022, Sortino 2.870, win rate 0.530, profit/loss ratio 3.597.
- Candidate `cross-v0.3.0-pool-candidate`: +120.42% return, +31.08% annualized return, 6.82% max drawdown, Sharpe 2.097, Sortino 2.960, win rate 0.552, profit/loss ratio 4.263.

Interpretation:
- JoinQuant confirms the local direction: removing `510300`, `510880`, and `159920` improves return, drawdown, Sharpe, win rate, and profit/loss ratio in the 2019-2021 training window.
- This is still an ETF-pool deletion selected from training-period attribution, so it has higher selection-bias risk than a pure risk-control rule.
- The candidate is eligible to be promoted into the official cross-signal training mainline, but it must be clearly marked as training-confirmed and later tested on reserved validation windows after the rule set is frozen.

Can this result be used to change rules? yes, training-window authority confirmation
Reason: JoinQuant confirmed that the candidate pool improves the current official training result and the transaction/log path contains no unexpected removed-symbol or runtime anomalies. Reserved validation periods were still not inspected; this is not validation approval.

### JoinQuant Confirmation For Official v0.3.1 Mainline

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: official training-mainline confirmation

JoinQuant headline result:
- Strategy return: +120.42%.
- Annualized return: +31.08%.
- Excess return: +34.32%.
- Benchmark return: +64.10%.
- Max drawdown: 6.82%.
- Sharpe ratio: 2.097.
- Sortino ratio: 2.960.
- Win rate: 0.552.
- Profit/loss ratio: 4.263.
- Alpha: 0.220.
- Beta: 0.348.
- Information ratio: 0.736.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 99 `[buy]` lines and 97 `[sell]` lines.
- Transaction export contained 196 rows: 99 buys and 97 sells.
- Transaction status: 195 fully filled rows, 1 canceled row.
- The only canceled row was `2019-12-12 513880.XSHG` with a `-0` share sell at `09:35`, matching the known sparse-liquidity execution issue.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume market-order cancellation.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

Interpretation:
- The official mainline now reproduces the training-confirmed candidate result exactly at the headline metric level and transaction-count level.
- The `v0.3.1` mainline is the current 2019-2021 training-period safety point.
- This still does not permit validation-period tuning; reserved validation periods remain unseen for rule-selection purposes.

Can this result be used to change rules? already adopted, confirmation only
Reason: This run confirms the official strategy file matches the previously confirmed candidate behavior. It adds operational confidence but does not introduce a new rule or parameter.

### v0.3.1 Training Robustness Check

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Backtest period: 2019-01-01 to 2021-12-31
Protocol role: training-only robustness diagnosis
Data boundary: local diagnostics used only the approved 2019-2021 training data root and 2018 warm-up buffer.

Purpose:
- Check whether the strong JoinQuant training result is structurally healthy before adding more indicators or changing parameters.
- Split the result by year, ETF, signal flags, sell reason, transaction friction, and max-drawdown interval.
- This is not validation; reserved validation periods remain unseen.

Local replay headline result:
- Local replay return: +113.44%.
- Annualized return: +28.84%.
- Max drawdown: 6.94%.
- Sharpe ratio: 2.049.
- Sortino ratio: 3.201.
- Buy/sell count: 100 / 97.
- Closed-trade win rate: 54.64%.
- Profit/loss ratio: 3.619.
- Average exposure: 73.55%.
- Position-count days: 0 positions 39 days, 1 position 94 days, 2 positions 163 days, 3 positions 434 days.

Yearly decomposition from local replay:
- 2019: +35.68%, max drawdown 4.67%, 29 buys / 26 sells.
- 2020: +48.00%, max drawdown 6.94%, 32 buys / 32 sells.
- 2021: +6.29%, max drawdown 6.50%, 39 buys / 39 sells.

Interpretation:
- The result is not supported by a single year only. 2019 and 2020 both contribute strongly.
- 2021 is the weak year: positive but much flatter, with more churn. Future training-only work should inspect 2021-style sideways/noisy markets, but must avoid tuning to a single weak subperiod.

ETF attribution from local closed trades:
- `513100`: 15 trades, PnL +5829.2, win rate 53.33%, P/L 6.183, average hold 17.67 days.
- `159915`: 10 trades, PnL +5616.2, win rate 60.00%, P/L 9.844, average hold 20.20 days.
- `159928`: 16 trades, PnL +3222.9, win rate 50.00%, P/L 5.125, average hold 14.00 days.
- `513500`: 14 trades, PnL +2387.8, win rate 71.43%, P/L 4.575, average hold 18.79 days.
- `513050`: 6 trades, PnL +2317.3, win rate 50.00%, P/L 3.510, average hold 23.17 days.
- `518880`: 10 trades, PnL +1354.2, win rate 80.00%, P/L 2.590, average hold 21.40 days.
- `513880`: 6 trades, PnL +1249.4, win rate 50.00%, P/L 4.291, average hold 19.67 days.
- `159985`: 10 trades, PnL +573.4, win rate 30.00%, P/L 1.277, average hold 12.90 days.
- `512100`: 10 trades, PnL +277.4, win rate 40.00%, P/L 1.216, average hold 14.10 days.

Interpretation:
- The top return engines are diversified: Nasdaq, ChiNext, consumer, S&P 500, and China internet all contribute.
- Gold and Nikkei provide smaller but positive diversification.
- Soymeal and CSI 1000 are weak but still positive in the current 9-ETF pool. They should not be removed based only on this diagnostic because the pool has already been selected from training attribution and further deletion would raise overfitting risk.

Entry-signal flag attribution:
- RSI6 up through RSI12 appeared in all 97 closed trades and is the core cross trigger.
- KDJ K/J up appeared in 87 trades, PnL +22792.7, win rate 55.17%.
- RSI6 up through RSI24 appeared in 95 trades, PnL +22469.1, win rate 54.74%.
- Near-MA20 location appeared in 96 trades, PnL +22249.0, win rate 54.17%.
- Non-negative MA20 slope appeared in 73 trades, PnL +21866.3, win rate 58.90%.
- MA5 > MA10 appeared in 51 trades, PnL +20650.6, win rate 60.78%.
- Close > MA60 appeared in 85 trades, PnL +18882.1, win rate 52.94%.
- MA10 > MA20 appeared in 63 trades, PnL +16141.3, win rate 60.32%.
- BOLL midline upward cross appeared in 58 trades, PnL +13218.3, win rate 56.90%.
- Volume flags appeared in about half the trades and were positive: `vol5_gt_vol20` 52 trades, PnL +12513.6, win rate 59.62%; `volume_above_vol20_and_up` 55 trades, PnL +11814.1, win rate 58.18%.
- MACD upward cross appeared in 29 trades, PnL +6177.6, win rate 44.83%.
- Low-BOLL location (`close_between_boll_lower_mid`) appeared in 12 trades, PnL +1003.4, win rate 33.33%.

Interpretation:
- The strategy is still genuinely cross-signal driven: RSI/KDJ/MA-location/trend confirmation form the main structure.
- Volume is helpful as confirmation but not universal; earlier hard volume gates failed, so this supports keeping volume as a soft component rather than a hard filter.
- MACD and lower-BOLL entries are weaker standalone, but they are not enough evidence for removal because indicator interactions and portfolio opportunity cost matter.

Sell-reason attribution:
- ATR stops: 28 closed trades, PnL +14181.2, win rate 60.71%.
- Signal sells: 69 closed trades, PnL +8646.6, win rate 52.17%.

Interpretation:
- ATR stops are not merely loss exits; they also lock in profitable trend trades.
- Signal sells remain positive after the 5-day minimum hold and should not be removed globally.

Friction sensitivity from local replay:
- Base local friction: +113.44% return, +28.84% annualized return, 6.94% max drawdown, Sharpe 2.049.
- Double commission and double slippage (`commission_rate=0.0006`, `slippage_rate=0.002`): +100.07% return, +26.09% annualized return, 7.27% max drawdown, Sharpe 1.873.
- Stress friction (`commission_rate=0.0010`, `slippage_rate=0.003`): +85.21% return, +22.87% annualized return, 7.79% max drawdown, Sharpe 1.656.

Interpretation:
- The strategy is friction-sensitive but does not collapse under materially heavier assumed costs.
- Trade count is moderate rather than high-frequency, so transaction friction is not the only source of training profitability.

Max-drawdown interval:
- Local max drawdown interval: 2020-02-13 to 2020-03-19, drawdown 6.94%.
- This matches the JoinQuant max-drawdown interval directionally and corresponds to the COVID crash/rebound period.
- Key local orders in the drawdown interval: ATR stops on `513100` at 2020-02-17, `513500` at 2020-02-19, and `518880` at 2020-03-02; new buys in `159985`, `159928`, and `513050`; then ATR/signal exits around 2020-03-10 to 2020-03-12.

Interpretation:
- The worst drawdown is not caused by a single stale position. It comes from rapid crash-period cross-asset repricing where new reversal entries were attempted before the market fully stabilized.
- This suggests the next research direction should be crash/regime robustness, not arbitrary indicator threshold tuning.

Recommendation:
- Treat `cross-v0.3.1` as the current training-period safety point.
- Do not delete more ETFs or tune narrow thresholds solely from these diagnostics.
- Reasonable next training-only experiments, if continuing optimization before validation: broad crash-regime risk control, entry pacing after clustered ATR stops, or a coarse market-volatility state filter. These must be tested as structural rules and recorded, not optimized by fine thresholds.

Can this result be used to change rules? diagnostic only
Reason: The diagnostics identify strengths and weaknesses but do not itself test a new rule. Any follow-up change must be implemented as a separate test-first experiment.

### Portfolio ATR-Stress Buy-Scale Candidate

Version: `cross-v0.3.1` local experimental replay
Candidate file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Backtest period: 2019-01-01 to 2021-12-31
Protocol role: training-only structural risk experiment

Hypothesis:
- The max-drawdown interval showed clustered ATR stops during a crash-like regime.
- If the whole portfolio has recently produced multiple ATR stops, new buy signals may still be valid but should initially use reduced size until the stress cluster fades.
- This is a broad crash/regime rule, not a fine indicator threshold.

Rule tested:
- Count portfolio-level ATR stops over a recent trading-day lookback.
- If there are enough recent ATR stops, multiply all new-buy target values by a coarse scale.
- Existing positions and sell rules are unchanged.

Training sweep:
- Baseline: +113.44% return, +28.84% annualized return, 6.94% max drawdown, Sharpe 2.049, Sortino 3.201.
- 5-day lookback, 2 stops, scale 0.50: +103.52%, 6.64% max drawdown, Sharpe 2.002.
- 5-day lookback, 2 stops, scale 0.75: +108.55%, 6.79% max drawdown, Sharpe 2.032.
- 5-day lookback, 3 stops: identical to baseline.
- 10-day lookback, 2 stops, scale 0.50: +102.12%, 6.29% max drawdown, Sharpe 2.000.
- 10-day lookback, 2 stops, scale 0.75: +107.93%, 6.62% max drawdown, Sharpe 2.033.
- 10-day lookback, 3 stops, scale 0.50: +113.21%, 6.64% max drawdown, Sharpe 2.055.
- 10-day lookback, 3 stops, scale 0.75: +113.31%, 6.79% max drawdown, Sharpe 2.053.
- 15-day lookback, 2 stops, scale 0.50: +105.78%, 5.95% max drawdown, Sharpe 2.076.
- 15-day lookback, 2 stops, scale 0.75: +109.50%, 6.11% max drawdown, Sharpe 2.071.
- 15-day lookback, 3 stops, scale 0.50: +117.10%, 6.53% max drawdown, Sharpe 2.111, Sortino 3.319.
- 15-day lookback, 3 stops, scale 0.75: +115.06%, 6.51% max drawdown, Sharpe 2.082.

Candidate selected for JoinQuant training confirmation:
- `portfolio_atr_stress_lookback_days=15`
- `portfolio_atr_stress_min_stops=3`
- `portfolio_atr_stress_buy_scale=0.50`

Reason:
- It is the best local training balance among the coarse structural candidates: higher return, lower drawdown, higher Sharpe and Sortino than the `v0.3.1` local baseline.
- The parameters are broad and interpretable: about three trading weeks, three portfolio-level stop events, and a half-size risk-control response.
- The rule only affects new buys under clustered stop stress; it does not change indicator scores, sell thresholds, ATR stop math, or ETF pool.

Overfitting risk:
- Medium. This candidate was selected from a small training-only sweep, and it directly addresses the known 2020 drawdown interval.
- It must be confirmed in JoinQuant over 2019-2021 before any adoption and later tested on reserved validation windows after the rule set is frozen.

Can this result be used to change rules? candidate only
Reason: Local training replay supports the structure, but JoinQuant remains the performance authority. The candidate file is ready for JoinQuant 2019-2021 confirmation; it is not adopted into the official mainline.

### JoinQuant Training Confirmation For ATR-Stress Candidate

Version: `cross-v0.3.1-atr-stress-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Platform: JoinQuant
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: training-window candidate confirmation

JoinQuant headline result:
- Strategy return: +122.47%.
- Annualized return: +31.50%.
- Excess return: +35.57%.
- Benchmark return: +64.10%.
- Max drawdown: 6.38%.
- Sharpe ratio: 2.160.
- Sortino ratio: 3.057.
- Win rate: 0.552.
- Profit/loss ratio: 4.466.
- Alpha: 0.225.
- Beta: 0.342.
- Information ratio: 0.759.

Comparison to official `cross-v0.3.1` JoinQuant training result:
- Official `cross-v0.3.1`: +120.42% return, +31.08% annualized return, 6.82% max drawdown, Sharpe 2.097, Sortino 2.960, win rate 0.552, profit/loss ratio 4.263.
- ATR-stress candidate: +122.47% return, +31.50% annualized return, 6.38% max drawdown, Sharpe 2.160, Sortino 3.057, win rate 0.552, profit/loss ratio 4.466.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1-atr-stress-candidate]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 99 `[buy]` lines and 97 `[sell]` lines.
- Transaction export contained 196 rows: 99 buys and 97 sells.
- Transaction status: 195 fully filled rows, 1 canceled row.
- The only canceled row was `2019-12-12 513880.XSHG` with a `-0` share sell at `09:35`, matching the known sparse-liquidity execution issue.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume market-order cancellation.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

ATR-stress trigger audit:
- Buy logs with explicit stress field: 99.
- `stress=1.00`: 95 buys.
- `stress=0.50`: 4 buys.
- Triggered buys:
  - 2020-03-03 `159985.XSHE`, target 4578, transaction value 4559.0.
  - 2020-03-05 `159928.XSHE`, target 2288, transaction value 2179.8.
  - 2020-03-06 `513050.XSHG`, target 4583, transaction value 4519.8.
  - 2020-03-16 `159985.XSHE`, target 4513, transaction value 4448.2.

Interpretation:
- JoinQuant confirms that the ATR-stress candidate improves the official training result across return, annualized return, max drawdown, Sharpe, Sortino, and profit/loss ratio.
- The improvement comes from only four half-size buys, all clustered in March 2020 during the COVID crash/rebound regime.
- This concentration makes the rule more explainable as crash-regime risk control, but also materially raises overfitting risk because it mostly repairs one known training-window stress episode.

Can this result be used to change rules? candidate only, do not adopt yet
Reason: The rule has a professional risk-control rationale and JoinQuant confirms the training improvement, but the trigger audit shows only four clustered training events. Adoption requires reserved-period validation after the rule is explicitly frozen, and should be judged primarily on drawdown control and non-collapse rather than small extra return.

### First Reserved Validation: Official v0.3.1 Mainline

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant
Validation period: 2022-01-01 to 2023-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: first reserved weak/sideways validation

Important protocol note:
- This is the first inspected reserved validation result after freezing `docs/validation_protocol.md`.
- This result must not be used to tune thresholds, add indicators, remove ETFs, or search for a new validation-fitting variant.

JoinQuant headline result:
- Strategy return: +15.49%.
- Annualized return: +7.72%.
- Excess return: +66.30%.
- Benchmark return: -30.55%.
- Max drawdown: 13.38%.
- Max drawdown interval: 2022-02-24 to 2022-11-22.
- Sharpe ratio: 0.346.
- Sortino ratio: 0.499.
- Win rate: 0.385.
- Profit/loss ratio: 1.490.
- Alpha: 0.075.
- Beta: 0.180.
- Information ratio: 1.423.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 68 `[buy]` lines and 65 `[sell]` lines.
- Transaction export contained 133 rows: 68 buys and 65 sells.
- Transaction status: 133 fully filled rows, 0 canceled/rejected rows.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 0.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

Interpretation:
- The official mainline passes the first validation window operationally: no runtime errors, no warnings, no canceled transactions, and no removed-symbol leakage.
- Performance is positive in an adverse benchmark period: +15.49% strategy return versus -30.55% benchmark return.
- Risk-adjusted metrics are weaker than training, as expected in a difficult unseen regime: Sharpe 0.346, Sortino 0.499, win rate 38.5%.
- The max drawdown of 13.38% is materially higher than the 2019-2021 training drawdown but still avoided benchmark-like collapse.

Pass/fail/hold judgment:
- Official `cross-v0.3.1` passes first reserved validation as a robust baseline, not as a finished production-ready strategy.
- The result supports keeping `v0.3.1` as the official cross-signal baseline for further reserved-period comparison.
- No rule changes are allowed from this result alone.

Next allowed action:
- Run the frozen ATR-stress candidate over the same 2022-01-01 to 2023-12-31 validation window and compare under the pre-written validation protocol.

### First Reserved Validation: ATR-Stress Candidate

Version: `cross-v0.3.1-atr-stress-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Platform: JoinQuant
Validation period: 2022-01-01 to 2023-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: first reserved weak/sideways validation for risk-control candidate

Important protocol note:
- This result is compared only against the already recorded official `cross-v0.3.1` validation result.
- No threshold, ETF-pool, or indicator change may be made from this validation result.

JoinQuant headline result:
- Strategy return: +16.01%.
- Annualized return: +7.97%.
- Excess return: +67.03%.
- Benchmark return: -30.55%.
- Max drawdown: 12.94%.
- Max drawdown interval: 2022-02-24 to 2022-11-22.
- Sharpe ratio: 0.373.
- Sortino ratio: 0.536.
- Win rate: 0.385.
- Profit/loss ratio: 1.512.
- Alpha: 0.077.
- Beta: 0.175.
- Information ratio: 1.437.

Comparison to official `cross-v0.3.1` first validation:
- Official `cross-v0.3.1`: +15.49% return, +7.72% annualized return, 13.38% max drawdown, Sharpe 0.346, Sortino 0.499, win rate 0.385, profit/loss ratio 1.490.
- ATR-stress candidate: +16.01% return, +7.97% annualized return, 12.94% max drawdown, Sharpe 0.373, Sortino 0.536, win rate 0.385, profit/loss ratio 1.512.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1-atr-stress-candidate]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 68 `[buy]` lines and 65 `[sell]` lines.
- Transaction export contained 133 rows: 68 buys and 65 sells.
- Transaction status: 133 fully filled rows, 0 canceled/rejected rows.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 0.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

ATR-stress trigger audit:
- Buy logs with explicit stress field: 68.
- `stress=1.00`: 65 buys.
- `stress=0.50`: 3 buys.
- Triggered buys:
  - 2022-05-13 `518880.XSHG`, target 3015, transaction value 2722.3.
  - 2022-05-17 `159985.XSHE`, target 3012, transaction value 2925.0.
  - 2022-05-18 `513880.XSHG`, target 3011, transaction value 2940.6.

Interpretation:
- The candidate improves all key comparison metrics slightly versus official `v0.3.1`: return, annualized return, max drawdown, Sharpe, Sortino, and profit/loss ratio.
- The validation improvement is small but directionally aligned with the training result.
- The rule triggered 3 times in validation, clustered in May 2022, so it was not inactive. However, the improvement still depends on a small number of clustered events.

Pass/fail/hold judgment:
- ATR-stress candidate passes the first reserved validation comparison as a live candidate.
- It should not yet be merged into the official mainline because trigger count remains small and the rule is still exposed to event-cluster overfitting.
- It should proceed to the next reserved validation window as a frozen candidate alongside official `v0.3.1`.

Next allowed action:
- Run official `v0.3.1` and ATR-stress candidate over 2024-01-01 to the latest available date, using the same frozen protocol.

### Second Reserved Validation: Official v0.3.1 Mainline

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant
Validation period: 2024-01-01 to 2026-07-08
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: recent-market reserved validation

Important protocol note:
- This result is recorded after the first 2022-2023 validation pair.
- This result must not be used to tune thresholds, add indicators, remove ETFs, or search for a new validation-fitting variant.

JoinQuant headline result:
- Strategy return: +56.99%.
- Annualized return: +20.41%.
- Excess return: +13.27%.
- Benchmark return: +38.60%.
- Max drawdown: 10.65%.
- Max drawdown interval: 2025-11-03 to 2026-07-08.
- Sharpe ratio: 1.276.
- Sortino ratio: 1.800.
- Win rate: 0.506.
- Profit/loss ratio: 2.786.
- Alpha: 0.132.
- Beta: 0.313.
- Information ratio: 0.352.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 78 `[buy]` lines and 77 `[sell]` lines.
- Transaction export contained 155 rows: 78 buys and 77 sells.
- Transaction status: 155 fully filled rows, 0 canceled/rejected rows.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 0.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

Interpretation:
- Official `v0.3.1` passes the recent-market validation operationally and financially.
- Performance is meaningfully positive in a rising but volatile recent market: +56.99% strategy return versus +38.60% benchmark return.
- Risk-adjusted metrics are stronger than the 2022-2023 validation window and remain weaker than the 2019-2021 training window, which is a reasonable out-of-sample pattern.
- The max drawdown of 10.65% is lower than the 2022-2023 validation drawdown and within the observed cross-signal risk range.

Pass/fail/hold judgment:
- Official `cross-v0.3.1` passes the second reserved validation window.
- The official mainline now has two positive reserved validation periods after the 2019-2021 training window.
- No rule changes are allowed from this result alone.

Next allowed action:
- Complete the same transaction-level record for the ATR-stress candidate over 2024-01-01 to 2026-07-08, then compare under the frozen validation protocol.

### Second Reserved Validation: ATR-Stress Candidate

Version: `cross-v0.3.1-atr-stress-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Platform: JoinQuant
Validation period: 2024-01-01 to 2026-07-08
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: recent-market reserved validation for risk-control candidate

Important protocol note:
- This result is compared only against the already recorded official `cross-v0.3.1` recent-market validation result.
- No threshold, ETF-pool, or indicator change may be made from this validation result.

JoinQuant headline result:
- Strategy return: +56.99%.
- Annualized return: +20.41%.
- Excess return: +13.27%.
- Benchmark return: +38.60%.
- Max drawdown: 10.65%.
- Max drawdown interval: 2025-11-03 to 2026-07-08.
- Sharpe ratio: 1.276.
- Sortino ratio: 1.800.
- Win rate: 0.506.
- Profit/loss ratio: 2.786.
- Alpha: 0.132.
- Beta: 0.313.
- Information ratio: 0.352.

Comparison to official `cross-v0.3.1` recent-market validation:
- Official `cross-v0.3.1`: +56.99% return, +20.41% annualized return, 10.65% max drawdown, Sharpe 1.276, Sortino 1.800, win rate 0.506, profit/loss ratio 2.786.
- ATR-stress candidate: identical headline metrics.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1-atr-stress-candidate]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 78 `[buy]` lines and 77 `[sell]` lines.
- Transaction export contained 155 rows: 78 buys and 77 sells.
- Transaction status: 155 fully filled rows, 0 canceled/rejected rows.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 0.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Expected 9-symbol pool check: all expected symbols traded at least once; no unexpected symbols appeared in the transaction export.

ATR-stress trigger audit:
- Buy logs with explicit stress field: 78.
- `stress=1.00`: 78 buys.
- `stress=0.50`: 0 buys.

Interpretation:
- The ATR-stress candidate produced the identical recent-market path as official `v0.3.1` because the stress rule did not trigger.
- This is favorable for side-effect control: the rule did not suppress upside in the recent rising/volatile market.
- This result does not add new evidence that the rule improves drawdown under stress, because it was inactive in this validation window.

Pass/fail/hold judgment:
- ATR-stress candidate passes the second reserved validation as harmless/inactive.
- Across validation so far, it improved 2022-2023 slightly and was inactive in 2024-2026.
- It remains a viable risk-control candidate, but final adoption should still wait for the next reserved stress window because the rule has low trigger count.

Next allowed action:
- Run official `v0.3.1` and ATR-stress candidate over the stress validation window 2015-01-01 to 2018-12-31, using the same frozen protocol.

### Stress Reserved Validation: Official v0.3.1 Mainline

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant
Validation period: 2015-01-01 to 2018-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: stress reserved validation

Important protocol note:
- This stress window was reserved before inspection.
- This result must not be used to tune thresholds, add indicators, remove ETFs, or search for a new validation-fitting variant.

JoinQuant headline result:
- Strategy return: +23.58%.
- Annualized return: +5.58%.
- Excess return: +45.05%.
- Benchmark return: -14.80%.
- Max drawdown: 7.49%.
- Max drawdown interval: 2016-07-29 to 2016-11-09.
- Sharpe ratio: 0.192.
- Sortino ratio: 0.256.
- Win rate: 0.443.
- Profit/loss ratio: 1.660.
- Alpha: 0.023.
- Beta: 0.092.
- Information ratio: 0.393.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 123 `[buy]` lines and 124 `[sell]` lines.
- Shared transaction export contained 247 rows: 123 buys and 124 sells.
- Transaction status: 245 fully filled rows and 2 canceled rows.
- The 2 canceled rows were `159928.XSHE` zero-share sell cancellations on 2016-08-03 and 2017-03-09; both were followed by normal sells on the next trading day.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 4, matching the two zero-volume/canceled-order events above.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.
- Traded expected symbols in this older window: `159928`, `159915`, `518880`, `513500`, `513100`, `512100`, and `513050`. `513880` and `159985` did not trade in this window.

Interpretation:
- Official `v0.3.1` passes the stress validation operationally: no runtime errors, no removed-symbol leakage, and warnings are explained by real zero-volume canceled sells.
- The strategy produced positive absolute return while the benchmark was negative, which supports cross-asset defensive behavior in the 2015-2018 market regime.
- Risk-adjusted metrics are weaker than later windows, but the max drawdown remained controlled at 7.49%.

Pass/fail/hold judgment:
- Official `cross-v0.3.1` passes the stress reserved validation as a robust baseline.
- No rule changes are allowed from this result alone.

### Stress Reserved Validation: ATR-Stress Candidate

Version: `cross-v0.3.1-atr-stress-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Platform: JoinQuant
Validation period: 2015-01-01 to 2018-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: stress reserved validation for risk-control candidate

Important protocol note:
- This result is compared only against the already recorded official `cross-v0.3.1` stress validation result.
- No threshold, ETF-pool, or indicator change may be made from this validation result.

JoinQuant headline result:
- Strategy return: +23.58%.
- Annualized return: +5.58%.
- Excess return: +45.05%.
- Benchmark return: -14.80%.
- Max drawdown: 7.49%.
- Max drawdown interval: 2016-07-29 to 2016-11-09.
- Sharpe ratio: 0.192.
- Sortino ratio: 0.256.
- Win rate: 0.443.
- Profit/loss ratio: 1.660.
- Alpha: 0.023.
- Beta: 0.092.
- Information ratio: 0.393.

Comparison to official `cross-v0.3.1` stress validation:
- Official `cross-v0.3.1`: +23.58% return, +5.58% annualized return, 7.49% max drawdown, Sharpe 0.192, Sortino 0.256, win rate 0.443, profit/loss ratio 1.660.
- ATR-stress candidate: identical headline metrics.

Log and transaction checks:
- Strategy log initialized as `[cross-v0.3.1-atr-stress-candidate]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 123 `[buy]` lines and 124 `[sell]` lines.
- The official and candidate logs are not byte-identical because the version label differs and the candidate buy logs include the `stress` field.
- Parsed buy/sell event sequence is identical between official and candidate logs.
- Shared transaction export contained 247 rows: 123 buys and 124 sells.
- Transaction status: 245 fully filled rows and 2 canceled rows.
- The 2 canceled rows were `159928.XSHE` zero-share sell cancellations on 2016-08-03 and 2017-03-09; both were followed by normal sells on the next trading day.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 4, matching the two zero-volume/canceled-order events above.
- Removed symbols check: 0 buy logs, 0 sell logs, and 0 transaction rows for `510300`, `510880`, or `159920`.

ATR-stress trigger audit:
- Buy logs with explicit stress field: 123.
- `stress=1.00`: 123 buys.
- `stress=0.50`: 0 buys.

Interpretation:
- The ATR-stress candidate produced the identical stress-window trading path as official `v0.3.1` because the stress rule did not trigger.
- This is favorable for side-effect control: the rule did not harm the 2015-2018 stress window.
- This result does not add evidence that the rule improves this older stress window, because it was inactive.
- One shared transaction export is sufficient evidence for both stress-window runs because the parsed buy/sell event sequence is identical and the transaction row counts, side counts, code distribution, and canceled-order events match the logs.

Pass/fail/hold judgment:
- ATR-stress candidate passes the stress reserved validation as harmless/inactive.
- Across recorded windows, the candidate improved training and 2022-2023 slightly, was inactive in 2024-2026, and was inactive in 2015-2018.
- It remains a viable risk-control candidate, but the low trigger count means final adoption should be decided only after completing the early out-of-sample supplement and summarizing all frozen evidence.

Next allowed action:
- Run official `v0.3.1` and ATR-stress candidate over the early out-of-sample supplement 2010-01-01 to 2014-12-31, then prepare a frozen evidence summary.

### Early Out-Of-Sample Supplement: Official v0.3.1 Mainline

Version: `cross-v0.3.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant
Validation period: 2010-01-01 to 2014-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: early out-of-sample supplement with incomplete ETF pool

Important protocol note:
- This window is not a normal full-pool validation window. Many ETFs in the current pool were not listed or not tradable for much of the period.
- This result must not be used to tune thresholds, add indicators, remove ETFs, or search for a new validation-fitting variant.

JoinQuant headline result:
- Strategy return: -0.61%.
- Annualized return: -0.13%.
- Excess return: +0.57%.
- Benchmark return: -1.17%.
- Max drawdown: 5.36%.
- Max drawdown interval: 2014-02-24 to 2014-05-16.
- Sharpe ratio: -0.822.
- Sortino ratio: -0.709.
- Win rate: 0.349.
- Profit/loss ratio: 1.075.
- Alpha: -0.039.
- Beta: 0.057.
- Information ratio: 0.006.

Log checks:
- Strategy log initialized as `[cross-v0.3.1]` with `max_hold=3`, `base_ratio=0.95`, and `min_signal_hold=5`.
- Strategy log contained 44 `[buy]` lines and 43 `[sell]` lines.
- Log errors: `ERROR=0`, `Traceback=0`, `Exception=0`.
- Warnings: 0.
- Removed symbols check: 0 buy logs and 0 sell logs for `510300`, `510880`, or `159920`.
- First buy log: 2012-03-12 `159915.XSHE`.
- Last buy log: 2014-12-23 `513500.XSHG`.
- First sell log: 2012-03-23 `159915.XSHE`.
- Last sell log: 2014-12-31 `159928.XSHE`.
- Actual traded symbols from the log: `159915`, `159928`, `513100`, `513500`, and `518880`.

ETF-availability observation:
- The log had long early stretches with skip summaries such as `paused=9`, `paused=8`, and `paused=4`.
- This is consistent with an early window where many current-pool ETFs were not yet listed or not usable for signal scoring.

Interpretation:
- Official `v0.3.1` did not collapse operationally in the early supplement: no runtime errors, no warnings, and no removed-symbol trades.
- Financial performance was flat/slightly negative, but benchmark-relative return was slightly positive.
- The result mainly proves that the strategy can sit mostly idle and avoid severe damage when the ETF pool is incomplete. It does not prove the complete-pool strategy is weak.

Can this result be used to change rules? no
Reason: This is an early supplement with incomplete ETF availability. It is useful for robustness context only, not for parameter, indicator, or pool decisions.

### Early Out-Of-Sample Supplement: ATR-Stress Candidate

Version: `cross-v0.3.1-atr-stress-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Platform: JoinQuant
Validation period: 2010-01-01 to 2014-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: early out-of-sample supplement with incomplete ETF pool

Important protocol note:
- The user provided the JoinQuant summary screenshot and confirmed it is identical to the official mainline summary.
- A separate candidate transaction export was not required because the candidate only changes new-buy sizing when ATR-stress triggers; identical headline summary in this sparse early window is treated as same-path evidence unless contradicted by later transaction details.

JoinQuant headline result:
- Strategy return: -0.61%.
- Annualized return: -0.13%.
- Excess return: +0.57%.
- Benchmark return: -1.17%.
- Max drawdown: 5.36%.
- Max drawdown interval: 2014-02-24 to 2014-05-16.
- Sharpe ratio: -0.822.
- Sortino ratio: -0.709.
- Win rate: 0.349.
- Profit/loss ratio: 1.075.
- Alpha: -0.039.
- Beta: 0.057.
- Information ratio: 0.006.

Comparison to official `cross-v0.3.1` early supplement:
- Headline metrics are identical to the official mainline.
- The most likely explanation is that the ATR-stress rule did not trigger in this sparse early window.

Interpretation:
- ATR-stress did not provide additional benefit in 2010-2014, but it also did not harm the result.
- Because many pool ETFs were unavailable, this window should not be used as a strong argument for or against the candidate.

Can this result be used to change rules? no
Reason: This is an early supplement and the candidate was effectively inactive. It supports side-effect control, not parameter or rule tuning.

### Frozen Cross-Period Summary

Version: official `cross-v0.3.1` and `cross-v0.3.1-atr-stress-candidate`
Code files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
Protocol role: frozen evidence summary after training and reserved validation windows

Summary table:
- 2019-2021 training: official +120.42%, 6.82% max drawdown, Sharpe 2.097; ATR-stress +122.47%, 6.38% max drawdown, Sharpe 2.160.
- 2022-2023 validation: official +15.49%, 13.38% max drawdown, Sharpe 0.346; ATR-stress +16.01%, 12.94% max drawdown, Sharpe 0.373.
- 2024-2026 validation: official +56.99%, 10.65% max drawdown, Sharpe 1.276; ATR-stress identical and inactive.
- 2015-2018 stress validation: official +23.58%, 7.49% max drawdown, Sharpe 0.192; ATR-stress identical and inactive.
- 2010-2014 early supplement: official -0.61%, 5.36% max drawdown, Sharpe -0.822; ATR-stress identical by summary and effectively inactive.

Recommendation:
- Keep official `cross-v0.3.1` as the cross-signal baseline.
- Keep ATR-stress as a valid low-frequency risk-control candidate, but do not automatically merge it from the current evidence alone. It improved the windows where it triggered and did not harm inactive windows, but the trigger count is small and clustered.
- Do not tune or add indicators from validation results. Any next improvement should start as a new 2019-2021 training-only experiment with a pre-written hypothesis.

Detailed summary:
- See `cross_signal_strategy/docs/validation_summary.md`.

### Training-Only Iteration: Confirmation And Sell-Timing Probes

Version: `cross-v0.3.1`
Code path: local replay only; no official strategy code changed
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: new training-only research cycle after frozen validation summary

Baseline local replay:
- Return: +113.44%.
- Annualized return: +28.84%.
- Max drawdown: 6.94%.
- Sharpe: 2.049.
- Sortino: 3.201.
- Buy/sell count: 100 buys and 97 sells.
- Average exposure: 73.6%.

Training diagnostics:
- `trend_score` and `volume_score` are useful context, but not sufficient standalone decision drivers.
- Normal `signal_sell` exits are often followed by positive 3/5/10-day returns, but global sell-delay rules still damage capital recycling.
- ATR stops remain strong contributors and should not be weakened casually.

Experiments tested and rejected:
- Weak-confirmation buy-size cuts reduced return without improving drawdown enough.
- Confirmation-first, trend-first, volume/trend-first, and quality-sum buy ranking all underperformed the current buy-score ranking.
- Extending normal signal-sell minimum hold from 5 to 7/10/15 days underperformed.
- Raising `sell_threshold` from 30 to 35/40/45 underperformed.

Interpretation:
- The current buy-score ranking is more load-bearing than simple post-hoc attribution suggests.
- The next promising research direction is not global confirmation gating or global sell-delay. It should focus on a targeted sell-fly detector or a broader structural redesign that preserves capital recycling.

Can this result be used to change rules? no
Reason: All tested variants failed on the training window. They are recorded to prevent repeated overfitting searches.

### Training-Only Iteration: Sell-Fly Diagnostics And ATR Multiplier Probe

Version: `cross-v0.3.1`
Code path: local replay diagnostics plus temporary ATR-2.0 JoinQuant candidate file
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only research cycle; no validation-period influence

Diagnostic tooling added:
- `trade_diagnostics` now records exit-score snapshots for closed trades.
- `sell_diagnostics` can measure sell-fly events by forward return after signal sells and summarize them by exit-time features.
- Tests were added before implementation.

Sell-fly diagnostic observations:
- There were 69 local `signal_sell` exits in the training replay.
- At 5 trading days after signal sells, 68 had forward data, 14 met the `>=3%` sell-fly diagnostic threshold, average forward return was +0.70%, and estimated missed upside on flagged cases was about 7511.6 local currency units.
- Sell-fly was more common when exit-time `volume_score` and trend context remained constructive, but simple protection rules based on those features underperformed.

Rejected sell-fly protection variants:
- Exit-time volume/trend/buy-score protections all reduced return and Sharpe.
- Replacement-aware protection when no eligible replacement existed was close but not compelling: +113.93% return versus +113.44% baseline, same 6.94% max drawdown, but lower Sharpe 2.026 versus 2.049 and far fewer trades.

ATR multiplier probe:
- `trailing_atr_mult=2.0`: +115.87% return, +29.33% annualized return, 6.97% max drawdown, Sharpe 2.076, Sortino 3.249.
- `trailing_atr_mult=2.5` baseline: +113.44% return, +28.84% annualized return, 6.94% max drawdown, Sharpe 2.049, Sortino 3.201.
- `trailing_atr_mult=3.0`: +113.27% return, 7.67% max drawdown, Sharpe 2.041.
- `trailing_atr_mult=3.5`: +96.60% return, 9.23% max drawdown, Sharpe 1.824.
- `trailing_atr_mult=1.5` produced the same local path as 2.0 because the stop-floor clamp still dominated many stop calculations.

Candidate created:
- `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr2_candidate.py`
- Version: `cross-v0.3.1-atr2-candidate`
- Only intended strategy change versus official mainline: `trailing_atr_mult=2.0` instead of `2.5`.

Interpretation:
- Global sell-fly protection is not ready. It is too easy to protect weak positions and damage capital recycling.
- ATR 2.0 is the first new training-only candidate from this cycle with a small but coherent local improvement. The rationale is broad and professional: slightly tighter trailing protection after an up-cross entry may preserve gains without changing buy signals.
- The improvement is modest and must be confirmed in JoinQuant 2019-2021 before any adoption or reserved validation.

Can this result be used to change rules? candidate only
Reason: Local training replay supports preparing a JoinQuant training candidate, but JoinQuant remains the authority. Do not merge ATR 2.0 into the official mainline unless JoinQuant training confirms the improvement.

### Training-Only Iteration: Pool And Entry-Quality Structure Scan

Version: `cross-v0.3.1`
Code path: local replay with cached training scores; no validation-period data used
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only structural scan after ATR-2.0 rejection

Baseline local replay:
- Return +113.44%, annualized +28.84%, max drawdown 6.94%, Sharpe 2.049, Sortino 3.201.
- Buy count 100, sell count 97, average exposure 73.6%, empty days 39, full-position days 434.

ETF attribution observations:
- Strongest realized contributors: `513100` +5829.2, `159915` +5616.2, `159928` +3222.9, `513500` +2387.8, `513050` +2317.3.
- Weakest realized contributors: `159985` +573.4 and `512100` +277.4.
- `159985` looked weak as a standalone realized contributor, but deleting it reduced local return and worsened drawdown, which suggests it may still provide useful path diversification.

Variant scan:
- Remove `159985`: +108.97% return, 9.64% max drawdown, Sharpe 2.017.
- Remove `512100`: +114.57% return, 6.99% max drawdown, Sharpe 2.094.
- Remove both `159985` and `512100`: +109.74% return, 8.42% max drawdown, Sharpe 2.073.
- Require `buy_score >= 70`: +85.05% return, 11.56% max drawdown, Sharpe 1.661.
- Require `buy_score >= 80`: +51.41% return, 5.17% max drawdown, Sharpe 1.593.
- Remove `location_score == 17` buys: +88.02% return, 7.31% max drawdown, Sharpe 1.808.
- Require `trend_score >= 8`: +108.39% return, 7.32% max drawdown, Sharpe 1.984.
- Require `trend_score >= 9`: +88.31% return, 7.38% max drawdown, Sharpe 1.794.
- Composite entry-quality filter: +50.37% return, 10.13% max drawdown, Sharpe 1.230.

Candidate created:
- `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py`
- Version: `cross-v0.3.1-no-512100-candidate`
- Only intended strategy change versus official mainline: remove `512100.XSHG` from `get_default_etf_pool()`.

Interpretation:
- Broad entry-quality filters were harmful because they reduced capital utilization and blocked useful rebound entries.
- Simple training-attribution deletion is dangerous. `159985` is the clearest warning: it looks weak in realized PnL but worsens the path when removed.
- Removing only `512100` is a small, low-complexity candidate because it modestly improves local return and Sharpe while simplifying A-share exposure. The local edge is small and must be confirmed in JoinQuant 2019-2021 before any adoption.

Can this result be used to change rules? candidate only
Reason: Local training replay supports preparing a JoinQuant training candidate, but the effect is modest and pool deletion has selection-bias risk. Do not merge into official mainline unless JoinQuant training confirms a meaningful improvement.

### JoinQuant Training Check: No-512100 Pool Candidate

Version: `cross-v0.3.1-no-512100-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py`
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: JoinQuant training authority check for local pool candidate

JoinQuant headline result:
- Strategy return: +119.31%.
- Annualized return: +30.86%.
- Excess return: +33.65%.
- Benchmark return: +64.10%.
- Alpha: 0.221.
- Beta: 0.330.
- Sharpe: 2.977.
- Sortino: 0.711.
- Max drawdown: 6.82%.
- Win rate: 0.548.
- Profit/loss ratio: 4.399.
- Trades shown by JoinQuant summary: 51 wins, 42 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-no-512100-candidate`.
- ERROR-level log count: 0.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume market-order cancellation.

Comparison with official mainline:
- Official `cross-v0.3.1` training return was +122.47%, annualized +31.50%, max drawdown 6.38%, Sharpe 3.057, Sortino 0.759, win rate 0.552, profit/loss ratio 4.466.
- The no-512100 candidate is worse on return, annualized return, max drawdown, Sharpe, Sortino, win rate, and profit/loss ratio.

Interpretation:
- The local replay edge did not survive the JoinQuant training authority check.
- This reinforces the previous pool-design caution: deleting a weak-looking ETF from one attribution view can damage the actual JoinQuant path.

Can this result be used to change rules? no
Reason: The candidate failed on the training authority check. Do not run validation for this candidate and do not remove `512100.XSHG` from the official mainline.

### Training-Only Iteration: Entry Combo Attribution And MACD-RSI Filter Candidate

Version: `cross-v0.3.1`
Code path: local replay diagnostics plus temporary combo-filter JoinQuant candidate file
Backtest period: 2019-01-02 to 2021-12-31
Protocol role: training-only research cycle; no validation-period influence

Diagnostic tooling added:
- `attribution_diagnostics` can label buy-entry score snapshots into stable signal-source combo keys.
- Combo labels currently include RSI up-cross, MACD up-cross, KDJ up-cross, low-location entry, trend support/strong trend, and volume confirmation.
- Tests were added before implementation.

Entry-combo attribution observations:
- Closed local training trades: 97.
- Distinct entry combos: 14.
- Strongest combos by realized PnL:
  - `kdj_up+rsi_up+strong_trend+volume_confirmed`: n=10, PnL +5477.9, win rate 0.700, profit/loss ratio 9.583.
  - `kdj_up+macd_up+rsi_up+strong_trend+volume_confirmed`: n=4, PnL +4610.1, win rate 1.000.
  - `kdj_up+rsi_up+strong_trend`: n=6, PnL +4552.8, win rate 0.667, profit/loss ratio 13.258.
- Weakest combos by realized PnL:
  - `macd_up+rsi_up+trend_support+volume_confirmed`: n=5, PnL -1415.1, win rate 0.200, profit/loss ratio 0.060.
  - `kdj_up+macd_up+rsi_up+trend_support`: n=4, PnL -666.9, win rate 0.250, profit/loss ratio 0.044.
  - `kdj_up+low_location+rsi_up+trend_support+volume_confirmed`: n=8, PnL -431.6, win rate 0.250, profit/loss ratio 0.712.

Local pressure tests:
- Baseline: +113.44% return, +28.84% annualized, 6.94% max drawdown, Sharpe 2.049, Sortino 3.201, 100 buys, average exposure 0.736.
- Block only `macd_up+rsi_up+trend_support+volume_confirmed`: +118.75% return, +29.90% annualized, 6.81% max drawdown, Sharpe 2.117, Sortino 3.327, 99 buys, average exposure 0.732.
- Block three negative-looking combos together: +116.34% return, +29.42% annualized, 7.19% max drawdown, Sharpe 2.097, Sortino 3.298, 94 buys, average exposure 0.704.

Candidate created:
- `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
- Version: `cross-v0.3.1-combo-candidate`
- Only intended strategy change versus official mainline: skip new buy entries when RSI and MACD are up, volume is confirmed, trend is supportive but not strong, and KDJ has not crossed up.

Interpretation:
- KDJ appears to be an important timing confirmation for this cross-signal framework. RSI+MACD without KDJ in a merely supportive trend looks like a weak rebound/chop pattern in the training replay.
- Expanding the block list was worse than the single narrow filter, so the candidate intentionally remains minimal.
- This is still a selection-bias-prone training-only candidate because it is derived from entry-combo attribution. JoinQuant 2019-2021 must confirm it before any merge or validation.

Can this result be used to change rules? candidate only
Reason: Local training replay supports preparing a JoinQuant training candidate, but JoinQuant remains the authority. Do not merge into official mainline unless JoinQuant training confirms improvement against `cross-v0.3.1`.

### JoinQuant Training Check: Entry Combo Filter Candidate

Version: `cross-v0.3.1-combo-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
Backtest period: 2019-01-01 to 2021-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: JoinQuant training authority check for local entry-combo candidate

JoinQuant headline result:
- Strategy return: +125.82%.
- Annualized return: +32.18%.
- Excess return: +37.61%.
- Benchmark return: +64.10%.
- Alpha: 0.232.
- Beta: 0.347.
- Sharpe: 3.109.
- Sortino: 0.799.
- Max drawdown: 6.70%.
- Win rate: 0.558.
- Profit/loss ratio: 4.845.
- Trades shown by JoinQuant summary: 53 wins, 42 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-combo-candidate`.
- ERROR-level log count: 0.
- Warnings: 2, both from the known `2019-12-12 513880.XSHG` zero-volume market-order cancellation.

Comparison with official mainline:
- Official `cross-v0.3.1` training return was +122.47%, annualized +31.50%, max drawdown 6.38%, Sharpe 3.057, Sortino 0.759, win rate 0.552, profit/loss ratio 4.466.
- The combo candidate improves return, annualized return, Sharpe, Sortino, win rate, and profit/loss ratio.
- The only headline deterioration is max drawdown, from 6.38% to 6.70%.

Interpretation:
- The local entry-combo hypothesis survived the JoinQuant training authority check.
- The improvement is coherent but modest, and the candidate was derived from training-period attribution, so overfitting risk remains.
- Freeze this rule before validation. Do not use validation results to retune the combo filter.

Can this result be used to change rules? candidate ready for validation
Reason: JoinQuant training confirms the local candidate, but validation periods must now be run unchanged before any mainline merge.

### JoinQuant Validation Check: Entry Combo Filter Candidate 2022-2023

Version: `cross-v0.3.1-combo-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
Backtest period: 2022-01-01 to 2023-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: frozen-rule out-of-sample validation; no tuning allowed from this result

JoinQuant headline result:
- Strategy return: +17.36%.
- Annualized return: +8.62%.
- Excess return: +68.99%.
- Benchmark return: -30.55%.
- Alpha: 0.084.
- Beta: 0.178.
- Sharpe: 0.432.
- Sortino: 0.622.
- Max drawdown: 11.63%.
- Win rate: 0.385.
- Profit/loss ratio: 1.560.
- Trades shown by JoinQuant summary: 25 wins, 40 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-combo-candidate`.
- ERROR-level log count: 0.
- WARNING-level log count: 0.

Comparison with official mainline:
- Official `cross-v0.3.1` 2022-2023 result was +15.49% return, +7.72% annualized, 13.38% max drawdown, Sharpe 0.346, Sortino 0.499, win rate 0.385, profit/loss ratio 1.490.
- The combo candidate improves return, annualized return, max drawdown, Sharpe, Sortino, and profit/loss ratio.
- Win rate is unchanged at 0.385.

Interpretation:
- This is the first frozen-rule validation period, and it supports the entry-combo filter.
- Improvement is especially meaningful because 2022-2023 is a weak/sideways period where the strategy is meant to avoid bad rebound entries.
- Do not adjust or expand the combo filter from this result. Continue with the next reserved validation periods unchanged.

Can this result be used to change rules? no direct rule change
Reason: This is validation evidence for the already-frozen candidate. It supports continuing validation, not retuning.

### Sell Confirmation Candidate Prepared: Raise Normal Signal Sell Threshold To 35

Version: `cross-v0.3.2-sell35-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_sell35_candidate.py`
Backtest period for next required check: 2019-01-01 to 2021-12-31 only
Initial capital for next required check: 20000
Execution schedule: daily `09:35`
Protocol role: training-only candidate; validation windows must not be inspected unless this candidate is frozen after JoinQuant training confirmation

Hypothesis:
- Post-sell diagnostics on the 2019-2021 JoinQuant training log suggest that normal signal sells around `sell_score 32-34` may be too weak as forced exits.
- ATR stops remain productive and must stay unconditional.
- Candidate change: raise only the normal signal `sell_threshold` from `30` to `35`; keep buy logic, ETF pool, ATR stop, position sizing, and indicator parameters unchanged.

Training-log diagnostic snapshot that motivated the candidate:
- Parsed filled JoinQuant training events from the recorded `cross-v0.3.1-combo-candidate` log: 98 buys, 95 sells, 95 closed lots.
- Normal signal sells: 68 lots, realized PnL about +9311.1, win rate 0.529.
- ATR stops: 27 lots, realized PnL about +15990.8, win rate 0.630.
- Normal signal sells had positive post-sell drift: 10-trading-day mean forward return about +0.89%, with 19 of 64 available samples above +3%.
- The weak score buckets were most suspicious: `sell_score 33` had negative realized PnL and +3.00% average 10-day forward return; `sell_score 34` had slightly negative realized PnL and +0.60% average 10-day forward return.

Implementation checks:
- Added tests before implementation for version, unchanged parameters except `sell_threshold`, unchanged ETF pool, ATR stop remaining unconditional, and weak `sell_score 34` normal sell being blocked.
- `python -m pytest tests/test_cross_signal_sell35_candidate_strategy.py tests/test_cross_signal_strategy.py -q` passed.
- `python -m py_compile cross_signal_strategy\archive\candidates\smart_trade_joinquant_cross_signal_etf_sell35_candidate.py cross_signal_strategy\smart_trade_joinquant_cross_signal_etf.py` passed.

Can this result be used to change rules? not yet
Reason: This is only a prepared training candidate. It needs JoinQuant 2019-2021 training confirmation before any adoption or reserved-period validation.

### Local Training Check: Sell35 Candidate

Version: `cross-v0.3.2-sell35-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_sell35_candidate.py`
Backtest period: 2019-01-02 to 2021-12-31 local training replay
Initial capital: 20000
Protocol role: training-only local direction check; JoinQuant remains performance authority

Local replay result:
- Mainline equivalent `sell_threshold=30`: +118.75% return, +29.90% annualized, 6.81% max drawdown, Sharpe 2.117, Sortino 3.327, daily win rate 0.526, 99 buys, 96 sells, 96 closed trades, trade win rate 0.552, profit/loss ratio 4.034.
- Candidate `sell_threshold=35`: +86.08% return, +23.07% annualized, 5.97% max drawdown, Sharpe 1.757, Sortino 2.713, daily win rate 0.533, 92 buys, 89 sells, 89 closed trades, trade win rate 0.528, profit/loss ratio 3.303.

Interpretation:
- Raising the normal signal sell threshold from 30 to 35 reduces drawdown, but it materially damages return, Sharpe, Sortino, trade win rate, and profit/loss ratio in the local training replay.
- This suggests that the weak sell buckets are not safe to remove wholesale. Some `sell_score 32-34` exits may sell early, but the low-threshold sell mechanism still appears useful for capital recycling and risk release.
- Do not send this candidate to JoinQuant unless a narrower training-only hypothesis is developed. The broad `sell_threshold=35` candidate is rejected at the local direction-check stage.

Can this result be used to change rules? no
Reason: The training-only local check failed. Record as a failed broad-threshold experiment; do not inspect validation windows.

### Weak Replacement-Aware Signal-Sell Candidate Prepared

Version: `cross-v0.3.2-weak-replacement-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py`
Backtest period for next required JoinQuant check: 2019-01-01 to 2021-12-31 only
Initial capital for next required JoinQuant check: 20000
Execution schedule: daily `09:35`
Protocol role: training-only candidate; validation windows must not be inspected unless this candidate is frozen after JoinQuant training confirmation

Hypothesis:
- Broadly raising `sell_threshold` to 35 failed because normal signal sells are load-bearing for capital recycling.
- A narrower rule may help: protect only weak normal signal sells (`sell_threshold <= sell_score < 35`) when the current holding still has `buy_score >= 35` and selling it would not free a slot for any eligible replacement buy.
- ATR stops remain unconditional.

Local training probe:
- Mainline equivalent baseline: +118.75% return, +29.90% annualized, 6.81% max drawdown, Sharpe 2.117, Sortino 3.327, 99 buys, 96 sells, trade win rate 0.552, profit/loss ratio 4.034.
- Broad no-replacement protection for all weak sells: +108.24% return, 7.67% max drawdown, Sharpe 1.973, protected 9 sells. Rejected.
- Weak no-replacement with `buy_score >= 35`: +119.82% return, +30.12% annualized, 6.86% max drawdown, Sharpe 2.120, Sortino 3.335, 99 buys, 96 sells, trade win rate 0.552, profit/loss ratio 4.107, protected 2 sells.

- Other no-replacement variants were worse than baseline.

Implementation checks:
- Added tests before implementation for version, weak sell protection without replacement, no protection when replacement exists, no protection for stronger sells, and no protection when `buy_score < 35`.
- `python -m pytest tests/test_cross_signal_weak_replacement_candidate.py tests/test_cross_signal_strategy.py -q` passed.
- `python -m py_compile cross_signal_strategy\archive\candidates\smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py cross_signal_strategy\smart_trade_joinquant_cross_signal_etf.py` passed.

Interpretation:
- The local edge is small and protects only two sells, so this candidate has path-noise risk.
- It is nevertheless narrower and more economically coherent than a global threshold increase: keep weak-sell positions only when they still have some buy support and there is no replacement opportunity.
- Requires JoinQuant training confirmation before any adoption or reserved validation.

Can this result be used to change rules? not yet
Reason: This is only a prepared training candidate. It needs JoinQuant 2019-2021 training confirmation before adoption or validation.

### JoinQuant Early Supplemental Validation Check: Entry Combo Filter Candidate 2010-2014

Version: `cross-v0.3.1-combo-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
Backtest period: 2010-01-01 to 2014-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: frozen-rule early supplemental validation; no tuning allowed from this result

JoinQuant headline result:
- Strategy return: +1.20%.
- Annualized return: +0.25%.
- Excess return: +2.40%.
- Benchmark return: -1.17%.
- Alpha: -0.035.
- Beta: 0.057.
- Sharpe: -0.672.
- Sortino: -0.763.
- Max drawdown: 5.23%.
- Win rate: 0.366.
- Profit/loss ratio: 1.172.
- Trades shown by JoinQuant summary: 15 wins, 26 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-combo-candidate`.
- ERROR-level log count: 0.
- WARNING-level log count: 0.
- Early-window availability note: the log starts with many `paused=9` days because most ETFs in the current pool were not yet listed or tradable in 2010. Treat this as a limited-pool supplemental check, not a full-pool performance test.

Comparison with official mainline:
- Official `cross-v0.3.1` 2010-2014 result was -0.61% return, -0.13% annualized, 5.36% max drawdown, Sharpe -0.709, Sortino -0.822, win rate 0.349, profit/loss ratio 1.075, with 15 wins and 28 losses.
- The combo candidate improves return, annualized return, max drawdown, Sharpe, Sortino, win rate, profit/loss ratio, and reduces loss count.

Interpretation:
- This early supplemental validation supports the candidate directionally, but the period has limited ETF availability and low trade count.
- The useful evidence is that the filter does not damage early-period behavior and slightly improves the main quality metrics.
- Together with 2022-2023 and 2024-2026, this strengthens the case for adoption; 2015-2018 remains the only mixed window.

Can this result be used to change rules? no direct rule change
Reason: This is supplemental validation evidence for the already-frozen candidate. It supports adoption judgment, not retuning.

### JoinQuant Stress Validation Check: Entry Combo Filter Candidate 2015-2018

Version: `cross-v0.3.1-combo-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
Backtest period: 2015-01-01 to 2018-12-31
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: frozen-rule stress validation; no tuning allowed from this result

JoinQuant headline result:
- Strategy return: +23.21%.
- Annualized return: +5.50%.
- Excess return: +44.61%.
- Benchmark return: -14.80%.
- Alpha: 0.022.
- Beta: 0.087.
- Sharpe: 0.247.
- Sortino: 0.389.
- Max drawdown: 7.38%.
- Win rate: 0.444.
- Profit/loss ratio: 1.674.
- Trades shown by JoinQuant summary: 52 wins, 65 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-combo-candidate`.
- ERROR-level log count: 0.
- WARNING-level log count: 6.
- WARNING details: three 159928.XSHE 09:35 zero-volume market-order events, each paired with an unfilled/cancelled market-order warning: 2016-08-03 close order, 2017-03-09 close order, and 2017-08-02 open order. Treat these as JoinQuant execution facts for the stress window, not strategy-code errors.

Comparison with official mainline:
- Official `cross-v0.3.1` 2015-2018 result was +23.58% return, +5.58% annualized, 7.49% max drawdown, Sharpe 0.256, Sortino 0.393, win rate 0.443, profit/loss ratio 1.660, with 54 wins and 68 losses.
- The combo candidate slightly worsens total return, annualized return, Sharpe, and Sortino.
- The combo candidate slightly improves max drawdown, win rate, profit/loss ratio, and reduces trade count.

Interpretation:
- This stress validation is broadly neutral to slightly mixed, not a failure.
- The candidate does not collapse in 2015-2018, but its edge is weaker here than in 2022-2023 and 2024-2026.
- Since this is a frozen validation window, do not tune the filter from this result. The adoption decision should weigh all reserved periods together.

Can this result be used to change rules? no direct rule change
Reason: This is stress-validation evidence for the already-frozen candidate. It may affect adoption judgment, but must not be used for retuning.

### JoinQuant Validation Check: Entry Combo Filter Candidate 2024-2026

Version: `cross-v0.3.1-combo-candidate`
Code file: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
Backtest period: 2024-01-01 to 2026-07-08
Initial capital: 20000
Execution schedule: daily `09:35`
Protocol role: frozen-rule recent-market validation; no tuning allowed from this result

JoinQuant headline result:
- Strategy return: +58.17%.
- Annualized return: +20.78%.
- Excess return: +14.12%.
- Benchmark return: +38.60%.
- Alpha: 0.136.
- Beta: 0.311.
- Sharpe: 1.842.
- Sortino: 0.374.
- Max drawdown: 9.98%.
- Win rate: 0.513.
- Profit/loss ratio: 2.877.
- Trades shown by JoinQuant summary: 39 wins, 37 losses.

Operational checks:
- Version log confirmed `cross-v0.3.1-combo-candidate`.
- ERROR-level log count: 0.
- WARNING-level log count: 0.

Comparison with official mainline:
- Official `cross-v0.3.1` 2024-2026 result was +56.99% return, +20.41% annualized, 10.65% max drawdown, Sharpe 1.800, Sortino 0.352, win rate 0.506, profit/loss ratio 2.786.
- The combo candidate improves return, annualized return, max drawdown, Sharpe, Sortino, win rate, and profit/loss ratio.

Interpretation:
- This is the second frozen-rule validation period, and it also supports the entry-combo filter.
- The improvement is modest but directionally broad: higher return and lower drawdown at the same time.
- Together with the 2022-2023 validation, the candidate now has two out-of-sample periods supporting it. Continue with stress validation unchanged.

Can this result be used to change rules? no direct rule change
Reason: This is validation evidence for the already-frozen candidate. It supports continuing validation, not retuning.

## 2026-07-10 Low-Bounce Entry Filter JoinQuant Training Check

Period: 2019-01-01 to 2021-12-31
Version: `cross-v0.3.2-low-bounce-candidate`
Authority: JoinQuant

Candidate result:
- Strategy return: +124.73%.
- Annualized return: +31.96%.
- Max drawdown: 7.00%.
- Sharpe: 3.127.
- Sortino: 0.778.
- Win rate: 0.582.
- Profit/loss ratio: 5.117.
- Profitable/loss trades: 53/38.
- Buy/sell log events: 94/92.

Official `cross-v0.3.2` training baseline:
- Strategy return: +125.82%.
- Annualized return: +32.18%.
- Max drawdown: 6.70%.
- Sharpe: 3.109.
- Sortino: 0.799.
- Win rate: 0.558.
- Profit/loss ratio: 4.845.
- Profitable/loss trades: 53/42.

Operational checks:
- Version initialization confirmed `cross-v0.3.2-low-bounce-candidate`.
- ERROR-level log count: 0. Text matches for `error=` came from JoinQuant's `StockOrder` object representation and were logged at INFO level.
- WARNING-level log count: 2. Both were the known 2019-12-12 `513880.XSHG` zero-volume market-order matching event.
- The changed event counts and changed performance confirm that the candidate altered the JoinQuant trade path.

Decision:
- Reject the candidate and keep official `cross-v0.3.2` unchanged.
- Do not run reserved validation. The candidate failed the JoinQuant training gate because return and annualized return fell while max drawdown and Sortino worsened.
- The higher win rate and profit/loss ratio show that the filter removed some losing entries, but it also removed higher-value opportunity. This is not a favorable trade under the strategy's primary objective.

## 2026-07-10 Cross-v0.3.2 Training Stability Diagnostic

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2`
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Baseline:
- Total return: +118.75%.
- Max drawdown: 6.81%.
- Average exposure: 0.732.
- Filled buy/sell events: 99/96.

Annual stability:
- 2019: +35.68% return, 4.67% max drawdown, 0.711 average exposure, +5763.90 closed-trade PnL, 15 wins and 11 losses.
- 2020: +48.45% return, 6.81% max drawdown, 0.761 average exposure, +12481.40 closed-trade PnL, 18 wins and 14 losses.
- 2021: +8.60% return, 4.97% max drawdown, 0.722 average exposure, +5644.10 closed-trade PnL, 20 wins and 18 losses.

Contribution concentration:
- Largest winning trade / gross profit: 8.32%.
- Top three winning trades / gross profit: 23.51%.
- Largest ETF positive contribution / gross profit: 22.00%.
- Interpretation: performance is not dependent on one trade or one ETF, although 2020 is the strongest calendar-year contributor.

Exit quality:
- ATR stop: 28 trades, 17 wins and 11 losses, +14831.10 realized PnL, 5.85 profit/loss ratio, 18.89 average trading-day hold.
- Signal sell: 68 trades, 36 wins and 32 losses, +9058.30 realized PnL, 2.88 profit/loss ratio, 17.07 average trading-day hold.
- Interpretation: ATR exits are the higher-payoff exit path, but signal sells also add positive PnL and remain necessary for capital recycling.

Holding periods:
- Average: 17.60 trading days.
- Median: 13 trading days.
- Buckets: 3 trades at 0-4 days, 29 at 5-9, 33 at 10-19, and 31 at 20+.

T-1 entry trend groups:
- Strong up (`trend_score >= 20`): 26 trades, 18 wins and 8 losses, +14847.60 realized PnL, 24.50 average trading-day hold.
- Mild up (`0 < trend_score < 20`): 69 trades, 35 wins and 34 losses, +9256.40 realized PnL, 15.19 average trading-day hold.
- Sideways (`trend_score == 0`): 1 losing trade, -214.60 realized PnL.
- Down (`trend_score < 0`): 0 closed trades.
- Interpretation: trend participation is the principal payoff engine even though reversal crosses provide entry timing.

Descriptive volatility split:
- Median entry normalized ATR (`ATR / close`): 1.3812%.
- Above-median group: 48 trades, 25 wins and 23 losses, +15363.90 realized PnL.
- At/below-median group: 48 trades, 28 wins and 20 losses, +8525.50 realized PnL.
- This median is a balanced ex-post diagnostic split, not a strategy parameter or candidate threshold.

Doubled-friction replay:
- Stress definition: double commission rate, minimum commission, and slippage, then rerun the complete path.
- Total return: +100.33%, down 18.42 percentage points from baseline.
- Max drawdown: 7.34%, versus 6.81% baseline.
- Interpretation: the strategy remains profitable under the local stress model, but small-capital performance is materially friction-sensitive.

Decision:
- Keep official `cross-v0.3.2` unchanged.
- Do not add a new indicator from this report alone.
- Use the report to design at most one broad, training-only structural hypothesis at a time. Any candidate still requires tests first and JoinQuant 2019-2021 confirmation before reserved validation.

## 2026-07-10 Cross-v0.3.2 Training Friction Decomposition

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2`
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Method:
- Precompute and freeze each date/code T-1 score once.
- Rerun the complete broker, order sizing, lot rounding, cash constraint, close-marking, and ATR state path for every friction scenario.
- Baseline assumptions: 0.03% commission rate, CNY 5 minimum commission, and 0.10% one-way slippage.
- Component stresses double exactly one assumption; the combined stress doubles all three.

Results:
- Baseline: +118.75% return, 6.81% max drawdown, 99 buys, 96 sells.
- Commission rate x2: +117.34% return (-1.41pp), 6.82% max drawdown, 99 buys, 96 sells.
- Minimum commission x2: +112.67% return (-6.07pp), 7.03% max drawdown, 99 buys, 96 sells.
- Slippage x2: +106.61% return (-12.13pp), 7.14% max drawdown, 99 buys, 96 sells.
- All friction x2: +100.33% return (-18.42pp), 7.34% max drawdown, 99 buys, 96 sells.
- Sum of standalone component return deltas: -19.62pp.
- Combined-path interaction versus the standalone sum: +1.20pp.

Interpretation:
- Slippage is the dominant component, representing about 61.8% of the absolute standalone component loss.
- Minimum commission contributes about 30.9% and matters because the strategy starts with only CNY 20000 and creates many small ETF tickets.
- Percentage commission contributes about 7.2%, substantially less than execution price and minimum-ticket cost.
- The stable event counts show that the loss is not caused by a trade-count explosion. Exact code/date path equality was not asserted from counts alone.
- Broadly suppressing trades is not supported: prior sell-threshold and replacement-aware experiments either failed or had no JoinQuant effect. Execution quality is the next evidence-based research direction.

Decision:
- Keep official `cross-v0.3.2` unchanged.
- Do not add or retune technical indicators from this result.
- Confirm the actual Guojin PTrade ETF minimum-commission schedule before making a live-cost assumption.
- Treat execution timing/order style as the next candidate research area, with tests first and the same training-only protocol.

## 2026-07-10 Cross-v0.3.2 Capital Utilization Diagnostic

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2`
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Portfolio occupancy:
- Trading days: 730.
- Daily-mean exposure ratio: 0.725; daily-mean cash ratio: 0.275.
- The earlier 0.732 exposure figure is value-weighted (`sum exposure / sum total value`); 0.725 is the arithmetic mean of daily exposure ratios. Both are correct under their stated aggregation.
- Position-count days: 0 holdings 43 days, 1 holding 86 days, 2 holdings 172 days, 3 holdings 429 days.
- Days with at least one vacant slot: 301.
- Total vacant slot-days: 473 out of 2190 available slot-days; occupied-slot ratio 78.40%.

Vacant-slot causes:
- Below official 60-point buy threshold: 326 slot-days.
- No reversal candidate available: 136 slot-days.
- Official blocked entry combo: 6 slot-days.
- RSI overheat: 4 slot-days.
- Location filter: 1 slot-day.
- No eligible-but-unfilled or sell-conflict slot was observed.

De-duplicated below-threshold shadow episodes:
- Total: 199 independent episodes from 326 candidate-days.
- 50-59: 51 episodes; average 5/10/20-day returns +0.93%/+0.44%/+1.38%; win rates 62.75%/54.90%/58.82%.
- 40-49: 71 episodes; average 5/10/20-day returns +0.42%/+0.77%/+1.97%.
- 30-39: 44 episodes; average 5/10/20-day returns -0.02%/+1.06%/+2.83%.
- 20-29: 19 episodes; average 5/10/20-day returns -0.10%/+0.59%/+2.28%.
- Below 20: 14 episodes; sample too small and short-horizon return negative, so it is not actionable.
- These are ex-post fixed-horizon diagnostics, not executable strategy results.

Other rejected groups:
- Blocked combo: 4 independent episodes; 20-day average -3.30% and 25% win rate, supporting the existing v0.3.2 block.
- Overheat: 3 episodes; 5/10/20-day averages all negative, supporting the existing overheat filter.
- Location filter: 1 episode with negative returns at all horizons; too small for a new conclusion but not contradictory.

Decision:
- Preserve every existing filter.
- Test exactly one local candidate: 50-59 point backup fills only when primary candidates leave slots vacant.
- Do not globally lower `buy_threshold=60` and do not inspect validation periods.

## 2026-07-10 Backup Cross-Signal Slot-Fill Local Candidate

Period: 2019-01-01 to 2021-12-31
Candidate: primary threshold remains 60; 50-59 point reversal candidates fill only slots left after primary candidates
Protocol role: local training direction check only

Results:
- Official baseline: +118.75% return, 6.81% max drawdown, 99 buys, 96 sells, 0.732 value-weighted exposure.
- Backup-fill candidate: +86.39% return, 9.17% max drawdown, 110 buys, 107 sells, 0.810 value-weighted exposure.
- Filled backup buys: 50.
- Return delta: -32.35 percentage points.
- Max-drawdown delta: +2.36 percentage points.

Interpretation:
- The candidate fails decisively despite positive isolated shadow returns.
- Fixed-horizon shadow labels omitted capital opportunity cost, later primary signals, and the strategy's actual ATR/signal exit path.
- Higher exposure is not automatically better; selective cash protects the ability to enter stronger signals later.

Decision:
- Reject locally. Do not prepare a JoinQuant candidate and do not run reserved validation.
- Keep official `cross-v0.3.2` unchanged.
- Do not revisit score-only backup filling or global buy-threshold loosening without a new independent signal dimension.

## 2026-07-10 CMF(20) Observation-Only Training Attribution

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only CMF
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Calculate standard CMF(20) from adjusted OHLCV ending on the frozen T-1 signal date.
- Use only the standard zero line; do not search periods or thresholds.
- If supported, require `CMF > 0` only for mild-trend entries (`0 < trend_score < 20`).
- Leave strong-trend entries (`trend_score >= 20`) unchanged.

Formula and safety checks:
- Money-flow multiplier: `(2 * close - high - low) / (high - low)`.
- Flat daily ranges contribute zero multiplier.
- Rolling zero volume produces a missing CMF value rather than division by zero.
- The diagnostic adapter verifies `cmf_data_date == signal_date` and rejects any frame containing data after the signal date.
- CMF is attached only to frozen diagnostic snapshots and does not affect ranking, sizing, or orders.

Overall attribution:
- `CMF <= 0`: 23 trades, +3482.50 PnL, 65.22% win rate, 2.859 profit/loss ratio, 17.39% ATR-stop rate.
- `CMF > 0`: 73 trades, +20406.90 PnL, 52.05% win rate, 4.401 profit/loss ratio, 32.88% ATR-stop rate.
- The aggregate positive-CMF advantage must not be read without the trend split.

Trend split:
- Mild trend, `CMF <= 0`: 17 trades, +3412.60 PnL, 64.71% win rate, 3.924 profit/loss ratio.
- Mild trend, `CMF > 0`: 52 trades, +5843.80 PnL, 46.15% win rate, 2.218 profit/loss ratio.
- Strong trend, `CMF <= 0`: 6 trades, +69.90 PnL, 66.67% win rate, 1.099 profit/loss ratio.
- Strong trend, `CMF > 0`: 20 trades, +14777.70 PnL, 70.00% win rate, 15.931 profit/loss ratio.
- Sideways, `CMF > 0`: 1 losing trade, -214.60 PnL.

Entry-year split:
- 2019 `CMF <= 0`: 9 trades, +2855.80 PnL; `CMF > 0`: 20 trades, +4359.00 PnL.
- 2020 `CMF <= 0`: 8 trades, +685.60 PnL; `CMF > 0`: 24 trades, +13551.70 PnL.
- 2021 `CMF <= 0`: 6 trades, -58.90 PnL; `CMF > 0`: 29 trades, +2496.20 PnL.
- Quality direction varies materially by year and does not support a stable mild-trend zero-line filter.

Decision:
- Do not implement the mild-trend CMF candidate.
- Keep CMF observation-only and retain official `cross-v0.3.2` unchanged.
- Do not reinterpret the strong-trend subgroup as a new rule in the same experiment; that would be post-hoc selection from a small six-trade comparison group.
- Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-10 Strong-Trend Idle-Capacity Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only capacity snapshots
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Preserve the official buy gate, ranking, ATR logic, and normal per-slot target.
- A strong-trend buy is `trend_score >= 20`, using the existing strategy classification rather than a new threshold.
- Capacity exists only when all same-day official buy candidates have been allocated, at least one portfolio slot remains unused, and estimated cash above the official reserve can fund another copy of the selected buy target.
- If multiple strong buys occur on the same day, only the highest-ranked one can claim one unused slot.
- Create one fixed candidate only if the capacity subset has at least 10 closed trades, at least 3 profitable trades in each training year, favorable close excursion dominates adverse excursion, and neither one trade nor one ETF contributes more than half of gross profit.

Safety and measurement:
- Capacity snapshots are taken from the official order plan before execution and cannot change target values.
- Only actually filled strong-trend buys enter the trade-quality report.
- MFE/MAE use daily closes from the filled buy date through the sell date, consistent with the strategy's closing-price trailing-high convention.
- Future closes are used only for ex-post attribution and are never visible to the strategy.
- Commissions are deliberately excluded from the cash-headroom estimate because this experiment tests strategy allocation capacity, not fee optimization. The existing local replay still charges its normal modeled costs.

Results:
- All strong-trend entries: 27 filled buys, 26 closed trades, 1 open at the training boundary, +14847.60 realized PnL, 69.23% win rate, average MFE +11.54%, average MAE -1.15%.
- Capacity-eligible subset: 5 entries/5 closed trades, +1371.00 PnL, 60.00% win rate, 4.249 profit/loss ratio, average MFE +8.49%, average MAE -1.07%.
- 2019 capacity subset: 2 trades, +1081.40 PnL, 100.00% win rate.
- 2020 capacity subset: 2 trades, +334.40 PnL, 50.00% win rate.
- 2021 capacity subset: 1 trade, -44.80 PnL, 0.00% win rate.
- Largest winning trade contributed 39.69% of capacity gross profit; the largest ETF contributed 60.31%.

Decision:
- The observation gate fails. Do not create a strong-trend idle-slot sizing candidate.
- The aggregate strong-trend edge is real in this training replay, but it usually occurs when no complete extra slot remains after official candidates are allocated.
- Five capacity trades are too few for a concentration increase, the yearly sample is inadequate, 2021 is negative, and ETF concentration exceeds the pre-registered limit.
- Keep official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-08-12 Profit Giveback And Fixed One-ATR Break-Even Candidate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` baseline versus one isolated local candidate
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Observation-only diagnostic:
- 89 closed baseline trades were measured from actual local fills.
- 67 trades reached at least one entry ATR of closing-price profit; 19 later
  finished at or below zero, for a 28.36% conditional round-trip rate.
- Annual conditional rates were 10.53%, 33.33%, and 37.50% for 2019, 2020,
  and 2021. Forward path values were labels only and never entered orders.

Fixed candidate:
- Activation: stored highest close reaches `entry cost + 1.0 * entry ATR`.
- Floor after activation: entry cost, applied from a later decision only.
- Baseline: +120.61% total, +30.27% annualized, 7.47% drawdown, 2.172
  Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio.
- Candidate: +114.28% total, +29.01% annualized, 7.57% drawdown, 2.096
  Sharpe, 3.292 Sortino, 48.94% win rate, 4.148 profit/loss ratio.
- Candidate annual returns: +36.55%/+45.91%/+7.55%, versus baseline
  +35.84%/+49.74%/+8.46% in 2019/2020/2021.
- The path changed on 47 filled-order days across all three years (2/27/18).

Decision:
- Reject the candidate before JoinQuant and validation.
- The giveback problem is real, but a mechanical break-even floor lowers both
  return and trade quality by forcing too many early exits.
- Keep the official ATR trailing stop and no-profit-floor policy unchanged.

## 2026-07-18 Ordinary-Buy Minute Execution Overlay Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` order path with one pre-registered execution-only counterfactual
Engine: local replay and read-only one-minute training data; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked execution rule:
- Freeze each formal 09:35 ordinary-buy code and share amount before reading later minutes.
- Use the raw 09:35 arrival price as one passive buy limit.
- Scan only later executable bars before 10:05; the 09:35 bar cannot fill its own order.
- Require `low < limit`; equality is not treated as a fill because bar data cannot prove queue priority.
- If still unfilled, use the first executable minute at or after 10:05 as the market fallback.
- Leave signal sells, ATR exits, target sizing, fees, and every formal strategy rule unchanged.

Coverage and result:
- 92 eligible ordinary buys; 92 matched (100% coverage).
- 75 passive-limit fills and 17 market fallbacks.
- Overall average signed execution improvement: +0.0263% (+2.63 basis points).
- 2019/2020/2021: +0.0102% / -0.0078% / +0.0673%.
- Non-QDII/QDII: +0.0412% / +0.0040%.

Decision:
- Reject before full portfolio replay because 2020 failed the locked positive-per-year execution gate.
- Do not create JoinQuant or PTrade candidates and do not inspect reserved validation periods.
- Do not search neighboring times, cycle counts, price offsets, fallbacks, ETF exceptions, or sell overlays.
- Keep all three formal `cross-v0.3.2` strategy entries unchanged at 09:35.

## 2026-07-18 Fixed 09:35 Versus 10:00 Execution-Time Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` signal and risk path with one fixed execution-time candidate
Engine: local minute execution replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Results:
- `09:35`: +120.61% total return, 30.27% annualized, 7.47% maximum drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, and 4.440 profit/loss ratio.
- `10:00`: +127.65% total return, 31.65% annualized, 7.15% maximum drawdown, 2.280 Sharpe, 3.670 Sortino, 59.09% win rate, and 4.413 profit/loss ratio.
- Candidate annual returns were better in 2019 and 2020 but worse in 2021: +39.92%/+50.86%/+7.85% versus +35.84%/+49.74%/+8.46%.
- Across 135 matched orders, side-adjusted execution averaged about -0.012%. QDII improved by +0.0307%, but non-QDII worsened by -0.0425%.
- The complete filled-order path changed on 78 days, so aggregate performance cannot be attributed to a uniformly better fill.

Decision:
- Reject `10:00` under the locked gate and retain formal `09:35` execution.
- Do not run JoinQuant or any reserved validation period because the local structural gate failed.
- Close the timing family. Nearby times and subgroup-specific clocks would be post-hoc searches.

## 2026-07-16 ETF Share-Flow Shadow Diagnostic

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` closed-trade path with observation-only shares-outstanding attribution
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up, approved 2019-2021 training prices, and isolated read-only 2018-2021 share histories only

Locked hypothesis and timing:
- For `159915`, `512100`, `159928`, `518880`, and `159985`, calculate `log(shares[T-1] / shares[T-6])` over exactly five share observations.
- Use only a share row whose date exactly matches the frozen T-1 signal date. QDII is blocked because exact historical publication timing is unproved.
- Neutralize a window crossing the registered `159928` 2021-06-25 share split and resume only from a post-split baseline.
- Compare `positive` against flat-or-negative `non_positive`; do not search periods, magnitudes, or interactions.
- Add metadata to defensive score copies only. Scores, ranking, position sizing, orders, sells, and ATR logic are unchanged.

Coverage and aggregate result:
- All 52 eligible domestic closed buys were comparable (100% eligible coverage); 37 QDII buys were excluded.
- Positive: 24 trades, +3795.30 PnL, +1.39% average return, 54.17% win rate, 3.398 profit/loss ratio.
- Non-positive: 28 trades, +7422.60 PnL, +3.70% average return, 50.00% win rate, 3.624 profit/loss ratio.

Annual result:
- 2019 positive: 7 trades, +1.35% average return, 42.86% win rate. Non-positive: 8 trades, +8.10%, 50.00%.
- 2020 positive: 9 trades, +1.24%, 55.56%. Non-positive: 7 trades, +5.92%, 57.14%.
- 2021 positive: 8 trades, +1.59%, 62.50%. Non-positive: 13 trades, -0.20%, 46.15%.

Decision:
- Reject a shares-outstanding sign confirmation or veto before candidate creation. Non-positive flow led average return in 2019 and 2020, while positive flow led both average return and win rate in 2021.
- The result is not sparse, but it is regime-dependent. Aggregate PnL cannot override the pre-registered annual-direction gate.
- Do not search neighboring lookbacks, thresholds, z-scores, fund-size/NAV interactions, QDII assumptions, code exceptions, or sell rules.
- Keep official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-14 Controlled-Breakout Anti-Chase Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` closed-trade path with observation-only breakout extension labels
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Resistance is the highest adjusted high of exactly 20 valid bars ending T-2.
- T-1 close above resistance is a breakout.
- A breakout is extended when `RSI6 >= 75` or close is at least 10% above MA20; otherwise it is controlled.
- A candidate may reject only extended breakouts, and only if both groups have at least 6 closed trades overall, at least 2 per year, and extended breakouts have lower average return and win rate every year.
- Breakout cannot create a buy, add score, alter ranking or sizing, or change sells.

Safety checks:
- The adapter requires the adjusted frame to end exactly on T-1 and rejects later rows.
- The resistance window excludes T-1 and ends on T-2.
- Score snapshots are copied defensively; labels and continuous diagnostics are observation-only.

Results:
- Controlled breakout: 15 trades, +9823.80 PnL, +7.83% average return, 73.33% win rate, 11.365 profit/loss ratio.
- Extended breakout: 2 trades, +139.70 PnL, +1.32% average return, 50.00% win rate, 2.337 profit/loss ratio.
- Controlled annual counts: 5/6/4. Extended annual counts: 1/0/1.
- The controlled group averaged RSI6 70.52, 3.29% above MA20, +2.67% over 5 days, +3.07% over 10 days, +6.93% over 20 days, and +9.04% from the same prior-20-day low.
- Both extended trades were RSI extensions rather than MA20-distance extensions: their RSI6 values were 75.10 and 82.37, while MA20 distances were only 2.47% and 2.83%.

Decision:
- Reject the anti-chase candidate before implementation. The extended subset failed the overall and every annual sample gate, and its single 2019 trade had higher win rate than controlled breakouts.
- The observed controlled breakouts look like early or measured strength rather than late chasing, but this diagnostic cannot reward them or add a new buy rule.
- Do not search nearby thresholds, periods, AND variants, breakout bonuses, or sell interactions.
- Keep official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-13 Multiple-Testing Risk Audit

Period: 2019-01-01 to 2021-12-31
Version: frozen official `cross-v0.3.2` local training path
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Retained-trial evidence:
- The failed/non-adopted ledger contains 47 experiments.
- Counting the selected mainline gives a minimum trial count of 48.
- This is a lower bound, not an exact historical search count.

Training result:
- Total return +120.61%, annualized return 30.27%, annualized Sharpe 2.172.
- Annual returns: 2019 +35.84%, 2020 +49.74%, 2021 +8.46%.
- PSR p-value 0.000123988; minimum-48 Bonferroni p-value 0.00595144.
- The PSR/Bonferroni approximation passes at no more than 403 trials.
- Newey-West/HAC automatic lag 6, t-statistic 3.837, p-value 0.0000622008, and minimum-48 Bonferroni p-value 0.00298564.

Limitations:
- This is training-only evidence and not out-of-sample validation.
- Canonical DSR is unavailable because all candidate Sharpe values were not retained.
- PBO is unavailable because aligned daily return curves for all candidates were not retained.
- The 48-trial corrections are optimistic upper bounds if undocumented early or adopted variants exist.

Decision:
- Keep official `cross-v0.3.2` unchanged.
- Do not increment the failed-experiment counter or reopen any exhausted research family; this is governance infrastructure, not a candidate.
- Retain aligned daily candidate curves for any future authorized experiment so exact cross-trial diagnostics become possible.

## 2026-07-13 Fixed MACD(6,13,5) Single-Variable Candidate

Period: 2019-01-01 to 2021-12-31
Warm-up: approved read-only 2018 daily bars only; excluded from trades and returns
Version: `cross-v0.3.2-macd-6-13-5-candidate`
Engine: two independent local replays; JoinQuant remains the performance authority

Locked design before the run:
- Baseline MACD is `(12,26,9)` and candidate MACD is `(6,13,5)`.
- MACD periods are the only changed values. ETF pool, RSI, KDJ, BOLL, MA, ADX, ATR, scoring, thresholds, sizing, exits, and execution are identical.
- A candidate can pass only if it changes filled orders in every training year, improves total and annualized return, and does not worsen maximum drawdown, Sharpe, Sortino, win rate, profit/loss ratio, or any annual return.
- No nearby-period search is permitted after the result.

Results:
- Baseline: +120.61% total, 30.27% annualized, 7.47% max drawdown, 2.172 Sharpe, 3.415 Sortino, 56.18% win rate, 4.440 profit/loss ratio, 92 buys, 89 sells.
- Candidate: +84.69% total, 22.76% annualized, 7.00% max drawdown, 1.766 Sharpe, 2.670 Sortino, 50.00% win rate, 2.834 profit/loss ratio, 97 buys, 94 sells.
- Annual baseline versus candidate: 2019 35.84% versus 17.02%; 2020 49.74% versus 51.94%; 2021 8.46% versus 3.87%.
- Filled-order paths changed on 89 days: 38 in 2019, 11 in 2020, and 40 in 2021.

Decision:
- Reject MACD(6,13,5). The 0.47 percentage-point drawdown improvement and 2.20-point 2020 gain do not offset the 35.92-point total-return loss and broad quality deterioration.
- Do not run JoinQuant or validation because the local gate failed.
- Keep official `cross-v0.3.2` and MACD(12,26,9) unchanged. Close the one-shot MACD research budget.

## 2026-07-14 Horizontal Support/Resistance Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only horizontal structure labels
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked definition and future-function boundary:
- Resistance is the maximum adjusted high and support is the minimum adjusted low over exactly 20 valid bars strictly before T-1, so the latest level input is T-2.
- T-1 close and official ATR(14) are used only to calculate normalized distance from those pre-existing levels.
- Fixed pressure groups are breakout, within one ATR below resistance, and more than one ATR below resistance.
- Fixed support groups are breakdown, within one ATR above support, and more than one ATR above support.
- The only actionable hypothesis was that mild-trend near-resistance entries would have lower average return and lower win rate than all other mild-trend entries in every training year, with at least 15 total and 3 annual trades in each group.
- No reserved validation period was read or run.

Overall closed-trade attribution:
- Breakout: 17 trades, +9963.50 PnL, +7.06% average return, 70.59% win rate, 10.468 profit/loss ratio.
- Near resistance: 37 trades, +8070.40 PnL, +2.63% average return, 54.05% win rate, 3.294 profit/loss ratio.
- Room to resistance: 35 trades, +6230.20 PnL, +1.69% average return, 51.43% win rate, 3.508 profit/loss ratio.
- All 89 closed-buy snapshots were more than one ATR above prior support; no near-support or support-breakdown trade existed.

Pre-registered mild-trend annual comparison:
- 2019: near resistance 4 trades, +6.25% average return, 50.00% win rate; comparison 14 trades, about +0.35%, 42.86% win rate.
- 2020: near resistance 5 trades, +3.56%, 40.00%; comparison 12 trades, +3.97%, 66.67%.
- 2021: near resistance 13 trades, +0.61%, 61.54%; comparison 14 trades, about +0.75%, 50.00%.

Decision:
- Reject the near-resistance filter before candidate creation. It failed both locked metrics in 2019 and the win-rate condition in 2021.
- Do not promote the strong descriptive breakout group because a breakout candidate was not pre-registered. Doing so now would be post-hoc winner selection.
- Do not search other lookbacks, ATR boundaries, pivots, support exceptions, Fibonacci levels, or volume profiles.
- Keep official `cross-v0.3.2` unchanged and close the one-shot horizontal-price-structure budget.

## 2026-07-11 US-QDII Previous-NAV Premium Observation

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` closed-trade path with observation-only `513100/513500` premium attribution
Engine: local replay plus no-order JoinQuant capability probes; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked design before attribution:
- Use only actual closed mainline buys in `513100` and `513500`.
- Calculate premium from the raw T-day 09:35 market price and point-in-time reference proxy; never use same-day end-of-day NAV.
- Keep fixed groups `<=2%`, `2-5%`, `5-10%`, and `>10%`.
- Treat `>5%` as the sole elevated group. Do not move this boundary after results.
- Require at least 80% reference coverage, at least 10 elevated trades, at least two years with 3 elevated trades, underperformance in qualifying years, and at least 3 elevated trades in each ETF before a candidate can exist.

Results:
- Targeted closed trades: 28; covered: 27; missing: 1; coverage: 96.43%.
- `<=2%`: 24 trades, +6509.90 PnL, +2.79% average return, 62.50% win rate, 5.884 profit/loss ratio.
- `2-5%`: 1 trade, +1105.10 PnL, +12.88% return.
- `5-10%`: 2 trades, +388.80 PnL, +2.55% average return, 50.00% win rate, 3.046 profit/loss ratio, 8.16% average premium.
- `>10%`: no actual closed mainline buy.
- Both elevated trades occurred in `513100` during 2020; `513500` had none.

Decision:
- Reject candidate creation. Sample-size, cross-year, and cross-code gates all failed.
- Do not lower the 5% boundary, add nearby bands, or extend the proxy to dynamic-IOPV ETFs after seeing this result.
- Official `cross-v0.3.2` remains unchanged and no reserved validation run is permitted for this failed observation.

## 2026-07-11 Local Zero-Trade And Unfilled-Sell Correctness Replay

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` signal path with corrected local execution semantics
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Correctness changes:
- A missing 09:35 minute bar now produces an unfilled order instead of terminating the replay.
- A 09:35 bar with both zero volume and zero trade count now produces an unfilled order instead of filling at a stale close.
- Planned sells no longer clear buy date, entry ATR, or highest-close state before fill confirmation.
- New buys are checked against the broker's actual post-execution holdings, preventing an unfilled sell from creating a phantom slot.
- No 10:35 retry, premium filter, or signal change was introduced.

Source evidence:
- `159915` on 2020-12-16 and 2021-02-09 had zero volume and zero trades from the open through 10:30, with the first positive minute at 10:31.
- `159915` on 2021-02-08 had zero volume and zero trades for the full session.
- `513880` on 2019-12-12 had zero volume at 09:35 but first traded at 09:38, proving that one zero-volume minute is insufficient to label the legal cause as suspension.

Corrected replay summary:
- End value: 44122.30.
- Total return: +120.61%.
- Maximum drawdown: 7.47%.
- Filled buys/sells: 92/89.
- Maximum actual holdings: 3.
- Unfilled plans: 17 total, comprising 15 zero-trade execution rejections and 2 buys rejected because a preceding unfilled sell did not release a slot.

Interpretation:
- The changed local return is a consequence of more realistic execution, not a strategy improvement and not a basis for parameter tuning.
- The local engine can now represent the non-tradable interval visible in minute data without pretending that it knows the formal suspension reason.
- Exact legal halt status still requires an exchange/broker status source. The current minute data supports conservative fill/no-fill simulation only.

## 2026-07-11 Reversal-First Candidate Ranking Comparison

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` versus one isolated local ranking candidate
Engine: two local replays sharing identical precomputed T-1 score snapshots
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked candidate before comparison:
- Keep the official eligible-candidate set, buy threshold, entry filters, maximum holdings, position sizing, sell rules, ATR logic, and execution model unchanged.
- Official ranking: descending `buy_score`, then descending `reversal_score`, then code.
- Candidate ranking: descending `reversal_score`, then descending `buy_score`, then code.
- Do not search weights or test additional ranking permutations.
- Advance only if at least 10 buy-decision days change, every training year contains a changed decision, annual returns do not worsen, total return improves, and max drawdown, Sharpe, and Sortino do not worsen.

Local comparison:
- Official: +118.75% return, 6.81% max drawdown, Sharpe 2.117, Sortino 3.327, 99 buys, 96 sells.
- Reversal-first: +121.69% return, 6.81% max drawdown, Sharpe 2.157, Sortino 3.403, 99 buys, 96 sells.
- Annual returns were identical in 2019 and 2020. 2021 changed from +8.60% to +10.06%.
- The candidate changed the bought code on only one day: 2021-12-27.

Root-cause audit of the only changed decision:
- Both signals used `signal_date=2021-12-24`.
- Official chose `159928`: buy score 70, reversal 35, location 15, trend 14, volume 6.
- Reversal-first chose `513500`: buy score 69, reversal 45, location 15, trend 9, volume 0.
- From 2021-12-27 09:35 to the 2021-12-31 training boundary, `159928` moved -3.29% while `513500` moved +0.95%.
- The changed position remained inside the final four-trading-day boundary segment. The headline improvement is therefore dominated by one terminal mark-to-market difference rather than repeated closed-trade ranking evidence.

Decision:
- Reject reversal-first ranking locally and do not prepare a JoinQuant candidate.
- The two ranking methods are behaviorally identical on all but one training day. A one-event terminal-boundary gain is not evidence of a robust ranking improvement.
- Preserve official `buy_score -> reversal_score -> code` ordering and do not tune ranking weights from this event.
- Do not inspect reserved validation periods for this failed local candidate.

## 2026-07-11 Kaufman ER(10) Direction Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only ER fields
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Use standard Kaufman Efficiency Ratio with period 10: `abs(C_t-C_t-10) / sum(abs(delta C), 10)`.
- A complete zero-movement window has ER 0; incomplete windows remain missing.
- Compare only one-day ER direction: rising versus non-rising. Do not search ER levels or alternate periods.
- If supported, require rising ER only for mild-trend entries (`0 < trend_score < 20`). Strong-trend entries remain unchanged.
- Create one candidate only if mild rising and non-rising groups both have at least 15 trades, each has at least 3 trades in every training year, and rising ER improves both average return and win rate in every year.

Safety checks:
- ER is calculated from the adjusted close series ending exactly on the frozen T-1 signal date.
- The adapter rejects any row after the signal date and returns defensive score copies.
- ER fields never enter official scoring, filtering, ranking, sizing, or order logic.

Overall attribution:
- Declining ER: 51 trades, +10478.90 PnL, +2.65% average return, 54.90% win rate, 3.420 profit/loss ratio.
- Rising ER: 44 trades, +13563.50 PnL, +3.20% average return, 56.82% win rate, 5.000 profit/loss ratio.
- Mild trend, declining ER: 38 trades, +6597.10 PnL, +1.89% average return, 52.63% win rate, 3.092 profit/loss ratio.
- Mild trend, rising ER: 30 trades, +2812.30 PnL, +1.12% average return, 50.00% win rate, 2.059 profit/loss ratio.
- Strong trend, rising ER was strong descriptively, but that subgroup was not the locked mild-trend hypothesis and the existing strategy already identifies strong trends.

Pre-registered mild-trend annual comparison:
- 2019 declining: 13 trades, +1944.80 PnL, +2.37% average return, 46.15% win rate. Rising: 5 trades, +82.90 PnL, +0.34% average return, 40.00% win rate.
- 2020 declining: 11 trades, +2420.00 PnL, +2.08% average return, 63.64% win rate. Rising: 8 trades, +3100.50 PnL, +4.21% average return, 50.00% win rate.
- 2021 declining: 14 trades, +2232.30 PnL, +1.28% average return, 50.00% win rate. Rising: 17 trades, -371.10 PnL, -0.10% average return, 52.94% win rate.

Decision:
- Reject the mild-trend rising-ER confirmation before candidate creation.
- Rising path efficiency is not a stable mild-trend quality signal: it underperformed on both average return and win rate in 2019, lost the win-rate comparison in 2020, and had negative average return in 2021.
- Do not tune ER thresholds or periods, and do not promote the post-hoc strong-trend subgroup.
- Keep ER observation-only and official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-10 BOLL(20,2) BandWidth Direction Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only BandWidth fields
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Preserve the existing standard BOLL period and multiplier: `20` and `2.0`.
- Calculate `BandWidth = (upper - lower) / middle` from the adjusted daily close series ending on the frozen T-1 signal date.
- Compare only one-day direction: rising versus non-rising. Do not search absolute width thresholds, alternate periods, or multi-day slopes.
- If supported, require rising BandWidth only for mild-trend entries (`0 < trend_score < 20`). Strong-trend entries remain unchanged.
- Create one candidate only if mild rising and mild non-rising groups both have at least 15 trades, each has at least 3 trades in every training year, and rising width improves both average return and win rate in every year.

Safety checks:
- The adapter verifies that the BandWidth frame ends exactly on the base signal date and rejects any row after T-1.
- It returns defensive score copies and attaches BandWidth only to diagnostic snapshots.
- BandWidth never enters the official score, buy filter, candidate ranking, position sizing, or sell logic.

Overall attribution:
- Declining width: 49 trades, +9607.20 PnL, +2.23% average return, 57.14% win rate, 3.660 profit/loss ratio.
- Rising width: 47 trades, +14282.20 PnL, +3.47% average return, 53.19% win rate, 4.351 profit/loss ratio.
- Mild trend, declining: 38 trades, +1317.00 PnL, +0.19% average return, 50.00% win rate, 1.393 profit/loss ratio.
- Mild trend, rising: 31 trades, +7939.40 PnL, +3.05% average return, 51.61% win rate, 4.040 profit/loss ratio.

Pre-registered mild-trend annual comparison:
- 2019 declining: 12 trades, -776.90 PnL, -1.01% average return, 33.33% win rate. Rising: 7 trades, +2651.60 PnL, +5.85% average return, 57.14% win rate.
- 2020 declining: 12 trades, +882.70 PnL, +0.76% average return, 50.00% win rate. Rising: 7 trades, +4637.80 PnL, +6.78% average return, 71.43% win rate.
- 2021 declining: 14 trades, +1211.20 PnL, +0.72% average return, 64.29% win rate. Rising: 17 trades, +650.00 PnL, +0.36% average return, 41.18% win rate.

Decision:
- Reject the mild-trend rising-BandWidth confirmation before candidate creation.
- The relationship is strong in 2019 and 2020 but reverses in 2021 on a non-trivial sample. A single fixed BandWidth-direction rule is therefore regime-dependent rather than stable.
- Do not change BOLL(20,2), search width thresholds, replace one-day direction with a tuned slope, or add a post-hoc 2021 regime exception.
- Keep BandWidth observation-only and official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-10 Active Cross-Sequence Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` trading path with observation-only cross-event offsets
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Reuse the official three-trading-day cross window and latest-direction semantics. Do not change indicator periods or the cross window.
- Record how many trading days before T-1 each active RSI6/RSI12, RSI6/RSI24, KDJ K/D, KDJ J/D, and MACD DIF/DEA cross occurred.
- `oscillators_lead_macd` requires every currently active RSI/KDJ upward cross to occur earlier than the active MACD upward cross.
- `macd_leads_oscillators` requires every active RSI/KDJ upward cross to occur later than MACD. Same-day, mixed, no-MACD, and MACD-only states remain separate.
- If supported, block only mild-trend `macd_leads_oscillators` entries. Create a candidate only if both clean order groups have at least 10 trades, at least 3 per training year, and oscillator-leading improves both average return and win rate every year.

Safety checks:
- The adapter recomputes indicators from the adjusted frame ending on the frozen T-1 signal date and rejects any later row.
- Cross-event detection matches the mainline difference rule and keeps only the latest direction inside the window.
- Sequence fields are attached to defensive diagnostic snapshots and never affect the official scores, ranking, or orders.

Overall closed-trade attribution:
- No active MACD confirmation: 70 trades, +16316.10 PnL, +2.60% average return, 57.14% win rate, 3.896 profit/loss ratio.
- Oscillators lead MACD: 11 trades, +3244.80 PnL, +3.32% average return, 45.45% win rate, 4.903 profit/loss ratio.
- Same day: 10 trades, +4617.50 PnL, +5.65% average return, 60.00% win rate, 7.041 profit/loss ratio.
- Mixed timing: 5 trades, -289.00 PnL, -0.57% average return, 40.00% win rate, 0.551 profit/loss ratio.
- No `macd_leads_oscillators` closed trade was observed.

Mild-trend evidence:
- No active MACD confirmation: 51 trades, +5981.40 PnL, +1.06% average return, 52.94% win rate.
- Oscillators lead MACD: 7 trades, +1744.80 PnL, +2.94% average return, 28.57% win rate.
- Same day: 7 trades, +1609.20 PnL, +3.82% average return, 57.14% win rate.
- Mixed: 4 trades, -79.00 PnL, +0.03% average return, 50.00% win rate.
- Mild oscillator-leading counts were only 2/3/2 in 2019/2020/2021. The two 2021 trades both lost, for -218.00 PnL.

Decision:
- Reject the proposed sequence candidate before implementation because the pre-registered MACD-leading comparison group has zero closed trades and the oscillator-leading group is too small and unstable.
- The observed strategy path is primarily early RSI/KDJ reversal participation without an active MACD upward cross, not a repeatable two-stage MACD confirmation process.
- Do not convert the post-hoc mixed or same-day groups into filters. Same-day was strong overall but negative in 2021; mixed was weak but contained only five trades.
- Keep official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-07-10 09:35 ATR-Normalized Gap Observation Gate

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` closed-trade path with observation-only execution-gap attribution
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Locked hypothesis before attribution:
- Calculate `gap_atr = (T-day 09:35 raw price - T-1 close) / T-1 ATR`.
- Use the raw 09:35 minute close, not the local fill price after slippage.
- Keep four fixed groups: `<=0`, `(0, 0.5]`, `(0.5, 1.0]`, and `>1.0 ATR`.
- Consider one candidate only if `>1 ATR` contains at least 10 closed trades, at least 3 in every training year, and has both lower average return and lower win rate than all other buys in every year.
- The possible candidate would skip only `>1 ATR` T-day entries. No other gap bucket or trend interaction was pre-registered.

Future-function boundary:
- T-1 close, ATR, trend score, and signal date come from the frozen entry-score snapshot.
- The diagnostic rejects any trade whose signal date is not strictly before the buy date.
- T-day 09:35 price is used only as execution-time information, which is available when the order decision is made.
- Later closes are used only for ex-post MFE/MAE attribution and never enter strategy code.

Overall results:
- `gap_atr > 1`: 5 trades, +3309.80 PnL, +8.15% average trade return, 60.00% win rate, 9.120 profit/loss ratio, +12.40% average MFE, -1.23% average MAE.
- `0.5 < gap_atr <= 1`: 13 trades, +1199.70 PnL, +1.05% average return, 38.46% win rate, 1.733 profit/loss ratio.
- `0 < gap_atr <= 0.5`: 19 trades, +5328.60 PnL, +3.38% average return, 57.89% win rate, 4.812 profit/loss ratio.
- `gap_atr <= 0`: 59 trades, +14051.30 PnL, +2.61% average return, 57.63% win rate, 4.170 profit/loss ratio.

Year and trend evidence:
- `>1 ATR` had only 2/2/1 trades in 2019/2020/2021.
- 2019: +1081.40 PnL, +7.18% average return, 100% win rate.
- 2020: +2258.80 PnL, +13.27% average return, 50% win rate.
- 2021: -30.40 PnL from one trade.
- Four of the five `>1 ATR` trades were strong-trend entries; they produced +3340.20 PnL and +10.23% average return.
- The post-hoc `0.5-1 ATR` mild-trend subgroup was weak, but it was not the locked hypothesis and must not be converted into a rule from this experiment.

Decision:
- Reject the proposed `>1 ATR` high-gap entry filter before candidate creation.
- Large positive gaps are rare and mostly belong to profitable strong-trend continuation trades in this training path. Blocking them would contradict the observed evidence and the strategy's need to preserve strong trends.
- Do not tune a smaller gap threshold or add a mild-trend interaction after seeing these results. That would be post-hoc selection.
- Keep official `cross-v0.3.2` unchanged. Do not run JoinQuant or reserved validation for this failed observation gate.

## 2026-08-16 ATR-Stress Local Pre-Check On cross-v0.3.2 Path

Period: 2019-01-01 to 2021-12-31
Version: official `cross-v0.3.2` baseline versus baseline + frozen ATR-stress rule (15/3/0.50)
Engine: local replay (`cross_signal_strategy/research/atr_stress_adoption_precheck.py`); JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only; no validation-period data read

Purpose:
- Before any JoinQuant run, confirm on the current mainline that the already-validated frozen ATR-stress candidate still improves training return and drawdown, and that the rule actually triggers.
- Two replays share identical data, adapter, execution model, and order path logic; the candidate only injects the three frozen stress keys at planner level. No formal strategy file was changed for the replay.

Results:
- Baseline: end value 44122.30, total return +120.61%, annualized (244-day compounding) +30.27%, max drawdown 7.47%, Sharpe 2.172, Sortino 3.415, annual 2019/2020/2021 +35.84%/+49.74%/+8.46%, 92 buys/89 sells, 25 filled ATR stops, 17 unfilled plans.
- Candidate: end value 45000.50, total return +125.00%, annualized +31.13%, max drawdown 6.03%, Sharpe 2.262, Sortino 3.581, annual +35.84%/+52.68%/+8.49%, 92 buys/89 sells, 25 filled ATR stops, 17 unfilled plans.
- Baseline alignment: the local replay exactly reproduced the recorded corrected baseline (44122.30, +120.61%, 7.47%, 92/89), so the comparison is trustworthy.
- Stress audit: 28 stress-active trading days; six half-size filled buys: 2020-03-03 159985, 2020-03-05 159928, 2020-03-06 513050, 2020-03-23 159985, 2020-09-15 513880, 2020-09-22 512100. The September 2020 pair is new on the v0.3.2 path (the v0.3.1-era JoinQuant candidate showed only the four March 2020 buys).

Interpretation:
- The rule improves total return (+4.39pp), max drawdown (-1.44pp), Sharpe, and Sortino without changing any entry or exit decision: buy/sell counts are identical, so the improvement is pure sizing.
- The 2020 annual gain (+2.94pp) is consistent with stress-triggered half-size buys being protective during the March crash cluster and the September 2020 pullback.
- This is local training evidence only; JoinQuant training confirmation and the four reserved validation windows remain mandatory before adoption and PTrade sync.

Can this result be used to change rules? candidate staging only
Reason: The frozen rule was already validated on `cross-v0.3.1`; this pre-check extends confidence to the current `cross-v0.3.2` path. JoinQuant remains the authority and must confirm before the mainline adoption is completed.

## 2026-08-16 JoinQuant v0.3.3 ATR-Stress Staging Runs

Version: `cross-v0.3.3` / build `20260816.1`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
Platform: JoinQuant (authority)
Protocol role: staged adoption runs; validation results are recorded only, never used for tuning

### Run 1: Training Confirmation 2019-2021

Baseline `cross-v0.3.2` record: +125.82% return, +32.18% annualized, 6.70% max drawdown, 53W/42L.
v0.3.3 result: +129.25% return, +32.86% annualized, +39.70% excess (benchmark +64.10%), alpha 0.239, beta 0.340, max drawdown 6.28%, Sharpe 2.275, Sortino 3.245, win rate 0.558, profit/loss ratio 5.297, 53W/42L, max drawdown interval 2020/09/03-2020/09/09.
Judgment: passed. Return +3.43pp, max drawdown -0.42pp, identical 53/42 path. The drawdown interval moved from the March 2020 crash cluster to the September 2020 pullback, matching the local pre-check's additional September stress triggers. Sharpe/Sortino are not compared against the old 3.109/0.799 record because the platform reporting caliber appears to have changed between runs; all cross-window v0.3.3 comparisons use the same current caliber.

### Run 2: First Reserved Validation 2022-2023

Baseline `cross-v0.3.2` record: +17.36% return, +8.62% annualized, 11.63% max drawdown, Sharpe 0.432, Sortino 0.622, win rate 0.385, profit/loss ratio 1.560, 25W/40L.
v0.3.3 result: +17.90% return, +8.88% annualized, +69.76% excess (benchmark -30.55%), alpha 0.086, beta 0.176, max drawdown 11.17%, Sharpe 0.459, Sortino 0.658, win rate 0.385, profit/loss ratio 1.584, 25W/40L, max drawdown interval 2022/02/24-2022/11/22.
Judgment: passed and improved on every metric: return +0.54pp, max drawdown -0.46pp, Sharpe and Sortino higher, identical 25/40 path. The improvement is pure sizing (stress half-size buys during the 2022 stop clusters), which is exactly the risk-insurance role the rule is meant to play in weak markets.

### Run 3: Second Reserved Validation 2024-01-01 to 2026-07-08

Baseline `cross-v0.3.2` record: +58.17% return, +20.78% annualized, 9.98% max drawdown, win rate 0.513, profit/loss ratio 2.877, 39W/37L.
v0.3.3 result: +58.17% return, +20.78% annualized, +14.12% excess (benchmark +38.60%), alpha 0.136, beta 0.311, max drawdown 9.98%, Sharpe 1.307, Sortino 1.842, win rate 0.513, profit/loss ratio 2.877, 39W/37L, max drawdown interval 2025/11/03-2026/07/08.
Judgment: passed as harmless/inactive. Every headline number is identical to the baseline record, so the stress rule did not trigger in this window (same as the v0.3.1-era validation, where it logged 0 half-size buys). It did not suppress upside in a rising market. Note: the current-caliber Sharpe 1.307 cannot be compared against the old record's 1.842 because that older record also listed Sortino 0.374, which is internally impossible (Sortino cannot be below Sharpe); the two values were evidently reported in a different or swapped caliber. The 1.307/1.842 pair in this run is internally consistent.

### Run 4: Stress Reserved Validation 2015-01-01 to 2018-12-31

Baseline `cross-v0.3.2` record: +23.21% return, +5.50% annualized, 7.38% max drawdown, win rate 0.444, profit/loss ratio 1.674, 52W/65L.
v0.3.3 result: +23.21% return, +5.50% annualized, +44.61% excess (benchmark -14.80%), alpha 0.022, beta 0.087, max drawdown 7.38%, Sharpe 0.186, Sortino 0.247, win rate 0.444, profit/loss ratio 1.674, 52W/65L, max drawdown interval 2015/06/12-2016/01/05.
Judgment: passed as harmless/inactive. Return, drawdown, win rate, profit/loss ratio, and the 52/65 path are identical to the baseline record, so the stress rule did not trigger in this window (same as the v0.3.1-era validation). The drawdown interval documented here belongs to the v0.3.2 path; the older 2016-07-29~2016-11-09 interval came from the v0.3.1 path and is not comparable. Sharpe/Sortino magnitudes differ from the old record (0.247/0.389) because of the platform reporting-caliber change noted in Run 1; both pairs are internally consistent, and the identical daily path is the decisive comparison.

### Run 5: Early Out-Of-Sample Supplement 2010-01-01 to 2014-12-31

Baseline `cross-v0.3.2` record: +1.20% return, +0.25% annualized, 5.23% max drawdown, win rate 0.366, profit/loss ratio 1.172, 15W/26L.
v0.3.3 result: +1.20% return, +0.25% annualized, +2.40% excess (benchmark -1.17%), alpha -0.035, beta 0.057, max drawdown 5.23%, Sharpe -0.763, Sortino -0.672, win rate 0.366, profit/loss ratio 1.172, 15W/26L, max drawdown interval 2012/03/13-2012/08/17.
Judgment: passed as harmless/inactive. Identical return, drawdown, win rate, profit/loss ratio, and 15/26 path; the stress rule did not trigger in this sparse early window. The old record's Sharpe -0.672 / Sortino -0.763 pair appears as Sharpe -0.763 / Sortino -0.672 in the current platform reporting, i.e. the current caliber swaps the two labels relative to the old records while preserving the identical daily path.

### Frozen Cross-Window Summary For cross-v0.3.3 (ATR-Stress Adoption)

| Period | Role | v0.3.2 Return | v0.3.3 Return | v0.3.2 Max DD | v0.3.3 Max DD | Stress Trigger | Judgment |
|---|---|---:|---:|---:|---:|---|---|
| 2019-2021 | training | +125.82% | +129.25% | 6.70% | 6.28% | 6 half-size buys (2020-03, 2020-09) | improved return and drawdown, identical 53/42 path |
| 2022-2023 | first validation | +17.36% | +17.90% | 11.63% | 11.17% | triggered (2022 stop clusters) | improved on every metric, identical 25/40 path |
| 2024-2026 | recent validation | +58.17% | +58.17% | 9.98% | 9.98% | none | identical path, harmless |
| 2015-2018 | stress validation | +23.21% | +23.21% | 7.38% | 7.38% | none | identical path, harmless |
| 2010-2014 | early supplement | +1.20% | +1.20% | 5.23% | 5.23% | none | identical path, harmless |

Adoption conclusion: the staged evidence supports adopting the frozen ATR-stress rule as `cross-v0.3.3`. In the two windows where the rule triggered (training and the out-of-sample 2022-2023 weak market), both return and max drawdown improved; in the three inactive windows the path was byte-identical to `cross-v0.3.2`. No window deteriorated. The improvement remains modest and concentrated in a small number of half-size buys, so the rule is positioned as drawdown insurance, not an alpha source. Remaining gates before PTrade sync: JoinQuant log/transaction audit (version line, `stress=` lines, ERROR=0, no removed-symbol trades) and then the PTrade parity sync.

### 2026-08-16 ATR-Stress Half-Size Buy Trade Attribution

Period: 2019-01-01 to 2021-12-31, training replay only
Tool: `cross_signal_strategy/research/atr_stress_trade_attribution.py` (read-only, approved training data only)

Purpose: answer whether the six frozen half-size buys "avoided further losses" or "missed a rally" by comparing each realized trade at actual half size against the counterfactual full size with identical entry/exit prices.

Results:
- 2020-03-03 159985: hold return -2.27%, half PnL -101.2, full counterfactual -204.6, half better +103.4.
- 2020-03-05 159928: hold return -1.48%, half -32.2, full -64.4, half better +32.2.
- 2020-03-06 513050: hold return -5.69%, half -257.3, full -514.6, half better +257.3.
- 2020-03-23 159985: hold return -2.78%, half -123.2, full -246.4, half better +123.2.
- 2020-09-15 513880: hold return -1.58%, half -93.5, full -188.7, half better +95.2.
- 2020-09-22 512100: hold return -1.05%, half -62.0, full -125.0, half better +63.0.
- One additional planned stress buy (2020-03-18 159985) did not fill at 09:35 and was excluded.

Interpretation:
- All six realized stress buys were losing trades; halving avoided losses in every case (6:0, no missed rally). Total direct delta +674.3 versus the full-replay difference of +878.2 (the remainder comes from the unfilled plan, compounding, and lot rounding).
- This is consistent with the mechanism: reversal entries taken inside a stop-loss cluster are systematically at elevated risk of further drawdown.
- The counterfactual risk remains: a future stress-period buy could turn into a V-bottom and the half size would miss half the rally. The cost is bounded (half of one slot for at most the 15-day stress window) and the rule expires automatically when stops stop clustering.

## 2026-08-16 Profit-Tiered ATR Tightening Experiment

Version: `cross-v0.3.3-profit-tier-candidate`
Protocol role: user-authorized fixed variant; training-only; Step 0 observation then Step 1 local A/B
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Frozen variant:
- Trailing ATR multiplier ×0.8 when current profit > 5%, ×0.6 when current profit > 15% (multi-factor V2.6 mechanism), applied inside `calc_stop_price` via a new `profit_pct` argument; profit measured as current price over entry cost minus one. Stop floor 5% and cap 15% unchanged.

Step 0 binding observation:
- 36 binding stop-check events (profit > 5% AND unfloored stop above the 5% floor): 4/24/8 in 2019/2020/2021, ETFs 159915/159928/513100/518880; 1 high-tier event.
- Baseline stop distance on binding days 5.02%-7.74%, tightened 5.00%-6.19%.
- Extra-trigger events (tightened stop fires while baseline does not, same day): 0.

Step 1 local A/B:
- Candidate changed 0 filled orders. Total return +125.00%, max drawdown 6.03%, Sharpe 2.262, Sortino 3.581, annual +35.84%/+52.68%/+8.49%, 92 buys/89 sells, 25 ATR stops — all identical to the `cross-v0.3.3` baseline.
- Pre-registered gate "at least 3 filled orders change" failed.

Interpretation:
- The multi-factor V2.6 profit-tier mechanism is an exact no-op in this framework: the frozen entry ATR (median about 1.4% of price) plus the 5% stop floor means most stops are floor-dominated, and on the few binding days no 09:35 price ever fell into the gap between the tightened and baseline stops.
- The giveback weakness (28.4% round-trip rate) is real but is not reachable by multiplier tiering under the current stop construction; addressing it would require changing the floor/stop construction itself, which is a different mechanism and was not part of this locked variant.

Decision:
- Reject the candidate before JoinQuant and validation. Keep official `cross-v0.3.3` unchanged. The family is exhausted; no tier threshold, multiplier factor, peak-profit measurement, profit floor, or per-ETF override search is allowed. The separate pre-registered gold-specific stop direction remains governed by its own future budget entry.

## 2026-08-16 Gold-Specific Stop Experiment

Version: `cross-v0.3.3-gold-stop-candidate`
Protocol role: user-authorized fixed variant; training-only; Step 0 observation then Step 1 local A/B
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Frozen variant:
- 518880 uses stop_floor 0.03 and trailing_atr_mult 2.0 (multi-factor V2.8 values); all other ETFs keep 5%/2.5×; stop cap 0.15 unchanged. `calc_stop_price` gains a `code` argument.

Step 0 binding observation:
- 223 binding gold stop-check days (73/92/58 in 2019/2020/2021); 6 same-day extra-trigger events (2019-07-01/02, 2019-09-09, 2021-08-09/10, 2021-11-24). Gates (10 binding, 3 extra triggers) passed.

Step 1 local A/B:
- Total return +125.00%→+120.96%, max drawdown 6.03%→6.08%, Sharpe 2.262→2.210, Sortino 3.581→3.492; annual 2019 +35.84%→+34.34%, 2020 +52.68%→+53.25%, 2021 +8.49%→+7.33%. Buys 92→94, sells 89→91, gold ATR stops 2→5, 162 changed filled-order positions.
- Per-trade gold attribution: baseline 2019-07 trade exited +9.0% on 2019-08-02 (signal sell); candidate stopped it 2019-07-01 at +4.7% (clipped winner) and the cascade changed the whole path. The 2021-08-09 stop exited two days before the baseline ATR stop for +0.3pp of avoided loss; 2021-11-24 only swapped the sell reason at the same price.

Interpretation:
- Gold's winning reversal trades in this framework tolerate pullbacks of 3-4% below the peak while the bounce develops; the 3% floor exits those pullbacks and clips winners. The multi-factor V2.8 gold-stop result does not transfer because that framework enters gold through a different rotation path with different exit semantics.

Decision:
- Reject the candidate before JoinQuant and validation. Keep official `cross-v0.3.3` unchanged. The family is exhausted; no nearby gold floor/multiplier values and no per-ETF stop extension are allowed.

## 2026-08-16 Profit-Giveback Direct Exit Observation

Version: `cross-v0.3.3-giveback-observation` (read-only counterfactual, no candidate file)
Protocol role: user-authorized fixed variant; training-only Step 0 observation
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Frozen variant:
- At the daily 09:35 stop check: peak closing-price profit ≥ 5% AND current 09:35 profit ≤ peak profit − 3pp → immediate full exit (same-day buys exempt). Everything else unchanged.

Observation result:
- 79 firing events across 21 affected closed trades.
- Per-share delta versus official exits: total -0.352; 2019 -0.380; 2020 -0.101; 2021 +0.129.
- Dominant clips: 2019-02-11 159928 (rule exit 2.245 vs official 2.666, -0.421/share) and 2020-04-17 513050 (1.523 vs 1.927, -0.404/share). Savings on other trades were small (+0.004 to +0.157/share).

Interpretation:
- The framework's payoff is concentrated in a few large trend winners that routinely give back more than 3pp of profit mid-hold before resuming. A profit-giveback exit therefore clips the payoff source while salvaging only small amounts elsewhere. This completes the evidence that profit-protection overlays do not transfer to this framework: break-even floor (-6.3pp), gold stop (clipped winners), and now giveback exit (negative counterfactual) all failed for the same structural reason.

Decision:
- Reject at Step 0 before any candidate. Keep official `cross-v0.3.3` unchanged. The family is exhausted; no activation/giveback threshold or mechanism search is allowed. The official ATR stop remains the only profit protection.

## 2026-08-16 Intraday-High Trailing Anchor Experiment

Version: `cross-v0.3.3-high-anchor-candidate`
Protocol role: user-authorized fixed variant; training-only; Step 0 observation then Step 1 local A/B
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Frozen variant:
- The trailing-high anchor is updated with each session's intraday HIGH instead of its close; stop formula, 2.5× multiplier, 5% floor, 15% cap, and frozen entry ATR are unchanged.

Step 0 binding observation:
- 1604 binding stop-check days (499/580/525 in 2019/2020/2021) and 38 same-day extra-trigger events across all nine ETFs. Gates (10 binding, 3 extra triggers) passed.

Step 1 local A/B:
- Total return +125.00%→+119.40%, max drawdown 6.03%→6.06%, Sharpe 2.262→2.299, Sortino 3.581→3.674; annual 2019 +35.84%→+30.55%, 2020 +52.68%→+53.82%, 2021 +8.49%→+9.26%. Buys 92→94, sells 89→91, ATR stops 25→29, 175 changed filled-order positions.
- Per-trade attribution across the nine changed exits: seven small saves (+0.001 to +0.090 per share) and two clips, dominated by 2019-02-11 159928 (candidate exit 2019-02-26 at 2.232 versus official 2019-04-12 at 2.666, -0.434 per share on an +18.8% winner); total per-share delta -0.228.

Interpretation:
- The peak-day upper wick raises the high anchor into the winner's normal pullback band, turning the noise the close anchor was designed to filter back into stop triggers. This validates the original close-anchor design rule with data.

Decision:
- Reject the candidate before JoinQuant and validation. Keep official `cross-v0.3.3` unchanged. The family is exhausted; no anchor blends, multiplier re-calibrations, or threshold changes are allowed.

## 2026-08-16 Profit-Gated Direct-Sell Matrix Observation

Version: `cross-v0.3.3-profit-gated-matrix-observation` (read-only counterfactual, no candidate file)
Protocol role: user-authorized fixed 4×3 matrix; training-only Step 0 observation
Engine: local replay; JoinQuant remains the performance authority
Data boundary: approved 2018 warm-up plus approved 2019-2021 training data only

Frozen matrix:
- Direct-sell channels: sell-score thresholds 32/35/38/40 crossed with profit bands 2-4%/3-5%/4-6%, bypassing the price-structure confirmation while keeping the 5-day minimum hold and the ADX strong-uptrend exemption.

Observation result:
- 38/40 thresholds: 0 firing events (sell scores that high arrive only after profit leaves the band).
- 32/35 thresholds: 19 and 18 firing events; every variant's total per-share delta was negative (A1 -0.051, B1 -0.077, A2/B2 -0.423, A3/B3 -0.313). The dominant clip is the 513050 +34% winner: its mid-hold pullback satisfies high-score-plus-small-profit and would exit at about 1.523 versus the official 1.927.
- All 12 variants failed the gates; the pre-registered selection rule found no passing variant.

Interpretation:
- The profit band cannot distinguish a small winner that will keep winning from one that will fail; large winners pass through the 2-6% profit zone repeatedly with elevated sell scores on pullbacks. This completes the sell-side evidence: threshold changes, protections, giveback exits, and profit-gated bypasses all fail for the same structural reason.

Decision:
- Reject at Step 0 before any candidate. Keep official `cross-v0.3.3` unchanged. The family is exhausted; no nearby thresholds, bands, or mechanism variants are allowed.

## 2026-08-21 Fixed 14:45 Dual-Timepoint Candidate

Version: `cross-v0.3.3-dual-timepoint-1445-candidate`
Protocol role: user-authorized single fixed variant; training-only local A/B
Engine: local causal 09:35/14:45 replay; JoinQuant remains the performance authority
Data boundary: approved read-only 2018 warm-up plus 2019-2021 training data only; no validation, pressure, recent, full-period, or 2026 price data

Frozen hypothesis and implementation:
- Preserve the official 09:35 path and add one full 14:45 buy/sell decision from minute labels strictly before 14:45, with 14:44 as the final signal minute.
- Recompute the unchanged RSI/KDJ/MACD/ADX/BOLL/MA/ATR scoring path on a provisional T-day bar; use raw partial volume; execute at the 14:45 minute open; share all broker, position, hold, ATR, close-anchor, sold-today, ranking, sizing, and risk state.
- Keep all indicators, parameters, thresholds, ETF pool, fees, and official 09:35 behavior unchanged. Candidate variants: exactly one.

Engineering audit before the completed run:
- The first CLI attempt produced no report because 512100 on 2019-01-02 had 224 pre-14:45 minute rows with missing `prev_close`; the strict daily/minute boundary check stopped the process.
- A failing regression test was added. The frame validator remains strict, while the dual adapter now records an invalid code-date as missing coverage and continues other ETFs. This does not invent a previous close or weaken causality.
- A complete read-only score scan then found usable 14:45 coverage of 1765/2134/2132 and missing counts of 431/53/55 for 2019/2020/2021. The aborted implementation run was not a candidate result or a second variant.

Nominal A/B result:
- Total return: baseline +125.0025%, candidate +84.9970%.
- Maximum drawdown: 6.0316% -> 7.4919%.
- Closed-trade win rate: 56.18% -> 47.66%.
- Profit/loss ratio: historical baseline 4.440, candidate 2.8131.
- Annual win rates: 56.00%/58.62%/54.29% -> 53.57%/48.65%/42.86%.
- Buys/sells: 92/89 -> 109/107.
- Positive-to-negative round trips: 31 -> 40; only 512100 had a per-code reduction.
- Maximum consecutive losing trades: 5 -> 5.

Double-friction result:
- Total return: +112.7772% -> +73.1887%.
- Maximum drawdown: 6.2540% -> 8.2763%.

Frozen gate outcome:
- Failed return retention, nominal drawdown, candidate profit/loss ratio, overall win rate, annual win-rate consistency, total round-trip reduction, cross-ETF round-trip breadth, stressed return retention, and stressed drawdown.
- Passed only the buy/sell count ceiling, non-worsening maximum loss streak, and nonzero annual score coverage requirements.

Interpretation and decision:
- A second full intraday signal pass increases reaction frequency but lowers accuracy. Partial-day indicator changes are too noisy to distinguish durable reversals from normal intraday movement, so the mechanism adds recycling, increases positive-to-negative outcomes, and worsens both nominal and stressed drawdown.
- Decision: `STOP`. Reject before JoinQuant and PTrade, keep formal `cross-v0.3.3` unchanged, exhaust the family, and do not search nearby times, per-ETF times, afternoon-only sides, indicator subsets, thresholds, or hold/cooldown interactions.

## 2026-08-22 Fresh-Unextended Fast-Entry Candidate Preparation

Version: `cross-v0.3.3-fresh-unextended-entry-candidate`
Protocol role: user-authorized single fixed variant; local engineering screen followed by one official JoinQuant training run
Data boundary: approved read-only 2018 warm-up plus 2019-2021 training data only; no validation data inspected

Frozen hypothesis and rule:
- Some 50-59 score observations may be delayed primary entries rather than generally weak entries when reversal contribution is already at least 35, every contributing bullish cross is only age 0/1, and the T-1 close has extended no more than 1 ATR14 from the earliest contributing cross close.
- The official score>=60 primary queue remains first and unchanged. The candidate can fill only a slot left by that queue.
- All existing RSI overheat, price-position, blocked-combination, sell-score, ATR cooldown, sizing, sell, and risk rules remain unchanged.
- Exactly one variant is allowed: score 50-59, reversal>=35, maximum cross age 1, maximum extension 1 ATR. No neighboring or per-ETF search is permitted.

TDD and causal checks:
- Candidate rule, main-queue priority, no-displacement, and future-row rejection tests were written and observed failing before implementation.
- The local adapter records `max_data_date` and rejects any row later than `signal_date`.
- The standalone JoinQuant candidate derives cross close only from the daily frame already ending at `prev_date`; T-day data does not enter the signal.

Local engineering screen (not performance authority):
- Baseline return +125.00%, drawdown 6.03%, win rate 56.18%, profit/loss ratio 4.878.
- Candidate return +98.11%, drawdown 6.03%, win rate 49.48%, profit/loss ratio 3.197.
- Double-friction baseline/candidate return +108.15%/+81.01%; drawdown 6.39%/7.78%.
- The fresh channel filled 19 buys and appeared in 2019, 2020, and 2021, so it is neither sparse nor a no-op.

Interpretation and next step:
- The local result is a clear adverse warning, especially for win rate and doubled-friction drawdown, but local minute execution is known not to reproduce changed JoinQuant order paths authoritatively.
- Do not reject, approve, or retune from the local values. Run the standalone JoinQuant candidate once over 2019-01-01 to 2021-12-31 with CNY 20,000 and daily frequency, then apply the predeclared official gates.
- Official nominal gates versus the official `cross-v0.3.3` baseline: win rate +3 percentage points or more; at least 3 fewer positive-to-negative round trips; no worse maximum drawdown; at least 95% total-return retention; profit/loss ratio at least 3.0; and positive return in every training year. Run the unchanged rule under doubled friction only if all nominal gates pass.
- Formal JoinQuant/PTrade `cross-v0.3.3` remains unchanged.

Official JoinQuant outcome (2026-08-22):
- The frozen build and fingerprint were confirmed as
  `20260822.1-candidate` / `25783cc30ba4`.
- Candidate total return was +111.14% versus the official baseline +129.25%; maximum drawdown was 6.29% versus 6.28%.
- Candidate win rate was 49.0% versus 55.8%, profit/loss ratio was 3.904 versus 5.297, and positive-to-negative round trips increased from 31 to 39.
- The added channel closed 4 winners and 15 losers; two large winners supplied about 88.5% of its gross profit.
- Decision: `REJECT`. The family is exhausted, the standalone JoinQuant file is archived, and formal `cross-v0.3.3` remains unchanged. No doubled-friction or validation run is permitted because the nominal gates failed.

## 2026-08-22 Late-MACD / BOLL-Upper Step 0 Observation

- Source: 98 official filled buys from formal `cross-v0.3.3` build `20260820.1`, fingerprint `77e44d93d255`.
- Frozen shape: T-1 MACD bullish-cross age 0; active RSI and KDJ bullish crosses each age 1/2; close at or above BOLL upper.
- Frozen gate: at least 3 matches across at least 2 training years before one new-buy veto candidate may exist.
- Observed matches: 2, both in 2019 — `513100` on 2019-03-15 and `159928` on 2019-12-31.
- Decision: `STOP`. The 3-event/2-year gate failed; no candidate was created, no threshold was relaxed, and formal `cross-v0.3.3` remains unchanged.

User-directed candidate override:
- After disclosure that the two matched trades were 1 win/1 loss and that directly vetoing both would have reduced realized PnL by about CNY 355.20, the user explicitly requested the exact JoinQuant rule for a portfolio-path backtest.
- Standalone candidate: `smart_trade_joinquant_cross_signal_etf_late_macd_boll_filter_candidate.py`.
- Version/build/fingerprint: `cross-v0.3.3-late-macd-boll-filter-candidate` / `20260822.2-candidate` / `a46fff884685`.
- This does not modify or promote formal `cross-v0.3.3`; no rule relaxation or neighboring variant is authorized.

## 2026-08-23 v0.4 Dimension-Capped Score Invalid-Implementation Provenance

- Classification: `invalid_implementation`. The first local run did not execute the approved frozen rule: sell RSI contributed 12 instead of 10, sell MACD contributed 5 instead of 4, and an ordinary raw sell conflict on a buy candidate could be suppressed by ADX instead of blocking the buy independently of ADX and holding period.
- Preserved artifact: `cross_signal_strategy/reports/dimension_capped_score_v04_invalid_implementation_2019_2021.md`, 6,606,607 bytes, SHA-256 `e4a1f30e02f2861b8cdb5f0740d27ef07acce002cb5b9307e86b8154aa7b8c76`. This is provenance only and is not the canonical corrected report.
- Observed invalid-run materiality: 196 filled-order days changed, with 62/64/70 changes in 2019/2020/2021. Closed trades were 89 baseline versus 85 candidate, a 95.51% retention rate.
- Observed invalid-run nominal baseline/candidate: return +125.00%/+78.13%; annualized return 31.13%/21.29%; maximum drawdown 6.03%/6.37%; win rate 56.18%/51.76%; Sharpe 2.262/1.672; Sortino 3.581/2.533; profit/loss ratio 4.878/2.831; buys 92/88; sells and closed trades 89/85.
- Observed invalid-run annual baseline/candidate: 2019 +35.84%/+21.45%; 2020 +52.68%/+43.89%; 2021 +8.49%/+1.93%.
- Observed invalid-run doubled-friction baseline/candidate: return +108.15%/+63.32%; annualized return 27.77%/17.82%; maximum drawdown 6.39%/6.93%; win rate 51.69%/45.88%; Sharpe 2.039/1.422; Sortino 3.186/2.125; profit/loss ratio 3.966/2.347. Candidate annual returns were +18.61%/+39.44%/-1.26%.
- Observed invalid-run gate reasons: candidate win rate did not strictly improve; candidate return, Sharpe, Sortino, and profit/loss ratio each retained less than 95% of baseline; doubled-friction return retained less than 95%; doubled-friction win rate was below baseline. These observations do not approve or reject the approved rule.
- Governance correction: retract the former `STOP`/exhausted conclusion for the approved v0.4 family. Exactly one corrective replay of the same rule is pending after the implementation-only fixes; no nearby score, threshold, filter, ranking, indicator, ETF/year, friction, protection, hold, or portfolio variant is permitted.
- Approved empirical status: not run. No corrected canonical report exists, no JoinQuant/PTrade candidate was created, and both formal `cross-v0.3.3` files remain unchanged.
- Validation influence: none. No reserved period was read or used.

## 2026-08-23 v0.4 Dimension-Capped Score Corrective Replay

- Unique empirical command: `python -m cross_signal_strategy.research.dimension_capped_training_ab`, executed once. The same process ran to completion; the frozen CLI contract maps the emitted failed gate to exit `1`, which is the designed `STOP` status rather than a runtime error. Complete stdout was captured separately and normalizes exactly to the canonical report; PowerShell redirection stored CRLF line endings while the canonical writer stored LF.
- Corrected canonical artifact: `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`, 9,740,215 bytes, SHA-256 `14395dbf09f506c914bd5da241454b3af4291cf2e701e7653738f50f4840ccd6`. Approved rule fingerprint: `0493e7fbeb80cdaa6d8ab0fe9c47d3fa8ca8b680e6556ca805de4d6e742f7f63`.
- Invalid provenance remains byte-preserved separately at `cross_signal_strategy/reports/dimension_capped_score_v04_invalid_implementation_2019_2021.md`, 6,606,607 bytes, SHA-256 `e4a1f30e02f2861b8cdb5f0740d27ef07acce002cb5b9307e86b8154aa7b8c76`. Both reports coexist; the invalid report does not determine the corrected conclusion.
- Materiality: 196 changed filled-order days, split 62/64/70 across 2019/2020/2021. Closed trades were 89 baseline versus 85 candidate, retaining 95.51%.
- Corrected nominal baseline/candidate: return +125.00%/+78.13%; annualized return 31.13%/21.29%; maximum drawdown 6.03%/6.37%; win rate 56.18%/51.76%; Sharpe 2.262/1.672; Sortino 3.581/2.533; profit/loss ratio 4.878/2.831; buys 92/88; sells and closed trades 89/85. Annual returns were +35.84%/+21.45% in 2019, +52.68%/+43.89% in 2020, and +8.49%/+1.93% in 2021.
- Corrected doubled-friction baseline/candidate: return +108.15%/+63.32%; annualized return 27.77%/17.82%; maximum drawdown 6.39%/6.93%; win rate 51.69%/45.88%; Sharpe 2.039/1.422; Sortino 3.186/2.125; profit/loss ratio 3.966/2.347. Candidate annual returns were +18.61%/+39.44%/-1.26%.
- Audit reconciliation: nominal and doubled-friction candidate arms each persisted 6,570 score attempts (6,111 scored, 459 skipped) and 186 planned orders (173 fills). Each arm reconciles independently against its own replay order sequence and buy/sell performance counts.
- Failed gates: candidate win rate did not strictly improve; nominal return, Sharpe, Sortino, and profit/loss ratio each retained less than 95% of baseline; doubled-friction return retained less than 95%; doubled-friction win rate was below baseline.
- Terminal action: `STOP`. The corrected failure is appended as a distinct ledger entry, failed/non-adopted count increases 78→79, the correction slot closes, and the v0.4 family is exhausted. No parameter, gate, threshold, pool, year, friction, ranking, hold, ATR, ADX, or protection rule was changed after seeing the result.
- JoinQuant/PTrade status: no candidate created. Both formal `cross-v0.3.3` files remain unchanged. Validation influence: none; no reserved period was read or used.











