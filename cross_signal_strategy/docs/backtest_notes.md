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
Code files: `cross_signal_strategy/baseline_report.py`, `cross_signal_strategy/local_training_run.py`
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
Code file: `cross_signal_strategy/local_signal_adapter.py`
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
Code files: `cross_signal_strategy/local_adjustment.py`, `cross_signal_strategy/local_signal_adapter.py`, `cross_signal_strategy/local_training_run.py`
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
Code files: `cross_signal_strategy/order_path_diagnostics.py`, `cross_signal_strategy/local_backtester.py`
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
Code files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local_order_planner.py`
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
Code file: `cross_signal_strategy/baseline_report.py`
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
Code file: `cross_signal_strategy/trade_diagnostics.py`
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

Version: temporary JoinQuant probe `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_probe_513880.py`
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
Code files: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`, `cross_signal_strategy/local_order_planner.py`
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
Code file: `cross_signal_strategy/sell_diagnostics.py`
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
Code file: `cross_signal_strategy/attribution_diagnostics.py`
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
- `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_pool_candidate.py` was created as a temporary training-confirmation file.
- It only changes `STRATEGY_VERSION` and removes `510300.XSHG`, `510880.XSHG`, and `159920.XSHE` from `get_default_etf_pool()`.
- It is not the official adopted strategy until JoinQuant 2019-2021 training confirms the improvement and logs/transactions are reviewed.

Can this result be used to change rules? candidate only
Reason: ETF-pool deletion is highly exposed to training-window selection bias. The `510300/510880/159920` removal candidate improves local training return and drawdown, but it should be confirmed in JoinQuant training before adoption and must later face reserved validation.

### JoinQuant Training Confirmation For ETF-Pool Candidate

Version: `cross-v0.3.0-pool-candidate`
Code file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_pool_candidate.py`
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
Candidate file: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
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
