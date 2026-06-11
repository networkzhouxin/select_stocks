# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese ETF quantitative trading strategy system. Automated buy/sell signal generation, ATR-based risk management, and momentum ranking for 3-5 widely-traded Chinese ETFs. Targets small capital (initial 20K CNY). Multi-platform: JoinQuant (聚宽) for backtesting, PTrade for live trading.

## Working Process

- **代码修改前必须先确认**：任何代码修改，先向用户说明方案和影响，等用户明确同意后才能动手。不要直接改代码。即使是在探索性讨论中提出的优化方向，也必须等用户确认"实施"后才执行。

## Key Files

- `smart_trade_joinquant_v10_etf.py` — **V10.0 JoinQuant, highest absolute return** (58% return over 10yr, 27.9% max drawdown, 3 ETFs)
- `smart_trade_ptrade_v10_etf.py` — V10.0 PTrade version (production-ready, dual-mode backtest+live)
- `smart_trade_joinquant_v11_etf.py` — V11.0 JoinQuant (5 ETFs, expanded pool)
- `smart_trade_joinquant_v13_etf.py` — **V13.0 JoinQuant, best risk-adjusted of signal-driven** (57.5% return, ~12.5% max drawdown, bear market half-position)
- `smart_trade_joinquant_v15_etf.py` — **V15.5 JoinQuant, original momentum rotation** (210% return over 11yr, ~10.8% annualized, 5万起始, momentum rotation, optimized 10-ETF pool: 4 A-share + 3 cross-market + 3 cross-asset)
- `smart_trade_joinquant_v15_7_etf.py` — **V15.7 JoinQuant** (212.8% return, 10万起始, buy price fix + bond slot-filling, 10-ETF pool)
- `smart_trade_joinquant_v15_7_expanded_etf.py` — **V15.7-Expanded JoinQuant** (267.9% return, 10万起始, 12-ETF pool: +日经+中概互联)
- `smart_trade_joinquant_v15_9_etf.py` — **V15.9 JoinQuant, current best of momentum** (256.9% return, 2万起始, 12-ETF + unified max_hold=3)
- `smart_trade_joinquant_multifactor_etf.py` — **Multi-Factor V2.10 JoinQuant** (+385.88%阶段验证 with 防追高+后备补位; V2.10 core +371.7%, 2万起始, 7-factor scoring, 12-ETF pool, 聚宽实测)
- `smart_trade_ptrade_multifactor_etf.py` — **Multi-Factor V2.10 PTrade版** (实盘/模拟部署用, 策略逻辑与聚宽版同步; PTrade回测仅验证无报错)
- `smart_trade_ptrade_v15_7_etf.py` — **V15.7 PTrade版** (实盘/模拟部署用, 10-ETF pool)
- `策略说明文档.md` — Complete strategy documentation for V15.x (Chinese)
- `多因子ETF策略说明文档.md` — Complete strategy documentation for Multi-Factor V2.10 (Chinese)
- `docs/帮助.html` / `docs/财务数据.html` — Latest local official PTrade API/reference docs
- `smart_trade_v10_tdx.txt` / `smart_trade_v10_tdx_main.txt` — TDX (通达信) indicator formulas

## Architecture

### Strategy Signal Pipeline

1. **Data**: `get_price()` fetches 120 daily bars ending at T-1 (previous trading day) — never uses today's data for signals
2. **Indicators**: MA10/20/60, EMA12/26, ATR(14), KDJ(9,3,3), RSI(6), MACD(12,26,9), ROC(20), VR(量比)
3. **Signals**: 4 buy conditions (BU1-BU4, weighted 1.0-1.5) + 4 sell conditions (SE1-SE4) → raw score
4. **Trend adjustment**: 5-dimension trend score (0-5) mapped to coefficient (-2 to +2), adjusts buy/sell scores
5. **Signal grading**: Score thresholds → levels 0-3 (强买/中买/弱买)
6. **Ranking**: `sort_score = buy_score × 0.6 + risk_adj_momentum × 0.4`
7. **Position sizing**: `base_ratio × signal_strength × volatility_inverse`, capped at 95%
8. **Execution**: ATR trailing stop (2.5×), max loss stop (3.5×), trend hold mode (score≥4 + profitable → ATR-only)

### Platform Adaptation (JoinQuant → PTrade)

| Aspect | JoinQuant | PTrade |
|--------|-----------|--------|
| Stock codes | `.XSHG` / `.XSHE` | `.SS` / `.SZ` |
| Total value | `context.portfolio.total_value` | `context.portfolio.portfolio_value` |
| Cash | `context.portfolio.available_cash` | `context.portfolio.cash` |
| Position amount | `pos.total_amount` | `pos.amount` |
| Position cost | `pos.avg_cost` | `pos.cost_basis` |
| Current price | `get_current_data()[code].last_price` | `data[code].price` / `get_snapshot()` |
| Halt check | `current_data[code].paused` | `get_stock_status([code], 'HALT')` |
| Scheduling | `run_daily(func, time)` | `run_daily(context, func, time)` |
| Backtest mode | `run_daily` works normally | `handle_data` drives all logic (daily `run_daily` fixed at 15:00) |
| Live detection | N/A | `is_trade()` returns True for live |
| Snapshot fields | N/A | `get_snapshot()` returns `last_px`, `high_px`, `preclose_px` etc. |

### PTrade Live Trading Constraints

- **`run_daily` + `run_interval` total ≤ 5**: exceeding causes thread blocking, tasks silently won't fire.
- **`order_target` has 6-second sync delay**: portfolio syncs with broker every 6s. Calling `order_target(code, 0)` twice within 6s will duplicate the sell order. Use `g.sold_today` flag to guard.
- **`order` without `limit_price`**: system uses `get_snapshot` latest price; if snapshot fails, order fails.
- **`get_price`/`get_history` not thread-safe**: don't call from `run_daily` and `handle_data` simultaneously.
- **Persistence**: `g` is auto-pickled. Variables prefixed with `__` (e.g. `g.__is_live`) are excluded — use this for non-serializable objects. On restart, `initialize` runs first, then persisted data overwrites.
- **Official docs**: Latest local PTrade docs are under `docs/帮助.html` and `docs/财务数据.html`.
- **Suspended/zero-volume data**: Official docs state `get_price/get_history` do not skip suspended days; suspended daily bars are filled from previous data with `volume=0`. PTrade `_get_price_data()` filters `volume <= 0` to mimic JoinQuant `skip_paused=True`.
- **Order status callbacks**: PTrade order status `"5"` means 部撤, `"6"` means 已撤, `"9"` means 废单. Treat `"5"` with `business_amount > 0` as partial fill waiting for trade callback, not a pure failure.
- **Broker**: 国金证券 PTrade.

### Capital Tiers (Multi-Factor V2.10)

| Tier | Total Assets | Max Holdings | Base Position |
|------|-------------|--------------|---------------|
| micro | <1.5万 | 3 | 70% |
| small | 1.5-5万 | 3 | 70% |
| medium | 5-10万 | 3 | 65% |
| large | >10万 | 3 | 65% |

> **V2.10**: Unified max_hold=3 for all tiers. ETF prices are low enough that even 2万 can hold 3 positions. The old tier restrictions (micro=1, small=2) created an unnecessary structural disadvantage for small capital.

## Critical Design Rules

- **No future functions**: Signals always computed on `prev_date` (T-1) data. Current price used only for stop-loss execution and order placement.
- **All parameters are academic defaults**: ATR(14), MACD(12,26,9), KDJ(9,3,3), RSI(6). Zero parameter optimization — this is intentional to avoid overfitting.
- **No profit-taking**: V11.1 proved that partial profit-taking (+20% sell half) destroys trend-following. 盈亏比 dropped from 3.7:1 to 1.14:1. Let profits run via ATR trailing stop only.
- **Stop loss clamped to [5%, 15%]**: `stop_floor=0.05` prevents noise shakeout (V2.7 WF验证), `stop_cap=0.15` prevents excessive single-trade loss. Gold ETF(518880) exception: `stop_floor=0.03` (V2.8).
- **Trend hold mode**: When trend_score ≥ 4 AND profitable → skip signal-based selling, use only ATR stop. Core mechanism for capturing big trends.
- **Cooldown**: 5-day cooldown between buy/sell signals on same ETF to avoid whipsaws.
- **ETF correlation matters**: Don't add 510050 (overlaps 510300) or 159901 (overlaps 159915+510300). Only add truly uncorrelated ETFs like 510880 (红利) and 512100 (中证1000).
- **Highest price uses closing price, not intraday high**: Intraday highs contain noise (upper wicks/spikes). ATR multiplier (2.5×) is calibrated against closing prices — using intraday high would systematically tighten stops, contradicting "let profits run".

## Platform Backtesting Rules

- **JoinQuant backtest is the authority for strategy performance.** `run_daily` executes at the exact time specified (09:30, 09:35, 15:00, 15:30), matching real trading behavior.
- **PTrade daily backtest CANNOT validate strategy returns.** `run_daily` and `handle_data` are both fixed at 15:00 regardless of time parameter — all logic executes at close price in a single pass. This fundamentally distorts entry timing, stop-loss behavior, and signal response vs real 09:35 execution. PTrade V10 backtest returned +6.26% vs JoinQuant's +45.18% — the gap is caused by the backtest mechanism, not strategy quality.
- **PTrade daily backtest is only useful for verifying code runs without errors.**
- **PTrade live trading matches JoinQuant**: `run_daily` honors the specified time (00:00~23:59), so the 09:30/09:35/15:00/15:30 schedule works identically to JoinQuant.
- **Workflow**: JoinQuant backtest (validate returns) → PTrade backtest (validate no errors) → PTrade live.

## Version History Lessons

- **V6-V7**: Individual stocks, high risk (49% max drawdown), poor for small capital
- **V8**: Switch to ETF improved everything; ATR stops introduced
- **V9**: Over-complicated (adaptive MACD, regime detection) → worse results
- **V10**: Simplified back, added trend hold + momentum ranking → **optimal** (45.18%)
- **V10.1/V10.2**: Attempted KAMA/adaptive indicators → degraded performance, confirming V10.0 is the complexity ceiling
- **V11**: Only change is ETF pool (3→5), all logic identical to V10.0
- **V12**: Two changes tested: (1) removed signal-based selling, (2) replaced 510300 with 512100 (中证1000). Result: total return dropped from 58% to 42.5%. Signal sells look bad standalone (9 trades, 8 losses) but serve critical capital-recycling role — without them, capital gets trapped in stagnant positions. 512100 contributed +5.8%, nearly identical to 510300's +5.4%, so the swap was neutral. **Lesson: don't remove signal sells; don't swap ETFs based on volatility alone — trend persistence matters more.**
- **V13**: One change vs V10: bear market position reduction (all ETFs below MA60 → halve position size). Result: 57.5% return (vs V10's 58%), max drawdown ~12.5% (vs V10's 27.9%). Worst years dramatically improved: 2018 -4.0% (was -15.2%), 2023 -7.6% (was -25.8%). 10-year 12 bear-mode triggers, condition strict enough to avoid false positives. **Lesson: same return with half the drawdown — bear market filter is the single most valuable risk-management addition. V13 is the best risk-adjusted version.**
- **V14**: V13 + 5 ETF pool (added 510880 红利, 512100 中证1000) + bear market detection decoupled from ETF pool (沪深300 < MA60 and MA60 declining). Result: 37.98% return, 24.21% max drawdown, 120 trades, P/L ratio 1.419, Sharpe -0.097. Two problems: (1) 510880 红利ETF is low-volatility mean-reverting, fundamentally unsuited for trend-following, diluted alpha; (2) more trades on weaker ETFs dragged P/L ratio from 1.675 to 1.419. **Lesson: expanding ETF pool hurts when new ETFs lack trend persistence. 红利ETF's defensive nature contradicts trend-following. Stick with 3 core ETFs. V13 remains the best version of the signal-driven framework.**
- **V13.1** (reverted): V13 + expanded ETF pool from 3 A-share to 10 cross-asset (same pool as V15.7) + bond fallback when no holdings/signals + buy price fix. Result (2万起始): 44.5% total return (**-13pp vs V13.0**), 11.4% max drawdown, 3 loss years (worst -8.1%). 114 buy signals over 11yr across 10 ETFs, ATR stop rate 85% — signals designed for A-share ETFs (MA20 breakout, MACD golden cross, RSI oversold) systematically misfired on gold/Nasdaq/soybean (90%+ stop rate on these). Bond fallback triggered 15 times (useful but insufficient). **Lesson: signal-driven and momentum rotation are fundamentally different frameworks — the same ETF pool does NOT work for both. Signal-driven (BU1-BU4) relies on A-share-specific technical patterns; cross-asset ETFs have different price dynamics that these signals can't capture. V13's 3-ETF A-share pool is optimal for its framework. Don't cross-pollinate ETF pools between signal-driven and momentum frameworks. V13.0 remains the best signal-driven version.**
- **V15.0**: Completely new framework — momentum rotation instead of signal-driven. Every 3 trading days rebalance, always hold top N ETFs by risk-adjusted momentum (ROC20/volatility). 10-ETF pool (3 broad-base + 4 sector + 2 cross-market + 1 gold). Filters: positive momentum + price > MA20, otherwise don't buy; if all ETFs fail filter → auto cash. ATR trailing stop as safety net. First iteration with 沪深300-based bear market filter triggered 234 times (too sensitive for weekly rotation, suppressed rebounds). After removing bear filter: 223.4% total return over 11yr (2万起始, ~11.3% annualized). **Lesson: momentum rotation dramatically improves capital utilization (~90% vs ~40%); explicit bear market filter is counterproductive for weekly rotation — "natural cash" (no positive momentum ETF → auto empty) is sufficient.**
- **V15.1**: Dual momentum filter (ROC20>0 AND ROC60>0), dynamic ATR stop (2.0x in high volatility, 2.5x normally). With original 10-ETF pool (8 A-share + 2 cross-asset): 170% return (2万起始), 119.5% (5万起始). Then **ETF pool restructured** from 8 A-share + 2 cross-asset → 4 A-share + 3 cross-market + 3 cross-asset. New pool: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 513100纳指, 513500标普500, 159920恒生, 518880黄金, 511010国债, 159985豆粕. Result (5万起始): **210% total return (~10.8% annualized), worst year -1.3% (2018), only 2 loss years both <1.5%**. Key findings: (1) 511010国债ETF bought 86 times with 0% stop rate — acts as "productive cash", earning bond returns instead of idle cash; (2) 513500标普500 + 513100纳指 account for 27% of trades, providing strong returns when A-shares weak; (3) 159985豆粕 has 42% stop rate, weakest link but marginal impact. **Lesson: cross-asset diversification is the single biggest improvement — same momentum framework, just better ETF pool structure, nearly doubled returns (119.5%→210%) while reducing worst year from -8.7% to -1.3%. Bond ETF as "productive cash" is a key innovation. V15.1 with optimized pool is the current best version.**
- **V15.2** (reverted): Soft ROC60 filter (allow -10%<ROC60<0 at 70% position). Result: 140% return — worse than V15.1's 170%. Mid-term negative momentum trades are fundamentally bad regardless of position sizing. Reverted to V15.1.
- **V15.3** (reverted): Two changes: (1) medium max_hold 3→2, base_ratio 0.70→0.80; (2) 20% switch threshold (new candidate must beat weakest holding by 20%+ to replace). Result: 225% total return (+15pp vs V15.1) but max drawdown 22.4% (vs ~15%). The switch threshold reduced trades by 24% (547 vs 723 buys) with 67% win rate on rotation sells. However, concentrated positions amplified volatility — worst year -3.2% (2021) vs V15.1's -0.9%. **Lesson: switch threshold is effective at reducing churn, but max_hold=2 concentrates risk too much. The two changes have opposite risk profiles and should not be bundled.**
- **V15.4** (reverted): V15.1 + 20% switch threshold only (max_hold stays at 3). Result: 185.9% total return, 16.5% max drawdown, Sharpe 0.607, 盈亏比 1.819 (highest of all versions). Trade quality improved but total return dropped 24pp vs V15.1. With 3 holdings, the threshold protects the weakest #3 position from being replaced, causing capital to stay in mediocre positions too long. **Lesson: switch threshold works with 2 holdings (V15.3) but backfires with 3 holdings (V15.4). The threshold's value depends on portfolio concentration — it helps when protecting strong positions but hurts when shielding weak ones. V15.1's simple "always pick top N" remains optimal for max_hold=3.**
- **V15.6** (reverted): Replaced 159985豆粕ETF with 162411华宝油气. Result (10万起始): 202.1% total return, 12.43% max drawdown, 2 loss years (-0.2%, -2.3%). 华宝油气 stop rate 38.1% (vs 豆粕42%, marginal improvement), but total return dropped 8pp. 2022 loss worsened due to oil price volatility. **Lesson: commodity slot in pool is for diversification "insurance", not alpha. Swapping one commodity for another has negligible impact. Both豆粕 and 华宝油气 are the weakest link — the key is having one uncorrelated commodity exposure, not which one. Stick with 豆粕 (original).**
- **V15.7**: Two changes: (1) buy price fix — use T-day 09:35 real-time price (`current_data[code].last_price`) instead of T-1 close (`sig['close']`) for share calculation and stop-loss baseline; (2) bond slot-filling — when candidates < max_hold, fill empty slots with bond ETF. Result (10万起始): 212.8% total return, 13.45% max drawdown. Bond slot-filling never triggered in 11yr backtest — bond ETF's stable positive momentum means it naturally enters top-N via regular ranking whenever few candidates qualify. **Lesson: bond slot-filling is logically correct but redundant in practice. The buy price fix is the only meaningful change. V15.7 ≈ V15.5 in actual behavior.**
- **V15.8** (reverted): V15.7 + two changes: (1) 6-day cooldown after ATR stop-loss (ban re-buying stopped ETF for 2 rotation cycles); (2) immediately buy bond ETF after stop-loss to avoid cash idle. Result (10万起始): 171.8% total return (**-41pp vs V15.7**), 14.37% max drawdown, worst year -4.5%. Cooldown reduced stops only marginally (155 vs 163) but blocked 56 buy opportunities (651 vs 707 buys). Many post-stop re-entries are correct — ETF bounces back within days, and immediate re-buy captures the rebound. Bond parking (80 occurrences) over-allocated to bonds, crowding out stronger momentum candidates. **Lesson: in V15's momentum rotation, post-stop "whipsaw" re-entries are often CORRECT re-entries, not waste. Cooldown destroys value by blocking profitable rebounds. Don't add defensive mechanisms that fight the core momentum signal. V15's ATR stops are mostly noise-triggered, and the correct response IS to re-enter when momentum confirms. V15.7 remains the best version.**
- **V15.7-Sector** (experimental, not adopted): V15.7 framework with sector ETF pool (6 A-share sector: 军工/医药/有色/消费/证券/芯片 + 2 cross-market: 纳指/标普 + 2 cross-asset: 黄金/国债). Result (10万起始): 201.1% total return (**-12pp vs V15.7**), 18.5% max drawdown (**+5pp worse**), worst years -5.5% and -4.4% (vs original's -2.2% and -1.1%). Cross-market+cross-asset ETFs (纳指/标普/黄金/国债) accounted for 52% of all trades, proving they are the real return engine in both versions. Sector ETFs had short-lived momentum spikes followed by reversals that triggered frequent stops, especially in 2021-2022 when sectors rotated violently. **Lesson: sector ETFs are strictly worse than broad-base ETFs for momentum rotation — higher volatility does NOT mean higher returns, it means more stop-loss friction. The original 4+3+3 pool structure (broad-base + cross-market + cross-asset) is optimal. Don't replace broad-base with sector ETFs.**
- **V15.7-Global** (experimental, not adopted): V15.7 framework with no A-share equities — pure overseas+cross-asset pool (纳指/标普/恒生/中概互联/黄金/国债/豆粕/华宝油气/日经/南方原油). Result (10万起始): 161.9% total return (**-51pp vs V15.7**), 14.5% max drawdown. Missed A-share rallies entirely (2015 +14%→0%, 2019 +20%→+8%, 2025 +27%→+7%). 华宝油气+南方原油 combined 162 buys with 37% stop rate — too many correlated energy assets. **Lesson: A-share equities are not a drag — they are an irreplaceable alpha source. The momentum framework already auto-reduces A-share exposure when weak. Permanently removing A-share just forfeits upside. Also, QDII-heavy pools face real-world premium/quota constraints.**
- **V15.7-Expanded**: V15.7 pool expanded from 10→12 ETFs by adding 513880日经ETF(2019+) and 513050中概互联ETF(2017+). Pool: 4 A-share + 5 cross-market + 3 cross-asset. Result (10万起始): **267.9% total return (+55pp vs V15.7)**, **12.73% max drawdown (-0.7pp better)**, 2 loss years (-1.1%, -2.6%). 日经ETF: 45 buys, 26.7% stop rate; 中概互联: 55 buys, 25.5% stop rate — both genuinely active with healthy stop rates. Key improvement years: 2017 +22.0% (vs +15.8%), 2020 +20.5% (vs +13.4%), 2024 +15.5% (vs +7.6%). **Lesson: wider cross-market diversification compounds the benefit — Japan and China ADR provide momentum opportunities uncorrelated with existing pool. Unlike sector expansion (which hurt) or A-share removal (which hurt), adding genuinely uncorrelated cross-market ETFs is a Pareto improvement (higher return + lower drawdown). The 4+5+3 structure is the new optimum.**
- **V15.9** (**current best**): V15.7-Expanded + unified max_hold=3 for all capital tiers. Old tiers: micro=1/small=2/medium=3/large=3; new: all=3. Rationale: ETF prices are low enough (100-1100元/手, except 国债14000元/手) that even 2万 can hold 3 positions. Result (2万起始): **256.9% total return**, 14.15% max drawdown, 2 loss years (-2.0%, -3.8%). Critical improvement: **2021 return -2.0% vs old small-tier's -7.3%** — the extra holding slot provides diversification that halves losses in bad years. First trade day verified: 20K successfully bought 3 ETFs (国债100股+消费1800股+沪深300 500股). Stayed in small tier for 9 years until crossing 5万 in 2024. **Lesson: for ETF strategies, capital-tier-based max_hold restrictions are unnecessary — ETFs are cheap enough for even micro capital to hold 3 positions. Unified max_hold=3 eliminates the structural disadvantage of small capital and makes strategy performance consistent across all capital levels. V15.9 = V15.7-Expanded pool (12 ETFs) + unified max_hold.**

## Multi-Factor Strategy V2.10 (smart_trade_joinquant_multifactor_etf.py + smart_trade_ptrade_multifactor_etf.py)

Separate framework from V15.x momentum rotation. Uses 7 classic technical indicators for comprehensive scoring instead of pure momentum. PTrade version synced to V2.10.

### Architecture
- **Factors**: RSI(14), MACD(12,26,9), Bollinger(25,1.8), ROC20(momentum), Volume ratio, KDJ(9,3,3), MA trend(10/20/60). Fixed weights (V2.10 WF验证), discrete scoring buckets, 3-day smoothing.
- **Rotation**: Tuesday + Thursday (fixed weekday calendar, no start-date dependency)
- **Guards**: Switch threshold 8pts, min hold 5 days, ATR trailing stop (dynamic 2.0x/2.5x + 利润分段收紧), MA10 trend stop exemption, volatility-inverse position sizing, new-buy overheat filter (`price/MA20 > 1.08` and `RSI > 75`)
- **Backup fill**: Buy queue is `primary_buy + backup_buy`; if a target-pool candidate is skipped by 防追高, later qualified candidates can fill the slot instead of leaving cash idle.
- **Choppy market logging**: Daily 09:30 market-state log uses MA20 cross count, MA60 slope, and distance to MA60. It is observation-only and must not affect trading unless separately tested.
- **Bear market**: Daily detection at 09:30 (runs in `update_tier`, uses T-1 data). 000300.SS < MA60 and MA60 declining → A-share ETF positions halved (only affects 510300/159915/512100/159928/510880; cross-market and cross-asset ETFs unaffected). Result stored in `g.market_bearish`.
- **Pool**: 12 ETFs (5 A-share + 5 cross-market + 2 cross-asset). Removed 511010 国债ETF (historically removing bond fallback added +18pp, and bond ETF's low-vol mean-reverting nature unsuited for trend-following framework).
- **No bond fallback**: Holds cash when candidates < max_hold.

### Key Iteration Lessons
- **V1.0→V2.0**: Daily rotation + no switch threshold = 4274 trades, -91.3%. Fixed by adding 5-day rebalance, 8pt switch threshold, 5-day min hold, trend-following scoring. Result: +234%.
- **Continuous vs discrete scoring**: Continuous (linear mapping) makes rankings unstable → more unnecessary switching → worse returns. Discrete buckets act as natural "noise filter". **Always use discrete scoring.**
- **ADX adaptive weights**: Tested and removed. Dynamic per-ETF factor weights increased ranking instability. Fixed weights +7pp better. **Don't use ADX adaptive weights.**
- **ROC60 penalty**: Penalizing mid-term negative momentum blocked V-shaped rebounds (2020: -8.5pp). Even light penalty (-15%→×0.7) hurt. **Don't penalize ROC60.**
- **空仓 (empty position) mechanism**: 28 triggers caused missed rebounds. **Don't add empty-position clearing.**
- **趋势持有 (trend hold) protection**: Higher switch threshold for profitable positions — no measurable benefit. **Don't add special trend protection.**
- **QDII volume screening**: Fixing QDII volume to neutral score — no benefit. **Don't screen QDII volume.**
- **4-factor vs 7-factor**: Removing RSI/MACD/KDJ (shared dimension with momentum) reduced returns. 7 "redundant" factors provide **ensemble smoothing** — multiple correlated-but-not-identical factors averaged together stabilize rankings. **Redundancy = stability. Keep all 7.**
- **Dynamic ATR multiplier (2.0x high vol / 2.5x normal)**: No measurable impact in current framework but kept as insurance.
- **国债ETF removal**: Removed 511010 from pool. Bond ETF's low-vol mean-reverting nature unsuited for trend-following. Holding cash is better.
- **Factor weights are NOT optimized**: 0.25/0.18/0.15/0.12/0.12/0.10/0.08 are design values, not backtested. Changing momentum from 0.25→0.30 had negligible effect, confirming strategy is not sensitive to weight precision.
- **Out-of-sample validation**: 2010-2014 (before main backtest period) returned +37% (annualized 6.4%) with incomplete ETF pool, confirming strategy logic is valid and not purely overfitted.

### V2.4→V2.5 Iteration (2026-05, on JoinQuant)
Three optimization attempts, two succeeded:

1. **止损豁免上限10%** (+9pp): When stop triggers but ETF is still in target, exempt ONLY if drawdown from highest <10%. If ≥10%, force stop regardless of score. Prevents "score lag" problem where price drops faster than scoring updates. Added `force_stopped` guard to prevent same-day rebuy after force-stop. Sample-out (2010-2014) confirmed improvement.

2. **候选不足空仓** (**-16pp**): When candidates < max_hold, raised keep threshold for existing positions from 55→60. Result: worse returns, higher drawdown. **Lesson: don't force cash holding — let strategy decide what to keep. Same lesson as V2.x 空仓机制. Reverted.**

3. **去掉8分换仓门槛** (**-115pp**): Removed switch_threshold=8.0. Result: trades exploded from 529→792, return collapsed. **Lesson: 8pt switch threshold is load-bearing — 不要动. Reverted.**

4. **熊市A股减仓** (+12.8pp, 回撤-0.9pp): 沪深300 < MA60 and MA60 declining → A-share ETF positions halved. Only affects 5 A-share ETFs; cross-market/cross-asset ETFs unaffected. MA20 tested with near-identical results, MA60 retained for longer track record. Condition triggered ~12 times in 10yr. **Lesson: same as V13 — bear market filter is zero-cost risk reduction.**

### Performance
| Version | Period | Capital | Return | Annualized | Max DD | Sharpe | Loss Years |
|---------|--------|---------|--------|-----------|--------|--------|-----------|
| V2.4 | 2015-2026 | 2万 | +251.5% | ~12% | ~15.8% | 0.63 | 2/11 |
| V2.4 | 2015-2026 | 10万 | +232.0% | ~11.5% | ~11.5% | — | — |
| V2.4 | 2010-2014 (out) | 2万 | +37% | ~6.4% | ~8.5% | — | — |
| **V2.5** | **2015-2026** | **2万** | **+306.2%** | **13.8%** | **12.85%** | **0.809** | **—** |
| V2.5 | 2010-2014 (out) | 2万 | +37.64% | 6.81% | 6.64% | 0.405 | — |
| **V2.6** | **2015-2026** | **2万** | **+372%** | **15.4%** | **14.4%** | **0.957** | **—** |
| **V2.10** | **2015-2026** | **2万** | **+371.7%** | **15.4%** | **15.2%** | **0.950** | **1/12** |
| **V2.10 + 防追高/后备补位** | **2015-2026** | **2万** | **+385.88%** | **15.66%** | **15.19%** | **0.985** | **—** |

> **V2.10聚宽实测**: +371.7%, 年化15.35%, 最大回撤15.19%, 夏普0.95, Beta 0.25, 盈亏比2.15, 仅1年亏损(-6.6%, 2018)。12个时间段全覆盖测试全部正收益+正超额。
> **2026-06-12阶段验证**: 防追高+后备补位版本 +385.88%, 年化15.66%, 最大回撤15.19%, 夏普0.985, 盈亏比2.181。该结果尚未完整WF验证，不替代V2.10核心参数结论。

### V2.5→V2.6 Iteration (2026-05, on JoinQuant)

**Final adopted (4 changes):**

1. **利润分段ATR收紧** (**+28pp**, highest single improvement): ATR multiplier tightens as profit grows — normal (2.5x) when profit<5%, moderate (2.0x) at 5-15%, tight (1.5x) at >15%. Prevents large accumulated profits from evaporating in grinding declines. **Lesson: risk management should adapt to profit cushion, not just volatility.**

2. **资本档位优化** (+8pp): medium base_ratio 0.60→0.65, large max_hold 4→3 + base_ratio 0.55→0.65. Higher capital utilization, unified max_hold=3.

3. **RSI >80极值修正** (+1pp): 55→70. Trend-following should not penalize extreme strength.

4. **MA10趋势止损** (夏普+0.009/回撤-0.8pp): Before exempting an ATR stop, check if price < MA10 and MA10 declining. If short-term trend is broken, deny exemption and execute stop. Replaced both DD 20% and 10% stop-exemption-cap — simpler and more precise.

**Tested and removed (proved unnecessary with MA10 present):**

5. **14:45午盘止损**: Removed — MA10 provides adequate same-day protection; afternoon check was redundant.
6. **20%极端DD止损**: Removed — MA10 catches trend breakdowns earlier than a raw DD threshold.
7. **10%止损豁免上限**: Removed — MA10 trend check is more discerning than a fixed percentage.
8. **档位5%迟滞**: Removed — unified max_hold=3 means tier changes barely affect behavior; simple thresholds are sufficient.
9. **移动补仓**: Removed — triggered only 3 times in 11yr backtest; negligible benefit.

**Tested and reverted (load-bearing parameters — DO NOT CHANGE):**

- **8分换仓门槛**: 5分翻车, 6分翻车, 0分翻车. Load-bearing, not tunable.
- **最低持仓5天**: 0天 = -48pp. Essential anti-whipsaw protection.
- **持仓保留门槛 55分**: 去掉惯性保护(55→60) OOS仅 -1.4pp/年（WF验证）. 全周期 -14pp（聚宽实测）. 保留是微弱正贡献，非铁律.
- **ROC20**: 14/15/25/blend all worse. Load-bearing.
- **Fast实验版**: 全局加速 = -231pp. 慢就是快.

### vs V15.9-Hybrid (ROC+LR daily rotation)
Direct A/B test on same period (2015-2026, 2万起始):
- **Multi-Factor V2.3: +251.5%, 15.8% max DD, 623 buys, 2 loss years (with stop-loss exemption)**
- **V15.9-Hybrid: +155.7%, 18.6% max DD, 1051 buys, 4 loss years**
- Multi-Factor wins on every metric. V15.9-Hybrid's daily rotation + no switch threshold = 70% more trades, higher costs, worse returns.
- **Lesson: "simple code" ≠ "simple trading behavior". 7-factor ensemble + switch threshold + min hold period produces fewer, better trades than pure momentum with daily rotation. Trading stability matters more than signal purity.**
- **ROC3 dynamic stop-loss**: Tested tightening ATR to 1.5x when 3-day ROC < -3%. Result: -62pp. Too many false triggers in normal pullbacks. **Don't add short-term momentum stop tightening.**
- **止损豁免 (stop-loss exemption)**: When ATR stop triggers but scoring says ETF is still a top candidate, skip the stop. Saves double commission (sell+buy back) and keeps position in place. Result: **+35pp** (215.9%→251.5%). Resetting highest price on exemption hurt (-44pp), so keep highest/ATR unchanged on exemption. **Key mechanism: let scoring override stop-loss when signals conflict.**

### V2.6后续优化实验（2026-05）

15+项实验中4项成功：利润分段ATR、资本档位优化、RSI极值修正、MA10趋势止损。其余全部失败或移除（DD 20%、14:45午盘、10%豁免上限、档位迟滞、移动补仓、换仓门槛调参、最低持有调参、信号分档仓位、动态减仓、得分动量、KDJ超卖惩罚、布林squeeze等）。

**铁律：8分换仓门槛、5天最低持有、ROC20、55分保留门槛——这四个参数动任何一个都翻车。** 14:45午盘和DD 20%在MA10趋势止损存在时是冗余的。

**核心教训：该策略的价值在于其"慢"设计——离散分档、3日平滑、8分门槛、5日最低持仓。任何试图"聪明"一点的改动几乎都会破坏这个平衡。372%/0.96夏普距离这个框架的理论天花板（~375%）已非常接近。**

### V2.7参数微调（2026-06）

22个参数全量单变量扫描 + 权重敏感性 + Walk-Forward验证。本地回测引擎复刻聚宽逻辑，2015-01~2026-03-11，初始2万。

**最终采纳（3个参数改动）：**

| 参数 | 旧值 | 新值 | 改善 | 机制 |
|------|------|------|------|------|
| `bb_period` | 20 | 25 | +6.8pp | 布林带窗口加长，趋势中不因短期波动触轨降分 |
| `bb_std` | 2.0 | 1.8 | +8.5pp | 带宽微收窄，60-80%最优评分区间更易维持 |
| `stop_floor` | 0.03 | 0.05 | +5.5pp | 止损地板3%→5%，过滤ETF日波动噪音止损 |

**组合效果：** bb_period+bb_std+stop_floor三者配合：收益+11.8pp，夏普1.17→1.20，交易529→506。布林带放宽让更多ETF留在高分区间→候选池扩大→换仓更频繁；stop_floor收紧恰好过滤掉这些新增换仓中的噪音止损。两者互补，缺一不可（单独改布林带反而-8.3pp）。聚宽估计业绩：~384%/年化~15.6%/DD~17.3%/夏普~1.01。

**22参数全量扫描结论：** 16/22参数默认值最优。6个偏离中有3个有意义（上述3个），2个边际（high_vol_threshold 0.30→0.20 +2.8pp, stop_cap 0.15→0.12 +2.8pp），1个无效（trailing_atr_mult_high_vol）。仅布林带两个信号参数可调，其余10个信号指标参数（RSI/MACD/KDJ/动量/成交量/平滑）全部默认值最优。

**因子权重敏感性：** 9组权重配置（等权/动量30%/动量15%/均线40%/RSI+KDJ上调/布林上调/动量70%/均线70%），**全部跑输基准**。动量25%是唯一不被淘汰的配置——更高（30%-71pp）或更低（15%-105pp）都崩。权重在正常范围内不敏感，极端集中时崩盘。

**Walk-Forward过拟合验证（8窗口，160次回测）：**

| 参数 | 稳定性 | 各窗口最优值 |
|------|--------|------------|
| switch_threshold=8 | **100% (8/8)** | 每窗都是8 |
| momentum_period=20 | **100% (8/8)** | 每窗都是20 |
| min_hold_days=5 | **75% (6/8)** | 前2窗选3(训练期太短)，后6窗全5 |
| hold_threshold=55 | **38% (3/8)** | 从60→52→50漂移，不稳定 |

**结论：换仓门槛8.0和动量周期20是结构性的——它们在任意时间段都是最优，不是过拟合。** 最低持有5天在训练期>4年后稳定。保留门槛不稳定但对OOS影响<1pp年化。平均OOS年化+15.5%，所有窗口为正。8个测试年全部正收益（最差2021年-6.5%）。

**品种差异化参数实验：**
- 信号端（QDII动量周期/黄金因子权重等）：全部翻车
- 风控端止损参数：**黄金唯一成功**——stop_floor=0.03, trailing_atr_mult=2.0 (+16.9pp, 夏普1.19→1.22)。黄金是均值回复型资产，趋势跟随的宽止损逻辑不适用。QDII放宽止损反而崩盘(-27.5pp)，QDII跳空是方向性信号而非噪音。
- 换仓门槛7/9：-30/-61pp。

### V2.8黄金品种级止损（2026-06）

**唯一采纳：** 黄金ETF(518880)使用品种级止损参数 `stop_floor=0.03, trailing_atr_mult=2.0`（其余11个品种保持V2.7参数不变）。黄金是趋势跟随框架中唯一的均值回复型资产，统一放宽止损(0.03→0.05)对黄金是负优化。

实现方式：`g.code_stop_params = {'518880.XSHG': {'stop_floor': 0.03, 'trailing_atr_mult': 2.0}}`，`calc_stop_price` 新增 `code` 参数查找品种级覆盖。

**品种差异化止损全覆盖实验（7组）：** 仅黄金收紧(+16.9pp)有效。豆粕收紧(-4.6pp)、QDII收紧(-2.0pp)、QDII放宽(-27.5pp)、A股收紧(-21.6pp)全部翻车。A股收紧破坏趋势跟随、QDII双向都不行、豆粕有趋势性不同于黄金。

**Walk-Forward验证（8窗口V2.7 vs V2.8）：** V2.8胜7/8 OOS窗口，仅在2019年微输1.7pp。改善逐年累积，夏普无退化。**黄金收紧不是过拟合。**

本地回测验证：+16.9pp vs V2.7，夏普 1.19→1.22。聚宽预计总业绩：~400%/年化~16%。

### V2.9因子权重再平衡（2026-06）

**42组细粒度权重扫描 + 8窗口Walk-Forward验证。**

均线趋势权重被严重低配。从 15%→21%（其余6因子等比例微降），全周期收益从 354%→400%（+46pp），夏普 1.19→1.28。

**Walk-Forward：9/9窗口全胜（含全期），夏普全部改善。** 改善逐年累积：Test2019 +7.5pp → Test2026 +46.0pp，无一年例外。这是所有实验中验证最强的一次。

新权重：
| 因子 | 旧 | 新 | 变化 |
|------|-----|-----|------|
| ma_trend | 15% | 21% | +6pp |
| momentum | 25% | 23.2% | -1.8pp |
| macd | 18% | 16.7% | -1.3pp |
| rsi | 12% | 11.2% | -0.8pp |
| kdj | 12% | 11.2% | -0.8pp |
| bollinger | 10% | 9.3% | -0.7pp |
| volume | 8% | 7.4% | -0.6pp |

逻辑：MA10>MA20>MA60 的多头排列是 ETF 趋势跟随中唯一同时覆盖短中长三个时间维度的信号，原 15% 权重远低于其真实预测力。动量（25%）虽然最强但不稳定，均线趋势更稳健且方向一致性更高。

**总结：信号端优化已触达框架天花板。** V2.6→V2.10 信号端四项改动（布林带/黄金止损/均线权重/hold_threshold验证），聚宽实测收益不变（~372%），但风控更稳健、参数有WF支撑。真正能改善策略的是执行端（QDII溢价检查、国债逆回购、最小交易额过滤）。

### V2.10均线权重Walk-Forward微调（2026-06）

**Walk-Forward验证（8窗口，28个MA权重值 8%-35%）：**
- 各窗口训练集最优：前5窗选22%，后3窗选24%（随时间漂移，更近窗口指向更高权重）
- OOS实际表现：24%在7/8年优于或平于22%，仅2026年输0.1pp
- 平均OOS：24%(+16.7%) > 22%(+16.2%) > 21%(+16.1%)
- **采纳 24%**（OOS一致性最高、近期窗口训练最优、与22%同处平坦区但方向更对）

新权重：

| 因子 | V2.9 | V2.10 | 变化 |
|------|------|-------|------|
| ma_trend | 21% | 24% | +3pp |
| momentum | 23.2% | 22.3% | -0.9pp |
| macd | 16.7% | 16.1% | -0.6pp |
| rsi | 11.2% | 10.8% | -0.4pp |
| kdj | 11.2% | 10.8% | -0.4pp |
| bollinger | 9.3% | 8.9% | -0.4pp |
| volume | 7.4% | 7.1% | -0.3pp |

**验证结论：均线权重最优区间 20-27%。24%在OOS上一致性最优。但区间内任何选择差异<1pp，不敏感。**

### V2.10参数WF补充验证（2026-06）

对之前未验证的关键参数做了8窗口Walk-Forward扫描：

**score_buy_threshold（买入门槛 55/58/60/62/65）：**
- **60是最优值** — 平均OOS最高(+17.4%)、训练集也最高(+183.6%)、两边都下降（55→60上升、60→65下降），真正凹形最优
- 65太高导致候选不足(OOS -2.7pp)，55太低放进低质量ETF
- 结论：**60是训练和OOS同时最优，非常稳健**

**hold_threshold（持仓保留门槛 50/55/58/60）：**
- 55平均OOS最高(+17.4%)但仅比58高0.2pp
- 训练集上55 vs 60差距41pp（大部分是噪音），OOS仅差1.4pp
- 去掉惯性保护(55→60)聚宽实测 -13.8pp（全周期）
- 结论：55是微弱正贡献(+1.4pp OOS)，保留但不再视为"铁律"

**起点敏感性测试（3个起点）：**
- 2015-01（牛市中部）年化15.35% → 2016-01（熔断后）15.24% → 2017-01（熊市前）15.65%
- 年化和夏普几乎不变，起始时机的运气成分被策略消化

**压力点测试（5个极端场景）：**
- 牛市顶入市 +3.5%（大盘 -43%）、熔断低点入市 +95.6%、全年熊市 -8.2%（大盘 -25%）、疫情年 +31.7%、震荡熊市 +0.1%（大盘 -26%）
- 全部正收益或打平，回撤全部在10-15%，无一崩盘

### 2026-06-12交易端防守实验

**Adopted (stage-validated, not full WF yet):**
- **防追高过滤**: New buys only. Skip candidate when real-time price is >8% above MA20 and RSI >75. Does not force-sell or affect existing holdings.
- **后备补位**: If a target-pool candidate is skipped by 防追高, continue checking backup candidates from the qualified pool to fill remaining slots.
- **震荡市日志**: Detects choppy market with MA20 crossing count, MA60 slope, and distance to MA60. Log only; no trading impact.
- **PTrade partial-cancel handling**: `status == "5"` with `business_amount > 0` is a partial fill, not a pure failure. Wait for trade callback.

**JoinQuant result**: +385.88%, annualized 15.66%, max DD 15.19%, Sharpe 0.985, P/L ratio 2.181. This is better than V2.10 core but not yet a full WF-proven core-parameter change.

**Tested and removed:**
- **盈利回落保护**: +381.78%, annualized 15.57%, Sharpe 0.982, P/L ratio 2.158. Worse than 防追高 version; removed.
- **评分衰减保护**: +370.89%, annualized 15.33%, max DD 16.69%, Sharpe 0.965. Worse return and worse drawdown; removed.

**PTrade validation**: `logs/PTrade2023.txt` only covers 2023-01-03 to 2023-01-31. It had no ERROR/Exception/废单/重复卖出/订单超时/资金不足 and max holdings stayed at 3. Treat it only as a short compatibility check, not performance validation.

## Pending Tasks

（无待办事项）

## Chinese Variable Reference

`买分`=buy score, `卖分`=sell score, `趋势分`=trend score, `趋势系数`=trend coefficient, `阳线`=bullish candle, `阴线`=bearish candle, `实体`=candle body, `量比`=volume ratio, `档位`=tier, `仓位`=position, `止损`=stop loss
