# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese ETF quantitative trading strategy system. Automated buy/sell signal generation, ATR-based risk management, and momentum ranking for 3-5 widely-traded Chinese ETFs. Targets small capital (initial 20K CNY). Multi-platform: JoinQuant (聚宽) for backtesting, PTrade for live trading.

## Key Files

- `smart_trade_joinquant_v10_etf.py` — **V10.0 JoinQuant, highest absolute return** (58% return over 10yr, 27.9% max drawdown, 3 ETFs)
- `smart_trade_ptrade_v10_etf.py` — V10.0 PTrade version (production-ready, dual-mode backtest+live)
- `smart_trade_joinquant_v11_etf.py` — V11.0 JoinQuant (5 ETFs, expanded pool)
- `smart_trade_joinquant_v13_etf.py` — **V13.0 JoinQuant, best risk-adjusted of signal-driven** (57.5% return, ~12.5% max drawdown, bear market half-position)
- `smart_trade_joinquant_v15_etf.py` — **V15.5 JoinQuant, original momentum rotation** (210% return over 11yr, ~10.8% annualized, 5万起始, momentum rotation, optimized 10-ETF pool: 4 A-share + 3 cross-market + 3 cross-asset)
- `smart_trade_joinquant_v15_7_etf.py` — **V15.7 JoinQuant** (212.8% return, 10万起始, buy price fix + bond slot-filling, 10-ETF pool)
- `smart_trade_joinquant_v15_7_expanded_etf.py` — **V15.7-Expanded JoinQuant** (267.9% return, 10万起始, 12-ETF pool: +日经+中概互联)
- `smart_trade_joinquant_v15_9_etf.py` — **V15.9 JoinQuant, current best** (256.9% return, 2万起始, 12-ETF + unified max_hold=3)
- `smart_trade_ptrade_v15_7_etf.py` — **V15.7 PTrade版** (实盘/模拟部署用, 10-ETF pool)
- `策略说明文档.md` — Complete strategy documentation (Chinese)
- `PTrade-API.html` — Official PTrade API reference
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
- **Broker**: 国金证券 PTrade. API docs at `PTrade-API.html` (local copy).

### Capital Tiers

| Tier | Total Assets | Max Holdings | Base Position |
|------|-------------|--------------|---------------|
| micro | <1.5万 | 1 | 85% |
| small | 1.5-5万 | 2 | 70% |
| medium | 5-10万 | 3 | 55% |
| large | >10万 | 3 | 45% |

## Critical Design Rules

- **No future functions**: Signals always computed on `prev_date` (T-1) data. Current price used only for stop-loss execution and order placement.
- **All parameters are academic defaults**: ATR(14), MACD(12,26,9), KDJ(9,3,3), RSI(6). Zero parameter optimization — this is intentional to avoid overfitting.
- **No profit-taking**: V11.1 proved that partial profit-taking (+20% sell half) destroys trend-following. 盈亏比 dropped from 3.7:1 to 1.14:1. Let profits run via ATR trailing stop only.
- **Stop loss clamped to [3%, 15%]**: `stop_floor=0.03` prevents noise shakeout, `stop_cap=0.15` prevents excessive single-trade loss.
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

## Chinese Variable Reference

`买分`=buy score, `卖分`=sell score, `趋势分`=trend score, `趋势系数`=trend coefficient, `阳线`=bullish candle, `阴线`=bearish candle, `实体`=candle body, `量比`=volume ratio, `档位`=tier, `仓位`=position, `止损`=stop loss
