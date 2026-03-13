# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese ETF quantitative trading strategy system. Automated buy/sell signal generation, ATR-based risk management, and momentum ranking for 3-5 widely-traded Chinese ETFs. Targets small capital (initial 20K CNY). Multi-platform: JoinQuant (聚宽) for backtesting, PTrade for live trading.

## Key Files

- `smart_trade_joinquant_v10_etf.py` — **V10.0 JoinQuant, highest absolute return** (58% return over 10yr, 27.9% max drawdown, 3 ETFs)
- `smart_trade_ptrade_v10_etf.py` — V10.0 PTrade version (production-ready, dual-mode backtest+live)
- `smart_trade_joinquant_v11_etf.py` — V11.0 JoinQuant (5 ETFs, expanded pool)
- `smart_trade_joinquant_v13_etf.py` — **V13.0 JoinQuant, best risk-adjusted of signal-driven** (57.5% return, ~12.5% max drawdown, bear market half-position)
- `smart_trade_joinquant_v15_etf.py` — **V15.1 JoinQuant, highest return & best risk-adjusted** (210% return over 11yr, ~10.8% annualized, 5万起始, momentum rotation, optimized 10-ETF pool: 4 A-share + 3 cross-market + 3 cross-asset)
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
- **V15.0**: Completely new framework — momentum rotation instead of signal-driven. Every 3 trading days rebalance, always hold top N ETFs by risk-adjusted momentum (ROC20/volatility). 10-ETF pool (3 broad-base + 4 sector + 2 cross-market + 1 gold). Filters: positive momentum + price > MA20, otherwise don't buy; if all ETFs fail filter → auto cash. ATR trailing stop as safety net. First iteration with 沪深300-based bear market filter triggered 234 times (too sensitive for weekly rotation, suppressed rebounds). After removing bear filter: 223.4% total return over 11yr (2万起始, ~11.3% annualized). **Lesson: momentum rotation dramatically improves capital utilization (~90% vs ~40%); explicit bear market filter is counterproductive for weekly rotation — "natural cash" (no positive momentum ETF → auto empty) is sufficient.**
- **V15.1**: Dual momentum filter (ROC20>0 AND ROC60>0), dynamic ATR stop (2.0x in high volatility, 2.5x normally). With original 10-ETF pool (8 A-share + 2 cross-asset): 170% return (2万起始), 119.5% (5万起始). Then **ETF pool restructured** from 8 A-share + 2 cross-asset → 4 A-share + 3 cross-market + 3 cross-asset. New pool: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 513100纳指, 513500标普500, 159920恒生, 518880黄金, 511010国债, 159985豆粕. Result (5万起始): **210% total return (~10.8% annualized), worst year -1.3% (2018), only 2 loss years both <1.5%**. Key findings: (1) 511010国债ETF bought 86 times with 0% stop rate — acts as "productive cash", earning bond returns instead of idle cash; (2) 513500标普500 + 513100纳指 account for 27% of trades, providing strong returns when A-shares weak; (3) 159985豆粕 has 42% stop rate, weakest link but marginal impact. **Lesson: cross-asset diversification is the single biggest improvement — same momentum framework, just better ETF pool structure, nearly doubled returns (119.5%→210%) while reducing worst year from -8.7% to -1.3%. Bond ETF as "productive cash" is a key innovation. V15.1 with optimized pool is the current best version.**
- **V15.2** (reverted): Soft ROC60 filter (allow -10%<ROC60<0 at 70% position). Result: 140% return — worse than V15.1's 170%. Mid-term negative momentum trades are fundamentally bad regardless of position sizing. Reverted to V15.1.

## Chinese Variable Reference

`买分`=buy score, `卖分`=sell score, `趋势分`=trend score, `趋势系数`=trend coefficient, `阳线`=bullish candle, `阴线`=bearish candle, `实体`=candle body, `量比`=volume ratio, `档位`=tier, `仓位`=position, `止损`=stop loss
