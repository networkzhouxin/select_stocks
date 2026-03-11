# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese ETF quantitative trading strategy system. Automated buy/sell signal generation, ATR-based risk management, and momentum ranking for 3-5 widely-traded Chinese ETFs. Targets small capital (initial 20K CNY). Multi-platform: JoinQuant (聚宽) for backtesting, PTrade for live trading.

## Key Files

- `smart_trade_joinquant_v10_etf.py` — **V10.0 JoinQuant, the proven best** (45.18% return, 0.833 Sharpe, 3 ETFs)
- `smart_trade_ptrade_v10_etf.py` — V10.0 PTrade version (production-ready, dual-mode backtest+live)
- `smart_trade_joinquant_v11_etf.py` — V11.0 JoinQuant (5 ETFs, expanded pool)
- `smart_trade_ptrade_v11_etf.py` — V11.0 PTrade version
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

## Version History Lessons

- **V6-V7**: Individual stocks, high risk (49% max drawdown), poor for small capital
- **V8**: Switch to ETF improved everything; ATR stops introduced
- **V9**: Over-complicated (adaptive MACD, regime detection) → worse results
- **V10**: Simplified back, added trend hold + momentum ranking → **optimal** (45.18%)
- **V10.1/V10.2**: Attempted KAMA/adaptive indicators → degraded performance, confirming V10.0 is the complexity ceiling
- **V11**: Only change is ETF pool (3→5), all logic identical to V10.0

## Chinese Variable Reference

`买分`=buy score, `卖分`=sell score, `趋势分`=trend score, `趋势系数`=trend coefficient, `阳线`=bullish candle, `阴线`=bearish candle, `实体`=candle body, `量比`=volume ratio, `档位`=tier, `仓位`=position, `止损`=stop loss
