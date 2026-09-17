# -*- coding: utf-8 -*-
"""
趋势腿策略 JoinQuant 版 v0.1
============================
信号驱动趋势跟随（非打分、非轮动），单品种判断。

池（9 只趋势型 ETF）：
  A股: 510300沪深300, 159915创业板, 512100中证1000, 159928消费
  跨市场: 513100纳指, 513500标普500, 159920恒生, 513880日经, 513050中概互联

规则：
  买入 = 多头趋势(close>MA20 且 MA20>MA60) + 突破20日新高；RSI>75 防追高暂缓
  卖出 = ATR(14)跟踪止损(2.5x, 利润分段收紧) 或 MA20死叉MA60（任一触发）
  仓位 = max_hold=3 等分，base_ratio=0.95 留5%现金；候选不足留现金不凑数
  防whipsaw = 冷却5天 + 最低持有5天；合格数>3 时按 ROC20 降序取前3
  执行 = 09:35 主流程；信号只用 T-1 数据（avoid_future_data 兜底）

平台约定（对齐 Multi-Factor 聚宽版）：
  - set_option('avoid_future_data', True)：访问未来数据直接报错
  - get_price(fq='pre', skip_paused=True)：前复权、跳过停牌
"""

import numpy as np
import pandas as pd
from jqdata import *


def get_default_params():
    return {
        "lookback": 120,
        "max_hold": 3,
        "base_ratio": 0.95,
        "atr_period": 14,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "min_hold_days": 5,
        "cooldown_days": 5,
        "overheat_rsi": 75,
        "rsi_period": 14,
        "ma_fast": 20,
        "ma_slow": 60,
        "high_period": 20,
        "roc_period": 20,
        # 熊市过滤开关：510300 收盘<MA60 且 MA60 下行 → A股ETF 暂停新买入
        "bear_filter": True,
    }


def get_default_etf_pool():
    return [
        "510300.XSHG",  # 沪深300
        "159915.XSHE",  # 创业板
        "512100.XSHG",  # 中证1000
        "159928.XSHE",  # 消费
        "513100.XSHG",  # 纳指
        "513500.XSHG",  # 标普500
        "159920.XSHE",  # 恒生
        "513880.XSHG",  # 日经
        "513050.XSHG",  # 中概互联
    ]


A_SHARE_CODES = {"510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE"}


# ============================================================
#  技术指标
# ============================================================

def calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    # 防未来函数兜底：任何对未来数据的访问（如 get_price 越过当前时点）直接报错
    set_option("avoid_future_data", True)

    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type="stock")

    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    g.highest_since_buy = {}
    g.buy_date = {}
    g.last_sell_date = {}

    run_daily(do_trading, time="09:35")
    run_daily(after_close, time="15:30")


def get_prev_trade_date(context):
    days = get_trade_days(end_date=context.current_dt.date(), count=2)
    if len(days) >= 2:
        return days[0]
    return context.previous_date


def hold_codes(context):
    return {c for c, p in context.portfolio.positions.items() if p.total_amount > 0}


def held_trading_days(code, today):
    """买入日计 0 天的持仓交易日数；无买入日视为很久。"""
    buy = g.buy_date.get(code)
    if buy is None:
        return 999
    return len(get_trade_days(start_date=buy, end_date=today)) - 1


def in_cooldown(code, today):
    sell = g.last_sell_date.get(code)
    if sell is None:
        return False
    elapsed = len(get_trade_days(start_date=sell, end_date=today)) - 1
    return elapsed <= g.params["cooldown_days"]


# ============================================================
#  信号计算（只用 end_date 及以前数据）
# ============================================================

def calc_trend_signal(code, end_date):
    p = g.params
    df = get_price(code, end_date=end_date, count=p["lookback"],
                   frequency="daily", fields=["close", "high", "low"],
                   skip_paused=True, fq="pre")
    if df is None or len(df) < p["ma_slow"] + 1:
        return None
    close, high, low = df["close"], df["high"], df["low"]
    ma_fast = close.rolling(p["ma_fast"]).mean()
    ma_slow = close.rolling(p["ma_slow"]).mean()
    ma_fast_prev = ma_fast.shift(1)
    ma_slow_prev = ma_slow.shift(1)

    cur = close.iloc[-1]
    atr = calc_atr(high, low, close, p["atr_period"]).iloc[-1]
    rsi = calc_rsi(close, p["rsi_period"]).iloc[-1]
    roc = cur / close.iloc[-p["roc_period"] - 1] - 1
    new_high = cur > close.iloc[-p["high_period"] - 1:-1].max()
    dead_cross = (ma_fast.iloc[-1] < ma_slow.iloc[-1]
                  and ma_fast_prev.iloc[-1] >= ma_slow_prev.iloc[-1])
    return {
        "code": code, "close": cur,
        "ma_fast": ma_fast.iloc[-1], "ma_slow": ma_slow.iloc[-1],
        "ma_slow_prev": ma_slow_prev.iloc[-1],
        "atr": atr, "rsi": rsi, "roc20": roc,
        "new_high": bool(new_high), "dead_cross": bool(dead_cross),
    }


def buy_signal_ok(sig):
    for key in ("close", "ma_fast", "ma_slow"):
        v = sig.get(key)
        if v is None or pd.isna(v):
            return False
    return (sig["new_high"]
            and sig["close"] > sig["ma_fast"]
            and sig["ma_fast"] > sig["ma_slow"])


def calc_stop_price(sig, highest, price, entry_cost):
    if sig is None:
        return None
    atr = sig.get("atr")
    if atr is None or pd.isna(atr) or atr <= 0 or highest <= 0:
        return None
    profit = price / entry_cost - 1 if entry_cost > 0 else 0.0
    mult = g.params["trailing_atr_mult"]
    if profit > 0.15:
        mult *= 0.6
    elif profit > 0.05:
        mult *= 0.8
    pct = mult * atr / highest
    pct = max(g.params["stop_floor"], min(g.params["stop_cap"], pct))
    return highest * (1 - pct)


# ============================================================
#  交易
# ============================================================

def execute_sell(context, code, reason, price):
    pos = context.portfolio.positions[code]
    if pos.avg_cost > 0:
        pnl_pct = (price - pos.avg_cost) / pos.avg_cost * 100
        log.info("[卖出] %s 原因=%s 现价%.3f 成本%.3f 盈亏%.1f%%" % (
            code, reason, price, pos.avg_cost, pnl_pct))
    order_target(code, 0)
    g.highest_since_buy.pop(code, None)
    g.buy_date.pop(code, None)
    g.last_sell_date[code] = context.current_dt.date()


def do_trading(context):
    p = g.params
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    today = context.current_dt.date()

    # ---- 卖出阶段：ATR 止损（始终）→ 死叉（满足最低持有）----
    for code in sorted(hold_codes(context)):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0 or current_data[code].paused:
            continue
        price = current_data[code].last_price
        if price <= 0:
            continue
        days_held = held_trading_days(code, today)
        sig = calc_trend_signal(code, prev_date)
        if days_held > 0:
            stop_price = calc_stop_price(
                sig, g.highest_since_buy.get(code, price), price, pos.avg_cost)
            if stop_price is not None and price <= stop_price:
                execute_sell(context, code, "atr_stop", price)
                continue
        if days_held >= p["min_hold_days"] and sig is not None and sig["dead_cross"]:
            execute_sell(context, code, "dead_cross", price)
            continue

    # ---- 买入阶段：合格候选按 ROC20 降序取前 N ----
    holds = hold_codes(context)
    slots = p["max_hold"] - len(holds)
    if slots <= 0:
        return

    # 熊市过滤：510300 收盘<MA60 且 MA60 下行 → A股ETF 暂停新买入
    bear = False
    if p.get("bear_filter", True):
        sig_510300 = calc_trend_signal("510300.XSHG", prev_date)
        if sig_510300 is not None and sig_510300["ma_slow_prev"] is not None:
            bear = (sig_510300["close"] < sig_510300["ma_slow"]
                    and sig_510300["ma_slow"] < sig_510300["ma_slow_prev"])

    qualified = []
    for code in g.etf_pool:
        if code in holds or in_cooldown(code, today):
            continue
        if bear and code in A_SHARE_CODES:
            continue
        if current_data[code].paused:
            continue
        sig = calc_trend_signal(code, prev_date)
        if sig is None or not buy_signal_ok(sig):
            continue
        if sig["rsi"] is not None and not pd.isna(sig["rsi"]) and sig["rsi"] > p["overheat_rsi"]:
            continue  # 防追高
        price = current_data[code].last_price
        if price <= 0:
            continue
        qualified.append((code, sig["roc20"], price))

    qualified.sort(key=lambda x: (-x[1], x[0]))

    total = context.portfolio.total_value
    for code, roc, price in qualified[:slots]:
        target = total * p["base_ratio"] / p["max_hold"]
        order_target_value(code, target)
        g.highest_since_buy[code] = price
        g.buy_date[code] = today
        log.info("[买入] %s ROC20=%.1f%% @%.3f 目标市值%.0f" % (code, roc * 100, price, target))


def after_close(context):
    current_data = get_current_data()
    holds = hold_codes(context)
    for code in holds:
        price = current_data[code].last_price
        if price > 0:
            g.highest_since_buy[code] = max(g.highest_since_buy.get(code, price), price)

    total = context.portfolio.total_value
    cash = context.portfolio.available_cash
    log.info("=" * 60)
    log.info("[收盘] 总值%.0f 现金%.0f 持仓%d/%d" % (
        total, cash, len(holds), g.params["max_hold"]))
    for code in sorted(holds):
        pos = context.portfolio.positions[code]
        pnl = (pos.price - pos.avg_cost) / pos.avg_cost * 100 if pos.avg_cost > 0 else 0
        highest = g.highest_since_buy.get(code, pos.price)
        log.info("  %s 成本%.3f 现%.3f 高%.3f 盈亏%.1f%%" % (
            code, pos.avg_cost, pos.price, highest, pnl))
    log.info("=" * 60)
