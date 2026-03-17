# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V15.7-全球版 - 纯海外+跨资产动量轮动
=============================================
基于V15.7动量轮动框架，ETF池替换为纯海外+跨资产，不含A股权益。
策略逻辑、参数、止损体系与V15.7 100%一致，仅ETF池不同。

实验目的：验证去掉A股后收益是否更好。
风险提示：QDII类ETF实盘中存在溢价和限购风险，回测结果可能无法完全复现。

ETF池（4海外权益+3跨资产+3替补 = 10只）：
  海外权益4只：
  - 513100 纳指ETF（美国科技）
  - 513500 标普500ETF（美国全市场）
  - 159920 恒生ETF（港股）
  - 513050 中概互联网ETF（海外中概）[2017上市]
  跨资产3只：
  - 518880 黄金ETF（避险）
  - 511010 国债ETF（债券兜底）
  - 159985 豆粕ETF（商品周期）
  补充3只（增加分散度）：
  - 162411 华宝油气（能源商品）
  - 513880 日经ETF（日本市场）[2019上市]
  - 501018 南方原油（原油）[2016上市]
"""

import numpy as np
import pandas as pd
from jqdata import *


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- ETF标的池（全球版：纯海外+跨资产）----
    g.etf_pool = [
        # ---- 海外权益 4只 ----
        '513100.XSHG',   # 纳指ETF（美国科技）
        '513500.XSHG',   # 标普500ETF（美国全市场）
        '159920.XSHE',   # 恒生ETF（港股）
        '513050.XSHG',   # 中概互联网ETF（海外中概）[2017上市]
        # ---- 跨资产 3只 ----
        '518880.XSHG',   # 黄金ETF（避险）
        '511010.XSHG',   # 国债ETF（债券兜底）
        '159985.XSHE',   # 豆粕ETF（商品周期）
        # ---- 补充 3只 ----
        '162411.XSHE',   # 华宝油气（能源商品）
        '513880.XSHG',   # 日经ETF（日本市场）[2019上市]
        '501018.XSHG',   # 南方原油（原油）[2016上市]
    ]
    g.bond_etf = '511010.XSHG'

    # ---- 资金档位配置（与V15.7一致）----
    g.capital_tiers = {
        'micro': {'max_hold': 1, 'base_position_ratio': 0.90},
        'small': {'max_hold': 2, 'base_position_ratio': 0.80},
        'medium': {'max_hold': 3, 'base_position_ratio': 0.70},
        'large': {'max_hold': 3, 'base_position_ratio': 0.60},
    }

    # ---- 策略参数（与V15.7一致）----
    g.params = {
        'rebalance_interval': 3,
        'momentum_period': 20,
        'momentum_period_long': 60,
        'atr_period': 14,
        'trailing_atr_mult': 2.5,
        'trailing_atr_mult_high_vol': 2.0,
        'high_vol_threshold': 0.30,
        'stop_floor': 0.03,
        'stop_cap': 0.15,
        'trend_weight': 0.3,
    }

    g.current_tier = None
    g.day_count = 0
    g.highest_since_buy = {}
    g.entry_atr = {}

    run_daily(update_tier, time='09:30')
    run_daily(do_trading, time='09:35')
    run_daily(update_highest, time='15:00')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


# ============================================================
#  动态资金档位
# ============================================================
def update_tier(context):
    total = context.portfolio.total_value
    if total < 15000:
        new_tier = 'micro'
    elif total < 50000:
        new_tier = 'small'
    elif total < 100000:
        new_tier = 'medium'
    else:
        new_tier = 'large'

    if new_tier != g.current_tier:
        old_tier = g.current_tier or '初始化'
        g.current_tier = new_tier
        cfg = g.capital_tiers[new_tier]
        log.info('[档位变更] %s -> %s | 总资产:%.0f | 最大持仓:%d' % (
            old_tier, new_tier, total, cfg['max_hold']))


def get_tier_param(param_name):
    return g.capital_tiers[g.current_tier][param_name]


# ============================================================
#  动量计算（与V15.7一致）
# ============================================================
def calc_momentum(code, end_date):
    mom_long = g.params['momentum_period_long']
    df = get_price(code, end_date=end_date, count=mom_long + 10,
                   frequency='daily',
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   skip_paused=True, fq='pre')

    if df is None or len(df) < mom_long:
        return None

    C = df['close']
    H = df['high']
    L = df['low']
    mom_period = g.params['momentum_period']

    roc_short = C.iloc[-1] / C.iloc[-mom_period] - 1
    roc_long = C.iloc[-1] / C.iloc[-mom_long] - 1

    returns = C.pct_change().iloc[-mom_period:]
    vol = returns.std() * np.sqrt(252)
    if vol <= 0 or pd.isna(vol):
        return None

    risk_adj_mom = roc_short / vol

    ma20 = C.iloc[-20:].mean()
    if C.iloc[-1] < ma20:
        return None
    if roc_short < 0:
        return None
    if roc_long < 0:
        return None

    trend_strength = (C.iloc[-1] - ma20) / ma20

    tw = g.params['trend_weight']
    sort_score = risk_adj_mom * (1 - tw) + trend_strength * tw

    atr_period = g.params['atr_period']
    TR = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = TR.iloc[-atr_period:].mean()

    return {
        'code': code,
        'momentum': sort_score,
        'risk_adj_mom': risk_adj_mom,
        'trend_strength': trend_strength,
        'roc': roc_short,
        'roc_long': roc_long,
        'volatility': vol,
        'close': C.iloc[-1],
        'atr': atr if not pd.isna(atr) else C.iloc[-1] * 0.02,
    }


# ============================================================
#  ATR跟踪止损（与V15.7一致）
# ============================================================
def calc_trailing_stop_price(highest_price, atr_value):
    vol_pct = atr_value / highest_price * np.sqrt(252 / g.params['atr_period'])
    if vol_pct > g.params['high_vol_threshold']:
        atr_mult = g.params['trailing_atr_mult_high_vol']
    else:
        atr_mult = g.params['trailing_atr_mult']

    pct_stop = atr_mult * atr_value / highest_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def check_stop_loss(context, current_data):
    stopped = []
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost

        if code in g.highest_since_buy and code in g.entry_atr:
            highest = g.highest_since_buy[code]
            stop_price = calc_trailing_stop_price(highest, g.entry_atr[code])
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[ATR止损] %s 最高%.3f 现价%.3f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, drawdown * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                stopped.append(code)

    return stopped


# ============================================================
#  核心交易逻辑（与V15.7一致）
# ============================================================
def do_trading(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()

    stopped_codes = check_stop_loss(context, current_data)

    g.day_count += 1
    if g.day_count < g.params['rebalance_interval'] and not stopped_codes:
        return
    if g.day_count >= g.params['rebalance_interval']:
        g.day_count = 0

    candidates = []
    for code in g.etf_pool:
        if current_data[code].paused:
            continue
        result = calc_momentum(code, prev_date)
        if result is not None:
            candidates.append(result)

    candidates.sort(key=lambda x: x['momentum'], reverse=True)

    max_hold = get_tier_param('max_hold')
    target_list = candidates[:max_hold]

    bond = g.bond_etf
    bond_in_targets = any(t['code'] == bond for t in target_list)
    if len(target_list) < max_hold and not bond_in_targets:
        if not current_data[bond].paused:
            target_list.append({
                'code': bond,
                'momentum': 0, 'risk_adj_mom': 0, 'trend_strength': 0,
                'roc': 0, 'roc_long': 0,
                'volatility': 0.03,
                'close': current_data[bond].last_price,
                'atr': current_data[bond].last_price * 0.005,
                '_is_bond_fill': True,
            })
            log.info('[国债填空] 候选%d只不足%d，国债补位' % (
                len(target_list) - 1, max_hold))

    target_codes = set(t['code'] for t in target_list)

    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if code not in target_codes:
            profit_pct = (current_data[code].last_price - pos.avg_cost) / pos.avg_cost
            log.info('[轮动卖出] %s 盈亏%.1f%%（被更强标的替换）' % (code, profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)

    current_holds = set()
    for code in context.portfolio.positions:
        if context.portfolio.positions[code].total_amount > 0:
            current_holds.add(code)

    to_buy = [sig for sig in target_list if sig['code'] not in current_holds]
    if not to_buy:
        return

    available = context.portfolio.available_cash
    slots = max_hold - len(current_holds & target_codes)

    if slots <= 0 or available < 500:
        return

    base_ratio = get_tier_param('base_position_ratio')

    for sig in to_buy:
        if slots <= 0 or available < 500:
            break

        code = sig['code']
        price = current_data[code].last_price
        is_bond_fill = sig.get('_is_bond_fill', False)

        alloc = available / slots * base_ratio

        if not is_bond_fill:
            target_vol = 0.15
            actual_vol = max(sig['volatility'], 0.05)
            vol_mult = min(target_vol / actual_vol, 1.5)
            vol_mult = max(vol_mult, 0.4)
            alloc *= vol_mult

        alloc = min(alloc, available * 0.95)

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            if available >= price * 100 * 1.003:
                shares = 100
            else:
                continue

        if is_bond_fill:
            log.info('[国债填空买入] %s %d股 @%.3f' % (code, shares, price))
        else:
            log.info('[轮动买入] %s 综合=%.3f 动量=%.3f 趋势=%.3f ROC20=%.1f%% ROC60=%.1f%% 波动率=%.1f%% %d股 @%.3f' % (
                code, sig['momentum'], sig['risk_adj_mom'], sig['trend_strength'],
                sig['roc'] * 100, sig['roc_long'] * 100,
                sig['volatility'] * 100, shares, price))

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['atr']
        available -= shares * price * 1.003
        slots -= 1


# ============================================================
#  每日更新最高价
# ============================================================
def update_highest(context):
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        cur = get_current_data()[code].last_price
        if code in g.highest_since_buy:
            if cur > g.highest_since_buy[code]:
                g.highest_since_buy[code] = cur
        else:
            g.highest_since_buy[code] = max(cur, pos.avg_cost)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        context.portfolio.total_value,
        context.portfolio.available_cash,
        len(hold),
        get_tier_param('max_hold')))

    for code, pos in hold.items():
        profit_pct = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        highest = g.highest_since_buy.get(code, pos.price)
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f 盈亏:%.1f%%' % (
            code, pos.avg_cost, pos.price, highest, profit_pct))
    log.info('=' * 60)
