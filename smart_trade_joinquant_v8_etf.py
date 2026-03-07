# -*- coding: utf-8 -*-
"""
智能买卖点量化交易策略 V8.0 - ETF版（最终版+动态资金适配）
======================================
核心信号逻辑不变（经8轮回测验证），仅根据账户资金规模动态调整运营参数：

资金档位：
  <1.5万（微型）：集中1只ETF，高仓位搏收益
  1.5-5万（小型）：标准2只ETF（回测验证档位）
  5-10万（中型）：3只ETF全覆盖，分散更好
  >10万（大型）：3只ETF+可持有更久

标的池（仅宽基）：
  - 510300 沪深300ETF（大盘蓝筹）
  - 159915 创业板ETF（成长弹性）
  - 510500 中证500ETF（中盘均衡）
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

    set_slippage(PriceRelatedSlippage(0.001))  # ETF滑点更小
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,            # ETF免印花税
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- ETF标的池（仅宽基，经回测验证行业ETF会拖累收益）----
    g.etf_pool = [
        '510300.XSHG',   # 沪深300ETF（大盘蓝筹）
        '159915.XSHE',   # 创业板ETF（成长弹性）
        '510500.XSHG',   # 中证500ETF（中盘均衡）
    ]

    # ---- 资金档位配置 ----
    # 每个档位：(最大持仓数, 信号门槛, 买入冷却, 卖出冷却, 跟踪止损, 最大止损, 仓位比例表)
    g.capital_tiers = {
        'micro': {                          # <1.5万
            'max_hold': 1,                  # 集中1只，2万以下买2只每只太少
            'signal_level': 2,
            'cooldown_buy': 5,
            'cooldown_sell': 5,
            'trailing_stop_pct': 0.05,
            'max_loss_pct': 0.08,
            'position_ratio': {3: 0.90, 2: 0.75, 1: 0.50},  # 集中火力
        },
        'small': {                          # 1.5万-5万（回测验证档位）
            'max_hold': 2,
            'signal_level': 2,
            'cooldown_buy': 5,
            'cooldown_sell': 5,
            'trailing_stop_pct': 0.05,
            'max_loss_pct': 0.08,
            'position_ratio': {3: 0.80, 2: 0.60, 1: 0.45},
        },
        'medium': {                         # 5万-10万
            'max_hold': 3,                  # 3只ETF全覆盖
            'signal_level': 2,
            'cooldown_buy': 5,
            'cooldown_sell': 5,
            'trailing_stop_pct': 0.05,
            'max_loss_pct': 0.08,
            'position_ratio': {3: 0.70, 2: 0.55, 1: 0.40},  # 单只比例降低（持仓更多）
        },
        'large': {                          # >10万
            'max_hold': 3,
            'signal_level': 2,
            'cooldown_buy': 5,
            'cooldown_sell': 5,
            'trailing_stop_pct': 0.06,      # 略宽止损，资金厚可以扛波动
            'max_loss_pct': 0.09,
            'position_ratio': {3: 0.60, 2: 0.45, 1: 0.35},
        },
    }

    g.current_tier = None  # 当前档位名称

    # ---- 运行时状态 ----
    g.buy_signal_history = {}
    g.sell_signal_history = {}
    g.highest_since_buy = {}

    run_daily(update_tier, time='09:30')    # 每日开盘前更新档位
    run_daily(market_open, time='09:35')
    run_daily(update_highest, time='15:00')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


# ============================================================
#  动态资金档位
# ============================================================
def update_tier(context):
    """每日根据总资产更新资金档位"""
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
        tier_cfg = g.capital_tiers[new_tier]
        log.info('[档位变更] %s → %s | 总资产:%.0f | 最大持仓:%d | 止损:%.0f%%' % (
            old_tier, new_tier, total,
            tier_cfg['max_hold'], tier_cfg['trailing_stop_pct'] * 100))


def get_tier_param(param_name):
    """获取当前档位的参数值"""
    return g.capital_tiers[g.current_tier][param_name]


# ============================================================
#  技术指标计算（完整保留全部V7逻辑）
# ============================================================
def calc_sma(series, n, m):
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


def calc_indicators(code, end_date, count=120):
    df = get_price(code, end_date=end_date, count=count,
                   frequency='daily',
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   skip_paused=True, fq='pre')

    if df is None or len(df) < 80:
        return None

    O = df['open']
    C = df['close']
    H = df['high']
    L = df['low']
    V = df['volume']

    # ------ 均线 ------
    MA5 = C.rolling(5).mean()
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean()

    # ------ KDJ ------
    low9 = L.rolling(9).min()
    high9 = H.rolling(9).max()
    RSV = (C - low9) / (high9 - low9) * 100
    RSV = RSV.fillna(50)
    K0 = calc_sma(RSV, 3, 1)
    D0 = calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    # ------ RSI6 ------
    LC = C.shift(1)
    diff_c = C - LC
    pos = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = calc_sma(pos.fillna(0), 6, 1)
    sma_abs = calc_sma(abs_diff.fillna(0), 6, 1)
    RSI6 = sma_pos / sma_abs.replace(0, np.nan) * 100
    RSI6 = RSI6.fillna(50)

    # ------ CCI ------
    TYP0 = (H + L + C) / 3
    ma_typ = TYP0.rolling(14).mean()
    avedev_typ = TYP0.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    CCI0 = (TYP0 - ma_typ) / (0.015 * avedev_typ.replace(0, np.nan))
    CCI0 = CCI0.fillna(0)

    # ------ 量比 ------
    V5 = V.rolling(5).mean()
    VR = V / V5.shift(1).replace(0, np.nan)
    VR = VR.fillna(1)

    # ------ K线形态 ------
    实体 = (C - O).abs()
    上影 = H - pd.concat([C, O], axis=1).max(axis=1)
    下影 = pd.concat([C, O], axis=1).min(axis=1) - L
    阳线 = C >= O
    阴线 = C <= O

    # ------ BIAS ------
    BIAS20 = (C - MA20) / MA20 * 100

    # ------ MACD ------
    DIF0 = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    DEA0 = DIF0.ewm(span=9, adjust=False).mean()

    # 趋势判断（五档）
    MA20_5ago = MA20.shift(5)
    趋势分 = (
        (C > MA20).astype(int)
        + (MA20 > MA20_5ago).astype(int)
        + (C > MA60).astype(int)
        + (MA20 > MA60).astype(int)
        + (DIF0 > DEA0).astype(int)
    )
    趋势系数 = 趋势分.map(lambda x: {0: -2, 1: -1, 2: 0, 3: 1}.get(x, 2))

    # 拉伸度（标准化融合）
    RSI标准 = RSI6 - 50
    KDJ标准 = (J0 - 50) * 0.6
    CCI标准 = CCI0 * 0.2
    BIAS标准 = BIAS20 * 4

    拉伸 = RSI标准 * 0.3 + KDJ标准 * 0.3 + CCI标准 * 0.2 + BIAS标准 * 0.2
    拉伸线 = 拉伸.ewm(span=2, adjust=False).mean()

    # 背离检测
    price_low_20 = C.rolling(20).min()
    stretch_low_20 = 拉伸线.rolling(20).min()
    价格新低 = C <= price_low_20
    拉伸未新低 = 拉伸线 > stretch_low_20 * 0.85
    底背离 = 价格新低 & 拉伸未新低 & (拉伸线 < -10)

    price_high_20 = C.rolling(20).max()
    stretch_high_20 = 拉伸线.rolling(20).max()
    价格新高 = C >= price_high_20
    拉伸未新高 = 拉伸线 < stretch_high_20 * 0.85
    顶背离 = 价格新高 & 拉伸未新高 & (拉伸线 > 10)

    # 买入信号（6个条件）
    BU1 = (C > MA10) & (C.shift(1) <= MA10.shift(1)) & (VR > 1.0)
    BU2 = ((J0 < 15) | (RSI6 < 25)) & 阳线 & (实体 > 0)
    BU3 = (下影 > 实体 * 2) & (下影 > 上影 * 2) & (拉伸线 < -5)
    BU4 = ((MA20 > MA20.shift(1))
           & (L <= MA20 * 1.02)
           & (C > MA20)
           & 阳线
           & (趋势系数 >= 1))
    BU5 = (J0.shift(1) < 10) & 阳线 & (实体 > 0)
    BU6 = 底背离

    买原分 = (BU1.astype(float) * 1.0
             + BU2.astype(float) * 1.5
             + BU3.astype(float) * 1.0
             + BU4.astype(float) * 1.5
             + BU5.astype(float) * 1.0
             + BU6.astype(float) * 1.5)

    买分 = 买原分.copy()
    买分 = 买分 + (趋势系数 >= 1).astype(float) * 1.0
    买分 = 买分 - (趋势系数 <= -1).astype(float) * 1.0

    # 卖出信号（6个条件）
    SE1 = (C < MA10) & (C.shift(1) >= MA10.shift(1))
    SE2 = (C < C.shift(1) * 0.97) & (VR > 1.5)
    SE3 = (上影 > 实体 * 2) & (上影 > 下影 * 2) & (拉伸线 > 15)
    high5 = C.rolling(5).max()
    SE4 = C < high5 * 0.93
    SE5 = (J0.shift(1) > 90) & 阴线 & (实体 > 0)
    SE6 = 顶背离

    卖原分 = (SE1.astype(float) * 1.0
             + SE2.astype(float) * 1.5
             + SE3.astype(float) * 1.0
             + SE4.astype(float) * 1.5
             + SE5.astype(float) * 1.0
             + SE6.astype(float) * 1.5)

    卖分 = 卖原分.copy()
    卖分 = 卖分 + (趋势系数 < 0).astype(float) * 1.0
    卖分 = 卖分 - (趋势系数 >= 2).astype(float) * 0.5

    # 信号分级
    idx = -1

    result = {
        'code': code,
        'close': C.iloc[idx],
        '买分': 买分.iloc[idx],
        '卖分': 卖分.iloc[idx],
        '趋势系数': 趋势系数.iloc[idx],
        '拉伸线': 拉伸线.iloc[idx],
        'BU_details': [BU1.iloc[idx], BU2.iloc[idx], BU3.iloc[idx],
                       BU4.iloc[idx], BU5.iloc[idx], BU6.iloc[idx]],
        'SE_details': [SE1.iloc[idx], SE2.iloc[idx], SE3.iloc[idx],
                       SE4.iloc[idx], SE5.iloc[idx], SE6.iloc[idx]],
    }

    bs = result['买分']
    ts = result['趋势系数']
    if bs >= 3 or (bs >= 2.5 and ts >= 1):
        result['买入级别'] = 3
    elif bs >= 2:
        result['买入级别'] = 2
    elif bs >= 1.5:
        result['买入级别'] = 1
    else:
        result['买入级别'] = 0

    ss = result['卖分']
    if ss >= 3 or (ss >= 2 and ts < 0):
        result['卖出级别'] = 3
    elif ss >= 2:
        result['卖出级别'] = 2
    elif ss >= 1:
        result['卖出级别'] = 1
    else:
        result['卖出级别'] = 0

    return result


# ============================================================
#  冷却与记录
# ============================================================
def check_cooldown(history_dict, code, today, cooldown_days):
    if code not in history_dict:
        return False
    recent = [d for d in history_dict[code] if (today - d).days <= cooldown_days]
    return len(recent) >= 1


def record_signal(history_dict, code, today):
    if code not in history_dict:
        history_dict[code] = []
    history_dict[code].append(today)
    history_dict[code] = [d for d in history_dict[code] if (today - d).days <= 30]


# ============================================================
#  跟踪止损：每日更新最高价
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
#  仓位计算
# ============================================================
def calc_position_size(context, signal_level, price):
    available = context.portfolio.available_cash
    ratio_map = get_tier_param('position_ratio')
    ratio = ratio_map.get(signal_level, 0.45)

    amount = available * ratio
    shares = int(amount / price / 100) * 100

    if shares < 100:
        if available >= price * 100 * 1.003:
            shares = 100
        else:
            shares = 0

    return shares


# ============================================================
#  每日交易主函数
# ============================================================
def market_open(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()

    # ========== 第一步：检查持仓卖出 ==========
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue

        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost

        # --- 跟踪止损 ---
        trailing_stop = get_tier_param('trailing_stop_pct')
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            drawdown = (highest - cur_price) / highest
            if drawdown >= trailing_stop:
                log.info('[跟踪止损|%s] %s 最高%.3f 现价%.3f 回撤%.1f%%(阈值%.0f%%) 盈亏%.1f%%' % (
                    g.current_tier, code, highest, cur_price,
                    drawdown * 100, trailing_stop * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- 最大亏损止损 ---
        max_loss = get_tier_param('max_loss_pct')
        if profit_pct <= -max_loss:
            log.info('[最大止损] %s 亏损%.1f%%' % (code, profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            record_signal(g.sell_signal_history, code, today)
            continue

        # --- 信号卖出 ---
        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        sell_level = sig['卖出级别']

        if check_cooldown(g.sell_signal_history, code, today, get_tier_param('cooldown_sell')):
            if sell_level < 3:
                continue

        if sell_level >= 2:
            log.info('[信号卖出] %s 级别=%d 卖分=%.1f 趋势=%d 盈亏=%.1f%%' % (
                code, sell_level, sig['卖分'], sig['趋势系数'], profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            record_signal(g.sell_signal_history, code, today)

    # ========== 第二步：检查是否可以买入 ==========
    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])

    if hold_count >= get_tier_param('max_hold'):
        return

    if context.portfolio.available_cash < 500:
        return

    # ========== 第三步：扫描ETF池信号 ==========
    buy_signals = []
    for code in g.etf_pool:
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        if check_cooldown(g.buy_signal_history, code, today, get_tier_param('cooldown_buy')):
            continue

        if current_data[code].paused:
            continue

        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        buy_level = sig['买入级别']
        sell_level = sig['卖出级别']

        if buy_level == 0 or sell_level >= 1:
            continue

        if buy_level < get_tier_param('signal_level'):
            continue

        # ETF也做趋势过滤：至少偏多
        ts = sig['趋势系数']
        if ts < 1:
            continue

        buy_signals.append(sig)

    if not buy_signals:
        return

    # 按买分降序
    buy_signals.sort(key=lambda x: x['买分'], reverse=True)

    slots = get_tier_param('max_hold') - hold_count
    for sig in buy_signals[:slots]:
        code = sig['code']
        price = sig['close']
        level = sig['买入级别']

        shares = calc_position_size(context, level, price)
        if shares <= 0:
            continue

        level_name = {3: '强买', 2: '中买', 1: '弱买'}[level]
        log.info('[%s] %s 买分=%.1f 趋势=%d 拉伸=%.1f %d股 @%.3f' % (
            level_name, code, sig['买分'], sig['趋势系数'],
            sig['拉伸线'], shares, price))
        log.info('  BU: %s' % sig['BU_details'])

        order(code, shares)
        g.highest_since_buy[code] = price
        record_signal(g.buy_signal_history, code, today)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    log.info('=' * 55)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        context.portfolio.total_value,
        context.portfolio.available_cash,
        len(hold),
        get_tier_param('max_hold')))

    for code, pos in hold.items():
        profit_pct = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        highest = g.highest_since_buy.get(code, pos.price)
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f 盈亏:%.1f%% 持:%d' % (
            code, pos.avg_cost, pos.price, highest, profit_pct, pos.total_amount))
    log.info('=' * 55)
