# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V13.1 - 跨资产扩池+国债兜底
=============================================
基于V13.0（信号驱动最优版，57.5%收益，12.5%回撤），两项改动：
  1. ETF池从3只A股宽基扩展到10只跨资产（借鉴V15.1的池子结构）
     信号系统（BU1-BU4、SE1-SE4）全部通用技术指标，对任何ETF适用
  2. 空仓时国债兜底：无持仓且无买入信号时，买入国债ETF作为"生息现金"
  3. 买入价修正：用T日09:35实时价替代T-1收盘价

不改动的部分：
  - 信号系统（4买4卖+趋势评分+信号分级）100%不变
  - 趋势持有模式（趋势分>=4且盈利→只靠止损）100%不变
  - 熊市半仓（所有ETF在MA60下方→仓位减半）100%不变
  - ATR止损体系（跟踪止损+最大止损）100%不变
  - 所有参数100%不变

ETF池（4A股+3跨市场+3跨资产）：
  - 510300 沪深300  - 159915 创业板  - 512100 中证1000  - 159928 消费ETF
  - 513100 纳指ETF  - 513500 标普500  - 159920 恒生ETF
  - 518880 黄金ETF  - 511010 国债ETF  - 159985 豆粕ETF
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

    # ---- ETF标的池（V13.1：从3只扩展到10只跨资产）----
    g.etf_pool = [
        # ---- A股 4只 ----
        '510300.XSHG',   # 沪深300（大盘均衡）
        '159915.XSHE',   # 创业板（成长弹性）
        '512100.XSHG',   # 中证1000（小盘弹性）
        '159928.XSHE',   # 消费ETF（内需消费）
        # ---- 跨市场 3只 ----
        '513100.XSHG',   # 纳指ETF（海外科技）
        '513500.XSHG',   # 标普500ETF（海外均衡）
        '159920.XSHE',   # 恒生ETF（港股）
        # ---- 跨资产 3只 ----
        '518880.XSHG',   # 黄金ETF（避险）
        '511010.XSHG',   # 国债ETF（债券对冲/兜底）
        '159985.XSHE',   # 豆粕ETF（商品周期）
    ]
    g.bond_etf = '511010.XSHG'  # 国债ETF作为兜底标的

    # ---- 资金档位配置（与V13.0一致）----
    g.capital_tiers = {
        'micro': {'max_hold': 1, 'base_position_ratio': 0.85},
        'small': {'max_hold': 2, 'base_position_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_position_ratio': 0.55},
        'large': {'max_hold': 3, 'base_position_ratio': 0.45},
    }

    # ---- 策略参数（与V13.0 100%一致）----
    g.params = {
        'atr_period': 14,
        'trailing_atr_mult': 2.5,
        'max_loss_atr_mult': 3.5,
        'stop_floor': 0.03,
        'stop_cap': 0.15,
        'momentum_period': 20,
        'cooldown_days': 5,
        'buy_threshold': 2.0,
        'sell_threshold': 2.0,
        'trend_hold_score': 4,
    }

    g.current_tier = None
    g.buy_signal_history = {}
    g.sell_signal_history = {}
    g.highest_since_buy = {}
    g.entry_atr = {}

    run_daily(update_tier, time='09:30')
    run_daily(market_open, time='09:35')
    run_daily(update_highest, time='15:00')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


# ============================================================
#  动态资金档位（与V13.0一致）
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
#  技术指标计算（与V13.0 100%一致）
# ============================================================
def calc_sma(series, n, m):
    """通达信SMA函数"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


def calc_indicators(code, end_date, count=120):
    """计算技术指标并生成买卖信号（与V13.0 100%一致）"""
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
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean()
    EMA12 = C.ewm(span=12, adjust=False).mean()
    EMA26 = C.ewm(span=26, adjust=False).mean()

    # ------ ATR ------
    TR = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    ATR = TR.rolling(g.params['atr_period']).mean()

    # ------ KDJ ------
    low9 = L.rolling(9).min()
    high9 = H.rolling(9).max()
    RSV = (C - low9) / (high9 - low9) * 100
    RSV = RSV.fillna(50)
    K0 = calc_sma(RSV, 3, 1)
    D0 = calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    # ------ RSI(6) ------
    LC = C.shift(1)
    diff_c = C - LC
    pos = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = calc_sma(pos.fillna(0), 6, 1)
    sma_abs = calc_sma(abs_diff.fillna(0), 6, 1)
    RSI6 = sma_pos / sma_abs.replace(0, np.nan) * 100
    RSI6 = RSI6.fillna(50)

    # ------ MACD ------
    DIF0 = EMA12 - EMA26
    DEA0 = DIF0.ewm(span=9, adjust=False).mean()
    MACD0 = (DIF0 - DEA0) * 2

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

    # ------ 动量 ------
    mom_period = g.params['momentum_period']
    ROC = (C / C.shift(mom_period) - 1) * 100
    volatility = C.pct_change().rolling(mom_period).std() * np.sqrt(252)
    risk_adj_momentum = ROC / (volatility.replace(0, np.nan) * 100)
    risk_adj_momentum = risk_adj_momentum.fillna(0)

    # ======================================================
    #  趋势评分（5维度）
    # ======================================================
    EMA20 = C.ewm(span=20, adjust=False).mean()
    趋势分 = (
        (C > MA20).astype(int)
        + (EMA20 > EMA20.shift(10)).astype(int)
        + (C > MA60).astype(int)
        + (DIF0 > 0).astype(int)
        + (ROC > 0).astype(int)
    )
    趋势系数 = 趋势分.map(lambda x: {0: -2, 1: -1, 2: 0, 3: 1}.get(x, 2))

    # ======================================================
    #  买入信号（4个条件，与V13.0一致）
    # ======================================================
    BU1 = (C > MA20) & (C.shift(1) <= MA20.shift(1)) & (VR > 1.2)
    BU2 = ((J0 < 10) | (RSI6 < 20)) & 阳线 & (实体 > 0)
    BU3 = (DIF0 > DEA0) & (DIF0.shift(1) <= DEA0.shift(1)) & (DIF0 < 0)

    price_low_20 = C.rolling(20).min()
    rsi_low_20 = RSI6.rolling(20).min()
    BU4 = (C <= price_low_20) & (RSI6 > rsi_low_20 * 1.1) & (RSI6 < 40)

    买原分 = (BU1.astype(float) * 1.5
              + BU2.astype(float) * 1.0
              + BU3.astype(float) * 1.5
              + BU4.astype(float) * 1.0)

    买分 = 买原分.copy()
    买分 = 买分 + (趋势系数 >= 1).astype(float) * 1.0
    买分 = 买分 - (趋势系数 <= -1).astype(float) * 1.0

    # ======================================================
    #  卖出信号（4个条件，与V13.0一致）
    # ======================================================
    SE1 = (C < MA20) & (C.shift(1) >= MA20.shift(1)) & (VR > 1.0)
    SE2 = ((J0 > 90) | (RSI6 > 80)) & 阴线 & (实体 > 0)
    SE3 = (C < C.shift(1) * 0.97) & (VR > 1.5)

    price_high_20 = C.rolling(20).max()
    rsi_high_20 = RSI6.rolling(20).max()
    SE4 = (C >= price_high_20) & (RSI6 < rsi_high_20 * 0.9) & (RSI6 > 60)

    卖原分 = (SE1.astype(float) * 1.5
              + SE2.astype(float) * 1.0
              + SE3.astype(float) * 1.5
              + SE4.astype(float) * 1.0)

    卖分 = 卖原分.copy()
    卖分 = 卖分 + (趋势系数 < 0).astype(float) * 1.0
    卖分 = 卖分 - (趋势系数 >= 2).astype(float) * 0.5

    # ======================================================
    #  信号分级（与V13.0一致）
    # ======================================================
    idx = -1
    result = {
        'code': code,
        'close': C.iloc[idx],
        'ATR': ATR.iloc[idx],
        'volatility': volatility.iloc[idx] if not pd.isna(volatility.iloc[idx]) else 0.2,
        'risk_adj_momentum': risk_adj_momentum.iloc[idx],
        'ROC': ROC.iloc[idx] if not pd.isna(ROC.iloc[idx]) else 0,
        '买分': 买分.iloc[idx],
        '卖分': 卖分.iloc[idx],
        '趋势分': 趋势分.iloc[idx],
        '趋势系数': 趋势系数.iloc[idx],
        'BU_details': [BU1.iloc[idx], BU2.iloc[idx], BU3.iloc[idx], BU4.iloc[idx]],
        'SE_details': [SE1.iloc[idx], SE2.iloc[idx], SE3.iloc[idx], SE4.iloc[idx]],
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
#  ATR动态止损（与V13.0一致）
# ============================================================
def calc_trailing_stop_price(code, highest_price, atr_value):
    atr_stop = highest_price - g.params['trailing_atr_mult'] * atr_value
    pct_stop = (highest_price - atr_stop) / highest_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def calc_max_loss_price(entry_price, entry_atr):
    atr_stop = entry_price - g.params['max_loss_atr_mult'] * entry_atr
    pct_stop = (entry_price - atr_stop) / entry_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return entry_price * (1 - pct_stop)


# ============================================================
#  冷却与记录（与V13.0一致）
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
#  跟踪止损：每日更新最高价（与V13.0一致）
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
#  波动率调整仓位（与V13.0一致）
# ============================================================
def calc_position_size(context, signal, signal_level, bear_mode=False):
    available = context.portfolio.available_cash
    base_ratio = get_tier_param('base_position_ratio')

    strength_mult = {3: 1.0, 2: 0.8, 1: 0.6}.get(signal_level, 0.5)

    target_vol = 0.15
    actual_vol = max(signal['volatility'], 0.05)
    vol_mult = min(target_vol / actual_vol, 1.5)
    vol_mult = max(vol_mult, 0.4)

    ratio = base_ratio * strength_mult * vol_mult

    if bear_mode:
        ratio *= 0.5

    ratio = min(ratio, 0.95)

    price = get_current_data()[signal['code']].last_price  # V13.1：用实时价
    amount = available * ratio
    shares = int(amount / price / 100) * 100

    if shares < 100:
        if available >= price * 100 * 1.003:
            shares = 100
        else:
            shares = 0

    return shares, price  # V13.1：同时返回实时价


# ============================================================
#  熊市判断（与V13.0一致，但扩展到10只ETF池）
# ============================================================
def is_bear_market(prev_date):
    """
    判断是否处于系统性熊市：ETF池中所有标的都在MA60下方。
    V13.1：池子扩展到10只跨资产ETF后，条件更严格——
    需要A股/海外/商品/黄金全部在MA60下方才触发，
    因此只在真正的全球系统性风险时才触发（如2008年）。
    """
    for code in g.etf_pool:
        df = get_price(code, end_date=prev_date, count=60,
                       frequency='daily', fields=['close'],
                       skip_paused=True, fq='pre')
        if df is None or len(df) < 60:
            continue
        ma60 = df['close'].mean()
        if df['close'].iloc[-1] > ma60:
            return False
    return True


# ============================================================
#  每日交易主函数
# ============================================================
def market_open(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    p = g.params

    # ========== 第一步：检查持仓卖出（与V13.0一致）==========
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost

        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        current_atr = sig['ATR']
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = cur_price * 0.02

        # --- ATR跟踪止损 ---
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            stop_price = calc_trailing_stop_price(code, highest, current_atr)
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[ATR跟踪止损] %s 最高%.3f 现价%.3f ATR=%.4f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, current_atr,
                    drawdown * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- ATR最大亏损止损 ---
        if code in g.entry_atr:
            max_loss_price = calc_max_loss_price(pos.avg_cost, g.entry_atr[code])
            if cur_price <= max_loss_price:
                log.info('[ATR最大止损] %s 成本%.3f 现价%.3f 亏损%.1f%%' % (
                    code, pos.avg_cost, cur_price, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- 趋势持有模式 ---
        trend_score = sig['趋势分']
        if trend_score >= p['trend_hold_score'] and profit_pct > 0:
            continue

        # --- 信号卖出 ---
        sell_level = sig['卖出级别']

        if check_cooldown(g.sell_signal_history, code, today, p['cooldown_days']):
            if sell_level < 3:
                continue

        if sell_level >= 2:
            log.info('[信号卖出] %s 级别=%d 卖分=%.1f 趋势=%d 盈亏=%.1f%%' % (
                code, sell_level, sig['卖分'], sig['趋势系数'], profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            record_signal(g.sell_signal_history, code, today)

    # ========== 第二步：检查是否可以买入 ==========
    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])

    if hold_count >= get_tier_param('max_hold'):
        return

    if context.portfolio.available_cash < 500:
        return

    # ========== 第三步：扫描ETF池信号（扩展到10只）==========
    buy_candidates = []
    for code in g.etf_pool:
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        if check_cooldown(g.buy_signal_history, code, today, p['cooldown_days']):
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

        if sig['买分'] < p['buy_threshold']:
            continue

        if sig['趋势系数'] < 1:
            continue

        buy_candidates.append(sig)

    # ========== 第四步：熊市检测（与V13.0一致）==========
    bear_mode = is_bear_market(prev_date)
    if bear_mode:
        log.info('[熊市模式] 所有ETF在MA60下方，仓位缩减50%%')

    # ========== 第五步：买入候选或国债兜底 ==========
    if buy_candidates:
        # 有信号：按买分+动量排序买入（与V13.0一致）
        buy_candidates.sort(
            key=lambda x: x['买分'] * 0.6 + x['risk_adj_momentum'] * 0.4,
            reverse=True
        )

        slots = get_tier_param('max_hold') - hold_count
        for sig in buy_candidates[:slots]:
            code = sig['code']
            level = sig['买入级别']

            shares, price = calc_position_size(context, sig, level, bear_mode=bear_mode)
            if shares <= 0:
                continue

            level_name = {3: '强买', 2: '中买', 1: '弱买'}.get(level, '买')
            bear_tag = '[熊市半仓] ' if bear_mode else ''
            log.info('%s[%s] %s 买分=%.1f 趋势=%d 动量=%.2f 波动率=%.1f%% ATR=%.4f %d股 @%.3f' % (
                bear_tag, level_name, code, sig['买分'], sig['趋势系数'],
                sig['risk_adj_momentum'], sig['volatility'] * 100,
                sig['ATR'], shares, price))
            log.info('  BU: %s' % sig['BU_details'])

            order(code, shares)
            g.highest_since_buy[code] = price  # V13.1：用实时价
            g.entry_atr[code] = sig['ATR']
            record_signal(g.buy_signal_history, code, today)

    elif hold_count == 0:
        # V13.1新增：无持仓且无买入信号 → 国债兜底
        bond = g.bond_etf
        if not current_data[bond].paused:
            bond_pos = context.portfolio.positions.get(bond)
            if bond_pos is None or bond_pos.total_amount <= 0:
                available = context.portfolio.available_cash
                price = current_data[bond].last_price
                shares = int(available * 0.90 / price / 100) * 100
                if shares >= 100:
                    log.info('[国债兜底] 无持仓无信号，买入%s %d股 @%.3f' % (bond, shares, price))
                    order(bond, shares)
                    g.highest_since_buy[bond] = price
                    g.entry_atr[bond] = price * 0.005


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
        atr = g.entry_atr.get(code, 0)
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f 入场ATR:%.4f 盈亏:%.1f%%' % (
            code, pos.avg_cost, pos.price, highest, atr, profit_pct))
    log.info('=' * 60)
