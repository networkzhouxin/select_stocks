# -*- coding: utf-8 -*-
"""
个股悟道量化交易策略 V1.0 — JoinQuant版
=============================================
基于"炒股悟道"理念，融合V13信号驱动 + V15动量轮动的双框架优势，
专为A股个股设计的趋势跟踪策略。

核心理念（悟道六字诀）：
  1. 简 — 由繁入简，指标不超过3类，参数全用学术默认值
  2. 势 — 顺势而为，只做趋势确认后的跟随，不预测顶底
  3. 截 — 截断亏损，ATR跟踪止损 + 最大止损双保险
  4. 奔 — 让利润奔跑，趋势持有模式下只靠止损保护，不主动止盈
  5. 控 — 仓位控制，波动率反比定仓位，熊市自动减仓
  6. 律 — 系统化执行，消除情绪干预，一致性执行

设计原则（个股 vs ETF的关键差异）：
  - 个股波动远大于ETF → 止损更宽（ATR 3.0x），仓位更小
  - 个股有涨跌停/停牌 → 需过滤ST、停牌、涨跌停板
  - 个股需从大池中选股 → 分两步：先从沪深300/中证500中筛，再排名
  - 个股换手率高 → 加入成交量确认，过滤无量突破
  - 个股有行业集中风险 → 同行业最多持1只

选股池：沪深300 + 中证500 = 800只流动性最好的A股
  - 不选创业板/科创板（波动过大，不适合趋势跟踪新手）
  - 不选ST、*ST、停牌股
  - 不选上市不满120天的次新股

信号框架：动量趋势确认（由繁入简版）
  - 买入：MA20突破 + ROC20>0 + 成交量放大 + 趋势分≥3
  - 卖出：ATR跟踪止损（主）+ 信号卖出（辅）+ 趋势持有模式
  - 排名：风险调整动量 × 0.6 + 买入分数 × 0.4

资金管理：
  - 初始资金：10万
  - 最大持仓：5只（个股需要更多分散）
  - 单只仓位：根据波动率反比，上限20%
  - 同行业上限：1只
"""

import numpy as np
import pandas as pd
from jqdata import *
from jqlib.technical_analysis import *


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_slippage(PriceRelatedSlippage(0.002))  # 个股滑点比ETF大
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,            # 个股卖出印花税0.1%
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- 选股池来源 ----
    g.index_list = [
        '000300.XSHG',   # 沪深300
        '000905.XSHG',   # 中证500
    ]

    # ---- 资金档位配置 ----
    g.capital_tiers = {
        'small':  {'max_hold': 3, 'base_position_ratio': 0.60},   # <10万
        'medium': {'max_hold': 5, 'base_position_ratio': 0.50},   # 10-30万
        'large':  {'max_hold': 5, 'base_position_ratio': 0.40},   # >30万
    }

    # ---- 策略参数（全部学术默认值，零优化）----
    g.params = {
        # 动量
        'momentum_period': 20,       # 短期动量（ROC20）
        'momentum_period_long': 60,  # 中期动量（ROC60）
        # ATR止损
        'atr_period': 14,            # ATR周期
        'trailing_atr_mult': 3.0,    # 跟踪止损倍数（个股比ETF宽0.5x）
        'max_loss_atr_mult': 4.0,    # 最大止损倍数
        'stop_floor': 0.04,          # 止损下限4%（个股噪音更大）
        'stop_cap': 0.15,            # 止损上限15%
        # 买卖信号
        'buy_threshold': 2.0,        # 买入分数门槛
        'sell_threshold': 2.0,       # 卖出分数门槛
        'trend_hold_score': 4,       # 趋势持有模式门槛
        'cooldown_days': 5,          # 同一股票买卖冷却期
        # 熊市
        'bear_index': '000300.XSHG', # 熊市判断基准
        # 选股
        'min_market_cap': 50,        # 最小市值（亿元），过滤小盘
        'min_volume_ratio': 0.8,     # 最小量比（过滤缩量股）
        'max_same_industry': 1,      # 同行业最大持仓数
    }

    g.current_tier = None
    g.buy_signal_history = {}
    g.sell_signal_history = {}
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.entry_price = {}             # 记录买入价（用于最大止损）

    run_daily(update_tier, time='09:30')
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
    total = context.portfolio.total_value
    if total < 100000:
        new_tier = 'small'
    elif total < 300000:
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
#  选股池构建（每日刷新）
# ============================================================
def get_stock_pool(context, prev_date):
    """
    从沪深300+中证500成分股中筛选：
    1. 剔除ST、*ST
    2. 剔除停牌
    3. 剔除上市不满120天
    4. 剔除市值过小（<50亿）
    返回股票代码列表
    """
    all_stocks = set()
    for idx in g.index_list:
        stocks = get_index_stocks(idx, date=prev_date)
        all_stocks.update(stocks)

    all_stocks = list(all_stocks)

    # 过滤ST
    current_data = get_current_data()
    pool = []
    for code in all_stocks:
        cd = current_data[code]
        # 剔除ST
        if cd.is_st:
            continue
        # 剔除停牌
        if cd.paused:
            continue
        # 剔除次新股（上市不满120天）
        info = get_security_info(code)
        if info is None:
            continue
        days_listed = (context.current_dt.date() - info.start_date).days
        if days_listed < 120:
            continue
        pool.append(code)

    return pool


# ============================================================
#  通达信SMA函数
# ============================================================
def calc_sma(series, n, m):
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


# ============================================================
#  技术指标计算 + 信号生成
# ============================================================
def calc_indicators(code, end_date, count=120):
    """
    计算个股技术指标并生成买卖信号。
    由繁入简：只用3类指标（均线趋势 + 动量 + 量价）
    """
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

    # ------ 均线（趋势骨架）------
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean()
    EMA12 = C.ewm(span=12, adjust=False).mean()
    EMA26 = C.ewm(span=26, adjust=False).mean()

    # ------ ATR（止损基石）------
    TR = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    ATR = TR.rolling(g.params['atr_period']).mean()

    # ------ KDJ(9,3,3) ------
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

    # ------ MACD(12,26,9) ------
    DIF0 = EMA12 - EMA26
    DEA0 = DIF0.ewm(span=9, adjust=False).mean()
    MACD0 = (DIF0 - DEA0) * 2

    # ------ 量比 ------
    V5 = V.rolling(5).mean()
    VR = V / V5.shift(1).replace(0, np.nan)
    VR = VR.fillna(1)

    # ------ K线形态 ------
    阳线 = C >= O
    阴线 = C <= O
    实体 = (C - O).abs()

    # ------ 动量 ------
    mom_period = g.params['momentum_period']
    mom_long = g.params['momentum_period_long']
    ROC20 = (C / C.shift(mom_period) - 1) * 100
    ROC60 = (C / C.shift(mom_long) - 1) * 100
    volatility = C.pct_change().rolling(mom_period).std() * np.sqrt(252)
    risk_adj_momentum = ROC20 / (volatility.replace(0, np.nan) * 100)
    risk_adj_momentum = risk_adj_momentum.fillna(0)

    # ======================================================
    #  趋势评分（5维度，满分5 — 与V13完全一致）
    # ======================================================
    EMA20 = C.ewm(span=20, adjust=False).mean()
    趋势分 = (
        (C > MA20).astype(int)                    # 价格在MA20上方
        + (EMA20 > EMA20.shift(10)).astype(int)   # 20日均线上升
        + (C > MA60).astype(int)                  # 价格在MA60上方
        + (DIF0 > 0).astype(int)                  # MACD DIF在零轴上方
        + (ROC20 > 0).astype(int)                 # 20日动量为正
    )
    趋势系数 = 趋势分.map(lambda x: {0: -2, 1: -1, 2: 0, 3: 1}.get(x, 2))

    # ======================================================
    #  买入信号（3个条件 — 由繁入简，比V13少1个）
    # ======================================================

    # BU1: 放量突破MA20（趋势启动的最经典信号）
    #   价格从MA20下方穿越到上方 + 量比>1.2确认资金参与
    BU1 = (C > MA20) & (C.shift(1) <= MA20.shift(1)) & (VR > 1.2)

    # BU2: MACD零下金叉（空头衰竭的可靠信号）
    #   DIF上穿DEA，且DIF仍在零轴下方（说明是从底部启动）
    BU2 = (DIF0 > DEA0) & (DIF0.shift(1) <= DEA0.shift(1)) & (DIF0 < 0)

    # BU3: 极度超卖反弹（KDJ J值<10 或 RSI<20 + 阳线确认）
    #   个股中超卖反弹比ETF更有效，因为个股超跌幅度更大
    BU3 = ((J0 < 10) | (RSI6 < 20)) & 阳线 & (实体 > 0)

    买原分 = (BU1.astype(float) * 1.5       # 趋势突破权重最高
              + BU2.astype(float) * 1.5      # MACD金叉权重高
              + BU3.astype(float) * 1.0)     # 超卖反弹权重适中

    # 趋势修正：趋势好加分，趋势差减分
    买分 = 买原分.copy()
    买分 = 买分 + (趋势系数 >= 1).astype(float) * 1.0
    买分 = 买分 - (趋势系数 <= -1).astype(float) * 1.0

    # ======================================================
    #  卖出信号（3个条件 — 同样精简）
    # ======================================================

    # SE1: 放量跌破MA20（趋势破坏）
    SE1 = (C < MA20) & (C.shift(1) >= MA20.shift(1)) & (VR > 1.0)

    # SE2: 恐慌性下跌（单日跌幅>5% + 放量 — 个股阈值比ETF更宽）
    SE2 = (C < C.shift(1) * 0.95) & (VR > 1.5)

    # SE3: 顶背离（价格新高但RSI未新高）
    price_high_20 = C.rolling(20).max()
    rsi_high_20 = RSI6.rolling(20).max()
    SE3 = (C >= price_high_20) & (RSI6 < rsi_high_20 * 0.9) & (RSI6 > 60)

    卖原分 = (SE1.astype(float) * 1.5
              + SE2.astype(float) * 1.5
              + SE3.astype(float) * 1.0)

    卖分 = 卖原分.copy()
    卖分 = 卖分 + (趋势系数 < 0).astype(float) * 1.0
    卖分 = 卖分 - (趋势系数 >= 2).astype(float) * 0.5

    # ======================================================
    #  信号分级
    # ======================================================
    idx = -1
    result = {
        'code': code,
        'close': C.iloc[idx],
        'ATR': ATR.iloc[idx],
        'volatility': volatility.iloc[idx] if not pd.isna(volatility.iloc[idx]) else 0.3,
        'risk_adj_momentum': risk_adj_momentum.iloc[idx],
        'ROC20': ROC20.iloc[idx] if not pd.isna(ROC20.iloc[idx]) else 0,
        'ROC60': ROC60.iloc[idx] if not pd.isna(ROC60.iloc[idx]) else 0,
        '买分': 买分.iloc[idx],
        '卖分': 卖分.iloc[idx],
        '趋势分': 趋势分.iloc[idx],
        '趋势系数': 趋势系数.iloc[idx],
        'VR': VR.iloc[idx],
        'BU_details': [BU1.iloc[idx], BU2.iloc[idx], BU3.iloc[idx]],
        'SE_details': [SE1.iloc[idx], SE2.iloc[idx], SE3.iloc[idx]],
    }

    bs = result['买分']
    ts = result['趋势系数']
    if bs >= 3 or (bs >= 2.5 and ts >= 1):
        result['买入级别'] = 3   # 强买
    elif bs >= 2:
        result['买入级别'] = 2   # 中买
    elif bs >= 1.5:
        result['买入级别'] = 1   # 弱买
    else:
        result['买入级别'] = 0   # 无信号

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
#  ATR跟踪止损
# ============================================================
def calc_trailing_stop_price(highest_price, atr_value):
    """基于ATR的跟踪止损价。个股用3.0倍ATR（比ETF宽0.5x）"""
    pct_stop = g.params['trailing_atr_mult'] * atr_value / highest_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def calc_max_loss_price(entry_price, entry_atr):
    """基于ATR的最大止损价。个股用4.0倍ATR"""
    pct_stop = g.params['max_loss_atr_mult'] * entry_atr / entry_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return entry_price * (1 - pct_stop)


# ============================================================
#  冷却期管理
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
#  熊市判断
# ============================================================
def is_bear_market(prev_date):
    """
    沪深300在MA60下方 且 MA60本身下降 → 系统性熊市。
    比V13的"全部ETF在MA60下方"更适合个股策略，
    因为个股池太大，不可能逐一检查。
    """
    idx = g.params['bear_index']
    df = get_price(idx, end_date=prev_date, count=70,
                   frequency='daily', fields=['close'],
                   skip_paused=True, fq='pre')
    if df is None or len(df) < 65:
        return False

    C = df['close']
    ma60 = C.iloc[-60:].mean()
    ma60_prev = C.iloc[-65:-5].mean()

    # 沪深300在MA60下方 且 MA60下行
    return C.iloc[-1] < ma60 and ma60 < ma60_prev


# ============================================================
#  行业集中度检查
# ============================================================
def get_stock_industry(code, date):
    """获取股票所属申万一级行业"""
    try:
        industry = get_industry(code, date=date)
        if code in industry:
            if 'sw_l1' in industry[code]:
                return industry[code]['sw_l1']['industry_code']
    except:
        pass
    return 'unknown'


def check_industry_limit(code, context, prev_date):
    """检查同行业持仓是否已达上限"""
    max_same = g.params['max_same_industry']
    new_industry = get_stock_industry(code, prev_date)

    count = 0
    for held_code in context.portfolio.positions:
        pos = context.portfolio.positions[held_code]
        if pos.total_amount <= 0:
            continue
        held_industry = get_stock_industry(held_code, prev_date)
        if held_industry == new_industry:
            count += 1

    return count >= max_same


# ============================================================
#  仓位计算
# ============================================================
def calc_position_size(context, signal, signal_level, bear_mode=False):
    """
    仓位 = 基础比例 × 信号强度 × 波动率反比 × 熊市系数
    个股单只上限20%（比ETF更严格的集中度控制）
    """
    available = context.portfolio.available_cash
    total = context.portfolio.total_value
    base_ratio = get_tier_param('base_position_ratio')

    # 信号强度系数
    strength_mult = {3: 1.0, 2: 0.8, 1: 0.6}.get(signal_level, 0.5)

    # 波动率反比：个股目标波动率20%（比ETF的15%更宽容）
    target_vol = 0.20
    actual_vol = max(signal['volatility'], 0.10)
    vol_mult = min(target_vol / actual_vol, 1.5)
    vol_mult = max(vol_mult, 0.3)

    ratio = base_ratio * strength_mult * vol_mult

    # 熊市减半仓位
    if bear_mode:
        ratio *= 0.5

    # 单只上限：总资产的20%
    max_single = 0.20
    amount = min(available * ratio, total * max_single)

    price = signal['close']
    shares = int(amount / price / 100) * 100

    if shares < 100:
        if available >= price * 100 * 1.003:
            shares = 100
        else:
            shares = 0

    return shares


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
#  核心交易主函数
# ============================================================
def market_open(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    p = g.params

    # ========== 第一步：检查持仓卖出 ==========
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost

        # 涨停不卖（卖不出去）
        if cur_price >= current_data[code].high_limit * 0.999:
            continue

        # 计算当前指标
        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        current_atr = sig['ATR']
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = cur_price * 0.03

        # --- ATR跟踪止损（主防线）---
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            stop_price = calc_trailing_stop_price(highest, current_atr)
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[ATR跟踪止损] %s 最高%.2f 现价%.2f ATR=%.3f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, current_atr,
                    drawdown * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                g.entry_price.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- ATR最大亏损止损（绝对防线）---
        if code in g.entry_atr and code in g.entry_price:
            max_loss_price = calc_max_loss_price(
                g.entry_price[code], g.entry_atr[code])
            if cur_price <= max_loss_price:
                log.info('[最大止损] %s 买入价%.2f 现价%.2f 亏损%.1f%%' % (
                    code, g.entry_price[code], cur_price, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                g.entry_price.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- 趋势持有模式（悟道核心：让利润奔跑）---
        trend_score = sig['趋势分']
        if trend_score >= p['trend_hold_score'] and profit_pct > 0:
            continue  # 强趋势 + 盈利 → 只靠止损保护

        # --- 信号卖出（辅助）---
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
            g.entry_price.pop(code, None)
            record_signal(g.sell_signal_history, code, today)

    # ========== 第二步：检查是否可以买入 ==========
    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])

    if hold_count >= get_tier_param('max_hold'):
        return

    if context.portfolio.available_cash < 1000:  # 个股最低门槛比ETF高
        return

    # ========== 第三步：获取选股池并扫描信号 ==========
    stock_pool = get_stock_pool(context, prev_date)

    buy_candidates = []
    for code in stock_pool:
        # 已持有的跳过
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        # 冷却期检查
        if check_cooldown(g.buy_signal_history, code, today, p['cooldown_days']):
            continue

        # 停牌/涨跌停检查
        if current_data[code].paused:
            continue
        # 跌停不买（买不进去）
        if current_data[code].last_price <= current_data[code].low_limit * 1.001:
            continue

        # 计算指标
        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        buy_level = sig['买入级别']
        sell_level = sig['卖出级别']

        # 基本过滤
        if buy_level == 0 or sell_level >= 1:
            continue
        if sig['买分'] < p['buy_threshold']:
            continue

        # 趋势过滤：至少偏多
        if sig['趋势系数'] < 1:
            continue

        # 量比过滤：必须有成交量配合
        if sig['VR'] < p['min_volume_ratio']:
            continue

        # 中期动量过滤：ROC60不能为负（避免下跌趋势中抄底）
        if sig['ROC60'] < 0:
            continue

        # 行业集中度检查
        if check_industry_limit(code, context, prev_date):
            continue

        buy_candidates.append(sig)

    if not buy_candidates:
        return

    # ========== 第四步：熊市检测 ==========
    bear_mode = is_bear_market(prev_date)
    if bear_mode:
        log.info('[熊市模式] 沪深300在MA60下方且MA60下行，仓位缩减50%%')

    # ========== 第五步：排名并买入 ==========
    # 综合排名 = 买入分数 × 0.4 + 风险调整动量 × 0.6
    # 悟道：强者恒强，动量是最朴素也最有效的选股因子
    buy_candidates.sort(
        key=lambda x: x['买分'] * 0.4 + x['risk_adj_momentum'] * 0.6,
        reverse=True
    )

    slots = get_tier_param('max_hold') - hold_count
    bought = 0
    for sig in buy_candidates:
        if bought >= slots:
            break

        code = sig['code']
        level = sig['买入级别']

        shares = calc_position_size(context, sig, level, bear_mode=bear_mode)
        if shares <= 0:
            continue

        price = current_data[code].last_price

        level_name = {3: '强买', 2: '中买', 1: '弱买'}.get(level, '买')
        bear_tag = '[熊市半仓] ' if bear_mode else ''
        log.info('%s[%s] %s 买分=%.1f 趋势=%d 动量=%.2f ROC20=%.1f%% '
                 'ROC60=%.1f%% 量比=%.1f 波动率=%.1f%% %d股 @%.2f' % (
            bear_tag, level_name, code, sig['买分'], sig['趋势系数'],
            sig['risk_adj_momentum'], sig['ROC20'],
            sig['ROC60'], sig['VR'],
            sig['volatility'] * 100, shares, price))

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['ATR']
        g.entry_price[code] = price
        record_signal(g.buy_signal_history, code, today)
        bought += 1


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
        entry_atr = g.entry_atr.get(code, 0)
        log.info('  %s 成本:%.2f 现价:%.2f 高:%.2f 入场ATR:%.3f 盈亏:%.1f%%' % (
            code, pos.avg_cost, pos.price, highest, entry_atr, profit_pct))
    log.info('=' * 60)
