# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V10.1 - 自适应增强版
=============================================
在V10.0基础上增加 Kaufman 自适应机制，使策略自动适应不同ETF的市场特征。

核心改进 vs V10.0：
  1. Kaufman效率比率(ER)：衡量每只ETF当前是"趋势型"还是"震荡型"
     - ER高(趋势强) → 使用快速指标、宽止损、偏重趋势信号
     - ER低(震荡多) → 使用慢速指标、窄止损、偏重均值回归信号
  2. KAMA自适应均线：替代固定MA20，在趋势中紧跟价格，在震荡中过滤噪音
  3. 自适应MACD：快慢线周期根据ER自动调整，不再是固定12/26
  4. 自适应信号权重：趋势型市场加大突破信号权重，震荡型市场加大超卖反弹权重
  5. 自适应ATR止损倍数：趋势强时宽止损(3x)，震荡时窄止损(2x)

理论依据：
  - Kaufman Efficiency Ratio (Perry Kaufman, 1995)
  - KAMA (Kaufman Adaptive Moving Average)
  - 自适应参数不是"为每个标的设不同参数"（那是过拟合）
  - 而是"同一个算法，根据实时数据自动产生不同参数"（这是自适应）

无未来函数保证：
  - 所有数据通过 get_prev_trade_date() 取前一交易日
  - set_option('avoid_future_data', True)
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

    # ---- ETF标的池 ----
    g.etf_pool = [
        '510300.XSHG',   # 沪深300ETF
        '159915.XSHE',   # 创业板ETF
        '510500.XSHG',   # 中证500ETF
    ]

    # ---- 资金档位配置 ----
    g.capital_tiers = {
        'micro': {'max_hold': 1, 'base_position_ratio': 0.85},
        'small': {'max_hold': 2, 'base_position_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_position_ratio': 0.55},
        'large': {'max_hold': 3, 'base_position_ratio': 0.45},
    }

    # ---- 策略参数 ----
    g.params = {
        'atr_period': 14,
        'er_period': 10,             # 效率比率计算周期（Kaufman标准值）
        'kama_fast': 2,              # KAMA最快周期（标准值）
        'kama_slow': 30,             # KAMA最慢周期（标准值）
        'trailing_atr_base': 2.5,    # ATR止损基础倍数
        'trailing_atr_trend': 3.5,   # 强趋势时ATR止损倍数
        'trailing_atr_chop': 2.0,    # 震荡时ATR止损倍数
        'max_loss_atr_mult': 3.5,
        'stop_floor': 0.03,
        'stop_cap': 0.15,
        'momentum_period': 20,
        'cooldown_days': 5,
        'buy_threshold': 2.0,
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
#  Kaufman自适应核心函数
# ============================================================
def calc_sma(series, n, m):
    """通达信SMA函数"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


def calc_efficiency_ratio(close, period=10):
    """
    Kaufman效率比率 (Efficiency Ratio)
    ER = |方向变化| / 总路径长度
    ER接近1：强趋势（价格朝一个方向走，噪音小）
    ER接近0：震荡（价格来回走，噪音大）

    这是一个纯粹的数据度量，不包含任何拟合参数。
    不同ETF因为波动特征不同，ER自然产生不同值。
    """
    direction = (close - close.shift(period)).abs()
    volatility_path = close.diff().abs().rolling(period).sum()
    er = direction / volatility_path.replace(0, np.nan)
    return er.fillna(0).clip(0, 1)


def calc_kama(close, er, fast_period=2, slow_period=30):
    """
    Kaufman自适应移动均线 (KAMA)
    - 趋势强(ER高) → KAMA接近快速EMA → 紧跟价格
    - 震荡多(ER低) → KAMA接近慢速EMA → 过滤噪音

    对比固定MA20：
    - 510300(低波)：KAMA偏慢，减少假突破信号
    - 159915(高波趋势)：KAMA偏快，更早捕捉趋势转折
    - 无需手动设定不同参数，算法自动适应
    """
    fast_sc = 2.0 / (fast_period + 1)   # 最快平滑常数 = 0.667
    slow_sc = 2.0 / (slow_period + 1)   # 最慢平滑常数 = 0.0645

    # 自适应平滑常数：ER高→接近fast_sc，ER低→接近slow_sc
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])
    return kama


def calc_adaptive_ema(close, er, fast_span=5, slow_span=40):
    """
    自适应EMA：用ER调节EMA的有效周期
    ER高 → 有效span接近fast_span → 反应快
    ER低 → 有效span接近slow_span → 反应慢

    用于构造自适应MACD，替代固定EMA(12)和EMA(26)
    """
    fast_alpha = 2.0 / (fast_span + 1)
    slow_alpha = 2.0 / (slow_span + 1)

    # alpha随ER变化
    alpha = er * (fast_alpha - slow_alpha) + slow_alpha

    result = pd.Series(index=close.index, dtype=float)
    result.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        a = alpha.iloc[i]
        result.iloc[i] = a * close.iloc[i] + (1 - a) * result.iloc[i - 1]
    return result


# ============================================================
#  技术指标计算（自适应版）
# ============================================================
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
    p = g.params

    # ======================================================
    #  Step 1: 计算效率比率（核心自适应因子）
    # ======================================================
    ER = calc_efficiency_ratio(C, p['er_period'])

    # ======================================================
    #  Step 2: 自适应均线体系
    # ======================================================
    # KAMA替代MA20：自动适应每只ETF的趋势/震荡特征
    KAMA = calc_kama(C, ER, p['kama_fast'], p['kama_slow'])

    # 固定均线（仍保留用于趋势判断）
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean()

    # ======================================================
    #  Step 3: 自适应MACD
    # ======================================================
    # 传统MACD用固定EMA(12)和EMA(26)
    # 自适应MACD：快线有效周期5-20，慢线有效周期15-50
    # ER高(趋势) → 快线~5/慢线~15 → 反应灵敏 → 创业板受益
    # ER低(震荡) → 快线~20/慢线~50 → 过滤噪音 → 沪深300受益
    adaptive_fast = calc_adaptive_ema(C, ER, fast_span=5, slow_span=20)
    adaptive_slow = calc_adaptive_ema(C, ER, fast_span=15, slow_span=50)
    A_DIF = adaptive_fast - adaptive_slow
    A_DEA = A_DIF.ewm(span=9, adjust=False).mean()
    A_MACD = (A_DIF - A_DEA) * 2

    # 同时保留标准MACD用于对比
    DIF0 = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    DEA0 = DIF0.ewm(span=9, adjust=False).mean()

    # ======================================================
    #  Step 4: ATR
    # ======================================================
    TR = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs()
    ], axis=1).max(axis=1)
    ATR = TR.rolling(p['atr_period']).mean()

    # ======================================================
    #  Step 5: KDJ / RSI
    # ======================================================
    low9 = L.rolling(9).min()
    high9 = H.rolling(9).max()
    RSV = (C - low9) / (high9 - low9) * 100
    RSV = RSV.fillna(50)
    K0 = calc_sma(RSV, 3, 1)
    D0 = calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    LC = C.shift(1)
    diff_c = C - LC
    pos_diff = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = calc_sma(pos_diff.fillna(0), 6, 1)
    sma_abs = calc_sma(abs_diff.fillna(0), 6, 1)
    RSI6 = sma_pos / sma_abs.replace(0, np.nan) * 100
    RSI6 = RSI6.fillna(50)

    # ======================================================
    #  Step 6: 量比 / K线形态 / 动量
    # ======================================================
    V5 = V.rolling(5).mean()
    VR = V / V5.shift(1).replace(0, np.nan)
    VR = VR.fillna(1)

    实体 = (C - O).abs()
    上影 = H - pd.concat([C, O], axis=1).max(axis=1)
    下影 = pd.concat([C, O], axis=1).min(axis=1) - L
    阳线 = C >= O
    阴线 = C <= O

    mom_period = p['momentum_period']
    ROC = (C / C.shift(mom_period) - 1) * 100
    volatility = C.pct_change().rolling(mom_period).std() * np.sqrt(252)
    risk_adj_momentum = ROC / (volatility.replace(0, np.nan) * 100)
    risk_adj_momentum = risk_adj_momentum.fillna(0)

    # ======================================================
    #  趋势评分（5维度，使用KAMA替代固定MA20）
    # ======================================================
    趋势分 = (
        (C > KAMA).astype(int)                     # 价格在KAMA上方（自适应版）
        + (KAMA > KAMA.shift(10)).astype(int)      # KAMA趋势向上
        + (C > MA60).astype(int)                   # 价格在MA60上方
        + (A_DIF > 0).astype(int)                  # 自适应DIF在零轴上方
        + (ROC > 0).astype(int)                    # 20日动量为正
    )
    趋势系数 = 趋势分.map(lambda x: {0: -2, 1: -1, 2: 0, 3: 1}.get(x, 2))

    # ======================================================
    #  当前ER值（用于自适应信号权重和止损倍数）
    # ======================================================
    er_now = ER.iloc[-1]

    # ======================================================
    #  买入信号（4条件，权重根据ER自适应调整）
    # ======================================================
    # 当ER高（趋势型）：趋势信号(BU1/BU3)权重增加
    # 当ER低（震荡型）：均值回归信号(BU2/BU4)权重增加
    trend_weight = 1.0 + er_now * 0.5      # ER=0→1.0, ER=0.5→1.25, ER=1→1.5
    revert_weight = 1.0 + (1 - er_now) * 0.5  # ER=0→1.5, ER=0.5→1.25, ER=1→1.0

    # BU1: 放量突破KAMA（趋势跟踪，用KAMA替代MA20更准确）
    BU1 = (C > KAMA) & (C.shift(1) <= KAMA.shift(1)) & (VR > 1.2)

    # BU2: 极度超卖反转（均值回归）
    BU2 = ((J0 < 10) | (RSI6 < 20)) & 阳线 & (实体 > 0)

    # BU3: 自适应MACD零下金叉（动量反转，用自适应MACD）
    BU3 = (A_DIF > A_DEA) & (A_DIF.shift(1) <= A_DEA.shift(1)) & (A_DIF < 0)

    # BU4: 底背离（价格新低但RSI未新低）
    price_low_20 = C.rolling(20).min()
    rsi_low_20 = RSI6.rolling(20).min()
    BU4 = (C <= price_low_20) & (RSI6 > rsi_low_20 * 1.1) & (RSI6 < 40)

    # 自适应加权
    买原分 = (BU1.astype(float) * 1.5 * trend_weight       # 趋势信号×趋势权重
              + BU2.astype(float) * 1.0 * revert_weight     # 均值回归×回归权重
              + BU3.astype(float) * 1.5 * trend_weight      # 趋势信号×趋势权重
              + BU4.astype(float) * 1.0 * revert_weight)    # 均值回归×回归权重

    买分 = 买原分.copy()
    买分 = 买分 + (趋势系数 >= 1).astype(float) * 1.0
    买分 = 买分 - (趋势系数 <= -1).astype(float) * 1.0

    # ======================================================
    #  卖出信号（4条件，同样自适应加权）
    # ======================================================
    # SE1: 放量跌破KAMA
    SE1 = (C < KAMA) & (C.shift(1) >= KAMA.shift(1)) & (VR > 1.0)

    # SE2: 极度超买回落
    SE2 = ((J0 > 90) | (RSI6 > 80)) & 阴线 & (实体 > 0)

    # SE3: 恐慌性下跌
    SE3 = (C < C.shift(1) * 0.97) & (VR > 1.5)

    # SE4: 顶背离
    price_high_20 = C.rolling(20).max()
    rsi_high_20 = RSI6.rolling(20).max()
    SE4 = (C >= price_high_20) & (RSI6 < rsi_high_20 * 0.9) & (RSI6 > 60)

    卖原分 = (SE1.astype(float) * 1.5 * trend_weight
              + SE2.astype(float) * 1.0 * revert_weight
              + SE3.astype(float) * 1.5
              + SE4.astype(float) * 1.0 * revert_weight)

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
        'ER': er_now,
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
#  ATR自适应止损
# ============================================================
def calc_trailing_stop_price(highest_price, atr_value, er_value):
    """
    自适应ATR跟踪止损：
    ER高(趋势强) → 用更大的ATR倍数(3.5x) → 给趋势空间
    ER低(震荡) → 用较小的ATR倍数(2.0x) → 及时止损

    举例：
    510300在稳定上涨时(ER≈0.5)，止损倍数≈2.75x ATR
    159915在急涨阶段(ER≈0.7)，止损倍数≈3.05x ATR
    159915在横盘时(ER≈0.2)，止损倍数≈2.30x ATR
    → 同一只ETF在不同阶段也有不同止损！
    """
    p = g.params
    # 线性插值：ER=0→chop倍数, ER=1→trend倍数
    atr_mult = p['trailing_atr_chop'] + er_value * (p['trailing_atr_trend'] - p['trailing_atr_chop'])

    atr_stop = highest_price - atr_mult * atr_value
    pct_stop = (highest_price - atr_stop) / highest_price
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def calc_max_loss_price(entry_price, entry_atr):
    p = g.params
    atr_stop = entry_price - p['max_loss_atr_mult'] * entry_atr
    pct_stop = (entry_price - atr_stop) / entry_price
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return entry_price * (1 - pct_stop)


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
#  跟踪止损更新
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
#  波动率调整仓位
# ============================================================
def calc_position_size(context, signal, signal_level):
    available = context.portfolio.available_cash
    base_ratio = get_tier_param('base_position_ratio')

    strength_mult = {3: 1.0, 2: 0.8, 1: 0.6}.get(signal_level, 0.5)

    # 波动率反比
    target_vol = 0.15
    actual_vol = max(signal['volatility'], 0.05)
    vol_mult = min(target_vol / actual_vol, 1.5)
    vol_mult = max(vol_mult, 0.4)

    # ER加成：趋势越强，可以给更大仓位（因为趋势行情胜率更高）
    er_mult = 0.85 + signal['ER'] * 0.3  # ER=0→0.85, ER=0.5→1.0, ER=1→1.15

    ratio = base_ratio * strength_mult * vol_mult * er_mult
    ratio = min(ratio, 0.95)

    price = signal['close']
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

        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        current_atr = sig['ATR']
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = cur_price * 0.02

        er_value = sig['ER']

        # --- 自适应ATR跟踪止损 ---
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            stop_price = calc_trailing_stop_price(highest, current_atr, er_value)
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[自适应止损] %s ER=%.2f 最高%.3f 现%.3f ATR=%.4f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, er_value, highest, cur_price, current_atr,
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
                log.info('[最大止损] %s 成本%.3f 现价%.3f 亏损%.1f%%' % (
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
            log.info('[信号卖出] %s 级别=%d 卖分=%.1f 趋势=%d ER=%.2f 盈亏=%.1f%%' % (
                code, sell_level, sig['卖分'], sig['趋势系数'],
                er_value, profit_pct * 100))
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

    # ========== 第三步：扫描ETF池 ==========
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

    if not buy_candidates:
        return

    # 排序：买分强度(60%) + 风险调整动量(25%) + ER趋势质量(15%)
    buy_candidates.sort(
        key=lambda x: x['买分'] * 0.60 + x['risk_adj_momentum'] * 0.25 + x['ER'] * 0.15,
        reverse=True
    )

    slots = get_tier_param('max_hold') - hold_count
    for sig in buy_candidates[:slots]:
        code = sig['code']
        price = sig['close']
        level = sig['买入级别']

        shares = calc_position_size(context, sig, level)
        if shares <= 0:
            continue

        level_name = {3: '强买', 2: '中买', 1: '弱买'}.get(level, '买')
        log.info('[%s] %s 买分=%.1f 趋势=%d ER=%.2f 动量=%.2f 波动%.1f%% %d股 @%.3f' % (
            level_name, code, sig['买分'], sig['趋势系数'],
            sig['ER'], sig['risk_adj_momentum'],
            sig['volatility'] * 100, shares, price))
        log.info('  BU: %s' % sig['BU_details'])

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['ATR']
        record_signal(g.buy_signal_history, code, today)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    log.info('=' * 65)
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
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f ATR:%.4f 盈亏:%.1f%%' % (
            code, pos.avg_cost, pos.price, highest, atr, profit_pct))
    log.info('=' * 65)
