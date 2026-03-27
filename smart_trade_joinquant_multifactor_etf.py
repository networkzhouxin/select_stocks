# -*- coding: utf-8 -*-
"""
多因子自适应ETF量化策略 V2.0 — Multi-Factor Adaptive ETF Strategy
================================================================
V2.0 修复（基于V1.0回测-91.3%的教训）：
  1. 轮动周期 3天→5天，减少60%换手
  2. 换仓门槛：新标的得分必须比持仓最低分高8分以上才替换
  3. 统一 max_hold=3，消除micro/small档位边界跳动
  4. 因子打分3日平滑，降低单日噪音
  5. 最低持仓期5天，避免买入即卖
  6. 因子评分统一为趋势跟随方向（去掉与轮动矛盾的均值回归打分）
  7. 基础买入门槛提高到60分，减少低质量交易

V1.0核心问题：4274次交易×5元最低手续费=21370元，超过2万本金。
排名每天微变导致持仓天天换，根本无法持有趋势。

ETF池（5A股 + 5跨市场 + 3跨资产 = 13只）：
  A股: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 510880红利
  跨市场: 513100纳指, 513500标普500, 159920恒生, 513880日经, 513050中概互联
  跨资产: 518880黄金, 511010国债, 159985豆粕
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

    # ---- ETF标的池（多市场+多资产，13只）----
    g.etf_pool = [
        # ---- A股 5只 ----
        '510300.XSHG',   # 沪深300（大盘均衡）
        '159915.XSHE',   # 创业板（成长弹性）
        '512100.XSHG',   # 中证1000（小盘弹性）
        '159928.XSHE',   # 消费ETF（内需消费）
        '510880.XSHG',   # 红利ETF（高股息防御）
        # ---- 跨市场 5只 ----
        '513100.XSHG',   # 纳指ETF（美国科技）
        '513500.XSHG',   # 标普500ETF（美国均衡）
        '159920.XSHE',   # 恒生ETF（港股）
        '513880.XSHG',   # 日经ETF（日本市场）
        '513050.XSHG',   # 中概互联ETF（海外中概）
        # ---- 跨资产 3只 ----
        '518880.XSHG',   # 黄金ETF（避险）
        '511010.XSHG',   # 国债ETF（债券/兜底）
        '159985.XSHE',   # 豆粕ETF（商品周期）
    ]
    g.bond_etf = '511010.XSHG'

    # ---- 资金档位（V2.0：统一max_hold=3）----
    g.capital_tiers = {
        'micro':  {'max_hold': 3, 'base_ratio': 0.70},  # <1.5万
        'small':  {'max_hold': 3, 'base_ratio': 0.70},  # 1.5-5万
        'medium': {'max_hold': 3, 'base_ratio': 0.60},  # 5-10万
        'large':  {'max_hold': 4, 'base_ratio': 0.55},  # >10万
    }

    # ---- 策略参数 ----
    g.params = {
        'lookback': 120,              # 数据回望天数
        'rebalance_weekdays': [1, 3],  # 轮动日：周二=1,周四=3（V2.1: 自然周历，无基准日依赖）
        'switch_threshold': 8.0,      # 换仓门槛：新标的得分需高于持仓最低分此值（V2.0新增）
        'min_hold_days': 5,           # 最低持仓天数（V2.0新增）
        'smooth_days': 3,             # 因子平滑天数（V2.0新增）
        'rsi_period': 14,             # RSI周期
        'macd_fast': 12,              # MACD快线
        'macd_slow': 26,              # MACD慢线
        'macd_signal': 9,             # MACD信号线
        'bb_period': 20,              # 布林带周期
        'bb_std': 2.0,                # 布林带标准差倍数
        'kdj_n': 9,                   # KDJ周期
        'kdj_m1': 3,                  # KDJ平滑1
        'kdj_m2': 3,                  # KDJ平滑2
        'adx_period': 14,             # ADX周期
        'momentum_period': 20,        # 动量周期
        'vol_ma_period': 20,          # 成交量均线周期
        'atr_period': 14,             # ATR周期
        'trailing_atr_mult': 2.5,     # 跟踪止损ATR倍数
        'stop_floor': 0.03,           # 止损下限3%
        'stop_cap': 0.15,             # 止损上限15%
        'score_buy_threshold': 60,    # 综合得分 > 此值才买入（V2.0: 55→60）
    }

    # ---- 因子权重（基准，会被ADX自适应调整）----
    g.base_weights = {
        'rsi': 0.12,         # RSI趋势确认
        'macd': 0.18,        # MACD趋势（V2.0加重）
        'bollinger': 0.10,   # 布林带趋势位置
        'momentum': 0.25,    # 动量（V2.0加重，核心因子）
        'volume': 0.08,      # 量价配合（V2.0降低，噪音大）
        'kdj': 0.12,         # KDJ
        'ma_trend': 0.15,    # 均线趋势
    }

    g.current_tier = None
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}           # V2.0新增：记录买入日期，用于最低持仓期
    g.holding_scores = {}     # V2.0新增：缓存持仓的上次得分

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
        old = g.current_tier or '初始化'
        g.current_tier = new_tier
        cfg = g.capital_tiers[new_tier]
        log.info('[档位] %s -> %s | 总资产:%.0f | 最大持仓:%d' % (
            old, new_tier, total, cfg['max_hold']))


def get_tier_param(name):
    return g.capital_tiers[g.current_tier][name]


# ============================================================
#  技术指标计算（全部基于T-1数据，无未来函数）
# ============================================================

def calc_rsi(close, period=14):
    """RSI(14)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD(12,26,9)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def calc_bollinger(close, period=20, std_mult=2.0):
    """布林带(20,2)"""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ(9,3,3)"""
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_adx(high, low, close, period=14):
    """ADX(14)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_atr(high, low, close, period=14):
    """ATR(14)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
#  多因子综合评分（V2.0：趋势跟随 + 3日平滑）
# ============================================================
def calc_multi_factor_score(code, end_date):
    """
    V2.0改进：
    - 因子评分统一为趋势跟随方向（去掉均值回归的超卖高分）
    - 使用3日平滑减少单日噪音
    - 要求ROC20 > 0才给合格分（负动量直接低分）
    """
    p = g.params
    df = get_price(code, end_date=end_date, count=p['lookback'],
                   frequency='daily',
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   skip_paused=True, fq='pre')
    if df is None or len(df) < p['lookback'] - 10:
        return None

    C = df['close']
    H = df['high']
    L = df['low']
    V = df['volume']
    sd = p['smooth_days']  # 平滑天数

    if C.iloc[-1] <= 0 or V.iloc[-5:].sum() == 0:
        return None

    # ============ 1. RSI — 趋势跟随评分 ============
    rsi = calc_rsi(C, p['rsi_period'])
    rsi_vals = rsi.iloc[-sd:]
    if rsi_vals.isnull().any():
        return None
    rsi_val = rsi_vals.mean()  # 3日平滑

    # V2.0: 纯趋势方向 — RSI越高越强（但>80超买要小心）
    if rsi_val < 30:
        rsi_score = 20       # 弱势
    elif rsi_val < 40:
        rsi_score = 35       # 偏弱
    elif rsi_val < 50:
        rsi_score = 50       # 中性
    elif rsi_val < 60:
        rsi_score = 65       # 偏强
    elif rsi_val < 70:
        rsi_score = 80       # 强势
    elif rsi_val < 80:
        rsi_score = 75       # 较强但接近过热
    else:
        rsi_score = 55       # 过热，谨慎

    # ============ 2. MACD — 3日平滑 ============
    dif, dea, macd_hist = calc_macd(C, p['macd_fast'], p['macd_slow'], p['macd_signal'])
    # 3日平滑
    dif_val = dif.iloc[-sd:].mean()
    dea_val = dea.iloc[-sd:].mean()
    hist_val = macd_hist.iloc[-sd:].mean()
    hist_prev = macd_hist.iloc[-sd - 3:-3].mean()  # 前一个周期

    macd_score = 50
    if dif_val > dea_val:
        macd_score += 20
    else:
        macd_score -= 20
    if hist_val > 0 and hist_val > hist_prev:
        macd_score += 15     # 红柱放大
    elif hist_val > 0:
        macd_score += 5      # 红柱缩小
    elif hist_val < 0 and hist_val > hist_prev:
        macd_score -= 5      # 绿柱缩小
    else:
        macd_score -= 15     # 绿柱放大
    # 金叉检测（用原始值，不平滑）
    if dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] >= dea.iloc[-1]:
        macd_score += 10
    elif dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] <= dea.iloc[-1]:
        macd_score -= 10
    macd_score = max(0, min(100, macd_score))

    # ============ 3. 布林带 — 趋势位置 ============
    bb_upper, bb_mid, bb_lower = calc_bollinger(C, p['bb_period'], p['bb_std'])
    bb_u = bb_upper.iloc[-1]
    bb_l = bb_lower.iloc[-1]
    bb_width = bb_u - bb_l
    if bb_width <= 0 or pd.isna(bb_width):
        bb_score = 50
    else:
        bb_pos = (C.iloc[-1] - bb_l) / bb_width
        # V2.0: 纯趋势 — 越接近上轨越强，但突破上轨过热
        if bb_pos < 0.2:
            bb_score = 20     # 弱势
        elif bb_pos < 0.4:
            bb_score = 40     # 偏弱
        elif bb_pos < 0.6:
            bb_score = 60     # 中性偏强
        elif bb_pos < 0.8:
            bb_score = 80     # 强势
        elif bb_pos < 0.95:
            bb_score = 75     # 很强但接近上轨
        else:
            bb_score = 55     # 突破上轨，可能过热

    # ============ 4. 动量 ROC20 — 核心因子 ============
    mom_period = p['momentum_period']
    # 3日平滑ROC
    roc_today = C.iloc[-1] / C.iloc[-mom_period] - 1
    roc_y1 = C.iloc[-2] / C.iloc[-mom_period - 1] - 1
    roc_y2 = C.iloc[-3] / C.iloc[-mom_period - 2] - 1
    roc = (roc_today + roc_y1 + roc_y2) / 3.0

    if roc > 0.15:
        mom_score = 95
    elif roc > 0.10:
        mom_score = 85
    elif roc > 0.05:
        mom_score = 75
    elif roc > 0.02:
        mom_score = 65
    elif roc > 0:
        mom_score = 55
    elif roc > -0.03:
        mom_score = 40       # 轻微负动量
    elif roc > -0.08:
        mom_score = 25
    else:
        mom_score = 10       # 深度负动量

    # ============ 5. 成交量 — 3日vs20日量比 ============
    vol_ma = V.iloc[-p['vol_ma_period']:].mean()
    vol_recent = V.iloc[-3:].mean()
    if vol_ma <= 0 or pd.isna(vol_ma):
        vol_score = 50
    else:
        vol_ratio = vol_recent / vol_ma
        price_up = C.iloc[-1] > C.iloc[-3]
        if price_up and vol_ratio > 1.3:
            vol_score = 85
        elif price_up and vol_ratio > 0.8:
            vol_score = 65
        elif price_up:
            vol_score = 50
        elif not price_up and vol_ratio > 1.3:
            vol_score = 25
        elif not price_up and vol_ratio > 0.8:
            vol_score = 40
        else:
            vol_score = 50   # 下跌缩量，中性

    # ============ 6. KDJ — 3日平滑 ============
    k, d, j = calc_kdj(H, L, C, p['kdj_n'], p['kdj_m1'], p['kdj_m2'])
    k_val = k.iloc[-sd:].mean()
    d_val = d.iloc[-sd:].mean()
    j_val = j.iloc[-sd:].mean()
    kdj_score = 50
    if k_val > d_val:
        kdj_score += 15
    else:
        kdj_score -= 15
    # V2.0: 趋势方向 — J值高=强势
    if j_val > 80:
        kdj_score += 10      # 强势
    elif j_val > 50:
        kdj_score += 5       # 偏强
    elif j_val < 20:
        kdj_score -= 15      # 弱势
    # KDJ金叉（用原始值）
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] >= d.iloc[-1]:
        kdj_score += 10
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] <= d.iloc[-1]:
        kdj_score -= 10
    kdj_score = max(0, min(100, kdj_score))

    # ============ 7. 均线趋势（MA10/20/60）============
    ma10 = C.iloc[-10:].mean()
    ma20 = C.iloc[-20:].mean()
    ma60 = C.iloc[-60:].mean()
    cur = C.iloc[-1]

    ma_score = 50
    if cur > ma10:
        ma_score += 10
    else:
        ma_score -= 10
    if cur > ma20:
        ma_score += 10
    else:
        ma_score -= 10
    if cur > ma60:
        ma_score += 10
    else:
        ma_score -= 10
    if ma10 > ma20 > ma60:
        ma_score += 15       # 多头排列
    elif ma10 < ma20 < ma60:
        ma_score -= 15       # 空头排列
    # MA20斜率
    ma20_5d_ago = C.iloc[-25:-5].mean()
    if ma20 > ma20_5d_ago:
        ma_score += 5
    else:
        ma_score -= 5
    ma_score = max(0, min(100, ma_score))

    # ============ ADX自适应权重 ============
    adx, plus_di, minus_di = calc_adx(H, L, C, p['adx_period'])
    adx_val = adx.iloc[-sd:].mean()  # 3日平滑
    if pd.isna(adx_val):
        adx_val = 25

    weights = {}
    for wk in g.base_weights:
        weights[wk] = g.base_weights[wk]

    if adx_val > 30:
        trend_boost = min((adx_val - 30) / 30, 0.5)
        weights['momentum'] = weights['momentum'] + 0.08 * trend_boost
        weights['macd'] = weights['macd'] + 0.06 * trend_boost
        weights['ma_trend'] = weights['ma_trend'] + 0.06 * trend_boost
        weights['rsi'] = weights['rsi'] - 0.06 * trend_boost
        weights['bollinger'] = weights['bollinger'] - 0.04 * trend_boost
        weights['kdj'] = weights['kdj'] - 0.05 * trend_boost
        weights['volume'] = weights['volume'] - 0.05 * trend_boost
    elif adx_val < 20:
        range_boost = min((20 - adx_val) / 15, 0.5)
        weights['rsi'] = weights['rsi'] + 0.06 * range_boost
        weights['bollinger'] = weights['bollinger'] + 0.04 * range_boost
        weights['kdj'] = weights['kdj'] + 0.05 * range_boost
        weights['momentum'] = weights['momentum'] - 0.08 * range_boost
        weights['macd'] = weights['macd'] - 0.04 * range_boost
        weights['ma_trend'] = weights['ma_trend'] - 0.03 * range_boost

    # 归一化权重
    total_w = 0.0
    for _v in weights.values():
        total_w += _v
    if total_w > 0:
        normed = {}
        for k_w in weights:
            normed[k_w] = weights[k_w] / total_w
        weights = normed

    # ============ 综合得分 ============
    scores = {
        'rsi': rsi_score,
        'macd': macd_score,
        'bollinger': bb_score,
        'momentum': mom_score,
        'volume': vol_score,
        'kdj': kdj_score,
        'ma_trend': ma_score,
    }

    final_score = 0.0
    for k_s in scores:
        final_score += scores[k_s] * weights[k_s]

    # ============ ATR ============
    atr = calc_atr(H, L, C, p['atr_period'])
    atr_val = atr.iloc[-1]
    if pd.isna(atr_val):
        atr_val = cur * 0.02

    # ============ 波动率 ============
    returns = C.pct_change().iloc[-20:]
    vol = returns.std() * np.sqrt(252)
    if pd.isna(vol) or vol <= 0:
        vol = 0.20

    return {
        'code': code,
        'final_score': final_score,
        'scores': scores,
        'adx': adx_val,
        'roc': roc,
        'close': cur,
        'atr': atr_val,
        'volatility': vol,
        'rsi': rsi_val,
    }


# ============================================================
#  ATR跟踪止损
# ============================================================
def calc_stop_price(highest, atr_val):
    p = g.params
    pct_stop = p['trailing_atr_mult'] * atr_val / highest
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return highest * (1 - pct_stop)


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
            stop_price = calc_stop_price(highest, g.entry_atr[code])
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[止损] %s 最高%.3f 现%.3f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, drawdown * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                g.buy_date.pop(code, None)
                g.holding_scores.pop(code, None)
                stopped.append(code)

    return stopped


# ============================================================
#  核心交易逻辑（V2.0：换仓门槛 + 最低持仓期）
# ============================================================
def do_trading(context):
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    today = context.current_dt.date()

    # ======== 1. 每日止损检查 ========
    stopped_codes = check_stop_loss(context, current_data)

    # ======== 2. 判断是否轮动日（V2.1: 自然周历，无任何基准日依赖）========
    weekday = today.weekday()  # 周一=0, 周二=1, ..., 周五=4
    is_rebalance_day = (weekday in g.params['rebalance_weekdays'])
    if not is_rebalance_day and not stopped_codes:
        return  # 非轮动日且无止损，跳过

    # ======== 3. 计算所有ETF多因子得分 ========
    all_results = []
    for code in g.etf_pool:
        if current_data[code].paused:
            continue
        result = calc_multi_factor_score(code, prev_date)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return

    all_results.sort(key=lambda x: x['final_score'], reverse=True)

    # 打印TOP5
    log.info('[TOP5]')
    for i, r in enumerate(all_results[:5]):
        log.info('  #%d %s 分:%.1f ADX:%.1f RSI:%.1f ROC:%.1f%%' % (
            i + 1, r['code'], r['final_score'], r['adx'],
            r['rsi'], r['roc'] * 100))

    # ======== 4. V2.0换仓逻辑：门槛 + 最低持仓期 ========
    threshold = g.params['score_buy_threshold']
    switch_th = g.params['switch_threshold']
    min_hold = g.params['min_hold_days']
    max_hold = get_tier_param('max_hold')

    # 合格候选（得分>阈值）
    candidates = [r for r in all_results if r['final_score'] > threshold]

    # 当前持仓信息
    current_holds = {}
    for code in context.portfolio.positions:
        if context.portfolio.positions[code].total_amount > 0:
            current_holds[code] = True

    # 更新持仓得分
    score_map = {}
    for r in all_results:
        score_map[r['code']] = r['final_score']
    for code in current_holds:
        if code in score_map:
            g.holding_scores[code] = score_map[code]

    # 决定目标持仓：保留现有持仓中仍合格的，用候选替换不合格的
    target_codes = set()
    protected_codes = set()  # 受最低持仓期保护的标的

    # 先处理现有持仓
    for code in list(current_holds.keys()):
        # 最低持仓期保护
        if code in g.buy_date:
            days_held = len(get_trade_days(start_date=g.buy_date[code], end_date=today))
            if days_held <= min_hold:
                target_codes.add(code)
                protected_codes.add(code)
                continue

        # 持仓得分仍 > 阈值-5（给持仓一定容忍度，避免微跌就卖）
        hold_score = g.holding_scores.get(code, 0)
        if hold_score > threshold - 5:
            target_codes.add(code)

    # 如果目标数 < max_hold，从候选中补充
    for r in candidates:
        if len(target_codes) >= max_hold:
            break
        if r['code'] not in target_codes:
            target_codes.add(r['code'])

    # 换仓门槛检查：如果目标已满，候选要替换持仓需要高出switch_threshold分
    if len(target_codes) >= max_hold:
        # 找出目标中得分最低的非保护持仓
        removable = []
        for code in target_codes:
            if code in current_holds and code not in protected_codes:
                removable.append((code, g.holding_scores.get(code, 0)))
        removable.sort(key=lambda x: x[1])

        for r in candidates:
            if r['code'] in target_codes:
                continue
            if not removable:
                break
            worst_code, worst_score = removable[0]
            if r['final_score'] > worst_score + switch_th:
                target_codes.discard(worst_code)
                target_codes.add(r['code'])
                removable.pop(0)
                log.info('[换仓] %s(%.1f) 替换 %s(%.1f) 差%.1f分' % (
                    r['code'], r['final_score'], worst_code, worst_score,
                    r['final_score'] - worst_score))

    # 国债兜底
    bond = g.bond_etf
    bond_in_targets = bond in target_codes
    if len(target_codes) < max_hold and not bond_in_targets:
        if not current_data[bond].paused:
            target_codes.add(bond)
            g.holding_scores[bond] = 0
            log.info('[国债兜底] 候选不足%d，国债补位' % max_hold)

    # ======== 5. 卖出 ========
    for code in list(current_holds.keys()):
        if code not in target_codes:
            profit = (current_data[code].last_price - context.portfolio.positions[code].avg_cost) / context.portfolio.positions[code].avg_cost
            log.info('[轮动卖出] %s 盈亏%.1f%% 得分:%.1f' % (
                code, profit * 100, g.holding_scores.get(code, 0)))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            g.buy_date.pop(code, None)
            g.holding_scores.pop(code, None)

    # ======== 6. 买入 ========
    to_buy_codes = [c for c in target_codes if c not in current_holds]
    if not to_buy_codes:
        return

    # 构建买入信号映射
    sig_map = {}
    for r in all_results:
        sig_map[r['code']] = r
    # 国债兜底的虚拟信号
    if bond in to_buy_codes and bond not in sig_map:
        sig_map[bond] = {
            'code': bond, 'final_score': 0, 'roc': 0, 'volatility': 0.03,
            'close': current_data[bond].last_price,
            'atr': current_data[bond].last_price * 0.005,
            '_is_bond_fill': True,
        }

    available = context.portfolio.available_cash
    current_target_holds = len(set(current_holds.keys()) & target_codes)
    slots = max_hold - current_target_holds

    if slots <= 0 or available < 500:
        return

    base_ratio = get_tier_param('base_ratio')

    for code in to_buy_codes:
        if slots <= 0 or available < 500:
            break
        if code not in sig_map:
            continue

        sig = sig_map[code]
        price = current_data[code].last_price
        is_bond = sig.get('_is_bond_fill', False)

        if is_bond:
            alloc = available * 0.95
        else:
            alloc = available / slots * base_ratio
            target_vol = 0.15
            actual_vol = max(sig['volatility'], 0.05)
            vol_mult = min(target_vol / actual_vol, 1.5)
            vol_mult = max(vol_mult, 0.4)
            alloc = alloc * vol_mult

        alloc = min(alloc, available * 0.95)

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            if available >= price * 100 * 1.003:
                shares = 100
            else:
                continue

        if is_bond:
            log.info('[国债兜底买入] %s %d股 @%.3f' % (code, shares, price))
        else:
            log.info('[买入] %s 分:%.1f ROC:%.1f%% 波动%.1f%% %d股 @%.3f' % (
                code, sig['final_score'], sig['roc'] * 100,
                sig['volatility'] * 100, shares, price))

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['atr']
        g.buy_date[code] = today
        g.holding_scores[code] = sig['final_score']
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
    hold = {}
    for c in positions:
        if positions[c].total_amount > 0:
            hold[c] = positions[c]

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        context.portfolio.total_value,
        context.portfolio.available_cash,
        len(hold),
        get_tier_param('max_hold')))

    for code in hold:
        pos = hold[code]
        pnl = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        highest = g.highest_since_buy.get(code, pos.price)
        score = g.holding_scores.get(code, 0)
        log.info('  %s 成本:%.3f 现:%.3f 高:%.3f 盈亏:%.1f%% 分:%.1f' % (
            code, pos.avg_cost, pos.price, highest, pnl, score))
    log.info('=' * 60)
