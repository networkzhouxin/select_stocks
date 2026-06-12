# -*- coding: utf-8 -*-
"""
多因子ETF量化策略 V2.10
============================
基于7个经典技术指标（RSI/MACD/布林带/动量/成交量/KDJ/均线趋势）综合评分，
固定因子权重，ATR跟踪止损+止损豁免，多市场多资产ETF轮动。

V2.10 vs V2.9:
  - 均线趋势权重 21%→24%，Walk-Forward验证(8窗口) 24%在7/8年OOS优于22%
    近3个窗口训练最优均为24%，OOS曲线平坦但24%一致性最高

V2.9 vs V2.8:
  - 均线趋势权重 15%→21%，其余因子等比例微降 (+46pp, 8/8 walk-forward窗口全胜)
    均线趋势是多时间维度趋势确认的最可靠信号，原权重严重低配

V2.8 vs V2.7:
  - 黄金ETF(518880)单独使用更紧止损: stop_floor 0.03, trailing_atr_mult 2.0 (+16.9pp)

V2.7 vs V2.6:
  - bb_period 20→25, bb_std 2.0→1.8（布林带更平滑宽容，趋势中不因触轨降分）
  - stop_floor 0.03→0.05（过滤ETF日波动噪音止损，减少无效交易）

核心机制：
  - 7因子离散分档评分 + 3日平滑 + 固定权重（稳定排名，减少噪音换仓）
  - 周二+周四固定轮动（无起始日依赖）
  - 换仓门槛8分（新标的必须高出持仓最低分8分才替换）
  - 最低持仓期5天（防止买入即卖）
  - ATR跟踪止损 + 利润分段收紧（盈利5-15%→2.0x, >15%→1.5x）
  - MA10趋势止损（止损豁免前检查短期趋势是否破坏）
  - 止损豁免（在target中→MA10未破→不卖；MA10已破→执行止损）
  - 动态资金档位（根据总资产自动调整max_hold和base_ratio）
  - 每日收盘后更新最高价+ATR（次日止损更准确）
  - 波动率反比仓位（低波动多买，高波动少买）
  - 买入按得分排序（最强标的优先获得资金）
  - 候选不足时持有现金（不强制兜底，让市场决定持仓数）
  - 组合回撤监控（盘后日志显示当前回撤幅度）

ETF池（5A股 + 5跨市场 + 2跨资产 = 12只）：
  A股: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 510880红利
  跨市场: 513100纳指, 513500标普500, 159920恒生, 513880日经, 513050中概互联
  跨资产: 518880黄金, 159985豆粕

因子权重（42组扫描+walk-forward验证）：
  均线趋势=0.24, 动量ROC20=0.223, MACD=0.161, RSI=0.108, KDJ=0.108, 布林带=0.089, 成交量=0.071

回测业绩（万三+最低5元佣金）：
  V2.10聚宽实测(2015-2026): +371.7%，年化15.35%，最大回撤15.19%，夏普0.95
  2010-2014（样本外）：+39%，年化6.4%，弱市+标的不全仍正收益

版本历史：
  V1.0: 每日轮动无门槛，-91.3%（手续费吞噬本金）
  V2.0: 加门槛+持仓期+离散分档，+234%
  V2.3: 去ADX自适应+7pp，去国债兜底+18pp，固定权重
  V2.4: 止损豁免+35pp（触发止损但得分仍高则不卖）
  V2.5: 止损豁免+回撤上限（得分高可豁免，但回撤≥10%时强制止损，防范得分滞后于价格）
  V2.6: 利润分段ATR(+28pp) + 资本档位优化(+8pp) + RSI极值修正 + MA10趋势止损
  V2.7: bb_period 20→25 + bb_std 2.0→1.8 + stop_floor 0.03→0.05 (+11.8pp vs V2.6本地回测)
  V2.8: 黄金ETF品种级止损参数(stop_floor=0.03, atr_mult=2.0) +16.9pp
  V2.9: 均线趋势权重 15%→21% (+46pp, 8/8 walk-forward全胜)
  V2.10: 均线趋势权重 21%→24%，Walk-Forward验证 24% OOS最优(7/8年胜22%)
"""

import numpy as np
import pandas as pd
from datetime import timedelta
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
        open_tax=0, close_tax=0,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type='stock')

    # ---- ETF标的池 ----
    g.etf_pool = [
        '510300.XSHG',   # 沪深300
        '159915.XSHE',   # 创业板
        '512100.XSHG',   # 中证1000
        '159928.XSHE',   # 消费ETF
        '510880.XSHG',   # 红利ETF
        '513100.XSHG',   # 纳指ETF
        '513500.XSHG',   # 标普500ETF
        '159920.XSHE',   # 恒生ETF
        '513880.XSHG',   # 日经ETF
        '513050.XSHG',   # 中概互联ETF
        '518880.XSHG',   # 黄金ETF
        '159985.XSHE',   # 豆粕ETF
    ]


    # ---- 资金档位 ----
    g.capital_tiers = {
        'micro':  {'max_hold': 3, 'base_ratio': 0.70},
        'small':  {'max_hold': 3, 'base_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_ratio': 0.65},
        'large':  {'max_hold': 3, 'base_ratio': 0.65},
    }

    # ---- 策略参数 ----
    g.params = {
        'lookback': 120,
        'rebalance_weekdays': [1, 3],
        'min_hold_days': 5,
        'smooth_days': 3,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 25,
        'bb_std': 1.8,
        'kdj_n': 9,
        'kdj_m1': 3,
        'kdj_m2': 3,
        'momentum_period': 20,
        'vol_ma_period': 20,
        'atr_period': 14,
        'trailing_atr_mult': 2.5,
        'trailing_atr_mult_high_vol': 2.0,
        'high_vol_threshold': 0.30,
        'stop_floor': 0.05,
        'stop_cap': 0.15,
        'score_buy_threshold': 60,
        'switch_threshold': 8.0,
    }

    # ---- 因子权重 ----
    g.base_weights = {
        'rsi': 0.108,
        'macd': 0.161,
        'bollinger': 0.089,
        'momentum': 0.223,
        'volume': 0.071,
        'kdj': 0.108,
        'ma_trend': 0.24,
    }

    g.current_tier = None
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.holding_scores = {}
    g.portfolio_high = 0          # 组合历史最高净值（用于监控回撤）
    g.market_bearish = False

    # 品种级止损参数：黄金均值回复型，用更紧的止损
    g.code_stop_params = {
        '518880.XSHG': {'stop_floor': 0.03, 'trailing_atr_mult': 2.0},
    }

    run_daily(update_tier, time='09:30')
    run_daily(do_trading, time='09:35')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    return get_trade_days(end_date=context.current_dt.date(), count=2)[0]


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

    # 每日熊市检测
    _detect_bear_market(context)
    detect_choppy_market(context)


def _detect_bear_market(context):
    """每日熊市检测：沪深300 < MA60 且 MA60 下行，结果存 g.market_bearish"""
    today = context.current_dt.date()
    try:
        prev_date = get_trade_days(end_date=today, count=2)[0]
    except Exception:
        prev_date = today - timedelta(days=1)
    g.a_share_codes = {'510300.XSHG', '159915.XSHE', '512100.XSHG', '159928.XSHE', '510880.XSHG'}
    hs300_data = get_price('000300.XSHG', end_date=prev_date, count=65,
                           frequency='daily', fields=['close'])
    g.market_bearish = False
    if hs300_data is not None and len(hs300_data) >= 61:
        hs300_close = hs300_data['close'].iloc[-1]
        hs300_ma = hs300_data['close'].iloc[-60:].mean()
        hs300_ma_prev = hs300_data['close'].iloc[-61:-1].mean()
        g.market_bearish = hs300_close < hs300_ma and hs300_ma < hs300_ma_prev
        direction = '下行' if hs300_ma < hs300_ma_prev else '上行'
        status = '触发' if g.market_bearish else '未触发'
        log.info('[熊市检测] 000300.XSHG收盘%.2f MA60=%.2f(%s) %s' % (
            hs300_close, hs300_ma, direction, status))
    else:
        log.warning('[熊市检测] 数据不足，跳过')


def detect_choppy_market(context):
    """识别震荡市，仅用于日志监控，不参与交易决策。"""
    today = context.current_dt.date()
    try:
        prev_date = get_trade_days(end_date=today, count=2)[0]
    except Exception:
        prev_date = today - timedelta(days=1)

    df = get_price('000300.XSHG', end_date=prev_date, count=80,
                   frequency='daily', fields=['close'])
    if df is None or len(df) < 65:
        g.market_state = '未知'
        g.choppy_score = 0
        log.warning('[市场状态] 数据不足，跳过震荡监控')
        return

    close = df['close']
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    recent_close = close.iloc[-40:]
    recent_ma20 = ma20.iloc[-40:]
    cross_count = 0
    prev_side = None
    for price, ma_val in zip(recent_close, recent_ma20):
        if pd.isna(ma_val):
            continue
        side = 1 if price > ma_val else -1
        if prev_side is not None and side != prev_side:
            cross_count += 1
        prev_side = side

    ma60_now = ma60.iloc[-1]
    ma60_prev = ma60.iloc[-21]
    cur = close.iloc[-1]
    if pd.isna(ma60_now) or pd.isna(ma60_prev) or ma60_prev == 0:
        g.market_state = '未知'
        g.choppy_score = 0
        log.warning('[市场状态] MA60数据不足，跳过震荡监控')
        return

    ma60_slope = ma60_now / ma60_prev - 1
    dist_ma60 = cur / ma60_now - 1 if ma60_now != 0 else 0

    score = 0
    if cross_count >= 5:
        score += 1
    if abs(ma60_slope) < 0.02:
        score += 1
    if abs(dist_ma60) < 0.05:
        score += 1

    if score >= 3:
        state = '震荡市'
    elif score == 2:
        state = '轻微震荡'
    else:
        state = '趋势市'

    g.market_state = state
    g.choppy_score = score
    log.info('[市场状态] %s | 穿越MA20:%d次 MA60斜率:%.1f%% 距MA60:%.1f%% 震荡分:%d/3' % (
        state, cross_count, ma60_slope * 100, dist_ma60 * 100, score))


def get_tier_param(name):
    return g.capital_tiers[g.current_tier][name]


# ============================================================
#  技术指标计算
# ============================================================

def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def calc_bollinger(close, period, std_mult):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def calc_kdj(high, low, close, n, m1, m2):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
#  多因子综合评分
# ============================================================
def calc_multi_factor_score(code, end_date):
    p = g.params
    df = get_price(code, end_date=end_date, count=p['lookback'],
                   frequency='daily',
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   skip_paused=True, fq='pre')
    if df is None or len(df) < p['lookback'] - 10:
        return None

    C, H, L, V = df['close'], df['high'], df['low'], df['volume']
    sd = p['smooth_days']

    if C.iloc[-1] <= 0 or V.iloc[-5:].sum() == 0:
        return None

    # ---- 1. RSI ----
    rsi = calc_rsi(C, p['rsi_period'])
    rsi_vals = rsi.iloc[-sd:]
    if rsi_vals.isnull().any():
        return None
    rsi_val = rsi_vals.mean()

    if rsi_val < 30:
        rsi_score = 20
    elif rsi_val < 40:
        rsi_score = 35
    elif rsi_val < 50:
        rsi_score = 50
    elif rsi_val < 60:
        rsi_score = 65
    elif rsi_val < 70:
        rsi_score = 80
    elif rsi_val < 80:
        rsi_score = 75
    else:
        rsi_score = 70

    # ---- 2. MACD ----
    dif, dea, macd_hist = calc_macd(C, p['macd_fast'], p['macd_slow'], p['macd_signal'])
    dif_val = dif.iloc[-sd:].mean()
    dea_val = dea.iloc[-sd:].mean()
    hist_val = macd_hist.iloc[-sd:].mean()
    hist_prev = macd_hist.iloc[-sd - 3:-3].mean()

    macd_score = 50
    macd_score += 20 if dif_val > dea_val else -20
    if hist_val > 0 and hist_val > hist_prev:
        macd_score += 15
    elif hist_val > 0:
        macd_score += 5
    elif hist_val < 0 and hist_val > hist_prev:
        macd_score -= 5
    else:
        macd_score -= 15
    if dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] >= dea.iloc[-1]:
        macd_score += 10
    elif dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] <= dea.iloc[-1]:
        macd_score -= 10
    macd_score = max(0, min(100, macd_score))

    # ---- 3. 布林带 + squeeze ----
    bb_upper, bb_mid, bb_lower = calc_bollinger(C, p['bb_period'], p['bb_std'])
    bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    if bb_width <= 0 or pd.isna(bb_width):
        bb_score = 50
    else:
        bb_pos = (C.iloc[-1] - bb_lower.iloc[-1]) / bb_width
        if bb_pos < 0.2:
            bb_score = 20
        elif bb_pos < 0.4:
            bb_score = 40
        elif bb_pos < 0.6:
            bb_score = 60
        elif bb_pos < 0.8:
            bb_score = 80
        elif bb_pos < 0.95:
            bb_score = 75
        else:
            bb_score = 55
        # squeeze加分
        bb_width_20 = (bb_upper - bb_lower).iloc[-20:]
        if len(bb_width_20.dropna()) >= 10:
            avg_w = bb_width_20.mean()
            if avg_w > 0:
                ratio = bb_width / avg_w
                if ratio < 0.6:
                    bb_score += 5
                elif ratio < 0.8:
                    bb_score += 2
        bb_score = max(0, min(100, bb_score))

    # ---- 4. 动量ROC20（3日平滑）----
    mp = p['momentum_period']
    roc = (C.iloc[-1] / C.iloc[-mp] - 1
           + C.iloc[-2] / C.iloc[-mp - 1] - 1
           + C.iloc[-3] / C.iloc[-mp - 2] - 1) / 3.0

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
        mom_score = 40
    elif roc > -0.08:
        mom_score = 25
    else:
        mom_score = 10

    # ---- 5. 成交量 ----
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
            vol_score = 50

    # ---- 6. KDJ ----
    k, d, j = calc_kdj(H, L, C, p['kdj_n'], p['kdj_m1'], p['kdj_m2'])
    k_val, d_val, j_val = k.iloc[-sd:].mean(), d.iloc[-sd:].mean(), j.iloc[-sd:].mean()

    kdj_score = 50
    kdj_score += 15 if k_val > d_val else -15
    if j_val > 80:
        kdj_score += 10
    elif j_val > 50:
        kdj_score += 5
    elif j_val < 20:
        kdj_score -= 15
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] >= d.iloc[-1]:
        kdj_score += 10
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] <= d.iloc[-1]:
        kdj_score -= 10
    kdj_score = max(0, min(100, kdj_score))

    # ---- 7. 均线趋势 ----
    ma10, ma20, ma60 = C.iloc[-10:].mean(), C.iloc[-20:].mean(), C.iloc[-60:].mean()
    cur = C.iloc[-1]

    ma_score = 50
    ma_score += 10 if cur > ma10 else -10
    ma_score += 10 if cur > ma20 else -10
    ma_score += 10 if cur > ma60 else -10
    if ma10 > ma20 > ma60:
        ma_score += 15
    elif ma10 < ma20 < ma60:
        ma_score -= 15
    ma20_5d_ago = C.iloc[-25:-5].mean()
    ma_score += 5 if ma20 > ma20_5d_ago else -5
    ma_score = max(0, min(100, ma_score))

    # ---- 综合得分（固定权重）----
    scores = {
        'rsi': rsi_score, 'macd': macd_score, 'bollinger': bb_score,
        'momentum': mom_score, 'volume': vol_score, 'kdj': kdj_score,
        'ma_trend': ma_score,
    }
    final_score = 0.0
    for ks in g.base_weights:
        final_score += scores[ks] * g.base_weights[ks]

    # ---- ATR + 波动率 ----
    atr_val = calc_atr(H, L, C, p['atr_period']).iloc[-1]
    if pd.isna(atr_val):
        atr_val = cur * 0.02

    vol = C.pct_change().iloc[-20:].std() * np.sqrt(252)
    if pd.isna(vol) or vol <= 0:
        vol = 0.20

    return {
        'code': code, 'final_score': final_score,
        'roc': roc, 'close': cur,
        'atr': atr_val, 'volatility': vol, 'rsi': rsi_val,
        'ma20': ma20,
    }


# ============================================================
#  ATR跟踪止损（动态倍数）
# ============================================================
def calc_stop_price(code, highest, atr_val, atr_mult_override=None, profit_pct=None):
    p = g.params
    # 品种级参数覆盖（黄金等均值回复型品种用更紧止损）
    cp = g.code_stop_params.get(code, {})
    if atr_mult_override is not None:
        atr_mult = atr_mult_override
    else:
        vol_pct = atr_val / highest * np.sqrt(252.0 / p['atr_period'])
        base_mult = cp.get('trailing_atr_mult', p['trailing_atr_mult'])
        high_vol_mult = cp.get('trailing_atr_mult_high_vol', p['trailing_atr_mult_high_vol'])
        if vol_pct > p['high_vol_threshold']:
            atr_mult = high_vol_mult
        else:
            atr_mult = base_mult
        if profit_pct is not None and profit_pct > 0:
            if profit_pct > 0.15:
                atr_mult *= 0.6
            elif profit_pct > 0.05:
                atr_mult *= 0.8
    stop_floor = cp.get('stop_floor', p['stop_floor'])
    stop_cap = cp.get('stop_cap', p['stop_cap'])
    pct_stop = atr_mult * atr_val / highest
    pct_stop = max(stop_floor, min(stop_cap, pct_stop))
    return highest * (1 - pct_stop)


def is_overheated_for_buy(code, sig, price):
    """新买入防追高过滤：RSI过热时暂缓新买入，不影响持仓。"""
    ma20 = sig.get('ma20')
    rsi = sig.get('rsi')
    if rsi is None or pd.isna(rsi):
        return False

    dist_ma20 = price / ma20 - 1 if ma20 is not None and not pd.isna(ma20) and ma20 > 0 else 0
    if rsi > 75:
        log.info('[防追高] %s 价格距MA20 %.1f%% RSI %.1f，暂缓新买入' % (
            code, dist_ma20 * 100, rsi))
        return True
    return False


def build_rank_map(results):
    """Return deterministic full-pool rank after final_score sorting."""
    return {r['code']: i for i, r in enumerate(results)}


def sort_removable_positions(removable, rank_map):
    """Worst first: low score, then weaker full-pool rank, then code."""
    default_rank = len(rank_map)
    return sorted(
        removable,
        key=lambda x: (x[1], -rank_map.get(x[0], default_rank), x[0]))


def sort_buy_codes(codes, sig_map, rank_map):
    """Best first: high score, then stronger full-pool rank, then code."""
    default_rank = len(rank_map)
    return sorted(
        codes,
        key=lambda c: (
            -sig_map.get(c, {}).get('final_score', 0),
            rank_map.get(c, default_rank),
            c))


def check_stop_triggered(context, current_data, atr_mult_override=None):
    """检查哪些持仓触发了止损线（仅检测，不执行卖出）"""
    triggered = []
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0 or pos.avg_cost <= 0:
            continue
        if current_data[code].paused:
            continue
        cur_price = current_data[code].last_price
        if code in g.highest_since_buy and code in g.entry_atr:
            pnl = (cur_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            stop_price = calc_stop_price(code, g.highest_since_buy[code], g.entry_atr[code], atr_mult_override, pnl)
            if cur_price <= stop_price:
                triggered.append(code)
    return triggered


def execute_stop(code, context, current_data):
    """执行止损卖出"""
    pos = context.portfolio.positions[code]
    cur_price = current_data[code].last_price
    pnl = (cur_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
    dd = (g.highest_since_buy[code] - cur_price) / g.highest_since_buy[code]
    log.info('[止损] %s 最高%.3f 现%.3f 回撤%.1f%% 盈亏%.1f%%' % (
        code, g.highest_since_buy[code], cur_price, dd * 100, pnl * 100))
    order_target(code, 0)
    g.highest_since_buy.pop(code, None)
    g.entry_atr.pop(code, None)
    g.buy_date.pop(code, None)
    g.holding_scores.pop(code, None)


# ============================================================
#  核心交易逻辑
# ============================================================
def do_trading(context):
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()
    today = context.current_dt.date()

    # 1. 检测止损（仅检测，不执行）
    stop_triggered = check_stop_triggered(context, current_data)

    # 2. 是否轮动日
    if today.weekday() not in g.params['rebalance_weekdays'] and not stop_triggered:
        log.info('[非轮动日] 止损检查通过，无触发')
        return

    # 3. 打印资金状态
    is_rebalance = today.weekday() in g.params['rebalance_weekdays']
    trigger_reason = '轮动日' if is_rebalance else '止损触发%d只' % len(stop_triggered)
    log.info('[%s] 档位:%s 总值:%.0f 现金:%.0f' % (
        trigger_reason, g.current_tier, context.portfolio.total_value, context.portfolio.available_cash))

    # 4. 全池评分（T-1日数据）
    all_results = []
    for code in g.etf_pool:
        if current_data[code].paused:
            continue
        result = calc_multi_factor_score(code, prev_date)
        if result is not None:
            all_results.append(result)

    if not all_results:
        # 无评分结果时，触发的止损必须执行
        if stop_triggered:
            log.info('[评分为空] 无可评分标的，%d只止损强制执行' % len(stop_triggered))
        for code in stop_triggered:
            execute_stop(code, context, current_data)
        return

    current_holds = {}
    for code in context.portfolio.positions:
        if context.portfolio.positions[code].total_amount > 0:
            current_holds[code] = True

    all_results.sort(key=lambda x: x['final_score'], reverse=True)
    rank_map = build_rank_map(all_results)

    log.info('[TOP5]')
    for i, r in enumerate(all_results[:5]):
        log.info('  #%d %s 分:%.1f RSI:%.1f ROC:%.1f%%' % (
            i + 1, r['code'], r['final_score'],
            r['rsi'], r['roc'] * 100))

    # 当前持仓得分（含评分用的T-1收盘价，便于复盘）
    if context.portfolio.positions:
        score_close_map = {}
        for r in all_results:
            score_close_map[r['code']] = (r['final_score'], r['close'])
        held = [(c, score_close_map.get(c, (0, 0))) for c in context.portfolio.positions
                if context.portfolio.positions[c].total_amount > 0]
        if held:
            held.sort(key=lambda x: x[1][0], reverse=True)
            log.info('[持仓得分] %s' % ' | '.join(
                '%s:%.1f(T-1收盘:%.3f)' % (c, sc[0], sc[1]) for c, sc in held))

    # 5. 换仓逻辑
    threshold = g.params['score_buy_threshold']
    switch_th = g.params['switch_threshold']
    min_hold = g.params['min_hold_days']
    max_hold = get_tier_param('max_hold')

    candidates = [r for r in all_results if r['final_score'] > threshold]
    log.info('[候选] %d/%d只达标(>%d分)' % (len(candidates), len(all_results), threshold))

    # 更新持仓得分
    score_map = {}
    for r in all_results:
        score_map[r['code']] = r['final_score']
    for code in current_holds:
        if code in score_map:
            g.holding_scores[code] = score_map[code]

    # 决定目标持仓
    target_codes = set()
    protected_codes = set()

    for code in list(current_holds.keys()):
        if code in g.buy_date:
            days_held = len(get_trade_days(start_date=g.buy_date[code], end_date=today))
            if days_held <= min_hold:
                target_codes.add(code)
                protected_codes.add(code)
                continue

        if g.holding_scores.get(code, 0) > threshold - 5:
            target_codes.add(code)

    for r in candidates:
        if len(target_codes) >= max_hold:
            break
        if r['code'] not in target_codes:
            target_codes.add(r['code'])

    # 换仓门槛
    if len(target_codes) >= max_hold:
        removable = [(c, g.holding_scores.get(c, 0))
                     for c in target_codes
                     if c in current_holds and c not in protected_codes]
        removable = sort_removable_positions(removable, rank_map)

        for r in candidates:
            if r['code'] in target_codes or not removable:
                continue
            worst_code, worst_score = removable[0]
            if r['final_score'] > worst_score + switch_th:
                target_codes.discard(worst_code)
                target_codes.add(r['code'])
                removable.pop(0)
                log.info('[换仓] %s(%.1f) 替换 %s(%.1f) 差%.1f分' % (
                    r['code'], r['final_score'], worst_code, worst_score,
                    r['final_score'] - worst_score))

    # 6. 执行止损（不在target→直接止损；在target→MA10趋势未破→豁免）
    force_stopped = set()
    for code in stop_triggered:
        if code in target_codes:
            cur_price = current_data[code].last_price
            trend_broken = False
            try:
                ma_df = get_price(code, end_date=prev_date, count=15,
                                 frequency='daily', fields=['close'],
                                 skip_paused=True, fq='pre')
                if ma_df is not None and len(ma_df) >= 11:
                    ma10 = ma_df['close'].iloc[-10:].mean()
                    ma10_prev = ma_df['close'].iloc[-11:-1].mean()
                    if cur_price < ma10 and ma10 < ma10_prev:
                        trend_broken = True
            except Exception:
                pass
            if trend_broken:
                log.info('[趋势止损] %s 得分%.1f 价格%.3f<MA10=%.3f MA10下行 不豁免' % (
                    code, g.holding_scores.get(code, 0), cur_price, ma10))
                execute_stop(code, context, current_data)
                force_stopped.add(code)
            else:
                log.info('[止损豁免] %s 得分%.1f 保留持仓' % (
                    code, g.holding_scores.get(code, 0)))
        else:
            execute_stop(code, context, current_data)

    # 7. 轮动卖出（停牌标的跳过）
    for code in list(current_holds.keys()):
        if code not in target_codes and code not in stop_triggered:
            if current_data[code].paused:
                log.info('[跳过卖出] %s 停牌中，保留持仓' % code)
                continue
            pos = context.portfolio.positions[code]
            pnl = (current_data[code].last_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            log.info('[轮动卖出] %s 盈亏%.1f%% 得分:%.1f' % (
                code, pnl * 100, g.holding_scores.get(code, 0)))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            g.buy_date.pop(code, None)
            g.holding_scores.pop(code, None)

    sig_map = {}
    for r in all_results:
        sig_map[r['code']] = r

    # 8. 买入（目标池优先；若防追高跳过，则用候选池后备标的补位）
    primary_buy = [c for c in target_codes if c not in current_holds and c not in force_stopped]
    primary_buy = sort_buy_codes(primary_buy, sig_map, rank_map)

    backup_buy = []
    seen = set(primary_buy)
    for r in candidates:
        code = r['code']
        if code in seen or code in target_codes or code in current_holds or code in force_stopped:
            continue
        backup_buy.append(code)
        seen.add(code)

    to_buy = primary_buy + backup_buy
    if not to_buy:
        log.info('[无换仓] 持仓与目标一致')
        return

    # 聚宽回测中available_cash即时更新，无需加sold_proceeds
    available = context.portfolio.available_cash
    # 用实时持仓数算槽位，避免卖出未结算导致的超买(持仓4/3的bug)
    actual_hold_count = len([c for c in context.portfolio.positions
                             if context.portfolio.positions[c].total_amount > 0])
    slots = max_hold - actual_hold_count
    if slots <= 0 or available < 500:
        log.info('[跳过买入] 无空仓位(slots=%d)或资金不足(%.0f)' % (slots, available))
        return

    base_ratio = get_tier_param('base_ratio')

    for code in to_buy:
        if slots <= 0 or available < 500:
            break
        if code not in sig_map:
            continue

        sig = sig_map[code]
        price = current_data[code].last_price
        if is_overheated_for_buy(code, sig, price):
            continue

        alloc = available / slots * base_ratio
        actual_vol = max(sig['volatility'], 0.05)
        alloc *= max(0.4, min(1.5, 0.15 / actual_vol))
        alloc = min(alloc, available * 0.95)
        if g.market_bearish and code in g.a_share_codes:
            alloc *= 0.5

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            if available >= price * 100 * 1.003:
                shares = 100
            else:
                log.info('[资金不足] %s 需%.0f元买100股，可用%.0f' % (code, price * 100, available))
                continue

        if code in backup_buy:
            log.info('[后备补位] %s 来自候选池后备队列，补足空余仓位' % code)
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
#  盘后：更新最高价/ATR + 记录
# ============================================================
def after_close(context):
    today = context.current_dt.date()
    positions = context.portfolio.positions
    hold = {}

    current_data = get_current_data()
    for code in list(positions.keys()):
        pos = positions[code]
        if pos.total_amount <= 0:
            continue
        hold[code] = pos
        cur = current_data[code].last_price

        # 更新最高价
        if code in g.highest_since_buy:
            if cur > g.highest_since_buy[code]:
                g.highest_since_buy[code] = cur
        else:
            g.highest_since_buy[code] = max(cur, pos.avg_cost)

        # 动态更新ATR
        if code in g.entry_atr:
            atr_df = get_price(code, end_date=today,
                               count=g.params['atr_period'] + 5,
                               frequency='daily',
                               fields=['close', 'high', 'low'],
                               skip_paused=True, fq='pre')
            if atr_df is not None and len(atr_df) >= g.params['atr_period']:
                new_atr = calc_atr(atr_df['high'], atr_df['low'],
                                   atr_df['close'], g.params['atr_period']).iloc[-1]
                if not pd.isna(new_atr) and new_atr > 0:
                    g.entry_atr[code] = new_atr

    # 组合回撤监控
    total_value = context.portfolio.total_value
    if total_value > g.portfolio_high:
        g.portfolio_high = total_value
    portfolio_dd = (g.portfolio_high - total_value) / g.portfolio_high * 100 if g.portfolio_high > 0 else 0

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d 组合回撤:%.1f%%' % (
        g.current_tier, total_value,
        context.portfolio.available_cash, len(hold), get_tier_param('max_hold'),
        portfolio_dd))

    for code, pos in hold.items():
        cur_price = pos.price
        pnl = (cur_price - pos.avg_cost) / pos.avg_cost * 100 if pos.avg_cost > 0 else 0
        highest = g.highest_since_buy.get(code, cur_price)
        score = g.holding_scores.get(code, 0)
        log.info('  %s 成本:%.3f 现:%.3f 高:%.3f 盈亏:%.1f%% 分:%.1f' % (
            code, pos.avg_cost, cur_price, highest, pnl, score))
    log.info('=' * 60)
