# -*- coding: utf-8 -*-
"""
多因子ETF量化策略 V2.5
============================
基于7个经典技术指标（RSI/MACD/布林带/动量/成交量/KDJ/均线趋势）综合评分，
固定因子权重，ATR跟踪止损+止损豁免，多市场多资产ETF轮动。

核心机制：
  - 7因子离散分档评分 + 3日平滑 + 固定权重（稳定排名，减少噪音换仓）
  - 周二+周四固定轮动（无起始日依赖）
  - 换仓门槛8分（新标的必须高出持仓最低分8分才替换）
  - 最低持仓期5天（防止买入即卖）
  - ATR跟踪止损 + 止损豁免（得分仍在目标中且回撤<10%→不卖；回撤≥10%→强制止损）
  - 每日收盘后更新最高价+ATR（次日止损更准确）
  - 波动率反比仓位（低波动多买，高波动少买）
  - 买入按得分排序（最强标的优先获得资金）
  - 候选不足时持有现金（不强制兜底，让市场决定持仓数）
  - 组合回撤监控（盘后日志显示当前回撤幅度）

ETF池（5A股 + 5跨市场 + 2跨资产 = 12只）：
  A股: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 510880红利
  跨市场: 513100纳指, 513500标普500, 159920恒生, 513880日经, 513050中概互联
  跨资产: 518880黄金, 159985豆粕

因子权重（固定，未优化）：
  动量ROC20=0.25, MACD=0.18, 均线趋势=0.15, RSI=0.12, KDJ=0.12, 布林带=0.10, 成交量=0.08

回测业绩（万三+最低5元佣金）：
  2015-2026（11年）：+251.5%，年化~12%，最大回撤~15.8%，夏普0.63
  2010-2014（样本外）：+37%，年化6.4%，弱市+标的不全仍正收益
  2008金融危机：仅-2.7%（同期沪深300 -65%）

版本历史：
  V1.0: 每日轮动无门槛，-91.3%（手续费吞噬本金）
  V2.0: 加门槛+持仓期+离散分档，+234%
  V2.3: 去ADX自适应+7pp，去国债兜底+18pp，固定权重
  V2.4: 止损豁免+35pp（触发止损但得分仍高则不卖）
  V2.5: 止损豁免+回撤上限（得分高可豁免，但回撤≥10%时强制止损，防范得分滞后于价格）
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
        'medium': {'max_hold': 3, 'base_ratio': 0.60},
        'large':  {'max_hold': 4, 'base_ratio': 0.55},
    }

    # ---- 策略参数（全部学术默认值）----
    g.params = {
        'lookback': 120,
        'rebalance_weekdays': [1, 3],
        'min_hold_days': 5,
        'smooth_days': 3,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2.0,
        'kdj_n': 9,
        'kdj_m1': 3,
        'kdj_m2': 3,
        'momentum_period': 20,
        'vol_ma_period': 20,
        'atr_period': 14,
        'trailing_atr_mult': 2.5,
        'trailing_atr_mult_high_vol': 2.0,
        'high_vol_threshold': 0.30,
        'stop_floor': 0.03,
        'stop_cap': 0.15,
        'score_buy_threshold': 60,
        'switch_threshold': 8.0,
        'stop_exempt_max_dd': 0.10,  # 止损豁免最大回撤：超过则强制止损
    }

    # ---- 因子权重 ----
    g.base_weights = {
        'rsi': 0.12,
        'macd': 0.18,
        'bollinger': 0.10,
        'momentum': 0.25,
        'volume': 0.08,
        'kdj': 0.12,
        'ma_trend': 0.15,
    }

    g.current_tier = None
    g.highest_since_buy = {}
    g.entry_atr = {}
    g.buy_date = {}
    g.holding_scores = {}
    g.portfolio_high = 0          # 组合历史最高净值（用于监控回撤）

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
        rsi_score = 55

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
    }


# ============================================================
#  ATR跟踪止损（动态倍数）
# ============================================================
def calc_stop_price(highest, atr_val):
    p = g.params
    vol_pct = atr_val / highest * np.sqrt(252.0 / p['atr_period'])
    if vol_pct > p['high_vol_threshold']:
        atr_mult = p['trailing_atr_mult_high_vol']
    else:
        atr_mult = p['trailing_atr_mult']
    pct_stop = atr_mult * atr_val / highest
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return highest * (1 - pct_stop)


def check_stop_triggered(context, current_data):
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
            stop_price = calc_stop_price(g.highest_since_buy[code], g.entry_atr[code])
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

    all_results.sort(key=lambda x: x['final_score'], reverse=True)

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
        removable.sort(key=lambda x: x[1])

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

    # 6. 执行止损（不在目标中→直接止损；在目标中→回撤超限才止损）
    force_stopped = set()
    max_exempt_dd = g.params['stop_exempt_max_dd']
    for code in stop_triggered:
        if code in target_codes:
            highest = g.highest_since_buy.get(code, 1)
            cur_price = current_data[code].last_price
            dd = (highest - cur_price) / highest if highest > 0 else 0
            if dd < max_exempt_dd:
                log.info('[止损豁免] %s 得分%.1f 回撤%.1f%%<%.0f%% 保留持仓' % (
                    code, g.holding_scores.get(code, 0), dd * 100, max_exempt_dd * 100))
            else:
                log.info('[止损豁免超限] %s 得分%.1f 回撤%.1f%%>=%.0f%% 强制止损' % (
                    code, g.holding_scores.get(code, 0), dd * 100, max_exempt_dd * 100))
                execute_stop(code, context, current_data)
                force_stopped.add(code)
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

    # 8. 买入（按得分排序）
    to_buy = [c for c in target_codes if c not in current_holds and c not in force_stopped]
    if not to_buy:
        log.info('[无换仓] 持仓与目标一致')
        return

    sig_map = {}
    for r in all_results:
        sig_map[r['code']] = r
    to_buy.sort(key=lambda c: sig_map.get(c, {}).get('final_score', 0), reverse=True)

    # 聚宽回测中available_cash即时更新，无需加sold_proceeds
    available = context.portfolio.available_cash
    slots = max_hold - len(set(current_holds.keys()) & target_codes)
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

        alloc = available / slots * base_ratio
        actual_vol = max(sig['volatility'], 0.05)
        alloc *= max(0.4, min(1.5, 0.15 / actual_vol))
        alloc = min(alloc, available * 0.95)

        shares = int(alloc / price / 100) * 100
        if shares < 100:
            if available >= price * 100 * 1.003:
                shares = 100
            else:
                log.info('[资金不足] %s 需%.0f元买100股，可用%.0f' % (code, price * 100, available))
                continue

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
        pnl = (pos.price - pos.avg_cost) / pos.avg_cost * 100 if pos.avg_cost > 0 else 0
        highest = g.highest_since_buy.get(code, pos.price)
        score = g.holding_scores.get(code, 0)
        log.info('  %s 成本:%.3f 现:%.3f 高:%.3f 盈亏:%.1f%% 分:%.1f' % (
            code, pos.avg_cost, pos.price, highest, pnl, score))
    log.info('=' * 60)
