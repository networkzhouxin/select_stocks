# -*- coding: utf-8 -*-
"""
多因子ETF量化策略 V2.4 - PTrade版（回测+实盘兼容）
=============================================
从聚宽V2.4逐行移植，所有交易逻辑、参数、因子评分100%不变。
根据PTrade官方API文档适配，同时兼容回测和实盘两种模式。

核心机制（与聚宽版一致）：
  - 7因子离散分档评分 + 3日平滑 + 固定权重
  - 周二+周四固定轮动
  - 最低持仓期5天
  - ATR跟踪止损 + 止损豁免（触发止损但得分仍在目标中→不卖）
  - 每日收盘后更新最高价+ATR
  - 波动率反比仓位 + 买入按得分排序
  - 候选不足时持有现金

实盘健壮性（参考PTrade V15.9.1）：
  - 所有order/order_target传limit_price
  - 止损/轮动卖出用跌停价作limit_price
  - 实盘用snapshot的trade_status检测停牌
  - on_trade_response成交回调确认买入
  - on_order_response委托回调检测失败
  - 盘前清理缓存/pending_orders
  - 持仓entry_atr恢复机制
  - sold_today防重复卖出

适配要点：
  1. 代码格式：.XSHG/.XSHE → .SS/.SZ
  2. Portfolio：portfolio_value/cash
  3. Position：amount/cost_basis/last_sale_price
  4. 行情：data[code].price / get_snapshot
  5. 停牌：snapshot trade_status / get_stock_status
  6. run_daily：run_daily(context, func, time)
  7. 回测：日频run_daily固定15:00，用handle_data驱动
  8. 限价精度：ETF限价单3位小数

ETF池（5A股 + 5跨市场 + 2跨资产 = 12只）：
  A股: 510300沪深300, 159915创业板, 512100中证1000, 159928消费, 510880红利
  跨市场: 513100纳指, 513500标普500, 159920恒生, 513880日经, 513050中概互联
  跨资产: 518880黄金, 159985豆粕

因子权重（固定，未优化）：
  动量ROC20=0.25, MACD=0.18, 均线趋势=0.15, RSI=0.12, KDJ=0.12, 布林带=0.10, 成交量=0.08

聚宽回测业绩（万三+最低5元佣金）：
  2015-2026（11年）：+251.5%，年化~12%，最大回撤~15.8%，夏普0.63
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.SS')

    try:
        set_commission(commission_ratio=0.0003, min_commission=5.0, type='ETF')
        set_slippage(slippage=0.001)
    except Exception:
        pass

    # ---- ETF标的池（PTrade .SS/.SZ格式，12只）----
    g.etf_pool = [
        '510300.SS',    # 沪深300
        '159915.SZ',    # 创业板
        '512100.SS',    # 中证1000
        '159928.SZ',    # 消费ETF
        '510880.SS',    # 红利ETF
        '513100.SS',    # 纳指ETF
        '513500.SS',    # 标普500ETF
        '159920.SZ',    # 恒生ETF
        '513880.SS',    # 日经ETF
        '513050.SS',    # 中概互联ETF
        '518880.SS',    # 黄金ETF
        '159985.SZ',    # 豆粕ETF
    ]

    # ---- 资金档位 ----
    g.capital_tiers = {
        'micro':  {'max_hold': 3, 'base_ratio': 0.70},
        'small':  {'max_hold': 3, 'base_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_ratio': 0.60},
        'large':  {'max_hold': 4, 'base_ratio': 0.55},
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
    g.portfolio_high = 0
    g.sold_today = {}
    g.__last_snapshot = {}
    g.__pending_orders = {}

    try:
        set_universe(g.etf_pool)
    except Exception:
        pass

    try:
        g.__is_live = is_trade()
    except Exception:
        g.__is_live = False

    if g.__is_live:
        run_daily(context, _update_tier_wrapper, time='09:30')
        run_daily(context, _do_trading_wrapper, time='09:35')
        run_daily(context, _after_close_wrapper, time='15:30')


def handle_data(context, data):
    g.__data = data
    if g.__is_live:
        return
    _update_tier(context)
    _do_trading(context)
    _update_highest_and_atr(context)


def before_trading_start(context, data):
    g.__data = data
    g.sold_today = {}
    g.__last_snapshot = {}
    log.info('[盘前] 清理缓存完毕')

    if g.__pending_orders:
        for code in list(g.__pending_orders.keys()):
            log.warning('[订单超时] %s 前日买入未确认，清理' % code)
        g.__pending_orders = {}

    if g.__is_live:
        today = context.blotter.current_dt.date()
        prev_date = _get_prev_trade_date(context)
        positions = _positions(context)
        hold_codes = [c for c, p in positions.items() if _pos_amount(p) > 0]
        log.info('[盘前恢复] 今日%s prev_date=%s 持仓%d只(%s)' % (
            today, prev_date, len(hold_codes), ', '.join(hold_codes) if hold_codes else '空仓'))
        missing_atr = [c for c in hold_codes if c not in g.entry_atr]
        missing_high = [c for c in hold_codes if c not in g.highest_since_buy]
        missing_buy = [c for c in hold_codes if g.buy_date.get(c) is None]
        if missing_atr or missing_high or missing_buy:
            log.info('[盘前恢复] 缺失: entry_atr=%s highest=%s buy_date=%s' % (
                missing_atr or '无', missing_high or '无', missing_buy or '无'))
        else:
            log.info('[盘前恢复] 数据完整，无需恢复')
            return
        # 一次性查询交割单，所有恢复函数共用
        deliver_records = _fetch_deliver_records(prev_date)
        _recover_missing_atr(context, deliver_records)
        _recover_missing_highest(context, deliver_records)
        _recover_missing_buy_date(context, deliver_records)


def after_trading_end(context, data):
    if not g.__is_live:
        _after_close(context)
    g.sold_today = {}


# ============================================================
#  run_daily包装函数
# ============================================================
def _update_tier_wrapper(context):
    _update_tier(context)

def _do_trading_wrapper(context):
    _do_trading(context)

def _after_close_wrapper(context):
    _update_highest_and_atr(context)
    _after_close(context)
    g.sold_today = {}


# ============================================================
#  Portfolio/Position兼容层
# ============================================================
def _total_value(context):
    return context.portfolio.portfolio_value

def _available_cash(context):
    return context.portfolio.cash

def _positions(context):
    return context.portfolio.positions

def _pos_amount(pos):
    return pos.amount

def _pos_cost(pos):
    return pos.cost_basis

def _pos_price(pos):
    return pos.last_sale_price


# ============================================================
#  行情数据获取
# ============================================================
def _get_current_price(code):
    if g.__is_live:
        try:
            snap = get_snapshot(code)
            if snap and code in snap:
                snap = snap[code]
            if snap and snap.get('last_px', 0) > 0:
                g.__last_snapshot[code] = snap
                return snap['last_px']
        except Exception:
            pass

    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].price
        except Exception:
            pass

    try:
        df = get_history(1, '1d', 'close', [code], include=True)
        return float(df[code].iloc[-1])
    except Exception:
        pass
    return None


def _is_paused(code):
    if g.__is_live and code in g.__last_snapshot:
        ts = g.__last_snapshot[code].get('trade_status', '')
        if ts in ('HALT', 'SUSP', 'STOPT', 'DELISTED'):
            return True
        if ts in ('TRADE', 'OCALL', 'BREAK', 'ENDTR', 'POSTR'):
            return False

    try:
        status = get_stock_status([code], 'HALT')
        return status.get(code, False)
    except Exception:
        pass

    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].is_open == 0
        except Exception:
            pass
    return False


def _get_prev_trade_date(context):
    today = context.blotter.current_dt.date()
    try:
        result = get_trade_days(end_date=today, count=2)[0]
        if isinstance(result, str):
            return datetime.strptime(result, '%Y-%m-%d').date()
        return result
    except Exception:
        pass
    try:
        all_days = get_all_trades_days(date=today.strftime('%Y%m%d'))
        past = [d for d in all_days if d < today]
        if past:
            return past[-1]
    except Exception:
        pass
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def _get_price_data(code, end_date, count):
    if hasattr(end_date, 'strftime'):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = str(end_date)

    try:
        df = get_price(code, end_date=end_date_str, count=count,
                       frequency='1d',
                       fields=['open', 'close', 'high', 'low', 'volume'],
                       fq='pre')
        if df is not None and len(df) > 0:
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]
            return df
    except Exception:
        pass

    try:
        df = pd.DataFrame({
            'open': get_history(count, '1d', 'open', [code], fq='pre')[code],
            'close': get_history(count, '1d', 'close', [code], fq='pre')[code],
            'high': get_history(count, '1d', 'high', [code], fq='pre')[code],
            'low': get_history(count, '1d', 'low', [code], fq='pre')[code],
            'volume': get_history(count, '1d', 'volume', [code], fq='pre')[code],
        })
        try:
            df = df[df.index <= pd.Timestamp(end_date_str)]
        except Exception:
            pass
        df = df[df['volume'] > 0]
        return df
    except Exception as e:
        log.error('获取行情失败 %s: %s' % (code, str(e)))
        return None


def _get_sell_limit_price(code, cur_price):
    """卖出限价：实盘用跌停价确保成交，回测用当前价"""
    sell_lmt = round(cur_price, 3)
    if g.__is_live and code in g.__last_snapshot:
        try:
            down_px = float(g.__last_snapshot[code].get('down_px', 0))
            if down_px > 0:
                sell_lmt = round(down_px, 3)
        except (ValueError, TypeError):
            pass
    return sell_lmt


# ============================================================
#  动态资金档位
# ============================================================
def _update_tier(context):
    total = _total_value(context)
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


def _get_tier_param(name):
    return g.capital_tiers[g.current_tier][name]


# ============================================================
#  技术指标计算（与聚宽版一致）
# ============================================================
def _calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _calc_macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def _calc_bollinger(close, period, std_mult):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def _calc_kdj(high, low, close, n, m1, m2):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def _calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
#  多因子综合评分（与聚宽版100%一致）
# ============================================================
def _calc_multi_factor_score(code, end_date):
    p = g.params
    df = _get_price_data(code, end_date, count=p['lookback'])
    if df is None or len(df) < p['lookback'] - 10:
        return None

    C, H, L, V = df['close'], df['high'], df['low'], df['volume']
    sd = p['smooth_days']

    if C.iloc[-1] <= 0 or V.iloc[-5:].sum() == 0:
        return None

    # 1. RSI
    rsi = _calc_rsi(C, p['rsi_period'])
    rsi_vals = rsi.iloc[-sd:]
    if rsi_vals.isnull().any():
        return None
    rsi_val = rsi_vals.mean()

    if rsi_val < 30: rsi_score = 20
    elif rsi_val < 40: rsi_score = 35
    elif rsi_val < 50: rsi_score = 50
    elif rsi_val < 60: rsi_score = 65
    elif rsi_val < 70: rsi_score = 80
    elif rsi_val < 80: rsi_score = 75
    else: rsi_score = 55

    # 2. MACD
    dif, dea, macd_hist = _calc_macd(C, p['macd_fast'], p['macd_slow'], p['macd_signal'])
    dif_val = dif.iloc[-sd:].mean()
    dea_val = dea.iloc[-sd:].mean()
    hist_val = macd_hist.iloc[-sd:].mean()
    hist_prev = macd_hist.iloc[-sd - 3:-3].mean()

    macd_score = 50
    macd_score += 20 if dif_val > dea_val else -20
    if hist_val > 0 and hist_val > hist_prev: macd_score += 15
    elif hist_val > 0: macd_score += 5
    elif hist_val < 0 and hist_val > hist_prev: macd_score -= 5
    else: macd_score -= 15
    if dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] >= dea.iloc[-1]: macd_score += 10
    elif dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] <= dea.iloc[-1]: macd_score -= 10
    macd_score = max(0, min(100, macd_score))

    # 3. 布林带 + squeeze
    bb_upper, bb_mid, bb_lower = _calc_bollinger(C, p['bb_period'], p['bb_std'])
    bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    if bb_width <= 0 or pd.isna(bb_width):
        bb_score = 50
    else:
        bb_pos = (C.iloc[-1] - bb_lower.iloc[-1]) / bb_width
        if bb_pos < 0.2: bb_score = 20
        elif bb_pos < 0.4: bb_score = 40
        elif bb_pos < 0.6: bb_score = 60
        elif bb_pos < 0.8: bb_score = 80
        elif bb_pos < 0.95: bb_score = 75
        else: bb_score = 55
        bb_width_20 = (bb_upper - bb_lower).iloc[-20:]
        if len(bb_width_20.dropna()) >= 10:
            avg_w = bb_width_20.mean()
            if avg_w > 0:
                ratio = bb_width / avg_w
                if ratio < 0.6: bb_score += 5
                elif ratio < 0.8: bb_score += 2
        bb_score = max(0, min(100, bb_score))

    # 4. 动量ROC20
    mp = p['momentum_period']
    roc = (C.iloc[-1] / C.iloc[-mp] - 1
           + C.iloc[-2] / C.iloc[-mp - 1] - 1
           + C.iloc[-3] / C.iloc[-mp - 2] - 1) / 3.0

    if roc > 0.15: mom_score = 95
    elif roc > 0.10: mom_score = 85
    elif roc > 0.05: mom_score = 75
    elif roc > 0.02: mom_score = 65
    elif roc > 0: mom_score = 55
    elif roc > -0.03: mom_score = 40
    elif roc > -0.08: mom_score = 25
    else: mom_score = 10

    # 5. 成交量
    vol_ma = V.iloc[-p['vol_ma_period']:].mean()
    vol_recent = V.iloc[-3:].mean()
    if vol_ma <= 0 or pd.isna(vol_ma):
        vol_score = 50
    else:
        vol_ratio = vol_recent / vol_ma
        price_up = C.iloc[-1] > C.iloc[-3]
        if price_up and vol_ratio > 1.3: vol_score = 85
        elif price_up and vol_ratio > 0.8: vol_score = 65
        elif price_up: vol_score = 50
        elif not price_up and vol_ratio > 1.3: vol_score = 25
        elif not price_up and vol_ratio > 0.8: vol_score = 40
        else: vol_score = 50

    # 6. KDJ
    k, d, j = _calc_kdj(H, L, C, p['kdj_n'], p['kdj_m1'], p['kdj_m2'])
    k_val, d_val, j_val = k.iloc[-sd:].mean(), d.iloc[-sd:].mean(), j.iloc[-sd:].mean()
    kdj_score = 50
    kdj_score += 15 if k_val > d_val else -15
    if j_val > 80: kdj_score += 10
    elif j_val > 50: kdj_score += 5
    elif j_val < 20: kdj_score -= 15
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] >= d.iloc[-1]: kdj_score += 10
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] <= d.iloc[-1]: kdj_score -= 10
    kdj_score = max(0, min(100, kdj_score))

    # 7. 均线趋势
    ma10, ma20, ma60 = C.iloc[-10:].mean(), C.iloc[-20:].mean(), C.iloc[-60:].mean()
    cur = C.iloc[-1]
    ma_score = 50
    ma_score += 10 if cur > ma10 else -10
    ma_score += 10 if cur > ma20 else -10
    ma_score += 10 if cur > ma60 else -10
    if ma10 > ma20 > ma60: ma_score += 15
    elif ma10 < ma20 < ma60: ma_score -= 15
    ma20_5d_ago = C.iloc[-25:-5].mean()
    ma_score += 5 if ma20 > ma20_5d_ago else -5
    ma_score = max(0, min(100, ma_score))

    # 综合得分
    scores = {
        'rsi': rsi_score, 'macd': macd_score, 'bollinger': bb_score,
        'momentum': mom_score, 'volume': vol_score, 'kdj': kdj_score,
        'ma_trend': ma_score,
    }
    final_score = 0.0
    for ks in g.base_weights:
        final_score += scores[ks] * g.base_weights[ks]

    atr_val = _calc_atr(H, L, C, p['atr_period']).iloc[-1]
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
#  ATR跟踪止损
# ============================================================
def _calc_stop_price(highest, atr_val):
    p = g.params
    vol_pct = atr_val / highest * np.sqrt(252.0 / p['atr_period'])
    if vol_pct > p['high_vol_threshold']:
        atr_mult = p['trailing_atr_mult_high_vol']
    else:
        atr_mult = p['trailing_atr_mult']
    pct_stop = atr_mult * atr_val / highest
    pct_stop = max(p['stop_floor'], min(p['stop_cap'], pct_stop))
    return highest * (1 - pct_stop)


def _check_stop_triggered(context):
    """检查哪些持仓触发止损线（仅检测，不执行）"""
    triggered = []
    positions = _positions(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0 or _pos_cost(pos) <= 0:
            continue
        if g.__is_live and g.sold_today.get(code, False):
            continue
        if _is_paused(code):
            continue
        cur_price = _get_current_price(code)
        if cur_price is None:
            continue
        if code in g.highest_since_buy and code in g.entry_atr:
            stop_price = _calc_stop_price(g.highest_since_buy[code], g.entry_atr[code])
            if cur_price <= stop_price:
                triggered.append(code)
    return triggered


def _execute_stop(code, context):
    """执行止损卖出"""
    pos = _positions(context)[code]
    cur_price = _get_current_price(code)
    if cur_price is None:
        cur_price = _pos_cost(pos)
    cost = _pos_cost(pos)
    pnl = (cur_price - cost) / cost if cost > 0 else 0
    dd = (g.highest_since_buy[code] - cur_price) / g.highest_since_buy[code]
    log.info('[止损] %s 最高%.3f 现%.3f 回撤%.1f%% 盈亏%.1f%%' % (
        code, g.highest_since_buy[code], cur_price, dd * 100, pnl * 100))
    sell_lmt = _get_sell_limit_price(code, cur_price)
    order_target(code, 0, limit_price=sell_lmt)
    g.sold_today[code] = True
    g.highest_since_buy.pop(code, None)
    g.entry_atr.pop(code, None)
    g.buy_date.pop(code, None)
    g.holding_scores.pop(code, None)


# ============================================================
#  核心交易逻辑（止损豁免 + 换仓门槛 + 最低持仓期）
# ============================================================
def _do_trading(context):
    prev_date = _get_prev_trade_date(context)
    today = context.blotter.current_dt.date()

    # 1. 检测止损（仅检测，不执行）
    stop_triggered = _check_stop_triggered(context)

    # 2. 是否轮动日
    if today.weekday() not in g.params['rebalance_weekdays'] and not stop_triggered:
        log.info('[非轮动日] 止损检查通过，无触发')
        return

    # 3. 打印资金状态
    is_rebalance = today.weekday() in g.params['rebalance_weekdays']
    trigger_reason = '轮动日' if is_rebalance else '止损触发%d只' % len(stop_triggered)
    log.info('[%s] 档位:%s 总值:%.0f 现金:%.0f' % (
        trigger_reason, g.current_tier, _total_value(context), _available_cash(context)))

    # 4. 全池评分
    all_results = []
    for code in g.etf_pool:
        if _is_paused(code):
            continue
        result = _calc_multi_factor_score(code, prev_date)
        if result is not None:
            all_results.append(result)

    if not all_results:
        if stop_triggered:
            log.info('[评分为空] 无可评分标的，%d只止损强制执行' % len(stop_triggered))
        for code in stop_triggered:
            _execute_stop(code, context)
        return

    all_results.sort(key=lambda x: x['final_score'], reverse=True)

    log.info('[TOP5]')
    for i, r in enumerate(all_results[:5]):
        log.info('  #%d %s 分:%.1f RSI:%.1f ROC:%.1f%%' % (
            i + 1, r['code'], r['final_score'], r['rsi'], r['roc'] * 100))

    # 持仓得分（含评分用的T-1收盘价）
    positions = _positions(context)
    if positions:
        score_close_map = {}
        for r in all_results:
            score_close_map[r['code']] = (r['final_score'], r['close'])
        held = [(c, score_close_map.get(c, (0, 0))) for c in positions
                if _pos_amount(positions[c]) > 0]
        if held:
            held.sort(key=lambda x: x[1][0], reverse=True)
            log.info('[持仓得分] %s' % ' | '.join(
                '%s:%.1f(T-1收盘:%.3f)' % (c, sc[0], sc[1]) for c, sc in held))

    # 5. 换仓逻辑
    threshold = g.params['score_buy_threshold']
    min_hold = g.params['min_hold_days']
    max_hold = _get_tier_param('max_hold')

    candidates = [r for r in all_results if r['final_score'] > threshold]
    log.info('[候选] %d/%d只达标(>%d分)' % (len(candidates), len(all_results), threshold))

    current_holds = {}
    for code in positions:
        if _pos_amount(positions[code]) > 0:
            current_holds[code] = True

    score_map = {}
    for r in all_results:
        score_map[r['code']] = r['final_score']
    for code in current_holds:
        if code in score_map:
            g.holding_scores[code] = score_map[code]

    target_codes = set()
    protected_codes = set()

    for code in list(current_holds.keys()):
        if code in g.buy_date:
            try:
                days_held = len(get_trade_days(start_date=g.buy_date[code], end_date=today))
            except Exception:
                days_held = (today - g.buy_date[code]).days
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

    # 换仓：候选分高于持仓最低分即替换
    if len(target_codes) >= max_hold:
        removable = [(c, g.holding_scores.get(c, 0))
                     for c in target_codes
                     if c in current_holds and c not in protected_codes]
        removable.sort(key=lambda x: x[1])

        for r in candidates:
            if r['code'] in target_codes or not removable:
                continue
            worst_code, worst_score = removable[0]
            if r['final_score'] > worst_score:
                target_codes.discard(worst_code)
                target_codes.add(r['code'])
                removable.pop(0)
                log.info('[换仓] %s(%.1f) 替换 %s(%.1f) 差%.1f分' % (
                    r['code'], r['final_score'], worst_code, worst_score,
                    r['final_score'] - worst_score))

    # 6. 执行止损（止损豁免：仍在目标中的不卖）
    sold_proceeds = 0  # 追踪卖出释放的资金（PTrade实盘cash有6秒同步延迟）
    for code in stop_triggered:
        if code in target_codes:
            log.info('[止损豁免] %s 得分%.1f仍在目标中，保留持仓' % (
                code, g.holding_scores.get(code, 0)))
        else:
            # 记录卖出市值
            cur_price = _get_current_price(code)
            if cur_price and code in positions:
                sold_proceeds += cur_price * _pos_amount(positions[code])
            _execute_stop(code, context)

    # 7. 轮动卖出
    for code in list(current_holds.keys()):
        if code not in target_codes and code not in stop_triggered:
            if _is_paused(code):
                log.info('[跳过卖出] %s 停牌中，保留持仓' % code)
                continue
            if g.__is_live and g.sold_today.get(code, False):
                continue
            cur_price = _get_current_price(code)
            cost = _pos_cost(positions[code])
            if cur_price and cost > 0:
                pnl = (cur_price - cost) / cost
                log.info('[轮动卖出] %s 盈亏%.1f%% 得分:%.1f' % (
                    code, pnl * 100, g.holding_scores.get(code, 0)))
            # 记录卖出市值
            if cur_price and code in positions:
                sold_proceeds += cur_price * _pos_amount(positions[code])
            sell_lmt = _get_sell_limit_price(code, cur_price or cost)
            order_target(code, 0, limit_price=sell_lmt)
            g.sold_today[code] = True
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            g.buy_date.pop(code, None)
            g.holding_scores.pop(code, None)

    # 8. 买入（按得分排序）
    to_buy = [c for c in target_codes if c not in current_holds]
    if not to_buy:
        log.info('[无换仓] 持仓与目标一致')
        return

    sig_map = {}
    for r in all_results:
        sig_map[r['code']] = r
    to_buy.sort(key=lambda c: sig_map.get(c, {}).get('final_score', 0), reverse=True)

    # 可用资金：实盘cash有6秒同步延迟需补偿，回测cash即时更新不需要
    if g.__is_live and sold_proceeds > 0:
        available = _available_cash(context) + sold_proceeds
        log.info('[资金] 当前现金:%.0f + 卖出释放:%.0f = 可用:%.0f' % (
            _available_cash(context), sold_proceeds, available))
    else:
        available = _available_cash(context)
    slots = max_hold - len(set(current_holds.keys()) & target_codes)
    if slots <= 0 or available < 500:
        log.info('[跳过买入] 无空仓位(slots=%d)或资金不足(%.0f)' % (slots, available))
        return

    base_ratio = _get_tier_param('base_ratio')

    for code in to_buy:
        if slots <= 0 or available < 500:
            break
        if code not in sig_map:
            continue

        price = _get_current_price(code)
        if price is None or price <= 0:
            continue

        sig = sig_map[code]
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

        order(code, shares, limit_price=round(price, 3))

        if g.__is_live:
            g.__pending_orders[code] = {'price': price, 'atr': sig['atr']}
        else:
            g.highest_since_buy[code] = price
            g.entry_atr[code] = sig['atr']

        g.buy_date[code] = today
        g.holding_scores[code] = sig['final_score']
        available -= shares * price * 1.003
        slots -= 1


# ============================================================
#  更新最高价 + ATR
# ============================================================
def _update_highest_and_atr(context):
    today = context.blotter.current_dt.date()
    positions = _positions(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        cur = _get_current_price(code)
        if cur is None:
            continue

        if code in g.highest_since_buy:
            if cur > g.highest_since_buy[code]:
                g.highest_since_buy[code] = cur
        else:
            g.highest_since_buy[code] = max(cur, _pos_cost(pos))

        if code in g.entry_atr:
            atr_df = _get_price_data(code, today, count=g.params['atr_period'] + 5)
            if atr_df is not None and len(atr_df) >= g.params['atr_period']:
                new_atr = _calc_atr(atr_df['high'], atr_df['low'],
                                    atr_df['close'], g.params['atr_period']).iloc[-1]
                if not pd.isna(new_atr) and new_atr > 0:
                    g.entry_atr[code] = new_atr


# ============================================================
#  恢复缺失ATR（实盘重启安全网）
# ============================================================
def _recover_missing_atr(context, deliver_records=None):
    positions = _positions(context)
    prev_date = _get_prev_trade_date(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        if code in g.entry_atr:
            continue
        atr_period = g.params['atr_period']
        df = _get_price_data(code, prev_date, count=atr_period + 5)
        if df is not None and len(df) >= atr_period:
            atr_val = _calc_atr(df['high'], df['low'], df['close'], atr_period).iloc[-1]
            if not pd.isna(atr_val):
                g.entry_atr[code] = atr_val
                _ensure_buy_date_and_highest(code, pos, prev_date, deliver_records)
                log.info('[ATR恢复] %s ATR=%.4f' % (code, atr_val))
                continue
        cost = _pos_cost(pos)
        g.entry_atr[code] = cost * 0.02
        _ensure_buy_date_and_highest(code, pos, prev_date, deliver_records)
        log.warning('[ATR恢复] %s 数据不足，用成本2%%估算' % code)


def _fetch_deliver_records(prev_date):
    """查询交割单（只查一次，供所有持仓共用）"""
    try:
        start = (prev_date - timedelta(days=180)).strftime('%Y%m%d')
        end = prev_date.strftime('%Y%m%d')
        log.info('[交割单] 查询区间%s~%s (prev_date=%s)' % (start, end, prev_date))
        records = get_deliver(start, end)
        if not records:
            log.warning('[交割单] 查询区间%s~%s 无记录' % (start, end))
            return []
        log.info('[交割单] 返回%d条记录' % len(records))
        return records
    except Exception as e:
        log.warning('[交割单] 查询失败: %s' % e)
        return []


def _ensure_buy_date_and_highest(code, pos, prev_date, deliver_records=None):
    """确保持仓有buy_date和highest_since_buy"""
    cost = _pos_cost(pos)
    if g.buy_date.get(code) is None:
        g.buy_date[code] = _query_buy_date_from_deliver(code, prev_date, deliver_records)
    if g.buy_date.get(code) is None:
        g.buy_date[code] = prev_date - timedelta(days=10)
        log.info('[买入日恢复] %s 设为%s（近似值 prev_date=%s）' % (code, g.buy_date[code], prev_date))
    if code not in g.highest_since_buy:
        g.highest_since_buy[code] = _recover_highest(code, cost, prev_date)


def _query_buy_date_from_deliver(code, prev_date, deliver_records=None):
    """通过交割单查询最近一次买入日期"""
    if deliver_records is None:
        deliver_records = _fetch_deliver_records(prev_date)
    if not deliver_records:
        return None

    # 标的代码：交割单中 stock_code 是6位数字码
    code_base = code.split('.')[0]

    buy_dates = []
    matched_codes = set()
    for r in deliver_records:
        # 标的匹配
        r_code = str(r.get('stock_code', ''))
        matched_codes.add(r_code)
        if r_code != code_base:
            continue

        # 买卖方向：entrust_bs='1' 买入, '2' 卖出; business_name='证券买入'/'证券卖出'
        entrust_bs = str(r.get('entrust_bs', ''))
        business_name = str(r.get('business_name', ''))
        if entrust_bs != '1' and '买入' not in business_name:
            continue

        # 成交日期：init_date / entrust_date / date_back，格式YYYYmmdd
        date_str = str(r.get('init_date', '') or r.get('entrust_date', '') or r.get('date_back', ''))
        if not date_str or len(date_str) < 8:
            continue
        try:
            trade_date = datetime.strptime(date_str[:8], '%Y%m%d').date()
            buy_dates.append(trade_date)
        except Exception:
            continue

    log.info('[交割单] %s 匹配code=%s 买入记录%d条 全量code列表:%s' % (
        code, code_base, len(buy_dates), sorted(matched_codes)))

    if buy_dates:
        most_recent = max(buy_dates)  # 最近一次买入
        log.info('[买入日恢复] %s 交割单最近买入%s' % (code, most_recent))
        return most_recent

    log.warning('[买入日恢复] %s 交割单中未找到买入记录' % code)
    return None


def _recover_highest(code, cost, prev_date):
    """恢复持仓最高价：从买入日到prev_date取收盘价最大值"""
    buy_date = g.buy_date.get(code, None)
    if buy_date is not None:
        df = _get_price_data(code, prev_date, count=500)
        if df is not None and len(df) > 0:
            df = df[df.index >= pd.Timestamp(buy_date)]
            if len(df) > 0:
                hist_high = df['close'].max()
                recovered = max(cost, hist_high)
                log.info('[最高价恢复] %s 买入日%s prev_date=%s 区间最高收盘%.3f 成本%.3f → 恢复%.3f' % (
                    code, buy_date, prev_date, hist_high, cost, recovered))
                return recovered
    # 兜底：用120天数据找最高收盘价
    df = _get_price_data(code, prev_date, count=120)
    if df is not None and len(df) > 0:
        hist_high = df['close'].max()
        recovered = max(cost, hist_high)
        # 补一个近似买入日（无法从交割单查到时的兜底）
        if g.buy_date.get(code) is None:
            g.buy_date[code] = prev_date - timedelta(days=10)
            log.info('[买入日恢复] %s 设为%s（近似值 prev_date=%s）' % (code, g.buy_date[code], prev_date))
        log.info('[最高价恢复] %s 买入日未知 prev_date=%s 120日最高收盘%.3f 成本%.3f → 恢复%.3f' % (
            code, prev_date, hist_high, cost, recovered))
        return recovered
    log.warning('[最高价恢复] %s 无历史数据，用成本价' % code)
    return cost


def _recover_missing_highest(context, deliver_records=None):
    """恢复缺失的最高价（entry_atr存在但highest_since_buy缺失的边缘情况）"""
    positions = _positions(context)
    prev_date = _get_prev_trade_date(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        if code in g.highest_since_buy:
            continue
        _ensure_buy_date_and_highest(code, pos, prev_date, deliver_records)


def _recover_missing_buy_date(context, deliver_records=None):
    """恢复缺失的买入日期（highest_since_buy存在但buy_date缺失的边缘情况）"""
    positions = _positions(context)
    prev_date = _get_prev_trade_date(context)
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue
        if g.buy_date.get(code) is not None:
            continue
        g.buy_date[code] = _query_buy_date_from_deliver(code, prev_date, deliver_records)
        if g.buy_date.get(code) is None:
            g.buy_date[code] = prev_date - timedelta(days=10)
            log.info('[买入日恢复] %s 设为%s（近似值 prev_date=%s）' % (code, g.buy_date[code], prev_date))


# ============================================================
#  盘后记录
# ============================================================
def _after_close(context):
    if g.current_tier is None:
        return

    positions = _positions(context)
    hold = {}
    for code, pos in positions.items():
        if _pos_amount(pos) > 0:
            hold[code] = pos

    total_value = _total_value(context)
    if total_value > g.portfolio_high:
        g.portfolio_high = total_value
    portfolio_dd = (g.portfolio_high - total_value) / g.portfolio_high * 100 if g.portfolio_high > 0 else 0

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d 组合回撤:%.1f%%' % (
        g.current_tier, total_value, _available_cash(context),
        len(hold), _get_tier_param('max_hold'), portfolio_dd))

    for code, pos in hold.items():
        cur = _pos_price(pos)
        cost = _pos_cost(pos)
        pnl = (cur - cost) / cost * 100 if cost > 0 else 0
        highest = g.highest_since_buy.get(code, cur)
        score = g.holding_scores.get(code, 0)
        atr_val = g.entry_atr.get(code, None)
        if atr_val is not None and highest > 0:
            stop_price = _calc_stop_price(highest, atr_val)
            log.info('  %s 成本:%.3f 现:%.3f 高:%.3f 盈亏:%.1f%% 分:%.1f ATR:%.4f 止损价:%.3f' % (
                code, cost, cur, highest, pnl, score, atr_val, stop_price))
        else:
            log.info('  %s 成本:%.3f 现:%.3f 高:%.3f 盈亏:%.1f%% 分:%.1f' % (
                code, cost, cur, highest, pnl, score))
    log.info('=' * 60)


# ============================================================
#  委托/成交回调（仅交易模式可用）
# ============================================================
def on_order_response(context, order_list):
    try:
        if not g.__is_live:
            return
    except AttributeError:
        return

    if not isinstance(order_list, list):
        order_list = [order_list]

    for od in order_list:
        code = od.get('stock_code', '')
        status = od.get('status', '')
        error_info = od.get('error_info', '')

        if str(status) in ('5', '6', '9'):
            if code in g.__pending_orders:
                g.__pending_orders.pop(code, None)
                log.warning('[买入失败] %s 状态:%s 原因:%s' % (code, status, error_info))
            elif code in g.sold_today:
                log.error('[止损告警] %s 卖出失败！%s' % (code, error_info))
                g.sold_today.pop(code, None)


def on_trade_response(context, trade_list):
    try:
        if not g.__is_live:
            return
    except AttributeError:
        return

    if not isinstance(trade_list, list):
        trade_list = [trade_list]

    for trade in trade_list:
        code = trade.get('stock_code', '')
        direction = trade.get('entrust_bs', '')
        filled_qty = trade.get('business_amount', 0)
        filled_price = trade.get('business_price', 0)

        if not code or filled_qty <= 0:
            continue

        if direction == '1' and code in g.__pending_orders:
            pending = g.__pending_orders.pop(code)
            g.highest_since_buy[code] = filled_price
            g.entry_atr[code] = pending['atr']
            log.info('[成交确认] 买入 %s %d股 @%.3f' % (code, filled_qty, filled_price))
        elif direction == '2':
            log.info('[成交确认] 卖出 %s %d股 @%.3f' % (code, filled_qty, filled_price))
