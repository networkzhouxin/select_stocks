# -*- coding: utf-8 -*-
"""
智能ETF量化交易策略 V10.0 - PTrade版（回测+实盘兼容）
=============================================
从聚宽V10.0逐行移植，所有交易逻辑、参数、信号条件100%不变。
根据PTrade官方API文档适配，同时兼容回测和实盘两种模式。

适配要点（根据PTrade-API.html官方文档）：
  1. 代码格式：.XSHG/.XSHE → .SS/.SZ
  2. 佣金/滑点：set_commission + set_slippage（仅回测生效）
  3. Portfolio属性：portfolio_value/cash 替代 total_value/available_cash
  4. Position属性：amount/cost_basis/last_sale_price 替代 total_amount/avg_cost/price
  5. 当前行情：data[code].price 替代 get_current_data()
  6. 停牌检测：get_stock_status 替代 paused属性
  7. run_daily签名：run_daily(context, func, time) 需传context
  8. 回测兼容：日频回测run_daily固定15:00，改用handle_data驱动全部逻辑
  9. 交易日获取：get_trade_days兼容 + get_all_trades_days回退
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
#  初始化
# ============================================================
def initialize(context):
    set_benchmark('000300.SS')

    # ---- 佣金与滑点（仅回测生效，实盘由券商设置）----
    try:
        set_commission(commission_ratio=0.0003, min_commission=5.0, type='ETF')
        set_slippage(slippage=0.001)
    except Exception:
        pass  # 实盘模式下这些函数不可用，跳过

    # ---- ETF标的池（PTrade使用.SS/.SZ格式）----
    g.etf_pool = [
        '510300.SS',    # 沪深300ETF
        '159915.SZ',    # 创业板ETF
        '510500.SS',    # 中证500ETF
    ]

    # ---- 资金档位配置（与聚宽版100%一致）----
    g.capital_tiers = {
        'micro': {'max_hold': 1, 'base_position_ratio': 0.85},
        'small': {'max_hold': 2, 'base_position_ratio': 0.70},
        'medium': {'max_hold': 3, 'base_position_ratio': 0.55},
        'large': {'max_hold': 3, 'base_position_ratio': 0.45},
    }

    # ---- 策略参数（与聚宽版100%一致）----
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

    # ---- 模式检测 ----
    # is_trade()：实盘返回True，回测返回False
    try:
        g.__is_live = is_trade()
    except Exception:
        g.__is_live = False

    # 实盘模式：使用run_daily精确控制执行时间
    if g.__is_live:
        run_daily(context, _update_tier_wrapper, time='09:25')
        run_daily(context, _market_open_wrapper, time='09:35')
        run_daily(context, _update_highest_wrapper, time='14:50')


def handle_data(context, data):
    """
    PTrade必须定义的函数。
    回测模式：日频在15:00调用，分钟频在每分钟调用。
    实盘模式：按券商配置调用（主逻辑已由run_daily处理）。

    回测模式下，handle_data承担全部逻辑（因为日频run_daily固定15:00无法分时段）。
    实盘模式下，handle_data仅更新最高价（run_daily已处理买卖逻辑）。
    """
    # 将data存入g供其他函数使用
    g.__data = data

    if g.__is_live:
        # 实盘模式：买卖逻辑由run_daily在09:35执行，这里只做补充更新
        return

    # 回测模式：在handle_data中执行全部逻辑
    _update_tier(context)
    _do_trading(context)
    _update_highest_prices(context)


def before_trading_start(context, data):
    """盘前准备（回测8:30/实盘9:10）"""
    g.__data = data


def after_trading_end(context, data):
    """盘后记录"""
    _after_close(context)


# ============================================================
#  run_daily包装函数（实盘模式使用）
# ============================================================
def _update_tier_wrapper(context):
    _update_tier(context)


def _market_open_wrapper(context):
    _do_trading(context)


def _update_highest_wrapper(context):
    _update_highest_prices(context)


# ============================================================
#  Portfolio/Position兼容层
# ============================================================
def _get_total_value(context):
    """获取总资产（兼容PTrade属性名）"""
    return context.portfolio.portfolio_value


def _get_available_cash(context):
    """获取可用资金（兼容PTrade属性名）"""
    return context.portfolio.cash


def _get_positions(context):
    """获取持仓字典"""
    return context.portfolio.positions


def _pos_amount(pos):
    """获取持仓数量"""
    return pos.amount


def _pos_cost(pos):
    """获取持仓成本"""
    return pos.cost_basis


def _pos_price(pos):
    """获取持仓当前价格"""
    return pos.last_sale_price


# ============================================================
#  行情数据获取
# ============================================================
def _get_current_price(code):
    """
    获取当前价格（兼容回测和实盘）
    回测：从g.__data获取
    实盘：优先get_snapshot，回退到get_history
    """
    # 方式1：从handle_data的data参数获取
    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].price
        except Exception:
            pass

    # 方式2：实盘用get_snapshot
    if g.__is_live:
        try:
            snap = get_snapshot(code)
            if snap and 'price' in snap:
                return snap['price']
            if snap and 'last' in snap:
                return snap['last']
        except Exception:
            pass

    # 方式3：回退到最近收盘价
    try:
        df = get_history(1, '1d', 'close', [code], include=True)
        return float(df[code].iloc[-1])
    except Exception:
        pass

    return None


def _is_paused(code):
    """检查是否停牌（PTrade用get_stock_status）"""
    try:
        status = get_stock_status([code], 'HALT')
        return status.get(code, False)
    except Exception:
        pass
    # 回退：检查data
    if hasattr(g, '__data') and g.__data is not None:
        try:
            return g.__data[code].is_open == 0
        except Exception:
            pass
    return False


def _get_prev_trade_date(context):
    """获取前一个交易日"""
    today = context.current_dt.date()
    try:
        trade_days = get_trade_days(end_date=today, count=2)
        return trade_days[0]
    except Exception:
        pass
    try:
        # PTrade拼写：get_all_trades_days（带s）
        all_days = get_all_trades_days(date=today)
        past_days = [d for d in all_days if d < today]
        if past_days:
            return past_days[-1]
    except Exception:
        pass
    # 最后回退
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def _get_price_data(code, end_date, count=120):
    """
    获取历史行情数据，返回DataFrame。
    PTrade的get_price参数：count和start_date互斥。
    """
    # 将date对象转为字符串（PTrade要求字符串格式）
    if hasattr(end_date, 'strftime'):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = str(end_date)

    try:
        df = get_price(code,
                       end_date=end_date_str,
                       count=count,
                       frequency='1d',
                       fields=['open', 'close', 'high', 'low', 'volume'],
                       fq='pre')
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    try:
        # 回退：用get_history
        df_c = get_history(count, '1d', 'close', [code], fq='pre')
        df_o = get_history(count, '1d', 'open', [code], fq='pre')
        df_h = get_history(count, '1d', 'high', [code], fq='pre')
        df_l = get_history(count, '1d', 'low', [code], fq='pre')
        df_v = get_history(count, '1d', 'volume', [code], fq='pre')
        df = pd.DataFrame({
            'open': df_o[code],
            'close': df_c[code],
            'high': df_h[code],
            'low': df_l[code],
            'volume': df_v[code]
        })
        return df
    except Exception as e:
        log.error('获取行情失败 %s: %s' % (code, str(e)))
        return None


# ============================================================
#  动态资金档位（逻辑与聚宽版100%一致）
# ============================================================
def _update_tier(context):
    total = _get_total_value(context)
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


def _get_tier_param(param_name):
    return g.capital_tiers[g.current_tier][param_name]


# ============================================================
#  技术指标计算（与聚宽版100%一致）
# ============================================================
def _calc_sma(series, n, m):
    """通达信SMA函数"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


def _calc_indicators(code, end_date, count=120):
    """计算技术指标并生成买卖信号（与聚宽版100%一致）"""
    df = _get_price_data(code, end_date, count)

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
    K0 = _calc_sma(RSV, 3, 1)
    D0 = _calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    # ------ RSI(6) ------
    LC = C.shift(1)
    diff_c = C - LC
    pos = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = _calc_sma(pos.fillna(0), 6, 1)
    sma_abs = _calc_sma(abs_diff.fillna(0), 6, 1)
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
    #  买入信号（4个条件）
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
    #  卖出信号（4个条件）
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
    #  信号分级
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
#  ATR动态止损（与聚宽版100%一致）
# ============================================================
def _calc_trailing_stop_price(highest_price, atr_value):
    atr_stop = highest_price - g.params['trailing_atr_mult'] * atr_value
    pct_stop = (highest_price - atr_stop) / highest_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return highest_price * (1 - pct_stop)


def _calc_max_loss_price(entry_price, entry_atr):
    atr_stop = entry_price - g.params['max_loss_atr_mult'] * entry_atr
    pct_stop = (entry_price - atr_stop) / entry_price
    pct_stop = max(g.params['stop_floor'], min(g.params['stop_cap'], pct_stop))
    return entry_price * (1 - pct_stop)


# ============================================================
#  冷却与记录（与聚宽版100%一致）
# ============================================================
def _check_cooldown(history_dict, code, today, cooldown_days):
    if code not in history_dict:
        return False
    recent = [d for d in history_dict[code] if (today - d).days <= cooldown_days]
    return len(recent) >= 1


def _record_signal(history_dict, code, today):
    if code not in history_dict:
        history_dict[code] = []
    history_dict[code].append(today)
    history_dict[code] = [d for d in history_dict[code] if (today - d).days <= 30]


# ============================================================
#  更新最高价
# ============================================================
def _update_highest_prices(context):
    positions = _get_positions(context)
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


# ============================================================
#  波动率调整仓位（与聚宽版100%一致）
# ============================================================
def _calc_position_size(context, signal, signal_level):
    available = _get_available_cash(context)
    base_ratio = _get_tier_param('base_position_ratio')

    strength_mult = {3: 1.0, 2: 0.8, 1: 0.6}.get(signal_level, 0.5)

    target_vol = 0.15
    actual_vol = max(signal['volatility'], 0.05)
    vol_mult = min(target_vol / actual_vol, 1.5)
    vol_mult = max(vol_mult, 0.4)

    ratio = base_ratio * strength_mult * vol_mult
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
#  核心交易逻辑（与聚宽版100%一致）
# ============================================================
def _do_trading(context):
    today = context.current_dt.date()
    prev_date = _get_prev_trade_date(context)
    p = g.params
    positions = _get_positions(context)

    # ========== 第一步：检查持仓卖出 ==========
    for code in list(positions.keys()):
        pos = positions[code]
        if _pos_amount(pos) <= 0:
            continue

        if _is_paused(code):
            continue

        cur_price = _get_current_price(code)
        if cur_price is None:
            continue

        cost = _pos_cost(pos)
        profit_pct = (cur_price - cost) / cost

        # 计算当前ATR
        sig = _calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        current_atr = sig['ATR']
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = cur_price * 0.02

        # --- ATR跟踪止损 ---
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            stop_price = _calc_trailing_stop_price(highest, current_atr)
            if cur_price <= stop_price:
                drawdown = (highest - cur_price) / highest
                log.info('[ATR跟踪止损] %s 最高%.3f 现价%.3f ATR=%.4f 回撤%.1f%% 盈亏%.1f%%' % (
                    code, highest, cur_price, current_atr,
                    drawdown * 100, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                _record_signal(g.sell_signal_history, code, today)
                continue

        # --- ATR最大亏损止损 ---
        if code in g.entry_atr:
            max_loss_price = _calc_max_loss_price(cost, g.entry_atr[code])
            if cur_price <= max_loss_price:
                log.info('[ATR最大止损] %s 成本%.3f 现价%.3f 亏损%.1f%%' % (
                    code, cost, cur_price, profit_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                g.entry_atr.pop(code, None)
                _record_signal(g.sell_signal_history, code, today)
                continue

        # --- 趋势持有模式 ---
        trend_score = sig['趋势分']
        if trend_score >= p['trend_hold_score'] and profit_pct > 0:
            continue

        # --- 信号卖出 ---
        sell_level = sig['卖出级别']

        if _check_cooldown(g.sell_signal_history, code, today, p['cooldown_days']):
            if sell_level < 3:
                continue

        if sell_level >= 2:
            log.info('[信号卖出] %s 级别=%d 卖分=%.1f 趋势=%d 盈亏=%.1f%%' % (
                code, sell_level, sig['卖分'], sig['趋势系数'], profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            g.entry_atr.pop(code, None)
            _record_signal(g.sell_signal_history, code, today)

    # ========== 第二步：检查是否可以买入 ==========
    positions = _get_positions(context)  # 刷新（卖出后持仓已变）
    hold_count = len([c for c in positions if _pos_amount(positions[c]) > 0])

    if hold_count >= _get_tier_param('max_hold'):
        return

    if _get_available_cash(context) < 500:
        return

    # ========== 第三步：扫描ETF池信号 ==========
    buy_candidates = []
    for code in g.etf_pool:
        if code in positions and _pos_amount(positions[code]) > 0:
            continue

        if _check_cooldown(g.buy_signal_history, code, today, p['cooldown_days']):
            continue

        if _is_paused(code):
            continue

        sig = _calc_indicators(code, prev_date, count=120)
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

    buy_candidates.sort(
        key=lambda x: x['买分'] * 0.6 + x['risk_adj_momentum'] * 0.4,
        reverse=True
    )

    slots = _get_tier_param('max_hold') - hold_count
    for sig in buy_candidates[:slots]:
        code = sig['code']
        price = sig['close']
        level = sig['买入级别']

        shares = _calc_position_size(context, sig, level)
        if shares <= 0:
            continue

        level_name = {3: '强买', 2: '中买', 1: '弱买'}.get(level, '买')
        log.info('[%s] %s 买分=%.1f 趋势=%d 动量=%.2f 波动率=%.1f%% ATR=%.4f %d股 @%.3f' % (
            level_name, code, sig['买分'], sig['趋势系数'],
            sig['risk_adj_momentum'], sig['volatility'] * 100,
            sig['ATR'], shares, price))
        log.info('  BU: %s' % sig['BU_details'])

        order(code, shares)
        g.highest_since_buy[code] = price
        g.entry_atr[code] = sig['ATR']
        _record_signal(g.buy_signal_history, code, today)


# ============================================================
#  盘后记录
# ============================================================
def _after_close(context):
    if g.current_tier is None:
        return

    positions = _get_positions(context)
    hold = {code: pos for code, pos in positions.items() if _pos_amount(pos) > 0}

    log.info('=' * 60)
    log.info('[%s] 总值:%.2f 现金:%.2f 持仓:%d/%d' % (
        g.current_tier,
        _get_total_value(context),
        _get_available_cash(context),
        len(hold),
        _get_tier_param('max_hold')))

    for code, pos in hold.items():
        cur_price = _pos_price(pos)
        cost = _pos_cost(pos)
        profit_pct = (cur_price - cost) / cost * 100 if cost > 0 else 0
        highest = g.highest_since_buy.get(code, cur_price)
        atr = g.entry_atr.get(code, 0)
        log.info('  %s 成本:%.3f 现价:%.3f 高:%.3f 入场ATR:%.4f 盈亏:%.1f%%' % (
            code, cost, cur_price, highest, atr, profit_pct))
    log.info('=' * 60)
