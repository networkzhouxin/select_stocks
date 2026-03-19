# -*- coding: utf-8 -*-
"""
智能买卖点量化交易策略 V7.3 - 聚宽版
======================================
V7.3 基于4年回测(2022-2025)暴露的问题优化：
  V7.2问题：4年最大回撤49%，2023-2024连续2年水下

核心改进：
  1. 三层择时：大盘趋势+大盘波动率+个股趋势
  2. 动态仓位：根据市场环境调节总仓位上限
  3. 账户级风控：总资产回撤>15%强制清仓冷静
  4. 跟踪止损根据波动率自适应
  5. 保留全部6买6卖信号逻辑

资金：2万
市场：主板 + 创业板
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

    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- 策略参数 ----
    g.max_hold = 2
    g.signal_level = 2
    g.trend_filter = 1
    g.cooldown_buy = 5
    g.cooldown_sell = 5

    # ---- 风控参数 ----
    g.account_stop_loss = 0.12     # 账户总值从高点回撤12%暂停买入
    g.account_clear_loss = 0.18    # 账户总值从高点回撤18%全部清仓
    g.freeze_days = 15             # 触发账户风控后冻结交易天数
    g.max_loss_pct = 0.10          # 单股最大亏损10%兜底

    # ---- 运行时状态 ----
    g.buy_signal_history = {}
    g.sell_signal_history = {}
    g.highest_since_buy = {}
    g.account_high = 0             # 账户净值最高点
    g.freeze_until = None          # 冻结交易截止日期

    run_daily(market_open, time='09:35')
    run_daily(update_highest, time='15:00')
    run_daily(after_close, time='15:30')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


# ============================================================
#  三层市场环境判断
# ============================================================
def get_market_regime(context):
    """
    返回市场状态：
      'bull'   - 牛市/强势（积极买入）
      'normal' - 正常（正常买入）
      'weak'   - 弱势（谨慎，只买强信号）
      'bear'   - 熊市（禁止买入）
    """
    prev_date = get_prev_trade_date(context)

    df = get_price('000300.XSHG', end_date=prev_date, count=65,
                   frequency='daily', fields=['close'])
    if df is None or len(df) < 61:
        return 'normal'

    close = df['close']
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    cur = close.iloc[-1]

    # 20日收益率判断动量
    ret_20 = (cur - close.iloc[-21]) / close.iloc[-21]

    # 波动率（20日年化）
    daily_ret = close.pct_change().dropna()
    vol_20 = daily_ret.tail(20).std() * np.sqrt(252)

    # 三层判断
    above_ma20 = cur > ma20
    above_ma60 = cur > ma60
    ma20_rising = ma20 > close.rolling(20).mean().iloc[-6]  # MA20比5天前高

    if above_ma20 and above_ma60 and ma20_rising:
        if vol_20 < 0.25:
            return 'bull'
        else:
            return 'normal'
    elif above_ma20:
        return 'normal'
    elif not above_ma20 and not above_ma60:
        if ret_20 < -0.08 or vol_20 > 0.30:
            return 'bear'
        else:
            return 'weak'
    else:
        return 'weak'


# ============================================================
#  自适应跟踪止损比例
# ============================================================
def get_trailing_stop(code, end_date):
    """根据个股波动率动态调整止损比例：波动大的股票给更宽的止损"""
    df = get_price(code, end_date=end_date, count=25,
                   frequency='daily', fields=['close'])
    if df is None or len(df) < 20:
        return 0.08

    daily_ret = df['close'].pct_change().dropna()
    vol = daily_ret.tail(20).std()

    # 基础8%，根据波动率上下浮动
    # 日波动率2%的股票 → 止损约8%
    # 日波动率3%的股票 → 止损约10%
    # 日波动率4%的股票 → 止损约12%（上限）
    stop = max(0.06, min(0.12, vol * 4))
    return stop


# ============================================================
#  股票池
# ============================================================
def get_stock_pool(context, use_date=None):
    today = use_date or context.current_dt.date()

    all_stocks = get_all_securities(types=['stock'], date=today)

    pool = []
    for code in all_stocks.index:
        if code.startswith('60') or code.startswith('000') or code.startswith('001') \
                or code.startswith('300') or code.startswith('301'):
            name = all_stocks.loc[code, 'display_name']
            if 'ST' in name or 'st' in name:
                continue
            start_date = all_stocks.loc[code, 'start_date']
            if (today - start_date).days < 120:
                continue
            pool.append(code)

    paused_info = get_current_data()
    pool = [s for s in pool if not paused_info[s].paused]

    return pool


# ============================================================
#  技术指标计算（完整保留全部逻辑）
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

    MA5 = C.rolling(5).mean()
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean()

    low9 = L.rolling(9).min()
    high9 = H.rolling(9).max()
    RSV = (C - low9) / (high9 - low9) * 100
    RSV = RSV.fillna(50)
    K0 = calc_sma(RSV, 3, 1)
    D0 = calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    LC = C.shift(1)
    diff_c = C - LC
    pos = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = calc_sma(pos.fillna(0), 6, 1)
    sma_abs = calc_sma(abs_diff.fillna(0), 6, 1)
    RSI6 = sma_pos / sma_abs.replace(0, np.nan) * 100
    RSI6 = RSI6.fillna(50)

    TYP0 = (H + L + C) / 3
    ma_typ = TYP0.rolling(14).mean()
    avedev_typ = TYP0.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    CCI0 = (TYP0 - ma_typ) / (0.015 * avedev_typ.replace(0, np.nan))
    CCI0 = CCI0.fillna(0)

    V5 = V.rolling(5).mean()
    VR = V / V5.shift(1).replace(0, np.nan)
    VR = VR.fillna(1)

    实体 = (C - O).abs()
    上影 = H - pd.concat([C, O], axis=1).max(axis=1)
    下影 = pd.concat([C, O], axis=1).min(axis=1) - L
    阳线 = C >= O
    阴线 = C <= O

    BIAS20 = (C - MA20) / MA20 * 100

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
#  仓位计算（根据市场环境动态调节）
# ============================================================
def calc_position_size(context, signal_level, price, regime):
    available = context.portfolio.available_cash

    # 基础仓位比例
    ratio_map = {3: 0.70, 2: 0.50, 1: 0.35}
    ratio = ratio_map.get(signal_level, 0.35)

    # 根据市场环境缩放
    regime_scale = {'bull': 1.0, 'normal': 0.8, 'weak': 0.5, 'bear': 0.0}
    ratio = ratio * regime_scale.get(regime, 0.5)

    amount = available * ratio
    shares = int(amount / price / 100) * 100

    if shares < 100:
        if available >= price * 100 * 1.003:
            shares = 100
        else:
            shares = 0

    return shares


# ============================================================
#  账户级风控
# ============================================================
def check_account_risk(context):
    """
    返回:
      'normal' - 正常交易
      'pause'  - 暂停买入（回撤>12%）
      'clear'  - 全部清仓（回撤>18%）
      'frozen' - 冻结期内不交易
    """
    today = context.current_dt.date()
    total_value = context.portfolio.total_value

    # 更新账户最高值
    if total_value > g.account_high:
        g.account_high = total_value

    # 冻结期检查
    if g.freeze_until and today <= g.freeze_until:
        return 'frozen'
    elif g.freeze_until and today > g.freeze_until:
        g.freeze_until = None
        g.account_high = total_value  # 重置高点
        log.info('[风控解除] 冻结期结束，重置账户高点为 %.2f' % total_value)

    # 账户回撤计算
    if g.account_high > 0:
        drawdown = (g.account_high - total_value) / g.account_high

        if drawdown >= g.account_clear_loss:
            return 'clear'
        elif drawdown >= g.account_stop_loss:
            return 'pause'

    return 'normal'


# ============================================================
#  每日交易主函数
# ============================================================
def market_open(context):
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    current_data = get_current_data()

    # ========== 账户级风控检查 ==========
    risk_status = check_account_risk(context)

    if risk_status == 'clear':
        log.info('[账户风控-清仓] 总值从高点%.2f回撤超%.0f%%，全部清仓！' % (
            g.account_high, g.account_clear_loss * 100))
        for code in list(context.portfolio.positions.keys()):
            pos = context.portfolio.positions[code]
            if pos.total_amount > 0 and not current_data[code].paused:
                if current_data[code].last_price > current_data[code].low_limit:
                    order_target(code, 0)
                    g.highest_since_buy.pop(code, None)
        import datetime
        g.freeze_until = today + datetime.timedelta(days=g.freeze_days)
        log.info('[冻结交易] 至 %s' % g.freeze_until)
        return

    if risk_status == 'frozen':
        return

    # ========== 第一步：检查持仓卖出 ==========
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue

        if current_data[code].paused:
            continue

        if current_data[code].last_price <= current_data[code].low_limit:
            continue

        cur_price = current_data[code].last_price
        profit_pct = (cur_price - pos.avg_cost) / pos.avg_cost

        # --- 自适应跟踪止损 ---
        if code in g.highest_since_buy:
            highest = g.highest_since_buy[code]
            drawdown = (highest - cur_price) / highest
            trail_pct = get_trailing_stop(code, prev_date)
            if drawdown >= trail_pct:
                log.info('[跟踪止损] %s 最高%.2f 现价%.2f 回撤%.1f%%(阈值%.0f%%)' % (
                    code, highest, cur_price, drawdown * 100, trail_pct * 100))
                order_target(code, 0)
                g.highest_since_buy.pop(code, None)
                record_signal(g.sell_signal_history, code, today)
                continue

        # --- 最大亏损止损（兜底）---
        if profit_pct <= -g.max_loss_pct:
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

        if check_cooldown(g.sell_signal_history, code, today, g.cooldown_sell):
            if sell_level < 3:
                continue

        if sell_level >= 2:
            log.info('[信号卖出] %s 级别=%d 卖分=%.1f 趋势=%d 盈亏=%.1f%%' % (
                code, sell_level, sig['卖分'], sig['趋势系数'], profit_pct * 100))
            order_target(code, 0)
            g.highest_since_buy.pop(code, None)
            record_signal(g.sell_signal_history, code, today)

    # ========== 第二步：检查是否可以买入 ==========
    if risk_status == 'pause':
        log.info('[账户风控] 回撤超%.0f%%，暂停买入' % (g.account_stop_loss * 100))
        return

    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])

    if hold_count >= g.max_hold:
        return

    if context.portfolio.available_cash < 500:
        return

    # 大盘环境
    regime = get_market_regime(context)

    if regime == 'bear':
        log.info('[市场环境] 熊市，禁止买入')
        return

    # ========== 第三步：选股并买入 ==========
    pool = get_stock_pool(context, use_date=prev_date)

    # 弱势市场要求更高信号级别
    min_signal = g.signal_level
    if regime == 'weak':
        min_signal = max(min_signal, 3)  # 弱势只买强信号

    q = query(
        valuation.code,
        valuation.circulating_market_cap,
        indicator.roe,
        indicator.inc_revenue_year_on_year
    ).filter(
        valuation.code.in_(pool),
        valuation.circulating_market_cap > 30,
        valuation.circulating_market_cap < 500,
        indicator.roe > 3,
        indicator.inc_revenue_year_on_year > -20
    ).order_by(
        indicator.roe.desc()
    ).limit(500)

    df_val = get_fundamentals(q, date=prev_date)
    if df_val is None or len(df_val) == 0:
        return

    candidates = df_val['code'].tolist()

    buy_signals = []
    for code in candidates:
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        if check_cooldown(g.buy_signal_history, code, today, g.cooldown_buy):
            continue

        sig = calc_indicators(code, prev_date, count=120)
        if sig is None:
            continue

        buy_level = sig['买入级别']
        sell_level = sig['卖出级别']

        if buy_level == 0 or sell_level >= 1:
            continue

        # 动态信号门槛
        if buy_level < min_signal:
            continue

        ts = sig['趋势系数']
        if g.trend_filter == 1 and ts < 1:
            continue
        if g.trend_filter == 2 and ts < 2:
            continue

        if current_data[code].last_price >= current_data[code].high_limit:
            continue

        if sig['close'] > 180:
            continue

        buy_signals.append(sig)

    if not buy_signals:
        return

    buy_signals.sort(key=lambda x: (x['买分'], x['趋势系数']), reverse=True)

    slots = g.max_hold - hold_count
    for sig in buy_signals[:slots]:
        code = sig['code']
        price = sig['close']
        level = sig['买入级别']

        shares = calc_position_size(context, level, price, regime)
        if shares <= 0:
            continue

        level_name = {3: '强买', 2: '中买', 1: '弱买'}[level]
        log.info('[%s|%s] %s 买分=%.1f 趋势=%d 拉伸=%.1f %d股 @%.2f' % (
            level_name, regime, code, sig['买分'], sig['趋势系数'],
            sig['拉伸线'], shares, price))

        order(code, shares)
        g.highest_since_buy[code] = price
        record_signal(g.buy_signal_history, code, today)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    dd = 0
    if g.account_high > 0:
        dd = (g.account_high - context.portfolio.total_value) / g.account_high * 100

    log.info('=' * 55)
    log.info('总值:%.2f 高点:%.2f 回撤:%.1f%% 现金:%.2f 持仓:%d' % (
        context.portfolio.total_value,
        g.account_high,
        dd,
        context.portfolio.available_cash,
        len(hold)))

    for code, pos in hold.items():
        profit_pct = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        highest = g.highest_since_buy.get(code, pos.price)
        log.info('  %s 成本:%.2f 现价:%.2f 高:%.2f 盈亏:%.1f%% 持:%d' % (
            code, pos.avg_cost, pos.price, highest, profit_pct, pos.total_amount))
    log.info('=' * 55)
