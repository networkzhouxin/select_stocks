# -*- coding: utf-8 -*-
"""
智能买卖点量化交易策略 V7.0 - 聚宽版
======================================
完整移植通达信 V7.0 全部逻辑：
  - 6个买入条件 + 差异化权重
  - 6个卖出条件 + 差异化权重
  - 趋势五档评分 + 趋势系数加减分
  - 拉伸度标准化融合(RSI/KDJ/CCI/BIAS)
  - 顶底背离检测
  - 信号三级分类(强/中/弱)
  - 冷却去重机制
  - 基本面安全过滤

资金：2万
市场：主板 + 创业板（排除科创板、北交所、ST）
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

    # 滑点与手续费（贴近实盘）
    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,       # 卖出印花税千一
        open_commission=0.0003,  # 最低5元由聚宽自动处理
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- 策略参数 ----
    g.max_hold = 2          # 最大持仓数（2万资金，最多2只）
    g.signal_level = 2      # 买入信号级别：1=全部, 2=中+强, 3=只强买
    g.trend_filter = 1      # 趋势过滤：0=不限, 1=偏多以上, 2=只多头
    g.cooldown_buy = 6      # 买入冷却天数
    g.cooldown_sell = 8     # 卖出冷却天数

    # ---- 运行时状态 ----
    g.buy_signal_history = {}   # {stock: [日期列表]} 买入信号历史
    g.sell_signal_history = {}  # {stock: [日期列表]} 卖出信号历史

    # 每日运行
    run_daily(market_open, time='09:35')
    run_daily(after_close, time='15:30')


# ============================================================
#  盘前：获取股票池
# ============================================================
def get_stock_pool(context):
    """获取主板+创业板股票池，排除ST/停牌/次新/小市值"""
    today = context.current_dt.date()

    # 全部A股
    all_stocks = get_all_securities(types=['stock'], date=today)

    pool = []
    for code in all_stocks.index:
        # 只保留主板和创业板
        if code.startswith('60') or code.startswith('000') or code.startswith('001') \
                or code.startswith('300') or code.startswith('301'):
            # 排除ST
            name = all_stocks.loc[code, 'display_name']
            if 'ST' in name or 'st' in name:
                continue
            # 排除次新股（上市不满60天）
            start_date = all_stocks.loc[code, 'start_date']
            if (today - start_date).days < 60:
                continue
            pool.append(code)

    # 排除停牌
    paused_info = get_current_data()
    pool = [s for s in pool if not paused_info[s].paused]

    return pool


# ============================================================
#  技术指标计算
# ============================================================
def calc_sma(series, n, m):
    """通达信 SMA(X,N,M) = (M*X + (N-M)*前值) / N"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i - 1]) / n
    return result


def calc_indicators(code, end_date, count=120):
    """计算单只股票的全部技术指标，返回最后一天的信号"""

    # 获取日线数据（多取一些用于计算长周期指标）
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

    # ============================================================
    #  趋势判断（五档）
    # ============================================================
    MA20_5ago = MA20.shift(5)
    趋势分 = (
        (C > MA20).astype(int)
        + (MA20 > MA20_5ago).astype(int)
        + (C > MA60).astype(int)
        + (MA20 > MA60).astype(int)
        + (DIF0 > DEA0).astype(int)
    )
    趋势系数 = 趋势分.map(lambda x: {0: -2, 1: -1, 2: 0, 3: 1}.get(x, 2))

    # ============================================================
    #  拉伸度（标准化融合）
    # ============================================================
    RSI标准 = RSI6 - 50
    KDJ标准 = (J0 - 50) * 0.6
    CCI标准 = CCI0 * 0.2
    BIAS标准 = BIAS20 * 4

    拉伸 = RSI标准 * 0.3 + KDJ标准 * 0.3 + CCI标准 * 0.2 + BIAS标准 * 0.2
    拉伸线 = 拉伸.ewm(span=2, adjust=False).mean()

    # ============================================================
    #  背离检测
    # ============================================================
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

    # ============================================================
    #  买入信号（6个条件）
    # ============================================================
    # BU1: 放量站上MA10
    BU1 = (C > MA10) & (C.shift(1) <= MA10.shift(1)) & (VR > 1.0)

    # BU2: 极度超卖 + 阳线
    BU2 = ((J0 < 15) | (RSI6 < 25)) & 阳线 & (实体 > 0)

    # BU3: 长下影锤子线（低位）
    BU3 = (下影 > 实体 * 2) & (下影 > 上影 * 2) & (拉伸线 < -5)

    # BU4: 回踩上升MA20获支撑（限偏多以上）
    BU4 = ((MA20 > MA20.shift(1))
           & (L <= MA20 * 1.02)
           & (C > MA20)
           & 阳线
           & (趋势系数 >= 1))

    # BU5: J值极低后首阳
    BU5 = (J0.shift(1) < 10) & 阳线 & (实体 > 0)

    # BU6: 底背离
    BU6 = 底背离

    # 买入加权评分
    买原分 = (BU1.astype(float) * 1.0
             + BU2.astype(float) * 1.5
             + BU3.astype(float) * 1.0
             + BU4.astype(float) * 1.5
             + BU5.astype(float) * 1.0
             + BU6.astype(float) * 1.5)

    # 趋势修正
    买分 = 买原分.copy()
    买分 = 买分 + (趋势系数 >= 1).astype(float) * 1.0
    买分 = 买分 - (趋势系数 <= -1).astype(float) * 1.0

    # ============================================================
    #  卖出信号（6个条件）
    # ============================================================
    # SE1: 跌破MA10
    SE1 = (C < MA10) & (C.shift(1) >= MA10.shift(1))

    # SE2: 放量暴跌>3%
    SE2 = (C < C.shift(1) * 0.97) & (VR > 1.5)

    # SE3: 高位长上影线
    SE3 = (上影 > 实体 * 2) & (上影 > 下影 * 2) & (拉伸线 > 15)

    # SE4: 从5日高点回落>7%
    high5 = C.rolling(5).max()
    SE4 = C < high5 * 0.93

    # SE5: J值极高后首阴
    SE5 = (J0.shift(1) > 90) & 阴线 & (实体 > 0)

    # SE6: 顶背离
    SE6 = 顶背离

    # 卖出加权评分
    卖原分 = (SE1.astype(float) * 1.0
             + SE2.astype(float) * 1.5
             + SE3.astype(float) * 1.0
             + SE4.astype(float) * 1.5
             + SE5.astype(float) * 1.0
             + SE6.astype(float) * 1.5)

    # 趋势修正
    卖分 = 卖原分.copy()
    卖分 = 卖分 + (趋势系数 < 0).astype(float) * 1.0
    卖分 = 卖分 - (趋势系数 >= 2).astype(float) * 0.5

    # ============================================================
    #  信号分级（取最后一天）
    # ============================================================
    idx = -1  # 最后一天

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

    # 买入信号分级
    bs = result['买分']
    ts = result['趋势系数']
    if bs >= 3 or (bs >= 2.5 and ts >= 1):
        result['买入级别'] = 3  # 强买
    elif bs >= 2:
        result['买入级别'] = 2  # 中买
    elif bs >= 1.5:
        result['买入级别'] = 1  # 弱买
    else:
        result['买入级别'] = 0  # 无信号

    # 卖出信号分级
    ss = result['卖分']
    if ss >= 3 or (ss >= 2 and ts < 0):
        result['卖出级别'] = 3  # 强卖
    elif ss >= 2:
        result['卖出级别'] = 2  # 中卖
    elif ss >= 1:
        result['卖出级别'] = 1  # 弱卖
    else:
        result['卖出级别'] = 0  # 无信号

    return result


# ============================================================
#  冷却检查
# ============================================================
def check_cooldown(history_dict, code, today, cooldown_days):
    """检查是否在冷却期内"""
    if code not in history_dict:
        return False  # 无历史，不在冷却期
    recent = [d for d in history_dict[code] if (today - d).days <= cooldown_days]
    return len(recent) >= 1  # 冷却期内已有信号则冷却中


def record_signal(history_dict, code, today):
    """记录信号日期"""
    if code not in history_dict:
        history_dict[code] = []
    history_dict[code].append(today)
    # 只保留最近30天的记录
    history_dict[code] = [d for d in history_dict[code] if (today - d).days <= 30]


# ============================================================
#  仓位计算（适配2万小资金）
# ============================================================
def calc_position_size(context, signal_level, price):
    """
    根据信号强度和可用资金计算买入股数（100股整数倍）
    强买：可用资金的60%
    中买：可用资金的40%
    弱买：可用资金的25%
    """
    available = context.portfolio.available_cash
    ratio_map = {3: 0.60, 2: 0.40, 1: 0.25}
    ratio = ratio_map.get(signal_level, 0.25)

    amount = available * ratio
    shares = int(amount / price / 100) * 100  # 取整到100股

    # 最少买100股
    if shares < 100:
        if available >= price * 100 * 1.003:  # 含手续费
            shares = 100
        else:
            shares = 0

    return shares


# ============================================================
#  每日交易主函数
# ============================================================
def market_open(context):
    today = context.current_dt.date()
    current_data = get_current_data()

    # ========== 第一步：检查持仓卖出 ==========
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue

        # 停牌跳过
        if current_data[code].paused:
            continue

        sig = calc_indicators(code, today, count=120)
        if sig is None:
            continue

        sell_level = sig['卖出级别']

        # 检查卖出冷却（防止连续卖出同一只）
        if check_cooldown(g.sell_signal_history, code, today, g.cooldown_sell):
            # 冷却期内只有强卖才执行
            if sell_level < 3:
                continue

        if sell_level >= 1:
            # 跌停不能卖出
            if current_data[code].last_price <= current_data[code].low_limit:
                continue

            if sell_level == 3:
                # 强卖 → 清仓
                log.info('[强卖-清仓] %s 卖分=%.1f 趋势=%d' % (
                    code, sig['卖分'], sig['趋势系数']))
                order_target(code, 0)
                record_signal(g.sell_signal_history, code, today)

            elif sell_level == 2:
                # 中卖 → 减半仓
                target = int(pos.total_amount / 2 / 100) * 100
                log.info('[中卖-减仓] %s 卖分=%.1f 目标=%d股' % (
                    code, sig['卖分'], target))
                order_target(code, target)
                record_signal(g.sell_signal_history, code, today)

            else:
                # 弱卖 → 仅记录，不操作（观望）
                log.info('[弱卖-观望] %s 卖分=%.1f' % (code, sig['卖分']))

    # ========== 第二步：检查是否有买入名额 ==========
    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])

    if hold_count >= g.max_hold:
        return

    if context.portfolio.available_cash < 500:  # 资金不足
        return

    # ========== 第三步：选股并买入 ==========
    pool = get_stock_pool(context)

    # 按流通市值排序取前300只加速计算（2万资金无需扫全市场）
    q = query(
        valuation.code,
        valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(pool),
        valuation.circulating_market_cap > 5  # >5亿流通市值
    ).order_by(
        valuation.circulating_market_cap.asc()
    ).limit(500)

    df_val = get_fundamentals(q, date=today)
    if df_val is None or len(df_val) == 0:
        return

    candidates = df_val['code'].tolist()

    # 计算每只股票的信号
    buy_signals = []
    for code in candidates:
        # 跳过已持仓
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        # 买入冷却检查
        if check_cooldown(g.buy_signal_history, code, today, g.cooldown_buy):
            continue

        sig = calc_indicators(code, today, count=120)
        if sig is None:
            continue

        buy_level = sig['买入级别']
        sell_level = sig['卖出级别']

        # 无买入信号或同时有卖出信号 → 跳过
        if buy_level == 0 or sell_level >= 1:
            continue

        # 信号级别过滤
        if g.signal_level == 2 and buy_level < 2:
            continue
        if g.signal_level == 3 and buy_level < 3:
            continue

        # 趋势过滤
        ts = sig['趋势系数']
        if g.trend_filter == 1 and ts < 1:
            continue
        if g.trend_filter == 2 and ts < 2:
            continue

        # 涨停无法买入
        if current_data[code].last_price >= current_data[code].high_limit:
            continue

        buy_signals.append(sig)

    if not buy_signals:
        return

    # 按买分降序排列，优先买入最强信号
    buy_signals.sort(key=lambda x: x['买分'], reverse=True)

    # 买入（最多补到max_hold只）
    slots = g.max_hold - hold_count
    for sig in buy_signals[:slots]:
        code = sig['code']
        price = sig['close']
        level = sig['买入级别']

        shares = calc_position_size(context, level, price)
        if shares <= 0:
            continue

        level_name = {3: '强买-重仓', 2: '中买-半仓', 1: '弱买-轻仓'}[level]
        log.info('[%s] %s 买分=%.1f 趋势=%d 拉伸=%.1f 买入%d股' % (
            level_name, code, sig['买分'], sig['趋势系数'],
            sig['拉伸线'], shares))
        log.info('  触发条件: BU1=%s BU2=%s BU3=%s BU4=%s BU5=%s BU6=%s' % (
            tuple(sig['BU_details'])))

        order(code, shares)
        record_signal(g.buy_signal_history, code, today)


# ============================================================
#  盘后记录
# ============================================================
def after_close(context):
    positions = context.portfolio.positions
    hold = {code: pos for code, pos in positions.items() if pos.total_amount > 0}

    log.info('=' * 50)
    log.info('账户总值: %.2f  可用资金: %.2f  持仓数: %d' % (
        context.portfolio.total_value,
        context.portfolio.available_cash,
        len(hold)))

    for code, pos in hold.items():
        profit_pct = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        log.info('  %s  成本:%.2f  现价:%.2f  盈亏:%.1f%%  持股:%d' % (
            code, pos.avg_cost, pos.price, profit_pct, pos.total_amount))
    log.info('=' * 50)
