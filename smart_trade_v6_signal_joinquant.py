# -*- coding: utf-8 -*-
"""
智能买卖点策略 - 基于V6.0通达信指标
====================================
策略逻辑：
  1. 全市场扫描，出现强买信号（买分>=2）立即买入
  2. 浮盈达5%立即止盈卖出
  3. 浮亏达2%立即止损卖出
  4. 卖出后继续扫描寻找下一个标的

信号体系（完整移植V6.0）：
  买入5条件：站上MA10/极度超卖阳线/锤子线/回踩MA20/J值极低首阳
  卖出5条件：跌破MA10/放量暴跌/高位长上影/5日高点回落8%/J极高首阴
  强买 = 买分>=2（至少2个买入条件同时满足）

仓位管理（2万小资金专用）：
  - 最多同时持2只，单只仓位上限9000元
  - 单笔最大亏损=资金2%=400元 → 持仓市值×止损比例(2%)≈400 → 单只上限约2万
  - 股价过滤：只买100股金额在800~9000元的股票（即股价8~90元）
  - 最低成交额过滤：日均成交额>2000万（确保流动性，避免买入后卖不出）
  - 5日内最多亏损3次强制停手（连续止损说明市场不适合该策略）

标的池：沪深主板+创业板，排除ST/停牌/次新股/科创板
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
        close_tax=0.001,         # 卖出印花税0.1%
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    # ---- 策略参数 ----
    g.take_profit = 0.05         # 止盈5%
    g.stop_loss = -0.02          # 止损2%
    g.max_hold = 2               # 最大持仓2只（2万资金的最优分配）
    g.cooldown_days = 3          # 同一只股票卖出后冷却3天
    g.scan_count = 500           # 每日扫描股票数（按成交额排序取前N）

    # ---- 仓位管理参数（2万小资金专用）----
    g.max_single_position = 9000     # 单只最大持仓金额（元）
    g.min_stock_price = 5            # 最低股价（排除低价股）
    g.max_stock_price = 80           # 最高股价（确保100股买得起）
    g.min_daily_money = 20000000     # 最低日均成交额2000万（流动性门槛）
    g.max_loss_count = 3             # 5日内最大止损次数（触发后暂停买入）
    g.loss_pause_window = 5          # 止损统计窗口（天）
    g.capital_reserve = 0.03         # 保留3%现金（约600元，覆盖手续费）

    g.sell_history = {}              # 记录卖出日期，用于冷却
    g.loss_history = []              # 记录止损日期，用于连亏暂停

    run_daily(check_sell, time='09:31')
    run_daily(check_buy, time='09:35')
    run_daily(check_sell_intraday, time='14:50')


def get_prev_trade_date(context):
    today = context.current_dt.date()
    return get_trade_days(end_date=today, count=2)[0]


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
#  V6.0信号计算（完整移植）
# ============================================================
def calc_signals(code, end_date, count=60):
    """计算V6.0买卖信号，返回最新一天的信号"""
    df = get_price(code, end_date=end_date, count=count,
                   frequency='daily',
                   fields=['open', 'close', 'high', 'low', 'volume'],
                   skip_paused=True, fq='pre')

    if df is None or len(df) < 30:
        return None

    O = df['open']
    C = df['close']
    H = df['high']
    L = df['low']
    V = df['volume']

    # ====== 基础指标 ======
    MA5 = C.rolling(5).mean()
    MA10 = C.rolling(10).mean()
    MA20 = C.rolling(20).mean()
    MA60 = C.rolling(60).mean() if len(C) >= 60 else C.rolling(len(C)).mean()

    # KDJ(9,3,3)
    low9 = L.rolling(9).min()
    high9 = H.rolling(9).max()
    RSV = (C - low9) / (high9 - low9).replace(0, np.nan) * 100
    RSV = RSV.fillna(50)
    K0 = calc_sma(RSV, 3, 1)
    D0 = calc_sma(K0, 3, 1)
    J0 = 3 * K0 - 2 * D0

    # RSI(6)
    LC = C.shift(1)
    diff_c = C - LC
    pos = diff_c.clip(lower=0)
    abs_diff = diff_c.abs()
    sma_pos = calc_sma(pos.fillna(0), 6, 1)
    sma_abs = calc_sma(abs_diff.fillna(0), 6, 1)
    RSI6 = sma_pos / sma_abs.replace(0, np.nan) * 100
    RSI6 = RSI6.fillna(50)

    # CCI(14)
    TYP0 = (H + L + C) / 3
    CCI0 = (TYP0 - TYP0.rolling(14).mean()) / (0.015 * TYP0.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
    CCI0 = CCI0.fillna(0)

    # 量比
    V5 = V.rolling(5).mean()
    VR = V / V5.shift(1).replace(0, np.nan)
    VR = VR.fillna(1)

    # K线形态
    实体 = (C - O).abs()
    上影 = H - pd.concat([C, O], axis=1).max(axis=1)
    下影 = pd.concat([C, O], axis=1).min(axis=1) - L
    阳线 = C >= O
    阴线 = C <= O

    # BIAS
    BIAS20 = (C - MA20) / MA20 * 100

    # ====== 趋势 ======
    趋势分 = (
        (C > MA20).astype(int)
        + (MA20 > MA20.shift(5)).astype(int)
        + (C > MA60).astype(int)
        + (MA20 > MA60).astype(int)
    )

    # ====== 拉伸度 ======
    拉伸 = ((RSI6 - 50) + (J0 - 50) + CCI0 / 4 + BIAS20 * 5) / 4

    # ====== 卖出信号（5个条件）======
    SE1 = (C < MA10) & (C.shift(1) >= MA10.shift(1))
    SE2 = (C < C.shift(1) * 0.97) & (VR > 1.5)
    SE3 = (上影 > 实体 * 2) & (上影 > 下影 * 2) & (拉伸 > 10)
    SE4 = C < H.rolling(5).max() * 0.92
    SE5 = (J0.shift(1) > 90) & 阴线 & (实体 > 0)

    卖分 = (SE1.astype(float) + SE2.astype(float) + SE3.astype(float)
            + SE4.astype(float) + SE5.astype(float))

    # ====== 买入信号（5个条件）======
    BU1 = (C > MA10) & (C.shift(1) <= MA10.shift(1))
    BU2 = ((J0 < 20) | (RSI6 < 30)) & 阳线 & (实体 > 0)
    BU3 = (下影 > 实体 * 2) & (下影 > 上影 * 2) & (拉伸 < 10)
    BU4 = (MA20 > MA20.shift(1)) & (L <= MA20 * 1.02) & (C > MA20) & 阳线
    BU5 = (J0.shift(1) < 10) & 阳线 & (实体 > 0)

    买分 = (BU1.astype(float) + BU2.astype(float) + BU3.astype(float)
            + BU4.astype(float) + BU5.astype(float))

    # 冷却：4日内最多1次
    买冷却 = 买分.rolling(4).apply(lambda x: (x >= 1).sum(), raw=True) <= 1

    idx = -1
    return {
        'code': code,
        'close': C.iloc[idx],
        '买分': 买分.iloc[idx],
        '卖分': 卖分.iloc[idx],
        '趋势分': 趋势分.iloc[idx],
        '买冷却': 买冷却.iloc[idx] if not pd.isna(买冷却.iloc[idx]) else True,
        'BU': [BU1.iloc[idx], BU2.iloc[idx], BU3.iloc[idx], BU4.iloc[idx], BU5.iloc[idx]],
    }


# ============================================================
#  获取候选股票池
# ============================================================
def get_stock_pool(context):
    """获取候选池：主板+创业板，排除科创板/ST/停牌/次新/价格不合适/流动性差"""
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)

    # 全A股
    all_stocks = get_all_securities(types=['stock'], date=today).index.tolist()

    # 排除科创板（688开头）—— 只保留主板+创业板
    all_stocks = [s for s in all_stocks if not s.startswith('688')]

    # 排除ST
    st_info = get_extras('is_st', all_stocks, start_date=prev_date, end_date=prev_date, df=True)
    if len(st_info) > 0:
        st_stocks = st_info.columns[st_info.iloc[-1] == True].tolist()
        all_stocks = [s for s in all_stocks if s not in st_stocks]

    # 排除次新股（上市不满90天，次新波动大且无充足历史数据）
    all_stocks = [s for s in all_stocks if
                  (today - get_security_info(s).start_date).days > 90]

    # 排除停牌
    current_data = get_current_data()
    all_stocks = [s for s in all_stocks if not current_data[s].paused]

    # 股价过滤：100股金额必须在可接受范围内
    # 太贵买不起1手，太便宜大概率是问题股
    all_stocks = [s for s in all_stocks
                  if g.min_stock_price <= current_data[s].last_price <= g.max_stock_price]

    # 获取前5日平均成交额，过滤流动性差的 + 按成交额排序
    price_data = get_price(all_stocks, end_date=prev_date, count=5,
                           frequency='daily', fields=['money'],
                           skip_paused=True, fq='pre', panel=False)
    if price_data is not None and len(price_data) > 0:
        avg_money = price_data.groupby('code')['money'].mean()
        # 过滤：日均成交额 >= 2000万
        liquid_stocks = avg_money[avg_money >= g.min_daily_money]
        # 按成交额降序，取前N只
        liquid_stocks = liquid_stocks.sort_values(ascending=False)
        all_stocks = liquid_stocks.index[:g.scan_count].tolist()

    return all_stocks


# ============================================================
#  卖出检查：止盈5%/止损2%
# ============================================================
def check_sell(context):
    """检查持仓，达到止盈止损条件立即卖出"""
    today = context.current_dt.date()
    current_data = get_current_data()

    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount <= 0:
            continue
        if current_data[code].paused:
            continue

        cur_price = current_data[code].last_price
        cost = pos.avg_cost
        profit_pct = (cur_price - cost) / cost
        profit_amt = (cur_price - cost) * pos.total_amount

        # 止盈5%
        if profit_pct >= g.take_profit:
            log.info('[止盈] %s 成本:%.2f 现价:%.2f 盈利:%.1f%% +%.0f元' % (
                code, cost, cur_price, profit_pct * 100, profit_amt))
            order_target(code, 0)
            record_sell(code, today)
            continue

        # 止损2%
        if profit_pct <= g.stop_loss:
            log.info('[止损] %s 成本:%.2f 现价:%.2f 亏损:%.1f%% %.0f元' % (
                code, cost, cur_price, profit_pct * 100, profit_amt))
            order_target(code, 0)
            record_sell(code, today)
            record_loss(today)
            continue


def check_sell_intraday(context):
    """尾盘再检查一次止盈止损（防止盘中波动触发不到）"""
    check_sell(context)


# ============================================================
#  买入检查：扫描全市场强买信号
# ============================================================
def check_buy(context):
    """扫描股票池，寻找强买信号"""
    today = context.current_dt.date()
    prev_date = get_prev_trade_date(context)
    total_value = context.portfolio.total_value

    # ---- 连亏保护：5日内止损>=3次，暂停买入 ----
    if is_loss_paused(today):
        log.info('[暂停买入] %d日内止损达%d次，等待市场恢复' % (
            g.loss_pause_window, g.max_loss_count))
        return

    # 检查是否还有空仓位
    hold_count = len([c for c in context.portfolio.positions
                      if context.portfolio.positions[c].total_amount > 0])
    if hold_count >= g.max_hold:
        return

    # 预留现金（手续费缓冲）
    reserve = total_value * g.capital_reserve
    available = context.portfolio.available_cash - reserve
    if available < 1000:
        return

    # 获取候选池
    stock_pool = get_stock_pool(context)

    # 扫描信号
    buy_candidates = []
    for code in stock_pool:
        # 已持仓的跳过
        if code in context.portfolio.positions and \
                context.portfolio.positions[code].total_amount > 0:
            continue

        # 冷却期内跳过
        if is_in_cooldown(code, today):
            continue

        # 计算信号
        sig = calc_signals(code, prev_date, count=60)
        if sig is None:
            continue

        # 强买条件：买分>=2 且 无卖出信号 且 通过冷却
        if sig['买分'] >= 2 and sig['卖分'] == 0 and sig['买冷却']:
            buy_candidates.append(sig)

    if not buy_candidates:
        return

    # 按买分排序，买分相同则按趋势分排
    buy_candidates.sort(key=lambda x: (x['买分'], x['趋势分']), reverse=True)

    # ---- 仓位计算 ----
    slots = g.max_hold - hold_count

    for sig in buy_candidates[:slots]:
        code = sig['code']
        price = sig['close']

        # 重新计算可用现金（每次买入后更新）
        available = context.portfolio.available_cash - reserve
        if available < 1000:
            break

        # 单只仓位 = min(等额分配, 单只上限)
        per_stock_cash = available / slots
        position_cash = min(per_stock_cash, g.max_single_position)

        # 计算股数（向下取整到100股）
        shares = int(position_cash / price / 100) * 100
        if shares < 100:
            continue

        # 预估买入金额
        buy_amount = shares * price
        # 预估最大亏损 = 买入金额 × 止损比例
        max_loss = buy_amount * abs(g.stop_loss)

        bu_details = sig['BU']
        bu_names = ['MA10突破', '超卖阳线', '锤子线', '回踩MA20', 'J值首阳']
        triggered = [bu_names[i] for i in range(5) if bu_details[i]]

        log.info('[强买] %s 买分=%.0f 趋势=%.0f %d股 @%.2f 金额:%.0f 最大亏损:%.0f 条件:%s' % (
            code, sig['买分'], sig['趋势分'], shares, price,
            buy_amount, max_loss, '+'.join(triggered)))

        order(code, shares)
        slots -= 1


# ============================================================
#  冷却机制 + 连亏保护
# ============================================================
def is_in_cooldown(code, today):
    if code not in g.sell_history:
        return False
    last_sell = g.sell_history[code]
    return (today - last_sell).days <= g.cooldown_days


def record_sell(code, today):
    g.sell_history[code] = today
    # 清理超过30天的记录
    g.sell_history = {k: v for k, v in g.sell_history.items()
                      if (today - v).days <= 30}


def record_loss(today):
    """记录一次止损"""
    g.loss_history.append(today)
    # 只保留近期记录
    g.loss_history = [d for d in g.loss_history
                      if (today - d).days <= g.loss_pause_window]


def is_loss_paused(today):
    """5日内止损次数>=3次则暂停买入"""
    recent = [d for d in g.loss_history
              if (today - d).days <= g.loss_pause_window]
    return len(recent) >= g.max_loss_count
