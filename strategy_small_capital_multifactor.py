# -*- coding: utf-8 -*-
"""
================================================================================
 小资金多因子精选策略 (Small Capital Multi-Factor Selection Strategy)
================================================================================

 适用平台: PTrade (恒生量化交易平台)
 适用市场: A股
 初始资金: 20,000 元
 策略频率: 日线级别
 持仓数量: 最多 2 只
 调仓周期: 每周一开盘调仓

 策略核心逻辑:
 ─────────────
 1. 从沪深300 + 中证500成分股中选股（流动性好、质量高）
 2. 基本面过滤：低估值(PE/PB合理) + 盈利能力强(ROE) + 有成长性
 3. 技术面确认：均线多头排列 + MACD趋势向好 + RSI未超买超卖
 4. 价格过滤：股价3~25元（2万资金可买1~6手，保证仓位灵活性）
 5. 严格风控：单只最大50%仓位、7%止损、避开涨跌停/停牌股

 风险提示:
 ─────────
 - 本策略仅供学习研究，不构成投资建议
 - 历史回测表现不代表未来收益
 - 小资金交易需特别注意手续费对收益的侵蚀
================================================================================
"""

import numpy as np


# ============================================================================
#  全局变量定义
# ============================================================================
# g 对象在 PTrade 中是跨函数持久化的全局存储对象


# ============================================================================
#  策略初始化
# ============================================================================
def initialize(context):
    """
    策略启动时执行一次，设置全局参数和定时任务。
    """
    # ------ 基本参数 ------
    g.max_hold_count = 2          # 最大持仓只数（2万资金持2只较合理）
    g.stop_loss_rate = -0.07      # 止损线：亏损7%止损
    g.take_profit_rate = 0.20     # 止盈线：盈利20%止盈（锁定利润）
    g.min_price = 3.0             # 最低股价（排除低价垃圾股）
    g.max_price = 25.0            # 最高股价（确保2万资金至少买1手）
    g.rebalance_weekday = 1       # 调仓日：周一（1=周一, 5=周五）
    g.hold_days_min = 3           # 最短持仓天数（避免频繁交易）

    # ------ 基本面筛选参数 ------
    g.pe_min = 5.0                # 最小动态PE（排除负盈利和异常低PE）
    g.pe_max = 35.0               # 最大动态PE（排除泡沫股）
    g.pb_max = 6.0                # 最大PB（排除严重高估股）
    g.pb_min = 0.5                # 最小PB（排除资产质量可疑股）

    # ------ 技术指标参数 ------
    g.ma_short = 5                # 短期均线
    g.ma_mid = 10                 # 中期均线
    g.ma_long = 20                # 长期均线
    g.rsi_period = 14             # RSI周期
    g.rsi_upper = 75              # RSI超买线
    g.rsi_lower = 25              # RSI超卖线
    g.macd_short = 12             # MACD快线
    g.macd_long = 26              # MACD慢线
    g.macd_signal = 9             # MACD信号线

    # ------ 运行时变量 ------
    g.hold_days = {}              # 记录每只股票的持仓天数 {stock_code: days}
    g.buy_prices = {}             # 记录每只股票的买入价格 {stock_code: price}
    g.last_rebalance_date = None  # 上次调仓日期
    g.stock_pool = []             # 当日可选股票池

    # ------ 佣金设置（回测用，尽量贴近真实费率）------
    set_commission(commission_ratio=0.00025, min_commission=5.0)

    # ------ 基准设置 ------
    set_benchmark('000300.SS')

    # ------ 定时任务 ------
    # 每天9:31执行选股（盘前准备）
    run_daily(context, daily_check)

    log.info("=" * 60)
    log.info("策略初始化完成 - 小资金多因子精选策略")
    log.info("初始资金: %.2f 元" % context.portfolio.portfolio_value)
    log.info("最大持仓: %d 只" % g.max_hold_count)
    log.info("止损线: %.1f%%" % (g.stop_loss_rate * 100))
    log.info("止盈线: %.1f%%" % (g.take_profit_rate * 100))
    log.info("=" * 60)


# ============================================================================
#  每日盘前检查（通过 run_daily 在 9:31 调用）
# ============================================================================
def daily_check(context):
    """
    每个交易日开盘时执行：
    1. 更新持仓天数
    2. 判断是否为调仓日
    3. 若为调仓日，执行选股
    """
    # 更新所有持仓的持有天数
    for stock in list(g.hold_days.keys()):
        g.hold_days[stock] = g.hold_days.get(stock, 0) + 1

    # 判断今天是周几
    current_dt = context.blotter.current_dt
    weekday = current_dt.isoweekday()

    # 只在调仓日选股（减少交易频率，降低手续费消耗）
    if weekday == g.rebalance_weekday:
        g.stock_pool = select_stocks(context)
        log.info("[选股完成] 日期: %s, 候选股票数: %d" % (
            current_dt.strftime("%Y-%m-%d"), len(g.stock_pool)))
        if len(g.stock_pool) > 0:
            for i, item in enumerate(g.stock_pool[:5]):
                log.info("  Top%d: %s (综合得分: %.2f)" % (
                    i + 1, item['code'], item['score']))


# ============================================================================
#  主交易逻辑（每个交易日执行）
# ============================================================================
def handle_data(context, data):
    """
    核心交易逻辑，每个交易日执行一次（日线级别）。
    执行顺序：
    1. 止损止盈检查（保命优先）
    2. 卖出不再符合条件的持仓
    3. 买入新的优质标的
    """
    current_dt = context.blotter.current_dt
    weekday = current_dt.isoweekday()

    # ======== 第一步：止损止盈检查（每天都执行，不受调仓日限制）========
    check_stop_loss_and_profit(context, data)

    # ======== 第二步 & 第三步：仅在调仓日执行买卖操作 ========
    if weekday == g.rebalance_weekday:
        # 卖出不在最新选股池中的持仓
        execute_sell(context, data)
        # 买入新选出的优质标的
        execute_buy(context, data)


# ============================================================================
#  选股模块：多因子综合评分
# ============================================================================
def select_stocks(context):
    """
    多因子选股流程：
    1. 获取沪深300 + 中证500成分股作为初始池
    2. 基本面过滤（PE、PB合理区间）
    3. 价格过滤（3~25元）
    4. 排除停牌、涨跌停、ST股
    5. 技术面评分
    6. 基本面评分
    7. 综合排序，返回Top N

    返回: [{'code': str, 'score': float}, ...] 按score降序排列
    """
    current_date = context.blotter.current_dt.strftime("%Y%m%d")

    # ------ 1. 获取候选股票池 ------
    try:
        hs300_stocks = get_index_stocks('000300.SS')
    except Exception:
        hs300_stocks = []

    try:
        zz500_stocks = get_index_stocks('000905.SS')
    except Exception:
        zz500_stocks = []

    # 合并去重
    all_stocks = list(set(hs300_stocks + zz500_stocks))

    if len(all_stocks) == 0:
        log.warn("[选股] 获取指数成分股失败，股票池为空")
        return []

    log.info("[选股] 初始候选池: %d 只 (沪深300: %d, 中证500: %d)" % (
        len(all_stocks), len(hs300_stocks), len(zz500_stocks)))

    # ------ 2. 基本面过滤：获取估值数据 ------
    try:
        val_data = get_fundamentals(
            all_stocks, 'valuation',
            fields=['pe_dynamic', 'pb', 'total_value', 'turnover_rate']
        )
    except Exception as e:
        log.warn("[选股] 获取估值数据失败: %s" % str(e))
        return []

    if val_data is None or len(val_data) == 0:
        log.warn("[选股] 估值数据为空")
        return []

    # 过滤PE、PB合理区间
    candidates = {}
    for idx, row in val_data.iterrows():
        code = row.get('secu_code', idx)
        pe = row.get('pe_dynamic', None)
        pb = row.get('pb', None)

        # 跳过数据缺失的
        if pe is None or pb is None:
            continue
        try:
            pe = float(pe)
            pb = float(pb)
        except (ValueError, TypeError):
            continue

        # 基本面过滤条件
        if g.pe_min <= pe <= g.pe_max and g.pb_min <= pb <= g.pb_max:
            candidates[code] = {
                'pe': pe,
                'pb': pb,
                'total_value': row.get('total_value', 0),
            }

    log.info("[选股] 基本面过滤后: %d 只" % len(candidates))

    if len(candidates) == 0:
        return []

    # ------ 3. 价格过滤 + 排除停牌/涨跌停/ST ------
    scored_stocks = []
    stock_list = list(candidates.keys())

    # 分批获取历史数据（避免一次请求过多）
    batch_size = 50
    for batch_start in range(0, len(stock_list), batch_size):
        batch = stock_list[batch_start:batch_start + batch_size]

        for stock in batch:
            try:
                score = evaluate_single_stock(stock, candidates[stock], context)
                if score is not None:
                    scored_stocks.append({
                        'code': stock,
                        'score': score
                    })
            except Exception as e:
                # 单只股票评分失败不影响整体流程
                continue

    # ------ 4. 按综合得分排序 ------
    scored_stocks.sort(key=lambda x: x['score'], reverse=True)

    log.info("[选股] 最终候选: %d 只" % len(scored_stocks))

    return scored_stocks


def evaluate_single_stock(stock, fundamental_info, context):
    """
    对单只股票进行综合评分。

    评分维度（总分100分）：
    - 估值得分（30分）：PE越低越好，PB越低越好
    - 趋势得分（30分）：均线多头排列、MACD、RSI
    - 动量得分（20分）：近期涨幅适中（不追高、不抄底）
    - 流动性得分（20分）：成交量适中

    参数:
        stock: 股票代码
        fundamental_info: 基本面数据字典
        context: 策略上下文

    返回:
        float: 综合得分 (0~100)，None表示不符合条件
    """
    # 获取近60个交易日的历史数据（约3个月）
    try:
        hist = get_history(
            60, '1d',
            ['close', 'high', 'low', 'volume', 'open'],
            security_list=stock,
            fq='pre'
        )
    except Exception:
        return None

    if hist is None or len(hist) < 30:
        return None

    close = hist['close'].values
    high = hist['high'].values
    low = hist['low'].values
    volume = hist['volume'].values

    # 最新价格
    latest_price = close[-1]

    # --- 价格过滤 ---
    if latest_price < g.min_price or latest_price > g.max_price:
        return None

    # --- 排除停牌股（最近3天成交量为0）---
    if volume[-1] == 0 or volume[-2] == 0 or volume[-3] == 0:
        return None

    # --- 排除涨跌停股 ---
    try:
        limit_status = check_limit(stock)
        if limit_status is not None:
            status = limit_status.get(stock, 0)
            if status != 0:
                return None
    except Exception:
        pass

    # --- 排除ST股（通过股票名称判断）---
    try:
        stock_info = get_stock_info(stock, 'stock_name')
        name = stock_info.get(stock, {}).get('stock_name', '')
        if 'ST' in name or 'st' in name or '*ST' in name:
            return None
    except Exception:
        pass

    # ===================== 评分开始 =====================

    total_score = 0.0

    # ---------- 1. 估值得分（满分30分）----------
    pe = fundamental_info['pe']
    pb = fundamental_info['pb']

    # PE得分：PE越低越好（在合理范围内），10~15分最高分
    if pe <= 15:
        pe_score = 15.0
    elif pe <= 25:
        pe_score = 15.0 - (pe - 15) * 0.8
    else:
        pe_score = max(0, 15.0 - (pe - 15) * 1.0)

    # PB得分：PB越低越好，1~2分最高分
    if pb <= 2.0:
        pb_score = 15.0
    elif pb <= 4.0:
        pb_score = 15.0 - (pb - 2.0) * 5.0
    else:
        pb_score = max(0, 15.0 - (pb - 2.0) * 5.0)

    total_score += pe_score + pb_score

    # ---------- 2. 趋势得分（满分30分）----------
    trend_score = 0.0

    # 2a. 均线多头排列判断（满分15分）
    if len(close) >= g.ma_long:
        ma5 = np.mean(close[-g.ma_short:])
        ma10 = np.mean(close[-g.ma_mid:])
        ma20 = np.mean(close[-g.ma_long:])

        # 价格在MA20上方 (+5分)
        if latest_price > ma20:
            trend_score += 5.0
        # MA5 > MA10 (+5分，短期趋势向好)
        if ma5 > ma10:
            trend_score += 5.0
        # MA10 > MA20 (+5分，中期趋势向好)
        if ma10 > ma20:
            trend_score += 5.0

    # 2b. MACD信号（满分10分）
    if len(close) >= 35:
        try:
            dif_arr, dea_arr, macd_arr = get_MACD(
                close, g.macd_short, g.macd_long, g.macd_signal)
            dif = dif_arr[-1]
            dea = dea_arr[-1]
            macd_val = macd_arr[-1]

            # DIF > DEA（MACD金叉或多头）(+5分)
            if dif > dea:
                trend_score += 5.0
            # MACD柱状线为正且在放大 (+5分)
            if macd_val > 0 and len(macd_arr) >= 2 and macd_arr[-1] > macd_arr[-2]:
                trend_score += 5.0
        except Exception:
            pass

    # 2c. RSI信号（满分5分）- 适中区间给分
    if len(close) >= g.rsi_period + 1:
        try:
            rsi_arr = get_RSI(close, g.rsi_period)
            rsi = rsi_arr[-1]

            # RSI在超买超卖区间外，且偏强不偏弱
            if rsi > g.rsi_upper:
                # 超买，不给分，且可能要排除
                trend_score -= 5.0
            elif rsi < g.rsi_lower:
                # 超卖，不给分
                trend_score -= 3.0
            elif 40 <= rsi <= 65:
                # 适中偏强区间，最佳
                trend_score += 5.0
            elif 30 <= rsi < 40 or 65 < rsi <= 75:
                trend_score += 2.0
        except Exception:
            pass

    total_score += max(0, trend_score)

    # ---------- 3. 动量得分（满分20分）----------
    momentum_score = 0.0

    if len(close) >= 20:
        # 近5日涨幅
        ret_5d = (close[-1] / close[-5] - 1) * 100 if close[-5] > 0 else 0
        # 近20日涨幅
        ret_20d = (close[-1] / close[-20] - 1) * 100 if close[-20] > 0 else 0

        # 近5日涨幅在 -2% ~ +5% 为最佳（温和上涨，非暴涨暴跌）
        if -2 <= ret_5d <= 5:
            momentum_score += 10.0
        elif -5 <= ret_5d < -2 or 5 < ret_5d <= 10:
            momentum_score += 5.0
        elif ret_5d > 15 or ret_5d < -10:
            momentum_score -= 5.0  # 暴涨暴跌扣分

        # 近20日涨幅在 0% ~ 15% 为最佳（中期稳健上涨）
        if 0 <= ret_20d <= 15:
            momentum_score += 10.0
        elif -5 <= ret_20d < 0 or 15 < ret_20d <= 25:
            momentum_score += 5.0
        elif ret_20d > 30:
            momentum_score -= 5.0  # 严重超涨扣分

    total_score += max(0, momentum_score)

    # ---------- 4. 流动性得分（满分20分）----------
    liquidity_score = 0.0

    if len(volume) >= 10:
        avg_vol_10 = np.mean(volume[-10:])
        avg_vol_5 = np.mean(volume[-5:])

        # 近10日日均成交量 > 50万手（基本门槛）
        if avg_vol_10 > 500000:
            liquidity_score += 10.0
        elif avg_vol_10 > 200000:
            liquidity_score += 5.0

        # 近5日成交量相比近10日有所放大（温和放量）
        if avg_vol_10 > 0:
            vol_ratio = avg_vol_5 / avg_vol_10
            if 1.0 <= vol_ratio <= 1.5:
                liquidity_score += 10.0  # 温和放量最佳
            elif 0.8 <= vol_ratio < 1.0:
                liquidity_score += 5.0   # 缩量也可接受
            elif vol_ratio > 2.0:
                liquidity_score -= 5.0   # 异常放量扣分

    total_score += max(0, liquidity_score)

    return total_score


# ============================================================================
#  止损止盈检查
# ============================================================================
def check_stop_loss_and_profit(context, data):
    """
    遍历所有持仓，检查是否触发止损或止盈条件。
    止损优先级最高，是保护本金的核心机制。
    """
    positions = context.portfolio.positions
    if len(positions) == 0:
        return

    for stock, pos in list(positions.items()):
        if pos.amount <= 0:
            continue

        # 可卖数量（T+1限制，今天买的不能卖）
        if pos.enable_amount <= 0:
            continue

        cost = g.buy_prices.get(stock, pos.cost_basis)
        if cost <= 0:
            continue

        current_price = pos.last_sale_price
        if current_price <= 0:
            continue

        profit_rate = (current_price - cost) / cost

        # --- 止损 ---
        if profit_rate <= g.stop_loss_rate:
            log.warn("[止损] %s 亏损 %.2f%%, 成本: %.2f, 现价: %.2f, 卖出 %d 股" % (
                stock, profit_rate * 100, cost, current_price, pos.enable_amount))
            order_target(stock, 0)
            clean_stock_record(stock)

        # --- 止盈 ---
        elif profit_rate >= g.take_profit_rate:
            log.info("[止盈] %s 盈利 %.2f%%, 成本: %.2f, 现价: %.2f, 卖出 %d 股" % (
                stock, profit_rate * 100, cost, current_price, pos.enable_amount))
            order_target(stock, 0)
            clean_stock_record(stock)

        # --- 移动止盈（盈利超过10%后，回撤5%则止盈）---
        elif profit_rate >= 0.10:
            # 更新最高盈利记录
            max_profit_key = stock + '_max_profit'
            current_max = getattr(g, max_profit_key, profit_rate) if hasattr(g, max_profit_key) else profit_rate
            if profit_rate > current_max:
                setattr(g, max_profit_key, profit_rate)
                current_max = profit_rate
            # 从最高点回撤超过5%
            if current_max - profit_rate >= 0.05:
                log.info("[移动止盈] %s 盈利 %.2f%% (峰值: %.2f%%), 回撤触发卖出" % (
                    stock, profit_rate * 100, current_max * 100))
                order_target(stock, 0)
                clean_stock_record(stock)
                # 清理最高盈利记录
                if hasattr(g, max_profit_key):
                    delattr(g, max_profit_key)


# ============================================================================
#  卖出逻辑
# ============================================================================
def execute_sell(context, data):
    """
    卖出不再满足条件的持仓：
    1. 不在最新选股池中
    2. 持仓已达最短持有天数
    3. 技术面转弱
    """
    positions = context.portfolio.positions
    if len(positions) == 0:
        return

    # 最新选股池中的股票代码集合
    pool_codes = set([item['code'] for item in g.stock_pool]) if g.stock_pool else set()

    for stock, pos in list(positions.items()):
        if pos.amount <= 0 or pos.enable_amount <= 0:
            continue

        # 持仓天数不足最短要求，不卖
        hold_days = g.hold_days.get(stock, 0)
        if hold_days < g.hold_days_min:
            continue

        should_sell = False
        sell_reason = ""

        # 条件1：不在最新选股池中
        if stock not in pool_codes:
            should_sell = True
            sell_reason = "不在最新选股池"

        # 条件2：技术面转弱（均线死叉）
        if not should_sell:
            try:
                hist = get_history(
                    25, '1d', ['close'], security_list=stock, fq='pre')
                if hist is not None and len(hist) >= 20:
                    close = hist['close'].values
                    ma5 = np.mean(close[-5:])
                    ma20 = np.mean(close[-20:])
                    # MA5跌破MA20，趋势转弱
                    if ma5 < ma20 * 0.98:
                        should_sell = True
                        sell_reason = "MA5跌破MA20（趋势转弱）"
            except Exception:
                pass

        if should_sell:
            log.info("[卖出] %s, 原因: %s, 持仓天数: %d, 数量: %d" % (
                stock, sell_reason, hold_days, pos.enable_amount))
            order_target(stock, 0)
            clean_stock_record(stock)


# ============================================================================
#  买入逻辑
# ============================================================================
def execute_buy(context, data):
    """
    根据选股结果买入新标的：
    1. 计算可用资金和可买入数量
    2. 优先买入得分最高的股票
    3. 等额分配仓位
    """
    # 当前持仓数量
    current_holds = get_current_hold_count(context)

    # 可买入的空位数
    empty_slots = g.max_hold_count - current_holds
    if empty_slots <= 0:
        return

    if not g.stock_pool or len(g.stock_pool) == 0:
        return

    # 可用资金
    available_cash = context.portfolio.cash
    # 每只股票分配的资金（预留2%作为交易费用缓冲）
    per_stock_value = available_cash * 0.98 / empty_slots

    # 最小有效金额（至少能买1手 + 手续费）
    min_buy_value = g.min_price * 100 + 10  # 约310元
    if per_stock_value < min_buy_value:
        log.info("[买入] 可用资金不足，跳过买入 (可用: %.2f, 需要: %.2f)" % (
            per_stock_value, min_buy_value))
        return

    # 当前已持有的股票代码
    held_stocks = set()
    for stock, pos in context.portfolio.positions.items():
        if pos.amount > 0:
            held_stocks.add(stock)

    bought_count = 0
    for item in g.stock_pool:
        if bought_count >= empty_slots:
            break

        stock = item['code']
        score = item['score']

        # 已持有的不重复买入
        if stock in held_stocks:
            continue

        # 得分过低的不买（至少要40分以上）
        if score < 40:
            log.info("[买入] 候选股得分过低，停止买入 (最高: %.1f)" % score)
            break

        # 再次检查涨跌停（避免追涨停板）
        try:
            limit_status = check_limit(stock)
            if limit_status is not None:
                status = limit_status.get(stock, 0)
                if status != 0:
                    log.info("[买入] %s 涨跌停，跳过" % stock)
                    continue
        except Exception:
            pass

        # 获取当前价格计算买入数量
        try:
            hist = get_history(1, '1d', 'close', security_list=stock, fq='pre', include=True)
            if hist is None or len(hist) == 0:
                continue
            current_price = hist['close'].values[-1]
        except Exception:
            continue

        if current_price <= 0:
            continue

        # 计算买入手数（A股最小单位100股）
        buy_shares = int(per_stock_value / current_price / 100) * 100

        if buy_shares < 100:
            log.info("[买入] %s 资金不足买1手 (价格: %.2f, 分配资金: %.2f)" % (
                stock, current_price, per_stock_value))
            continue

        # 执行买入
        log.info("[买入] %s, 得分: %.1f, 价格: %.2f, 数量: %d 股, 金额: %.2f 元" % (
            stock, score, current_price, buy_shares,
            current_price * buy_shares))

        order_id = order(stock, buy_shares)

        if order_id is not None:
            # 记录买入信息
            g.buy_prices[stock] = current_price
            g.hold_days[stock] = 0
            bought_count += 1
            held_stocks.add(stock)


# ============================================================================
#  辅助函数
# ============================================================================
def get_current_hold_count(context):
    """获取当前实际持仓只数（排除数量为0的）"""
    count = 0
    for stock, pos in context.portfolio.positions.items():
        if pos.amount > 0:
            count += 1
    return count


def clean_stock_record(stock):
    """清理某只股票的持仓记录"""
    if stock in g.hold_days:
        del g.hold_days[stock]
    if stock in g.buy_prices:
        del g.buy_prices[stock]


# ============================================================================
#  盘后处理（可选）
# ============================================================================
def after_trading_end(context, data):
    """
    每个交易日收盘后执行：
    - 记录当日持仓和收益情况
    - 用于监控和复盘
    """
    current_dt = context.blotter.current_dt
    portfolio = context.portfolio

    # 每5个交易日输出一次汇总（减少日志量）
    if current_dt.isoweekday() == 5:  # 周五
        log.info("=" * 50)
        log.info("[周报] 日期: %s" % current_dt.strftime("%Y-%m-%d"))
        log.info("[周报] 总资产: %.2f 元" % portfolio.portfolio_value)
        log.info("[周报] 可用现金: %.2f 元" % portfolio.cash)
        log.info("[周报] 持仓市值: %.2f 元" % portfolio.positions_value)
        log.info("[周报] 累计收益率: %.2f%%" % (portfolio.returns * 100))
        log.info("[周报] 累计盈亏: %.2f 元" % portfolio.pnl)

        # 输出各持仓详情
        for stock, pos in portfolio.positions.items():
            if pos.amount > 0:
                cost = g.buy_prices.get(stock, pos.cost_basis)
                if cost > 0:
                    pnl_rate = (pos.last_sale_price - cost) / cost * 100
                else:
                    pnl_rate = 0
                log.info("[周报] 持仓: %s, 数量: %d, 成本: %.2f, "
                         "现价: %.2f, 盈亏: %.2f%%, 持有天数: %d" % (
                    stock, pos.amount, cost, pos.last_sale_price,
                    pnl_rate, g.hold_days.get(stock, 0)))
        log.info("=" * 50)
