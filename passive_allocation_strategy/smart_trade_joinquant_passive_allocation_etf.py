# -*- coding: utf-8 -*-
"""
被动配置 + 再平衡 策略 JoinQuant 版 v1
======================================
不预测涨跌，用股债黄金固定比例控制风险，靠定期再平衡机械高抛低吸。

池与目标权重（6 只，可配置）：
  红利 510880 20% | 沪深300 510300 15% | 创业板 159915 10%
  纳指 513100 15% | 黄金 518880 5% | 国债 511010 35%

再平衡（二选一触发）：
  1. 每季度第一个交易日
  2. 任一资产实际权重偏离目标 > 5%（绝对）

执行：order_target_percent 调到目标权重。
"""

from jqdata import *

TARGET_WEIGHTS = {
    "510300.XSHG": 0.15,  # 沪深300
    "510880.XSHG": 0.20,  # 红利
    "159915.XSHE": 0.10,  # 创业板
    "513100.XSHG": 0.15,  # 纳指
    "518880.XSHG": 0.05,  # 黄金
    "511010.XSHG": 0.35,  # 国债
}

REBALANCE_THRESHOLD = 0.05  # 偏离 5% 触发


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type="stock")

    g.target_weights = dict(TARGET_WEIGHTS)
    g.threshold = REBALANCE_THRESHOLD
    g.initialized = False

    run_daily(check_rebalance, time="14:50")


def _is_quarter_start(context):
    today = context.current_dt.date()
    if today.month not in (1, 4, 7, 10):
        return False
    days = get_trade_days(end_date=today, count=2)
    if len(days) < 2:
        return True
    return days[0].month != today.month


def _threshold_hit(context):
    total = context.portfolio.total_value
    if total <= 0:
        return False
    for code, target in g.target_weights.items():
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            if target > 0.05:
                return True
            continue
        current = pos.total_amount * pos.price / total
        if abs(current - target) > g.threshold:
            return True
    return False


def check_rebalance(context):
    if g.initialized and not (_is_quarter_start(context) or _threshold_hit(context)):
        return

    total = context.portfolio.total_value
    for code, target in g.target_weights.items():
        order_target_percent(code, target)

    g.initialized = True
    log.info("[再平衡] %s 总值%.0f 季度=%s" % (
        context.current_dt.date(), total, _is_quarter_start(context)))
