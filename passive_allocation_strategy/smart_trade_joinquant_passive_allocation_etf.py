# -*- coding: utf-8 -*-
"""
被动配置 + 再平衡 策略 JoinQuant 版 v1
======================================
不预测涨跌，用股权+黄金固定比例控制风险，债 = 现金（货基/逆回购），靠定期再平衡机械高抛低吸。

池与目标权重（6 只场内 ETF + 35% 现金，可配置）：
  红利 510880 20% | 沪深300 510300 10% | 创业板 159915 5%
  纳指 513100 15% | 标普500 513500 10% | 黄金 518880 5% | 债(现金) 35%

再平衡（二选一触发）：
  1. 每季度第一个交易日
  2. 任一资产（含现金）实际权重偏离目标 > 5%（绝对）

执行：先按比例算目标股数、差值向零取整到整手，order_target 调仓，先卖超配、再买低配；债部分持有现金、不下单。

债=现金的原因：国债 ETF(511010) 一手约 1.1 万，2 万小资金买不进（35%=7000 元 < 一手），
且国债 ETF 会随利率波动下跌，不是保本。改用货基/逆回购（场外、几乎不跌、1元起购）。
本地引擎把现金建模为年化 2%（货基收益）；聚宽现金不产息，因此聚宽回测比本地略低
（约 2% × 35% × 年数），属已知口径差异，不影响结构验证。

与本地回测（passive_allocation_strategy/local/engine.py）策略逻辑一致：
  季度触发 + 阈值触发（含现金口径）+ 万三/最低5元佣金 + 先卖后买。
已知执行模型差异（非策略逻辑差异）：
  - 本地现金计息 2%/年；聚宽现金 0%
  - 滑点：本地与聚宽均为 0.1%（已对齐）
  - 本地引擎整手（100股）；聚宽整手（100股）
  - 本地引擎按收盘价；聚宽 14:50 近收盘执行
"""

from jqdata import *

TARGET_WEIGHTS = {
    "510880.XSHG": 0.20,  # 上证红利
    "510300.XSHG": 0.10,  # 沪深300
    "159915.XSHE": 0.05,  # 创业板
    "513100.XSHG": 0.15,  # 纳指100
    "513500.XSHG": 0.10,  # 标普500
    "518880.XSHG": 0.05,  # 黄金
}

BOND_WEIGHT = 0.35  # 债 = 现金（货基/逆回购），持有现金不下单

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
    g.bond_weight = BOND_WEIGHT
    g.threshold = REBALANCE_THRESHOLD
    g.initialized = False
    g.rebalance_count = 0

    run_daily(check_rebalance, time="14:50")


def _is_quarter_start(context):
    today = context.current_dt.date()
    if today.month not in (1, 4, 7, 10):
        return False
    days = get_trade_days(end_date=today, count=2)
    if len(days) < 2:
        return True
    return days[0].month != today.month


def _current_weights(context):
    total = context.portfolio.total_value
    weights = {}
    for code in g.target_weights:
        pos = context.portfolio.positions.get(code)
        if pos is not None and pos.total_amount > 0:
            weights[code] = pos.total_amount * pos.price / total if total > 0 else 0.0
        else:
            weights[code] = 0.0
    weights["CASH"] = context.portfolio.available_cash / total if total > 0 else 0.0
    return weights


def _threshold_hit(weights):
    for code, target in g.target_weights.items():
        if abs(weights.get(code, 0.0) - target) > g.threshold:
            return True
    if abs(weights.get("CASH", 0.0) - g.bond_weight) > g.threshold:
        return True
    return False


def _target_shares(context, code, total):
    """目标股数：按比例算目标股数，与当前持仓的差向零取整到整手；
    差 < 1 手返回当前持仓（即无需下单）。停牌/无价返回 None。"""
    price = get_current_data()[code].last_price
    if price is None or price <= 0:
        return None
    pos = context.portfolio.positions.get(code)
    current = pos.total_amount if pos is not None else 0
    delta_float = total * g.target_weights[code] / price - current
    delta = int(abs(delta_float) // 100) * 100
    if delta_float < 0:
        delta = -delta
    return current + delta


def check_rebalance(context):
    weights = _current_weights(context)
    if not g.initialized:
        reason = "初始建仓"
    elif _is_quarter_start(context):
        reason = "季度"
    elif _threshold_hit(weights):
        reason = "阈值"
    else:
        return

    total = context.portfolio.total_value
    log.info("[再平衡#%d] %s | 触发:%s | 总值:%.0f" % (
        g.rebalance_count + 1, context.current_dt.date(), reason, total))
    for code in g.target_weights:
        log.info("  %s 调前%.1f%% -> 目标%.1f%%" % (
            code, weights.get(code, 0.0) * 100, g.target_weights[code] * 100))
    log.info("  现金 调前%.1f%% -> 目标%.1f%%" % (weights.get("CASH", 0.0) * 100, g.bond_weight * 100))

    positions = context.portfolio.positions
    # 先卖超配（释放现金），再买低配，与本地引擎 _rebalance 顺序一致。
    # 目标股数已修正为整手；修正后与当前持仓一致（差 < 1 手）则不下单，
    # 绝不发出会被平台拒绝的 < 1 手订单。
    for code, target in g.target_weights.items():
        tgt = _target_shares(context, code, total)
        if tgt is None:
            continue
        pos = positions.get(code)
        current = pos.total_amount if pos is not None else 0
        if tgt < current:
            log.info("  [卖] %s %d -> %d 股" % (code, current, tgt))
            order_target(code, tgt)
    for code, target in g.target_weights.items():
        tgt = _target_shares(context, code, total)
        if tgt is None:
            continue
        pos = positions.get(code)
        current = pos.total_amount if pos is not None else 0
        if tgt > current:
            log.info("  [买] %s %d -> %d 股" % (code, current, tgt))
            order_target(code, tgt)

    g.rebalance_count += 1
    g.initialized = True
