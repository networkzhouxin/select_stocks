# -*- coding: utf-8 -*-
"""Read-only pre-check: frozen ATR-stress candidate on top of cross-v0.3.2.

Runs the approved 2019-2021 training replay twice with identical data and the
same local execution model:

  baseline   : official cross-v0.3.2 parameters (no stress keys)
  candidate  : baseline + the frozen portfolio ATR-stress rule
               (lookback 15 trading days, min 3 ATR stops, buy scale 0.50)

Purpose: confirm on the current mainline that the already-validated frozen
candidate still improves (or at least does not worsen) training return and
max drawdown, and that the rule actually triggers, before any JoinQuant run.

This is not a parameter search, does not modify any formal strategy file, and
JoinQuant remains the performance authority.

Note: the formal JoinQuant mainline now carries the stress keys itself
(cross-v0.3.3, 2026-08-16). The baseline replay below therefore strips the
three frozen stress keys explicitly so the pre/post comparison stays valid.
"""

import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy  # noqa: E402
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine  # noqa: E402
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader  # noqa: E402
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner  # noqa: E402
from cross_signal_strategy.local_training_run import (  # noqa: E402
    build_training_signal_adapter,
    get_training_trade_dates,
)

FROZEN_STRESS_KEYS = {
    "portfolio_atr_stress_lookback_days": 15,
    "portfolio_atr_stress_min_stops": 3,
    "portfolio_atr_stress_buy_scale": 0.50,
}

RECORDED_BASELINE = {
    "total_return": 1.2061,
    "max_drawdown": 0.0747,
    "buy_count": 92,
    "sell_count": 89,
}


class RecordingPlanner(LocalCrossSignalOrderPlanner):
    """Records the daily stress scale and planned stress-buy targets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_stress_scale = {}
        self.planned_stress_buys = []

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        scale = self._portfolio_atr_stress_buy_scale(str(current_date))
        self.daily_stress_scale[str(current_date)] = scale
        orders = super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices
        )
        if scale < 1.0:
            for order in orders:
                if float(order["target_value"]) > 0.0:
                    self.planned_stress_buys.append((
                        str(current_date),
                        str(order["code"]),
                        float(order["target_value"]),
                    ))
        return orders


def _daily_returns(values, initial_cash):
    returns = []
    previous = initial_cash
    for value in values:
        returns.append(value / previous - 1.0)
        previous = value
    return returns


def _max_drawdown(values):
    peak = None
    max_dd = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def _sharpe_sortino(daily_returns):
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
    std = math.sqrt(variance)
    downside = [
        r for r in daily_returns
        if r < 0
    ]
    downside_var = (
        sum(r * r for r in downside) / len(daily_returns)
        if downside else 0.0
    )
    sharpe = mean / std * math.sqrt(244) if std > 0 else 0.0
    sortino = (
        mean / math.sqrt(downside_var) * math.sqrt(244)
        if downside_var > 0 else 0.0
    )
    return sharpe, sortino


def _annual_returns(dates, values, initial_cash):
    result = {}
    boundaries = {
        "2019": ("2019-01-01", "2019-12-31"),
        "2020": ("2020-01-01", "2020-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
    }
    for year, (start_text, end_text) in boundaries.items():
        in_year = [
            value for date, value in zip(dates, values)
            if start_text <= str(date) <= end_text
        ]
        if not in_year:
            continue
        year_start = initial_cash if year == "2019" else (
            values[len([d for d in dates if str(d) < start_text]) - 1]
            if any(str(d) < start_text for d in dates) else in_year[0]
        )
        result[year] = in_year[-1] / year_start - 1.0
    return result


def _replay(loader, trade_dates, params, initial_cash):
    adapter = build_training_signal_adapter(loader)
    planner = RecordingPlanner(adapter, params=params, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)

    values = [day.total_value for day in results]
    daily_returns = _daily_returns(values, initial_cash)
    sharpe, sortino = _sharpe_sortino(daily_returns)
    buy_count = 0
    sell_count = 0
    unfilled = 0
    stress_fills = []
    planned_by_key = {(d, c): v for d, c, v in planner.planned_stress_buys}
    for day in results:
        stress_active = planner.daily_stress_scale.get(str(day.date), 1.0) < 1.0
        for order in day.orders:
            if not order.filled:
                unfilled += 1
                continue
            if order.amount_delta > 0:
                buy_count += 1
                if stress_active:
                    stress_fills.append({
                        "date": str(day.date),
                        "code": str(order.code),
                        "shares": int(order.amount_delta),
                        "planned_target": planned_by_key.get(
                            (str(day.date), str(order.code))),
                    })
            elif order.amount_delta < 0:
                sell_count += 1

    annual = _annual_returns(
        [str(day.date) for day in results], values, initial_cash)
    n_days = len(values)
    annualized = (
        (values[-1] / initial_cash) ** (244.0 / n_days) - 1.0
        if n_days > 0 and values[-1] > 0 else 0.0
    )
    return {
        "end_value": values[-1],
        "total_return": values[-1] / initial_cash - 1.0,
        "annualized": annualized,
        "max_drawdown": _max_drawdown(values),
        "sharpe": sharpe,
        "sortino": sortino,
        "annual": annual,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "unfilled": unfilled,
        "atr_stop_count": len(planner.atr_stop_history),
        "stress_days": sum(
            1 for scale in planner.daily_stress_scale.values() if scale < 1.0
        ),
        "stress_fills": stress_fills,
    }


def _fmt_pct(value):
    return "%+.2f%%" % (value * 100.0)


def _print_block(label, stats):
    print("=" * 72)
    print("[%s]" % label)
    print("  期末资产       = %.2f" % stats["end_value"])
    print("  总收益         = %s" % _fmt_pct(stats["total_return"]))
    print("  年化(244日复利)= %s" % _fmt_pct(stats["annualized"]))
    print("  最大回撤       = %s" % _fmt_pct(stats["max_drawdown"]))
    print("  夏普           = %.3f" % stats["sharpe"])
    print("  索提诺         = %.3f" % stats["sortino"])
    print("  年度收益       = %s" % ", ".join(
        "%s %s" % (year, _fmt_pct(value))
        for year, value in sorted(stats["annual"].items())
    ))
    print("  成交: 买=%d 卖=%d 未成交计划=%d ATR止损成交=%d" % (
        stats["buy_count"], stats["sell_count"],
        stats["unfilled"], stats["atr_stop_count"]))
    print("  压力活跃日     = %d" % stats["stress_days"])
    if stats["stress_fills"]:
        print("  压力减半买入:")
        for fill in stats["stress_fills"]:
            print("    %s %s 股数=%d 计划目标=%.0f" % (
                fill["date"], fill["code"], fill["shares"],
                fill["planned_target"] or 0.0))


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    initial_cash = 20000.0
    print("训练回放窗口: %s ~ %s (%d 个交易日)" % (
        trade_dates[0], trade_dates[-1], len(trade_dates)))

    baseline_params = {
        key: value
        for key, value in strategy.get_default_params().items()
        if key not in FROZEN_STRESS_KEYS
    }
    candidate_params = dict(strategy.get_default_params())

    print()
    print("第 1 遍: 基线 cross-v0.3.2 (无 stress 参数)")
    baseline = _replay(loader, trade_dates, baseline_params, initial_cash)
    _print_block("基线 cross-v0.3.2", baseline)

    print()
    print("第 2 遍: 候选 = 基线 + 冻结 stress 规则 (15/3/0.50)")
    candidate = _replay(loader, trade_dates, candidate_params, initial_cash)
    _print_block("候选 v0.3.2+stress", candidate)

    print()
    print("=" * 72)
    print("[对比] 候选 - 基线")
    print("  总收益  %s -> %s  差值 %+.2fpp" % (
        _fmt_pct(baseline["total_return"]),
        _fmt_pct(candidate["total_return"]),
        (candidate["total_return"] - baseline["total_return"]) * 100.0))
    print("  最大回撤 %s -> %s  差值 %+.2fpp" % (
        _fmt_pct(baseline["max_drawdown"]),
        _fmt_pct(candidate["max_drawdown"]),
        (candidate["max_drawdown"] - baseline["max_drawdown"]) * 100.0))
    print("  夏普 %.3f -> %.3f" % (
        baseline["sharpe"], candidate["sharpe"]))
    print("  索提诺 %.3f -> %.3f" % (
        baseline["sortino"], candidate["sortino"]))
    print("  买/卖  %d/%d -> %d/%d" % (
        baseline["buy_count"], baseline["sell_count"],
        candidate["buy_count"], candidate["sell_count"]))

    recorded = RECORDED_BASELINE
    drift = abs(baseline["total_return"] - recorded["total_return"]) > 0.005
    print()
    print("[基线对齐检查] 已记录修正基线: 总收益 %+.2f%% 回撤 %.2f%% 买/卖 %d/%d" % (
        recorded["total_return"] * 100.0,
        recorded["max_drawdown"] * 100.0,
        recorded["buy_count"], recorded["sell_count"]))
    if drift:
        print("警告: 本地基线总收益与已记录基线差异 > 0.5pp，")
        print("      请核对引擎/数据是否演进，再解读候选对比结果。")

    print()
    print("[预检结论]")
    if drift:
        print("  基线漂移 -> 先排查引擎对齐，暂停后续步骤。")
        return 1
    if candidate["total_return"] + 1e-9 < baseline["total_return"]:
        print("  候选总收益劣化 -> 不通过，不上聚宽。")
        return 1
    if candidate["max_drawdown"] > baseline["max_drawdown"] + 1e-9:
        print("  候选回撤恶化 -> 不通过，不上聚宽。")
        return 1
    if not candidate["stress_fills"]:
        print("  stress 规则未触发 -> 回放未复现规则行为，需排查，不上聚宽。")
        return 1
    print("  通过: 收益不劣化、回撤不恶化、stress 规则实际触发。")
    print("  下一步: 聚宽 2019-2021 训练确认 + 4 个保留窗口验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
