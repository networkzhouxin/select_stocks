# -*- coding: utf-8 -*-
"""Read-only attribution of the six frozen ATR-stress half-size buys.

For every filled half-size buy in the 2019-2021 training replay, compare the
realized PnL at the actual half size against the counterfactual full size
(same entry and exit prices, identical order path). This answers directly
whether halving "avoided further losses" or "missed a rally" in each case.

Data boundary: approved 2018 warm-up and 2019-2021 training data only.
No formal strategy file is modified; JoinQuant remains the authority.
"""

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


class StressPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_scale = {}
        self.planned_stress_targets = {}

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        scale = self._portfolio_atr_stress_buy_scale(str(current_date))
        self.daily_scale[str(current_date)] = scale
        orders = super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices
        )
        if scale < 1.0:
            for order in orders:
                if float(order["target_value"]) > 0.0:
                    self.planned_stress_targets[(str(current_date), str(order["code"]))] = (
                        float(order["target_value"])
                    )
        return orders


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    params = strategy.get_default_params()
    adapter = build_training_signal_adapter(loader)
    planner = StressPlanner(adapter, params=params, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    results = engine.run(trade_dates, planner.plan_orders)

    # Ledger: per-code lot (entry_date, shares, avg_price) and realized trades.
    lots = {}
    stress_entries = []
    realized = []
    for day in results:
        for order in day.orders:
            if not order.filled:
                continue
            code = str(order.code)
            if order.amount_delta > 0:
                current = lots.get(code)
                if current is None:
                    lots[code] = {
                        "entry_date": str(day.date),
                        "shares": int(order.amount_delta),
                        "price": float(order.exec_price),
                    }
                else:
                    old_value = current["shares"] * current["price"]
                    new_shares = current["shares"] + int(order.amount_delta)
                    current["price"] = (old_value + int(order.amount_delta) * float(order.exec_price)) / new_shares
                    current["shares"] = new_shares
            elif order.amount_delta < 0:
                current = lots.get(code)
                if current is None or current["shares"] <= 0:
                    continue
                exit_price = float(order.exec_price)
                realized.append({
                    "code": code,
                    "entry_date": current["entry_date"],
                    "exit_date": str(day.date),
                    "entry_price": current["price"],
                    "exit_price": exit_price,
                    "shares": current["shares"],
                })
                lots.pop(code, None)

    # Mark-to-market for lots still open at the boundary.
    final_marks = results[-1].marks if results else {}
    for code, lot in sorted(lots.items()):
        exit_price = float(final_marks.get(code, lot["price"]))
        realized.append({
            "code": code,
            "entry_date": lot["entry_date"],
            "exit_date": str(results[-1].date),
            "entry_price": lot["price"],
            "exit_price": exit_price,
            "shares": lot["shares"],
            "open": True,
        })

    # Attribute the six stress entries.
    print("=" * 78)
    print("ATR-stress 减半买入事后归因 (2019-2021 训练回放, 只读)")
    print("=" * 78)
    print("%-12s %-8s %-12s %9s %9s %9s %9s %9s %8s %s" % (
        "买入日", "代码", "卖出日", "入场价", "出场价", "持有收益%",
        "半仓盈亏", "满仓反事实", "差额", "结论"))
    total_delta = 0.0
    helped = 0
    hurt = 0
    for (entry_date, code), target in sorted(planner.planned_stress_targets.items()):
        trade = next(
            (item for item in realized
             if item["code"] == code and item["entry_date"] == entry_date),
            None,
        )
        if trade is None:
            print("%-12s %-8s 未成交或未匹配到出场记录" % (entry_date, code))
            continue
        half_pnl = trade["shares"] * (trade["exit_price"] - trade["entry_price"])
        full_target = target / 0.5
        full_shares = int(full_target / trade["entry_price"] / 100) * 100
        full_pnl = full_shares * (trade["exit_price"] - trade["entry_price"])
        delta = half_pnl - full_pnl
        total_delta += delta
        hold_return = (trade["exit_price"] / trade["entry_price"] - 1.0) * 100.0
        verdict = "减半更优(躲过下跌)" if delta > 0 else (
            "满仓更优(踏空反弹)" if delta < 0 else "无差异")
        if delta > 1e-9:
            helped += 1
        elif delta < -1e-9:
            hurt += 1
        print("%-12s %-8s %-12s %9.3f %9.3f %8.2f%% %9.1f %9.1f %+9.1f %s" % (
            entry_date, code, trade["exit_date"],
            trade["entry_price"], trade["exit_price"], hold_return,
            half_pnl, full_pnl, delta, verdict))
    print("-" * 78)
    print("减半更优 %d 笔, 踏空反弹 %d 笔, 合计差额 %+.1f 元" % (
        helped, hurt, total_delta))
    print("(合计差额应约等于 候选期末资产 45000.50 - 基线期末资产 44122.30 = +878.2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
