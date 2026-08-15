# -*- coding: utf-8 -*-
"""Step 0 binding observation for the gold-specific stop family.

Read-only observation on the official cross-v0.3.3 training replay. For every
518880 stop-check day it compares the official stop (floor 5%, multiplier 2.5x)
with the frozen gold-specific stop (floor 3%, multiplier 2.0x), counts binding
days where the two stops differ, and counts days where the gold stop would have
triggered while the official stop would not (same-day extra triggers).

Gates: at least 10 binding days AND at least 3 extra-trigger days are required
to proceed to a candidate. No formal strategy file is modified; approved
training data only.
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

GOLD_CODE = "518880"
GOLD_FLOOR = 0.03
GOLD_MULT = 2.0
BASE_FLOOR = 0.05
BASE_MULT = 2.5
CAP = 0.15


def _stop_price(highest, atr_val, floor, mult):
    pct_stop = mult * atr_val / highest
    pct_stop = max(floor, min(CAP, pct_stop))
    return highest * (1.0 - pct_stop)


class GoldBindingPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding_events = []
        self.extra_trigger_events = []

    def _atr_stop_codes(self, broker, current_prices):
        for code, pos in broker.positions.items():
            if str(code) != GOLD_CODE:
                continue
            price = current_prices.get(code)
            if price is None or float(price) <= 0:
                continue
            price = float(price)
            highest = self.highest_since_buy.get(code)
            atr_val = self.entry_atr.get(code)
            if highest is None or atr_val is None or highest <= 0 or atr_val <= 0:
                continue
            stop_base = _stop_price(highest, atr_val, BASE_FLOOR, BASE_MULT)
            stop_gold = _stop_price(highest, atr_val, GOLD_FLOOR, GOLD_MULT)
            if abs(stop_gold - stop_base) > 1e-12:
                event = {
                    "date": self._current_date,
                    "code": str(code),
                    "price": price,
                    "stop_base": stop_base,
                    "stop_gold": stop_gold,
                }
                self.binding_events.append(event)
                if (
                    round(price, 3) <= round(stop_gold, 3)
                    and round(price, 3) > round(stop_base, 3)
                ):
                    self.extra_trigger_events.append(event)
        return super()._atr_stop_codes(broker, current_prices)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        self._current_date = str(current_date)
        return super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices)


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = GoldBindingPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    engine.run(trade_dates, planner.plan_orders)

    events = planner.binding_events
    extras = planner.extra_trigger_events
    print("=" * 78)
    print("Step 0 绑定事件观察: 黄金品种级止损 518880 (官方 v0.3.3 训练回放, 只读)")
    print("=" * 78)
    print("黄金止损检查日总数(绑定, 两套止损不同) = %d" % len(events))
    print("额外触发事件(黄金止损当日触发而基线不触发) = %d" % len(extras))
    by_year = {}
    for event in events:
        by_year.setdefault(event["date"][:4], 0)
        by_year[event["date"][:4]] += 1
    print("绑定日按年: %s" % ", ".join(
        "%s=%d" % (year, count) for year, count in sorted(by_year.items())))
    if extras:
        print("-" * 78)
        print("额外触发明细:")
        print("%-12s %9s %9s %9s %9s" % (
            "日期", "当日价", "基线止损", "黄金止损", "距峰值%"))
        for event in extras:
            highest_implied = event["stop_base"] / (1.0 - BASE_FLOOR)
            print("%-12s %9.3f %9.3f %9.3f %8.2f%%" % (
                event["date"], event["price"],
                event["stop_base"], event["stop_gold"],
                (event["price"] / highest_implied - 1.0) * 100.0))
    print("-" * 78)
    binding_ok = len(events) >= 10
    trigger_ok = len(extras) >= 3
    print("门槛判定: %s (绑定日%d>=10 %s; 额外触发%d>=3 %s)" % (
        "通过, 可进入 Step 1 候选" if binding_ok and trigger_ok else "未通过, 家族关闭",
        len(events), "是" if binding_ok else "否",
        len(extras), "是" if trigger_ok else "否"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
