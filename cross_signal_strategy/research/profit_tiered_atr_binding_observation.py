# -*- coding: utf-8 -*-
"""Step 0 binding-count observation for the profit-tiered ATR tightening family.

Read-only observation on the official cross-v0.3.3 training replay. It records
every daily stop check where the frozen profit-tiered variant would change the
effective stop: profit > 5% AND the unfloored baseline stop exceeds the 5%
floor. It also counts days where the tightened stop would have triggered while
the baseline stop would not.

Gate: at least 10 binding observations are required to proceed to a candidate.
No formal strategy file is modified; approved training data only.
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

PROFIT_LOW = 0.05
PROFIT_HIGH = 0.15
FACTOR_LOW = 0.8
FACTOR_HIGH = 0.6
FLOOR = 0.05
CAP = 0.15
BASE_MULT = 2.5


class BindingPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding_events = []
        self.extra_trigger_events = []

    def _atr_stop_codes(self, broker, current_prices):
        for code, pos in broker.positions.items():
            price = current_prices.get(code)
            if price is None or float(price) <= 0:
                continue
            price = float(price)
            highest = self.highest_since_buy.get(code)
            atr_val = self.entry_atr.get(code)
            if highest is None or atr_val is None or highest <= 0 or atr_val <= 0:
                continue
            if pos.avg_cost <= 0:
                continue
            profit = price / pos.avg_cost - 1.0
            unfloored_pct = BASE_MULT * atr_val / highest
            if profit > PROFIT_LOW and unfloored_pct > FLOOR:
                factor = FACTOR_HIGH if profit > PROFIT_HIGH else FACTOR_LOW
                pct_base = max(FLOOR, min(CAP, unfloored_pct))
                pct_tight = max(FLOOR, min(CAP, unfloored_pct * factor))
                stop_base = highest * (1.0 - pct_base)
                stop_tight = highest * (1.0 - pct_tight)
                event = {
                    "date": self._current_date,
                    "code": str(code),
                    "profit": profit,
                    "unfloored_pct": unfloored_pct,
                    "pct_base": pct_base,
                    "pct_tight": pct_tight,
                    "price": price,
                    "stop_base": stop_base,
                    "stop_tight": stop_tight,
                }
                self.binding_events.append(event)
                if price <= stop_tight and price > stop_base:
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
    planner = BindingPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    engine.run(trade_dates, planner.plan_orders)

    events = planner.binding_events
    extras = planner.extra_trigger_events
    print("=" * 78)
    print("Step 0 绑定事件观察: 利润分段 ATR 收紧 (官方 v0.3.3 训练回放, 只读)")
    print("=" * 78)
    print("绑定事件总数(停损检查日 x 持仓) = %d" % len(events))
    print("额外触发事件(收紧止损当日触发而基线不触发) = %d" % len(extras))
    by_year = {}
    for event in events:
        by_year.setdefault(event["date"][:4], 0)
        by_year[event["date"][:4]] += 1
    print("按年分布: %s" % ", ".join(
        "%s=%d" % (year, count) for year, count in sorted(by_year.items())))
    codes = sorted(set(event["code"] for event in events))
    print("涉及标的(%d): %s" % (len(codes), ",".join(codes)))
    high_tier = [e for e in events if e["profit"] > PROFIT_HIGH]
    print("浮盈>15%% 的高档事件 = %d" % len(high_tier))

    if extras:
        print("-" * 78)
        print("额外触发明细 (收紧后当日触发、基线不触发):")
        print("%-12s %-8s %8s %8s %8s %9s %9s" % (
            "日期", "代码", "浮盈%", "基线止损距%", "收紧止损距%",
            "当日价", "基线止损价"))
        for event in extras:
            print("%-12s %-8s %7.1f%% %7.2f%% %7.2f%% %9.3f %9.3f" % (
                event["date"], event["code"],
                event["profit"] * 100.0,
                event["pct_base"] * 100.0,
                event["pct_tight"] * 100.0,
                event["price"], event["stop_base"]))
    print("-" * 78)
    print("样本分布: 绑定日止损距离 基线[%.2f%%~%.2f%%] 收紧后[%.2f%%~%.2f%%]" % (
        min(e["pct_base"] for e in events) * 100.0 if events else 0.0,
        max(e["pct_base"] for e in events) * 100.0 if events else 0.0,
        min(e["pct_tight"] for e in events) * 100.0 if events else 0.0,
        max(e["pct_tight"] for e in events) * 100.0 if events else 0.0,
    ))
    print("门槛判定: %s" % (
        "通过 (>=10 绑定事件, 可进入 Step 1 候选)"
        if len(events) >= 10 else
        "未通过 (<10 绑定事件, 家族关闭, 不建候选、不搜索更宽参数)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
