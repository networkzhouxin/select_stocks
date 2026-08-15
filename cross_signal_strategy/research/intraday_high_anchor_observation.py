# -*- coding: utf-8 -*-
"""Step 0 binding observation for the intraday-high trailing anchor.

Read-only observation on the official cross-v0.3.3 training replay. For every
stop-check day it compares the official close-anchored stop with the frozen
high-anchored stop (multiplier 2.5, floor 5%, cap 15%, entry ATR unchanged),
counts binding days where the two stops differ, and counts days where the
high-anchored stop would have triggered while the close-anchored stop would not
(same-day extra triggers).

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

BASE_FLOOR = 0.05
CAP = 0.15
MULT = 2.5


def _stop_price(anchor, atr_val):
    pct_stop = MULT * atr_val / anchor
    pct_stop = max(BASE_FLOOR, min(CAP, pct_stop))
    return anchor * (1.0 - pct_stop)


class HighAnchorObservationPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highest_high = {}
        self.binding_events = []
        self.extra_trigger_events = []

    def _atr_stop_codes(self, broker, current_prices):
        for code, pos in broker.positions.items():
            price = current_prices.get(code)
            if price is None or float(price) <= 0:
                continue
            price = float(price)
            close_anchor = self.highest_since_buy.get(code)
            high_anchor = self.highest_high.get(code)
            atr_val = self.entry_atr.get(code)
            if (
                close_anchor is None or high_anchor is None
                or close_anchor <= 0 or high_anchor <= 0
                or atr_val is None or atr_val <= 0
            ):
                continue
            if str(self.buy_dates.get(code)) == str(self._current_date):
                continue
            stop_base = _stop_price(close_anchor, atr_val)
            stop_high = _stop_price(high_anchor, atr_val)
            if abs(stop_high - stop_base) > 1e-12:
                event = {
                    "date": self._current_date,
                    "code": str(code),
                    "price": price,
                    "stop_base": stop_base,
                    "stop_high": stop_high,
                }
                self.binding_events.append(event)
                if (
                    round(price, 3) <= round(stop_high, 3)
                    and round(price, 3) > round(stop_base, 3)
                ):
                    self.extra_trigger_events.append(event)
        return super()._atr_stop_codes(broker, current_prices)

    def on_orders_filled(self, current_date, orders):
        super().on_orders_filled(current_date, orders)
        for order in orders:
            if not getattr(order, "filled", False):
                continue
            code = str(order.code).split(".")[0]
            if order.amount_delta > 0:
                self.highest_high[code] = float(order.exec_price)

    def on_after_close(self, current_date, marks):
        super().on_after_close(current_date, marks)
        for code in list(self.highest_since_buy.keys()):
            try:
                frame = self.signal_adapter.loader.load_daily_frame(
                    code, str(current_date))
                rows = frame[frame["date"].astype(str) == str(current_date)]
                if rows.empty or "high" not in rows.columns:
                    continue
                high = float(rows.iloc[0]["high"])
                if high > 0:
                    self.highest_high[code] = max(
                        self.highest_high.get(code, high), high)
            except Exception:
                continue

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        self._current_date = str(current_date)
        return super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices)


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = HighAnchorObservationPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    engine.run(trade_dates, planner.plan_orders)

    events = planner.binding_events
    extras = planner.extra_trigger_events
    print("=" * 78)
    print("Step 0 绑定事件观察: 盘中最高锚点 (官方 v0.3.3 训练回放, 只读)")
    print("=" * 78)
    print("绑定事件总数(两套止损不同) = %d" % len(events))
    print("额外触发事件(盘中锚触发而收盘锚不触发) = %d" % len(extras))
    by_year = {}
    for event in events:
        by_year.setdefault(event["date"][:4], 0)
        by_year[event["date"][:4]] += 1
    print("绑定日按年: %s" % ", ".join(
        "%s=%d" % (year, count) for year, count in sorted(by_year.items())))
    codes = sorted(set(event["code"] for event in events))
    print("涉及标的(%d): %s" % (len(codes), ",".join(codes)))
    if extras:
        print("-" * 78)
        print("额外触发明细:")
        print("%-12s %-8s %9s %9s %9s" % (
            "日期", "代码", "当日价", "收盘锚止损", "盘中锚止损"))
        for event in extras:
            print("%-12s %-8s %9.3f %9.3f %9.3f" % (
                event["date"], event["code"], event["price"],
                event["stop_base"], event["stop_high"]))
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
