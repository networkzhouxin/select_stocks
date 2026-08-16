# -*- coding: utf-8 -*-
"""Step 0 trade-level counterfactual for the profit-gated direct-sell matrix.

One read-only replay of the official cross-v0.3.3 path computes the
counterfactual for all 12 frozen variants at once. A variant fires when the
sell score reaches its threshold AND the current 09:35 profit falls inside its
band, with the 5-day minimum hold and the ADX strong-uptrend exemption kept,
and the price-structure confirmation bypassed.

Per-variant gates: at least 5 affected closed trades, positive total per-share
delta, and positive delta in each of 2019/2020/2021. The pre-registered
selection rule then picks only the highest total per-share delta among the
gate-passing variants.

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

VARIANTS = {}
for _letter, _threshold in (("A", 32), ("B", 35), ("C", 38), ("D", 40)):
    for _band_index, _band in enumerate(
        ((0.02, 0.04), (0.03, 0.05), (0.04, 0.06)), start=1
    ):
        VARIANTS["%s%d" % (_letter, _band_index)] = (
            _threshold, _band[0], _band[1])


class MatrixObservationPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant_events = {key: [] for key in VARIANTS}

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        self._current_date = str(current_date)
        orders = super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices)
        sold = {
            str(order["code"])
            for order in orders
            if float(order.get("target_value", 0.0)) <= 0.0
        }
        score_map = self.last_scores or {}
        prices = current_prices or {}
        for raw_code, pos in broker.positions.items():
            code = str(raw_code)
            if code in sold:
                continue
            score = score_map.get(code)
            if not isinstance(score, dict):
                continue
            price = prices.get(code)
            if price is None or float(price) <= 0:
                continue
            if pos.avg_cost <= 0:
                continue
            buy_date = self.buy_dates.get(code)
            if buy_date is None:
                continue
            if not strategy.can_sell_by_signal(
                buy_date,
                self._current_date,
                min_hold_days=5,
                trade_days=self.trade_dates,
            ):
                continue
            if strategy.is_protected_by_strong_adx_uptrend(score, self.params):
                continue
            profit = float(price) / pos.avg_cost - 1.0
            sell_score = strategy._numeric_score(score.get("sell_score"))
            for key, (threshold, low, high) in VARIANTS.items():
                if sell_score >= threshold and low <= profit < high:
                    self.variant_events[key].append({
                        "date": self._current_date,
                        "code": code,
                        "price": float(price),
                        "profit": profit,
                        "sell_score": sell_score,
                    })
        return orders


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = MatrixObservationPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    results = engine.run(trade_dates, planner.plan_orders)

    # Official trade ledger.
    lots = {}
    trades = []
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
                        "entry_price": float(order.exec_price),
                    }
                else:
                    old_value = current["entry_price"]
                    current["entry_price"] = float(order.exec_price)
                    del old_value
            elif order.amount_delta < 0:
                current = lots.get(code)
                if current is None:
                    continue
                trades.append({
                    "code": code,
                    "entry_date": current["entry_date"],
                    "exit_date": str(day.date),
                    "entry_price": current["entry_price"],
                    "exit_price": float(order.exec_price),
                    "reason": str(order.reason),
                })
                lots.pop(code, None)
    final_marks = results[-1].marks if results else {}
    for code, lot in sorted(lots.items()):
        trades.append({
            "code": code,
            "entry_date": lot["entry_date"],
            "exit_date": str(results[-1].date),
            "entry_price": lot["entry_price"],
            "exit_price": float(final_marks.get(code, lot["entry_price"])),
            "reason": "open_at_boundary",
        })

    print("=" * 78)
    print("Step 0 交易级反事实观察: 利润门槛直卖矩阵 (官方 v0.3.3 训练回放, 只读)")
    print("=" * 78)
    print("%-4s %10s %10s %8s %8s %12s %12s %12s %12s %s" % (
        "变体", "卖出分>=", "利润区间", "触发数", "受影响",
        "2019差额", "2020差额", "2021差额", "总差额", "门槛"))
    summaries = {}
    for key, (threshold, low, high) in sorted(VARIANTS.items()):
        events = planner.variant_events[key]
        affected = []
        for trade in trades:
            firing = next(
                (
                    event for event in events
                    if event["code"] == trade["code"]
                    and trade["entry_date"] <= event["date"] < trade["exit_date"]
                ),
                None,
            )
            if firing is None:
                continue
            delta = firing["price"] - trade["exit_price"]
            affected.append({
                "year": trade["entry_date"][:4],
                "delta": delta,
            })
        by_year = {}
        for item in affected:
            by_year[item["year"]] = by_year.get(item["year"], 0.0) + item["delta"]
        total = sum(item["delta"] for item in affected)
        count_ok = len(affected) >= 5
        annual_ok = all(
            by_year.get(year, 0.0) > 0 for year in ("2019", "2020", "2021"))
        total_ok = total > 0
        gate = "通过" if (count_ok and annual_ok and total_ok) else "未通过"
        band_text = "%d%%~%d%%" % (int(low * 100), int(high * 100))
        print("%-4s %10d %10s %8d %8d %+12.3f %+12.3f %+12.3f %+12.3f %s" % (
            key, threshold, band_text, len(events), len(affected),
            by_year.get("2019", 0.0), by_year.get("2020", 0.0),
            by_year.get("2021", 0.0), total, gate))
        summaries[key] = {
            "threshold": threshold,
            "band": (low, high),
            "affected": affected,
            "by_year": by_year,
            "total": total,
            "gate": gate == "通过",
            "events": events,
        }

    passing = [key for key, item in summaries.items() if item["gate"]]
    print("-" * 78)
    if not passing:
        print("选择结果: 无变体通过门槛 -> 家族关闭, 不建候选、不搜索近邻变体")
        return 0
    winner = max(passing, key=lambda key: summaries[key]["total"])
    print("通过门槛的变体: %s" % ",".join(sorted(passing)))
    print("预注册选择规则 -> 选中变体: %s (总每股差额最高 %+.3f)" % (
        winner, summaries[winner]["total"]))
    item = summaries[winner]
    print("参数: 卖出分>=%d, 利润区间 %d%%~%d%%" % (
        item["threshold"], int(item["band"][0] * 100), int(item["band"][1] * 100)))
    print("受影响交易 %d 笔, 触发事件 %d 次" % (
        len(item["affected"]), len(item["events"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
