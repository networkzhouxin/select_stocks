# -*- coding: utf-8 -*-
"""Step 0 trade-level counterfactual for the profit-giveback direct exit.

Read-only observation on the official cross-v0.3.3 training replay. It records
every stop-check day where the frozen giveback rule (peak profit >= 5%, current
profit <= peak - 3pp) would fire while the official path still holds the
position, then maps each firing to its closed trade and compares the first
rule-exit price with the official exit price (same entry, same shares).

Gates (all required to proceed to a candidate):
  - at least 5 affected closed trades
  - positive total delta
  - positive delta in each of 2019 / 2020 / 2021

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

PEAK_ACTIVATION = 0.05
GIVEBACK = 0.03


class GivebackObservationPlanner(LocalCrossSignalOrderPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.giveback_events = []

    def _atr_stop_codes(self, broker, current_prices):
        for code, pos in broker.positions.items():
            price = current_prices.get(code)
            if price is None or float(price) <= 0:
                continue
            highest = self.highest_since_buy.get(code)
            if highest is None or highest <= 0 or pos.avg_cost <= 0:
                continue
            if str(self.buy_dates.get(code)) == str(self._current_date):
                continue
            peak_profit = highest / pos.avg_cost - 1.0
            current_profit = float(price) / pos.avg_cost - 1.0
            if (
                peak_profit >= PEAK_ACTIVATION
                and current_profit <= peak_profit - GIVEBACK
            ):
                self.giveback_events.append({
                    "date": str(self._current_date),
                    "code": str(code),
                    "peak_profit": peak_profit,
                    "current_profit": current_profit,
                    "price": float(price),
                })
        return super()._atr_stop_codes(broker, current_prices)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        self._current_date = str(current_date)
        return super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices)


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = GivebackObservationPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    results = engine.run(trade_dates, planner.plan_orders)

    # Build the official trade ledger: entry on filled buy, exit on filled sell.
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
                        "shares": int(order.amount_delta),
                        "price": float(order.exec_price),
                    }
                else:
                    old_value = current["shares"] * current["price"]
                    new_shares = current["shares"] + int(order.amount_delta)
                    current["price"] = (
                        old_value + int(order.amount_delta) * float(order.exec_price)
                    ) / new_shares
                    current["shares"] = new_shares
            elif order.amount_delta < 0:
                current = lots.get(code)
                if current is None or current["shares"] <= 0:
                    continue
                trades.append({
                    "code": code,
                    "entry_date": current["entry_date"],
                    "exit_date": str(day.date),
                    "entry_price": current["price"],
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
            "entry_price": lot["price"],
            "exit_price": float(final_marks.get(code, lot["price"])),
            "reason": "open_at_boundary",
        })

    events_by_trade = {}
    for event in planner.giveback_events:
        for trade in trades:
            if (
                trade["code"] == event["code"]
                and trade["entry_date"] <= event["date"] < trade["exit_date"]
            ):
                events_by_trade.setdefault(
                    (trade["code"], trade["entry_date"]), []).append(event)
                break

    affected = []
    for trade in trades:
        key = (trade["code"], trade["entry_date"])
        events = sorted(events_by_trade.get(key, []), key=lambda e: e["date"])
        firing = next(
            (event for event in events if event["date"] < trade["exit_date"]),
            None,
        )
        if firing is None:
            continue
        delta = firing["price"] - trade["exit_price"]
        affected.append({
            "code": trade["code"],
            "entry_date": trade["entry_date"],
            "exit_date": trade["exit_date"],
            "reason": trade["reason"],
            "entry_price": trade["entry_price"],
            "rule_exit_price": firing["price"],
            "official_exit_price": trade["exit_price"],
            "peak_profit": firing["peak_profit"],
            "current_profit": firing["current_profit"],
            "delta_per_share": delta,
            "year": trade["entry_date"][:4],
        })

    print("=" * 78)
    print("Step 0 交易级反事实观察: 利润回吐直接卖出 (官方 v0.3.3 训练回放, 只读)")
    print("=" * 78)
    print("回吐触发事件总数 = %d" % len(planner.giveback_events))
    print("受影响的已平仓交易数 = %d" % len(affected))
    if affected:
        print("-" * 78)
        print("%-12s %-8s %-12s %-14s %9s %9s %9s %8s %s" % (
            "买入日", "代码", "官方卖出日", "官方原因", "规则出场价",
            "官方出场价", "每股差额", "年份", "结论"))
        for item in affected:
            verdict = "救回收益" if item["delta_per_share"] > 0 else "砍掉赢家"
            print("%-12s %-8s %-12s %-14s %9.3f %9.3f %+9.3f %8s %s" % (
                item["entry_date"], item["code"], item["exit_date"],
                item["reason"], item["rule_exit_price"],
                item["official_exit_price"], item["delta_per_share"],
                item["year"], verdict))
        print("-" * 78)
        totals = {}
        for item in affected:
            totals.setdefault(item["year"], 0.0)
            totals[item["year"]] += item["delta_per_share"]
        print("按年合计差额: %s" % ", ".join(
            "%s=%+.3f" % (year, totals[year]) for year in sorted(totals)))
        print("总合计差额 = %+.3f (每股口径, 未含滑点/复利)" % sum(
            item["delta_per_share"] for item in affected))

    count_ok = len(affected) >= 5
    total_ok = sum(item["delta_per_share"] for item in affected) > 0
    annual_ok = all(
        sum(item["delta_per_share"] for item in affected if item["year"] == year) > 0
        for year in ("2019", "2020", "2021")
    )
    print("门槛判定: %s (受影响交易%d>=5 %s; 总差额%s; 逐年一致%s)" % (
        "通过, 可进入 Step 1 候选" if count_ok and total_ok and annual_ok
        else "未通过, 家族关闭(禁止搜索近邻变体)",
        len(affected), "是" if count_ok else "否",
        ">0 是" if total_ok else ">0 否",
        "是" if annual_ok else "否"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
