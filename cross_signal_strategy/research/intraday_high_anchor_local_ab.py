# -*- coding: utf-8 -*-
"""Step 1 local A/B: official cross-v0.3.3 versus intraday-high anchor.

Two training replays share the same data, adapter, and execution model. The
candidate planner only replaces the trailing-high anchor: it tracks the maximum
completed daily HIGH instead of the close. Read-only; no formal strategy file
is modified.

Pre-registered gates (all required to pass):
  - total return >= baseline
  - max drawdown <= baseline; Sharpe and Sortino >= baseline
  - every annual return >= baseline
  - at least 3 filled orders change
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


class HighAnchorPlanner(LocalCrossSignalOrderPlanner):
    """Official planner with the intraday-high trailing anchor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highest_high = {}

    def _atr_stop_codes(self, broker, current_prices):
        stopped = set()
        for code, pos in broker.positions.items():
            price = current_prices.get(code)
            if price is None or float(price) <= 0:
                continue
            anchor = self.highest_high.get(code)
            atr_val = self.entry_atr.get(code)
            if anchor is None or anchor <= 0 or atr_val is None or atr_val <= 0:
                continue
            if str(self.buy_dates.get(code)) == str(self._current_date):
                continue
            stop_price = strategy.calc_stop_price(
                anchor, atr_val, pos.avg_cost, self.params)
            if round(float(price), 3) <= round(stop_price, 3):
                stopped.add(code)
        return stopped

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

    def _clear_position_state(self, code):
        super()._clear_position_state(code)
        self.highest_high.pop(code, None)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        self._current_date = str(current_date)
        return super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices)


def _daily_returns(values, initial_cash):
    previous = initial_cash
    returns = []
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


def _sharpe_sortino(returns):
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)
    downside_var = sum(r * r for r in returns if r < 0) / len(returns)
    sharpe = mean / std * math.sqrt(244) if std > 0 else 0.0
    sortino = (
        mean / math.sqrt(downside_var) * math.sqrt(244)
        if downside_var > 0 else 0.0
    )
    return sharpe, sortino


def _annual_returns(dates, values, initial_cash):
    result = {}
    for year in ("2019", "2020", "2021"):
        start_text, end_text = year + "-01-01", year + "-12-31"
        in_year = [
            value for date, value in zip(dates, values)
            if start_text <= str(date) <= end_text
        ]
        if not in_year:
            continue
        prior = [
            value for date, value in zip(dates, values)
            if str(date) < start_text
        ]
        year_start = initial_cash if not prior else prior[-1]
        result[year] = in_year[-1] / year_start - 1.0
    return result


def _replay(loader, trade_dates, planner, initial_cash):
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    values = [day.total_value for day in results]
    returns = _daily_returns(values, initial_cash)
    sharpe, sortino = _sharpe_sortino(returns)
    fills = []
    lots = {}
    trades = []
    buy_count = 0
    sell_count = 0
    atr_stop_count = 0
    for day in results:
        for order in day.orders:
            if not order.filled:
                continue
            fills.append((
                str(day.date),
                str(order.code),
                int(order.amount_delta),
                str(order.reason),
            ))
            code = str(order.code)
            if order.amount_delta > 0:
                buy_count += 1
                current = lots.get(code)
                if current is None:
                    lots[code] = {
                        "entry_date": str(day.date),
                        "entry_price": float(order.exec_price),
                    }
            else:
                sell_count += 1
                if str(order.reason) == "atr_stop":
                    atr_stop_count += 1
                current = lots.get(code)
                if current is not None:
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
    return {
        "end_value": values[-1],
        "total_return": values[-1] / initial_cash - 1.0,
        "max_drawdown": _max_drawdown(values),
        "sharpe": sharpe,
        "sortino": sortino,
        "annual": _annual_returns(
            [str(day.date) for day in results], values, initial_cash),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "atr_stop_count": atr_stop_count,
        "trades": trades,
        "fills": fills,
    }


def main():
    loader = CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    baseline_planner = LocalCrossSignalOrderPlanner(adapter, trade_dates=trade_dates)
    candidate_planner = HighAnchorPlanner(adapter, trade_dates=trade_dates)

    print("本地 A/B: 官方 cross-v0.3.3 vs 盘中最高锚点候选 (2019-2021 训练回放)")
    baseline = _replay(loader, trade_dates, baseline_planner, 20000.0)
    candidate_run = _replay(loader, trade_dates, candidate_planner, 20000.0)

    def fmt(value):
        return "%+.2f%%" % (value * 100.0)

    print("-" * 72)
    print("%-16s %12s %12s" % ("指标", "基线 v0.3.3", "候选"))
    print("%-16s %12s %12s" % ("总收益", fmt(baseline["total_return"]), fmt(candidate_run["total_return"])))
    print("%-16s %12s %12s" % ("最大回撤", fmt(baseline["max_drawdown"]), fmt(candidate_run["max_drawdown"])))
    print("%-16s %12.3f %12.3f" % ("夏普", baseline["sharpe"], candidate_run["sharpe"]))
    print("%-16s %12.3f %12.3f" % ("索提诺", baseline["sortino"], candidate_run["sortino"]))
    for year in ("2019", "2020", "2021"):
        print("%-16s %12s %12s" % (
            "年度" + year,
            fmt(baseline["annual"].get(year, 0.0)),
            fmt(candidate_run["annual"].get(year, 0.0))))
    print("%-16s %12d %12d" % ("买/卖笔数", baseline["buy_count"], candidate_run["buy_count"]))
    print("%-16s %12d %12d" % (
        "卖笔数", baseline["sell_count"], candidate_run["sell_count"]))
    print("%-16s %12d %12d" % (
        "ATR止损成交", baseline["atr_stop_count"], candidate_run["atr_stop_count"]))

    baseline_fills = baseline["fills"]
    candidate_fills = candidate_run["fills"]
    changed = []
    for index in range(max(len(baseline_fills), len(candidate_fills))):
        base = baseline_fills[index] if index < len(baseline_fills) else None
        cand = candidate_fills[index] if index < len(candidate_fills) else None
        if base != cand:
            changed.append((base, cand))
    print("-" * 72)
    print("成交路径差异笔数 = %d (买/卖总数 %d vs %d)" % (
        len(changed), len(baseline_fills), len(candidate_fills)))
    for pair in changed[:12]:
        print("  基线 %s | 候选 %s" % (pair[0], pair[1]))

    # Per-trade exit comparison for trades whose exit date/reason changed.
    baseline_trades = {
        (t["code"], t["entry_date"]): t for t in baseline["trades"]
    }
    print("-" * 72)
    print("提前离场逐笔比较 (候选出场 vs 官方出场, 每股口径):")
    saved = clipped = 0
    total_delta = 0.0
    for trade in candidate_run["trades"]:
        key = (trade["code"], trade["entry_date"])
        official = baseline_trades.get(key)
        if official is None:
            continue
        if trade["exit_date"] == official["exit_date"]:
            continue
        delta = trade["exit_price"] - official["exit_price"]
        total_delta += delta
        if delta > 0:
            saved += 1
        else:
            clipped += 1
        print("  %s %s: 候选%s@%.3f vs 官方%s@%.3f (%+.3f, %s)" % (
            trade["entry_date"], trade["code"],
            trade["exit_date"], trade["exit_price"],
            official["exit_date"], official["exit_price"],
            delta, "救回" if delta > 0 else "砍掉"))
    print("  合计: 救回%d笔 砍掉%d笔 差额%+.3f" % (saved, clipped, total_delta))

    gates = {
        "总收益>=基线": candidate_run["total_return"] + 1e-9 >= baseline["total_return"],
        "回撤<=基线": candidate_run["max_drawdown"] <= baseline["max_drawdown"] + 1e-9,
        "夏普>=基线": candidate_run["sharpe"] + 1e-9 >= baseline["sharpe"],
        "索提诺>=基线": candidate_run["sortino"] + 1e-9 >= baseline["sortino"],
        "每年收益>=基线": all(
            candidate_run["annual"].get(year, 0.0) + 1e-9 >= baseline["annual"].get(year, 0.0)
            for year in ("2019", "2020", "2021")),
        "至少3笔成交改变": len(changed) >= 3,
    }
    print("-" * 72)
    for gate, passed in gates.items():
        print("[%s] %s" % ("通过" if passed else "未通过", gate))
    passed = all(gates.values())
    print("门控判定: %s" % (
        "通过, 可进入聚宽训练确认"
        if passed else
        "未通过, 候选被拒, 家族关闭(禁止搜索近邻变体)"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
