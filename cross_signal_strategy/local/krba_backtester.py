# -*- coding: utf-8 -*-
"""Causal 09:35 plus ATR-only 14:50 local engine for the KRBA candidate."""

from __future__ import annotations

from typing import Iterable, Mapping

from cross_signal_strategy.local.local_backtester import (
    DayResult,
    LocalBroker,
    OrderResult,
    Position,
    _bar_has_executable_trade,
    _planner_max_holdings,
)


PRICE_FIELD = {"09:35": "close", "14:50": "open"}


class KRBABacktestEngine:
    def __init__(self, loader, initial_cash: float, broker_kwargs=None) -> None:
        self.loader = loader
        self.broker = LocalBroker(initial_cash, **dict(broker_kwargs or {}))

    def run(self, trade_dates: Iterable[str], planner):
        results = []
        previous_date = None
        for current_date in [str(item) for item in trade_dates]:
            day_orders = []
            for decision_time in ("09:35", "14:50"):
                prices = self._current_prices(current_date, decision_time)
                plans = planner.plan_orders_at(
                    current_date,
                    previous_date,
                    self.broker,
                    decision_time,
                    current_prices=prices,
                )
                batch = self._execute(current_date, decision_time, plans, planner)
                planner.on_orders_processed(
                    current_date, decision_time, plans, batch
                )
                day_orders.extend(batch)
            marks = self._close_marks(current_date)
            planner.on_after_close(current_date, marks)
            positions = {
                code: Position(pos.code, pos.amount, pos.avg_cost)
                for code, pos in self.broker.positions.items()
            }
            results.append(
                DayResult(
                    date=current_date,
                    previous_date=previous_date,
                    orders=day_orders,
                    cash=self.broker.cash,
                    positions=positions,
                    marks=marks,
                    total_value=self.broker.total_value(marks),
                )
            )
            previous_date = current_date
        return results

    def _current_prices(self, date: str, decision_time: str):
        field = PRICE_FIELD[decision_time]
        prices = {}
        for code in self.broker.positions:
            try:
                bar = self.loader.get_minute_bar(code, date, decision_time)
            except (FileNotFoundError, KeyError):
                continue
            if _bar_has_executable_trade(bar):
                prices[code] = float(bar[field])
        return prices

    def _execute(self, date, decision_time, plans, planner):
        field = PRICE_FIELD[decision_time]
        orders = []
        max_holdings = _planner_max_holdings(planner)
        for plan in plans:
            code = str(plan["code"])
            target = float(plan["target_value"])
            reason = str(plan.get("reason", ""))
            if (
                target > 0
                and code not in self.broker.positions
                and max_holdings is not None
                and len(self.broker.positions) >= max_holdings
            ):
                orders.append(
                    OrderResult(
                        code, 0, 0.0, 0.0, f"{date} {decision_time}", False,
                        "no available holding slot after execution",
                    )
                )
                continue
            try:
                bar = self.loader.get_minute_bar(code, date, decision_time)
            except (FileNotFoundError, KeyError):
                orders.append(
                    OrderResult(
                        code, 0, 0.0, 0.0, f"{date} {decision_time}", False,
                        f"missing execution bar at {decision_time}",
                    )
                )
                continue
            price = float(bar[field])
            if not _bar_has_executable_trade(bar):
                orders.append(
                    OrderResult(
                        code, 0, price, 0.0, f"{date} {decision_time}", False,
                        f"no executable trade at {decision_time}",
                    )
                )
                continue
            order = self.broker.order_target_value(
                code, target, price, f"{date} {decision_time}"
            )
            if order.filled and reason:
                order.reason = reason
            orders.append(order)
        return orders

    def _close_marks(self, date: str):
        marks = {}
        for code in self.broker.positions:
            frame = self.loader.load_daily_frame(code, date)
            rows = frame.loc[frame["date"].astype(str) == str(date)]
            if rows.empty:
                raise KeyError(f"No daily close for {code} {date}")
            marks[code] = float(rows.iloc[0]["close"])
        return marks

