# -*- coding: utf-8 -*-
"""Run the fixed 09:35 and 14:45 batches against one local broker."""

from __future__ import annotations

from cross_signal_strategy.local.local_backtester import (
    DayResult,
    LocalBroker,
    OrderResult,
    Position,
    _bar_has_executable_trade,
    _planner_max_holdings,
)


DECISION_PRICE_FIELDS = {"09:35": "close", "14:45": "open"}


class DualTimepointBacktestEngine:
    def __init__(
        self,
        loader,
        initial_cash: float,
        decision_times=("09:35", "14:45"),
        broker_kwargs=None,
    ) -> None:
        allowed = ("09:35", "14:45")
        if tuple(decision_times) not in (("09:35",), allowed):
            raise ValueError(
                "Only morning baseline or fixed 09:35/14:45 candidate is allowed"
            )
        self.loader = loader
        self.decision_times = tuple(decision_times)
        self.broker = LocalBroker(
            initial_cash=initial_cash,
            **dict(broker_kwargs or {}),
        )

    def run(self, trade_dates, planner):
        results = []
        previous_date = None
        for current_date in [str(item) for item in trade_dates]:
            day_orders = []
            for decision_time in self.decision_times:
                current_prices = self._current_prices(current_date, decision_time)
                plans = planner.plan_orders_at(
                    current_date,
                    previous_date,
                    self.broker,
                    decision_time,
                    current_prices=current_prices,
                )
                batch_orders = self._execute_plans(
                    current_date, decision_time, plans, planner
                )
                planner.on_orders_processed(
                    current_date, decision_time, plans, batch_orders
                )
                day_orders.extend(batch_orders)

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

    def _current_prices(self, current_date, decision_time):
        prices = {}
        field = DECISION_PRICE_FIELDS[decision_time]
        for code in self.broker.positions:
            try:
                bar = self.loader.get_minute_bar(
                    code, current_date, decision_time
                )
            except (FileNotFoundError, KeyError):
                continue
            if decision_time == "09:35" or _bar_has_executable_trade(bar):
                prices[code] = float(bar[field])
        return prices

    def _execute_plans(
        self, current_date, decision_time, plans, planner
    ):
        orders = []
        max_holdings = _planner_max_holdings(planner)
        field = DECISION_PRICE_FIELDS[decision_time]
        for plan in plans:
            code = str(plan["code"])
            target_value = float(plan["target_value"])
            reason = str(plan.get("reason", ""))
            if (
                target_value > 0.0
                and code not in self.broker.positions
                and max_holdings is not None
                and len(self.broker.positions) >= max_holdings
            ):
                orders.append(
                    OrderResult(
                        code,
                        0,
                        0.0,
                        0.0,
                        "%s %s" % (current_date, decision_time),
                        False,
                        "no available holding slot after execution",
                    )
                )
                continue
            try:
                bar = self.loader.get_minute_bar(
                    code, current_date, decision_time
                )
            except (FileNotFoundError, KeyError):
                orders.append(
                    OrderResult(
                        code,
                        0,
                        0.0,
                        0.0,
                        "%s %s" % (current_date, decision_time),
                        False,
                        "missing execution bar at %s" % decision_time,
                    )
                )
                continue
            price = float(bar[field])
            if not _bar_has_executable_trade(bar):
                orders.append(
                    OrderResult(
                        code,
                        0,
                        price,
                        0.0,
                        "%s %s" % (current_date, decision_time),
                        False,
                        "no executable trade at %s" % decision_time,
                    )
                )
                continue
            order = self.broker.order_target_value(
                code,
                target_value,
                price,
                "%s %s" % (current_date, decision_time),
            )
            if order.filled and reason:
                order.reason = reason
            orders.append(order)
        return orders

    def _close_marks(self, current_date):
        marks = {}
        for code in self.broker.positions:
            frame = self.loader.load_daily_frame(code, current_date)
            rows = frame[frame["date"].astype(str) == current_date]
            if rows.empty:
                raise KeyError("No daily close for %s %s" % (code, current_date))
            marks[code] = float(rows.iloc[0]["close"])
        return marks
