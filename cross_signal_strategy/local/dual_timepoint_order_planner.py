# -*- coding: utf-8 -*-
"""Share cross-signal portfolio and safety state across two daily batches."""

from __future__ import annotations

from dataclasses import dataclass, field

from cross_signal_strategy.local.local_order_planner import (
    LocalCrossSignalOrderPlanner,
    strategy,
)


@dataclass
class DualTimepointOrderPlanner(LocalCrossSignalOrderPlanner):
    sold_today: set[str] = field(default_factory=set)
    failed_buy_codes: set[str] = field(default_factory=set)
    execution_date: str | None = None
    decision_time: str = "09:35"
    entry_score_snapshots: dict = field(default_factory=dict)
    exit_score_snapshots: dict = field(default_factory=dict)
    score_coverage: dict = field(default_factory=dict)
    _current_date: str | None = None

    def plan_orders_at(
        self,
        current_date,
        previous_date,
        broker,
        decision_time,
        current_prices=None,
    ):
        if self.execution_date != str(current_date):
            self.execution_date = str(current_date)
            self.sold_today.clear()
            self.failed_buy_codes.clear()
        self.decision_time = str(decision_time)[:5]
        if self.decision_time not in {"09:35", "14:45"}:
            raise ValueError("Only 09:35 and 14:45 are allowed")
        self._current_date = str(current_date)

        proposed_orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        blocked = self.sold_today | self.failed_buy_codes
        orders = [
            order
            for order in proposed_orders
            if not (
                float(order["target_value"]) > 0
                and str(order["code"]).split(".")[0] in blocked
            )
        ]
        for order in orders:
            code = str(order["code"]).split(".")[0]
            score = self.last_scores.get(code)
            if score is None:
                continue
            key = (str(current_date), self.decision_time, code)
            if order.get("reason") == "buy_signal":
                self.entry_score_snapshots[key] = dict(score)
            elif order.get("reason") in {"signal_sell", "atr_stop"}:
                self.exit_score_snapshots[key] = dict(score)
        return orders

    def _score_pool(self, current_date):
        scores = []
        for raw_code in self.etf_pool:
            code = str(raw_code).split(".")[0]
            score, reason = self.signal_adapter.score_at(
                code,
                current_date,
                self.decision_time,
                return_reason=True,
            )
            self.score_coverage[(str(current_date), self.decision_time, code)] = (
                "ok" if score is not None else str(reason or "unknown")
            )
            if score is None:
                continue
            item = dict(score)
            item["code"] = code
            if code in self.sold_today or code in self.failed_buy_codes:
                item["buy_allowed"] = False
            scores.append(item)
        return strategy.sort_candidates(scores)

    def _atr_stop_codes(self, broker, current_prices):
        stopped = super()._atr_stop_codes(broker, current_prices)
        return {
            code
            for code in stopped
            if str(self.buy_dates.get(code)) != str(self._current_date)
        }

    def on_orders_processed(
        self, current_date, decision_time, plans, results
    ) -> None:
        super().on_orders_filled(current_date, results)
        plan_by_code = {
            str(item["code"]).split(".")[0]: item for item in plans
        }
        result_by_code = {
            str(item.code).split(".")[0]: item for item in results
        }
        for code, plan in plan_by_code.items():
            result = result_by_code.get(code)
            if (
                float(plan["target_value"]) == 0.0
                and result is not None
                and result.filled
            ):
                self.sold_today.add(code)
            if float(plan["target_value"]) > 0.0 and (
                result is None or not result.filled
            ):
                self.failed_buy_codes.add(code)
