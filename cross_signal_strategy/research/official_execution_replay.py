# -*- coding: utf-8 -*-
"""Replay local signal intent against proven JoinQuant execution evidence.

The local minute source remains untouched.  Strategy intent is still produced
from the frozen local signal path, while actual fills use only the supplied
JoinQuant log evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Iterable, Mapping, Sequence

from cross_signal_strategy.local.local_backtester import (
    DayResult,
    LocalBacktestEngine,
    OrderResult,
    Position,
)
from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent


class OfficialExecutionReplayEngine(LocalBacktestEngine):
    """Validate local order intent, then apply exact official fill evidence."""

    def __init__(
        self,
        loader,
        initial_cash: float,
        intent_events: Sequence[OrderPathEvent],
        fill_events: Sequence[OrderPathEvent],
        execution_time: str = "09:35",
    ) -> None:
        super().__init__(
            loader=loader,
            initial_cash=initial_cash,
            execution_time=execution_time,
        )
        self.intent_events = tuple(intent_events)
        self.fill_events = tuple(fill_events)
        self._validate_evidence()

    def run(self, trade_dates: Iterable[str], order_plan) -> list[DayResult]:
        ordered_dates = [str(day) for day in trade_dates]
        intents_by_date = _events_by_date(self.intent_events)
        fills_by_date = _events_by_date(self.fill_events)
        results: list[DayResult] = []
        previous_date: str | None = None

        for current_date in ordered_dates:
            current_prices = self._current_prices(current_date)
            planned = list(
                self._call_order_plan(
                    order_plan,
                    current_date,
                    previous_date,
                    current_prices,
                )
            )
            official_intents = intents_by_date.get(current_date, ())
            planned_keys = [
                (current_date, _planned_side(item), _code(item["code"]))
                for item in planned
            ]
            expected_keys = [event.as_key() for event in official_intents]
            if _intent_signature(planned_keys) != _intent_signature(expected_keys):
                raise ValueError(
                    "official intent mismatch on %s: expected %r, actual %r"
                    % (current_date, expected_keys, planned_keys)
                )

            planned_by_key = defaultdict(deque)
            for plan, key in zip(planned, planned_keys):
                planned_by_key[key].append(plan)

            available_fills = defaultdict(deque)
            for event in fills_by_date.get(current_date, ()):
                available_fills[event.as_key()].append(event)

            orders: list[OrderResult] = []
            for intent in official_intents:
                plan = planned_by_key[intent.as_key()].popleft()
                queue = available_fills[intent.as_key()]
                if queue:
                    order = self._apply_fill(queue.popleft(), str(plan.get("reason", "")))
                else:
                    order = OrderResult(
                        code=_code(plan["code"]),
                        amount_delta=0,
                        exec_price=float(current_prices.get(_code(plan["code"]), 0.0)),
                        commission=0.0,
                        side_time=f"{current_date} {self.execution_time}",
                        filled=False,
                        reason="official intent was not filled",
                    )
                orders.append(order)

            marks = self._close_marks(current_date)
            owner = getattr(order_plan, "__self__", None)
            if owner is not None and hasattr(owner, "on_orders_filled"):
                owner.on_orders_filled(current_date, orders)
            if owner is not None and hasattr(owner, "on_after_close"):
                owner.on_after_close(current_date, marks)
            positions = {
                code: Position(pos.code, pos.amount, pos.avg_cost)
                for code, pos in self.broker.positions.items()
            }
            results.append(
                DayResult(
                    date=current_date,
                    previous_date=previous_date,
                    orders=orders,
                    cash=self.broker.cash,
                    positions=positions,
                    marks=marks,
                    total_value=self.broker.total_value(marks),
                )
            )
            previous_date = current_date

        unused_dates = sorted(set(intents_by_date) - set(ordered_dates))
        if unused_dates:
            raise ValueError("official intent dates were not replayed: %r" % unused_dates)
        return results

    def _validate_evidence(self) -> None:
        intent_counts = Counter(event.as_key() for event in self.intent_events)
        fill_counts = Counter(event.as_key() for event in self.fill_events)
        extras = {
            key: count - intent_counts[key]
            for key, count in fill_counts.items()
            if count > intent_counts[key]
        }
        if extras:
            raise ValueError("official fills require matching intent evidence: %r" % extras)
        for event in self.fill_events:
            if event.amount is None or int(event.amount) <= 0:
                raise ValueError("official fill amount must be positive")
            if event.price is None or float(event.price) <= 0:
                raise ValueError("official fill price must be positive")
            if event.commission is None or float(event.commission) < 0:
                raise ValueError("official fill commission is required")

    def _apply_fill(self, event: OrderPathEvent, reason: str) -> OrderResult:
        code = _code(event.code)
        amount = int(event.amount or 0)
        price = float(event.price or 0.0)
        commission = float(event.commission or 0.0)
        side_time = f"{event.date} {self.execution_time}"
        current = self.broker.positions.get(code)

        if event.side == "BUY":
            cost = amount * price + commission
            if cost > self.broker.cash + 1e-9:
                raise ValueError("official buy exceeds replay cash on %s %s" % (event.date, code))
            self.broker.cash -= cost
            if current is None:
                self.broker.positions[code] = Position(code, amount, price)
            else:
                old_value = current.amount * current.avg_cost
                current.amount += amount
                current.avg_cost = (old_value + amount * price) / current.amount
            amount_delta = amount
        elif event.side == "SELL":
            if current is None or current.amount < amount:
                raise ValueError("official sell exceeds replay position on %s %s" % (event.date, code))
            self.broker.cash += amount * price - commission
            current.amount -= amount
            if current.amount == 0:
                del self.broker.positions[code]
            amount_delta = -amount
        else:
            raise ValueError("unsupported official side %r" % event.side)

        return OrderResult(
            code=code,
            amount_delta=amount_delta,
            exec_price=price,
            commission=commission,
            side_time=side_time,
            filled=True,
            reason=reason,
        )


def _events_by_date(
    events: Sequence[OrderPathEvent],
) -> Mapping[str, tuple[OrderPathEvent, ...]]:
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event.date)].append(event)
    return {date: tuple(rows) for date, rows in grouped.items()}


def _planned_side(plan: Mapping[str, object]) -> str:
    return "SELL" if float(plan["target_value"]) <= 0.0 else "BUY"


def _intent_signature(keys: Sequence[tuple[str, str, str]]):
    sells = tuple(sorted(key for key in keys if key[1] == "SELL"))
    buys = tuple(key for key in keys if key[1] != "SELL")
    return sells, buys


def _code(value: object) -> str:
    return str(value).split(".")[0]
