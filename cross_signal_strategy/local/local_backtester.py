# -*- coding: utf-8 -*-
"""Minimal local event-backtest primitives for cross_signal_strategy."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Dict, Iterable, List, Mapping, Sequence


LOT_SIZE = 100


@dataclass
class Position:
    code: str
    amount: int
    avg_cost: float


@dataclass
class OrderResult:
    code: str
    amount_delta: int
    exec_price: float
    commission: float
    side_time: str
    filled: bool
    reason: str = ""


@dataclass
class DayResult:
    date: str
    previous_date: str | None
    orders: List[OrderResult]
    cash: float
    positions: Dict[str, Position]
    marks: Dict[str, float]
    total_value: float


class LocalBroker:
    """Small broker model matching the strategy's basic JoinQuant settings."""

    def __init__(
        self,
        initial_cash: float,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        slippage_rate: float = 0.001,
        price_tick: float = 0.001,
        lot_size: int = LOT_SIZE,
    ) -> None:
        self.cash = float(initial_cash)
        self.commission_rate = float(commission_rate)
        self.min_commission = float(min_commission)
        self.slippage_rate = float(slippage_rate)
        self.price_tick = float(price_tick)
        self.lot_size = int(lot_size)
        self.positions: Dict[str, Position] = {}

    def _commission(self, trade_value: float) -> float:
        return max(self.min_commission, abs(trade_value) * self.commission_rate)

    def _round_lot(self, amount: int) -> int:
        return int(amount // self.lot_size) * self.lot_size

    def _buy_exec_price(self, price: float) -> float:
        return self._round_price(float(price) * (1.0 + self.slippage_rate))

    def _sell_exec_price(self, price: float) -> float:
        return self._round_price(float(price) * (1.0 - self.slippage_rate))

    def _round_price(self, price: float) -> float:
        if self.price_tick <= 0:
            return float(price)
        return round(round(float(price) / self.price_tick) * self.price_tick, 3)

    def order_target_value(
        self,
        code: str,
        target_value: float,
        price: float,
        side_time: str,
    ) -> OrderResult:
        if price <= 0:
            return OrderResult(code, 0, 0.0, 0.0, side_time, False, "invalid price")

        current = self.positions.get(code)
        current_amount = current.amount if current else 0
        target_amount = self._round_lot(int(float(target_value) / float(price)))
        amount_delta = target_amount - current_amount

        if amount_delta == 0:
            return OrderResult(code, 0, float(price), 0.0, side_time, False, "no change")

        if amount_delta > 0:
            return self._buy(code, current, amount_delta, price, side_time)
        return self._sell(code, current, amount_delta, price, side_time)

    def _buy(
        self,
        code: str,
        current: Position | None,
        amount_delta: int,
        price: float,
        side_time: str,
    ) -> OrderResult:
        exec_price = self._buy_exec_price(price)
        trade_value = amount_delta * exec_price
        commission = self._commission(trade_value)
        total_cost = trade_value + commission

        while amount_delta > 0 and total_cost > self.cash:
            amount_delta -= self.lot_size
            trade_value = amount_delta * exec_price
            commission = self._commission(trade_value) if amount_delta > 0 else 0.0
            total_cost = trade_value + commission

        if amount_delta <= 0:
            return OrderResult(code, 0, exec_price, 0.0, side_time, False, "insufficient cash")

        self.cash -= total_cost
        if current is None:
            self.positions[code] = Position(code, amount_delta, exec_price)
        else:
            old_value = current.amount * current.avg_cost
            new_value = amount_delta * exec_price
            new_amount = current.amount + amount_delta
            current.avg_cost = (old_value + new_value) / new_amount
            current.amount = new_amount
        return OrderResult(code, amount_delta, exec_price, commission, side_time, True)

    def _sell(
        self,
        code: str,
        current: Position | None,
        amount_delta: int,
        price: float,
        side_time: str,
    ) -> OrderResult:
        if current is None or current.amount <= 0:
            return OrderResult(code, 0, self._sell_exec_price(price), 0.0, side_time, False, "no position")

        sell_amount = min(abs(amount_delta), current.amount)
        sell_amount = self._round_lot(sell_amount)
        if sell_amount <= 0:
            return OrderResult(code, 0, self._sell_exec_price(price), 0.0, side_time, False, "no lot")

        exec_price = self._sell_exec_price(price)
        trade_value = sell_amount * exec_price
        commission = self._commission(trade_value)
        self.cash += trade_value - commission
        current.amount -= sell_amount
        if current.amount <= 0:
            del self.positions[code]
        return OrderResult(code, -sell_amount, exec_price, commission, side_time, True)

    def total_value(self, marks: Mapping[str, float]) -> float:
        value = self.cash
        for code, pos in self.positions.items():
            value += pos.amount * float(marks.get(code, pos.avg_cost))
        return value


OrderPlan = Callable[[str, str | None, LocalBroker], Sequence[Mapping[str, object]]]


class LocalBacktestEngine:
    """Minimal daily loop using a fixed T-day execution time and close marks."""

    def __init__(
        self,
        loader,
        initial_cash: float,
        execution_time: str = "09:35",
        broker_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.loader = loader
        self.broker = LocalBroker(
            initial_cash=initial_cash,
            **dict(broker_kwargs or {}),
        )
        self.execution_time = str(execution_time)[:5]

    def run(self, trade_dates: Iterable[str], order_plan: OrderPlan) -> List[DayResult]:
        ordered_dates = [str(d) for d in trade_dates]
        results: List[DayResult] = []
        previous_date: str | None = None

        for current_date in ordered_dates:
            current_prices = self._current_prices(current_date)
            planned_orders = self._call_order_plan(order_plan, current_date, previous_date, current_prices)
            owner = getattr(order_plan, "__self__", None)
            max_holdings = _planner_max_holdings(owner)
            orders: List[OrderResult] = []
            for plan in planned_orders:
                code = str(plan["code"])
                target_value = float(plan["target_value"])
                plan_reason = str(plan.get("reason", ""))
                if (
                    target_value > 0.0
                    and code not in self.broker.positions
                    and max_holdings is not None
                    and len(self.broker.positions) >= max_holdings
                ):
                    orders.append(OrderResult(
                        code=code,
                        amount_delta=0,
                        exec_price=0.0,
                        commission=0.0,
                        side_time=f"{current_date} {self.execution_time}",
                        filled=False,
                        reason="no available holding slot after execution",
                    ))
                    continue
                try:
                    execution_bar = self.loader.get_minute_bar(
                        code,
                        current_date,
                        self.execution_time,
                    )
                except (FileNotFoundError, KeyError):
                    orders.append(OrderResult(
                        code=code,
                        amount_delta=0,
                        exec_price=0.0,
                        commission=0.0,
                        side_time=f"{current_date} {self.execution_time}",
                        filled=False,
                        reason=f"missing execution bar at {self.execution_time}",
                    ))
                    continue
                price = float(execution_bar["close"])
                if not _bar_has_executable_trade(execution_bar):
                    orders.append(OrderResult(
                        code=code,
                        amount_delta=0,
                        exec_price=price,
                        commission=0.0,
                        side_time=f"{current_date} {self.execution_time}",
                        filled=False,
                        reason=f"no executable trade at {self.execution_time}",
                    ))
                    continue
                order = self.broker.order_target_value(
                    code=code,
                    target_value=target_value,
                    price=price,
                    side_time=f"{current_date} {self.execution_time}",
                )
                if order.filled and plan_reason:
                    order.reason = plan_reason
                orders.append(order)

            marks = self._close_marks(current_date)
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

        return results

    def _call_order_plan(self, order_plan, current_date: str, previous_date: str | None, current_prices):
        try:
            params = inspect.signature(order_plan).parameters
            if "current_prices" in params:
                return order_plan(current_date, previous_date, self.broker, current_prices=current_prices)
        except (TypeError, ValueError):
            pass
        return order_plan(current_date, previous_date, self.broker)

    def _current_prices(self, current_date: str) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        candidates = set(self.broker.positions.keys())
        for code in candidates:
            try:
                prices[code] = float(
                    self.loader.get_minute_bar(code, current_date, self.execution_time)["close"]
                )
            except (FileNotFoundError, KeyError):
                continue
        return prices

    def _close_marks(self, current_date: str) -> Dict[str, float]:
        marks: Dict[str, float] = {}
        for code in self.broker.positions:
            frame = self.loader.load_daily_frame(code, current_date)
            rows = frame[frame["date"].astype(str) == current_date]
            if rows.empty:
                raise KeyError(f"No daily close for {code} {current_date}")
            marks[code] = float(rows.iloc[0]["close"])
        return marks


def _bar_has_executable_trade(bar: Mapping[str, object]) -> bool:
    def numeric(field: str) -> float:
        try:
            return float(bar.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return numeric("volume") > 0.0 or numeric("num_trades") > 0.0


def _planner_max_holdings(owner) -> int | None:
    params = getattr(owner, "params", None)
    if not isinstance(params, Mapping):
        return None
    try:
        value = int(params.get("max_hold", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
