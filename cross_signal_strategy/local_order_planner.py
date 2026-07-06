# -*- coding: utf-8 -*-
"""Convert local cross-signal scores into order plans for LocalBacktestEngine."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy


@dataclass
class LocalCrossSignalOrderPlanner:
    signal_adapter: object
    etf_pool: Iterable[str] | None = None
    params: dict | None = None
    buy_dates: Dict[str, str] = field(default_factory=dict)
    highest_since_buy: Dict[str, float] = field(default_factory=dict)
    entry_atr: Dict[str, float] = field(default_factory=dict)
    last_scores: Dict[str, dict] = field(default_factory=dict)
    trade_dates: List[str] | None = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = strategy.get_default_params()
        if self.etf_pool is None:
            self.etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
        else:
            self.etf_pool = [str(code).split(".")[0] for code in self.etf_pool]
        if self.trade_dates is not None:
            self.trade_dates = [str(day) for day in self.trade_dates]

    def plan_orders(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        current_prices: Mapping[str, float] | None = None,
    ) -> List[Mapping[str, float]]:
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}
        self.last_scores = score_map

        orders: List[Mapping[str, float]] = []
        sold_codes = self._atr_stop_codes(broker, current_prices or {})
        force_stopped = set(sold_codes)
        for code in sorted(sold_codes):
            orders.append({"code": code, "target_value": 0.0, "reason": "atr_stop"})
            self._clear_position_state(code)

        for code in list(broker.positions.keys()):
            if code in sold_codes:
                continue
            score = score_map.get(code)
            if score is None:
                continue
            if not strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params.get("min_signal_hold_days", 1),
                trade_days=self.trade_dates,
            ):
                continue
            if strategy.should_force_sell(score, atr_stop_triggered=False, params=self.params):
                orders.append({"code": code, "target_value": 0.0, "reason": "signal_sell"})
                sold_codes.add(code)
                self._clear_position_state(code)

        held_after_sell = [
            code for code in broker.positions.keys()
            if code not in sold_codes
        ]
        slots = int(self.params["max_hold"]) - len(held_after_sell)
        if slots <= 0:
            return orders

        target_value = self._target_value(broker, current_prices or {})
        candidates = [
            item for item in strategy.filter_buy_candidates(scores, held_after_sell, self.params)
            if item["code"] not in force_stopped
        ]
        bought = 0
        for score in candidates:
            if bought >= slots:
                break
            code = score["code"]
            orders.append({"code": code, "target_value": target_value, "reason": "buy_signal"})
            bought += 1

        return orders

    def _atr_stop_codes(self, broker, current_prices: Mapping[str, float]) -> set:
        stopped = set()
        for code, pos in broker.positions.items():
            if code not in current_prices:
                continue
            highest = self.highest_since_buy.get(code)
            atr_val = self.entry_atr.get(code)
            price = float(current_prices[code])
            if highest is None or atr_val is None or price <= 0:
                continue
            stop_price = strategy.calc_stop_price(highest, atr_val, pos.avg_cost, self.params)
            if round(price, 3) <= round(stop_price, 3):
                stopped.add(code)
        return stopped

    def on_orders_filled(self, current_date: str, orders) -> None:
        for order in orders:
            if not getattr(order, "filled", False):
                continue
            code = str(order.code).split(".")[0]
            if order.amount_delta > 0:
                self.buy_dates[code] = current_date
                self.highest_since_buy[code] = float(order.exec_price)
                score = self.last_scores.get(code)
                if score is None:
                    score, reason = self.signal_adapter.score(code, current_date, return_reason=True)
                if score is not None and score.get("atr") is not None:
                    self.entry_atr[code] = float(score["atr"])
            elif order.amount_delta < 0:
                self._clear_position_state(code)

    def on_after_close(self, current_date: str, marks: Mapping[str, float]) -> None:
        for code, price in marks.items():
            current_high = self.highest_since_buy.get(code, float(price))
            self.highest_since_buy[code] = max(current_high, float(price))

    def _clear_position_state(self, code: str) -> None:
        self.buy_dates.pop(code, None)
        self.highest_since_buy.pop(code, None)
        self.entry_atr.pop(code, None)

    def _score_pool(self, current_date: str) -> List[dict]:
        scores = []
        for code in self.etf_pool:
            score, reason = self.signal_adapter.score(code, current_date, return_reason=True)
            if score is None:
                continue
            score = dict(score)
            score["code"] = str(score.get("code", code)).split(".")[0]
            scores.append(score)
        return strategy.sort_candidates(scores)

    def _target_value(self, broker, current_prices: Mapping[str, float]) -> float:
        position_value = sum(
            pos.amount * float(current_prices.get(code, pos.avg_cost))
            for code, pos in broker.positions.items()
        )
        total_value = broker.cash + position_value
        return total_value * float(self.params["base_ratio"]) / int(self.params["max_hold"])
