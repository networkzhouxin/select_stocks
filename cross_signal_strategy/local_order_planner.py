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

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = strategy.get_default_params()
        if self.etf_pool is None:
            self.etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
        else:
            self.etf_pool = [str(code).split(".")[0] for code in self.etf_pool]

    def plan_orders(self, current_date: str, previous_date: str | None, broker) -> List[Mapping[str, float]]:
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}

        orders: List[Mapping[str, float]] = []
        sold_codes = set()
        for code in list(broker.positions.keys()):
            score = score_map.get(code)
            if score is None:
                continue
            if not strategy.can_sell_by_signal(self.buy_dates.get(code), current_date):
                continue
            if strategy.should_force_sell(score, atr_stop_triggered=False, params=self.params):
                orders.append({"code": code, "target_value": 0.0})
                sold_codes.add(code)
                self.buy_dates.pop(code, None)

        held_after_sell = [
            code for code in broker.positions.keys()
            if code not in sold_codes
        ]
        slots = int(self.params["max_hold"]) - len(held_after_sell)
        if slots <= 0:
            return orders

        target_value = self._target_value(broker)
        candidates = strategy.filter_buy_candidates(scores, held_after_sell, self.params)
        bought = 0
        for score in candidates:
            if bought >= slots:
                break
            code = score["code"]
            orders.append({"code": code, "target_value": target_value})
            self.buy_dates[code] = current_date
            bought += 1

        return orders

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

    def _target_value(self, broker) -> float:
        # Before close marks are available, use cash plus cost basis as a conservative total-value proxy.
        position_value = sum(pos.amount * pos.avg_cost for pos in broker.positions.values())
        total_value = broker.cash + position_value
        return total_value * float(self.params["base_ratio"]) / int(self.params["max_hold"])
