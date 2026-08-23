# -*- coding: utf-8 -*-
"""Isolated training-period order planner for the dimension-capped candidate."""

from __future__ import annotations

import sys
import types
from typing import List, Mapping


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.research.dimension_capped_score_candidate import (
    DimensionCappedScoreAdapter,
    is_dimension_capped_buy_candidate,
    should_dimension_capped_signal_sell,
    sort_dimension_capped_candidates,
)


class DimensionCappedOrderPlanner(LocalCrossSignalOrderPlanner):
    """Plan capped-score candidate trades without changing shared planner rules."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params = dict(self.params)
        self.params["min_signal_hold_days"] = 5

    def _score_pool(self, current_date: str) -> List[dict]:
        scores = []
        for code in self.etf_pool:
            score, reason = self.signal_adapter.score(code, current_date, return_reason=True)
            if score is None:
                continue
            score = dict(score)
            score["code"] = str(score.get("code", code)).split(".")[0]
            scores.append(score)
        return sort_dimension_capped_candidates(scores)

    def _should_force_signal_sell(self, score: Mapping[str, object]) -> bool:
        return should_dimension_capped_signal_sell(dict(score))

    def _candidate_target_value(self, broker, current_prices, current_date: str) -> float:
        equal_weight = (
            self._total_value(broker, current_prices)
            * float(self.params["base_ratio"])
            / int(self.params["max_hold"])
        )
        return equal_weight * self._portfolio_atr_stress_buy_scale(current_date)

    def plan_orders(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        current_prices: Mapping[str, float] | None = None,
    ) -> List[Mapping[str, float]]:
        prices = current_prices or {}
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}
        self.last_scores = score_map

        orders: List[Mapping[str, float]] = []
        sold_codes = self._atr_stop_codes(broker, prices)
        force_stopped = set(sold_codes)
        for code in sorted(sold_codes):
            orders.append({"code": code, "target_value": 0.0, "reason": "atr_stop"})

        for code in list(broker.positions.keys()):
            if code in sold_codes:
                continue
            score = score_map.get(code)
            if score is None:
                continue
            if not strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params["min_signal_hold_days"],
                trade_days=self.trade_dates,
            ):
                continue
            if self._should_force_signal_sell(score):
                orders.append({
                    "code": code,
                    "target_value": 0.0,
                    "reason": "dimension_capped_signal_sell",
                })
                sold_codes.add(code)

        held_after_sell = [
            code for code in broker.positions.keys() if code not in sold_codes
        ]
        slots = int(self.params["max_hold"]) - len(held_after_sell)
        if slots <= 0:
            return orders

        candidates = [
            score for score in scores
            if is_dimension_capped_buy_candidate(score, set(held_after_sell))
            and score["code"] not in force_stopped
        ]
        for score in candidates[:slots]:
            orders.append({
                "code": score["code"],
                "target_value": self._candidate_target_value(broker, prices, current_date),
                "reason": "dimension_capped_buy",
            })
        return orders
