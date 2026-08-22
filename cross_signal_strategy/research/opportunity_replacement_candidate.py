# -*- coding: utf-8 -*-
"""Training-only opportunity-cost replacement candidate for cross-v0.3.3."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner


@dataclass(frozen=True)
class OpportunityReplacementDecision:
    sell_code: str
    buy_code: str


@dataclass
class OpportunityReplacementOrderPlanner(LocalCrossSignalOrderPlanner):
    """Local planner that adds one full-capacity replacement after base exits."""

    def plan_orders(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        current_prices: Mapping[str, float] | None = None,
    ):
        orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        if orders:
            return orders

        held_codes = list(broker.positions.keys())
        signal_sell_eligible_codes = [
            code for code in held_codes
            if strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params.get("min_signal_hold_days", 1),
                trade_days=self.trade_dates,
            )
        ]
        decision = select_opportunity_replacement(
            all_scores=list(self.last_scores.values()),
            held_codes=held_codes,
            signal_sell_eligible_codes=signal_sell_eligible_codes,
            params=self.params,
        )
        if decision is None:
            return orders

        incoming = self.last_scores[decision.buy_code]
        total_value = self._total_value(broker, current_prices or {})
        return [
            {
                "code": decision.sell_code,
                "target_value": 0.0,
                "reason": "opportunity_replacement",
            },
            {
                "code": decision.buy_code,
                "target_value": self._scaled_buy_target_value(
                    total_value,
                    incoming,
                    current_date,
                ),
                "reason": "replacement_buy",
            },
        ]


def select_opportunity_replacement(
    all_scores: Sequence[Mapping[str, object]],
    held_codes: Iterable[str],
    signal_sell_eligible_codes: Iterable[str],
    params: Mapping[str, object] | None = None,
) -> OpportunityReplacementDecision | None:
    """Select one blocked-risk holding to replace with the best formal buy."""
    p = strategy.get_default_params()
    p.update(dict(params or {}))
    held = [str(code).split(".")[0] for code in held_codes]
    eligible = {str(code).split(".")[0] for code in signal_sell_eligible_codes}
    if len(held) != int(p["max_hold"]) or set(held) != eligible:
        return None
    ordered_scores = strategy.sort_candidates([dict(item) for item in all_scores])
    candidates = strategy.filter_buy_candidates(ordered_scores, held, p)
    if not candidates:
        return None

    held_set = set(held)
    replaceable = [
        item for item in ordered_scores
        if str(item.get("code", "")).split(".")[0] in held_set
        and float(item.get("sell_score", 0) or 0) >= float(p["sell_threshold"])
        and not strategy.should_force_sell(item, atr_stop_triggered=False, params=p)
    ]
    if not replaceable:
        return None

    outgoing = sorted(
        replaceable,
        key=lambda item: (
            -float(item.get("sell_score", 0) or 0),
            float(item.get("buy_score", 0) or 0),
            str(item.get("code", "")),
        ),
    )[0]
    return OpportunityReplacementDecision(
        sell_code=str(outgoing["code"]).split(".")[0],
        buy_code=str(candidates[0]["code"]).split(".")[0],
    )
