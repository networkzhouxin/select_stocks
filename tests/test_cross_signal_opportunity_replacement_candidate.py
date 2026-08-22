# -*- coding: utf-8 -*-
"""Tests for the full-capacity opportunity-cost replacement candidate."""

from __future__ import annotations

import pathlib
import sys
import types
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def score(code, buy_score, sell_score, **overrides):
    item = {
        "code": code,
        "buy_allowed": True,
        "buy_score": buy_score,
        "reversal_score": buy_score,
        "sell_score": sell_score,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "adx": 20.0,
        "plus_di": 20.0,
        "minus_di": 20.0,
        "ma20_slope_non_negative": False,
        "volume_score": 0,
        "trend_score": 0,
    }
    item.update(overrides)
    return item


def test_selects_highest_sell_risk_holding_for_best_eligible_candidate():
    try:
        from cross_signal_strategy.research.opportunity_replacement_candidate import (
            select_opportunity_replacement,
        )
    except ImportError:
        select_opportunity_replacement = None

    held_codes = ["AAA", "BBB", "CCC"]
    all_scores = [
        score("AAA", buy_score=25, sell_score=35),
        score("BBB", buy_score=20, sell_score=45),
        score("CCC", buy_score=30, sell_score=10),
        score("DDD", buy_score=72, sell_score=0),
        score("EEE", buy_score=65, sell_score=0),
    ]

    decision = None if select_opportunity_replacement is None else select_opportunity_replacement(
        all_scores=all_scores,
        held_codes=held_codes,
        signal_sell_eligible_codes=held_codes,
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is not None
    assert decision.sell_code == "BBB"
    assert decision.buy_code == "DDD"


def test_does_not_replace_healthy_holdings_below_the_official_sell_threshold():
    from cross_signal_strategy.research.opportunity_replacement_candidate import (
        select_opportunity_replacement,
    )

    held_codes = ["AAA", "BBB", "CCC"]
    decision = select_opportunity_replacement(
        all_scores=[
            score("AAA", buy_score=20, sell_score=29),
            score("BBB", buy_score=15, sell_score=18),
            score("CCC", buy_score=10, sell_score=0),
            score("DDD", buy_score=80, sell_score=0),
        ],
        held_codes=held_codes,
        signal_sell_eligible_codes=held_codes,
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is None


def test_requires_every_full_capacity_holding_to_finish_the_five_day_hold():
    from cross_signal_strategy.research.opportunity_replacement_candidate import (
        select_opportunity_replacement,
    )

    decision = select_opportunity_replacement(
        all_scores=[
            score("AAA", buy_score=25, sell_score=35),
            score("BBB", buy_score=20, sell_score=45),
            score("CCC", buy_score=30, sell_score=10),
            score("DDD", buy_score=72, sell_score=0),
        ],
        held_codes=["AAA", "BBB", "CCC"],
        signal_sell_eligible_codes=["AAA", "CCC"],
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is None


def test_does_not_replace_when_the_portfolio_has_an_open_slot():
    from cross_signal_strategy.research.opportunity_replacement_candidate import (
        select_opportunity_replacement,
    )

    decision = select_opportunity_replacement(
        all_scores=[
            score("AAA", buy_score=25, sell_score=35),
            score("BBB", buy_score=20, sell_score=45),
            score("DDD", buy_score=72, sell_score=0),
        ],
        held_codes=["AAA", "BBB"],
        signal_sell_eligible_codes=["AAA", "BBB"],
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is None


def test_only_replaces_a_risk_holding_blocked_by_existing_sell_protection():
    from cross_signal_strategy.research.opportunity_replacement_candidate import (
        select_opportunity_replacement,
    )

    held_codes = ["AAA", "BBB", "CCC"]
    decision = select_opportunity_replacement(
        all_scores=[
            score("AAA", buy_score=25, sell_score=45, close_below_ma20=True),
            score("BBB", buy_score=20, sell_score=10),
            score("CCC", buy_score=30, sell_score=0),
            score("DDD", buy_score=72, sell_score=0),
        ],
        held_codes=held_codes,
        signal_sell_eligible_codes=held_codes,
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is None


def test_breaks_equal_sell_score_ties_with_the_lower_current_buy_score():
    from cross_signal_strategy.research.opportunity_replacement_candidate import (
        select_opportunity_replacement,
    )

    held_codes = ["AAA", "BBB", "CCC"]
    decision = select_opportunity_replacement(
        all_scores=[
            score("AAA", buy_score=50, sell_score=35),
            score("BBB", buy_score=20, sell_score=35),
            score("CCC", buy_score=30, sell_score=0),
            score("DDD", buy_score=72, sell_score=0),
        ],
        held_codes=held_codes,
        signal_sell_eligible_codes=held_codes,
        params={"max_hold": 3, "buy_threshold": 60, "sell_threshold": 30},
    )

    assert decision is not None
    assert decision.sell_code == "BBB"


def test_planner_emits_one_sell_then_its_frozen_replacement_buy():
    try:
        from cross_signal_strategy.research.opportunity_replacement_candidate import (
            OpportunityReplacementOrderPlanner,
        )
    except ImportError:
        OpportunityReplacementOrderPlanner = None

    class Adapter:
        def __init__(self, scores):
            self.scores = {item["code"]: item for item in scores}

        def score(self, code, current_date, return_reason=False):
            result = dict(self.scores[code])
            return (result, None) if return_reason else result

    scores = [
        score("AAA", buy_score=25, sell_score=35),
        score("BBB", buy_score=20, sell_score=45),
        score("CCC", buy_score=30, sell_score=10),
        score("DDD", buy_score=72, sell_score=0),
    ]
    assert OpportunityReplacementOrderPlanner is not None
    planner = OpportunityReplacementOrderPlanner(
        Adapter(scores),
        etf_pool=["AAA", "BBB", "CCC", "DDD"],
        trade_dates=[
            "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07",
            "2019-01-08", "2019-01-09", "2019-01-10",
        ],
        buy_dates={"AAA": "2019-01-02", "BBB": "2019-01-02", "CCC": "2019-01-02"},
    )
    broker = SimpleNamespace(
        cash=1000.0,
        positions={
            "AAA": SimpleNamespace(amount=100, avg_cost=10.0),
            "BBB": SimpleNamespace(amount=100, avg_cost=10.0),
            "CCC": SimpleNamespace(amount=100, avg_cost=10.0),
        },
    )

    orders = planner.plan_orders(
        "2019-01-10",
        "2019-01-09",
        broker,
        current_prices={"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "DDD": 10.0},
    )

    assert [(item["code"], item["reason"]) for item in orders] == [
        ("BBB", "opportunity_replacement"),
        ("DDD", "replacement_buy"),
    ]
    assert orders[0]["target_value"] == 0.0
    assert orders[1]["target_value"] > 0.0
