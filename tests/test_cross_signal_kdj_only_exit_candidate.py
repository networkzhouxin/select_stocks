# -*- coding: utf-8 -*-
"""Tests for the isolated KDJ-only ordinary-exit candidate."""

import json
from pathlib import Path

from types import SimpleNamespace

import pytest


class StaticSignalAdapter:
    def __init__(self, scores):
        self.scores = {str(code): dict(score) for code, score in scores.items()}

    def score(self, code, current_date, return_reason=False):
        result = dict(self.scores[str(code).split(".")[0]])
        result["code"] = str(code).split(".")[0]
        return (result, None) if return_reason else result


def _score(code="513100", **overrides):
    values = {
        "code": code,
        "buy_score": 0,
        "sell_score": 0,
        "buy_allowed": True,
        "macd_cross_up": True,
        "macd_cross_down": False,
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
        "kdj_k_cross_down": True,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "volume_score": 0,
        "trend_score": 0,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "adx": 35.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "ma20_slope_non_negative": True,
        "atr": 0.05,
    }
    values.update(overrides)
    return values


def _broker():
    return SimpleNamespace(
        positions={"513100": SimpleNamespace(amount=1000, avg_cost=2.0)},
        cash=10000.0,
    )


def _planner(scores=None, buy_date="2019-01-02"):
    from cross_signal_strategy.research.kdj_only_exit_candidate import (
        KdjOnlyExitPlanner,
    )

    scores = scores or {"513100": _score()}
    planner = KdjOnlyExitPlanner(
        StaticSignalAdapter(scores),
        etf_pool=list(scores),
        trade_dates=[
            "2019-01-02",
            "2019-01-03",
            "2019-01-04",
            "2019-01-07",
            "2019-01-08",
            "2019-01-09",
            "2019-01-10",
        ],
    )
    planner.buy_dates["513100"] = buy_date
    planner.highest_since_buy["513100"] = 2.1
    planner.entry_atr["513100"] = 0.05
    return planner


def test_kdj_death_cross_sells_after_five_days_without_score_or_structure():
    planner = _planner()

    orders = planner.plan_orders(
        "2019-01-09", "2019-01-08", _broker(), {"513100": 2.05}
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "kdj_only_exit"}
    ]


def test_kdj_death_cross_does_not_bypass_five_day_hold():
    planner = _planner()

    orders = planner.plan_orders(
        "2019-01-08", "2019-01-07", _broker(), {"513100": 2.05}
    )

    assert orders == []


def test_official_sell_conditions_do_not_sell_without_kdj_death_cross():
    planner = _planner({
        "513100": _score(
            kdj_k_cross_down=False,
            sell_score=80,
            close_below_ma20=True,
            close_below_boll_mid=True,
            adx=10.0,
            plus_di=10.0,
            minus_di=20.0,
            ma20_slope_non_negative=False,
        )
    })

    orders = planner.plan_orders(
        "2019-01-09", "2019-01-08", _broker(), {"513100": 2.05}
    )

    assert orders == []


def test_atr_stop_remains_available_before_five_days():
    planner = _planner({"513100": _score(kdj_k_cross_down=False)})

    orders = planner.plan_orders(
        "2019-01-03", "2019-01-02", _broker(), {"513100": 1.90}
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "atr_stop"}
    ]


def test_same_day_kdj_sell_cannot_rebuy_the_same_etf_but_can_fill_next_candidate():
    planner = _planner({
        "513100": _score(buy_score=80),
        "518880": _score(code="518880", buy_score=70, kdj_k_cross_down=False),
    })

    orders = planner.plan_orders(
        "2019-01-09",
        "2019-01-08",
        _broker(),
        {"513100": 2.05, "518880": 3.0},
    )

    assert orders[0] == {
        "code": "513100",
        "target_value": 0.0,
        "reason": "kdj_only_exit",
    }
    assert all(order["code"] != "513100" for order in orders[1:])
    assert any(order["code"] == "518880" for order in orders[1:])


def test_candidate_contract_keeps_buy_logic_and_macd_decision_role():
    from cross_signal_strategy.research.kdj_only_exit_candidate import (
        candidate_rule_contract,
    )

    contract = candidate_rule_contract()

    assert contract == {
        "min_signal_hold_days": 5,
        "cross_window": 3,
        "buy_logic_changed": False,
        "macd_buy_score_preserved": True,
        "ordinary_exit_trigger": "recent_kdj_k_cross_down",
        "atr_stop_preserved": True,
    }


def _performance(**overrides):
    from cross_signal_strategy.research.kdj_only_exit_candidate import (
        KdjOnlyExitPerformance,
    )

    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.08,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "buy_count": 90,
        "sell_count": 88,
        "annual_returns": {2019: 0.20, 2020: 0.30, 2021: 0.15},
    }
    values.update(overrides)
    return KdjOnlyExitPerformance(**values)


def test_candidate_gate_rejects_any_worse_training_year():
    from cross_signal_strategy.research.kdj_only_exit_candidate import (
        evaluate_kdj_only_exit_gate,
    )

    decision = evaluate_kdj_only_exit_gate(
        _performance(),
        _performance(
            total_return=1.05,
            annualized_return=0.26,
            max_drawdown=0.07,
            sharpe_ratio=2.1,
            sortino_ratio=3.1,
            win_rate=0.56,
            profit_loss_ratio=4.1,
            annual_returns={2019: 0.21, 2020: 0.31, 2021: 0.14},
        ),
        {2019: 1, 2020: 1, 2021: 1},
    )

    assert not decision.passed
    assert "2021 candidate annual return worsens" in decision.reasons


def test_candidate_gate_accepts_only_broad_training_dominance():
    from cross_signal_strategy.research.kdj_only_exit_candidate import (
        evaluate_kdj_only_exit_gate,
    )

    decision = evaluate_kdj_only_exit_gate(
        _performance(),
        _performance(
            total_return=1.05,
            annualized_return=0.26,
            max_drawdown=0.07,
            sharpe_ratio=2.1,
            sortino_ratio=3.1,
            win_rate=0.56,
            profit_loss_ratio=4.1,
            annual_returns={2019: 0.21, 2020: 0.31, 2021: 0.16},
        ),
        {2019: 1, 2020: 1, 2021: 1},
    )

    assert decision.passed
    assert decision.reasons == ()


def test_failed_candidate_is_closed_in_research_budget_and_ledger():
    root = Path(__file__).resolve().parents[1]
    budget = json.loads(
        (root / "cross_signal_strategy" / "docs" / "research_budget.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = (
        root / "cross_signal_strategy" / "docs" / "failed_experiments.md"
    ).read_text(encoding="utf-8")
    families = {item["key"]: item for item in budget["families"]}

    assert budget["expected_failed_experiment_count"] == 71
    family = families["kdj_only_exit_user_authorized"]
    assert family["status"] == "exhausted"
    assert family["max_new_experiments"] == 0
    assert family["candidate_gate_passed"] is False
    assert family["minimum_signal_hold_days"] == 5
    assert family["validation_influence"] == "none"
    assert family["prohibit_alternatives"] is True
    assert "cross-v0.3.2-kdj-only-exit-candidate" in ledger
