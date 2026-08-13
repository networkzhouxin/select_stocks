# -*- coding: utf-8 -*-
"""Tests for the isolated MACD-death-cross fast-exit candidate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class StaticSignalAdapter:
    def __init__(self, score):
        self._score = dict(score)

    def score(self, code, current_date, return_reason=False):
        result = dict(self._score)
        result["code"] = str(code).split(".")[0]
        return (result, None) if return_reason else result


def _score(**overrides):
    values = {
        "code": "513100",
        "buy_score": 0,
        "sell_score": 10,
        "macd_cross_down": True,
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
        positions={
            "513100": SimpleNamespace(amount=1000, avg_cost=2.0),
        },
        cash=10000.0,
    )


def _planner(score, buy_date="2019-01-02"):
    from cross_signal_strategy.research.macd_fast_exit_candidate import (
        MacdFastExitPlanner,
    )

    trade_dates = [
        "2019-01-02",
        "2019-01-03",
        "2019-01-04",
        "2019-01-07",
        "2019-01-08",
        "2019-01-09",
        "2019-01-10",
    ]
    planner = MacdFastExitPlanner(
        StaticSignalAdapter(score),
        etf_pool=["513100"],
        trade_dates=trade_dates,
    )
    planner.buy_dates["513100"] = buy_date
    planner.highest_since_buy["513100"] = 2.1
    planner.entry_atr["513100"] = 0.05
    return planner


def test_macd_death_cross_exits_after_five_days_without_score_or_structure_confirmation():
    planner = _planner(_score())

    orders = planner.plan_orders(
        "2019-01-09",
        "2019-01-08",
        _broker(),
        {"513100": 2.05},
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "macd_fast_exit"}
    ]


def test_macd_fast_exit_does_not_rebuy_the_same_etf_on_the_sell_day():
    planner = _planner(_score(buy_score=80))

    orders = planner.plan_orders(
        "2019-01-09",
        "2019-01-08",
        _broker(),
        {"513100": 2.05},
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "macd_fast_exit"}
    ]


def test_macd_death_cross_does_not_bypass_five_day_minimum_hold():
    planner = _planner(_score())

    orders = planner.plan_orders(
        "2019-01-08",
        "2019-01-07",
        _broker(),
        {"513100": 2.05},
    )

    assert orders == []


def test_candidate_preserves_official_signal_sell_when_macd_does_not_cross_down():
    planner = _planner(_score(
        macd_cross_down=False,
        sell_score=30,
        close_below_ma20=True,
        adx=10.0,
        plus_di=10.0,
        minus_di=20.0,
        ma20_slope_non_negative=False,
    ))

    orders = planner.plan_orders(
        "2019-01-09",
        "2019-01-08",
        _broker(),
        {"513100": 2.05},
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "signal_sell"}
    ]


def test_candidate_preserves_atr_stop_before_five_day_minimum_hold():
    planner = _planner(_score(macd_cross_down=False))

    orders = planner.plan_orders(
        "2019-01-03",
        "2019-01-02",
        _broker(),
        {"513100": 1.90},
    )

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "atr_stop"}
    ]


def _performance(**overrides):
    from cross_signal_strategy.research.macd_fast_exit_candidate import (
        MacdFastExitPerformance,
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
    return MacdFastExitPerformance(**values)


def test_candidate_gate_requires_strict_training_period_dominance():
    from cross_signal_strategy.research.macd_fast_exit_candidate import (
        evaluate_macd_fast_exit_gate,
    )

    baseline = _performance()
    candidate = _performance(
        total_return=1.05,
        annualized_return=0.26,
        max_drawdown=0.075,
        sharpe_ratio=2.1,
        sortino_ratio=3.1,
        win_rate=0.56,
        profit_loss_ratio=4.1,
        annual_returns={2019: 0.21, 2020: 0.31, 2021: 0.16},
    )

    decision = evaluate_macd_fast_exit_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert decision.passed
    assert decision.reasons == ()


def test_candidate_gate_rejects_any_worse_training_year():
    from cross_signal_strategy.research.macd_fast_exit_candidate import (
        evaluate_macd_fast_exit_gate,
    )

    baseline = _performance()
    candidate = _performance(
        total_return=1.05,
        annualized_return=0.26,
        max_drawdown=0.075,
        sharpe_ratio=2.1,
        sortino_ratio=3.1,
        win_rate=0.56,
        profit_loss_ratio=4.1,
        annual_returns={2019: 0.21, 2020: 0.31, 2021: 0.14},
    )

    decision = evaluate_macd_fast_exit_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert not decision.passed
    assert "2021 candidate annual return worsens" in decision.reasons


def test_candidate_uses_official_five_day_hold_and_three_day_cross_window():
    from cross_signal_strategy.research.macd_fast_exit_candidate import (
        candidate_rule_contract,
    )

    contract = candidate_rule_contract()

    assert contract["min_signal_hold_days"] == 5
    assert contract["cross_window"] == 3
    assert contract["buy_logic_changed"] is False
    assert contract["official_signal_sell_preserved"] is True
    assert contract["atr_stop_preserved"] is True


def test_failed_candidate_is_closed_in_research_budget_and_ledger():
    root = Path(__file__).resolve().parents[1]
    budget_path = root / "cross_signal_strategy" / "docs" / "research_budget.json"
    ledger_path = root / "cross_signal_strategy" / "docs" / "failed_experiments.md"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    families = {item["key"]: item for item in budget["families"]}

    assert budget["expected_failed_experiment_count"] == 57
    family = families["macd_fast_exit_user_authorized"]
    assert family["status"] == "exhausted"
    assert family["max_new_experiments"] == 0
    assert family["candidate_gate_passed"] is False
    assert family["minimum_signal_hold_days"] == 5
    assert family["validation_influence"] == "none"
    assert family["prohibit_alternatives"] is True
    assert "cross-v0.3.2-macd-fast-exit-candidate" in ledger_path.read_text(
        encoding="utf-8"
    )
