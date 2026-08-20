# -*- coding: utf-8 -*-
"""Tests for the isolated MACD-free buy and KDJ-only exit candidate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        item = self.scores.get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        result = dict(item)
        return (result, None) if return_reason else result


def _score(**overrides):
    values = {
        "code": "513100",
        "buy_score": 70,
        "reversal_score": 40,
        "sell_score": 35,
        "sell_reversal_score": 20,
        "sell_risk_score": 15,
        "volume_score": 6,
        "trend_score": 10,
        "buy_allowed": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": True,
        "macd_cross_down": True,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 50,
        "plus_di": 40,
        "minus_di": 10,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _planner(score, buy_date="2019-07-01"):
    from cross_signal_strategy.research.macd_free_kdj_exit_candidate import (
        MacdFreeKdjExitPlanner,
    )

    return MacdFreeKdjExitPlanner(
        FakeSignalAdapter({"513100": score}),
        etf_pool=["513100"],
        buy_dates={"513100": buy_date},
        trade_dates=[
            "2019-07-01",
            "2019-07-02",
            "2019-07-03",
            "2019-07-04",
            "2019-07-05",
            "2019-07-08",
        ],
    )


def _broker():
    from cross_signal_strategy.local.local_backtester import LocalBroker, Position

    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["513100"] = Position("513100", 1000, 2.0)
    return broker


def test_candidate_keeps_macd_as_observation_but_removes_its_decision_points():
    from cross_signal_strategy.local.local_order_planner import strategy
    from cross_signal_strategy.research.macd_free_kdj_exit_candidate import (
        make_macd_observation_only,
    )

    result = make_macd_observation_only(_score())

    assert result["observed_macd_cross_up"] is True
    assert result["observed_macd_cross_down"] is True
    assert result["macd_cross_up"] is False
    assert result["macd_cross_down"] is False
    assert result["buy_score"] == 60
    assert result["reversal_score"] == 30
    assert result["sell_score"] == 25
    assert result["sell_reversal_score"] == 10
    assert not strategy.is_blocked_entry_combo(result)


def test_candidate_sells_on_kdj_death_cross_after_five_trading_days_without_confirmation():
    planner = _planner(_score(sell_score=0, sell_reversal_score=0, sell_risk_score=0))

    orders = planner.plan_orders("2019-07-08", "2019-07-05", _broker())

    assert orders == [
        {"code": "513100", "target_value": 0.0, "reason": "kdj_signal_sell"},
    ]


def test_candidate_keeps_five_trading_day_minimum_hold_for_kdj_exit():
    planner = _planner(_score(sell_score=0, sell_reversal_score=0, sell_risk_score=0))

    orders = planner.plan_orders("2019-07-05", "2019-07-04", _broker())

    assert orders == []


def test_candidate_does_not_use_formal_sell_score_without_kdj_death_cross():
    planner = _planner(
        _score(
            kdj_k_cross_down=False,
            kdj_j_cross_down=False,
            sell_score=80,
            sell_reversal_score=60,
            close_below_ma20=True,
            close_below_boll_mid=True,
        )
    )

    orders = planner.plan_orders("2019-07-08", "2019-07-05", _broker())

    assert orders == []


def test_candidate_keeps_atr_stop_independent_of_minimum_hold():
    planner = _planner(_score(kdj_k_cross_down=False, kdj_j_cross_down=False))
    planner.highest_since_buy["513100"] = 10.0
    planner.entry_atr["513100"] = 1.0
    broker = _broker()
    broker.positions["513100"].avg_cost = 9.0

    orders = planner.plan_orders(
        "2019-07-02",
        "2019-07-01",
        broker,
        current_prices={"513100": 8.0},
    )

    assert orders[0] == {
        "code": "513100",
        "target_value": 0.0,
        "reason": "atr_stop",
    }


def test_candidate_rule_is_fixed_and_not_parameter_searched():
    from cross_signal_strategy.research.macd_free_kdj_exit_candidate import (
        CANDIDATE_MIN_SIGNAL_HOLD_DAYS,
        MACD_BUY_POINTS,
        MACD_SELL_POINTS,
    )

    assert CANDIDATE_MIN_SIGNAL_HOLD_DAYS == 5
    assert MACD_BUY_POINTS == pytest.approx(10)
    assert MACD_SELL_POINTS == pytest.approx(10)


def test_candidate_gate_rejects_broad_training_degradation():
    from cross_signal_strategy.research.macd_free_kdj_exit_candidate import (
        MacdFreeKdjPerformance,
        evaluate_macd_free_kdj_gate,
    )

    baseline = MacdFreeKdjPerformance(
        total_return=1.2061,
        annualized_return=0.3027,
        max_drawdown=0.0747,
        sharpe_ratio=2.172,
        sortino_ratio=3.415,
        win_rate=0.5618,
        profit_loss_ratio=4.440,
        buy_count=92,
        sell_count=89,
        annual_returns={2019: 0.3584, 2020: 0.4974, 2021: 0.0846},
    )
    candidate = MacdFreeKdjPerformance(
        total_return=0.4187,
        annualized_return=0.1240,
        max_drawdown=0.0865,
        sharpe_ratio=1.276,
        sortino_ratio=1.924,
        win_rate=0.5595,
        profit_loss_ratio=1.664,
        buy_count=170,
        sell_count=168,
        annual_returns={2019: 0.1021, 2020: 0.3363, 2021: -0.0367},
    )

    decision = evaluate_macd_free_kdj_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 79, 2020: 88, 2021: 89},
    )

    assert not decision.passed
    assert "candidate total return does not improve" in decision.reasons
    assert "candidate profit/loss ratio worsens" in decision.reasons
    assert "2021 candidate annual return worsens" in decision.reasons


def test_failed_candidate_is_closed_in_research_budget_and_ledger():
    root = Path(__file__).resolve().parents[1]
    budget_path = root / "cross_signal_strategy" / "docs" / "research_budget.json"
    ledger_path = root / "cross_signal_strategy" / "docs" / "failed_experiments.md"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    families = {item["key"]: item for item in budget["families"]}

    assert budget["expected_failed_experiment_count"] == 64
    family = families["macd_free_kdj_exit_user_authorized"]
    assert family["status"] == "exhausted"
    assert family["max_new_experiments"] == 0
    assert family["candidate_gate_passed"] is False
    assert family["minimum_signal_hold_days"] == 5
    assert family["validation_influence"] == "none"
    assert family["prohibit_alternatives"] is True
    assert "cross-v0.3.2-macd-free-kdj-exit-candidate" in ledger_path.read_text(
        encoding="utf-8"
    )
