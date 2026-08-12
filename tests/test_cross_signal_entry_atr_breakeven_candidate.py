# -*- coding: utf-8 -*-
"""Tests for the isolated one-entry-ATR break-even candidate."""

from types import SimpleNamespace

import pytest


def _planner():
    from cross_signal_strategy.research.entry_atr_breakeven_candidate import (
        EntryAtrBreakevenPlanner,
    )

    return EntryAtrBreakevenPlanner(signal_adapter=object(), trade_dates=[])


def test_candidate_keeps_official_stop_before_one_entry_atr_profit():
    planner = _planner()
    planner.highest_since_buy["513050"] = 102.9
    planner.entry_atr["513050"] = 3.0
    broker = SimpleNamespace(
        positions={"513050": SimpleNamespace(avg_cost=100.0)},
    )

    assert planner._atr_stop_codes(broker, {"513050": 100.0}) == set()


def test_candidate_floors_stop_at_cost_after_one_entry_atr_profit():
    planner = _planner()
    planner.highest_since_buy["513050"] = 103.0
    planner.entry_atr["513050"] = 3.0
    broker = SimpleNamespace(
        positions={"513050": SimpleNamespace(avg_cost=100.0)},
    )

    assert planner._atr_stop_codes(broker, {"513050": 100.0}) == {"513050"}


def test_candidate_does_not_use_intraday_high_to_activate_break_even():
    planner = _planner()
    planner.highest_since_buy["513050"] = 102.0
    planner.entry_atr["513050"] = 3.0
    broker = SimpleNamespace(
        positions={"513050": SimpleNamespace(avg_cost=100.0)},
    )

    assert planner._atr_stop_codes(broker, {"513050": 100.0}) == set()


def test_candidate_rule_is_fixed_to_one_entry_atr_and_cost_floor():
    from cross_signal_strategy.research.entry_atr_breakeven_candidate import (
        BREAK_EVEN_ACTIVATION_ATR,
        BREAK_EVEN_FLOOR_RETURN,
    )

    assert BREAK_EVEN_ACTIVATION_ATR == pytest.approx(1.0)
    assert BREAK_EVEN_FLOOR_RETURN == pytest.approx(0.0)


def _performance(**overrides):
    from cross_signal_strategy.research.entry_atr_breakeven_candidate import (
        EntryAtrBreakevenPerformance,
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
    return EntryAtrBreakevenPerformance(**values)


def test_candidate_gate_requires_strict_training_period_dominance():
    from cross_signal_strategy.research.entry_atr_breakeven_candidate import (
        evaluate_entry_atr_breakeven_gate,
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

    decision = evaluate_entry_atr_breakeven_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert decision.passed
    assert decision.reasons == ()


def test_candidate_gate_rejects_any_worse_training_year():
    from cross_signal_strategy.research.entry_atr_breakeven_candidate import (
        evaluate_entry_atr_breakeven_gate,
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

    decision = evaluate_entry_atr_breakeven_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert not decision.passed
    assert "2021 candidate annual return worsens" in decision.reasons
