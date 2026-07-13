# -*- coding: utf-8 -*-
"""Tests for the isolated MACD(6,13,5) training candidate."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def performance(
    total_return,
    annualized_return,
    max_drawdown,
    sharpe,
    sortino,
    win_rate,
    profit_loss_ratio,
    annual_returns,
):
    from cross_signal_strategy.macd_parameter_candidate import MacdPerformance

    return MacdPerformance(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        buy_count=50,
        sell_count=45,
        annual_returns=annual_returns,
    )


def day(date, total_value, code=None, amount=100, reason="buy_signal"):
    from cross_signal_strategy.local_backtester import DayResult, OrderResult

    orders = []
    if code:
        orders.append(OrderResult(
            code,
            amount,
            10.0,
            0.0,
            "%s 09:35" % date,
            True,
            reason,
        ))
    return DayResult(
        date=date,
        previous_date=None,
        orders=orders,
        cash=float(total_value),
        positions={},
        marks={},
        total_value=float(total_value),
    )


def test_candidate_changes_only_macd_periods_from_official_v032():
    from cross_signal_strategy.local_order_planner import strategy
    from cross_signal_strategy.macd_parameter_candidate import (
        CANDIDATE_VERSION,
        candidate_params,
    )

    baseline = strategy.get_default_params()
    candidate = candidate_params()
    changed = {
        key: (baseline[key], candidate[key])
        for key in baseline
        if baseline[key] != candidate[key]
    }

    assert CANDIDATE_VERSION == "cross-v0.3.2-macd-6-13-5-candidate"
    assert changed == {
        "macd_fast": (12, 6),
        "macd_slow": (26, 13),
        "macd_signal": (9, 5),
    }


def test_macd_gate_is_pre_registered_as_strict_non_degradation():
    from cross_signal_strategy.macd_parameter_candidate import evaluate_macd_gate

    baseline = performance(
        total_return=1.20,
        annualized_return=0.30,
        max_drawdown=0.08,
        sharpe=2.0,
        sortino=3.0,
        win_rate=0.55,
        profit_loss_ratio=4.0,
        annual_returns={2019: 0.20, 2020: 0.50, 2021: 0.10},
    )
    improved = performance(
        total_return=1.25,
        annualized_return=0.31,
        max_drawdown=0.07,
        sharpe=2.1,
        sortino=3.1,
        win_rate=0.56,
        profit_loss_ratio=4.1,
        annual_returns={2019: 0.21, 2020: 0.51, 2021: 0.11},
    )

    passed = evaluate_macd_gate(
        baseline,
        improved,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )
    failed = evaluate_macd_gate(
        baseline,
        performance(
            total_return=1.25,
            annualized_return=0.31,
            max_drawdown=0.07,
            sharpe=2.1,
            sortino=3.1,
            win_rate=0.54,
            profit_loss_ratio=4.1,
            annual_returns={2019: 0.21, 2020: 0.51, 2021: 0.09},
        ),
        changed_days_by_year={2019: 1, 2020: 2},
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("win rate" in reason for reason in failed.reasons)
    assert any("2021 has no changed" in reason for reason in failed.reasons)
    assert any("2021 candidate annual return" in reason for reason in failed.reasons)


def test_macd_comparison_counts_all_filled_order_path_changes():
    from cross_signal_strategy.macd_parameter_candidate import build_macd_comparison

    baseline = [
        day("2019-12-31", 11000.0, "AAA", amount=100),
        day("2020-12-31", 12100.0, "AAA", amount=-100, reason="signal_sell"),
        day("2021-12-31", 13310.0),
    ]
    candidate = [
        day("2019-12-31", 11200.0, "BBB", amount=100),
        day("2020-12-31", 12320.0, "AAA", amount=-100, reason="atr_stop"),
        day("2021-12-31", 13552.0),
    ]

    report = build_macd_comparison(baseline, candidate, initial_cash=10000.0)

    assert report.changed_order_days == 2
    assert report.changed_days_by_year == {2019: 1, 2020: 1}
    assert report.changed_decisions[0].baseline_orders == (("AAA", "buy", "buy_signal"),)
    assert report.changed_decisions[0].candidate_orders == (("BBB", "buy", "buy_signal"),)
    assert report.changed_decisions[1].baseline_orders == (("AAA", "sell", "signal_sell"),)
    assert report.changed_decisions[1].candidate_orders == (("AAA", "sell", "atr_stop"),)
    assert report.baseline_performance.annual_returns[2019] == pytest.approx(0.10)
    assert report.candidate_performance.annual_returns[2020] == pytest.approx(0.10)


def test_macd_comparison_rejects_dates_outside_2019_2021_training_window():
    from cross_signal_strategy.macd_parameter_candidate import build_macd_comparison

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_macd_comparison(
            [day("2022-01-04", 10000.0)],
            [day("2022-01-04", 10000.0)],
            initial_cash=10000.0,
        )
