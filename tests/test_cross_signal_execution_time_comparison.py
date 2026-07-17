# -*- coding: utf-8 -*-
"""Tests for the pre-registered 09:35 versus 10:00 training experiment."""

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
    from cross_signal_strategy.research.execution_time_comparison import ExecutionTimePerformance

    return ExecutionTimePerformance(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        buy_count=10,
        sell_count=9,
        annual_returns=annual_returns,
    )


def day(date, total_value, price=None, code="513100", amount=100, reason="buy_signal"):
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult

    orders = []
    if price is not None:
        orders.append(OrderResult(
            code=code,
            amount_delta=amount,
            exec_price=float(price),
            commission=0.0,
            side_time="%s 09:35" % date,
            filled=True,
            reason=reason,
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


def test_execution_time_family_is_locked_to_one_preregistered_candidate():
    from cross_signal_strategy.research.execution_time_comparison import (
        BASELINE_TIME,
        CANDIDATE_TIME,
        EXECUTION_TIMES,
    )

    assert BASELINE_TIME == "09:35"
    assert CANDIDATE_TIME == "10:00"
    assert EXECUTION_TIMES == ("09:35", "10:00")


def test_execution_time_gate_requires_cross_year_and_cross_group_improvement():
    from cross_signal_strategy.research.execution_time_comparison import (
        ExecutionPriceStats,
        evaluate_execution_time_gate,
    )

    baseline = performance(
        1.20, 0.30, 0.08, 2.0, 3.0, 0.55, 4.0,
        {2019: 0.20, 2020: 0.50, 2021: 0.10},
    )
    candidate = performance(
        1.25, 0.31, 0.07, 2.1, 3.1, 0.56, 4.1,
        {2019: 0.21, 2020: 0.51, 2021: 0.11},
    )
    stable_prices = ExecutionPriceStats(
        matched_orders=12,
        average_signed_improvement=0.001,
        matched_by_year={2019: 4, 2020: 4, 2021: 4},
        average_by_year={2019: 0.001, 2020: 0.001, 2021: 0.001},
        matched_by_group={"qdii": 6, "non_qdii": 6},
        average_by_group={"qdii": 0.001, "non_qdii": 0.001},
    )

    passed = evaluate_execution_time_gate(baseline, candidate, stable_prices)
    failed = evaluate_execution_time_gate(
        baseline,
        performance(
            1.25, 0.31, 0.09, 2.1, 3.1, 0.54, 4.1,
            {2019: 0.21, 2020: 0.51, 2021: 0.09},
        ),
        ExecutionPriceStats(
            matched_orders=8,
            average_signed_improvement=0.001,
            matched_by_year={2019: 4, 2020: 4},
            average_by_year={2019: 0.001, 2020: 0.001},
            matched_by_group={"qdii": 4, "non_qdii": 4},
            average_by_group={"qdii": 0.001, "non_qdii": -0.001},
        ),
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("maximum drawdown" in reason for reason in failed.reasons)
    assert any("win rate" in reason for reason in failed.reasons)
    assert any("2021 has no matched" in reason for reason in failed.reasons)
    assert any("2021 candidate annual return" in reason for reason in failed.reasons)
    assert any("non_qdii average execution" in reason for reason in failed.reasons)


def test_comparison_measures_side_adjusted_prices_and_path_changes():
    from cross_signal_strategy.research.execution_time_comparison import (
        build_execution_time_comparison,
    )

    baseline = [
        day("2019-12-31", 11000.0, 10.0, code="513100", amount=100),
        day("2020-12-31", 12100.0, 10.0, code="159915", amount=-100, reason="signal_sell"),
        day("2021-12-31", 13310.0),
    ]
    candidate = [
        day("2019-12-31", 11200.0, 9.9, code="513100", amount=100),
        day("2020-12-31", 12320.0, 10.1, code="159915", amount=-100, reason="signal_sell"),
        day("2021-12-31", 13552.0, 8.0, code="513500", amount=100),
    ]

    report = build_execution_time_comparison(
        {"09:35": baseline, "10:00": candidate},
        initial_cash=10000.0,
    )

    assert report.price_stats.matched_orders == 2
    assert report.price_stats.average_signed_improvement == pytest.approx(0.01)
    assert report.price_stats.matched_by_year == {2019: 1, 2020: 1}
    assert report.price_stats.matched_by_group == {"non_qdii": 1, "qdii": 1}
    assert report.changed_order_days == 1
    assert report.changed_days_by_year == {2021: 1}
    assert report.candidate_performance.annual_returns[2019] == pytest.approx(0.12)


def test_comparison_rejects_unregistered_times_and_dates_outside_training():
    from cross_signal_strategy.research.execution_time_comparison import (
        build_execution_time_comparison,
    )

    valid = [day("2019-12-31", 10000.0)]
    with pytest.raises(ValueError, match="exactly"):
        build_execution_time_comparison({"09:35": valid, "09:45": valid}, 10000.0)

    outside = [day("2022-01-04", 10000.0)]
    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_execution_time_comparison({"09:35": outside, "10:00": outside}, 10000.0)
