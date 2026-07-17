# -*- coding: utf-8 -*-
"""Tests for the training-only cross-window single-variable comparison."""

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
    from cross_signal_strategy.research.cross_window_comparison import CrossWindowPerformance

    return CrossWindowPerformance(
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
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult

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


def test_window_params_change_only_cross_window_from_official_v032():
    from cross_signal_strategy.local.local_order_planner import strategy
    from cross_signal_strategy.research.cross_window_comparison import (
        WINDOWS,
        params_for_window,
    )

    baseline = strategy.get_default_params()
    assert WINDOWS == (1, 2, 3, 4)

    for window in WINDOWS:
        candidate = params_for_window(window)
        changed = {
            key: (baseline[key], candidate[key])
            for key in baseline
            if baseline[key] != candidate[key]
        }
        expected = {} if window == 3 else {"cross_window": (3, window)}
        assert changed == expected

    with pytest.raises(ValueError, match="one of"):
        params_for_window(5)


def test_cross_window_gate_is_pre_registered_as_strict_non_degradation():
    from cross_signal_strategy.research.cross_window_comparison import evaluate_window_gate

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

    passed = evaluate_window_gate(
        baseline,
        improved,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )
    failed = evaluate_window_gate(
        baseline,
        performance(
            total_return=1.25,
            annualized_return=0.31,
            max_drawdown=0.09,
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
    assert any("maximum drawdown" in reason for reason in failed.reasons)
    assert any("win rate" in reason for reason in failed.reasons)
    assert any("2021 has no changed" in reason for reason in failed.reasons)
    assert any("2021 candidate annual return" in reason for reason in failed.reasons)


def test_comparison_requires_all_windows_and_counts_path_changes_against_three():
    from cross_signal_strategy.research.cross_window_comparison import (
        build_cross_window_comparison,
    )

    baseline = [
        day("2019-12-31", 11000.0, "AAA"),
        day("2020-12-31", 12100.0),
        day("2021-12-31", 13310.0),
    ]
    results = {
        1: [
            day("2019-12-31", 11200.0, "BBB"),
            day("2020-12-31", 12320.0),
            day("2021-12-31", 13552.0),
        ],
        2: baseline,
        3: baseline,
        4: [
            day("2019-12-31", 10900.0, "AAA"),
            day("2020-12-31", 11990.0, "AAA", amount=-100, reason="signal_sell"),
            day("2021-12-31", 13189.0),
        ],
    }

    report = build_cross_window_comparison(results, initial_cash=10000.0)

    assert tuple(report.variants) == (1, 2, 3, 4)
    assert report.variants[1].changed_order_days == 1
    assert report.variants[1].changed_days_by_year == {2019: 1}
    assert report.variants[2].changed_order_days == 0
    assert report.variants[3].changed_order_days == 0
    assert report.variants[3].gate is None
    assert report.variants[4].changed_order_days == 1
    assert report.variants[4].changed_days_by_year == {2020: 1}
    assert report.variants[1].performance.annual_returns[2019] == pytest.approx(0.12)
    assert report.variants[1].performance.annual_returns[2020] == pytest.approx(0.10)

    with pytest.raises(ValueError, match="exactly"):
        build_cross_window_comparison({1: baseline, 2: baseline, 3: baseline}, 10000.0)


def test_comparison_rejects_nonidentical_or_outside_training_dates():
    from cross_signal_strategy.research.cross_window_comparison import (
        build_cross_window_comparison,
    )

    baseline = [day("2019-12-31", 10000.0)]
    nonidentical = {
        1: baseline,
        2: baseline,
        3: baseline,
        4: [day("2020-01-02", 10000.0)],
    }
    outside = {window: [day("2022-01-04", 10000.0)] for window in (1, 2, 3, 4)}

    with pytest.raises(ValueError, match="identical trading dates"):
        build_cross_window_comparison(nonidentical, 10000.0)
    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_cross_window_comparison(outside, 10000.0)
