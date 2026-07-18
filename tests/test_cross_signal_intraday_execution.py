# -*- coding: utf-8 -*-
"""Tests for the single pre-registered minute execution overlay."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def minute_frame(rows):
    defaults = {
        "code": "513100",
        "date": "2019-01-02",
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": 100.0,
        "num_trades": 1.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_policy_is_one_fixed_six_cycle_arrival_price_variant():
    from cross_signal_strategy.local.intraday_execution_overlay import (
        ARRIVAL_TIME,
        DECISION_CYCLES,
        DECISION_INTERVAL_MINUTES,
        FALLBACK_TIME,
    )

    assert ARRIVAL_TIME == "09:35"
    assert DECISION_INTERVAL_MINUTES == 5
    assert DECISION_CYCLES == 6
    assert FALLBACK_TIME == "10:05"


def test_limit_fill_uses_only_later_minutes_and_requires_strict_price_improvement():
    from cross_signal_strategy.local.intraday_execution_overlay import (
        choose_buy_execution,
    )

    frame = minute_frame([
        {"time": "09:35", "low": 9.0, "close": 10.0},
        {"time": "09:36", "low": 10.0, "close": 10.0},
        {"time": "09:37", "low": 9.999, "close": 10.0},
        {"time": "10:05", "low": 10.2, "close": 10.2},
    ])

    execution = choose_buy_execution(frame, "2019-01-02", arrival_price=10.0)

    assert execution.filled is True
    assert execution.route == "passive_limit"
    assert execution.fill_time == "09:37"
    assert execution.raw_price == pytest.approx(10.0)


def test_zero_volume_price_cross_does_not_fill_and_fallback_uses_first_tradable_minute():
    from cross_signal_strategy.local.intraday_execution_overlay import (
        choose_buy_execution,
    )

    frame = minute_frame([
        {"time": "09:35", "close": 10.0},
        {"time": "09:36", "low": 9.9, "volume": 0.0, "num_trades": 0.0},
        {"time": "10:05", "close": 10.2, "volume": 0.0, "num_trades": 0.0},
        {"time": "10:06", "open": 10.3, "high": 10.3, "low": 10.3, "close": 10.3},
    ])

    execution = choose_buy_execution(frame, "2019-01-02", arrival_price=10.0)

    assert execution.filled is True
    assert execution.route == "market_fallback"
    assert execution.fill_time == "10:06"
    assert execution.raw_price == pytest.approx(10.3)


def test_execution_rejects_dates_outside_training_and_never_uses_other_date_rows():
    from cross_signal_strategy.local.intraday_execution_overlay import (
        choose_buy_execution,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        choose_buy_execution(
            minute_frame([{"date": "2022-01-04", "time": "09:35"}]),
            "2022-01-04",
            arrival_price=10.0,
        )

    frame = minute_frame([
        {"date": "2019-01-02", "time": "09:35", "close": 10.0},
        {"date": "2019-01-03", "time": "09:36", "low": 9.0},
        {"date": "2019-01-02", "time": "10:05", "close": 10.2},
    ])
    execution = choose_buy_execution(frame, "2019-01-02", arrival_price=10.0)

    assert execution.route == "market_fallback"
    assert execution.fill_time == "10:05"


def test_execution_report_only_compares_filled_formal_ordinary_buys():
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult
    from cross_signal_strategy.research.intraday_execution_observation import (
        build_intraday_execution_report,
    )

    class Loader:
        def get_minute_bar(self, code, date, trade_time="09:35"):
            return {"close": 10.0, "volume": 100.0, "num_trades": 1.0}

        def load_minute_frame(self, code, date):
            return minute_frame([
                {"code": code, "date": date, "time": "09:35", "close": 10.0},
                {"code": code, "date": date, "time": "09:36", "low": 9.99},
                {"code": code, "date": date, "time": "10:05", "close": 10.2},
            ])

    day = DayResult(
        date="2019-01-02",
        previous_date=None,
        orders=[
            OrderResult("513100", 100, 10.01, 5.0, "2019-01-02 09:35", True, "buy_signal"),
            OrderResult("513500", 100, 10.01, 5.0, "2019-01-02 09:35", False, "buy_signal"),
            OrderResult("159915", -100, 9.99, 5.0, "2019-01-02 09:35", True, "signal_sell"),
        ],
        cash=10000.0,
        positions={},
        marks={},
        total_value=10000.0,
    )

    report = build_intraday_execution_report([day], Loader())

    assert report.eligible_buy_count == 1
    assert report.matched_buy_count == 1
    assert report.limit_fill_count == 1
    assert report.fallback_fill_count == 0
    assert report.observations[0].code == "513100"
    assert report.observations[0].candidate_exec_price == pytest.approx(10.0)
    assert report.observations[0].signed_improvement == pytest.approx(0.01 / 10.01)


def test_gate_requires_positive_improvement_in_every_year_and_etf_group():
    from cross_signal_strategy.research.intraday_execution_observation import (
        ExecutionGroupStats,
        evaluate_intraday_execution_gate,
    )

    passed = evaluate_intraday_execution_gate(
        eligible_buy_count=12,
        matched_buy_count=12,
        overall=ExecutionGroupStats(12, 0.001),
        by_year={
            2019: ExecutionGroupStats(4, 0.001),
            2020: ExecutionGroupStats(4, 0.001),
            2021: ExecutionGroupStats(4, 0.001),
        },
        by_group={
            "qdii": ExecutionGroupStats(6, 0.001),
            "non_qdii": ExecutionGroupStats(6, 0.001),
        },
    )
    failed = evaluate_intraday_execution_gate(
        eligible_buy_count=12,
        matched_buy_count=11,
        overall=ExecutionGroupStats(11, 0.001),
        by_year={
            2019: ExecutionGroupStats(4, 0.001),
            2020: ExecutionGroupStats(4, -0.001),
        },
        by_group={
            "qdii": ExecutionGroupStats(6, 0.001),
            "non_qdii": ExecutionGroupStats(5, -0.001),
        },
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("matched buy count" in reason for reason in failed.reasons)
    assert any("2020 average" in reason for reason in failed.reasons)
    assert any("2021 has no" in reason for reason in failed.reasons)
    assert any("non_qdii average" in reason for reason in failed.reasons)


def test_consumed_observation_remains_reproducible_without_reopening_budget(monkeypatch):
    from cross_signal_strategy.research import intraday_execution_observation as module

    expected = object()

    class Planner:
        plan_orders = object()

    class Engine:
        def __init__(self, loader, initial_cash):
            assert initial_cash == 20000.0

        def run(self, trade_dates, plan_orders):
            assert trade_dates == ["2019-01-02"]
            assert plan_orders is Planner.plan_orders
            return ["baseline"]

    loader = object()
    monkeypatch.setattr(module, "get_training_trade_dates", lambda value: ["2019-01-02"])
    monkeypatch.setattr(module, "build_training_signal_adapter", lambda value: "adapter")
    monkeypatch.setattr(
        module,
        "LocalCrossSignalOrderPlanner",
        lambda adapter, trade_dates: Planner(),
    )
    monkeypatch.setattr(module, "LocalBacktestEngine", Engine)
    monkeypatch.setattr(
        module,
        "build_intraday_execution_report",
        lambda baseline, value: expected,
    )

    assert module.run_training_intraday_execution_observation(loader=loader) is expected
