# -*- coding: utf-8 -*-
"""Tests for the standalone completed-week/daily-pullback research candidate."""

import math

import pandas as pd
import pytest

from cross_signal_strategy.research import (
    weekly_trend_daily_pullback_candidate as candidate,
)


DECISION_DATE = "2021-03-10"


def _weekly_frame(closes, *, current_week_close=None):
    dates = pd.date_range(end="2021-03-05", periods=len(closes), freq="W-FRI")
    rows = []
    for date, close in zip(dates, closes):
        value = float(close)
        rows.append(
            {
                "date": date,
                "open": value - 0.2,
                "high": value + 0.5,
                "low": value - 0.5,
                "close": value,
                "volume": 1000.0,
            }
        )
    if current_week_close is not None:
        value = float(current_week_close)
        rows.append(
            {
                "date": pd.Timestamp("2021-03-08"),
                "open": value - 0.2,
                "high": value + 0.5,
                "low": value - 0.5,
                "close": value,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_completed_weeks_exclude_the_entire_decision_week():
    current_monday_close = 999.0
    frame = _weekly_frame(range(1, 23), current_week_close=current_monday_close)

    weeks = candidate.aggregate_completed_weeks(frame, DECISION_DATE)

    assert weeks.iloc[-1]["last_trade_date"].date().isoformat() == "2021-03-05"
    assert current_monday_close not in weeks["close"].tolist()


def test_short_holiday_week_is_complete_after_its_calendar_week_ends():
    holiday_frame = pd.DataFrame(
        [
            {
                "date": "2021-02-08",
                "open": 10.0,
                "high": 10.4,
                "low": 9.8,
                "close": 10.2,
            },
            {
                "date": "2021-02-10",
                "open": 10.2,
                "high": 10.8,
                "low": 10.1,
                "close": 10.7,
            },
        ]
    )

    weeks = candidate.aggregate_completed_weeks(holiday_frame, "2021-02-22")

    assert weeks.iloc[-1]["close"] == pytest.approx(10.7)
    assert weeks.iloc[-1]["last_trade_date"].date().isoformat() == "2021-02-10"


def test_weekly_context_needs_21_completed_weeks():
    twenty_week_frame = _weekly_frame(range(1, 21), current_week_close=999.0)

    context, reason = candidate.build_weekly_context(
        twenty_week_frame,
        DECISION_DATE,
    )

    assert context is None
    assert reason == "insufficient_weekly_history"


def test_each_etf_frame_produces_its_own_weekly_gate():
    rising, rising_reason = candidate.build_weekly_context(
        _weekly_frame(range(1, 22), current_week_close=-999.0),
        DECISION_DATE,
    )
    falling, falling_reason = candidate.build_weekly_context(
        _weekly_frame(range(21, 0, -1), current_week_close=999.0),
        DECISION_DATE,
    )

    assert rising_reason is None
    assert falling_reason is None
    assert candidate.weekly_entry_allowed(rising) is True
    assert candidate.weekly_entry_allowed(falling) is False
    assert candidate.weekly_trend_broken(rising) is False
    assert candidate.weekly_trend_broken(falling) is True


def test_weekly_aggregation_rejects_malformed_columns():
    malformed = _weekly_frame(range(1, 22)).drop(columns=["high"])

    with pytest.raises(ValueError, match="missing required columns: high"):
        candidate.aggregate_completed_weeks(malformed, DECISION_DATE)


def test_weekly_aggregation_returns_empty_when_no_week_has_completed():
    current_week_only = pd.DataFrame(
        [
            {
                "date": "2021-03-08",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
            }
        ]
    )

    weeks = candidate.aggregate_completed_weeks(current_week_only, DECISION_DATE)

    assert weeks.empty
    assert list(weeks.columns) == [
        "open",
        "high",
        "low",
        "close",
        "last_trade_date",
    ]


def test_weekly_context_rejects_non_finite_indicator_values():
    frame = _weekly_frame(range(1, 22))
    frame.loc[frame.index[-1], "close"] = math.inf

    context, reason = candidate.build_weekly_context(frame, DECISION_DATE)

    assert context is None
    assert reason == "invalid_weekly_indicator"


@pytest.mark.parametrize(
    "snapshot",
    [
        {"weekly_close": 10.0, "weekly_ma20": 10.0, "weekly_ma20_prev": 9.0},
        {"weekly_close": 11.0, "weekly_ma20": 10.0, "weekly_ma20_prev": 10.0},
    ],
)
def test_weekly_entry_gate_is_strict_at_either_threshold(snapshot):
    assert candidate.weekly_entry_allowed(snapshot) is False


@pytest.mark.parametrize(
    "snapshot",
    [
        {"weekly_close": 10.0, "weekly_ma20": 10.0, "weekly_ma20_prev": 11.0},
        {"weekly_close": 9.0, "weekly_ma20": 10.0, "weekly_ma20_prev": 10.0},
    ],
)
def test_weekly_break_gate_is_strict_at_either_threshold(snapshot):
    assert candidate.weekly_trend_broken(snapshot) is False


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"weekly_close": 11.0, "weekly_ma20": 10.0},
        {"weekly_close": math.nan, "weekly_ma20": 10.0, "weekly_ma20_prev": 9.0},
    ],
)
def test_weekly_predicates_return_false_for_missing_or_non_finite_input(snapshot):
    assert candidate.weekly_entry_allowed(snapshot) is False
    assert candidate.weekly_trend_broken(snapshot) is False
