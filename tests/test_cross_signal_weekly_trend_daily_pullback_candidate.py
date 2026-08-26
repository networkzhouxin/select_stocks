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


def _eligible_snapshot(**overrides):
    values = {
        "code": "159915",
        "weekly_close": 11.0,
        "weekly_ma20": 10.0,
        "weekly_ma20_prev": 9.0,
        "close": 10.0,
        "boll_lower": 9.7,
        "boll_mid": 10.0,
        "k_prev": 19.0,
        "d_prev": 20.0,
        "k": 22.0,
        "d": 20.0,
        "rsi6_prev": 39.0,
        "rsi6": 40.0,
        "atr": 0.2,
    }
    values.update(overrides)
    return values


def _daily_death_cross_snapshot(**overrides):
    values = _eligible_snapshot(
        close=9.8,
        boll_mid=10.0,
        k_prev=21.0,
        d_prev=20.0,
        k=19.0,
        d=20.0,
    )
    values.update(overrides)
    return values


def test_entry_requires_weekly_gate_and_every_daily_condition():
    assert candidate.is_entry_eligible(_eligible_snapshot()) is True
    for broken in (
        {"weekly_close": 9.9, "weekly_ma20": 10.0},
        {"close": 9.7, "boll_lower": 9.7},
        {"close": 10.6, "boll_mid": 10.5},
        {"k_prev": 21.0, "d_prev": 20.0},
        {"k": 20.0, "d": 20.0},
        {"rsi6": 40.0, "rsi6_prev": 40.0},
        {"rsi6": 50.01},
    ):
        assert candidate.is_entry_eligible(_eligible_snapshot(**broken)) is False


def test_entry_equality_semantics_are_frozen():
    assert candidate.is_entry_eligible(
        _eligible_snapshot(close=10.0, boll_mid=10.0)
    ) is True
    assert candidate.is_entry_eligible(
        _eligible_snapshot(k_prev=20.0, d_prev=20.0)
    ) is True
    assert candidate.is_entry_eligible(_eligible_snapshot(rsi6=50.0)) is True


@pytest.mark.parametrize(
    "field",
    [
        "close",
        "boll_lower",
        "boll_mid",
        "k_prev",
        "d_prev",
        "k",
        "d",
        "rsi6_prev",
        "rsi6",
    ],
)
@pytest.mark.parametrize("invalid", [None, math.nan, math.inf])
def test_daily_entry_rejects_every_missing_or_non_finite_field(field, invalid):
    snapshot = _eligible_snapshot()
    if invalid is None:
        snapshot.pop(field)
    else:
        snapshot[field] = invalid

    assert candidate.is_daily_entry_eligible(snapshot) is False
    assert candidate.is_entry_eligible(snapshot) is False


def test_buy_queue_ranks_weekly_strength_then_kd_then_pool_order():
    weak_week = _eligible_snapshot(
        code="159915",
        weekly_close=11.0,
        weekly_ma20=10.0,
        k=24.0,
        d=20.0,
    )
    strong_small_cross = _eligible_snapshot(
        code="518880",
        weekly_close=12.0,
        weekly_ma20=10.0,
        k=21.0,
        d=20.0,
    )
    strong_large_cross = _eligible_snapshot(
        code="513100",
        weekly_close=12.0,
        weekly_ma20=10.0,
        k=23.0,
        d=20.0,
    )
    excluded = _eligible_snapshot(
        code="513500",
        weekly_close=13.0,
        weekly_ma20=10.0,
    )

    queue = candidate.build_buy_queue(
        [weak_week, strong_small_cross, strong_large_cross, excluded],
        excluded_codes={"513500"},
        etf_pool=["159915", "513100", "513500", "518880"],
    )

    assert [item["code"] for item in queue] == ["513100", "518880", "159915"]


def test_buy_queue_uses_pool_order_for_exact_ties_and_ignores_ineligible_codes():
    pool = ["159915", "513100", "518880"]
    tied_later = _eligible_snapshot(code="518880")
    tied_first = _eligible_snapshot(code="159915")
    ineligible = _eligible_snapshot(code="513100", rsi6=60.0)
    outside_pool = _eligible_snapshot(code="999999")

    queue = candidate.build_buy_queue(
        [tied_later, outside_pool, ineligible, tied_first],
        excluded_codes=set(),
        etf_pool=pool,
    )

    assert [item["code"] for item in queue] == ["159915", "518880"]


def test_exit_priority_is_atr_then_weekly_break_then_daily_failure():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.2, 11.0)
    broken_week_and_daily_death_cross = _daily_death_cross_snapshot(
        weekly_close=9.0,
        weekly_ma20=10.0,
        weekly_ma20_prev=11.0,
    )
    allowed_week_and_daily_death_cross = _daily_death_cross_snapshot()

    assert candidate.choose_exit_reason(
        state,
        broken_week_and_daily_death_cross,
        10.44,
        8,
        "513100",
    ) == "atr_stop"
    assert candidate.choose_exit_reason(
        state,
        broken_week_and_daily_death_cross,
        10.80,
        8,
        "513100",
    ) == "weekly_trend_break"
    assert candidate.choose_exit_reason(
        state,
        allowed_week_and_daily_death_cross,
        10.80,
        8,
        "513100",
    ) == "daily_pullback_failure"


def test_daily_failure_waits_five_sessions_but_weekly_break_does_not():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.2, 11.0)
    allowed_week_and_daily_death_cross = _daily_death_cross_snapshot()
    broken_week_and_daily_death_cross = _daily_death_cross_snapshot(
        weekly_close=9.0,
        weekly_ma20=10.0,
        weekly_ma20_prev=11.0,
    )

    assert candidate.choose_exit_reason(
        state,
        allowed_week_and_daily_death_cross,
        10.80,
        4,
        "513100",
    ) is None
    assert candidate.choose_exit_reason(
        state,
        broken_week_and_daily_death_cross,
        10.80,
        1,
        "513100",
    ) == "weekly_trend_break"


def test_upper_band_touch_is_not_an_exit():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.2, 11.0)
    upper_touch = _eligible_snapshot(close=12.0, boll_mid=10.0, boll_upper=12.0)

    assert candidate.choose_exit_reason(
        state,
        upper_touch,
        10.80,
        20,
        "513100",
    ) is None


def test_frozen_atr_stop_uses_entry_price_clamp_and_gold_three_percent_floor():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.01, 10.0)

    assert candidate.calc_frozen_atr_stop(state, "513100") == pytest.approx(9.5)
    assert candidate.calc_frozen_atr_stop(state, "518880.XSHG") == pytest.approx(9.7)


def test_highest_close_updates_only_from_positive_finite_t1_close():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.2, 10.5)

    candidate.update_highest_close_from_t1(state, 10.8)
    candidate.update_highest_close_from_t1(state, 10.2)
    candidate.update_highest_close_from_t1(state, math.inf)

    assert state.highest_close == pytest.approx(10.8)


def test_invalid_atr_state_fails_closed_without_hiding_weekly_exit():
    invalid_state = candidate.PositionSignalState("2021-01-04", 0.0, math.nan, 0.0)
    broken_week = _eligible_snapshot(
        weekly_close=9.0,
        weekly_ma20=10.0,
        weekly_ma20_prev=11.0,
    )

    assert math.isnan(candidate.calc_frozen_atr_stop(invalid_state, "513100"))
    assert candidate.choose_exit_reason(
        invalid_state,
        broken_week,
        9.0,
        1,
        "513100",
    ) == "weekly_trend_break"
