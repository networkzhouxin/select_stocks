# -*- coding: utf-8 -*-
"""Tests for the standalone completed-week/daily-pullback research candidate."""

import math
from types import SimpleNamespace

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


class _PlannerAdapter:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []

    def score(self, code, current_date, return_reason=False):
        self.calls.append((code, current_date, return_reason))
        value = self.snapshots.get(code)
        if value is None:
            result = (None, "stale_signal_date")
        else:
            result = (dict(value, code=code), None)
        return result if return_reason else result[0]


def _position(amount=100, avg_cost=10.0):
    return SimpleNamespace(amount=amount, avg_cost=avg_cost)


def _broker(cash, positions=None):
    held = dict(positions or {})
    total_value = float(cash) + sum(
        float(position.amount) * float(position.avg_cost)
        for position in held.values()
    )
    return SimpleNamespace(
        cash=float(cash),
        positions=held,
        total_value=total_value,
    )


def test_0935_plans_sells_before_fixed_slot_buys_and_no_same_day_rebuy():
    held_code = "513100"
    new_code = "159915"
    sold_today_code = "518880"
    adapter = _PlannerAdapter(
        {
            held_code: _daily_death_cross_snapshot(
                weekly_close=9.0,
                weekly_ma20=10.0,
                weekly_ma20_prev=11.0,
            ),
            new_code: _eligible_snapshot(code=new_code),
            sold_today_code: _eligible_snapshot(code=sold_today_code),
        }
    )
    planner = candidate.TrendPullbackOrderPlanner(
        adapter,
        etf_pool=[held_code, new_code, sold_today_code],
        trade_dates=["2021-03-01", "2021-03-08"],
    )
    planner.position_states[held_code] = candidate.PositionSignalState(
        "2021-03-01",
        10.0,
        0.2,
        11.0,
    )
    planner.sold_today = {sold_today_code}
    planner.sold_today_date = "2021-03-08"
    broker = _broker(19000.0, {held_code: _position()})

    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker,
        decision_time="09:35",
        current_prices={held_code: 10.80, new_code: 10.0, sold_today_code: 10.0},
    )

    assert plans[0] == {
        "code": held_code,
        "target_value": 0.0,
        "reason": "weekly_trend_break",
    }
    assert plans[1]["code"] == new_code
    assert plans[1]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
    assert plans[1]["reason"] == "weekly_pullback_entry"
    assert plans[1]["entry_atr"] == pytest.approx(0.2)
    assert sold_today_code not in [item["code"] for item in plans[1:]]


def test_planner_does_not_force_replacement_when_three_holdings_are_healthy():
    held_codes = ["A", "B", "C"]
    adapter = _PlannerAdapter(
        {
            **{code: _eligible_snapshot(code=code) for code in held_codes},
            "NEW": _eligible_snapshot(code="NEW", weekly_close=12.0),
        }
    )
    planner = candidate.TrendPullbackOrderPlanner(
        adapter,
        etf_pool=held_codes + ["NEW"],
        trade_dates=["2021-03-01", "2021-03-08"],
    )
    for code in held_codes:
        planner.position_states[code] = candidate.PositionSignalState(
            "2021-03-01",
            10.0,
            0.2,
            10.0,
        )
    broker = _broker(1000.0, {code: _position() for code in held_codes})

    plans = planner.plan_orders_at(
        "2021-03-08",
        "2021-03-05",
        broker,
        "09:35",
        current_prices={code: 10.0 for code in held_codes + ["NEW"]},
    )

    assert plans == []


def test_planner_buy_order_uses_deterministic_frozen_ranking():
    adapter = _PlannerAdapter(
        {
            "159915": _eligible_snapshot(code="159915", weekly_close=11.0),
            "513100": _eligible_snapshot(
                code="513100", weekly_close=12.0, k=23.0, d=20.0
            ),
            "518880": _eligible_snapshot(
                code="518880", weekly_close=12.0, k=21.0, d=20.0
            ),
        }
    )
    planner = candidate.TrendPullbackOrderPlanner(
        adapter,
        etf_pool=["159915", "513100", "518880"],
    )

    plans = planner.plan_orders_at(
        "2021-03-08",
        "2021-03-05",
        _broker(20000.0),
        "09:35",
        current_prices={"159915": 10.0, "513100": 10.0, "518880": 10.0},
    )

    assert [item["code"] for item in plans] == ["513100", "518880", "159915"]


@pytest.mark.parametrize("atr", [None, 0.0, math.nan, math.inf])
def test_planner_skips_entry_without_positive_finite_atr(atr):
    snapshot = _eligible_snapshot(code="159915")
    if atr is None:
        snapshot.pop("atr")
    else:
        snapshot["atr"] = atr
    planner = candidate.TrendPullbackOrderPlanner(
        _PlannerAdapter({"159915": snapshot}),
        etf_pool=["159915"],
    )

    plans = planner.plan_orders_at(
        "2021-03-08",
        "2021-03-05",
        _broker(20000.0),
        "09:35",
        current_prices={"159915": 10.0},
    )

    assert plans == []


def test_planner_requires_confirmed_positive_fill_before_creating_entry_state():
    adapter = _PlannerAdapter({"159915": _eligible_snapshot(code="159915")})
    planner = candidate.TrendPullbackOrderPlanner(adapter, etf_pool=["159915"])
    plans = planner.plan_orders_at(
        "2021-03-08",
        "2021-03-05",
        _broker(20000.0),
        "09:35",
        current_prices={"159915": 10.0},
    )
    rejected = SimpleNamespace(
        filled=False,
        code="159915",
        amount_delta=0,
        exec_price=10.0,
    )

    planner.on_orders_processed("2021-03-08", "09:35", plans, [rejected])
    assert "159915" not in planner.position_states

    fill = SimpleNamespace(
        filled=True,
        code="159915",
        amount_delta=600,
        exec_price=10.01,
    )
    planner.on_orders_processed("2021-03-08", "09:35", plans, [fill])

    state = planner.position_states["159915"]
    assert state.entry_date == "2021-03-08"
    assert state.entry_price == pytest.approx(10.01)
    assert state.entry_atr == pytest.approx(plans[0]["entry_atr"])
    assert state.highest_close == pytest.approx(10.01)


def test_0935_atr_survives_missing_signal_snapshot():
    code = "513100"
    planner = candidate.TrendPullbackOrderPlanner(
        _PlannerAdapter({code: None}),
        etf_pool=[code],
    )
    planner.position_states[code] = candidate.PositionSignalState(
        "2021-03-01",
        10.0,
        0.2,
        11.0,
    )
    broker = _broker(10000.0, {code: _position()})

    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker,
        decision_time="09:35",
        current_prices={code: 10.44},
    )

    assert plans == [{"code": code, "target_value": 0.0, "reason": "atr_stop"}]


def test_1450_calls_no_signal_adapter_and_only_returns_atr_sells():
    stopped_code = "513100"
    adapter = SimpleNamespace(
        score=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("14:50 loaded signals")
        )
    )
    planner = candidate.TrendPullbackOrderPlanner(adapter, etf_pool=[stopped_code])
    planner.position_states[stopped_code] = candidate.PositionSignalState(
        "2021-03-01",
        10.0,
        0.2,
        11.0,
    )
    broker = _broker(10000.0, {stopped_code: _position()})

    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker,
        decision_time="14:50",
        current_prices={stopped_code: 10.44},
    )

    assert plans == [
        {"code": stopped_code, "target_value": 0.0, "reason": "atr_stop"}
    ]


def test_order_results_remove_state_only_after_confirmed_sell_and_block_rebuy():
    code = "513100"
    planner = candidate.TrendPullbackOrderPlanner(
        _PlannerAdapter({code: _eligible_snapshot(code=code)}),
        etf_pool=[code],
    )
    planner.position_states[code] = candidate.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 11.0
    )
    plans = [{"code": code, "target_value": 0.0, "reason": "atr_stop"}]
    unfilled = SimpleNamespace(
        filled=False,
        code=code,
        amount_delta=0,
        exec_price=10.44,
    )
    planner.on_orders_processed("2021-03-08", "09:35", plans, [unfilled])
    assert code in planner.position_states

    fill = SimpleNamespace(
        filled=True,
        code=code,
        amount_delta=-100,
        exec_price=10.44,
    )
    planner.on_orders_processed("2021-03-08", "09:35", plans, [fill])

    assert code not in planner.position_states
    assert planner.sold_today == {code}
    assert planner.sold_today_date == "2021-03-08"


def test_after_close_updates_only_existing_position_highest_close():
    planner = candidate.TrendPullbackOrderPlanner(
        _PlannerAdapter({}),
        etf_pool=["513100"],
    )
    planner.position_states["513100"] = candidate.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 10.5
    )

    planner.on_after_close(
        "2021-03-08",
        {"513100": 10.8, "159915": 99.0},
    )

    assert planner.position_states["513100"].highest_close == pytest.approx(10.8)
    assert "159915" not in planner.position_states


def test_planner_rejects_any_decision_time_other_than_0935_or_1450():
    planner = candidate.TrendPullbackOrderPlanner(_PlannerAdapter({}), etf_pool=[])

    with pytest.raises(ValueError, match="09:35 and 14:50"):
        planner.plan_orders_at(
            "2021-03-08",
            "2021-03-05",
            _broker(20000.0),
            "10:00",
            current_prices={},
        )
