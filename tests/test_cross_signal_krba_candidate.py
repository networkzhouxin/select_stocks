# -*- coding: utf-8 -*-
"""Behavior tests for the frozen KRBA independent research candidate."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _snapshot(**overrides):
    value = {
        "k_prev": 19.0,
        "d_prev": 20.0,
        "k": 22.0,
        "d": 21.0,
        "rsi6": 29.0,
        "low": 9.7,
        "close": 9.9,
        "boll_lower": 9.8,
        "boll_mid": 10.5,
        "boll_upper": 11.2,
        "atr": 0.2,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "broken",
    [
        {"k_prev": 21.0},
        {"k": 20.0},
        {"rsi6": 30.01},
        {"low": 9.81},
        {"close": 9.8},
    ],
)
def test_entry_requires_same_day_kd_cross_rsi_location_and_lower_band_reclaim(broken):
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        is_entry_eligible,
    )

    assert is_entry_eligible(_snapshot()) is True
    assert is_entry_eligible(_snapshot(**broken)) is False


def test_low_zone_kdj_death_cross_does_not_exit_before_mean_reached():
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        PositionSignalState,
        choose_exit_reason,
    )

    state = PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=10.1,
    )
    bearish = _snapshot(
        k_prev=25.0,
        d_prev=24.0,
        k=20.0,
        d=22.0,
        close=9.95,
    )

    assert choose_exit_reason(
        state,
        bearish,
        current_price=9.95,
        hold_days=8,
    ) is None


def test_atr_exit_is_unconditional_and_has_priority_over_normal_exit():
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        PositionSignalState,
        choose_exit_reason,
    )

    state = PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=11.0,
        mean_reached=True,
        upper_reached=True,
    )

    assert choose_exit_reason(
        state,
        _snapshot(close=11.2, boll_upper=11.0),
        current_price=10.44,
        hold_days=1,
    ) == "atr_stop"


class _IntradayLoader:
    def get_minute_bar(self, code, current_date, trade_time):
        assert code == "AAA"
        assert current_date == "2020-01-06"
        if trade_time == "09:35":
            return {
                "open": 10.0,
                "close": 10.0,
                "volume": 1000.0,
                "num_trades": 10.0,
            }
        assert trade_time == "14:50"
        return {
            "open": 9.49,
            "high": 99.0,
            "low": 1.0,
            "close": 99.0,
            "volume": 1000.0,
            "num_trades": 10.0,
        }

    def load_daily_frame(self, code, current_date):
        return pd.DataFrame([{"date": current_date, "close": 9.8}])


class _NoAfternoonScorePlanner:
    params = {"max_hold": 3}

    def __init__(self):
        self.times = []
        self.processed = []

    def plan_orders_at(
        self,
        current_date,
        previous_date,
        broker,
        decision_time,
        current_prices=None,
    ):
        self.times.append(decision_time)
        if decision_time == "09:35":
            return []
        assert current_prices == {"AAA": pytest.approx(9.49)}
        return [{"code": "AAA", "target_value": 0.0, "reason": "atr_stop"}]

    def on_orders_processed(self, current_date, decision_time, plans, results):
        self.processed.append((decision_time, tuple(result.reason for result in results)))

    def on_after_close(self, current_date, marks):
        pass


def test_1450_execution_uses_arrival_open_and_never_minute_close():
    from cross_signal_strategy.local.krba_backtester import KRBABacktestEngine
    from cross_signal_strategy.local.local_backtester import Position

    planner = _NoAfternoonScorePlanner()
    engine = KRBABacktestEngine(_IntradayLoader(), initial_cash=1000.0)
    engine.broker.positions["AAA"] = Position("AAA", 100, 10.0)

    results = engine.run(["2020-01-06"], planner)

    afternoon = [order for order in results[0].orders if order.side_time.endswith("14:50")]
    assert len(afternoon) == 1
    assert afternoon[0].filled is True
    assert afternoon[0].exec_price == pytest.approx(9.481)
    assert planner.times == ["09:35", "14:50"]


def test_signal_adapter_uses_only_completed_t1_daily_bar():
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        KRBASignalAdapter,
    )

    dates = pd.date_range("2019-01-01", periods=30, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": range(30),
            "high": [float(i) + 1.0 for i in range(30)],
            "low": [max(float(i) - 1.0, 0.1) for i in range(30)],
            "close": [float(i) + 0.5 for i in range(30)],
            "volume": [1000.0] * 30,
        }
    )

    class Loader:
        def load_daily_frame(self, code, current_date):
            return frame.copy()

    snapshot = KRBASignalAdapter(Loader()).score("AAA", "2019-01-31")

    assert snapshot["signal_date"] == "2019-01-30"
    assert snapshot["max_data_date"] == "2019-01-30"


class _FixedAdapter:
    def __init__(self, snapshot):
        self.snapshot = dict(snapshot)
        self.calls = []

    def score(self, code, current_date, return_reason=False):
        self.calls.append((code, current_date))
        value = dict(self.snapshot, code=code, signal_date="2020-01-03")
        return (value, None) if return_reason else value


def test_planner_buys_fixed_slot_and_freezes_fill_atr_state():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        KRBAOrderPlanner,
    )

    adapter = _FixedAdapter(_snapshot())
    planner = KRBAOrderPlanner(
        adapter,
        etf_pool=["AAA"],
        trade_dates=["2020-01-03", "2020-01-06"],
    )
    broker = LocalBroker(20000.0)

    plans = planner.plan_orders_at(
        "2020-01-06", None, broker, "09:35", current_prices={}
    )
    assert plans == [
        {
            "code": "AAA",
            "target_value": pytest.approx(20000.0 * 0.95 / 3),
            "reason": "krba_entry",
        }
    ]

    fill = broker.order_target_value(
        "AAA", plans[0]["target_value"], 10.0, "2020-01-06 09:35"
    )
    fill.reason = "krba_entry"
    planner.on_orders_processed("2020-01-06", "09:35", plans, [fill])

    state = planner.position_states["AAA"]
    assert state.entry_date == "2020-01-06"
    assert state.entry_atr == pytest.approx(0.2)
    assert state.highest_close == pytest.approx(fill.exec_price)


def test_1450_planner_calls_no_signals_and_submits_only_atr_stop():
    from cross_signal_strategy.local.local_backtester import LocalBroker, Position
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        KRBAOrderPlanner,
        PositionSignalState,
    )

    adapter = _FixedAdapter(_snapshot())
    planner = KRBAOrderPlanner(adapter, etf_pool=["AAA"])
    planner.position_states["AAA"] = PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=11.0,
    )
    broker = LocalBroker(1000.0)
    broker.positions["AAA"] = Position("AAA", 100, 10.0)

    plans = planner.plan_orders_at(
        "2020-01-06", "2020-01-03", broker, "14:50",
        current_prices={"AAA": 10.44},
    )

    assert plans == [{"code": "AAA", "target_value": 0.0, "reason": "atr_stop"}]
    assert adapter.calls == []


def test_planner_arms_mean_from_t1_and_exits_after_five_sessions():
    from cross_signal_strategy.local.local_backtester import LocalBroker, Position
    from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
        KRBAOrderPlanner,
        PositionSignalState,
    )

    bearish_after_mean = _snapshot(
        k_prev=30.0,
        d_prev=29.0,
        k=25.0,
        d=27.0,
        close=10.4,
        boll_mid=10.5,
    )
    adapter = _FixedAdapter(bearish_after_mean)
    trade_dates = [
        "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
        "2020-01-08", "2020-01-09",
    ]
    planner = KRBAOrderPlanner(adapter, etf_pool=["AAA"], trade_dates=trade_dates)
    planner.position_states["AAA"] = PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=10.8,
        mean_reached=True,
    )
    broker = LocalBroker(1000.0)
    broker.positions["AAA"] = Position("AAA", 100, 10.0)

    plans = planner.plan_orders_at(
        "2020-01-09", "2020-01-08", broker, "09:35",
        current_prices={"AAA": 10.4},
    )

    assert plans[0] == {
        "code": "AAA",
        "target_value": 0.0,
        "reason": "mean_reached_weakness",
    }


def test_frozen_gate_requires_material_accuracy_and_return_retention():
    from types import SimpleNamespace

    from cross_signal_strategy.research.krba_training_replay import evaluate_gate

    baseline = SimpleNamespace(
        win_rate=0.50,
        total_return=1.00,
        profit_loss_ratio=4.0,
        max_drawdown=0.06,
        closed_trade_count=60,
    )
    passing = SimpleNamespace(
        win_rate=0.56,
        total_return=0.85,
        profit_loss_ratio=3.0,
        max_drawdown=0.065,
        closed_trade_count=60,
    )
    baseline_x2 = SimpleNamespace(total_return=0.80, max_drawdown=0.07)
    passing_x2 = SimpleNamespace(total_return=0.62, max_drawdown=0.075)

    decision = evaluate_gate(
        baseline,
        passing,
        baseline_x2,
        passing_x2,
        candidate_annual_returns={2019: 0.1, 2020: 0.2, 2021: 0.05},
        candidate_trades_by_year={2019: 20, 2020: 20, 2021: 20},
        max_single_profit_share=0.30,
    )
    assert decision.passed is True
    assert decision.reasons == ()

    too_small_accuracy_gain = SimpleNamespace(**vars(passing))
    too_small_accuracy_gain.win_rate = 0.54
    failed = evaluate_gate(
        baseline,
        too_small_accuracy_gain,
        baseline_x2,
        passing_x2,
        candidate_annual_returns={2019: 0.1, 2020: 0.2, 2021: 0.05},
        candidate_trades_by_year={2019: 20, 2020: 20, 2021: 20},
        max_single_profit_share=0.30,
    )
    assert failed.passed is False
    assert "win rate gain is below 5 percentage points" in failed.reasons


def test_doubled_friction_applies_ten_yuan_minimum_commission_to_real_order():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.krba_training_replay import DOUBLE_FRICTION

    broker = LocalBroker(20000.0, **DOUBLE_FRICTION)
    order = broker.order_target_value(
        "AAA", target_value=1000.0, price=10.0, side_time="2020-01-06 09:35"
    )

    assert order.filled is True
    assert order.commission == pytest.approx(10.0)
