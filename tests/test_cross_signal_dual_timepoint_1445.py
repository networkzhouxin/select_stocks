# -*- coding: utf-8 -*-
"""Tests for the fixed causal 09:35/14:45 cross-signal candidate."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def _t1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [9.8, 10.0],
            "high": [10.2, 10.3],
            "low": [9.7, 9.9],
            "close": [10.0, 10.1],
            "volume": [1000.0, 1100.0],
        }
    )


def _minutes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2020-01-06",
                "time": "09:30",
                "prev_close": 10.1,
                "open": 10.2,
                "high": 10.3,
                "low": 10.1,
                "close": 10.25,
                "volume": 100.0,
            },
            {
                "date": "2020-01-06",
                "time": "14:44",
                "prev_close": 10.1,
                "open": 10.4,
                "high": 10.6,
                "low": 10.35,
                "close": 10.5,
                "volume": 200.0,
            },
            {
                "date": "2020-01-06",
                "time": "14:45",
                "prev_close": 10.1,
                "open": 99.0,
                "high": 100.0,
                "low": 1.0,
                "close": 99.0,
                "volume": 999999.0,
            },
        ]
    )


def test_1445_frame_uses_only_completed_minutes_through_1444():
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    result = build_intraday_signal_frame(
        _t1_frame(), _minutes(), "2020-01-06", decision_time="14:45"
    )

    bar = result.frame.iloc[-1]
    assert bar["date"] == "2020-01-06"
    assert bar["open"] == pytest.approx(10.2)
    assert bar["high"] == pytest.approx(10.6)
    assert bar["low"] == pytest.approx(10.1)
    assert bar["close"] == pytest.approx(10.5)
    assert bar["volume"] == pytest.approx(300.0)
    assert list(result.frame.iloc[:-1]["date"]) == ["2020-01-02", "2020-01-03"]
    assert result.audit.decision_time == "14:45"
    assert result.audit.data_cutoff == "14:44"
    assert result.audit.last_minute == "14:44"
    assert result.audit.minute_count == 2
    assert result.audit.partial_volume is True


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "out_of_order",
        "cross_day",
        "daily_out_of_order",
        "t_day_daily",
        "bad_prev_close",
    ],
)
def test_1445_frame_fails_closed_on_ambiguous_or_misaligned_data(mutation):
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    daily, minutes = _t1_frame(), _minutes()
    if mutation == "duplicate":
        minutes = pd.concat([minutes, minutes.iloc[[0]]], ignore_index=True)
    elif mutation == "out_of_order":
        minutes = minutes.iloc[[1, 0, 2]].reset_index(drop=True)
    elif mutation == "cross_day":
        minutes.loc[0, "date"] = "2020-01-03"
    elif mutation == "daily_out_of_order":
        daily = daily.iloc[::-1].reset_index(drop=True)
    elif mutation == "t_day_daily":
        daily = pd.concat(
            [daily, daily.iloc[[-1]].assign(date="2020-01-06")],
            ignore_index=True,
        )
    else:
        minutes.loc[:, "prev_close"] = 10.9

    with pytest.raises(ValueError):
        build_intraday_signal_frame(daily, minutes, "2020-01-06", "14:45")


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("time", "missing required columns"),
        ("prev_close", "missing required columns"),
        ("volume", "missing required columns"),
    ],
)
def test_1445_frame_rejects_missing_required_columns(column, message):
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    with pytest.raises(ValueError, match=message):
        build_intraday_signal_frame(
            _t1_frame(), _minutes().drop(columns=[column]), "2020-01-06", "14:45"
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", 0.0),
        ("high", 10.0),
        ("low", 10.4),
        ("close", float("nan")),
        ("volume", -1.0),
    ],
)
def test_1445_frame_rejects_invalid_visible_ohlcv(column, value):
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    minutes = _minutes()
    minutes.loc[0, column] = value

    with pytest.raises(ValueError, match="Invalid point-in-time OHLCV"):
        build_intraday_signal_frame(_t1_frame(), minutes, "2020-01-06", "14:45")


def test_1445_frame_rejects_unregistered_time_and_empty_visible_window():
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    with pytest.raises(ValueError, match="pre-registered 14:45"):
        build_intraday_signal_frame(_t1_frame(), _minutes(), "2020-01-06", "14:30")

    late_only = _minutes().loc[_minutes()["time"] >= "14:45"].copy()
    with pytest.raises(ValueError, match="No completed minute"):
        build_intraday_signal_frame(_t1_frame(), late_only, "2020-01-06", "14:45")


def test_dual_adapter_delegates_0935_and_scores_real_partial_t_bar():
    from cross_signal_strategy.local.dual_timepoint_signal_adapter import (
        DualTimepointSignalAdapter,
    )
    from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_training_run import build_training_signal_adapter

    baseline = build_training_signal_adapter(
        CrossSignalTrainingDataLoader(TRAIN_ROOT)
    )
    adapter = DualTimepointSignalAdapter(baseline)

    morning = adapter.score_at("510300", "2019-07-01", "09:35")
    afternoon = adapter.score_at("510300", "2019-07-01", "14:45")

    assert morning == baseline.score("510300", "2019-07-01")
    assert afternoon["signal_date"] == "2019-07-01"
    assert afternoon["decision_time"] == "14:45"
    assert afternoon["data_cutoff"] == "14:44"
    assert afternoon["max_data_date"] == "2019-07-01"
    assert afternoon["partial_volume"] is True
    for field in (
        "rsi6",
        "rsi12",
        "rsi24",
        "k",
        "d",
        "j",
        "dif",
        "dea",
        "adx",
        "boll_mid",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "atr",
    ):
        assert pd.notna(afternoon[field]), field


def test_dual_adapter_rejects_any_unregistered_decision_time():
    from cross_signal_strategy.local.dual_timepoint_signal_adapter import (
        DualTimepointSignalAdapter,
    )
    from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_training_run import build_training_signal_adapter

    adapter = DualTimepointSignalAdapter(
        build_training_signal_adapter(CrossSignalTrainingDataLoader(TRAIN_ROOT))
    )

    with pytest.raises(ValueError, match="Only 09:35 and 14:45"):
        adapter.score_at("510300", "2019-07-01", "14:44")


class _DualEngineLoader:
    MORNING_CLOSE_WITH_SLIPPAGE = 10.01
    AFTERNOON_OPEN_WITH_SLIPPAGE = 20.02

    def get_minute_bar(self, code, current_date, trade_time):
        assert current_date == "2020-01-06"
        assert code in {"AAA", "BBB"}
        assert trade_time in {"09:35", "14:45"}
        return {
            "open": 20.0 if trade_time == "14:45" else 9.9,
            "close": 10.0 if trade_time == "09:35" else 20.1,
            "volume": 1000.0,
            "num_trades": 10.0,
        }

    def load_daily_frame(self, code, current_date):
        return pd.DataFrame([{"date": current_date, "close": 15.0}])


class _RecordingPlanner:
    params = {"max_hold": 3}

    def __init__(self):
        self.calls = []

    def plan_orders_at(
        self,
        current_date,
        previous_date,
        broker,
        decision_time,
        current_prices=None,
    ):
        self.calls.append(("plan", current_date, decision_time))
        code = "AAA" if decision_time == "09:35" else "BBB"
        return [{"code": code, "target_value": 5000.0, "reason": "buy_signal"}]

    def on_orders_processed(self, current_date, decision_time, plans, results):
        self.calls.append(("processed", current_date, decision_time))

    def on_after_close(self, current_date, marks):
        self.calls.append(("after_close", current_date))


def _candidate(
    code,
    buy_score=65,
    sell_score=0,
    reversal_score=40,
    volume_score=0,
    atr=0.1,
):
    return {
        "code": code,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reversal_score": reversal_score,
        "volume_score": volume_score,
        "buy_allowed": True,
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
        "adx": 10,
        "plus_di": 20,
        "minus_di": 10,
        "atr": atr,
    }


class _TimeVaryingAdapter:
    def __init__(self, scores_by_time):
        self.scores_by_time = scores_by_time
        self.calls = []

    def score_at(self, code, current_date, decision_time, return_reason=False):
        self.calls.append((code, current_date, decision_time, return_reason))
        item = self.scores_by_time.get(decision_time, {}).get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        value = dict(item)
        return (value, None) if return_reason else value


def _execute_plans(broker, plans, decision_time):
    results = []
    for plan in plans:
        result = broker.order_target_value(
            plan["code"],
            plan["target_value"],
            10.0,
            "2020-01-06 %s" % decision_time,
        )
        if result.filled:
            result.reason = plan["reason"]
        results.append(result)
    return results


def _buy_codes(plans):
    return {
        str(item["code"]).split(".")[0]
        for item in plans
        if float(item["target_value"]) > 0.0
    }


def _sell_codes(plans):
    return {
        str(item["code"]).split(".")[0]
        for item in plans
        if float(item["target_value"]) == 0.0
    }


def _filled_signature(days):
    return [
        (
            day.date,
            "BUY" if order.amount_delta > 0 else "SELL",
            str(order.code).split(".")[0],
            abs(order.amount_delta),
            order.reason,
        )
        for day in days
        for order in day.orders
        if order.filled and order.amount_delta != 0
    ]


def _all_order_signature(days):
    return [
        (
            day.date,
            order.code,
            order.amount_delta,
            order.exec_price,
            order.commission,
            order.side_time,
            order.filled,
            order.reason,
        )
        for day in days
        for order in day.orders
    ]


def test_local_engine_accepts_friction_injection_without_changing_defaults():
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine

    baseline = LocalBacktestEngine(_DualEngineLoader(), initial_cash=20000.0)
    stressed = LocalBacktestEngine(
        _DualEngineLoader(),
        initial_cash=20000.0,
        broker_kwargs={"commission_rate": 0.0006, "slippage_rate": 0.002},
    )

    assert baseline.broker.commission_rate == pytest.approx(0.0003)
    assert baseline.broker.slippage_rate == pytest.approx(0.001)
    assert stressed.broker.commission_rate == pytest.approx(0.0006)
    assert stressed.broker.slippage_rate == pytest.approx(0.002)


def test_dual_engine_runs_0935_then_1445_and_marks_close_once():
    from cross_signal_strategy.local.dual_timepoint_backtester import (
        DualTimepointBacktestEngine,
    )

    planner = _RecordingPlanner()
    engine = DualTimepointBacktestEngine(
        _DualEngineLoader(), initial_cash=20000.0
    )

    days = engine.run(["2020-01-06"], planner)

    assert planner.calls == [
        ("plan", "2020-01-06", "09:35"),
        ("processed", "2020-01-06", "09:35"),
        ("plan", "2020-01-06", "14:45"),
        ("processed", "2020-01-06", "14:45"),
        ("after_close", "2020-01-06"),
    ]
    assert [order.side_time[-5:] for order in days[0].orders] == ["09:35", "14:45"]
    assert days[0].orders[0].exec_price == pytest.approx(
        _DualEngineLoader.MORNING_CLOSE_WITH_SLIPPAGE
    )
    assert days[0].orders[1].exec_price == pytest.approx(
        _DualEngineLoader.AFTERNOON_OPEN_WITH_SLIPPAGE
    )
    assert days[0].marks == {"AAA": 15.0, "BBB": 15.0}


def test_afternoon_recomputes_candidates_but_keeps_same_day_hold_guard():
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )
    from cross_signal_strategy.local.local_backtester import LocalBroker

    morning_aaa = _candidate("AAA", buy_score=75, atr=0.11)
    morning_bbb = _candidate("BBB", buy_score=0)
    afternoon_aaa = _candidate("AAA", buy_score=0, sell_score=40)
    afternoon_aaa["close_below_ma20"] = True
    afternoon_bbb = _candidate("BBB", buy_score=75, atr=0.22)
    adapter = _TimeVaryingAdapter(
        {
            "09:35": {"AAA": morning_aaa, "BBB": morning_bbb},
            "14:45": {"AAA": afternoon_aaa, "BBB": afternoon_bbb},
        }
    )
    planner = DualTimepointOrderPlanner(
        adapter, etf_pool=["AAA", "BBB"], trade_dates=["2020-01-06"]
    )
    broker = LocalBroker(initial_cash=20000.0)

    morning = planner.plan_orders_at(
        "2020-01-06", None, broker, "09:35", current_prices={}
    )
    morning_results = _execute_plans(broker, morning, "09:35")
    planner.on_orders_processed(
        "2020-01-06", "09:35", morning, morning_results
    )
    afternoon = planner.plan_orders_at(
        "2020-01-06", None, broker, "14:45", current_prices={}
    )

    assert _buy_codes(morning) == {"AAA"}
    assert _buy_codes(afternoon) == {"BBB"}
    assert "AAA" not in _sell_codes(afternoon)
    assert planner.entry_atr["AAA"] == pytest.approx(0.11)
    assert {call[2] for call in adapter.calls} == {"09:35", "14:45"}


def test_morning_sell_and_failed_buy_are_not_retried_at_1445():
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )
    from cross_signal_strategy.local.local_backtester import LocalBroker

    adapter = _TimeVaryingAdapter(
        {
            "14:45": {
                "SOLD": _candidate("SOLD", buy_score=80),
                "FAILED": _candidate("FAILED", buy_score=79),
                "SAFE": _candidate("SAFE", buy_score=78),
            }
        }
    )
    planner = DualTimepointOrderPlanner(
        adapter, etf_pool=["SOLD", "FAILED", "SAFE"]
    )
    planner.execution_date = "2020-01-06"
    planner.sold_today.add("SOLD")
    planner.failed_buy_codes.add("FAILED")
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders_at(
        "2020-01-06", None, broker, "14:45", current_prices={}
    )

    assert _buy_codes(orders) == {"SAFE"}


def test_1445_filled_buy_freezes_partial_t_atr_until_after_close():
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )
    from cross_signal_strategy.local.local_backtester import LocalBroker

    adapter = _TimeVaryingAdapter(
        {"14:45": {"AAA": _candidate("AAA", buy_score=80, atr=0.22)}}
    )
    planner = DualTimepointOrderPlanner(adapter, etf_pool=["AAA"])
    broker = LocalBroker(initial_cash=20000.0)
    plans = planner.plan_orders_at(
        "2020-01-06", None, broker, "14:45", current_prices={}
    )
    results = _execute_plans(broker, plans, "14:45")

    planner.on_orders_processed("2020-01-06", "14:45", plans, results)

    assert planner.entry_atr["AAA"] == pytest.approx(0.22)
    assert planner.highest_since_buy["AAA"] == pytest.approx(results[0].exec_price)
    planner.on_after_close("2020-01-06", {"AAA": 11.0})
    assert planner.highest_since_buy["AAA"] == pytest.approx(11.0)


def test_full_training_morning_only_dual_engine_matches_official_local_path():
    from cross_signal_strategy.local.dual_timepoint_backtester import (
        DualTimepointBacktestEngine,
    )
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )
    from cross_signal_strategy.local.dual_timepoint_signal_adapter import (
        DualTimepointSignalAdapter,
    )
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    dates = get_training_trade_dates(loader)
    baseline_adapter = build_training_signal_adapter(loader)
    baseline_planner = LocalCrossSignalOrderPlanner(
        baseline_adapter, trade_dates=dates
    )
    baseline_days = LocalBacktestEngine(loader, 20000.0).run(
        dates, baseline_planner.plan_orders
    )

    candidate_adapter = DualTimepointSignalAdapter(
        build_training_signal_adapter(loader)
    )
    candidate_planner = DualTimepointOrderPlanner(
        candidate_adapter, trade_dates=dates
    )
    candidate_days = DualTimepointBacktestEngine(
        loader, 20000.0, decision_times=("09:35",)
    ).run(dates, candidate_planner)

    # Frozen cross-v0.3.3 training baseline: 92 buys plus 89 sells.
    assert len(_filled_signature(baseline_days)) == 181
    assert _filled_signature(candidate_days) == _filled_signature(baseline_days)
    assert _all_order_signature(candidate_days) == _all_order_signature(baseline_days)
    assert candidate_days[-1].total_value == pytest.approx(
        baseline_days[-1].total_value
    )
