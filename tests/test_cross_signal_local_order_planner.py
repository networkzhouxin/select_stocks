# -*- coding: utf-8 -*-
"""Tests for converting local cross-signal scores into order plans."""

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, code, current_date, return_reason=False):
        self.calls.append((code, current_date, return_reason))
        item = self.scores.get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        return (dict(item), None) if return_reason else dict(item)


def candidate(code, buy_score=65, sell_score=0, reversal_score=40, volume_score=0):
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
        "atr": 0.1,
    }


def test_planner_buys_top_candidates_up_to_empty_slots():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({
        "510300": candidate("510300", buy_score=70, reversal_score=30, volume_score=6),
        "159915": candidate("159915", buy_score=75, reversal_score=35, volume_score=6),
        "512100": candidate("512100", buy_score=55, reversal_score=45),
        "513100": candidate("513100", buy_score=72, sell_score=35, reversal_score=50),
    })
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300", "159915", "512100", "513100"])
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders("2019-07-01", "2019-06-28", broker)

    assert orders == [
        {"code": "159915", "target_value": pytest.approx(6333.333333), "reason": "buy_signal"},
        {"code": "510300", "target_value": pytest.approx(6333.333333), "reason": "buy_signal"},
    ]


def test_planner_sells_existing_position_before_buying_new_slots():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    held_sell = candidate("510300", buy_score=20, sell_score=40)
    held_sell.update({"close_below_ma20": True})
    adapter = FakeSignalAdapter({
        "510300": held_sell,
        "159915": candidate("159915", buy_score=75),
        "512100": candidate("512100", buy_score=70),
        "159928": candidate("159928", buy_score=65),
    })
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300", "159915", "512100", "159928"])
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders("2019-07-01", "2019-06-28", broker)

    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "signal_sell"}
    assert [o["code"] for o in orders[1:]] == ["159915", "512100", "159928"]


def test_planner_does_not_sell_position_bought_today_by_signal():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    held_sell = candidate("510300", buy_score=20, sell_score=40)
    held_sell.update({"close_below_ma20": True})
    adapter = FakeSignalAdapter({"510300": held_sell})
    planner = LocalCrossSignalOrderPlanner(
        adapter,
        etf_pool=["510300"],
        buy_dates={"510300": "2019-07-01"},
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders("2019-07-01", "2019-06-28", broker)

    assert orders == []


def test_planner_blocks_signal_sell_before_minimum_trading_day_hold():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    held_sell = candidate("510300", buy_score=20, sell_score=40)
    held_sell.update({"close_below_ma20": True})
    adapter = FakeSignalAdapter({"510300": held_sell})
    planner = LocalCrossSignalOrderPlanner(
        adapter,
        etf_pool=["510300"],
        buy_dates={"510300": "2019-07-01"},
        trade_dates=[
            "2019-07-01",
            "2019-07-02",
            "2019-07-03",
            "2019-07-04",
            "2019-07-05",
            "2019-07-08",
        ],
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    blocked = planner.plan_orders("2019-07-05", "2019-07-04", broker)
    allowed = planner.plan_orders("2019-07-08", "2019-07-05", broker)

    assert blocked == []
    assert allowed[0] == {"code": "510300", "target_value": 0.0, "reason": "signal_sell"}


def test_planner_atr_stop_ignores_minimum_signal_hold():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({"510300": candidate("510300", buy_score=80)})
    params = {
        "max_hold": 3,
        "base_ratio": 0.95,
        "buy_threshold": 60,
        "sell_threshold": 30,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "adx_trend_threshold": 25,
        "min_signal_hold_days": 5,
    }
    planner = LocalCrossSignalOrderPlanner(
        adapter,
        etf_pool=["510300"],
        params=params,
        buy_dates={"510300": "2019-07-01"},
        trade_dates=["2019-07-01", "2019-07-02"],
    )
    planner.highest_since_buy["510300"] = 10.0
    planner.entry_atr["510300"] = 1.0
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 9.0)

    orders = planner.plan_orders(
        "2019-07-02",
        "2019-07-01",
        broker,
        current_prices={"510300": 8.0},
    )

    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "atr_stop"}


def test_engine_runs_real_signal_planner_smoke_window_without_future_dates():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    train_root = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")
    loader = CrossSignalTrainingDataLoader(train_root)
    adapter = LocalSignalAdapter(loader)
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300", "159915", "512100"])
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)

    results = engine.run(
        ["2019-07-01", "2019-07-02", "2019-07-03"],
        planner.plan_orders,
    )

    assert [day.date for day in results] == ["2019-07-01", "2019-07-02", "2019-07-03"]
    assert all(day.total_value > 0 for day in results)
    assert all(day.previous_date is None or day.previous_date < day.date for day in results)


def test_planner_records_entry_atr_and_highest_after_filled_buy():
    from cross_signal_strategy.local_backtester import OrderResult
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({"510300": candidate("510300", buy_score=70)})
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300"])
    order = OrderResult(
        code="510300",
        amount_delta=1000,
        exec_price=3.0,
        commission=5.0,
        side_time="2019-07-01 09:35",
        filled=True,
    )

    planner.on_orders_filled("2019-07-01", [order])

    assert planner.entry_atr["510300"] == pytest.approx(0.1)
    assert planner.highest_since_buy["510300"] == pytest.approx(3.0)
    assert planner.buy_dates["510300"] == "2019-07-01"


def test_planner_atr_stop_sells_before_signal_logic_and_blocks_same_day_rebuy():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({
        "510300": candidate("510300", buy_score=80),
        "159915": candidate("159915", buy_score=70),
    })
    params = {
        "max_hold": 3,
        "base_ratio": 0.75,
        "buy_threshold": 60,
        "sell_threshold": 30,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "adx_trend_threshold": 25,
    }
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300", "159915"], params=params)
    planner.highest_since_buy["510300"] = 10.0
    planner.entry_atr["510300"] = 1.0
    broker = LocalBroker(initial_cash=15000.0)
    broker.positions["510300"] = Position("510300", 1000, 9.0)

    orders = planner.plan_orders(
        "2019-07-02",
        "2019-07-01",
        broker,
        current_prices={"510300": 8.0, "159915": 4.0},
    )

    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "atr_stop"}
    assert [o["code"] for o in orders[1:]] == ["159915"]
    assert "510300" not in [o["code"] for o in orders[1:]]


def test_planner_atr_stop_uses_etf_tick_precision_for_trigger():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({
        "518880": candidate("518880", buy_score=31, sell_score=24),
    })
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["518880"])
    planner.highest_since_buy["518880"] = 3.725
    planner.entry_atr["518880"] = 0.046
    broker = LocalBroker(initial_cash=14000.0)
    broker.positions["518880"] = Position("518880", 1700, 3.52352)

    orders = planner.plan_orders(
        "2020-03-02",
        "2020-02-28",
        broker,
        current_prices={"518880": 3.539},
    )

    assert orders[0] == {"code": "518880", "target_value": 0.0, "reason": "atr_stop"}


def test_planner_uses_0935_position_marks_for_new_buy_target_value():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({
        "159915": candidate("159915", buy_score=70, volume_score=6),
    })
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["159915"])
    broker = LocalBroker(initial_cash=17000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders(
        "2019-07-02",
        "2019-07-01",
        broker,
        current_prices={"510300": 4.0},
    )

    assert orders == [
        {"code": "159915", "target_value": pytest.approx(6650.0), "reason": "buy_signal"},
    ]


def test_planner_can_scale_zero_volume_score_buy_target_without_blocking_trade():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    no_volume = candidate("159915", buy_score=70)
    no_volume["volume_score"] = 0
    with_volume = candidate("510300", buy_score=68)
    with_volume["volume_score"] = 6
    adapter = FakeSignalAdapter({
        "159915": no_volume,
        "510300": with_volume,
    })
    params = {
        "max_hold": 3,
        "base_ratio": 0.90,
        "buy_threshold": 60,
        "sell_threshold": 30,
        "trailing_atr_mult": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
        "adx_trend_threshold": 25,
        "a_share_zero_volume_buy_scale": 0.50,
    }
    planner = LocalCrossSignalOrderPlanner(
        adapter,
        etf_pool=["159915", "510300"],
        params=params,
    )
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders("2019-07-02", "2019-07-01", broker)

    assert orders == [
        {"code": "159915", "target_value": pytest.approx(3000.0), "reason": "buy_signal"},
        {"code": "510300", "target_value": pytest.approx(6000.0), "reason": "buy_signal"},
    ]
