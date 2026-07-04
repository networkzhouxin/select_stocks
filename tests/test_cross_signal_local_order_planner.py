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


def candidate(code, buy_score=65, sell_score=0, reversal_score=40):
    return {
        "code": code,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reversal_score": reversal_score,
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
    }


def test_planner_buys_top_candidates_up_to_empty_slots():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner

    adapter = FakeSignalAdapter({
        "510300": candidate("510300", buy_score=70, reversal_score=30),
        "159915": candidate("159915", buy_score=75, reversal_score=35),
        "512100": candidate("512100", buy_score=55, reversal_score=45),
        "513100": candidate("513100", buy_score=72, sell_score=35, reversal_score=50),
    })
    planner = LocalCrossSignalOrderPlanner(adapter, etf_pool=["510300", "159915", "512100", "513100"])
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders("2019-07-01", "2019-06-28", broker)

    assert orders == [
        {"code": "159915", "target_value": pytest.approx(5000.0)},
        {"code": "510300", "target_value": pytest.approx(5000.0)},
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

    assert orders[0] == {"code": "510300", "target_value": 0.0}
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
