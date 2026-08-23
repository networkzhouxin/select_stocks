# -*- coding: utf-8 -*-
"""Tests for the isolated dimension-capped candidate order planner."""

from copy import deepcopy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_signal_strategy.local.local_backtester import LocalBroker, Position


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        value = deepcopy(self.scores.get(code))
        reason = None if value is not None else "no_data"
        return (value, reason) if return_reason else value


def _training_module():
    from cross_signal_strategy.research import dimension_capped_training_ab
    return dimension_capped_training_ab


def _candidate_score(code, **overrides):
    values = {
        "code": code,
        "buy_allowed": True,
        "buy_score": 40.0,
        "reversal_score": 18.0,
        "location_score": 10.0,
        "trend_score": 12.0,
        "volume_rank_score": 0.0,
        "sell_score": 0.0,
        "sell_weakness_score": 0.0,
        "sell_damage_score": 0.0,
        "close_far_above_ma20": False,
        "downside_continuation": False,
        "weak_repair_blocked": False,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
        "ma20_slope_non_negative": True,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _six_trade_dates():
    return [
        "2019-07-01", "2019-07-02", "2019-07-03",
        "2019-07-04", "2019-07-05", "2019-07-08",
    ]


def _held_severe_sell_fixture(buy_date, params=None):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score(
                "510300", buy_score=10.0,
                sell_score=24.0, sell_weakness_score=6.0, sell_damage_score=18.0,
            )
        }),
        etf_pool=["510300"],
        params=params,
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)
    return planner, broker


def _held_atr_stop_fixture(buy_date):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score("510300", buy_score=44.0),
            "159915": _candidate_score("159915", buy_score=42.0),
        }),
        etf_pool=["510300", "159915"],
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    planner.highest_since_buy["510300"] = 10.0
    planner.entry_atr["510300"] = 1.0
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 9.0)
    return planner, broker


def test_candidate_planner_sells_first_then_buys_ranked_empty_slots():
    module = _training_module()
    planner = module.DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score(
                "510300", buy_score=10, sell_score=24,
                sell_weakness_score=10, sell_damage_score=14,
            ),
            "513100": _candidate_score("513100", buy_score=44, location_score=10),
            "159915": _candidate_score("159915", buy_score=42, location_score=8, volume_rank_score=6),
        }),
        etf_pool=["510300", "513100", "159915"],
        buy_dates={"510300": "2019-06-20"},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders("2019-07-08", "2019-07-05", broker)
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "dimension_capped_signal_sell"}
    assert [item["code"] for item in orders[1:]] == ["513100", "159915"]


def test_candidate_target_is_equal_weight_and_volume_only_breaks_rank_ties():
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "159915": _candidate_score("159915", volume_rank_score=10.0),
            "513100": _candidate_score("513100", volume_rank_score=0.0),
        }),
        etf_pool=["159915", "513100"],
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=20000.0)
    orders = planner.plan_orders("2019-07-01", None, broker)
    assert [item["code"] for item in orders] == ["159915", "513100"]
    assert orders[0]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
    assert orders[1]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)


def test_candidate_signal_sell_waits_five_trading_days():
    planner, broker = _held_severe_sell_fixture(buy_date="2019-07-01")
    assert planner.plan_orders("2019-07-05", "2019-07-04", broker) == []
    assert planner.plan_orders("2019-07-08", "2019-07-05", broker)[0]["reason"] == "dimension_capped_signal_sell"


def test_candidate_freezes_five_day_signal_hold_when_params_request_one_day():
    module = _training_module()
    params = module.strategy.get_default_params()
    params["min_signal_hold_days"] = 1
    planner, broker = _held_severe_sell_fixture("2019-07-01", params=params)

    assert planner.plan_orders("2019-07-02", "2019-07-01", broker) == []


def test_candidate_atr_stop_ignores_five_day_signal_hold_and_blocks_same_day_rebuy():
    planner, broker = _held_atr_stop_fixture(buy_date="2019-07-01")
    orders = planner.plan_orders(
        "2019-07-02", "2019-07-01", broker,
        current_prices={"510300": 8.0, "159915": 4.0},
    )
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "atr_stop"}
    assert "510300" not in [item["code"] for item in orders[1:]]
