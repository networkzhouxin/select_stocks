# -*- coding: utf-8 -*-
"""Tests for the minimal local cross-signal event backtester."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def test_broker_buys_target_value_at_0935_with_lot_fee_and_slippage():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    broker = LocalBroker(initial_cash=20000.0)

    order = broker.order_target_value(
        code="510300",
        target_value=5000.0,
        price=loader.get_minute_bar("510300", "2019-01-02", "09:35")["close"],
        side_time="2019-01-02 09:35",
    )

    assert order.filled
    assert order.amount_delta == 1600
    assert order.exec_price == pytest.approx(3.063)
    assert order.commission == pytest.approx(5.0)
    assert broker.cash == pytest.approx(15094.2)
    assert broker.positions["510300"].amount == 1600
    assert broker.positions["510300"].avg_cost == pytest.approx(3.063)
    assert broker.total_value({"510300": 3.060}) == pytest.approx(19990.2)


def test_broker_sells_to_zero_with_sell_slippage_and_min_commission():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    price = loader.get_minute_bar("510300", "2019-01-02", "09:35")["close"]
    broker = LocalBroker(initial_cash=20000.0)
    broker.order_target_value("510300", 5000.0, price, "2019-01-02 09:35")

    order = broker.order_target_value("510300", 0.0, price, "2019-01-02 09:35")

    assert order.filled
    assert order.amount_delta == -1600
    assert order.exec_price == pytest.approx(3.057)
    assert order.commission == pytest.approx(5.0)
    assert "510300" not in broker.positions
    assert broker.cash == pytest.approx(19980.4)


def test_broker_rejects_buy_when_cash_cannot_cover_one_lot_plus_commission():
    from cross_signal_strategy.local_backtester import LocalBroker

    broker = LocalBroker(initial_cash=100.0)
    order = broker.order_target_value("510300", 5000.0, price=3.060, side_time="2019-01-02 09:35")

    assert not order.filled
    assert order.amount_delta == 0
    assert broker.cash == pytest.approx(100.0)
    assert broker.positions == {}


def test_engine_runs_one_day_order_plan_with_0935_execution_and_close_mark():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)

    def order_plan(current_date, previous_date, broker):
        assert current_date == "2019-01-02"
        assert previous_date is None
        assert broker.cash == pytest.approx(20000.0)
        return [{"code": "510300", "target_value": 5000.0}]

    results = engine.run(["2019-01-02"], order_plan)

    assert len(results) == 1
    day = results[0]
    assert day.date == "2019-01-02"
    assert day.orders[0].amount_delta == 1600
    assert day.orders[0].exec_price == pytest.approx(3.063)
    assert day.cash == pytest.approx(15094.2)
    assert day.positions["510300"].amount == 1600
    assert day.marks["510300"] == pytest.approx(3.017)
    assert day.total_value == pytest.approx(19921.4)


def test_engine_preserves_filled_order_plan_reason_for_diagnostics():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)

    def order_plan(current_date, previous_date, broker):
        return [{"code": "510300", "target_value": 5000.0, "reason": "buy_signal"}]

    results = engine.run(["2019-01-02"], order_plan)

    assert results[0].orders[0].filled
    assert results[0].orders[0].reason == "buy_signal"


def test_engine_passes_previous_training_trade_date_to_order_plan():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)
    seen = []

    def order_plan(current_date, previous_date, broker):
        seen.append((current_date, previous_date))
        return []

    engine.run(["2019-01-02", "2019-01-03"], order_plan)

    assert seen == [("2019-01-02", None), ("2019-01-03", "2019-01-02")]


def test_engine_does_not_fill_at_stale_0935_bar_with_no_trades():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    bar = loader.get_minute_bar("159915", "2021-02-09", "09:35")
    assert float(bar["volume"]) == 0.0
    assert float(bar["num_trades"]) == 0.0
    engine = LocalBacktestEngine(loader=loader, initial_cash=20000.0)

    def order_plan(current_date, previous_date, broker):
        return [{"code": "159915", "target_value": 5000.0, "reason": "buy_signal"}]

    results = engine.run(["2021-02-09"], order_plan)

    order = results[0].orders[0]
    assert order.filled is False
    assert order.amount_delta == 0
    assert order.reason == "no executable trade at 09:35"
    assert results[0].cash == pytest.approx(20000.0)
    assert results[0].positions == {}


def test_engine_records_missing_0935_bar_as_unfilled_instead_of_crashing():
    from cross_signal_strategy.local_backtester import LocalBacktestEngine

    class MissingBarLoader:
        def get_minute_bar(self, code, current_date, trade_time):
            raise KeyError("missing minute")

    engine = LocalBacktestEngine(loader=MissingBarLoader(), initial_cash=20000.0)

    def order_plan(current_date, previous_date, broker):
        return [{"code": "MISSING", "target_value": 5000.0, "reason": "buy_signal"}]

    results = engine.run(["2019-01-02"], order_plan)

    order = results[0].orders[0]
    assert order.filled is False
    assert order.amount_delta == 0
    assert order.reason == "missing execution bar at 09:35"
    assert results[0].cash == pytest.approx(20000.0)
    assert results[0].positions == {}


@pytest.mark.parametrize("buy_reason", ["buy_signal", "backup_buy_signal"])
def test_engine_does_not_exceed_max_hold_when_planned_sell_is_unfilled(buy_reason):
    from cross_signal_strategy.local_backtester import (
        LocalBacktestEngine,
        Position,
    )
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    class SellThenBuyPlan:
        params = {"max_hold": 3}

        def plan_orders(self, current_date, previous_date, broker, current_prices=None):
            return [
                {"code": "159915", "target_value": 0.0, "reason": "signal_sell"},
                {"code": "510300", "target_value": 5000.0, "reason": buy_reason},
            ]

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    engine = LocalBacktestEngine(loader=loader, initial_cash=5000.0)
    engine.broker.positions = {
        "159915": Position("159915", 1000, 3.0),
        "513100": Position("513100", 1000, 4.0),
        "513500": Position("513500", 1000, 3.0),
    }

    results = engine.run(["2021-02-09"], SellThenBuyPlan().plan_orders)

    sell, buy = results[0].orders
    assert sell.filled is False
    assert sell.reason == "no executable trade at 09:35"
    assert buy.filled is False
    assert buy.reason == "no available holding slot after execution"
    assert sorted(results[0].positions) == ["159915", "513100", "513500"]
    assert len(results[0].positions) == 3


def test_engine_uses_slot_released_by_a_filled_sell_for_later_buy():
    from cross_signal_strategy.local_backtester import (
        LocalBacktestEngine,
        Position,
    )
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    class SellThenBuyPlan:
        params = {"max_hold": 3}

        def plan_orders(self, current_date, previous_date, broker, current_prices=None):
            return [
                {"code": "159915", "target_value": 0.0, "reason": "signal_sell"},
                {"code": "510300", "target_value": 5000.0, "reason": "buy_signal"},
            ]

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    engine = LocalBacktestEngine(loader=loader, initial_cash=5000.0)
    engine.broker.positions = {
        "159915": Position("159915", 1000, 3.0),
        "513100": Position("513100", 1000, 4.0),
        "513500": Position("513500", 1000, 3.0),
    }

    results = engine.run(["2021-02-10"], SellThenBuyPlan().plan_orders)

    sell, buy = results[0].orders
    assert sell.filled is True
    assert buy.filled is True
    assert "159915" not in results[0].positions
    assert "510300" in results[0].positions
    assert len(results[0].positions) == 3
