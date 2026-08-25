from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from cross_signal_strategy.local.local_backtester import LocalBroker
from cross_signal_strategy.research.rsi_low_turn_outcomes import (
    Friction,
    FutureSnapshot,
    calculate_round_trip,
    mature_event_labels,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
ARRIVAL = datetime(2026, 8, 26, 9, 35, tzinfo=SHANGHAI)
ARRIVAL_PLUS_5 = ARRIVAL + timedelta(days=7)
ARRIVAL_PLUS_10 = ARRIVAL + timedelta(days=14)
SLOT_CAPITAL = 20000.0 * 0.95 / 3.0
NOMINAL = Friction(0.0003, 5.0, 0.001)
DOUBLED = Friction(0.0006, 10.0, 0.002)


class FakeFuturePriceSource:
    def __init__(self, snapshots):
        self.snapshots = dict(snapshots)

    def snapshot_for(self, event, horizon, as_of):
        return self.snapshots.get(
            horizon,
            FutureSnapshot(horizon, "pending_horizon_not_arrived", None, None, None, None),
        )


def valid_event():
    return {
        "event_id": "e1",
        "code": "513100",
        "arrival_date": date(2026, 8, 26),
        "entry_open": 2.035,
    }


def source_with_horizons(tmp_path, horizons):
    snapshots = {
        horizon: FutureSnapshot(
            horizon, "matured", 2.035 + horizon * 0.01, 0.05, -0.02,
            ARRIVAL + timedelta(days=horizon),
        )
        for horizon in horizons
    }
    return FakeFuturePriceSource(snapshots)


def source_missing_horizon_three(tmp_path):
    snapshots = {
        horizon: FutureSnapshot(
            horizon, "matured", 2.035 + horizon * 0.01, 0.05, -0.02,
            ARRIVAL + timedelta(days=horizon),
        )
        for horizon in (1, 5, 10)
    }
    snapshots[3] = FutureSnapshot(
        3, "pending_missing_executable_price", None, None, None, None,
    )
    return FakeFuturePriceSource(snapshots)


def test_round_trip_matches_local_broker_and_integer_lots():
    result = calculate_round_trip("513100", 2.000, 2.100, NOMINAL)
    broker = LocalBroker(
        SLOT_CAPITAL,
        commission_rate=0.0003,
        min_commission=5.0,
        slippage_rate=0.001,
    )
    buy = broker.order_target_value("513100", SLOT_CAPITAL, 2.000, "shadow_entry")
    sell = broker.order_target_value("513100", 0.0, 2.100, "shadow_exit")

    assert result.amount % 100 == 0
    assert result.amount == buy.amount_delta
    assert result.buy_exec_price == buy.exec_price
    assert result.sell_exec_price == sell.exec_price
    assert result.buy_commission == 5.0
    assert result.sell_commission == 5.0
    assert result.net_pnl == pytest.approx(broker.cash - SLOT_CAPITAL)
    assert result.net_return == pytest.approx(result.net_pnl / SLOT_CAPITAL)


def test_doubled_friction_uses_ten_yuan_minimum():
    result = calculate_round_trip("513100", 2.000, 2.100, DOUBLED)

    assert result.buy_commission == 10.0
    assert result.sell_commission == 10.0


def test_only_arrived_horizons_mature(tmp_path):
    labels = mature_event_labels(
        valid_event(), source_with_horizons(tmp_path, (1, 3, 5)), ARRIVAL_PLUS_5,
    )

    assert {label.horizon for label in labels if label.status == "matured"} == {1, 3, 5}


def test_missing_0935_price_is_not_substituted(tmp_path):
    labels = mature_event_labels(
        valid_event(), source_missing_horizon_three(tmp_path), ARRIVAL_PLUS_10,
    )
    item = next(label for label in labels if label.horizon == 3)

    assert item.status == "pending_missing_executable_price"
    assert item.exit_price is None
    assert item.nominal is None
    assert item.doubled is None
