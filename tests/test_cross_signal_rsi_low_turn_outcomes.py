from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from cross_signal_strategy.local.local_backtester import LocalBroker
from cross_signal_strategy.research.rsi_low_turn_outcomes import (
    EventOutcomeRecord,
    Friction,
    FutureSnapshot,
    MaturedLabel,
    RoundTripResult,
    build_summary,
    calculate_round_trip,
    evaluate_gate,
    mature_event_labels,
    wilson_interval,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
ARRIVAL = datetime(2026, 8, 26, 9, 35, tzinfo=SHANGHAI)
ARRIVAL_PLUS_5 = ARRIVAL + timedelta(days=7)
ARRIVAL_PLUS_10 = ARRIVAL + timedelta(days=14)
SLOT_CAPITAL = 20000.0 * 0.95 / 3.0
NOMINAL = Friction(0.0003, 5.0, 0.001)
DOUBLED = Friction(0.0006, 10.0, 0.002)
ETF_CODES = ("510300", "159915", "512100", "159928", "513100", "513500")


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


def make_records(count=60, five_day_wins=50):
    """Build a passing fixture spanning six ETFs and seven natural months."""
    records = []
    for index in range(count):
        five_day_return = 0.02 if index < five_day_wins else -0.001
        labels = {
            horizon: _matured_label(
                f"e{index}", horizon, five_day_return if horizon == 5 else 0.01,
            )
            for horizon in (1, 3, 5, 10)
        }
        records.append(EventOutcomeRecord(
            f"e{index}",
            ETF_CODES[index % len(ETF_CODES)],
            date(2026, 1 + index % 7, 1 + index // 7),
            labels,
        ))
    return tuple(records)


def mutate_passing_records(name):
    records = make_records()
    if name == "span_under_six_months":
        return tuple(replace(record, arrival_date=date(2026, 1 + index % 5, 1))
                     for index, record in enumerate(records))
    if name == "only_four_etfs":
        return tuple(replace(record, code=ETF_CODES[index % 4])
                     for index, record in enumerate(records))
    if name == "one_etf_over_40_percent":
        return tuple(replace(record, code=ETF_CODES[0] if index < 25 else ETF_CODES[1 + index % 5])
                     for index, record in enumerate(records))
    if name == "wilson_lower_not_above_half":
        return _with_horizon_returns(records, 5, [0.02] * 37 + [-0.001] * 23)
    if name == "five_day_mean_non_positive":
        return _with_horizon_returns(records, 5, [0.02] * 50 + [-0.15] * 10)
    if name == "five_day_median_non_positive":
        return _with_horizon_returns(records, 5, [0.02] * 29 + [-0.001] * 31)
    if name == "ten_day_mean_negative":
        return _with_horizon_returns(records, 10, [0.02] * 50 + [-0.15] * 10)
    if name == "ten_day_median_negative":
        return _with_horizon_returns(records, 10, [0.02] * 29 + [-0.001] * 31)
    if name == "top_winner_dependency":
        return _with_horizon_returns(records, 5, [0.8] + [0.01] * 40 + [-0.03] * 19)
    raise ValueError(f"unknown mutation: {name}")


def _matured_label(event_id, horizon, doubled_return):
    doubled = _round_trip(doubled_return)
    return MaturedLabel(event_id, horizon, "matured", 2.0, _round_trip(doubled_return / 2), doubled, None, None)


def _round_trip(net_return):
    return RoundTripResult(100, 2.0, 2.0, 10.0, 10.0, net_return * SLOT_CAPITAL, net_return)


def _with_horizon_returns(records, horizon, returns):
    return tuple(replace(
        record,
        labels={
            key: _matured_label(record.event_id, key, value if key == horizon else label.doubled.net_return)
            for key, label in record.labels.items()
        },
    ) for record, value in zip(records, returns))


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


@pytest.mark.parametrize("available_at", [None, ARRIVAL_PLUS_5 + timedelta(seconds=1)])
def test_matured_snapshot_without_arrived_availability_is_not_labeled(available_at):
    source = FakeFuturePriceSource({
        1: FutureSnapshot(1, "matured", 2.10, None, None, available_at),
    })

    labels = mature_event_labels(valid_event(), source, ARRIVAL_PLUS_5)
    item = next(label for label in labels if label.horizon == 1)

    assert item.status == "pending_missing_executable_price"
    assert item.exit_price is None
    assert item.nominal is None
    assert item.doubled is None


def test_wilson_interval_is_not_the_raw_rate():
    lower, upper = wilson_interval(6, 7)

    assert lower == pytest.approx(0.486872, abs=1e-6)
    assert upper == pytest.approx(0.974321, abs=1e-6)


def test_under_fifty_is_accumulating():
    gate = evaluate_gate(make_records(count=49, five_day_wins=40))

    assert gate.status == "accumulating"
    assert gate.reasons == ("fewer_than_50_matured_five_day_events",)


@pytest.mark.parametrize(("mutation", "reason"), [
    ("span_under_six_months", "observation_span_under_six_months"),
    ("only_four_etfs", "fewer_than_five_etfs"),
    ("one_etf_over_40_percent", "single_etf_share_over_40_percent"),
    ("wilson_lower_not_above_half", "five_day_wilson_lower_not_above_50_percent"),
    ("five_day_mean_non_positive", "five_day_double_mean_not_positive"),
    ("five_day_median_non_positive", "five_day_double_median_not_positive"),
    ("ten_day_mean_negative", "ten_day_double_mean_negative"),
    ("ten_day_median_negative", "ten_day_double_median_negative"),
    ("top_winner_dependency", "leave_top_winner_out_mean_not_positive"),
])
def test_each_gate_fails_closed(mutation, reason):
    gate = evaluate_gate(mutate_passing_records(mutation))

    assert gate.status == "stop"
    assert reason in gate.reasons


def test_all_frozen_gates_can_pass_together():
    gate = evaluate_gate(make_records(count=60, five_day_wins=50))

    assert gate.status == "pass"
    assert gate.reasons == ()


def test_eligible_gate_evaluates_all_failures_without_short_circuiting():
    records = _with_horizon_returns(
        mutate_passing_records("one_etf_over_40_percent"), 5, [0.02] * 30 + [-0.15] * 30,
    )

    gate = evaluate_gate(records)

    assert gate.status == "stop"
    assert "single_etf_share_over_40_percent" in gate.reasons
    assert "five_day_wilson_lower_not_above_50_percent" in gate.reasons
    assert "five_day_double_mean_not_positive" in gate.reasons
    assert "five_day_double_median_not_positive" in gate.reasons


def test_summary_contains_frozen_identity_metrics_and_gate_result():
    summary = build_summary(
        make_records(),
        collection_start=date(2026, 8, 26),
        generated_at=ARRIVAL_PLUS_10,
    )

    assert summary["version"] == "rsi-low-turn-shadow-v0.1"
    assert summary["collection_start"] == "2026-08-26"
    assert summary["generated_at"] == ARRIVAL_PLUS_10.isoformat()
    assert summary["counts"]["matured_five_day_events"] == 60
    assert summary["date_span"] == {"start": "2026-01-01", "end": "2026-07-08", "natural_months": 7}
    assert summary["etf_distribution"] == {code: 10 for code in ETF_CODES}
    assert summary["return_metrics"]["5"]["doubled"]["mean"] > 0
    assert summary["return_metrics"]["10"]["nominal"]["median"] > 0
    assert summary["wilson_interval"]["lower"] > 0.5
    assert summary["leave_top_winner_out_mean"] > 0
    assert summary["status"] == "pass"
    assert summary["reasons"] == []
