# -*- coding: utf-8 -*-
"""Behavior tests for the frozen late-MACD/BOLL-upper observation."""

from __future__ import annotations

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def late_snapshot(**overrides):
    snapshot = {
        "code": "513100",
        "current_date": "2020-02-03",
        "signal_date": "2020-01-23",
        "max_data_date": "2020-01-23",
        "close": 3.05,
        "boll_upper": 3.00,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "rsi6_cross_rsi12_up_age": 1,
        "rsi6_cross_rsi24_up_age": None,
        "macd_cross_up": True,
        "macd_cross_up_age": 0,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "kdj_k_cross_up_age": 2,
        "kdj_j_cross_up_age": None,
    }
    snapshot.update(overrides)
    return snapshot


def test_rule_matches_only_current_macd_after_prior_rsi_and_kdj_at_upper_band():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        is_late_macd_boll_upper_entry,
    )

    assert is_late_macd_boll_upper_entry(late_snapshot()) is True
    assert is_late_macd_boll_upper_entry(
        late_snapshot(macd_cross_up_age=1)
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(rsi6_cross_rsi12_up_age=0)
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(kdj_k_cross_up_age=0)
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(close=2.999999)
    ) is False


def test_rule_rejects_mixed_rsi_direction_and_missing_or_nonfinite_prices():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        is_late_macd_boll_upper_entry,
    )

    assert is_late_macd_boll_upper_entry(
        late_snapshot(rsi6_cross_rsi12_down=True)
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(macd_cross_up_age=None)
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(close=float("nan"))
    ) is False
    assert is_late_macd_boll_upper_entry(
        late_snapshot(boll_upper=None)
    ) is False


def test_observation_counts_only_official_filled_buys_and_applies_frozen_gate():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        observe_official_filled_buys,
    )
    from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent

    events = [
        OrderPathEvent(date="2019-02-01", side="BUY", code="513100"),
        OrderPathEvent(date="2019-03-01", side="SELL", code="513100"),
        OrderPathEvent(date="2020-02-03", side="BUY", code="513100"),
        OrderPathEvent(date="2020-03-02", side="BUY", code="513500"),
        OrderPathEvent(date="2021-01-04", side="BUY", code="513050"),
    ]
    snapshots = {
        ("513100", "2019-02-01"): late_snapshot(
            code="513100", current_date="2019-02-01",
            signal_date="2019-01-31", max_data_date="2019-01-31",
        ),
        ("513100", "2020-02-03"): late_snapshot(),
        ("513500", "2020-03-02"): late_snapshot(
            code="513500", current_date="2020-03-02",
            signal_date="2020-02-28", max_data_date="2020-02-28",
        ),
        ("513050", "2021-01-04"): late_snapshot(
            code="513050", current_date="2021-01-04",
            signal_date="2020-12-31", max_data_date="2020-12-31",
            close=2.9,
        ),
    }

    result = observe_official_filled_buys(
        events,
        lambda code, date: snapshots[(code, date)],
    )

    assert result.total_filled_buys == 4
    assert [(event.date, event.code) for event in result.matched_events] == [
        ("2019-02-01", "513100"),
        ("2020-02-03", "513100"),
        ("2020-03-02", "513500"),
    ]
    assert result.distinct_years == (2019, 2020)
    assert result.gate_passed is True


def test_observation_rejects_future_rows_before_counting_event():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        observe_official_filled_buys,
    )
    from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent

    event = OrderPathEvent(date="2020-02-03", side="BUY", code="513100")
    future = late_snapshot(max_data_date="2020-02-04")

    with pytest.raises(ValueError, match="future data"):
        observe_official_filled_buys([event], lambda code, date: future)


def test_observation_gate_fails_below_three_events_or_two_years():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        observe_official_filled_buys,
    )
    from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent

    events = [
        OrderPathEvent(date="2020-02-03", side="BUY", code="513100"),
        OrderPathEvent(date="2020-03-02", side="BUY", code="513500"),
    ]

    result = observe_official_filled_buys(
        events,
        lambda code, date: late_snapshot(
            code=code,
            current_date=date,
            signal_date="2020-01-23",
            max_data_date="2020-01-23",
        ),
    )

    assert len(result.matched_events) == 2
    assert result.distinct_years == (2020,)
    assert result.gate_passed is False


def test_mixed_encoding_log_decode_preserves_ascii_order_lines():
    from cross_signal_strategy.research.late_macd_boll_filter import (
        decode_joinquant_log_bytes,
    )

    raw = (
        b"2020-02-03 09:35:00 - INFO - order StockOrder(action=open) "
        b"trade price: 3.0, amount:100, commission:5.0\n"
        b"warning=" + bytes([0xBF, 0xE2]) + b"\n"
    )

    decoded = decode_joinquant_log_bytes(raw)

    assert "2020-02-03 09:35:00" in decoded
    assert "action=open" in decoded
