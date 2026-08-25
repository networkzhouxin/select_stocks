from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from cross_signal_strategy.research.rsi_low_turn_shadow import (
    RsiTurnInput,
    calculate_rsi6,
    detect_rsi_low_turn,
)


def make_input(**overrides):
    values = {
        "code": "513100",
        "arrival_dt": datetime(2026, 8, 26, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai")),
        "signal_date": date(2026, 8, 25),
        "r2": 24.0,
        "r1": 18.0,
        "r0": 21.0,
        "c1": 2.00,
        "c0": 2.01,
        "entry_open": 2.035,
        "price_proved": True,
    }
    values.update(overrides)
    return RsiTurnInput(**values)


def test_rsi6_matches_formal_formula():
    close = pd.Series([10, 9, 8, 8.5, 8.2, 8.8, 9.1, 8.9, 9.4, 9.7])
    actual = calculate_rsi6(close)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 6, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1 / 6, min_periods=6).mean()
    expected = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    pd.testing.assert_series_equal(actual, expected)


def test_exact_low_turn_needs_no_kdj_or_macd_confirmation():
    item = make_input(r2=24, r1=18, r0=21, c1=2.00, c0=2.01)
    decision = detect_rsi_low_turn(item)
    assert decision.signal_detected is True
    assert decision.valid_event is True
    assert decision.reasons == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"r2": 18},
        {"r0": 18},
        {"r1": 30.01},
        {"c0": 2.00},
    ],
)
def test_equal_or_failed_condition_is_not_a_turn(changes):
    item = replace(
        make_input(r2=24, r1=18, r0=21, c1=2.00, c0=2.01),
        **changes,
    )
    assert detect_rsi_low_turn(item).signal_detected is False
