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
