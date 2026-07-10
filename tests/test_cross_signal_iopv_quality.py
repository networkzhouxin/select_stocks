# -*- coding: utf-8 -*-
"""Tests for training-only IOPV data-quality diagnostics."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample_frame():
    return pd.DataFrame([
        {
            "code": "513100",
            "date": "2019-01-02",
            "time": "09:31",
            "close": 1.01,
            "volume": 100,
            "num_trades": 2,
            "iopv": 1.00,
        },
        {
            "code": "513100",
            "date": "2019-01-02",
            "time": "09:35",
            "close": 1.03,
            "volume": 0,
            "num_trades": 0,
            "iopv": 1.01,
        },
        {
            "code": "513100",
            "date": "2019-01-02",
            "time": "09:36",
            "close": 1.03,
            "volume": 0,
            "num_trades": 0,
            "iopv": 1.02,
        },
        {
            "code": "513100",
            "date": "2019-01-03",
            "time": "09:35",
            "close": 1.00,
            "volume": 10,
            "num_trades": 1,
            "iopv": None,
        },
        {
            "code": "513100",
            "date": "2019-01-03",
            "time": "09:36",
            "close": 1.00,
            "volume": 10,
            "num_trades": 1,
            "iopv": 0.0,
        },
        {
            "code": "513100",
            "date": "2019-01-03",
            "time": "09:37",
            "close": 1.00,
            "volume": 10,
            "num_trades": 1,
            "iopv": float("inf"),
        },
    ])


def test_profile_rejects_missing_required_columns():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    with pytest.raises(ValueError, match="Missing required columns: iopv"):
        profile_iopv_frame(sample_frame().drop(columns=["iopv"]), "513100", 2019)


def test_profile_rejects_rows_outside_training_window():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    frame = sample_frame()
    frame.loc[0, "date"] = "2022-01-04"

    with pytest.raises(ValueError, match="outside training window"):
        profile_iopv_frame(frame, "513100", 2019)


def test_profile_counts_iopv_validity_and_0935_coverage():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    stats = profile_iopv_frame(sample_frame(), "513100", 2019)

    assert stats.rows == 6
    assert stats.trading_days == 2
    assert stats.missing_iopv_rows == 1
    assert stats.nonpositive_iopv_rows == 1
    assert stats.nonfinite_iopv_rows == 1
    assert stats.valid_iopv_rows == 3
    assert stats.valid_iopv_rate == pytest.approx(0.5)
    assert stats.bar_0935_days == 2
    assert stats.valid_iopv_0935_days == 1
    assert stats.valid_iopv_0935_rate == pytest.approx(0.5)
    assert stats.missing_iopv_0935_dates == ("2019-01-03",)


def test_profile_calculates_premium_distribution_only_from_valid_pairs():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    stats = profile_iopv_frame(sample_frame(), "513100", 2019)

    expected = sorted([1.01 / 1.00 - 1.0, 1.03 / 1.01 - 1.0, 1.03 / 1.02 - 1.0])
    assert stats.premium_observations == 3
    assert stats.premium_min == pytest.approx(expected[0])
    assert stats.premium_median == pytest.approx(expected[1])
    assert stats.premium_max == pytest.approx(expected[2])
    assert stats.premium_0935_observations == 1
    assert stats.premium_0935_median == pytest.approx(1.03 / 1.01 - 1.0)


def test_profile_detects_duplicate_minutes_and_iopv_updates_without_trades():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    frame = sample_frame()
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    stats = profile_iopv_frame(frame, "513100", 2019)

    assert stats.duplicate_minute_rows == 2
    assert stats.no_trade_rows == 3
    assert stats.no_trade_valid_iopv_rows == 3
    assert stats.no_trade_iopv_change_rows == 2


def test_profile_separates_executable_0935_premium_from_no_trade_rows():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    frame = sample_frame()
    frame.loc[
        (frame["date"] == "2019-01-03") & (frame["time"] == "09:35"),
        "iopv",
    ] = 1.00

    stats = profile_iopv_frame(frame, "513100", 2019)

    assert stats.executable_0935_days == 1
    assert stats.executable_valid_iopv_0935_days == 1
    assert stats.executable_valid_iopv_0935_rate == pytest.approx(1.0)
    assert stats.premium_executable_0935_observations == 1
    assert stats.premium_executable_0935_median == pytest.approx(0.0)


def test_profile_rejects_code_or_year_mismatch():
    from cross_signal_strategy.iopv_quality_diagnostics import profile_iopv_frame

    with pytest.raises(ValueError, match="does not match requested code"):
        profile_iopv_frame(sample_frame(), "513500", 2019)

    with pytest.raises(ValueError, match="does not match requested year"):
        profile_iopv_frame(sample_frame(), "513100", 2020)


def test_audit_training_iopv_rejects_non_training_years_before_loading():
    from cross_signal_strategy.iopv_quality_diagnostics import audit_training_iopv

    with pytest.raises(ValueError, match="years must be within 2019-2021"):
        audit_training_iopv(codes=["513100"], years=[2022])


def test_audit_training_iopv_profiles_each_requested_code_year():
    from cross_signal_strategy.iopv_quality_diagnostics import audit_training_iopv

    calls = []

    class FakeLoader:
        def load_minute_frame(self, code, trade_date):
            calls.append((code, trade_date))
            return sample_frame()

    report = audit_training_iopv(
        codes=["513100"],
        years=[2019],
        loader_factory=FakeLoader,
    )

    assert calls == [("513100", "2019-01-02")]
    assert len(report) == 1
    assert report[0].code == "513100"
    assert report[0].year == 2019
