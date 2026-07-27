# -*- coding: utf-8 -*-
"""Tests for the fixed same-side contraction/re-expansion observation."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def observation(
    *,
    direction,
    group,
    year=2019,
    forward_5=0.02,
    code="AAA",
    day="03-01",
):
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        ReexpansionObservation,
    )

    return ReexpansionObservation(
        code=code,
        execution_date=f"{year}-{day}",
        signal_date=f"{year}-02-28",
        direction=direction,
        group=group,
        indicators=("macd",),
        forward_returns={1: forward_5 / 2.0, 3: forward_5, 5: forward_5, 10: forward_5},
    )


def test_classification_requires_same_side_contraction_then_reexpansion():
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        classify_same_side_reexpansion,
    )

    assert classify_same_side_reexpansion(
        fast=np.array([3.0, 2.0, 3.0]),
        slow=np.zeros(3),
    ) == "bullish"
    assert classify_same_side_reexpansion(
        fast=np.array([-3.0, -2.0, -3.0]),
        slow=np.zeros(3),
    ) == "bearish"

    assert classify_same_side_reexpansion(
        fast=np.array([-1.0, 0.5, 1.0]),
        slow=np.zeros(3),
    ) == "none"
    assert classify_same_side_reexpansion(
        fast=np.array([3.0, 2.0, 1.0]),
        slow=np.zeros(3),
    ) == "none"
    assert classify_same_side_reexpansion(
        fast=np.array([3.0, np.nan, 4.0]),
        slow=np.zeros(3),
    ) == "no_data"


def test_indicator_flags_are_symmetric_and_reject_future_rows(monkeypatch):
    from cross_signal_strategy.research import reexpansion_diagnostics as module

    dates = pd.bdate_range(end="2019-03-29", periods=40)
    close = np.linspace(10.0, 12.0, len(dates))
    frame = pd.DataFrame({
        "date": dates,
        "close": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "volume": np.full(len(dates), 1000.0),
    })

    monkeypatch.setattr(
        module.strategy,
        "calc_rsi",
        lambda values, period: pd.Series([0.0] * 37 + [3.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        module.strategy,
        "calc_macd",
        lambda values, fast, slow, signal: (
            pd.Series([0.0] * 37 + [-3.0, -2.0, -3.0]),
            pd.Series(np.zeros(40)),
            pd.Series(np.zeros(40)),
        ),
    )
    monkeypatch.setattr(
        module.strategy,
        "calc_kdj",
        lambda high, low, values, n, m1, m2: (
            pd.Series([0.0] * 37 + [3.0, 2.0, 3.0]),
            pd.Series(np.zeros(40)),
            pd.Series([0.0] * 37 + [-3.0, -2.0, -3.0]),
        ),
    )

    flags = module.build_reexpansion_flags(frame, signal_date="2019-03-29")

    assert flags["rsi6_rsi12"] == "none"
    assert flags["rsi6_rsi24"] == "none"
    assert flags["macd"] == "bearish"
    assert flags["kdj_k"] == "bullish"
    assert flags["kdj_j"] == "bearish"

    future = frame.copy()
    future.loc[len(future)] = {
        "date": pd.Timestamp("2019-04-01"),
        "close": 12.1,
        "high": 12.2,
        "low": 12.0,
        "volume": 1000.0,
    }
    with pytest.raises(ValueError, match="after signal_date"):
        module.build_reexpansion_flags(future, signal_date="2019-03-29")


def test_report_uses_five_day_primary_gate_and_keeps_other_horizons_descriptive():
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        build_reexpansion_report,
    )

    rows = []
    for year in (2019, 2020, 2021):
        for index in range(10):
            rows.append(observation(
                direction="bullish",
                group="novel_reexpansion",
                year=year,
                forward_5=0.04,
                code=f"B{index}",
                day=f"03-{index + 1:02d}",
            ))
            rows.append(observation(
                direction="bullish",
                group="existing_cross",
                year=year,
                forward_5=0.01 if index < 5 else -0.01,
                code=f"C{index}",
                day=f"04-{index + 1:02d}",
            ))
            rows.append(observation(
                direction="bearish",
                group="novel_reexpansion",
                year=year,
                forward_5=-0.04,
                code=f"D{index}",
                day=f"05-{index + 1:02d}",
            ))
            rows.append(observation(
                direction="bearish",
                group="existing_cross",
                year=year,
                forward_5=-0.01 if index < 5 else 0.01,
                code=f"E{index}",
                day=f"06-{index + 1:02d}",
            ))

    report = build_reexpansion_report(rows)

    assert report.gate.passed is True
    assert report.gate.primary_horizon == 5
    assert report.by_direction_group_horizon[
        "bullish:novel_reexpansion:5"
    ].observations == 30
    assert report.by_direction_group_horizon[
        "bearish:novel_reexpansion:5"
    ].directional_success_rate == pytest.approx(1.0)
    assert "bullish:novel_reexpansion:10" in report.by_direction_group_horizon


def test_report_rejects_sparse_or_annually_inconsistent_evidence():
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        build_reexpansion_report,
    )

    rows = []
    for year in (2019, 2020, 2021):
        for index in range(10):
            novel_return = -0.01 if year == 2020 else 0.04
            rows.append(observation(
                direction="bullish",
                group="novel_reexpansion",
                year=year,
                forward_5=novel_return,
                code=f"B{index}",
                day=f"03-{index + 1:02d}",
            ))
            rows.append(observation(
                direction="bullish",
                group="existing_cross",
                year=year,
                forward_5=0.01,
                code=f"C{index}",
                day=f"04-{index + 1:02d}",
            ))

    report = build_reexpansion_report(rows)

    assert report.gate.passed is False
    assert any("bearish" in reason and "fewer than 30" in reason for reason in report.gate.reasons)
    assert any("2020 bullish" in reason and "average return" in reason for reason in report.gate.reasons)


def test_report_rejects_any_observation_outside_training_window():
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        build_reexpansion_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_reexpansion_report([
            observation(
                direction="bullish",
                group="novel_reexpansion",
                year=2022,
            )
        ])


def test_report_allows_2018_warmup_signal_for_2019_execution():
    from cross_signal_strategy.research.reexpansion_diagnostics import (
        ReexpansionObservation,
        build_reexpansion_report,
    )

    report = build_reexpansion_report([
        ReexpansionObservation(
            code="AAA",
            execution_date="2019-01-02",
            signal_date="2018-12-28",
            direction="bullish",
            group="novel_reexpansion",
            indicators=("macd",),
            forward_returns={5: 0.01},
        )
    ])

    assert report.event_count == 1
