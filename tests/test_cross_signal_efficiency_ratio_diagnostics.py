# -*- coding: utf-8 -*-
"""Tests for observation-only Kaufman Efficiency Ratio diagnostics."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSignalAdapter:
    def __init__(self, frame, signal_date="2019-01-05"):
        self.frame = frame
        self.signal_date = signal_date

    def score(self, code, current_date, return_reason=False):
        result = {
            "code": code,
            "signal_date": self.signal_date,
            "trend_score": 10,
        }
        return (result, None) if return_reason else result

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), self.signal_date


def frame(closes, end="2019-01-05"):
    dates = pd.date_range(end=end, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "close": closes})


def closed_trade(direction, year=2019, pnl=100.0, trend_score=10):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date="%d-02-01" % year,
        sell_date="%d-02-08" % year,
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 + float(pnl) / 100.0,
        pnl=float(pnl),
        return_pct=float(pnl) / 10.0,
        entry_score={
            "trend_score": trend_score,
            "efficiency_ratio": 0.4,
            "efficiency_ratio_change": 0.1 if direction == "rising" else -0.1,
            "efficiency_ratio_direction": direction,
        },
    )


def stats(trades, average_return, win_rate):
    from cross_signal_strategy.efficiency_ratio_diagnostics import EfficiencyRatioStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    return EfficiencyRatioStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=average_return * trades * 100.0,
        gross_profit=max(0.0, average_return * trades * 120.0),
        gross_loss=max(0.0, -average_return * trades * 20.0),
        average_return=average_return,
        average_ratio=0.4,
        average_change=0.1,
    )


def test_calc_efficiency_ratio_handles_trend_noise_and_flat_windows():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        calc_efficiency_ratio,
    )

    trend = calc_efficiency_ratio(frame([0.0, 1.0, 2.0, 3.0]), period=3)
    noisy = calc_efficiency_ratio(frame([0.0, 1.0, 0.0, 1.0]), period=3)
    flat = calc_efficiency_ratio(frame([1.0, 1.0, 1.0, 1.0]), period=3)

    assert pd.isna(trend.iloc[2])
    assert trend.iloc[3] == pytest.approx(1.0)
    assert noisy.iloc[3] == pytest.approx(1.0 / 3.0)
    assert flat.iloc[3] == pytest.approx(0.0)


def test_efficiency_ratio_adapter_uses_frozen_frame_and_returns_copy():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        EfficiencyRatioSignalAdapter,
    )

    adapter = EfficiencyRatioSignalAdapter(
        FakeSignalAdapter(frame([0.0, 1.0, 2.0, 3.0, 2.0])),
        period=3,
    )

    first = adapter.score("AAA", "2019-01-06")
    first["efficiency_ratio"] = 999.0
    second = adapter.score("AAA", "2019-01-06")

    assert second["efficiency_ratio"] == pytest.approx(1.0 / 3.0)
    assert second["efficiency_ratio_direction"] == "declining"
    assert second["efficiency_ratio_data_date"] == "2019-01-05"
    assert second["efficiency_ratio_data_date"] == second["signal_date"]


def test_efficiency_ratio_adapter_rejects_data_after_signal_date():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        EfficiencyRatioSignalAdapter,
    )

    adapter = EfficiencyRatioSignalAdapter(
        FakeSignalAdapter(
            frame([0.0, 1.0, 2.0, 3.0], end="2019-01-06"),
            signal_date="2019-01-05",
        ),
        period=3,
    )

    with pytest.raises(ValueError, match="after signal_date"):
        adapter.score("AAA", "2019-01-07")


def test_efficiency_ratio_attribution_splits_direction_trend_and_year():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        build_efficiency_ratio_attribution,
    )

    report = build_efficiency_ratio_attribution([
        closed_trade("rising", year=2019, pnl=100.0, trend_score=10),
        closed_trade("declining", year=2019, pnl=-40.0, trend_score=10),
        closed_trade("rising", year=2020, pnl=60.0, trend_score=20),
    ])

    assert report.by_direction["rising"].realized_pnl == pytest.approx(160.0)
    assert report.by_trend_direction["mild_up:declining"].realized_pnl == pytest.approx(-40.0)
    assert report.by_trend_direction["strong_up:rising"].closed_trades == 1
    assert report.mild_by_year_direction["2019:rising"].closed_trades == 1
    assert "2020:rising" not in report.mild_by_year_direction


def test_efficiency_ratio_gate_requires_stable_mild_trend_improvement():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        evaluate_efficiency_ratio_gate,
    )

    rising = {
        year: stats(6, average_return=0.05, win_rate=0.67)
        for year in (2019, 2020, 2021)
    }
    non_rising = {
        year: stats(6, average_return=0.01, win_rate=0.33)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_efficiency_ratio_gate(rising, non_rising)
    failed = evaluate_efficiency_ratio_gate(
        {**rising, 2021: stats(6, average_return=-0.01, win_rate=0.17)},
        non_rising,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_efficiency_ratio_attribution_rejects_non_training_dates():
    from cross_signal_strategy.efficiency_ratio_diagnostics import (
        build_efficiency_ratio_attribution,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_efficiency_ratio_attribution([
            closed_trade("rising", year=2022, pnl=100.0)
        ])
