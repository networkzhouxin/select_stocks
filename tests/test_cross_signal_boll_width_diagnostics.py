# -*- coding: utf-8 -*-
"""Tests for observation-only BOLL BandWidth diagnostics."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSignalAdapter:
    def __init__(self, frame, signal_date="2019-01-03"):
        self.frame = frame
        self.signal_date = signal_date

    def score(self, code, current_date, return_reason=False):
        result = {
            "code": code,
            "current_date": current_date,
            "signal_date": self.signal_date,
            "trend_score": 10,
        }
        return (result, None) if return_reason else result

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), self.signal_date


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "close"])


def closed_trade(direction, year=2019, pnl=100.0, trend_score=10):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date="%d-01-02" % year,
        sell_date="%d-01-10" % year,
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 + float(pnl) / 100.0,
        pnl=float(pnl),
        return_pct=float(pnl) / 10.0,
        entry_score={
            "trend_score": trend_score,
            "boll_width_direction": direction,
            "boll_width": 0.2,
            "boll_width_change": 0.01 if direction == "rising" else -0.01,
        },
    )


def stats(trades, average_return, win_rate, profit_loss_ratio=1.0):
    from cross_signal_strategy.boll_width_diagnostics import BollWidthStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    gross_loss = 100.0 if losses else 0.0
    gross_profit = gross_loss * profit_loss_ratio if gross_loss else 100.0
    return BollWidthStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=gross_profit - gross_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        average_return=average_return,
        average_width=0.2,
        average_change=0.01,
    )


def test_calc_boll_bandwidth_uses_standard_rolling_band_definition():
    from cross_signal_strategy.boll_width_diagnostics import calc_boll_bandwidth

    data = frame([
        ("2019-01-01", 1.0),
        ("2019-01-02", 3.0),
        ("2019-01-03", 3.0),
    ])

    width = calc_boll_bandwidth(data, period=2, std_mult=2.0)

    assert pd.isna(width.iloc[0])
    assert width.iloc[1] == pytest.approx(2.0 * (2.0 ** 0.5))
    assert width.iloc[2] == pytest.approx(0.0)


def test_boll_width_adapter_uses_frozen_signal_frame_and_returns_copy():
    from cross_signal_strategy.boll_width_diagnostics import BollWidthSignalAdapter

    data = frame([
        ("2019-01-01", 1.0),
        ("2019-01-02", 3.0),
        ("2019-01-03", 4.0),
    ])
    adapter = BollWidthSignalAdapter(FakeSignalAdapter(data), period=2, std_mult=2.0)

    first = adapter.score("AAA", "2019-01-04")
    first["boll_width"] = 999.0
    second = adapter.score("AAA", "2019-01-04")

    assert second["boll_width"] != 999.0
    assert second["boll_width_data_date"] == "2019-01-03"
    assert second["boll_width_data_date"] == second["signal_date"]
    assert second["boll_width_direction"] == "declining"


def test_boll_width_adapter_rejects_data_after_signal_date():
    from cross_signal_strategy.boll_width_diagnostics import BollWidthSignalAdapter

    data = frame([
        ("2019-01-02", 1.0),
        ("2019-01-03", 2.0),
    ])
    adapter = BollWidthSignalAdapter(
        FakeSignalAdapter(data, signal_date="2019-01-02"),
        period=2,
        std_mult=2.0,
    )

    with pytest.raises(ValueError, match="after signal_date"):
        adapter.score("AAA", "2019-01-04")


def test_boll_width_attribution_splits_direction_trend_and_year():
    from cross_signal_strategy.boll_width_diagnostics import (
        build_boll_width_attribution,
    )

    report = build_boll_width_attribution([
        closed_trade("rising", year=2019, pnl=100.0, trend_score=10),
        closed_trade("declining", year=2019, pnl=-40.0, trend_score=10),
        closed_trade("rising", year=2020, pnl=60.0, trend_score=20),
    ])

    assert report.by_direction["rising"].closed_trades == 2
    assert report.by_direction["rising"].realized_pnl == pytest.approx(160.0)
    assert report.by_trend_direction["mild_up:declining"].realized_pnl == pytest.approx(-40.0)
    assert report.by_trend_direction["strong_up:rising"].realized_pnl == pytest.approx(60.0)
    assert report.by_year_direction["2019:rising"].closed_trades == 1
    assert report.by_year_direction["2020:rising"].closed_trades == 1
    assert report.mild_by_year_direction["2019:rising"].closed_trades == 1
    assert report.mild_by_year_direction["2019:declining"].closed_trades == 1
    assert "2020:rising" not in report.mild_by_year_direction


def test_boll_width_gate_requires_stable_mild_trend_improvement():
    from cross_signal_strategy.boll_width_diagnostics import (
        evaluate_boll_width_gate,
    )

    rising = {
        year: stats(6, average_return=0.05, win_rate=0.67, profit_loss_ratio=3.0)
        for year in (2019, 2020, 2021)
    }
    non_rising = {
        year: stats(6, average_return=0.01, win_rate=0.33, profit_loss_ratio=1.2)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_boll_width_gate(rising, non_rising)
    failed = evaluate_boll_width_gate(
        {**rising, 2021: stats(6, average_return=-0.01, win_rate=0.17, profit_loss_ratio=0.5)},
        non_rising,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_boll_width_attribution_rejects_non_training_dates():
    from cross_signal_strategy.boll_width_diagnostics import (
        build_boll_width_attribution,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_boll_width_attribution([
            closed_trade("rising", year=2022, pnl=100.0, trend_score=10)
        ])
