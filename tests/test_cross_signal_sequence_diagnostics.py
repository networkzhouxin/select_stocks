# -*- coding: utf-8 -*-
"""Tests for observation-only cross-event sequence diagnostics."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSignalAdapter:
    def __init__(self, frame, signal_date="2019-02-20"):
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


def signal_frame(end="2019-02-20", periods=40):
    dates = pd.bdate_range(end=end, periods=periods)
    close = np.linspace(10.0, 12.0, periods) + np.sin(np.arange(periods)) * 0.2
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "volume": np.full(periods, 1000.0),
    })


def closed_trade(sequence, year=2019, pnl=100.0, trend_score=10):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date="%d-03-01" % year,
        sell_date="%d-03-08" % year,
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 + float(pnl) / 100.0,
        pnl=float(pnl),
        return_pct=float(pnl) / 10.0,
        entry_score={
            "trend_score": trend_score,
            "cross_sequence": sequence,
        },
    )


def stats(trades, average_return, win_rate):
    from cross_signal_strategy.research.sequence_diagnostics import SequenceStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    return SequenceStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=average_return * trades * 100.0,
        gross_profit=max(0.0, average_return * trades * 120.0),
        gross_loss=max(0.0, -average_return * trades * 20.0),
        average_return=average_return,
    )


def test_latest_cross_event_keeps_latest_direction_and_days_ago():
    from cross_signal_strategy.research.sequence_diagnostics import latest_cross_event

    fast = np.array([-1.0, 1.0, -1.0, 1.0])
    slow = np.zeros(4)
    event = latest_cross_event(fast, slow, window=3)

    assert event.direction == "above"
    assert event.days_ago == 0

    down = latest_cross_event(np.array([-1.0, 1.0, -1.0]), np.zeros(3), window=2)
    assert down.direction == "below"
    assert down.days_ago == 0


def test_classify_cross_sequence_keeps_clean_and_mixed_orders_separate():
    from cross_signal_strategy.research.sequence_diagnostics import classify_cross_sequence

    assert classify_cross_sequence(0, [2, 1]) == "oscillators_lead_macd"
    assert classify_cross_sequence(2, [1, 0]) == "macd_leads_oscillators"
    assert classify_cross_sequence(1, [1, 1]) == "same_day"
    assert classify_cross_sequence(1, [2, 0]) == "mixed"
    assert classify_cross_sequence(None, [1]) == "no_macd_confirmation"
    assert classify_cross_sequence(1, []) == "macd_only"


def test_sequence_adapter_uses_frozen_t_minus_one_frame_and_returns_copy():
    from cross_signal_strategy.research.sequence_diagnostics import CrossSequenceSignalAdapter

    adapter = CrossSequenceSignalAdapter(FakeSignalAdapter(signal_frame()))

    first = adapter.score("AAA", "2019-02-21")
    first["cross_sequence"] = "mutated"
    second = adapter.score("AAA", "2019-02-21")

    assert second["cross_sequence"] != "mutated"
    assert second["sequence_data_date"] == "2019-02-20"
    assert second["sequence_data_date"] == second["signal_date"]


def test_sequence_adapter_rejects_data_after_signal_date():
    from cross_signal_strategy.research.sequence_diagnostics import CrossSequenceSignalAdapter

    adapter = CrossSequenceSignalAdapter(
        FakeSignalAdapter(signal_frame(end="2019-02-21"), signal_date="2019-02-20")
    )

    with pytest.raises(ValueError, match="after signal_date"):
        adapter.score("AAA", "2019-02-22")


def test_sequence_attribution_splits_sequence_trend_and_year():
    from cross_signal_strategy.research.sequence_diagnostics import build_sequence_attribution

    report = build_sequence_attribution([
        closed_trade("oscillators_lead_macd", year=2019, pnl=100.0, trend_score=10),
        closed_trade("macd_leads_oscillators", year=2019, pnl=-40.0, trend_score=10),
        closed_trade("same_day", year=2020, pnl=60.0, trend_score=20),
    ])

    assert report.by_sequence["oscillators_lead_macd"].realized_pnl == pytest.approx(100.0)
    assert report.by_trend_sequence["mild_up:macd_leads_oscillators"].realized_pnl == pytest.approx(-40.0)
    assert report.by_trend_sequence["strong_up:same_day"].closed_trades == 1
    assert report.mild_by_year_sequence["2019:oscillators_lead_macd"].closed_trades == 1
    assert "2020:same_day" not in report.mild_by_year_sequence


def test_sequence_gate_requires_stable_oscillator_lead_advantage():
    from cross_signal_strategy.research.sequence_diagnostics import evaluate_sequence_gate

    oscillator_lead = {
        year: stats(4, average_return=0.05, win_rate=0.75)
        for year in (2019, 2020, 2021)
    }
    macd_lead = {
        year: stats(4, average_return=0.01, win_rate=0.25)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_sequence_gate(oscillator_lead, macd_lead)
    failed = evaluate_sequence_gate(
        {**oscillator_lead, 2021: stats(4, average_return=-0.01, win_rate=0.25)},
        macd_lead,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_sequence_attribution_rejects_non_training_dates():
    from cross_signal_strategy.research.sequence_diagnostics import build_sequence_attribution

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_sequence_attribution([
            closed_trade("oscillators_lead_macd", year=2022, pnl=100.0)
        ])
