# -*- coding: utf-8 -*-
"""Tests for T-2-safe horizontal support/resistance diagnostics."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def price_frame(
    prior_high=110.0,
    prior_low=90.0,
    signal_close=108.0,
    signal_high=109.0,
    signal_low=107.0,
    prior_count=20,
    signal_date="2019-02-01",
):
    dates = pd.bdate_range(end=signal_date, periods=prior_count + 1)
    rows = []
    for index, date in enumerate(dates[:-1]):
        rows.append({
            "date": date,
            "open": 100.0,
            "high": float(prior_high if index == prior_count - 1 else 105.0),
            "low": float(prior_low if index == 0 else 95.0),
            "close": 100.0,
        })
    rows.append({
        "date": dates[-1],
        "open": signal_close,
        "high": signal_high,
        "low": signal_low,
        "close": signal_close,
    })
    return pd.DataFrame(rows)


class FakeSource:
    def __init__(self, frame, score=None, signal_date="2019-02-01"):
        self.frame = frame
        self.signal_date = signal_date
        self.base_score = score or {
            "code": "AAA",
            "signal_date": signal_date,
            "close": 108.0,
            "atr": 2.0,
            "trend_score": 10,
        }

    def score(self, code, current_date, return_reason=False):
        result = self.base_score
        return (result, None) if return_reason else result

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), self.signal_date


def closed_trade(
    pressure_bucket,
    support_bucket="away_from_support",
    year=2019,
    pnl=100.0,
    trend_score=10,
    resistance_distance_atr=0.5,
    support_distance_atr=4.0,
):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date=f"{year}-02-01",
        sell_date=f"{year}-02-08",
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 + float(pnl) / 100.0,
        pnl=float(pnl),
        return_pct=float(pnl) / 10.0,
        entry_score={
            "trend_score": trend_score,
            "pressure_bucket": pressure_bucket,
            "support_bucket": support_bucket,
            "resistance_distance_atr": resistance_distance_atr,
            "support_distance_atr": support_distance_atr,
        },
    )


def stats(trades, average_return, win_rate):
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        HorizontalStructureStats,
    )

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    return HorizontalStructureStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=average_return * trades * 100.0,
        gross_profit=max(0.0, average_return * trades * 120.0),
        gross_loss=max(0.0, -average_return * trades * 20.0),
        average_return=average_return,
        average_resistance_distance_atr=0.5,
        average_support_distance_atr=4.0,
    )


def test_levels_use_exactly_twenty_bars_strictly_before_signal_date():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        calc_horizontal_structure,
    )

    frame = price_frame(
        prior_high=110.0,
        prior_low=90.0,
        signal_close=108.0,
        signal_high=150.0,
        signal_low=50.0,
    )
    snapshot = calc_horizontal_structure(
        frame,
        signal_date="2019-02-01",
        atr=2.0,
        period=20,
    )

    assert snapshot.eligible is True
    assert snapshot.resistance == pytest.approx(110.0)
    assert snapshot.support == pytest.approx(90.0)
    assert snapshot.resistance_distance_atr == pytest.approx(1.0)
    assert snapshot.support_distance_atr == pytest.approx(9.0)
    assert snapshot.pressure_bucket == "near_resistance"
    assert snapshot.support_bucket == "away_from_support"
    assert snapshot.level_data_date < snapshot.signal_date


@pytest.mark.parametrize(
    ("signal_close", "atr", "pressure_bucket", "support_bucket"),
    [
        (112.0, 2.0, "breakout", "away_from_support"),
        (108.0, 2.0, "near_resistance", "away_from_support"),
        (107.999, 2.0, "room_to_resistance", "away_from_support"),
        (92.0, 2.0, "room_to_resistance", "near_support"),
        (91.999, 2.0, "room_to_resistance", "near_support"),
        (88.0, 2.0, "room_to_resistance", "breakdown"),
    ],
)
def test_structure_buckets_use_locked_one_atr_boundaries(
    signal_close,
    atr,
    pressure_bucket,
    support_bucket,
):
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        calc_horizontal_structure,
    )

    snapshot = calc_horizontal_structure(
        price_frame(
            signal_close=signal_close,
            signal_high=max(signal_close, 109.0),
            signal_low=min(signal_close, 91.0),
        ),
        signal_date="2019-02-01",
        atr=atr,
    )

    assert snapshot.pressure_bucket == pressure_bucket
    assert snapshot.support_bucket == support_bucket


def test_structure_returns_no_data_for_short_history_or_invalid_atr():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        calc_horizontal_structure,
    )

    short = calc_horizontal_structure(
        price_frame(prior_count=19),
        signal_date="2019-02-01",
        atr=2.0,
    )
    invalid_atr = calc_horizontal_structure(
        price_frame(),
        signal_date="2019-02-01",
        atr=0.0,
    )

    assert short.eligible is False
    assert short.pressure_bucket == "no_data"
    assert invalid_atr.eligible is False
    assert invalid_atr.support_bucket == "no_data"


def test_structure_rejects_any_row_after_signal_date():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        calc_horizontal_structure,
    )

    frame = price_frame()
    future = frame.iloc[-1].copy()
    future["date"] = pd.Timestamp("2019-02-04")
    frame = pd.concat([frame, pd.DataFrame([future])], ignore_index=True)

    with pytest.raises(ValueError, match="after signal_date"):
        calc_horizontal_structure(
            frame,
            signal_date="2019-02-01",
            atr=2.0,
        )


def test_adapter_enriches_a_defensive_copy_without_changing_base_score():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        HorizontalStructureSignalAdapter,
    )

    source = FakeSource(price_frame())
    adapter = HorizontalStructureSignalAdapter(source)

    enriched = adapter.score("AAA", "2019-02-04")

    assert enriched is not source.base_score
    assert "pressure_bucket" not in source.base_score
    assert enriched["pressure_bucket"] == "near_resistance"
    assert enriched["structure_period"] == 20
    assert enriched["structure_signal_date"] == "2019-02-01"
    assert enriched["structure_level_data_date"] < enriched["structure_signal_date"]


def test_gate_requires_annual_return_and_win_rate_underperformance():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        evaluate_near_resistance_gate,
    )

    near = {
        year: stats(6, average_return=-0.02, win_rate=0.33)
        for year in (2019, 2020, 2021)
    }
    other = {
        year: stats(6, average_return=0.03, win_rate=0.67)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_near_resistance_gate(near, other)
    reversed_year = evaluate_near_resistance_gate(
        {**near, 2021: stats(6, average_return=0.04, win_rate=0.83)},
        other,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert reversed_year.passed is False
    assert any("2021" in reason for reason in reversed_year.reasons)


def test_report_keeps_support_descriptive_and_gates_only_mild_pressure():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        build_horizontal_structure_report,
    )

    report = build_horizontal_structure_report([
        closed_trade("near_resistance", support_bucket="near_support", pnl=-40.0),
        closed_trade("breakout", support_bucket="away_from_support", pnl=100.0),
        closed_trade(
            "near_resistance",
            support_bucket="near_support",
            pnl=200.0,
            trend_score=20,
        ),
    ])

    assert report.by_pressure["near_resistance"].realized_pnl == pytest.approx(160.0)
    assert report.by_support["near_support"].closed_trades == 2
    assert report.mild_by_year_pressure["2019:near_resistance"].realized_pnl == pytest.approx(-40.0)
    assert report.mild_by_year_pressure["2019:breakout"].realized_pnl == pytest.approx(100.0)


def test_report_rejects_validation_dates():
    from cross_signal_strategy.horizontal_structure_diagnostics import (
        build_horizontal_structure_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_horizontal_structure_report([
            closed_trade("near_resistance", year=2022),
        ])

