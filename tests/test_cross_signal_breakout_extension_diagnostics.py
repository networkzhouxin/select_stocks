# -*- coding: utf-8 -*-
"""Tests for the fixed controlled-breakout anti-chase observation."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def price_frame(
    signal_close=112.0,
    signal_date="2019-02-01",
    add_future=False,
):
    dates = pd.bdate_range(end=signal_date, periods=21)
    closes = [90.0 + index for index in range(20)]
    rows = []
    for index, date in enumerate(dates[:-1]):
        close = closes[index]
        rows.append({
            "date": date,
            "open": close,
            "high": 110.0 if index == 19 else close + 1.0,
            "low": 88.0 if index == 0 else close - 1.0,
            "close": close,
        })
    rows.append({
        "date": dates[-1],
        "open": signal_close,
        "high": max(signal_close, 150.0),
        "low": min(signal_close, 80.0),
        "close": signal_close,
    })
    if add_future:
        future = dict(rows[-1])
        future["date"] = pd.Timestamp(signal_date) + pd.offsets.BDay(1)
        rows.append(future)
    return pd.DataFrame(rows)


class FakeSource:
    def __init__(self, frame, score=None, signal_date="2019-02-01"):
        self.frame = frame
        self.signal_date = signal_date
        self.base_score = score or {
            "code": "AAA",
            "signal_date": signal_date,
            "close": 112.0,
            "ma20": 105.0,
            "rsi6": 60.0,
            "atr": 2.0,
            "buy_allowed": True,
        }

    def score(self, code, current_date, return_reason=False):
        result = self.base_score
        return (result, None) if return_reason else result

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), self.signal_date


def closed_trade(label, year=2019, pnl=100.0, return_pct=None):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    trade_return = float(return_pct if return_pct is not None else pnl / 10.0)
    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date=f"{year}-02-01",
        sell_date=f"{year}-02-08",
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 * (1.0 + trade_return / 100.0),
        pnl=float(pnl),
        return_pct=trade_return,
        entry_score={
            "breakout_extension_label": label,
            "breakout_rsi6": 70.0 if label == "extended_breakout" else 60.0,
            "breakout_ma20_distance": 0.11 if label == "extended_breakout" else 0.03,
            "breakout_return_5": 0.08,
            "breakout_return_10": 0.12,
            "breakout_return_20": 0.18,
            "breakout_rise_from_low": 0.25,
        },
    )


@pytest.mark.parametrize(
    ("pressure", "rsi6", "close", "ma20", "expected"),
    [
        ("near_resistance", 90.0, 120.0, 100.0, "no_breakout"),
        ("breakout", 74.9, 109.9, 100.0, "controlled_breakout"),
        ("breakout", 75.0, 100.0, 100.0, "extended_breakout"),
        ("breakout", 40.0, 110.0, 100.0, "extended_breakout"),
        ("breakout", 80.0, 110.0, None, "extended_breakout"),
        ("breakout", 40.0, 105.0, None, "no_data"),
    ],
)
def test_classification_uses_only_locked_breakout_extension_boundaries(
    pressure,
    rsi6,
    close,
    ma20,
    expected,
):
    from cross_signal_strategy.breakout_extension_diagnostics import (
        classify_breakout_extension,
    )

    assert classify_breakout_extension(pressure, rsi6, close, ma20) == expected


def test_adapter_uses_t2_resistance_and_t1_trailing_diagnostics_on_copy():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        BreakoutExtensionSignalAdapter,
    )

    source = FakeSource(price_frame())
    adapter = BreakoutExtensionSignalAdapter(source)

    enriched = adapter.score("AAA", "2019-02-04")

    assert enriched is not source.base_score
    assert "breakout_extension_label" not in source.base_score
    assert enriched["breakout_extension_label"] == "controlled_breakout"
    assert enriched["breakout_extension_blocked"] is False
    assert enriched["breakout_level_data_date"] < enriched["breakout_signal_date"]
    assert enriched["breakout_return_5"] == pytest.approx(112.0 / 105.0 - 1.0)
    assert enriched["breakout_return_10"] == pytest.approx(112.0 / 100.0 - 1.0)
    assert enriched["breakout_return_20"] == pytest.approx(112.0 / 90.0 - 1.0)
    assert enriched["breakout_rise_from_low"] == pytest.approx(112.0 / 88.0 - 1.0)
    assert enriched["buy_allowed"] is True


def test_adapter_rejects_any_row_after_signal_date():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        BreakoutExtensionSignalAdapter,
    )

    adapter = BreakoutExtensionSignalAdapter(FakeSource(price_frame(add_future=True)))

    with pytest.raises(ValueError, match="after signal_date"):
        adapter.score("AAA", "2019-02-04")


def test_adapter_returns_no_data_without_exact_twenty_t2_bars():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        BreakoutExtensionSignalAdapter,
    )

    source = FakeSource(price_frame().iloc[1:].copy())
    enriched = BreakoutExtensionSignalAdapter(source).score("AAA", "2019-02-04")

    assert enriched["breakout_extension_label"] == "no_data"
    assert enriched["breakout_return_20"] is None


def test_observation_gate_requires_sample_and_annual_underperformance():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        BreakoutExtensionStats,
        evaluate_observation_gate,
    )

    controlled = {
        year: BreakoutExtensionStats(
            closed_trades=2,
            wins=2,
            average_return=0.05,
        )
        for year in (2019, 2020, 2021)
    }
    extended = {
        year: BreakoutExtensionStats(
            closed_trades=2,
            wins=0,
            average_return=-0.02,
        )
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_observation_gate(controlled, extended)
    reversed_year = evaluate_observation_gate(
        controlled,
        {
            **extended,
            2021: BreakoutExtensionStats(
                closed_trades=2,
                wins=2,
                average_return=0.06,
            ),
        },
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert reversed_year.passed is False
    assert any("2021" in reason for reason in reversed_year.reasons)


def test_observation_gate_rejects_insufficient_group_samples():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        BreakoutExtensionStats,
        evaluate_observation_gate,
    )

    controlled = {
        year: BreakoutExtensionStats(closed_trades=2, wins=2, average_return=0.05)
        for year in (2019, 2020, 2021)
    }
    extended = {
        2019: BreakoutExtensionStats(closed_trades=2, wins=0, average_return=-0.02),
        2020: BreakoutExtensionStats(closed_trades=2, wins=0, average_return=-0.02),
        2021: BreakoutExtensionStats(closed_trades=1, wins=0, average_return=-0.02),
    }

    decision = evaluate_observation_gate(controlled, extended)

    assert decision.passed is False
    assert any("fewer than 6" in reason for reason in decision.reasons)
    assert any("2021" in reason and "fewer than 2" in reason for reason in decision.reasons)


def test_report_groups_breakouts_and_rejects_validation_dates():
    from cross_signal_strategy.breakout_extension_diagnostics import (
        build_breakout_extension_report,
    )

    report = build_breakout_extension_report([
        closed_trade("controlled_breakout", pnl=100.0),
        closed_trade("extended_breakout", pnl=-40.0),
        closed_trade("no_breakout", pnl=20.0),
    ])

    assert report.by_label["controlled_breakout"].closed_trades == 1
    assert report.by_label["extended_breakout"].realized_pnl == pytest.approx(-40.0)
    assert report.by_year_label["2019:controlled_breakout"].average_return == pytest.approx(0.10)

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_breakout_extension_report([
            closed_trade("extended_breakout", year=2022),
        ])
