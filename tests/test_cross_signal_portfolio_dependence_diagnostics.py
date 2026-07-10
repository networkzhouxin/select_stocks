# -*- coding: utf-8 -*-
"""Tests for observation-only portfolio-dependence diagnostics."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def price_frame(prices, end="2019-02-01"):
    dates = pd.bdate_range(end=end, periods=len(prices))
    return pd.DataFrame({"date": dates, "close": prices})


class FakeFrameSource:
    def __init__(self, frames, signal_date="2019-02-01"):
        self.frames = frames
        self.signal_date = signal_date

    def load_signal_frame(self, code, current_date):
        return self.frames[code].copy(), self.signal_date


def closed_trade(bucket, year=2019, pnl=100.0, mae=-0.02, corr=0.9):
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
            "dependence_bucket": bucket,
            "dependence_max_correlation": corr,
            "test_mae": mae,
        },
    )


def stats(trades, average_return, average_mae):
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        DependenceTradeStats,
    )

    return DependenceTradeStats(
        closed_trades=trades,
        wins=trades if average_return > 0 else 0,
        losses=trades if average_return < 0 else 0,
        realized_pnl=average_return * trades * 100.0,
        gross_profit=max(0.0, average_return * trades * 100.0),
        gross_loss=max(0.0, -average_return * trades * 100.0),
        average_return=average_return,
        average_max_correlation=0.9,
        average_mae=average_mae,
    )


def test_return_correlation_uses_aligned_twenty_day_returns():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        calc_return_correlation,
    )

    daily_returns = [((index % 7) - 3) * 0.003 + 0.001 for index in range(24)]
    base = [100.0]
    inverse = [100.0]
    for daily_return in daily_returns:
        base.append(base[-1] * (1.0 + daily_return))
        inverse.append(inverse[-1] * (1.0 - daily_return))
    positive = [value * 2.0 for value in base]

    assert calc_return_correlation(
        price_frame(base),
        price_frame(positive),
        period=20,
    ) == pytest.approx(1.0)
    assert calc_return_correlation(
        price_frame(base),
        price_frame(inverse),
        period=20,
    ) == pytest.approx(-1.0)
    assert pd.isna(calc_return_correlation(
        price_frame(base[:20]),
        price_frame(positive[:20]),
        period=20,
    ))


def test_buy_snapshot_annotation_includes_held_and_earlier_planned_buys():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        annotate_planned_buy_dependence,
    )

    base = [100.0 + index for index in range(25)]
    frames = {
        "HELD": price_frame(base),
        "FIRST": price_frame([value * 2.0 for value in base]),
        "SECOND": price_frame([value * 3.0 for value in base]),
    }
    source = FakeFrameSource(frames)
    orders = [
        {"code": "FIRST", "reason": "buy_signal"},
        {"code": "SECOND", "reason": "buy_signal"},
    ]
    snapshots = {
        ("2019-02-04", "FIRST"): {"signal_date": "2019-02-01"},
        ("2019-02-04", "SECOND"): {"signal_date": "2019-02-01"},
    }

    annotate_planned_buy_dependence(
        orders=orders,
        entry_score_snapshots=snapshots,
        held_codes=["HELD"],
        source=source,
        current_date="2019-02-04",
    )

    first = snapshots[("2019-02-04", "FIRST")]
    second = snapshots[("2019-02-04", "SECOND")]
    assert first["dependence_reference_count"] == 1
    assert second["dependence_reference_count"] == 2
    assert first["dependence_bucket"] == "high"
    assert second["dependence_bucket"] == "high"
    assert first["dependence_data_date"] == "2019-02-01"


def test_buy_snapshot_annotation_rejects_data_after_signal_date():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        annotate_planned_buy_dependence,
    )

    frames = {
        "HELD": price_frame(range(100, 125), end="2019-02-04"),
        "FIRST": price_frame(range(200, 225), end="2019-02-04"),
    }
    snapshots = {
        ("2019-02-04", "FIRST"): {"signal_date": "2019-02-01"},
    }

    with pytest.raises(ValueError, match="after signal_date"):
        annotate_planned_buy_dependence(
            orders=[{"code": "FIRST", "reason": "buy_signal"}],
            entry_score_snapshots=snapshots,
            held_codes=["HELD"],
            source=FakeFrameSource(frames),
            current_date="2019-02-04",
        )


def test_dependence_report_splits_bucket_and_year_and_uses_mae():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        build_portfolio_dependence_report,
    )

    trades = [
        closed_trade("high", year=2019, pnl=100.0, mae=-0.08, corr=0.9),
        closed_trade("low", year=2020, pnl=-40.0, mae=-0.03, corr=0.2),
        closed_trade("no_reference", year=2021, pnl=60.0, mae=-0.01, corr=float("nan")),
    ]

    report = build_portfolio_dependence_report(
        trades,
        adverse_excursion_lookup=lambda trade: trade.entry_score["test_mae"],
    )

    assert report.by_bucket["high"].realized_pnl == pytest.approx(100.0)
    assert report.by_bucket["high"].average_mae == pytest.approx(-0.08)
    assert report.by_year_bucket["2020:low"].closed_trades == 1
    assert report.by_year_bucket["2021:no_reference"].closed_trades == 1


def test_dependence_gate_requires_annual_return_and_mae_underperformance():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        evaluate_dependence_gate,
    )

    high = {
        year: stats(6, average_return=-0.02, average_mae=-0.08)
        for year in (2019, 2020, 2021)
    }
    low = {
        year: stats(6, average_return=0.03, average_mae=-0.03)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_dependence_gate(high, low)
    failed = evaluate_dependence_gate(
        {**high, 2021: stats(6, average_return=0.04, average_mae=-0.02)},
        low,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_dependence_report_rejects_validation_dates():
    from cross_signal_strategy.portfolio_dependence_diagnostics import (
        build_portfolio_dependence_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_portfolio_dependence_report(
            [closed_trade("high", year=2022)],
            adverse_excursion_lookup=lambda trade: -0.05,
        )
