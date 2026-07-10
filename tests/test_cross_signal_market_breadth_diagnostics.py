# -*- coding: utf-8 -*-
"""Tests for observation-only ETF-pool market-breadth diagnostics."""

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


def closed_trade(bucket, year=2019, pnl=100.0, trend_score=10, breadth=0.4):
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
            "breadth_bucket": bucket,
            "market_breadth": breadth,
            "trend_score": trend_score,
        },
    )


def stats(trades, average_return, win_rate):
    from cross_signal_strategy.market_breadth_diagnostics import BreadthTradeStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    return BreadthTradeStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=average_return * trades * 100.0,
        gross_profit=max(0.0, average_return * trades * 120.0),
        gross_loss=max(0.0, -average_return * trades * 20.0),
        average_return=average_return,
        average_breadth=0.4,
    )


def test_above_ma20_uses_latest_t1_close_and_requires_full_history():
    from cross_signal_strategy.market_breadth_diagnostics import calc_above_ma

    assert calc_above_ma(price_frame(range(1, 21)), period=20) is True
    assert calc_above_ma(price_frame(range(20, 0, -1)), period=20) is False
    assert calc_above_ma(price_frame(range(1, 20)), period=20) is None


def test_pool_breadth_excludes_insufficient_history_from_denominator():
    from cross_signal_strategy.market_breadth_diagnostics import calculate_pool_breadth

    source = FakeFrameSource({
        "UP": price_frame(range(1, 21)),
        "DOWN": price_frame(range(20, 0, -1)),
        "NEW": price_frame(range(1, 10)),
    })

    snapshot = calculate_pool_breadth(
        source=source,
        pool_codes=["UP", "DOWN", "NEW"],
        current_date="2019-02-04",
        signal_date="2019-02-01",
    )

    assert snapshot.eligible_count == 2
    assert snapshot.above_count == 1
    assert snapshot.breadth == pytest.approx(0.5)
    assert snapshot.bucket == "majority"


def test_buy_snapshot_annotation_shares_one_t1_breadth_snapshot():
    from cross_signal_strategy.market_breadth_diagnostics import (
        annotate_planned_buy_breadth,
    )

    source = FakeFrameSource({
        "UP": price_frame(range(1, 21)),
        "DOWN": price_frame(range(20, 0, -1)),
    })
    snapshots = {
        ("2019-02-04", "UP"): {"signal_date": "2019-02-01"},
        ("2019-02-04", "DOWN"): {"signal_date": "2019-02-01"},
    }

    annotate_planned_buy_breadth(
        orders=[
            {"code": "UP", "reason": "buy_signal"},
            {"code": "DOWN", "reason": "buy_signal"},
        ],
        entry_score_snapshots=snapshots,
        source=source,
        pool_codes=["UP", "DOWN"],
        current_date="2019-02-04",
    )

    assert snapshots[("2019-02-04", "UP")]["market_breadth"] == pytest.approx(0.5)
    assert snapshots[("2019-02-04", "DOWN")]["market_breadth"] == pytest.approx(0.5)
    assert snapshots[("2019-02-04", "UP")]["breadth_data_date"] == "2019-02-01"


def test_pool_breadth_rejects_data_after_signal_date():
    from cross_signal_strategy.market_breadth_diagnostics import calculate_pool_breadth

    source = FakeFrameSource(
        {"UP": price_frame(range(1, 22), end="2019-02-04")},
        signal_date="2019-02-01",
    )

    with pytest.raises(ValueError, match="after signal_date"):
        calculate_pool_breadth(
            source=source,
            pool_codes=["UP"],
            current_date="2019-02-04",
            signal_date="2019-02-01",
        )


def test_breadth_report_gates_only_mild_trend_entries_by_year():
    from cross_signal_strategy.market_breadth_diagnostics import (
        build_market_breadth_report,
    )

    report = build_market_breadth_report([
        closed_trade("below_majority", year=2019, pnl=-40.0, trend_score=10),
        closed_trade("majority", year=2019, pnl=100.0, trend_score=10, breadth=0.6),
        closed_trade("below_majority", year=2020, pnl=200.0, trend_score=20),
    ])

    assert report.by_bucket["below_majority"].realized_pnl == pytest.approx(160.0)
    assert report.mild_by_year_bucket["2019:below_majority"].realized_pnl == pytest.approx(-40.0)
    assert "2020:below_majority" not in report.mild_by_year_bucket


def test_breadth_gate_requires_consistent_annual_return_and_win_underperformance():
    from cross_signal_strategy.market_breadth_diagnostics import evaluate_breadth_gate

    below = {
        year: stats(6, average_return=-0.02, win_rate=0.33)
        for year in (2019, 2020, 2021)
    }
    majority = {
        year: stats(6, average_return=0.03, win_rate=0.67)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_breadth_gate(below, majority)
    failed = evaluate_breadth_gate(
        {**below, 2021: stats(6, average_return=0.04, win_rate=0.83)},
        majority,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_breadth_report_rejects_validation_dates():
    from cross_signal_strategy.market_breadth_diagnostics import (
        build_market_breadth_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_market_breadth_report([
            closed_trade("below_majority", year=2022),
        ])

