# -*- coding: utf-8 -*-
"""Tests for post-sell diagnostics in the cross-signal training window."""

import pathlib
import sys
import types

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeLoader:
    def __init__(self, frames):
        self.frames = frames

    def load_daily_frame(self, code, trade_date):
        return self.frames[str(code).split(".")[0]].copy()


def test_post_sell_returns_use_training_trade_day_offsets_and_available_future_rows():
    from cross_signal_strategy.sell_diagnostics import post_sell_returns
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    loader = FakeLoader({
        "AAA": pd.DataFrame({
            "date": ["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-06"],
            "close": [10.0, 11.0, 12.0, 9.0],
        })
    })
    trade = ClosedTradeDiagnostic(
        code="AAA",
        buy_date="2021-01-01",
        sell_date="2021-01-04",
        sell_reason="signal_sell",
        amount=100,
        buy_price=9.0,
        sell_price=11.0,
        pnl=195.0,
        return_pct=22.2,
    )

    result = post_sell_returns(trade, loader, horizons=(1, 2, 5))

    assert result.code == "AAA"
    assert result.sell_reason == "signal_sell"
    assert result.forward_returns[1] == pytest.approx(12.0 / 11.0 - 1.0)
    assert result.forward_returns[2] == pytest.approx(9.0 / 11.0 - 1.0)
    assert result.forward_returns[5] is None


def test_summarize_post_sell_returns_groups_by_sell_reason():
    from cross_signal_strategy.sell_diagnostics import PostSellDiagnostic, summarize_post_sell_returns

    diagnostics = [
        PostSellDiagnostic("AAA", "2021-01-04", "signal_sell", 11.0, {3: 0.10, 5: -0.02}),
        PostSellDiagnostic("BBB", "2021-01-05", "signal_sell", 20.0, {3: 0.04, 5: None}),
        PostSellDiagnostic("CCC", "2021-01-06", "atr_stop", 8.0, {3: -0.03, 5: -0.05}),
    ]

    summary = summarize_post_sell_returns(diagnostics, horizons=(3, 5))

    assert summary["signal_sell"][3].count == 2
    assert summary["signal_sell"][3].mean_return == pytest.approx(0.07)
    assert summary["signal_sell"][3].positive_rate == pytest.approx(1.0)
    assert summary["signal_sell"][5].count == 1
    assert summary["atr_stop"][3].count == 1
    assert summary["atr_stop"][3].mean_return == pytest.approx(-0.03)


def test_sell_fly_diagnostic_uses_exit_score_and_forward_return():
    from cross_signal_strategy.sell_diagnostics import sell_fly_diagnostic
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    loader = FakeLoader({
        "AAA": pd.DataFrame({
            "date": ["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"],
            "close": [9.0, 10.0, 10.2, 10.5, 10.8],
        })
    })
    trade = ClosedTradeDiagnostic(
        code="AAA",
        buy_date="2021-01-01",
        sell_date="2021-01-04",
        sell_reason="signal_sell",
        amount=100,
        buy_price=9.0,
        sell_price=10.0,
        pnl=95.0,
        return_pct=11.1,
        exit_score={
            "sell_score": 34,
            "sell_reversal_score": 24,
            "sell_risk_score": 10,
            "close_below_ma20": True,
            "close_below_boll_mid": False,
        },
    )

    result = sell_fly_diagnostic(trade, loader, horizon=3, min_forward_return=0.03)

    assert result.code == "AAA"
    assert result.is_sell_fly is True
    assert result.forward_return == pytest.approx(0.08)
    assert result.missed_pnl == pytest.approx(80.0)
    assert result.exit_features["close_below_ma20"] is True
    assert result.exit_features["sell_reversal_score"] == 24


def test_summarize_sell_fly_by_feature_counts_flagged_and_unflagged_cases():
    from cross_signal_strategy.sell_diagnostics import SellFlyDiagnostic, summarize_sell_fly_by_feature

    diagnostics = [
        SellFlyDiagnostic("AAA", "2021-01-04", "signal_sell", 3, 0.08, 80.0, True, {"close_below_ma20": True}),
        SellFlyDiagnostic("BBB", "2021-01-05", "signal_sell", 3, -0.02, -20.0, False, {"close_below_ma20": True}),
        SellFlyDiagnostic("CCC", "2021-01-06", "signal_sell", 3, 0.05, 50.0, True, {"close_below_ma20": False}),
    ]

    summary = summarize_sell_fly_by_feature(diagnostics, "close_below_ma20")

    assert summary[True].count == 2
    assert summary[True].sell_fly_count == 1
    assert summary[True].sell_fly_rate == pytest.approx(0.5)
    assert summary[True].average_forward_return == pytest.approx(0.03)
    assert summary[False].count == 1
    assert summary[False].sell_fly_rate == pytest.approx(1.0)
