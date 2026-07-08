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
