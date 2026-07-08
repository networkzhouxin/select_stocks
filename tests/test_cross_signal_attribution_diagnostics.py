# -*- coding: utf-8 -*-
"""Tests for ETF-level attribution diagnostics."""

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def closed_trade(code, buy_date, sell_date, reason, pnl):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason=reason,
        amount=100,
        buy_price=10.0,
        sell_price=11.0,
        pnl=pnl,
        return_pct=10.0,
    )


def test_etf_attribution_summarizes_trade_quality_by_code():
    from cross_signal_strategy.attribution_diagnostics import build_etf_attribution

    trades = [
        closed_trade("AAA", "2021-01-01", "2021-01-06", "atr_stop", 100.0),
        closed_trade("AAA", "2021-01-07", "2021-01-11", "signal_sell", -40.0),
        closed_trade("BBB", "2021-01-04", "2021-01-05", "signal_sell", 30.0),
    ]

    report = build_etf_attribution(
        trades,
        trade_dates=[
            "2021-01-01",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
        ],
    )

    aaa = report.by_code["AAA"]
    assert aaa.closed_trades == 2
    assert aaa.wins == 1
    assert aaa.losses == 1
    assert aaa.realized_pnl == pytest.approx(60.0)
    assert aaa.win_rate == pytest.approx(0.5)
    assert aaa.profit_loss_ratio == pytest.approx(100.0 / 40.0)
    assert aaa.atr_stop_rate == pytest.approx(0.5)
    assert aaa.signal_sell_rate == pytest.approx(0.5)
    assert aaa.average_holding_days == pytest.approx((3 + 2) / 2)

    assert report.total_realized_pnl == pytest.approx(90.0)
    assert report.by_code["BBB"].average_holding_days == pytest.approx(1.0)
