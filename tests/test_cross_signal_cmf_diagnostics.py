# -*- coding: utf-8 -*-
"""Tests for cross-signal CMF(20) training diagnostics."""

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
    return pd.DataFrame(rows, columns=["date", "high", "low", "close", "volume"])


def closed_trade(cmf, trend_score, pnl, buy_date="2019-01-02", reason="signal_sell"):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date=buy_date,
        sell_date="2019-01-10",
        sell_reason=reason,
        amount=100,
        buy_price=10.0,
        sell_price=11.0,
        pnl=float(pnl),
        return_pct=10.0,
        entry_score={"cmf20": cmf, "trend_score": trend_score},
    )


def test_calc_cmf_uses_close_location_volume_and_handles_flat_range():
    from cross_signal_strategy.cmf_diagnostics import calc_cmf

    data = frame(
        [
            ("2019-01-01", 10.0, 0.0, 10.0, 100.0),
            ("2019-01-02", 10.0, 0.0, 0.0, 100.0),
            ("2019-01-03", 5.0, 5.0, 5.0, 100.0),
        ]
    )

    cmf = calc_cmf(data, period=2)

    assert pd.isna(cmf.iloc[0])
    assert cmf.iloc[1] == pytest.approx(0.0)
    assert cmf.iloc[2] == pytest.approx(-0.5)


def test_calc_cmf_returns_nan_when_rolling_volume_is_zero():
    from cross_signal_strategy.cmf_diagnostics import calc_cmf

    data = frame(
        [
            ("2019-01-01", 10.0, 9.0, 9.5, 0.0),
            ("2019-01-02", 10.0, 9.0, 9.8, 0.0),
        ]
    )

    assert pd.isna(calc_cmf(data, period=2).iloc[-1])


def test_cmf_signal_adapter_attaches_t_minus_one_value_and_returns_copy():
    from cross_signal_strategy.cmf_diagnostics import CmfSignalAdapter

    data = frame(
        [
            ("2019-01-01", 10.0, 9.0, 9.8, 100.0),
            ("2019-01-02", 10.5, 9.5, 10.4, 120.0),
            ("2019-01-03", 10.8, 10.0, 10.7, 130.0),
        ]
    )
    adapter = CmfSignalAdapter(FakeSignalAdapter(data), period=2)

    first, reason = adapter.score("AAA", "2019-01-04", return_reason=True)
    first["cmf20"] = 999.0
    second = adapter.score("AAA", "2019-01-04")

    assert reason is None
    assert second["cmf_period"] == 2
    assert second["cmf_data_date"] == "2019-01-03"
    assert second["cmf_data_date"] == second["signal_date"]
    assert second["cmf20"] != 999.0


def test_cmf_signal_adapter_rejects_data_after_signal_date():
    from cross_signal_strategy.cmf_diagnostics import CmfSignalAdapter

    data = frame(
        [
            ("2019-01-02", 10.0, 9.0, 9.8, 100.0),
            ("2019-01-03", 10.5, 9.5, 10.4, 120.0),
        ]
    )
    adapter = CmfSignalAdapter(
        FakeSignalAdapter(data, signal_date="2019-01-02"),
        period=2,
    )

    with pytest.raises(ValueError, match="after signal_date"):
        adapter.score("AAA", "2019-01-04")


def test_cmf_attribution_groups_sign_trend_and_entry_year():
    from cross_signal_strategy.cmf_diagnostics import build_cmf_attribution

    trades = [
        closed_trade(0.2, 20, 100.0, buy_date="2019-01-02", reason="atr_stop"),
        closed_trade(-0.1, 10, -40.0, buy_date="2019-02-01"),
        closed_trade(0.1, 10, 60.0, buy_date="2020-01-02"),
        closed_trade(-0.2, 20, 20.0, buy_date="2020-02-03"),
    ]

    report = build_cmf_attribution(trades)

    assert report.by_sign["positive"].closed_trades == 2
    assert report.by_sign["positive"].realized_pnl == pytest.approx(160.0)
    assert report.by_sign["positive"].win_rate == pytest.approx(1.0)
    assert report.by_sign["non_positive"].realized_pnl == pytest.approx(-20.0)
    assert report.by_trend_sign["mild_up:non_positive"].realized_pnl == pytest.approx(-40.0)
    assert report.by_trend_sign["strong_up:non_positive"].realized_pnl == pytest.approx(20.0)
    assert report.by_year_sign["2019:positive"].closed_trades == 1
    assert report.by_year_sign["2020:positive"].closed_trades == 1
