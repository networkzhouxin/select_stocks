# -*- coding: utf-8 -*-
"""Tests for T-1-safe 09:35 gap execution diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def trade(
    buy_date="2019-01-02",
    sell_date="2019-01-04",
    pnl=100.0,
    signal_date="2019-01-01",
    previous_close=9.0,
    atr=1.0,
    trend_score=10,
    buy_price=10.1,
):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    sell_price = buy_price + float(pnl) / 100.0
    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason="signal_sell",
        amount=100,
        buy_price=float(buy_price),
        sell_price=float(sell_price),
        pnl=float(pnl),
        return_pct=(sell_price / buy_price - 1.0) * 100.0,
        entry_score={
            "signal_date": signal_date,
            "close": previous_close,
            "atr": atr,
            "trend_score": trend_score,
        },
    )


def stats(trades, average_return, win_rate, profit_loss_ratio=1.0):
    from cross_signal_strategy.research.gap_execution_diagnostics import GapTradeStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    gross_loss = 100.0 if losses else 0.0
    gross_profit = gross_loss * profit_loss_ratio if gross_loss else 100.0
    return GapTradeStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=gross_profit - gross_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        average_return=average_return,
        average_gap_atr=1.2,
        average_mfe=0.08,
        average_mae=-0.05,
    )


def test_gap_atr_bucket_uses_locked_boundaries():
    from cross_signal_strategy.research.gap_execution_diagnostics import gap_atr_bucket

    assert gap_atr_bucket(-0.1) == "non_positive"
    assert gap_atr_bucket(0.0) == "non_positive"
    assert gap_atr_bucket(0.5) == "up_to_half"
    assert gap_atr_bucket(1.0) == "half_to_one"
    assert gap_atr_bucket(1.0001) == "above_one"
    assert gap_atr_bucket(float("nan")) == "unknown"


def test_gap_report_uses_raw_0935_price_not_slippage_adjusted_fill_price():
    from cross_signal_strategy.research.gap_execution_diagnostics import (
        build_gap_execution_report,
    )

    prices = {
        ("AAA", "2019-01-02"): 10.0,
        ("AAA", "2019-01-03"): 9.5,
        ("AAA", "2019-01-04"): 11.0,
    }
    report = build_gap_execution_report(
        trades=[trade()],
        trade_dates=["2019-01-02", "2019-01-03", "2019-01-04"],
        entry_price_lookup=lambda code, date: prices[(code, date)],
        close_price_lookup=lambda code, date: prices[(code, date)],
    )

    bucket = report.by_bucket["half_to_one"]
    assert bucket.closed_trades == 1
    assert bucket.average_gap_atr == pytest.approx(1.0)
    assert bucket.average_mfe == pytest.approx(0.10)
    assert bucket.average_mae == pytest.approx(-0.05)
    assert "above_one" not in report.by_bucket


def test_gap_report_splits_entry_year_and_trend_without_changing_trade_path():
    from cross_signal_strategy.research.gap_execution_diagnostics import (
        build_gap_execution_report,
    )

    trades = [
        trade(pnl=100.0, trend_score=20),
        trade(
            buy_date="2020-01-02",
            sell_date="2020-01-03",
            signal_date="2019-12-31",
            previous_close=10.0,
            atr=1.0,
            pnl=-40.0,
            trend_score=5,
        ),
    ]
    trade_dates = [
        "2019-01-02",
        "2019-01-03",
        "2019-01-04",
        "2020-01-02",
        "2020-01-03",
    ]
    entry_prices = {
        ("AAA", "2019-01-02"): 10.2,
        ("AAA", "2020-01-02"): 10.2,
    }

    report = build_gap_execution_report(
        trades=trades,
        trade_dates=trade_dates,
        entry_price_lookup=lambda code, date: entry_prices[(code, date)],
        close_price_lookup=lambda code, date: entry_prices.get((code, date), 10.0),
    )

    assert report.by_bucket["above_one"].closed_trades == 1
    assert report.by_bucket["up_to_half"].closed_trades == 1
    assert report.by_year_bucket["2019:above_one"].realized_pnl == pytest.approx(100.0)
    assert report.by_year_bucket["2020:up_to_half"].realized_pnl == pytest.approx(-40.0)
    assert report.by_trend_bucket["strong_up:above_one"].closed_trades == 1
    assert report.by_trend_bucket["mild_up:up_to_half"].closed_trades == 1


def test_gap_filter_gate_requires_consistent_annual_underperformance():
    from cross_signal_strategy.research.gap_execution_diagnostics import (
        evaluate_gap_filter_gate,
    )

    above = {
        year: stats(4, average_return=-0.02, win_rate=0.25, profit_loss_ratio=0.5)
        for year in (2019, 2020, 2021)
    }
    rest = {
        year: stats(20, average_return=0.04, win_rate=0.60, profit_loss_ratio=2.0)
        for year in (2019, 2020, 2021)
    }

    passed = evaluate_gap_filter_gate(above, rest)
    failed = evaluate_gap_filter_gate(
        {**above, 2021: stats(4, average_return=0.06, win_rate=0.75, profit_loss_ratio=3.0)},
        rest,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_gap_report_rejects_same_day_signal_and_non_training_dates():
    from cross_signal_strategy.research.gap_execution_diagnostics import (
        build_gap_execution_report,
    )

    with pytest.raises(ValueError, match="signal_date must precede buy_date"):
        build_gap_execution_report(
            trades=[trade(signal_date="2019-01-02")],
            trade_dates=["2019-01-02", "2019-01-03", "2019-01-04"],
            entry_price_lookup=lambda code, date: 10.0,
            close_price_lookup=lambda code, date: 10.0,
        )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_gap_execution_report(
            trades=[
                trade(
                    buy_date="2022-01-04",
                    sell_date="2022-01-05",
                    signal_date="2021-12-31",
                )
            ],
            trade_dates=["2022-01-04", "2022-01-05"],
            entry_price_lookup=lambda code, date: 10.0,
            close_price_lookup=lambda code, date: 10.0,
        )
