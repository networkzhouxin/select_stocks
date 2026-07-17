# -*- coding: utf-8 -*-
"""Tests for cross-signal training-window stability diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def day(date, total_value, cash, positions=None, orders=None):
    from cross_signal_strategy.local.local_backtester import DayResult

    return DayResult(
        date=date,
        previous_date=None,
        orders=orders or [],
        cash=float(cash),
        positions=positions or {},
        marks={code: 10.0 for code in (positions or {})},
        total_value=float(total_value),
    )


def trade(code, buy_date, sell_date, reason, pnl, trend_score=0, atr=0.2, close=10.0):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    sell_price = 10.0 + pnl / 100.0
    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason=reason,
        amount=100,
        buy_price=10.0,
        sell_price=sell_price,
        pnl=float(pnl),
        return_pct=(sell_price / 10.0 - 1.0) * 100.0,
        entry_score={
            "trend_score": trend_score,
            "atr": atr,
            "close": close,
        },
    )


def test_stability_report_splits_annual_performance_and_anchors_year_start_value():
    from cross_signal_strategy.local.local_backtester import OrderResult, Position
    from cross_signal_strategy.research.training_stability import build_training_stability_report

    results = [
        day(
            "2019-12-30",
            10000.0,
            9000.0,
            {"AAA": Position("AAA", 100, 10.0)},
            [OrderResult("AAA", 100, 10.0, 5.0, "2019-12-30 09:35", True, "buy_signal")],
        ),
        day(
            "2019-12-31",
            11000.0,
            11000.0,
            orders=[OrderResult("AAA", -100, 20.0, 5.0, "2019-12-31 09:35", True, "signal_sell")],
        ),
        day(
            "2020-01-02",
            10500.0,
            9500.0,
            {"BBB": Position("BBB", 100, 10.0)},
            [OrderResult("BBB", 100, 10.0, 5.0, "2020-01-02 09:35", True, "buy_signal")],
        ),
        day(
            "2020-01-03",
            12100.0,
            12100.0,
            orders=[OrderResult("BBB", -100, 26.0, 5.0, "2020-01-03 09:35", True, "atr_stop")],
        ),
    ]
    trades = [
        trade("AAA", "2019-12-30", "2019-12-31", "signal_sell", 990.0, trend_score=20),
        trade("BBB", "2020-01-02", "2020-01-03", "atr_stop", 1590.0, trend_score=-5),
    ]

    report = build_training_stability_report(results, trades, initial_cash=10000.0)

    assert report.annual[2019].start_value == pytest.approx(10000.0)
    assert report.annual[2019].total_return == pytest.approx(0.10)
    assert report.annual[2020].start_value == pytest.approx(11000.0)
    assert report.annual[2020].total_return == pytest.approx(0.10)
    assert report.annual[2020].max_drawdown == pytest.approx((11000.0 - 10500.0) / 11000.0)
    assert report.annual[2019].trade_stats.realized_pnl == pytest.approx(990.0)
    assert report.annual[2020].trade_stats.realized_pnl == pytest.approx(1590.0)


def test_stability_report_summarizes_concentration_exits_holding_and_entry_regimes():
    from cross_signal_strategy.research.training_stability import build_training_stability_report

    results = [
        day("2019-01-02", 10000.0, 10000.0),
        day("2019-01-03", 10100.0, 10100.0),
        day("2019-01-04", 10200.0, 10200.0),
        day("2019-01-07", 10300.0, 10300.0),
        day("2019-01-08", 10400.0, 10400.0),
        day("2019-01-09", 10500.0, 10500.0),
    ]
    trades = [
        trade("AAA", "2019-01-02", "2019-01-04", "signal_sell", 100.0, trend_score=20, atr=0.2),
        trade("AAA", "2019-01-03", "2019-01-07", "atr_stop", 60.0, trend_score=5, atr=0.4),
        trade("BBB", "2019-01-04", "2019-01-08", "signal_sell", 40.0, trend_score=0, atr=0.2),
        trade("CCC", "2019-01-07", "2019-01-09", "atr_stop", -50.0, trend_score=-5, atr=0.4),
    ]

    report = build_training_stability_report(results, trades, initial_cash=10000.0)

    concentration = report.concentration
    assert concentration.gross_profit == pytest.approx(200.0)
    assert concentration.largest_trade_profit_share == pytest.approx(0.5)
    assert concentration.top_three_trade_profit_share == pytest.approx(1.0)
    assert concentration.largest_code_profit_share == pytest.approx(0.8)

    assert report.exit_reasons["signal_sell"].closed_trades == 2
    assert report.exit_reasons["signal_sell"].realized_pnl == pytest.approx(140.0)
    assert report.exit_reasons["atr_stop"].win_rate == pytest.approx(0.5)
    assert report.holding_periods.average_days == pytest.approx(2.0)
    assert report.holding_periods.buckets == {
        "0-4": 4,
        "5-9": 0,
        "10-19": 0,
        "20+": 0,
    }

    assert report.entry_regimes["trend:strong_up"].realized_pnl == pytest.approx(100.0)
    assert report.entry_regimes["trend:mild_up"].realized_pnl == pytest.approx(60.0)
    assert report.entry_regimes["trend:sideways"].realized_pnl == pytest.approx(40.0)
    assert report.entry_regimes["trend:down"].realized_pnl == pytest.approx(-50.0)
    assert report.entry_regimes["volatility:normal"].closed_trades == 2
    assert report.entry_regimes["volatility:high"].closed_trades == 2
    assert report.volatility_cutoff == pytest.approx(0.03)


def test_volatility_regime_uses_training_sample_median_instead_of_fixed_market_threshold():
    from cross_signal_strategy.research.training_stability import build_training_stability_report

    results = [
        day("2019-01-02", 10000.0, 10000.0),
        day("2019-01-03", 10100.0, 10100.0),
    ]
    trades = [
        trade("AAA", "2019-01-02", "2019-01-03", "signal_sell", 50.0, atr=0.10),
        trade("BBB", "2019-01-02", "2019-01-03", "signal_sell", 60.0, atr=0.20),
    ]

    report = build_training_stability_report(results, trades, initial_cash=10000.0)

    assert report.volatility_cutoff == pytest.approx(0.015)
    assert report.entry_regimes["volatility:normal"].closed_trades == 1
    assert report.entry_regimes["volatility:high"].closed_trades == 1


def test_stability_report_compares_complete_doubled_friction_replay():
    from cross_signal_strategy.research.baseline_report import build_baseline_report
    from cross_signal_strategy.research.training_stability import build_training_stability_report

    baseline_results = [
        day("2019-01-02", 10000.0, 10000.0),
        day("2019-01-03", 11000.0, 11000.0),
    ]
    stressed_results = [
        day("2019-01-02", 9950.0, 9950.0),
        day("2019-01-03", 10800.0, 10800.0),
    ]
    stressed = build_baseline_report(stressed_results, initial_cash=10000.0)

    report = build_training_stability_report(
        baseline_results,
        [],
        initial_cash=10000.0,
        stressed_baseline=stressed,
    )

    assert report.friction_stress is not None
    assert report.friction_stress.baseline_return == pytest.approx(0.10)
    assert report.friction_stress.stressed_return == pytest.approx(0.08)
    assert report.friction_stress.return_delta == pytest.approx(-0.02)
    assert report.friction_stress.end_value_delta == pytest.approx(-200.0)


def test_stability_report_rejects_non_training_dates():
    from cross_signal_strategy.research.training_stability import build_training_stability_report

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_training_stability_report(
            [day("2022-01-04", 10000.0, 10000.0)],
            [],
            initial_cash=10000.0,
        )
