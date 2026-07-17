# -*- coding: utf-8 -*-
"""Tests for cross-signal training baseline diagnostics."""

import pathlib
import sys
import math

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_baseline_report_summarizes_returns_drawdown_and_closed_trades():
    from cross_signal_strategy.research.baseline_report import build_baseline_report
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult, Position

    results = [
        DayResult("2020-01-02", None, [
            OrderResult("AAA", 100, 10.0, 5.0, "2020-01-02 09:35", True),
        ], cash=8995.0, positions={"AAA": Position("AAA", 100, 10.0)}, marks={"AAA": 10.0}, total_value=9995.0),
        DayResult("2020-01-03", "2020-01-02", [], cash=8995.0, positions={"AAA": Position("AAA", 100, 10.0)}, marks={"AAA": 11.0}, total_value=10095.0),
        DayResult("2020-01-06", "2020-01-03", [
            OrderResult("AAA", -100, 12.0, 5.0, "2020-01-06 09:35", True),
            OrderResult("BBB", 100, 20.0, 5.0, "2020-01-06 09:35", True),
        ], cash=8185.0, positions={"BBB": Position("BBB", 100, 20.0)}, marks={"BBB": 20.0}, total_value=10185.0),
        DayResult("2020-01-07", "2020-01-06", [
            OrderResult("BBB", -100, 18.0, 5.0, "2020-01-07 09:35", True),
        ], cash=9980.0, positions={}, marks={}, total_value=9980.0),
    ]

    report = build_baseline_report(results, initial_cash=10000.0, periods_per_year=244)

    assert report.start_date == "2020-01-02"
    assert report.end_date == "2020-01-07"
    assert report.trading_days == 4
    assert report.total_return == pytest.approx(-0.002)
    assert report.max_drawdown == pytest.approx((10185.0 - 9980.0) / 10185.0)
    assert report.buy_count == 2
    assert report.sell_count == 2
    assert report.closed_trade_count == 2
    assert report.win_rate == pytest.approx(0.5)
    assert report.profit_loss_ratio == pytest.approx(190.0 / 210.0)
    daily_returns = [
        9995.0 / 10000.0 - 1.0,
        10095.0 / 9995.0 - 1.0,
        10185.0 / 10095.0 - 1.0,
        9980.0 / 10185.0 - 1.0,
    ]
    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((item - mean_return) ** 2 for item in daily_returns) / len(daily_returns)
    downside = [min(item, 0.0) for item in daily_returns]
    downside_variance = sum(item ** 2 for item in downside) / len(downside)
    assert report.daily_win_rate == pytest.approx(0.5)
    assert report.annualized_volatility == pytest.approx(math.sqrt(variance) * math.sqrt(244))
    assert report.sharpe_ratio == pytest.approx(mean_return / math.sqrt(variance) * math.sqrt(244))
    assert report.sortino_ratio == pytest.approx(mean_return / math.sqrt(downside_variance) * math.sqrt(244))
    assert report.average_exposure == pytest.approx((1000.0 + 1100.0 + 2000.0) / (9995.0 + 10095.0 + 10185.0 + 9980.0))
    assert report.position_count_days == {0: 1, 1: 3}
    assert report.full_position_days == 0
    assert report.empty_days == 1
    assert report.by_code["AAA"].closed_trades == 1
    assert report.by_code["AAA"].realized_pnl == pytest.approx(190.0)
    assert report.by_code["BBB"].closed_trades == 1
    assert report.by_code["BBB"].realized_pnl == pytest.approx(-210.0)
