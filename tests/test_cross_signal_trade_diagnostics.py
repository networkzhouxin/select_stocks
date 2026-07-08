# -*- coding: utf-8 -*-
"""Tests for cross-signal trade attribution diagnostics."""

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        item = self.scores.get((current_date, code), self.scores.get(code))
        if item is None:
            return (None, "no_data") if return_reason else None
        return (dict(item), None) if return_reason else dict(item)


def candidate(code, buy_score=70):
    return {
        "code": code,
        "buy_score": buy_score,
        "sell_score": 0,
        "reversal_score": 35,
        "location_score": 15,
        "trend_score": 14,
        "volume_score": 0,
        "buy_allowed": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 20.0,
        "plus_di": 25.0,
        "minus_di": 10.0,
        "atr": 0.1,
    }


def test_diagnostic_planner_captures_entry_score_snapshot_before_later_scores_change():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.trade_diagnostics import DiagnosticOrderPlanner

    adapter = FakeSignalAdapter({"AAA": candidate("AAA", buy_score=70)})
    planner = DiagnosticOrderPlanner(adapter, etf_pool=["AAA"])
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders("2019-01-02", None, broker)
    planner.last_scores["AAA"]["buy_score"] = 10

    assert orders[0]["reason"] == "buy_signal"
    assert planner.entry_score_snapshots[("2019-01-02", "AAA")]["buy_score"] == 70


def test_diagnostic_planner_captures_exit_score_snapshot_for_signal_sells():
    from cross_signal_strategy.local_backtester import LocalBroker, Position
    from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy
    from cross_signal_strategy.trade_diagnostics import DiagnosticOrderPlanner

    sell_score = candidate("AAA", buy_score=40)
    sell_score.update({
        "sell_score": 34,
        "close_below_ma20": True,
    })
    adapter = FakeSignalAdapter({"AAA": sell_score})
    params = strategy.get_default_params()
    params["min_signal_hold_days"] = 1
    planner = DiagnosticOrderPlanner(
        adapter,
        etf_pool=["AAA"],
        params=params,
        trade_dates=["2019-01-02", "2019-01-08"],
    )
    planner.buy_dates["AAA"] = "2019-01-02"
    broker = LocalBroker(initial_cash=20000.0)
    broker.positions["AAA"] = Position(code="AAA", amount=100, avg_cost=10.0)

    orders = planner.plan_orders("2019-01-08", "2019-01-07", broker, current_prices={"AAA": 9.5})

    assert orders[0]["reason"] == "signal_sell"
    assert planner.exit_score_snapshots[("2019-01-08", "AAA")]["sell_score"] == 34
    assert planner.exit_score_snapshots[("2019-01-08", "AAA")]["close_below_ma20"] is True


def test_closed_trade_diagnostics_use_entry_score_snapshot():
    from cross_signal_strategy.local_backtester import DayResult, OrderResult
    from cross_signal_strategy.trade_diagnostics import build_closed_trade_diagnostics

    results = [
        DayResult(
            "2019-01-02",
            None,
            [OrderResult("AAA", 100, 10.0, 5.0, "2019-01-02 09:35", True, "buy_signal")],
            cash=8995.0,
            positions={},
            marks={},
            total_value=9995.0,
        ),
        DayResult(
            "2019-01-08",
            "2019-01-07",
            [OrderResult("AAA", -100, 12.0, 5.0, "2019-01-08 09:35", True, "signal_sell")],
            cash=10190.0,
            positions={},
            marks={},
            total_value=10190.0,
        ),
    ]

    trades = build_closed_trade_diagnostics(
        results,
        entry_score_snapshots={("2019-01-02", "AAA"): {"buy_score": 70, "volume_score": 6}},
        exit_score_snapshots={("2019-01-08", "AAA"): {"sell_score": 34, "close_below_ma20": True}},
    )

    assert len(trades) == 1
    assert trades[0].pnl == pytest.approx(190.0)
    assert trades[0].return_pct == pytest.approx(20.0)
    assert trades[0].entry_score["buy_score"] == 70
    assert trades[0].entry_score["volume_score"] == 6
    assert trades[0].exit_score["sell_score"] == 34
    assert trades[0].exit_score["close_below_ma20"] is True
