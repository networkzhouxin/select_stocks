# -*- coding: utf-8 -*-
"""Tests for strong-trend idle-capacity diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score(code, trend_score):
    return {
        "code": code,
        "trend_score": trend_score,
        "buy_score": 70,
    }


def order(code, target_value, reason):
    return {
        "code": code,
        "target_value": float(target_value),
        "reason": reason,
    }


def closed_trade(code, buy_date, sell_date, pnl, entry_score=None):
    from cross_signal_strategy.trade_diagnostics import ClosedTradeDiagnostic

    buy_price = 10.0
    sell_price = buy_price + float(pnl) / 100.0
    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason="signal_sell",
        amount=100,
        buy_price=buy_price,
        sell_price=sell_price,
        pnl=float(pnl),
        return_pct=(sell_price / buy_price - 1.0) * 100.0,
        entry_score=entry_score or {"trend_score": 20},
    )


def day(date, orders=None, cash=10000.0, positions=None, total_value=10000.0):
    from cross_signal_strategy.local_backtester import DayResult

    positions = positions or {}
    return DayResult(
        date=date,
        previous_date=None,
        orders=orders or [],
        cash=float(cash),
        positions=positions,
        marks={code: 10.0 for code in positions},
        total_value=float(total_value),
    )


def filled_buy(code):
    from cross_signal_strategy.local_backtester import OrderResult

    return OrderResult(
        code=code,
        amount_delta=100,
        exec_price=10.0,
        commission=0.0,
        side_time="2019-01-02 09:35",
        filled=True,
        reason="buy_signal",
    )


def test_build_entry_contexts_uses_only_slot_left_after_all_primary_buys():
    from cross_signal_strategy.local_backtester import Position
    from cross_signal_strategy.strong_trend_capacity_diagnostics import (
        build_entry_contexts,
    )

    class Broker:
        cash = 8000.0
        positions = {"OLD": Position("OLD", 200, 10.0)}

    contexts = build_entry_contexts(
        date="2019-01-02",
        broker=Broker(),
        current_prices={"OLD": 10.0},
        orders=[
            order("OLD", 0.0, "signal_sell"),
            order("STRONG", 3000.0, "buy_signal"),
            order("MILD", 3000.0, "buy_signal"),
        ],
        scores={
            "STRONG": score("STRONG", 20),
            "MILD": score("MILD", 10),
        },
        params={"max_hold": 3, "base_ratio": 0.90},
    )

    assert contexts["STRONG"].is_strong is True
    assert contexts["STRONG"].unused_slots_after_orders == 1
    assert contexts["STRONG"].cash_headroom_ratio == pytest.approx(0.30)
    assert contexts["STRONG"].slot_ratio == pytest.approx(0.30)
    assert contexts["STRONG"].capacity_eligible is True
    assert contexts["MILD"].is_strong is False
    assert contexts["MILD"].capacity_eligible is False


def test_build_entry_contexts_assigns_one_unused_slot_to_top_ranked_strong_buy():
    from cross_signal_strategy.strong_trend_capacity_diagnostics import (
        build_entry_contexts,
    )

    class Broker:
        cash = 10000.0
        positions = {}

    contexts = build_entry_contexts(
        date="2019-01-02",
        broker=Broker(),
        current_prices={},
        orders=[
            order("FIRST", 3000.0, "buy_signal"),
            order("SECOND", 3000.0, "buy_signal"),
        ],
        scores={
            "FIRST": score("FIRST", 20),
            "SECOND": score("SECOND", 25),
        },
        params={"max_hold": 3, "base_ratio": 0.90},
    )

    assert contexts["FIRST"].capacity_eligible is True
    assert contexts["SECOND"].capacity_eligible is False


def test_capacity_report_calculates_close_based_excursions_and_year_quality():
    from cross_signal_strategy.strong_trend_capacity_diagnostics import (
        StrongTrendEntryContext,
        build_strong_trend_capacity_report,
    )

    trade_dates = ["2019-01-02", "2019-01-03", "2019-01-04"]
    results = [
        day("2019-01-02", orders=[filled_buy("AAA")]),
        day("2019-01-03"),
        day("2019-01-04"),
    ]
    contexts = {
        ("2019-01-02", "AAA"): StrongTrendEntryContext(
            date="2019-01-02",
            code="AAA",
            trend_score=20.0,
            is_strong=True,
            unused_slots_after_orders=1,
            cash_headroom_ratio=0.35,
            slot_ratio=0.30,
            capacity_eligible=True,
        )
    }
    prices = {
        ("AAA", "2019-01-02"): 11.0,
        ("AAA", "2019-01-03"): 9.0,
        ("AAA", "2019-01-04"): 12.0,
    }

    report = build_strong_trend_capacity_report(
        results=results,
        trades=[closed_trade("AAA", "2019-01-02", "2019-01-04", 200.0)],
        entry_contexts=contexts,
        trade_dates=trade_dates,
        close_price_lookup=lambda code, date: prices[(code, date)],
    )

    assert report.strong_entries == 1
    assert report.capacity_entries == 1
    assert report.capacity_open_entries == 0
    assert report.capacity_stats.closed_trades == 1
    assert report.capacity_stats.realized_pnl == pytest.approx(200.0)
    assert report.capacity_stats.average_mfe == pytest.approx(0.20)
    assert report.capacity_stats.average_mae == pytest.approx(-0.10)
    assert report.capacity_by_year[2019].win_rate == pytest.approx(1.0)
    assert report.concentration.largest_trade_profit_share == pytest.approx(1.0)
    assert report.concentration.largest_code_profit_share == pytest.approx(1.0)


def test_capacity_gate_requires_enough_profitable_entries_in_every_training_year():
    from cross_signal_strategy.strong_trend_capacity_diagnostics import (
        CapacityConcentration,
        StrongTrendPathStats,
        evaluate_capacity_gate,
    )

    annual = {
        year: StrongTrendPathStats(
            closed_trades=4,
            wins=3,
            losses=1,
            realized_pnl=100.0,
            gross_profit=140.0,
            gross_loss=40.0,
            average_return=0.03,
            average_mfe=0.10,
            average_mae=-0.04,
        )
        for year in (2019, 2020, 2021)
    }
    total = StrongTrendPathStats(
        closed_trades=12,
        wins=9,
        losses=3,
        realized_pnl=300.0,
        gross_profit=420.0,
        gross_loss=120.0,
        average_return=0.03,
        average_mfe=0.10,
        average_mae=-0.04,
    )
    concentration = CapacityConcentration(
        gross_profit=420.0,
        largest_trade_profit_share=0.30,
        largest_code_profit_share=0.40,
    )

    passed = evaluate_capacity_gate(total, annual, concentration)
    failed = evaluate_capacity_gate(
        total,
        {**annual, 2021: StrongTrendPathStats(closed_trades=2, realized_pnl=-10.0)},
        concentration,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_capacity_report_rejects_dates_outside_training_window():
    from cross_signal_strategy.strong_trend_capacity_diagnostics import (
        build_strong_trend_capacity_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_strong_trend_capacity_report(
            results=[day("2022-01-04")],
            trades=[],
            entry_contexts={},
            trade_dates=["2022-01-04"],
            close_price_lookup=lambda code, date: 10.0,
        )
