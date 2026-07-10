# -*- coding: utf-8 -*-
"""Tests for cross-signal training capital-utilization diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score(code, **overrides):
    item = {
        "code": code,
        "buy_allowed": True,
        "buy_score": 60,
        "sell_score": 0,
        "reversal_score": 12,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "volume_score": 0,
        "trend_score": 5,
    }
    item.update(overrides)
    return item


def day(date, total_value, cash, positions=None, orders=None):
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


def test_candidate_rejection_reason_matches_official_filter_order():
    from cross_signal_strategy.capital_utilization_diagnostics import (
        candidate_rejection_reason,
    )

    assert candidate_rejection_reason(score("AAA", buy_allowed=False)) == "overheat"
    assert candidate_rejection_reason(score("AAA", buy_score=49)) == "below_buy_threshold"
    assert candidate_rejection_reason(score("AAA", sell_score=30)) == "sell_conflict"
    assert candidate_rejection_reason(
        score(
            "AAA",
            close_between_boll_lower_mid=False,
            close_cross_boll_mid_up=False,
            close_near_ma20=False,
        )
    ) == "location_filter"
    assert candidate_rejection_reason(
        score(
            "AAA",
            macd_cross_up=True,
            kdj_k_cross_up=False,
            volume_score=4,
            trend_score=10,
        )
    ) == "blocked_entry_combo"
    assert candidate_rejection_reason(score("AAA")) == "eligible_unfilled"


def test_capital_report_attributes_vacant_slots_and_shadow_returns():
    from cross_signal_strategy.local_backtester import Position
    from cross_signal_strategy.capital_utilization_diagnostics import (
        build_capital_utilization_report,
    )

    trade_dates = ["2019-01-02", "2019-01-03", "2019-01-04"]
    results = [
        day("2019-01-02", 10000.0, 10000.0),
        day(
            "2019-01-03",
            10000.0,
            9000.0,
            positions={"HELD": Position("HELD", 100, 10.0)},
        ),
        day(
            "2019-01-04",
            10000.0,
            7000.0,
            positions={
                "H1": Position("H1", 100, 10.0),
                "H2": Position("H2", 100, 10.0),
                "H3": Position("H3", 100, 10.0),
            },
        ),
    ]
    daily_scores = {
        "2019-01-02": [
            score("AAA", buy_score=49),
            score(
                "BBB",
                buy_score=70,
                close_between_boll_lower_mid=False,
                close_cross_boll_mid_up=False,
                close_near_ma20=False,
            ),
        ],
        "2019-01-03": [score("CCC", buy_score=80)],
        "2019-01-04": [],
    }
    entry_prices = {
        ("AAA", "2019-01-02"): 10.0,
        ("BBB", "2019-01-02"): 20.0,
        ("CCC", "2019-01-03"): 10.0,
    }
    close_prices = {
        ("AAA", "2019-01-03"): 11.0,
        ("AAA", "2019-01-04"): 9.0,
        ("BBB", "2019-01-03"): 18.0,
        ("BBB", "2019-01-04"): 22.0,
        ("CCC", "2019-01-04"): 12.0,
    }

    report = build_capital_utilization_report(
        results=results,
        daily_scores=daily_scores,
        trade_dates=trade_dates,
        entry_price_lookup=lambda code, date: entry_prices[(code, date)],
        close_price_lookup=lambda code, date: close_prices[(code, date)],
        horizons=(1, 2),
        max_hold=3,
    )

    assert report.position_count_days == {0: 1, 1: 1, 3: 1}
    assert report.vacant_slot_days == 2
    assert report.total_vacant_slots == 5
    assert report.vacant_slot_reasons == {
        "below_buy_threshold": 1,
        "eligible_unfilled": 1,
        "location_filter": 1,
        "no_reversal_candidate": 2,
    }
    assert report.average_exposure == pytest.approx((0.0 + 0.1 + 0.3) / 3.0)
    assert report.average_cash_ratio == pytest.approx((1.0 + 0.9 + 0.7) / 3.0)

    below = report.shadow_by_reason["below_buy_threshold"]
    assert below.candidate_days == 1
    assert below.episodes == 1
    assert below.score_bands == {"40-49": 1}
    assert below.forward[1].observations == 1
    assert below.forward[1].average_return == pytest.approx(0.10)
    assert below.forward[2].average_return == pytest.approx(-0.10)
    assert below.forward_by_score_band["40-49"][1].average_return == pytest.approx(0.10)
    assert below.forward_by_score_band["40-49"][2].average_return == pytest.approx(-0.10)

    eligible = report.shadow_by_reason["eligible_unfilled"]
    assert eligible.forward[1].average_return == pytest.approx(0.20)
    assert eligible.forward[2].observations == 0


def test_shadow_returns_collapse_consecutive_candidate_days_into_one_episode():
    from cross_signal_strategy.capital_utilization_diagnostics import (
        build_capital_utilization_report,
    )

    trade_dates = ["2019-01-02", "2019-01-03", "2019-01-04"]
    results = [
        day("2019-01-02", 10000.0, 10000.0),
        day("2019-01-03", 10000.0, 10000.0),
        day("2019-01-04", 10000.0, 10000.0),
    ]
    daily_scores = {
        "2019-01-02": [score("AAA", buy_score=45)],
        "2019-01-03": [score("AAA", buy_score=47)],
        "2019-01-04": [],
    }
    entry_prices = {
        ("AAA", "2019-01-02"): 10.0,
        ("AAA", "2019-01-03"): 11.0,
    }
    close_prices = {
        ("AAA", "2019-01-03"): 11.0,
        ("AAA", "2019-01-04"): 12.0,
    }

    report = build_capital_utilization_report(
        results=results,
        daily_scores=daily_scores,
        trade_dates=trade_dates,
        entry_price_lookup=lambda code, date: entry_prices[(code, date)],
        close_price_lookup=lambda code, date: close_prices[(code, date)],
        horizons=(1,),
        max_hold=3,
    )

    below = report.shadow_by_reason["below_buy_threshold"]
    assert below.candidate_days == 2
    assert below.episodes == 1
    assert below.score_bands == {"40-49": 1}
    assert below.forward[1].observations == 1
    assert below.forward[1].average_return == pytest.approx(0.10)
    assert below.forward_by_score_band["40-49"][1].observations == 1


def test_shadow_score_bands_keep_50_59_separate_from_40_49():
    from cross_signal_strategy.capital_utilization_diagnostics import (
        build_capital_utilization_report,
    )

    report = build_capital_utilization_report(
        results=[day("2019-01-02", 10000.0, 10000.0)],
        daily_scores={
            "2019-01-02": [score("AAA", buy_score=55), score("BBB", buy_score=45)]
        },
        trade_dates=["2019-01-02"],
        entry_price_lookup=lambda code, date: 10.0,
        close_price_lookup=lambda code, date: 10.0,
        horizons=(1,),
        max_hold=3,
    )

    below = report.shadow_by_reason["below_buy_threshold"]
    assert below.score_bands == {"40-49": 1, "50-59": 1}


def test_capital_report_excludes_actual_buys_and_existing_holdings_from_shadow_pool():
    from cross_signal_strategy.local_backtester import OrderResult, Position
    from cross_signal_strategy.capital_utilization_diagnostics import (
        build_capital_utilization_report,
    )

    results = [
        day(
            "2019-01-02",
            10000.0,
            8000.0,
            positions={
                "HELD": Position("HELD", 100, 10.0),
                "BOUGHT": Position("BOUGHT", 100, 10.0),
            },
            orders=[
                OrderResult("BOUGHT", 100, 10.0, 5.0, "2019-01-02 09:35", True, "buy_signal")
            ],
        ),
    ]
    daily_scores = {
        "2019-01-02": [score("HELD", buy_score=90), score("BOUGHT", buy_score=80)]
    }

    report = build_capital_utilization_report(
        results=results,
        daily_scores=daily_scores,
        trade_dates=["2019-01-02"],
        entry_price_lookup=lambda code, date: 10.0,
        close_price_lookup=lambda code, date: 10.0,
        horizons=(1,),
        max_hold=3,
    )

    assert report.total_vacant_slots == 1
    assert report.vacant_slot_reasons == {"no_reversal_candidate": 1}
    assert report.shadow_by_reason == {}


def test_capital_report_rejects_dates_outside_training_window():
    from cross_signal_strategy.capital_utilization_diagnostics import (
        build_capital_utilization_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_capital_utilization_report(
            results=[day("2022-01-04", 10000.0, 10000.0)],
            daily_scores={},
            trade_dates=["2022-01-04"],
            entry_price_lookup=lambda code, date: 10.0,
            close_price_lookup=lambda code, date: 10.0,
        )
