# -*- coding: utf-8 -*-
"""Tests for the isolated reversal-first candidate ranking experiment."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        item = self.scores.get(code)
        return (dict(item), None) if return_reason else dict(item)


def candidate(code, buy_score, reversal_score):
    return {
        "code": code,
        "buy_score": buy_score,
        "sell_score": 0,
        "reversal_score": reversal_score,
        "location_score": 20,
        "trend_score": 10,
        "volume_score": 0,
        "buy_allowed": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "atr": 0.2,
    }


def day(date, total_value, buy_code=None):
    from cross_signal_strategy.local_backtester import DayResult, OrderResult

    orders = []
    if buy_code:
        orders.append(OrderResult(
            buy_code,
            100,
            10.0,
            0.0,
            "%s 09:35" % date,
            True,
            "buy_signal",
        ))
    return DayResult(
        date=date,
        previous_date=None,
        orders=orders,
        cash=float(total_value),
        positions={},
        marks={},
        total_value=float(total_value),
    )


def performance(total_return, max_drawdown, sharpe, sortino, annual):
    from cross_signal_strategy.ranking_candidate import RankingPerformance

    return RankingPerformance(
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        annual_returns=annual,
    )


def test_reversal_first_sort_changes_only_candidate_priority():
    from cross_signal_strategy.ranking_candidate import reversal_first_sort

    items = [
        candidate("TOTAL", buy_score=80, reversal_score=35),
        candidate("REV", buy_score=70, reversal_score=45),
        candidate("TIE", buy_score=75, reversal_score=45),
    ]

    ordered = reversal_first_sort(items)

    assert [item["code"] for item in ordered] == ["TIE", "REV", "TOTAL"]


def test_reversal_first_planner_selects_higher_reversal_when_one_slot_exists():
    from cross_signal_strategy.local_backtester import LocalBroker
    from cross_signal_strategy.local_order_planner import strategy
    from cross_signal_strategy.ranking_candidate import ReversalFirstOrderPlanner

    scores = {
        "TOTAL": candidate("TOTAL", buy_score=80, reversal_score=35),
        "REV": candidate("REV", buy_score=70, reversal_score=45),
    }
    params = strategy.get_default_params()
    params["max_hold"] = 1
    planner = ReversalFirstOrderPlanner(
        FakeSignalAdapter(scores),
        etf_pool=["TOTAL", "REV"],
        params=params,
    )

    orders = planner.plan_orders(
        "2019-01-02",
        None,
        LocalBroker(initial_cash=20000.0),
    )

    assert len(orders) == 1
    assert orders[0]["code"] == "REV"
    assert orders[0]["reason"] == "buy_signal"


def test_ranking_gate_requires_path_activity_and_strict_metric_non_degradation():
    from cross_signal_strategy.ranking_candidate import evaluate_ranking_gate

    baseline = performance(
        total_return=1.0,
        max_drawdown=0.08,
        sharpe=2.0,
        sortino=3.0,
        annual={2019: 0.20, 2020: 0.30, 2021: 0.10},
    )
    candidate_perf = performance(
        total_return=1.1,
        max_drawdown=0.07,
        sharpe=2.1,
        sortino=3.1,
        annual={2019: 0.21, 2020: 0.31, 2021: 0.11},
    )

    passed = evaluate_ranking_gate(
        baseline,
        candidate_perf,
        changed_days_by_year={2019: 4, 2020: 3, 2021: 3},
    )
    failed = evaluate_ranking_gate(
        baseline,
        performance(
            total_return=1.1,
            max_drawdown=0.09,
            sharpe=2.1,
            sortino=3.1,
            annual={2019: 0.21, 2020: 0.31, 2021: 0.09},
        ),
        changed_days_by_year={2019: 4, 2020: 3, 2021: 3},
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("drawdown" in reason for reason in failed.reasons)
    assert any("2021" in reason for reason in failed.reasons)


def test_ranking_comparison_counts_changed_buy_days_and_annual_returns():
    from cross_signal_strategy.ranking_candidate import build_ranking_comparison

    baseline = [
        day("2019-12-31", 11000.0, "AAA"),
        day("2020-12-31", 12100.0, "AAA"),
        day("2021-12-31", 13310.0, "AAA"),
    ]
    candidate_days = [
        day("2019-12-31", 11200.0, "BBB"),
        day("2020-12-31", 12320.0, "AAA"),
        day("2021-12-31", 13552.0, "CCC"),
    ]

    report = build_ranking_comparison(
        baseline,
        candidate_days,
        initial_cash=10000.0,
    )

    assert report.changed_buy_days == 2
    assert report.changed_days_by_year == {2019: 1, 2021: 1}
    assert report.changed_decisions[0].date == "2019-12-31"
    assert report.changed_decisions[0].baseline_codes == ("AAA",)
    assert report.changed_decisions[0].candidate_codes == ("BBB",)
    assert report.changed_decisions[1].date == "2021-12-31"
    assert report.changed_decisions[1].candidate_codes == ("CCC",)
    assert report.baseline_performance.annual_returns[2019] == pytest.approx(0.10)
    assert report.candidate_performance.annual_returns[2019] == pytest.approx(0.12)
    assert report.candidate_performance.annual_returns[2020] == pytest.approx(0.10)
    assert report.candidate_performance.annual_returns[2021] == pytest.approx(0.10)
    assert report.gate.passed is False


def test_ranking_comparison_rejects_non_training_dates():
    from cross_signal_strategy.ranking_candidate import build_ranking_comparison

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_ranking_comparison(
            [day("2022-01-04", 10000.0)],
            [day("2022-01-04", 10000.0)],
            initial_cash=10000.0,
        )
