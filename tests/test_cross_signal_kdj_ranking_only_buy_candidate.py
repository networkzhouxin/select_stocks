# -*- coding: utf-8 -*-
"""Tests for KDJ buy points that rank but cannot create eligibility."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PoolSignalAdapter:
    def __init__(self, scores):
        self._scores = {str(item["code"]): item for item in scores}

    def score(self, code, current_date, return_reason=False):
        value = self._scores.get(str(code))
        result = deepcopy(value) if value is not None else None
        reason = None if result is not None else "no_data"
        return (result, reason) if return_reason else result


def _module():
    try:
        from cross_signal_strategy.research import kdj_ranking_only_buy_candidate
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("ranking-only KDJ candidate is not implemented: %s" % exc)
    return kdj_ranking_only_buy_candidate


def _official_score(code="513100", **overrides):
    values = {
        "code": code,
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "k": 50.0,
        "buy_score": 61.0,
        "sell_score": 0.0,
        "reversal_score": 20.0,
        "volume_score": 0.0,
        "buy_allowed": True,
        "downside_continuation": False,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": False,
        "adx": 10.0,
        "plus_di": 10.0,
        "minus_di": 20.0,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _adapter(scores):
    module = _module()
    return module.KdjRankingOnlyBuyScoreAdapter(
        PoolSignalAdapter(scores),
        trade_dates=("2019-01-08",),
    )


def test_kdj_bonus_cannot_turn_official_41_into_an_eligible_buy():
    module = _module()
    candidate = _adapter([
        _official_score(k=19.0, buy_score=41.0),
    ]).score("513100", "2019-01-08")

    assert candidate["buy_extreme_zone_score"] == 20.0
    assert candidate["buy_rank_score"] == 61.0
    assert candidate["buy_score"] == 41.0
    assert module.strategy.filter_buy_candidates([candidate], []) == []


def test_officially_eligible_buy_keeps_qualification_and_gets_rank_bonus():
    module = _module()
    candidate = _adapter([
        _official_score(k=19.0, buy_score=61.0),
    ]).score("513100", "2019-01-08")

    assert candidate["buy_score"] == 61.0
    assert candidate["buy_rank_score"] == 81.0
    assert module.strategy.filter_buy_candidates([candidate], []) == [candidate]


def test_planner_uses_rank_bonus_only_after_official_eligibility():
    from cross_signal_strategy.local.local_backtester import LocalBroker

    module = _module()
    adapter = _adapter([
        _official_score("513100", k=19.0, buy_score=61.0),
        _official_score("513500", k=50.0, buy_score=70.0),
        _official_score("513050", k=19.0, buy_score=41.0),
    ])
    planner = module.KdjRankingOnlyBuyPlanner(
        adapter,
        etf_pool=("513100", "513500", "513050"),
        trade_dates=["2019-01-08"],
    )

    orders = planner.plan_orders(
        "2019-01-08",
        "2019-01-07",
        LocalBroker(initial_cash=20000.0),
    )

    assert [order["code"] for order in orders] == ["513100", "513500"]
    assert orders[0]["target_value"] == pytest.approx(6333.333333333333)


def test_sell_tier_remains_part_of_the_unified_sell_score():
    candidate = _adapter([
        _official_score(k=75.0, sell_score=25.0),
    ]).score("513100", "2019-01-08")

    assert candidate["sell_extreme_zone_score"] == 5.0
    assert candidate["sell_score"] == 30.0


def _performance(**overrides):
    from cross_signal_strategy.research.extreme_zone_score_candidate import (
        ExtremeZonePerformance,
    )

    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.06,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.56,
        "profit_loss_ratio": 4.0,
        "buy_count": 90,
        "sell_count": 88,
        "annual_returns": {2019: 0.2, 2020: 0.3, 2021: 0.1},
    }
    values.update(overrides)
    return ExtremeZonePerformance(**values)


def test_gate_requires_accuracy_friction_and_material_effect():
    module = _module()
    official = _performance(win_rate=0.56)
    current = _performance(total_return=0.96, win_rate=0.54)
    candidate = _performance(total_return=0.97, win_rate=0.57)
    official_stress = _performance(total_return=0.80, win_rate=0.51)
    candidate_stress = _performance(total_return=0.77, win_rate=0.52)

    passed = module.evaluate_kdj_ranking_only_gate(
        official,
        current,
        candidate,
        official_stress,
        candidate_stress,
        {2019: 1, 2020: 1, 2021: 1},
        changed_days_vs_official=1,
    )
    failed = module.evaluate_kdj_ranking_only_gate(
        official,
        current,
        candidate,
        official_stress,
        _performance(total_return=0.77, win_rate=0.50),
        {2019: 1, 2020: 1, 2021: 1},
        changed_days_vs_official=0,
    )

    assert passed.passed
    assert not failed.passed
    assert any("stress win rate" in reason for reason in failed.reasons)
    assert any("official path" in reason for reason in failed.reasons)


def test_training_runner_rejects_unapproved_data_root():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_ranking_only_training_comparison(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
