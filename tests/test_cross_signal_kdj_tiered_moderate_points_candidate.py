# -*- coding: utf-8 -*-
"""Tests for the isolated KDJ 20/10 buy and 10/5 sell candidate."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatedSignalAdapter:
    def __init__(self, scores_by_date):
        self._scores_by_date = scores_by_date

    def score(self, code, current_date, return_reason=False):
        score = self._scores_by_date.get(str(current_date))
        result = deepcopy(score) if score is not None else None
        reason = None if result is not None else "no_data"
        return (result, reason) if return_reason else result


def _module():
    try:
        from cross_signal_strategy.research import (
            kdj_tiered_moderate_points_candidate,
        )
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("moderate-points KDJ candidate is not implemented: %s" % exc)
    return kdj_tiered_moderate_points_candidate


def _official_score(current_date, **overrides):
    values = {
        "code": "513100",
        "current_date": current_date,
        "signal_date": current_date,
        "max_data_date": current_date,
        "k": 50.0,
        "downside_continuation": False,
        "buy_score": 40.0,
        "sell_score": 20.0,
        "reversal_score": 20.0,
        "volume_score": 0.0,
        "buy_allowed": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 35.0,
        "plus_di": 30.0,
        "minus_di": 10.0,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _adapter(scores_by_date, dates=None):
    module = _module()
    return module.KdjTieredModeratePointsScoreAdapter(
        DatedSignalAdapter(scores_by_date),
        trade_dates=dates or tuple(scores_by_date),
    )


@pytest.mark.parametrize(
    ("k_value", "buy_bonus", "sell_bonus"),
    [
        (20.0, 20.0, 0.0),
        (20.1, 10.0, 0.0),
        (30.0, 10.0, 0.0),
        (30.1, 0.0, 0.0),
        (69.9, 0.0, 0.0),
        (70.0, 0.0, 5.0),
        (79.9, 0.0, 5.0),
        (80.0, 0.0, 10.0),
    ],
)
def test_moderate_tiers_adjust_the_unified_scores(
    k_value,
    buy_bonus,
    sell_bonus,
):
    date = "2019-01-08"
    candidate = _adapter({date: _official_score(date, k=k_value)}).score(
        "513100", date
    )

    assert candidate["buy_extreme_zone_score"] == buy_bonus
    assert candidate["sell_extreme_zone_score"] == sell_bonus
    assert candidate["buy_score"] == 40.0 + buy_bonus
    assert candidate["sell_score"] == 20.0 + sell_bonus


def test_prior_extreme_state_is_not_retained():
    dates = ("2019-01-08", "2019-01-09")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=50.0),
    }

    candidate = _adapter(scores, dates).score("513100", dates[1])

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["sell_extreme_zone_score"] == 0.0


def test_current_downside_continuation_blocks_the_buy_bonus():
    date = "2019-01-08"
    candidate = _adapter(
        {date: _official_score(date, k=19.0, downside_continuation=True)}
    ).score("513100", date)

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["buy_score"] == 40.0


def _protected_sell_orders(**score_overrides):
    from cross_signal_strategy.local.local_backtester import LocalBroker, Position
    from cross_signal_strategy.local.local_order_planner import (
        LocalCrossSignalOrderPlanner,
    )

    dates = (
        "2019-01-02",
        "2019-01-03",
        "2019-01-04",
        "2019-01-07",
        "2019-01-08",
        "2019-01-09",
    )
    current = dates[-1]
    adapter = _adapter(
        {current: _official_score(current, k=80.0, **score_overrides)},
        dates,
    )
    planner = LocalCrossSignalOrderPlanner(
        adapter,
        etf_pool=["513100"],
        buy_dates={"513100": dates[0]},
        trade_dates=list(dates),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["513100"] = Position("513100", 1000, 3.0)
    return planner.plan_orders(current, dates[-2], broker)


def test_sell_score_at_threshold_still_requires_price_confirmation():
    assert _protected_sell_orders() == []


def test_price_confirmation_at_threshold_still_respects_adx_protection():
    assert _protected_sell_orders(close_below_boll_mid=True) == []


def test_severe_price_break_can_sell_after_minimum_hold():
    assert _protected_sell_orders(close_below_ma20=True) == [
        {"code": "513100", "target_value": 0.0, "reason": "signal_sell"}
    ]


def test_training_runner_rejects_unapproved_data_roots():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_tiered_moderate_points_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
