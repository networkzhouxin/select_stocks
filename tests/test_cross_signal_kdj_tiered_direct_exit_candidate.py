# -*- coding: utf-8 -*-
"""Tests for the isolated current-state KDJ tier direct-exit candidate."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module():
    from cross_signal_strategy.research import kdj_tiered_direct_exit_candidate

    return kdj_tiered_direct_exit_candidate


def _score(**overrides):
    values = {
        "sell_score": 35.0,
        "sell_extreme_zone_score": 5.0,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 35.0,
        "plus_di": 30.0,
        "minus_di": 10.0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("sell_score", "bonus"),
    [
        (30.0, 5.0),
        (35.0, 5.0),
        (30.0, 10.0),
        (45.0, 10.0),
    ],
)
def test_extreme_bonus_at_or_above_threshold_bypasses_price_and_adx(
    sell_score,
    bonus,
):
    module = _module()

    assert module.should_force_kdj_extreme_sell(
        _score(sell_score=sell_score, sell_extreme_zone_score=bonus),
        params={"sell_threshold": 30, "adx_trend_threshold": 25},
    )


def test_extreme_bonus_below_threshold_does_not_force_sell():
    module = _module()

    assert not module.should_force_kdj_extreme_sell(
        _score(sell_score=29.0, sell_extreme_zone_score=5.0),
        params={"sell_threshold": 30, "adx_trend_threshold": 25},
    )


def test_no_extreme_bonus_falls_back_to_official_sell_confirmation():
    module = _module()
    params = {"sell_threshold": 30, "adx_trend_threshold": 25}

    blocked = module.should_force_kdj_extreme_sell(
        _score(sell_extreme_zone_score=0.0),
        params=params,
    )
    confirmed = module.should_force_kdj_extreme_sell(
        _score(
            sell_extreme_zone_score=0.0,
            close_below_ma20=True,
            adx=10.0,
        ),
        params=params,
    )

    assert not blocked
    assert confirmed


def test_atr_stop_remains_an_unconditional_sell():
    module = _module()

    assert module.should_force_kdj_extreme_sell(
        _score(sell_score=0.0, sell_extreme_zone_score=0.0),
        atr_stop_triggered=True,
        params={"sell_threshold": 30, "adx_trend_threshold": 25},
    )


def test_training_runner_rejects_unapproved_data_roots():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_tiered_direct_exit_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
