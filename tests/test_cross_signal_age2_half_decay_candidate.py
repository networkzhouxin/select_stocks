# -*- coding: utf-8 -*-
"""Tests for the isolated bullish-cross age-2 half-decay candidate."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticSignalAdapter:
    def __init__(self, score):
        self._score = deepcopy(score)

    def score(self, code, current_date, return_reason=False):
        result = deepcopy(self._score)
        return (result, None) if return_reason else result


def _official_score(**overrides):
    values = {
        "code": "513100",
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi12_up_age": 2,
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi24_up_age": 1,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": True,
        "macd_cross_up_age": 2,
        "kdj_k_cross_up": True,
        "kdj_k_cross_up_age": 0,
        "kdj_j_cross_up": True,
        "kdj_j_cross_up_age": 2,
        "reversal_score": 45,
        "location_score": 10,
        "trend_score": 6,
        "volume_score": 4,
        "buy_score": 65,
        "buy_allowed": True,
        "sell_score": 28,
        "sell_reversal_score": 22,
        "sell_risk_score": 6,
        "macd_cross_down": True,
        "atr": 0.12,
        "close": 3.45,
        "unrelated_observation": {"nested": [1, 2, 3]},
    }
    values.update(overrides)
    return values


def _adapter(score):
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        Age2HalfDecaySignalAdapter,
    )

    return Age2HalfDecaySignalAdapter(StaticSignalAdapter(score))


def test_age2_bullish_crosses_receive_exactly_half_weight_while_age0_and_age1_stay_full():
    official = _official_score()

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["official_reversal_score"] == 45
    assert candidate["official_buy_score"] == 65
    assert candidate["age2_half_decay_penalty"] == pytest.approx(13.5)
    assert candidate["reversal_score"] == pytest.approx(31.5)
    assert candidate["buy_score"] == pytest.approx(51.5)


def test_age0_and_age1_bullish_crosses_are_identical_to_official_score():
    official = _official_score(
        rsi6_cross_rsi12_up_age=0,
        rsi6_cross_rsi24_up_age=1,
        macd_cross_up_age=1,
        kdj_k_cross_up_age=0,
        kdj_j_cross_up_age=1,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["age2_half_decay_penalty"] == 0
    assert candidate["reversal_score"] == official["reversal_score"]
    assert candidate["buy_score"] == official["buy_score"]


def test_mixed_rsi_direction_does_not_create_a_bullish_age2_penalty():
    official = _official_score(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi12_up_age=2,
        rsi6_cross_rsi24_up=False,
        rsi6_cross_rsi24_up_age=None,
        rsi6_cross_rsi12_down=True,
        reversal_score=0,
        location_score=-10,
        trend_score=0,
        volume_score=0,
        buy_score=0,
        macd_cross_up=False,
        macd_cross_up_age=None,
        kdj_k_cross_up=False,
        kdj_k_cross_up_age=None,
        kdj_j_cross_up=False,
        kdj_j_cross_up_age=None,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["age2_half_decay_penalty"] == 0
    assert candidate["reversal_score"] == 0
    assert candidate["buy_score"] == 0


def test_active_bullish_cross_without_age_metadata_fails_closed():
    official = _official_score(macd_cross_up=True)
    del official["macd_cross_up_age"]

    with pytest.raises(ValueError, match="macd_cross_up_age"):
        _adapter(official).score("513100", "2019-01-08")


def test_candidate_preserves_official_snapshot_and_sell_side_without_mutation():
    official = _official_score()
    original = deepcopy(official)
    adapter = _adapter(official)

    candidate, reason = adapter.score("513100", "2019-01-08", return_reason=True)
    candidate["unrelated_observation"]["nested"].append(99)
    second = adapter.score("513100", "2019-01-08")

    assert reason is None
    assert official == original
    assert second["unrelated_observation"] == original["unrelated_observation"]
    for key in (
        "code",
        "current_date",
        "signal_date",
        "max_data_date",
        "location_score",
        "trend_score",
        "volume_score",
        "buy_allowed",
        "sell_score",
        "sell_reversal_score",
        "sell_risk_score",
        "macd_cross_down",
        "atr",
        "close",
    ):
        assert second[key] == original[key]
