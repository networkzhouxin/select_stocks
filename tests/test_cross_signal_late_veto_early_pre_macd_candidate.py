# -*- coding: utf-8 -*-
"""Behavior tests for the stacked late-veto + early pre-MACD candidate."""

from __future__ import annotations

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def eligible_score(**overrides):
    score = {
        "code": "513100.XSHG",
        "buy_allowed": True,
        "buy_score": 55,
        "sell_score": 0,
        "reversal_score": 35,
        "location_score": 15,
        "trend_score": 10,
        "volume_score": 0,
        "close": 2.50,
        "boll_upper": 2.60,
        "rsi6": 70.0,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": True,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "rsi6_cross_rsi12_up_age": 0,
        "rsi6_cross_rsi24_up_age": None,
        "macd_cross_up": False,
        "macd_cross_up_age": None,
        "dif": -0.020,
        "dea": -0.010,
        "dif_prev": -0.025,
        "dea_prev": -0.010,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "kdj_k_cross_up_age": 1,
        "kdj_j_cross_up_age": None,
    }
    score.update(overrides)
    return score


def candidate_module():
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate,
    )

    return smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate


def test_early_channel_accepts_fresh_rsi_kdj_with_negative_narrowing_macd():
    candidate = candidate_module()

    assert candidate.is_early_pre_macd_entry(eligible_score()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"buy_score": 49.999},
        {"buy_score": 60},
        {"rsi6_cross_rsi12_up_age": 2},
        {"kdj_k_cross_up_age": 2},
        {"rsi6_cross_rsi12_down": True},
        {"macd_cross_up": True, "macd_cross_up_age": 0},
        {"dif": -0.005, "dea": -0.010},
        {"dif": -0.030, "dea": -0.010},
        {"close": 2.60},
        {"rsi6": 85.0},
    ],
)
def test_early_channel_rejects_every_boundary_and_near_miss(overrides):
    candidate = candidate_module()

    assert candidate.is_early_pre_macd_entry(
        eligible_score(**overrides)
    ) is False


def test_primary_queue_stays_ahead_of_early_queue_regardless_of_score_order():
    candidate = candidate_module()
    early = eligible_score(code="EARLY.XSHG", buy_score=59)
    primary = eligible_score(
        code="PRIMARY.XSHG",
        buy_score=60,
        macd_cross_up=True,
        macd_cross_up_age=1,
    )

    queue = candidate.build_new_buy_queue([early, primary], held_codes=[])

    assert [(item["code"], item["entry_channel"]) for item in queue] == [
        ("PRIMARY.XSHG", "primary"),
        ("EARLY.XSHG", "early_pre_macd"),
    ]


def test_existing_late_macd_upper_band_veto_still_removes_primary_candidate():
    candidate = candidate_module()
    late_primary = eligible_score(
        code="LATE.XSHG",
        buy_score=84,
        close=2.61,
        macd_cross_up=True,
        macd_cross_up_age=0,
        rsi6_cross_rsi12_up_age=2,
        kdj_k_cross_up_age=1,
    )

    assert candidate.is_late_macd_boll_upper_entry(late_primary) is True
    assert candidate.build_new_buy_queue([late_primary], held_codes=[]) == []


@pytest.mark.parametrize(
    "overrides,held_codes",
    [
        ({"sell_score": 30}, []),
        ({"close_far_above_ma20": True}, []),
        ({"close_cross_boll_mid_up": False}, []),
        ({"code": "HELD.XSHG"}, ["HELD.XSHG"]),
    ],
)
def test_early_channel_keeps_existing_buy_guards(overrides, held_codes):
    candidate = candidate_module()

    assert candidate.filter_early_pre_macd_buy_candidates(
        [eligible_score(**overrides)], held_codes=held_codes
    ) == []
