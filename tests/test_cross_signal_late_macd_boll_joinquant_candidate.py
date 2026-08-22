# -*- coding: utf-8 -*-
"""Behavior tests for the standalone late-MACD/BOLL-upper JoinQuant veto."""

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
        "buy_score": 84,
        "sell_score": 0,
        "reversal_score": 45,
        "location_score": 15,
        "trend_score": 20,
        "volume_score": 4,
        "close": 2.535,
        "boll_upper": 2.524726,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": True,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "rsi6_cross_rsi12_up_age": 2,
        "rsi6_cross_rsi24_up_age": None,
        "macd_cross_up": True,
        "macd_cross_up_age": 0,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "kdj_k_cross_up_age": 1,
        "kdj_j_cross_up_age": None,
    }
    score.update(overrides)
    return score


def test_candidate_vetoes_exact_late_macd_upper_band_buy_only():
    from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as formal
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_late_macd_boll_filter_candidate as candidate,
    )

    score = eligible_score()

    assert [item["code"] for item in formal.filter_buy_candidates(
        [score], held_codes=[]
    )] == ["513100.XSHG"]
    assert candidate.is_late_macd_boll_upper_entry(score) is True
    assert candidate.filter_buy_candidates([score], held_codes=[]) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"macd_cross_up_age": 1},
        {"rsi6_cross_rsi12_up_age": 0},
        {"kdj_k_cross_up_age": 0},
        {"close": 2.524725},
        {"rsi6_cross_rsi12_down": True},
    ],
)
def test_candidate_keeps_each_near_miss_eligible(overrides):
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_late_macd_boll_filter_candidate as candidate,
    )

    score = eligible_score(**overrides)

    assert candidate.is_late_macd_boll_upper_entry(score) is False
    assert [item["code"] for item in candidate.filter_buy_candidates(
        [score], held_codes=[]
    )] == ["513100.XSHG"]


def test_candidate_does_not_change_existing_official_buy_guards():
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_late_macd_boll_filter_candidate as candidate,
    )

    blocked_by_threshold = eligible_score(
        code="LOW.XSHG", buy_score=59, macd_cross_up_age=1
    )
    already_held = eligible_score(
        code="HELD.XSHG", macd_cross_up_age=1
    )

    assert candidate.filter_buy_candidates(
        [blocked_by_threshold, already_held], held_codes=["HELD.XSHG"]
    ) == []
