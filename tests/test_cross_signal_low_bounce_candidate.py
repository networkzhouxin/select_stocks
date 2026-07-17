# -*- coding: utf-8 -*-
"""Tests for the low-bounce entry-filter candidate."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def candidate_score(**overrides):
    score = {
        "code": "159928",
        "buy_allowed": True,
        "buy_score": 64,
        "sell_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
        "trend_score": 9,
        "volume_score": 6,
    }
    score.update(overrides)
    return score


def test_low_bounce_candidate_blocks_volume_low_reversal_without_macd():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_low_bounce_candidate as candidate,
    )

    blocked = candidate_score()

    assert candidate.is_blocked_entry_combo(blocked)
    assert candidate.filter_buy_candidates(
        [blocked],
        held_codes=[],
        params={"buy_threshold": 60, "sell_threshold": 30},
    ) == []


def test_low_bounce_candidate_keeps_macd_strong_trend_or_non_low_bounce_entries():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_low_bounce_candidate as candidate,
    )

    with_macd = candidate_score(macd_cross_up=True)
    strong_trend = candidate_score(trend_score=20)
    no_volume = candidate_score(volume_score=0)
    not_low_bounce = candidate_score(close_between_boll_lower_mid=False)

    kept = candidate.filter_buy_candidates(
        [with_macd, strong_trend, no_volume, not_low_bounce],
        held_codes=[],
        params={"buy_threshold": 60, "sell_threshold": 30},
    )

    assert kept == [with_macd, strong_trend, no_volume, not_low_bounce]
