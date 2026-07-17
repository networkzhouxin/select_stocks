# -*- coding: utf-8 -*-
"""Tests for entry-combo filter candidate behavior."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def candidate_score(**overrides):
    score = {
        "code": "510300",
        "buy_allowed": True,
        "buy_score": 60,
        "sell_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": True,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
        "trend_score": 12,
        "volume_score": 6,
    }
    score.update(overrides)
    return score


def test_combo_candidate_blocks_macd_rsi_volume_entry_without_kdj():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_combo_candidate as candidate,
    )

    blocked = candidate_score()

    assert candidate.is_blocked_entry_combo(blocked)
    assert candidate.filter_buy_candidates([blocked], held_codes=[], params={"buy_threshold": 45, "sell_threshold": 30}) == []


def test_combo_candidate_keeps_kdj_or_strong_trend_confirmed_entries():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_combo_candidate as candidate,
    )

    with_kdj = candidate_score(kdj_k_cross_up=True)
    strong_trend = candidate_score(trend_score=20)

    kept = candidate.filter_buy_candidates(
        [with_kdj, strong_trend],
        held_codes=[],
        params={"buy_threshold": 45, "sell_threshold": 30},
    )

    assert kept == [with_kdj, strong_trend]
