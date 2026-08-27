# -*- coding: utf-8 -*-
"""Behavior tests for the standalone JoinQuant fast-entry candidate."""

from __future__ import annotations

import pathlib
import sys
import types

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def candidate_score(code="FRESH", buy_score=55, **overrides):
    score = {
        "code": code,
        "buy_allowed": True,
        "buy_score": buy_score,
        "sell_score": 0,
        "reversal_score": 40,
        "close": 11.0,
        "atr": 1.0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "rsi6_cross_rsi12_up_age": 1,
        "rsi6_cross_rsi24_up_age": 0,
        "macd_cross_up_age": 0,
        "kdj_k_cross_up_age": 0,
        "kdj_j_cross_up_age": None,
        "volume_score": 0,
        "trend_score": 5,
    }
    score.update(overrides)
    return score


def test_joinquant_snapshot_enrichment_uses_only_visible_daily_closes():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_fresh_unextended_entry_candidate as candidate,
    )

    snapshot = candidate_score()
    closes = pd.Series([9.5, 10.0, 11.0])

    enriched = candidate.enrich_fresh_entry_snapshot(snapshot, closes)

    assert enriched["fresh_entry_earliest_cross_age"] == 1
    assert enriched["fresh_entry_cross_close"] == 10.0
    assert enriched["fresh_entry_extension_atr"] == 1.0


def test_joinquant_buy_queue_preserves_primary_priority_and_fixed_fast_gate():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_fresh_unextended_entry_candidate as candidate,
    )

    fresh = candidate.enrich_fresh_entry_snapshot(
        candidate_score("FRESH"),
        pd.Series([9.5, 10.0, 11.0]),
    )
    old = candidate.enrich_fresh_entry_snapshot(
        candidate_score("OLD", rsi6_cross_rsi12_up_age=2),
        pd.Series([9.5, 10.0, 11.0]),
    )
    extended = dict(fresh, code="EXTENDED", fresh_entry_extension_atr=1.01)
    scores = [
        dict(fresh, code="P1", buy_score=80),
        dict(fresh, code="P2", buy_score=70),
        fresh,
        old,
        extended,
    ]

    queue = candidate.build_new_buy_queue(scores, held_codes=[])

    assert [(item["code"], item["entry_channel"]) for item in queue] == [
        ("P1", "primary"),
        ("P2", "primary"),
        ("FRESH", "fresh_unextended"),
    ]
    assert [item["code"] for item in queue[:2]] == ["P1", "P2"]


def test_joinquant_fast_gate_rejects_missing_contributing_cross_age():
    from cross_signal_strategy.archive.candidates import (
        smart_trade_joinquant_cross_signal_etf_fresh_unextended_entry_candidate as candidate,
    )

    missing = candidate.enrich_fresh_entry_snapshot(
        candidate_score("MISSING", macd_cross_up_age=None),
        pd.Series([9.5, 10.0, 11.0]),
    )

    assert candidate.filter_fresh_unextended_buy_candidates(
        [missing], held_codes=[]
    ) == []
