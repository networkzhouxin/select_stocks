# -*- coding: utf-8 -*-
"""Tests for the frozen fresh, unextended fast-entry candidate."""

from __future__ import annotations

import pathlib
import sys
import types

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        item = self.scores.get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        result = dict(item)
        return (result, None) if return_reason else result


class FakeCausalSourceAdapter(FakeSignalAdapter):
    def __init__(self, scores, frame):
        super().__init__(scores)
        self.frame = frame

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), "2019-01-04"


def raw_candidate(code="FRESH", buy_score=55, **overrides):
    item = {
        "code": code,
        "signal_date": "2019-01-04",
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
    item.update(overrides)
    return item


def causal_frame():
    return pd.DataFrame(
        {
            "date": ["2019-01-02", "2019-01-03", "2019-01-04"],
            "close": [9.5, 10.0, 11.0],
        }
    )


def test_enrichment_uses_earliest_contributing_cross_and_t1_close():
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        enrich_fresh_entry_context,
    )

    enriched = enrich_fresh_entry_context(raw_candidate(), causal_frame())

    assert enriched["fresh_entry_earliest_cross_age"] == 1
    assert enriched["fresh_entry_earliest_cross_date"] == "2019-01-03"
    assert enriched["fresh_entry_cross_close"] == 10.0
    assert enriched["fresh_entry_extension_atr"] == 1.0


def test_enrichment_rejects_any_row_later_than_signal_date():
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        enrich_fresh_entry_context,
    )

    frame = pd.concat(
        [
            causal_frame(),
            pd.DataFrame({"date": ["2019-01-05"], "close": [99.0]}),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="later than signal_date"):
        enrich_fresh_entry_context(raw_candidate(), frame)


def test_signal_adapter_enriches_the_real_score_from_its_causal_frame():
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        FreshUnextendedSignalAdapter,
    )

    source = FakeCausalSourceAdapter(
        {"FRESH": raw_candidate("FRESH")},
        causal_frame(),
    )
    adapter = FreshUnextendedSignalAdapter(source)

    score, reason = adapter.score("FRESH", "2019-01-07", return_reason=True)

    assert reason is None
    assert score["max_data_date"] == "2019-01-04"
    assert score["fresh_entry_earliest_cross_age"] == 1
    assert score["fresh_entry_cross_close"] == 10.0
    assert score["fresh_entry_extension_atr"] == 1.0


def test_filter_accepts_only_fresh_unextended_50_59_scores():
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        enrich_fresh_entry_context,
        filter_fresh_unextended_buy_candidates,
    )

    valid = enrich_fresh_entry_context(raw_candidate("VALID"), causal_frame())
    old = enrich_fresh_entry_context(
        raw_candidate(
            "OLD",
            rsi6_cross_rsi12_up_age=2,
            rsi6_cross_rsi24_up_age=0,
            macd_cross_up_age=0,
            kdj_k_cross_up_age=0,
        ),
        causal_frame(),
    )
    extended = dict(valid, code="EXTENDED", fresh_entry_extension_atr=1.0001)
    weak_reversal = dict(valid, code="WEAK_REV", reversal_score=34)
    below_band = dict(valid, code="BELOW", buy_score=49)
    primary = dict(valid, code="PRIMARY", buy_score=60)
    overheated = dict(valid, code="HOT", buy_allowed=False)

    kept = filter_fresh_unextended_buy_candidates(
        [valid, old, extended, weak_reversal, below_band, primary, overheated],
        held_codes=[],
    )

    assert [item["code"] for item in kept] == ["VALID"]


def test_filter_rejects_score_when_any_contributing_cross_age_is_missing():
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        enrich_fresh_entry_context,
        filter_fresh_unextended_buy_candidates,
    )

    missing_age = enrich_fresh_entry_context(
        raw_candidate("MISSING", macd_cross_up_age=None),
        causal_frame(),
    )

    assert filter_fresh_unextended_buy_candidates(
        [missing_age], held_codes=[]
    ) == []


def test_planner_places_primary_before_fresh_candidate_without_displacement():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        FreshUnextendedEntryOrderPlanner,
        enrich_fresh_entry_context,
    )

    fresh = enrich_fresh_entry_context(raw_candidate("FRESH"), causal_frame())
    scores = {
        "PRIMARY": dict(fresh, code="PRIMARY", buy_score=70),
        "FRESH": fresh,
    }
    planner = FreshUnextendedEntryOrderPlanner(
        FakeSignalAdapter(scores),
        etf_pool=["PRIMARY", "FRESH"],
    )

    orders = planner.plan_orders("2019-01-07", "2019-01-04", LocalBroker(20000.0))

    assert [(order["code"], order["reason"]) for order in orders] == [
        ("PRIMARY", "buy_signal"),
        ("FRESH", "fresh_unextended_buy_signal"),
    ]


def test_planner_never_displaces_three_primary_candidates():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.fresh_unextended_entry_candidate import (
        FreshUnextendedEntryOrderPlanner,
        enrich_fresh_entry_context,
    )

    fresh = enrich_fresh_entry_context(raw_candidate("FRESH"), causal_frame())
    scores = {
        "P1": dict(fresh, code="P1", buy_score=80),
        "P2": dict(fresh, code="P2", buy_score=75),
        "P3": dict(fresh, code="P3", buy_score=70),
        "FRESH": fresh,
    }
    planner = FreshUnextendedEntryOrderPlanner(
        FakeSignalAdapter(scores),
        etf_pool=["P1", "P2", "P3", "FRESH"],
    )

    orders = planner.plan_orders("2019-01-07", "2019-01-04", LocalBroker(20000.0))

    assert [order["code"] for order in orders] == ["P1", "P2", "P3"]
    assert all(order["reason"] == "buy_signal" for order in orders)
