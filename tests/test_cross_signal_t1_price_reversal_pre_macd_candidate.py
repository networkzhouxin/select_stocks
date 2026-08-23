# -*- coding: utf-8 -*-
"""Behavior tests for the frozen T-1 reversal/pre-MACD entry candidate."""

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
        score = self.scores.get(code)
        if score is None:
            return (None, "no_data") if return_reason else None
        result = dict(score)
        return (result, None) if return_reason else result


class FakeCausalSource(FakeSignalAdapter):
    def __init__(self, scores, frame, signal_date="2019-01-04"):
        super().__init__(scores)
        self.frame = frame
        self.signal_date = signal_date

    def load_signal_frame(self, code, current_date):
        return self.frame.copy(), self.signal_date


def score(code="ALT", buy_score=41, **overrides):
    result = {
        "code": code,
        "signal_date": "2019-01-04",
        "buy_allowed": True,
        "buy_score": buy_score,
        "sell_score": 0,
        "reversal_score": 30,
        "trend_score": 6,
        "volume_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "downside_continuation": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "macd_cross_up": False,
        "t1_price_reversal_context_complete": True,
        "t1_low_not_lower_than_t2": True,
        "t1_close_above_t2_high": True,
        "t1_price_reversal_confirmed": True,
    }
    result.update(overrides)
    return result


def reversal_frame(t1_low=9.5, t1_close=10.6, t2_low=9.5, t2_high=10.5):
    return pd.DataFrame(
        {
            "date": ["2019-01-02", "2019-01-03", "2019-01-04"],
            "low": [9.8, t2_low, t1_low],
            "high": [10.2, t2_high, 10.8],
            "close": [10.0, 10.1, t1_close],
        }
    )


def test_enrichment_uses_only_t1_and_t2_and_does_not_change_scores():
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        enrich_t1_price_reversal_context,
    )

    raw = score(buy_score=41, sell_score=17)
    enriched = enrich_t1_price_reversal_context(raw, reversal_frame())

    assert enriched["buy_score"] == 41
    assert enriched["sell_score"] == 17
    assert enriched["t1_price_reversal_context_complete"] is True
    assert enriched["t2_date"] == "2019-01-03"
    assert enriched["t1_date"] == "2019-01-04"
    assert enriched["t1_low_not_lower_than_t2"] is True
    assert enriched["t1_close_above_t2_high"] is True
    assert enriched["t1_price_reversal_confirmed"] is True


def test_enrichment_rejects_future_rows_and_requires_two_completed_rows():
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        enrich_t1_price_reversal_context,
    )

    future = pd.concat(
        [
            reversal_frame(),
            pd.DataFrame(
                {
                    "date": ["2019-01-07"],
                    "low": [1.0],
                    "high": [99.0],
                    "close": [99.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="later than signal_date"):
        enrich_t1_price_reversal_context(score(), future)
    with pytest.raises(ValueError, match="at least two"):
        enrich_t1_price_reversal_context(score(), reversal_frame().tail(1))


def test_adapter_enriches_official_score_from_matching_causal_frame():
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        T1PriceReversalSignalAdapter,
    )

    adapter = T1PriceReversalSignalAdapter(
        FakeCausalSource({"ALT": score()}, reversal_frame())
    )
    enriched, reason = adapter.score("ALT", "2019-01-07", return_reason=True)

    assert reason is None
    assert enriched["signal_date"] == "2019-01-04"
    assert enriched["max_data_date"] == "2019-01-04"
    assert enriched["t1_price_reversal_confirmed"] is True


def test_filter_requires_both_oscillator_groups_no_macd_and_price_confirmation():
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        filter_t1_price_reversal_pre_macd_candidates,
    )

    valid = score("VALID")
    cases = [
        valid,
        score("PRIMARY", buy_score=60),
        score("NO_RSI", rsi6_cross_rsi12_up=False),
        score("NO_KDJ", kdj_k_cross_up=False),
        score("HAS_MACD", macd_cross_up=True),
        score("LOWER_LOW", t1_low_not_lower_than_t2=False),
        score("NO_BREAK", t1_close_above_t2_high=False),
        score("FALLING", downside_continuation=True),
        score("HOT", buy_allowed=False),
        score("SELL_RISK", sell_score=30),
        score("NO_LOCATION", close_between_boll_lower_mid=False),
    ]

    kept = filter_t1_price_reversal_pre_macd_candidates(cases, held_codes=[])

    assert [item["code"] for item in kept] == ["VALID"]


def test_planner_places_official_primary_first_then_fills_leftover_slot():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        T1PriceReversalPreMacdOrderPlanner,
    )

    scores = {
        "PRIMARY": score("PRIMARY", buy_score=70),
        "ALT": score("ALT", buy_score=41),
    }
    planner = T1PriceReversalPreMacdOrderPlanner(
        FakeSignalAdapter(scores), etf_pool=["PRIMARY", "ALT"]
    )

    orders = planner.plan_orders(
        "2019-01-07", "2019-01-04", LocalBroker(20000.0)
    )

    assert [(item["code"], item["reason"]) for item in orders] == [
        ("PRIMARY", "buy_signal"),
        ("ALT", "t1_price_reversal_pre_macd_buy"),
    ]
    snapshot = planner.entry_score_snapshots[("2019-01-07", "ALT")]
    assert snapshot["entry_channel"] == "t1_price_reversal_pre_macd"
    assert snapshot["buy_score"] == 41


def test_planner_never_displaces_three_official_primary_candidates():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        T1PriceReversalPreMacdOrderPlanner,
    )

    scores = {
        "P1": score("P1", buy_score=80),
        "P2": score("P2", buy_score=75),
        "P3": score("P3", buy_score=70),
        "ALT": score("ALT", buy_score=41),
    }
    planner = T1PriceReversalPreMacdOrderPlanner(
        FakeSignalAdapter(scores), etf_pool=["P1", "P2", "P3", "ALT"]
    )

    orders = planner.plan_orders(
        "2019-01-07", "2019-01-04", LocalBroker(20000.0)
    )

    assert [item["code"] for item in orders] == ["P1", "P2", "P3"]
    assert all(item["reason"] == "buy_signal" for item in orders)


def performance(**overrides):
    from cross_signal_strategy.research.extreme_zone_score_candidate import (
        ExtremeZonePerformance,
    )

    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.06,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "buy_count": 10,
        "sell_count": 9,
        "annual_returns": {2019: 0.2, 2020: 0.3, 2021: 0.1},
    }
    values.update(overrides)
    return ExtremeZonePerformance(**values)


def test_frozen_gate_requires_materiality_accuracy_payoff_and_stress_guards():
    from cross_signal_strategy.research.t1_price_reversal_pre_macd_candidate import (
        evaluate_t1_price_reversal_gate,
    )

    baseline = performance()
    passing = performance(win_rate=0.56)
    assert evaluate_t1_price_reversal_gate(
        baseline,
        passing,
        baseline,
        performance(win_rate=0.55),
        direct_fill_count=3,
        direct_fill_years=(2019, 2021),
    ).passed is True

    rejected = evaluate_t1_price_reversal_gate(
        baseline,
        performance(total_return=0.94, win_rate=0.54),
        baseline,
        performance(total_return=0.94, win_rate=0.54),
        direct_fill_count=2,
        direct_fill_years=(2019,),
    )
    assert rejected.passed is False
    assert any("fewer than 3" in reason for reason in rejected.reasons)
    assert any("fewer than 2 training years" in reason for reason in rejected.reasons)
    assert any("win rate does not improve" in reason for reason in rejected.reasons)
    assert any("double-friction" in reason for reason in rejected.reasons)
