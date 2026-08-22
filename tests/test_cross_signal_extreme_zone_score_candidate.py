# -*- coding: utf-8 -*-
"""Tests for the isolated KDJ extreme-zone score candidate."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticSignalAdapter:
    def __init__(self, score):
        self._score = score

    def score(self, code, current_date, return_reason=False):
        result = deepcopy(self._score) if self._score is not None else None
        reason = None if result is not None else "no_data"
        return (result, reason) if return_reason else result


def _candidate_module():
    try:
        from cross_signal_strategy.research import extreme_zone_score_candidate
    except ImportError as exc:  # pragma: no cover - exercised only in TDD red phase
        pytest.fail("extreme-zone candidate is not implemented: %s" % exc)
    return extreme_zone_score_candidate


def _official_score(**overrides):
    values = {
        "code": "513100",
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "k": 50.0,
        "downside_continuation": False,
        "buy_score": 55.0,
        "sell_score": 25.0,
        "reversal_score": 25.0,
        "location_score": 15.0,
        "trend_score": 10.0,
        "volume_score": 5.0,
        "sell_reversal_score": 15.0,
        "sell_risk_score": 10.0,
        "buy_allowed": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
    }
    values.update(overrides)
    return values


def _adapter(score):
    module = _candidate_module()
    return module.ExtremeZoneScoreAdapter(StaticSignalAdapter(score))


def test_oversold_k_adds_five_to_unified_buy_score_without_a_gold_cross():
    official = _official_score(
        k=20.0,
        kdj_k_cross_up=False,
        kdj_j_cross_up=False,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["official_buy_score"] == 55.0
    assert candidate["buy_extreme_zone_score"] == 5.0
    assert candidate["buy_score"] == 60.0


def test_downside_continuation_blocks_only_the_oversold_buy_bonus():
    official = _official_score(k=19.0, downside_continuation=True)

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["buy_score"] == 55.0


def test_overbought_k_adds_five_to_the_same_final_sell_score_without_a_death_cross():
    official = _official_score(
        k=80.0,
        kdj_k_cross_down=False,
        kdj_j_cross_down=False,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["official_sell_score"] == 25.0
    assert candidate["sell_extreme_zone_score"] == 5.0
    assert candidate["sell_score"] == 30.0


def test_neutral_or_missing_k_does_not_change_either_score():
    for k_value in (50.0, None, float("nan")):
        candidate = _adapter(_official_score(k=k_value)).score(
            "513100", "2019-01-08"
        )

        assert candidate["buy_extreme_zone_score"] == 0.0
        assert candidate["sell_extreme_zone_score"] == 0.0
        assert candidate["buy_score"] == 55.0
        assert candidate["sell_score"] == 25.0


def test_candidate_preserves_t_minus_one_metadata_and_does_not_mutate_source():
    official = _official_score(k=18.0, nested={"values": [1]})
    original = deepcopy(official)
    adapter = _adapter(official)

    candidate, reason = adapter.score(
        "513100", "2019-01-08", return_reason=True
    )
    candidate["nested"]["values"].append(2)
    second = adapter.score("513100", "2019-01-08")

    assert reason is None
    assert official == original
    assert second["nested"] == original["nested"]
    assert second["signal_date"] == "2019-01-07"
    assert second["max_data_date"] == "2019-01-07"


def _performance(**overrides):
    module = _candidate_module()
    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.08,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "buy_count": 90,
        "sell_count": 88,
        "annual_returns": {2019: 0.20, 2020: 0.30, 2021: 0.15},
    }
    values.update(overrides)
    return module.ExtremeZonePerformance(**values)


def test_gate_requires_accuracy_improvement_without_material_damage():
    module = _candidate_module()
    baseline = _performance()
    candidate = _performance(
        total_return=0.97,
        annualized_return=0.245,
        max_drawdown=0.084,
        sharpe_ratio=1.96,
        sortino_ratio=2.94,
        win_rate=0.58,
        profit_loss_ratio=3.92,
        annual_returns={2019: 0.19, 2020: 0.29, 2021: 0.145},
    )

    decision = module.evaluate_extreme_zone_gate(
        baseline,
        candidate,
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert decision.passed
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("candidate_overrides", "reason"),
    [
        ({"win_rate": 0.55}, "candidate win rate does not improve"),
        ({"total_return": 0.949}, "candidate retains less than 95% of baseline return"),
        ({"max_drawdown": 0.086}, "candidate maximum drawdown worsens by more than 0.5pp"),
        ({"sharpe_ratio": 1.89}, "candidate Sharpe ratio worsens by more than 5%"),
        ({"sortino_ratio": 2.84}, "candidate Sortino ratio worsens by more than 5%"),
        ({"profit_loss_ratio": 3.79}, "candidate profit/loss ratio worsens by more than 5%"),
        (
            {"annual_returns": {2019: 0.20, 2020: 0.30, 2021: -0.01}},
            "2021 candidate annual return turns non-positive",
        ),
    ],
)
def test_gate_rejects_each_accuracy_or_damage_failure(candidate_overrides, reason):
    module = _candidate_module()
    passing_values = {
        "total_return": 1.01,
        "annualized_return": 0.255,
        "max_drawdown": 0.08,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.58,
        "profit_loss_ratio": 4.0,
        "annual_returns": {2019: 0.21, 2020: 0.31, 2021: 0.16},
    }
    passing_values.update(candidate_overrides)
    candidate = _performance(**passing_values)

    decision = module.evaluate_extreme_zone_gate(
        _performance(),
        candidate,
        changed_days_by_year={2019: 1, 2020: 1, 2021: 1},
    )

    assert not decision.passed
    assert reason in decision.reasons


def test_gate_requires_changed_filled_orders_in_every_training_year():
    module = _candidate_module()
    decision = module.evaluate_extreme_zone_gate(
        _performance(),
        _performance(total_return=1.01, win_rate=0.58),
        changed_days_by_year={2019: 1, 2020: 0, 2021: 1},
    )

    assert not decision.passed
    assert "2020 has no changed filled-order day" in decision.reasons


def test_training_runner_rejects_unapproved_data_roots():
    module = _candidate_module()
    loader = SimpleNamespace(root="G:/unapproved/training")

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_extreme_zone_training_ab(loader=loader)


def test_binding_summary_distinguishes_bonus_events_from_threshold_crossings():
    module = _candidate_module()
    scores = [
        _official_score(code="BUY_BIND", k=20.0, buy_score=55.0, sell_score=0.0),
        _official_score(code="BUY_NO_BIND", k=19.0, buy_score=54.0, sell_score=0.0),
        _official_score(code="BUY_BLOCKED", k=18.0, buy_score=55.0, downside_continuation=True),
        _official_score(code="SELL_BIND", k=80.0, buy_score=0.0, sell_score=25.0),
        _official_score(code="SELL_NO_BIND", k=81.0, buy_score=0.0, sell_score=24.0),
    ]

    summary = module.summarize_extreme_zone_bindings(scores)

    assert summary.oversold_bonus_events == 2
    assert summary.overbought_bonus_events == 2
    assert summary.buy_threshold_crossings == 1
    assert summary.sell_threshold_crossings == 1
    assert summary.buy_crossings_by_code == {"BUY_BIND": 1}
    assert summary.sell_crossings_by_code == {"SELL_BIND": 1}
