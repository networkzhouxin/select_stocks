# -*- coding: utf-8 -*-
"""Tests for the isolated tiered three-session KDJ state candidate."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys

import pytest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatedSignalAdapter:
    def __init__(self, scores_by_date):
        self._scores_by_date = scores_by_date

    def score(self, code, current_date, return_reason=False):
        score = self._scores_by_date.get(str(current_date))
        result = deepcopy(score) if score is not None else None
        reason = None if result is not None else "no_data"
        return (result, reason) if return_reason else result


def _candidate_module():
    try:
        from cross_signal_strategy.research import kdj_tiered_persistence_candidate
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("tiered-persistence candidate is not implemented: %s" % exc)
    return kdj_tiered_persistence_candidate


def _official_score(current_date, **overrides):
    values = {
        "code": "513100",
        "current_date": current_date,
        "signal_date": current_date,
        "max_data_date": current_date,
        "k": 50.0,
        "downside_continuation": False,
        "buy_score": 50.0,
        "sell_score": 20.0,
        "reversal_score": 25.0,
        "location_score": 15.0,
        "trend_score": 10.0,
        "volume_score": 5.0,
        "sell_reversal_score": 15.0,
        "sell_risk_score": 5.0,
    }
    values.update(overrides)
    return values


def _adapter(scores_by_date, dates=None):
    module = _candidate_module()
    trade_dates = dates or tuple(scores_by_date)
    return module.KdjTieredPersistenceScoreAdapter(
        DatedSignalAdapter(scores_by_date),
        trade_dates=trade_dates,
    )


@pytest.mark.parametrize(
    ("k_value", "expected_bonus"),
    [(20.0, 10.0), (20.1, 5.0), (30.0, 5.0), (30.1, 0.0)],
)
def test_buy_state_uses_fixed_strong_and_near_extreme_tiers(k_value, expected_bonus):
    date = "2019-01-08"
    candidate = _adapter({date: _official_score(date, k=k_value)}).score(
        "513100", date
    )

    assert candidate["buy_extreme_zone_score"] == expected_bonus
    assert candidate["buy_score"] == 50.0 + expected_bonus


@pytest.mark.parametrize(
    ("k_value", "expected_bonus"),
    [(69.9, 0.0), (70.0, 5.0), (79.9, 5.0), (80.0, 10.0)],
)
def test_sell_state_uses_fixed_near_and_strong_extreme_tiers(k_value, expected_bonus):
    date = "2019-01-08"
    candidate = _adapter({date: _official_score(date, k=k_value)}).score(
        "513100", date
    )

    assert candidate["sell_extreme_zone_score"] == expected_bonus
    assert candidate["sell_score"] == 20.0 + expected_bonus


def test_strong_state_persists_for_exactly_three_decision_sessions():
    dates = ("2019-01-08", "2019-01-09", "2019-01-10", "2019-01-11")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=50.0),
        dates[2]: _official_score(dates[2], k=50.0),
        dates[3]: _official_score(dates[3], k=50.0),
    }
    adapter = _adapter(scores, dates)

    assert adapter.score("513100", dates[2])["buy_extreme_zone_score"] == 10.0
    assert adapter.score("513100", dates[3])["buy_extreme_zone_score"] == 0.0


def test_same_direction_retention_takes_maximum_tier_without_accumulating():
    dates = ("2019-01-08", "2019-01-09", "2019-01-10")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=25.0),
        dates[2]: _official_score(dates[2], k=28.0),
    }

    candidate = _adapter(scores, dates).score("513100", dates[2])

    assert candidate["buy_extreme_zone_score"] == 10.0
    assert candidate["buy_score"] == 60.0


def test_most_recent_opposite_direction_replaces_older_stronger_direction():
    dates = ("2019-01-08", "2019-01-09", "2019-01-10")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=50.0),
        dates[2]: _official_score(dates[2], k=75.0),
    }

    candidate = _adapter(scores, dates).score("513100", dates[2])

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["sell_extreme_zone_score"] == 5.0


def test_current_downside_continuation_blocks_retained_buy_state():
    dates = ("2019-01-08", "2019-01-09", "2019-01-10")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=50.0),
        dates[2]: _official_score(dates[2], k=50.0, downside_continuation=True),
    }

    candidate = _adapter(scores, dates).score("513100", dates[2])

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["buy_score"] == 50.0


def test_candidate_preserves_current_t_minus_one_metadata_and_source_data():
    dates = ("2019-01-08", "2019-01-09")
    current = _official_score(
        dates[1],
        signal_date="2019-01-08",
        max_data_date="2019-01-08",
        nested={"values": [1]},
    )
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: current,
    }
    original = deepcopy(scores)
    adapter = _adapter(scores, dates)

    candidate, reason = adapter.score("513100", dates[1], return_reason=True)
    candidate["nested"]["values"].append(2)
    second = adapter.score("513100", dates[1])

    assert reason is None
    assert scores == original
    assert second["nested"] == {"values": [1]}
    assert second["signal_date"] == "2019-01-08"
    assert second["max_data_date"] == "2019-01-08"


def test_unknown_decision_date_does_not_borrow_future_or_unordered_state():
    dates = ("2019-01-08", "2019-01-09")
    adapter = _adapter(
        {
            dates[0]: _official_score(dates[0], k=19.0),
            dates[1]: _official_score(dates[1], k=50.0),
        },
        dates,
    )

    score, reason = adapter.score("513100", "2019-01-07", return_reason=True)

    assert score is None
    assert reason == "no_data"


def test_training_runner_rejects_unapproved_data_roots():
    module = _candidate_module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_tiered_persistence_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
