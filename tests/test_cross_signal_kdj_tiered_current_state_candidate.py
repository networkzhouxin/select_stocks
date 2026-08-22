# -*- coding: utf-8 -*-
"""Tests for the isolated current-session tiered KDJ state candidate."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys
from types import SimpleNamespace

import pytest


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
        from cross_signal_strategy.research import kdj_tiered_current_state_candidate
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("current-state tier candidate is not implemented: %s" % exc)
    return kdj_tiered_current_state_candidate


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
    }
    values.update(overrides)
    return values


def _adapter(scores_by_date, dates=None):
    module = _candidate_module()
    return module.KdjTieredCurrentStateScoreAdapter(
        DatedSignalAdapter(scores_by_date),
        trade_dates=dates or tuple(scores_by_date),
    )


@pytest.mark.parametrize(
    ("k_value", "buy_bonus", "sell_bonus"),
    [
        (20.0, 10.0, 0.0),
        (20.1, 5.0, 0.0),
        (30.0, 5.0, 0.0),
        (30.1, 0.0, 0.0),
        (69.9, 0.0, 0.0),
        (70.0, 0.0, 5.0),
        (79.9, 0.0, 5.0),
        (80.0, 0.0, 10.0),
    ],
)
def test_current_state_uses_the_frozen_tier_boundaries(
    k_value,
    buy_bonus,
    sell_bonus,
):
    date = "2019-01-08"
    candidate = _adapter({date: _official_score(date, k=k_value)}).score(
        "513100", date
    )

    assert candidate["buy_extreme_zone_score"] == buy_bonus
    assert candidate["sell_extreme_zone_score"] == sell_bonus
    assert candidate["buy_score"] == 50.0 + buy_bonus
    assert candidate["sell_score"] == 20.0 + sell_bonus


def test_prior_extreme_state_expires_immediately_when_current_k_is_neutral():
    dates = ("2019-01-08", "2019-01-09")
    scores = {
        dates[0]: _official_score(dates[0], k=19.0),
        dates[1]: _official_score(dates[1], k=50.0),
    }

    candidate = _adapter(scores, dates).score("513100", dates[1])

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["sell_extreme_zone_score"] == 0.0
    assert candidate["buy_score"] == 50.0
    assert candidate["sell_score"] == 20.0


def test_current_downside_continuation_blocks_current_buy_bonus_only():
    date = "2019-01-08"
    candidate = _adapter(
        {date: _official_score(date, k=19.0, downside_continuation=True)}
    ).score("513100", date)

    assert candidate["buy_extreme_zone_score"] == 0.0
    assert candidate["buy_score"] == 50.0


def test_training_runner_rejects_unapproved_data_roots():
    module = _candidate_module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_tiered_current_state_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
