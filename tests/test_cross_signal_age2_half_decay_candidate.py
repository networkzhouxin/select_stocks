# -*- coding: utf-8 -*-
"""Tests for the isolated bullish-cross age-2 half-decay candidate."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticSignalAdapter:
    def __init__(self, score):
        self._score = deepcopy(score)

    def score(self, code, current_date, return_reason=False):
        result = deepcopy(self._score)
        return (result, None) if return_reason else result


def _official_score(**overrides):
    values = {
        "code": "513100",
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi12_up_age": 2,
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi24_up_age": 1,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": True,
        "macd_cross_up_age": 2,
        "kdj_k_cross_up": True,
        "kdj_k_cross_up_age": 0,
        "kdj_j_cross_up": True,
        "kdj_j_cross_up_age": 2,
        "reversal_score": 45,
        "location_score": 10,
        "trend_score": 6,
        "volume_score": 4,
        "buy_score": 65,
        "buy_allowed": True,
        "sell_score": 28,
        "sell_reversal_score": 22,
        "sell_risk_score": 6,
        "macd_cross_down": True,
        "atr": 0.12,
        "close": 3.45,
        "unrelated_observation": {"nested": [1, 2, 3]},
    }
    values.update(overrides)
    return values


def _adapter(score):
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        Age2HalfDecaySignalAdapter,
    )

    return Age2HalfDecaySignalAdapter(StaticSignalAdapter(score))


def test_age2_bullish_crosses_receive_exactly_half_weight_while_age0_and_age1_stay_full():
    official = _official_score()

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["official_reversal_score"] == 45
    assert candidate["official_buy_score"] == 65
    assert candidate["age2_half_decay_penalty"] == pytest.approx(13.5)
    assert candidate["reversal_score"] == pytest.approx(31.5)
    assert candidate["buy_score"] == pytest.approx(51.5)


def test_age0_and_age1_bullish_crosses_are_identical_to_official_score():
    official = _official_score(
        rsi6_cross_rsi12_up_age=0,
        rsi6_cross_rsi24_up_age=1,
        macd_cross_up_age=1,
        kdj_k_cross_up_age=0,
        kdj_j_cross_up_age=1,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["age2_half_decay_penalty"] == 0
    assert candidate["reversal_score"] == official["reversal_score"]
    assert candidate["buy_score"] == official["buy_score"]


def test_mixed_rsi_direction_does_not_create_a_bullish_age2_penalty():
    official = _official_score(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi12_up_age=2,
        rsi6_cross_rsi24_up=False,
        rsi6_cross_rsi24_up_age=None,
        rsi6_cross_rsi12_down=True,
        reversal_score=0,
        location_score=-10,
        trend_score=0,
        volume_score=0,
        buy_score=0,
        macd_cross_up=False,
        macd_cross_up_age=None,
        kdj_k_cross_up=False,
        kdj_k_cross_up_age=None,
        kdj_j_cross_up=False,
        kdj_j_cross_up_age=None,
    )

    candidate = _adapter(official).score("513100", "2019-01-08")

    assert candidate["age2_half_decay_penalty"] == 0
    assert candidate["reversal_score"] == 0
    assert candidate["buy_score"] == 0


def test_active_bullish_cross_without_age_metadata_fails_closed():
    official = _official_score(macd_cross_up=True)
    del official["macd_cross_up_age"]

    with pytest.raises(ValueError, match="macd_cross_up_age"):
        _adapter(official).score("513100", "2019-01-08")


def test_candidate_preserves_official_snapshot_and_sell_side_without_mutation():
    official = _official_score()
    original = deepcopy(official)
    adapter = _adapter(official)

    candidate, reason = adapter.score("513100", "2019-01-08", return_reason=True)
    candidate["unrelated_observation"]["nested"].append(99)
    second = adapter.score("513100", "2019-01-08")

    assert reason is None
    assert official == original
    assert second["unrelated_observation"] == original["unrelated_observation"]
    for key in (
        "code",
        "current_date",
        "signal_date",
        "max_data_date",
        "location_score",
        "trend_score",
        "volume_score",
        "buy_allowed",
        "sell_score",
        "sell_reversal_score",
        "sell_risk_score",
        "macd_cross_down",
        "atr",
        "close",
    ):
        assert second[key] == original[key]


def _performance(**overrides):
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        Age2HalfDecayPerformance,
    )

    values = {
        "total_return": 1.00,
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
    return Age2HalfDecayPerformance(**values)


def _passing_candidate(**overrides):
    values = {
        "total_return": 1.05,
        "annualized_return": 0.26,
        "max_drawdown": 0.08,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "annual_returns": {2019: 0.21, 2020: 0.30, 2021: 0.15},
    }
    values.update(overrides)
    return _performance(**values)


def test_strict_gate_passes_only_when_returns_improve_and_all_other_metrics_are_not_worse():
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        evaluate_age2_half_decay_gate,
    )

    decision = evaluate_age2_half_decay_gate(
        _performance(),
        _passing_candidate(),
        changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
    )

    assert decision.passed
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("candidate_overrides", "reason"),
    [
        ({"total_return": 1.00}, "candidate total return does not improve"),
        ({"annualized_return": 0.25}, "candidate annualized return does not improve"),
        ({"max_drawdown": 0.081}, "candidate maximum drawdown worsens"),
        ({"sharpe_ratio": 1.99}, "candidate Sharpe ratio worsens"),
        ({"sortino_ratio": 2.99}, "candidate Sortino ratio worsens"),
        ({"win_rate": 0.549}, "candidate win rate worsens"),
        ({"profit_loss_ratio": 3.99}, "candidate profit/loss ratio worsens"),
        (
            {"annual_returns": {2019: 0.21, 2020: 0.299, 2021: 0.15}},
            "2020 candidate annual return worsens",
        ),
    ],
)
def test_strict_gate_rejects_each_independent_performance_regression(
    candidate_overrides,
    reason,
):
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        evaluate_age2_half_decay_gate,
    )

    decision = evaluate_age2_half_decay_gate(
        _performance(),
        _passing_candidate(**candidate_overrides),
        changed_days_by_year={2019: 1, 2020: 1, 2021: 1},
    )

    assert not decision.passed
    assert reason in decision.reasons


def test_strict_gate_rejects_a_year_without_changed_filled_order_days():
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        evaluate_age2_half_decay_gate,
    )

    decision = evaluate_age2_half_decay_gate(
        _performance(),
        _passing_candidate(),
        changed_days_by_year={2019: 1, 2020: 0, 2021: 1},
    )

    assert not decision.passed
    assert "2020 has no changed filled-order day" in decision.reasons


def _order(code, amount_delta, reason, filled=True):
    return SimpleNamespace(
        code=code,
        amount_delta=amount_delta,
        reason=reason,
        filled=filled,
    )


def _day(date, *orders):
    return SimpleNamespace(date=date, orders=list(orders), total_value=20000.0)


def test_filled_order_path_comparison_counts_changed_days_in_each_training_year():
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        compare_filled_order_paths,
    )

    baseline = [
        _day("2019-03-01", _order("513100.XSHG", 100, "buy_signal")),
        _day("2020-04-02", _order("510300.XSHG", 100, "buy_signal")),
        _day("2021-05-06", _order("513050.XSHG", -100, "signal_sell")),
    ]
    candidate = [
        _day("2019-03-01"),
        _day("2020-04-02", _order("510300.XSHG", 100, "buy_signal", filled=False)),
        _day("2021-05-06", _order("513050.XSHG", -100, "atr_stop")),
    ]

    changed = compare_filled_order_paths(baseline, candidate)

    assert changed.changed_days_by_year == {2019: 1, 2020: 1, 2021: 1}
    assert changed.changed_order_days == 3
    assert tuple(item.date for item in changed.decisions) == (
        "2019-03-01",
        "2020-04-02",
        "2021-05-06",
    )


def test_filled_order_path_comparison_rejects_different_dates_and_nontraining_dates():
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        compare_filled_order_paths,
    )

    with pytest.raises(ValueError, match="identical trading dates"):
        compare_filled_order_paths([_day("2019-01-02")], [_day("2019-01-03")])
    with pytest.raises(ValueError, match="outside 2019-2021"):
        compare_filled_order_paths([_day("2022-01-04")], [_day("2022-01-04")])


def test_markdown_report_records_frozen_change_metrics_gate_and_next_action():
    from cross_signal_strategy.research.age2_half_decay_candidate import (
        Age2HalfDecayComparisonReport,
        Age2HalfDecayGateDecision,
        FilledOrderPathComparison,
        format_age2_half_decay_comparison,
    )

    report = Age2HalfDecayComparisonReport(
        baseline_report=None,
        candidate_report=None,
        baseline=_performance(),
        candidate=_passing_candidate(),
        path=FilledOrderPathComparison(
            changed_order_days=4,
            changed_days_by_year={2019: 1, 2020: 2, 2021: 1},
            decisions=(),
        ),
        gate=Age2HalfDecayGateDecision(True, ()),
    )

    markdown = format_age2_half_decay_comparison(report)

    assert "2019-2021" in markdown
    assert "age 0/1" in markdown
    assert "age 2" in markdown
    assert "0.5" in markdown
    assert "Baseline" in markdown and "Candidate" in markdown
    assert "2019: 1" in markdown and "2020: 2" in markdown and "2021: 1" in markdown
    assert "PASS" in markdown
    assert "separate JoinQuant candidate" in markdown
