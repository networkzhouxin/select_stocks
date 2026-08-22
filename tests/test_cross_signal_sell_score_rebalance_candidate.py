# -*- coding: utf-8 -*-
"""Tests for the isolated sell-score rebalance candidate."""

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


def _module():
    try:
        from cross_signal_strategy.research import sell_score_rebalance_candidate
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("sell-score rebalance candidate is not implemented: %s" % exc)
    return sell_score_rebalance_candidate


def _score(date, **overrides):
    values = {
        "code": "512100",
        "current_date": date,
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "k": 50.0,
        "buy_score": 40.0,
        "sell_score": 0.0,
        "reversal_score": 0.0,
        "sell_reversal_score": 0.0,
        "sell_risk_score": 0.0,
        "buy_allowed": True,
        "downside_continuation": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_down": False,
        "kdj_k_cross_down": False,
        "kdj_j_cross_down": False,
        "close_below_boll_mid": False,
        "fell_back_inside_boll": False,
        "far_above_ma20_and_rsi6_down": False,
        "close_below_ma20": False,
        "close_below_falling_ma10": False,
        "ma20_slope_non_negative": False,
        "adx": 10.0,
        "plus_di": 10.0,
        "minus_di": 20.0,
    }
    values.update(overrides)
    return values


def _adapter(score):
    module = _module()
    date = str(score["current_date"])
    return module.SellScoreRebalanceAdapter(
        DatedSignalAdapter({date: score}),
        trade_dates=(date,),
    )


def test_rebalanced_score_uses_capped_indicator_families_and_kdj_state_bonus():
    date = "2019-01-08"
    candidate = _adapter(_score(
        date,
        k=75.0,
        sell_score=45.0,
        rsi6_cross_rsi12_down=True,
        rsi6_cross_rsi24_down=True,
        macd_cross_down=True,
        kdj_k_cross_down=True,
        kdj_j_cross_down=True,
        close_below_boll_mid=True,
        close_below_ma20=True,
        close_below_falling_ma10=True,
        downside_continuation=True,
    )).score("512100", date)

    assert candidate["official_sell_score"] == pytest.approx(45.0)
    assert candidate["pre_rebalance_sell_score"] == pytest.approx(50.0)
    assert candidate["sell_reversal_score"] == pytest.approx(36.0)
    assert candidate["sell_risk_score"] == pytest.approx(12.0)
    assert candidate["sell_extreme_zone_score"] == pytest.approx(5.0)
    assert candidate["sell_score"] == pytest.approx(53.0)


def test_correlated_price_weakness_takes_only_the_highest_bucket():
    date = "2019-01-08"
    candidate = _adapter(_score(
        date,
        close_below_boll_mid=True,
        fell_back_inside_boll=True,
        far_above_ma20_and_rsi6_down=True,
        close_below_ma20=True,
        close_below_falling_ma10=True,
    )).score("512100", date)

    assert candidate["sell_reversal_score"] == 0.0
    assert candidate["sell_risk_score"] == 12.0
    assert candidate["sell_score"] == 12.0


def test_two_rsi_down_crosses_plus_ma20_break_reach_thirty_without_macd():
    date = "2019-01-08"
    candidate = _adapter(_score(
        date,
        rsi6_cross_rsi12_down=True,
        rsi6_cross_rsi24_down=True,
        close_below_ma20=True,
    )).score("512100", date)

    assert not candidate["macd_cross_down"]
    assert candidate["sell_reversal_score"] == 20.0
    assert candidate["sell_risk_score"] == 10.0
    assert candidate["sell_score"] == 30.0


def test_rebalanced_threshold_keeps_official_confirmation_and_adx_guard():
    module = _module()
    date = "2019-01-08"
    protected = _adapter(_score(
        date,
        rsi6_cross_rsi12_down=True,
        rsi6_cross_rsi24_down=True,
        kdj_k_cross_down=True,
        close_below_boll_mid=True,
        adx=35.0,
        plus_di=30.0,
        minus_di=10.0,
        ma20_slope_non_negative=True,
    )).score("512100", date)
    severe_break = _adapter(_score(
        date,
        rsi6_cross_rsi12_down=True,
        rsi6_cross_rsi24_down=True,
        close_below_ma20=True,
        adx=35.0,
        plus_di=30.0,
        minus_di=10.0,
        ma20_slope_non_negative=True,
    )).score("512100", date)

    assert protected["sell_score"] == 31.0
    assert not module.strategy.should_force_sell(protected)
    assert severe_break["sell_score"] == 30.0
    assert module.strategy.should_force_sell(severe_break)


def _performance(**overrides):
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
        "buy_count": 90,
        "sell_count": 88,
        "annual_returns": {2019: 0.2, 2020: 0.3, 2021: 0.1},
    }
    values.update(overrides)
    return ExtremeZonePerformance(**values)


def test_gate_requires_accuracy_above_current_and_official_paths():
    module = _module()
    official = _performance(win_rate=0.56)
    current = _performance(total_return=0.96, win_rate=0.54)
    candidate = _performance(total_return=0.97, win_rate=0.57)
    official_stress = _performance(total_return=0.80, win_rate=0.51)
    candidate_stress = _performance(total_return=0.77, win_rate=0.52)

    passed = module.evaluate_sell_score_rebalance_gate(
        official,
        current,
        candidate,
        official_stress,
        candidate_stress,
        {2019: 1, 2020: 1, 2021: 1},
    )
    failed = module.evaluate_sell_score_rebalance_gate(
        official,
        current,
        _performance(total_return=0.97, win_rate=0.55),
        official_stress,
        candidate_stress,
        {2019: 1, 2020: 1, 2021: 1},
    )

    assert passed.passed
    assert not failed.passed
    assert any("win rate" in reason for reason in failed.reasons)


def test_training_runner_rejects_unapproved_data_root():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_sell_score_rebalance_training_comparison(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )


def _trade(code, buy_date, sell_date, return_pct):
    from cross_signal_strategy.research.trade_diagnostics import (
        ClosedTradeDiagnostic,
    )

    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason="signal_sell",
        amount=100,
        buy_price=10.0,
        sell_price=10.0 * (1.0 + return_pct / 100.0),
        pnl=return_pct * 10.0,
        return_pct=return_pct,
    )


def test_target_attribution_pairs_same_entry_even_when_exit_date_changes():
    module = _module()
    current = [
        _trade("512100", "2019-09-30", "2019-10-21", -1.49),
        _trade("513880", "2021-03-04", "2021-03-23", -0.48),
    ]
    candidate = [
        _trade("512100", "2019-09-30", "2019-10-17", 0.74),
        _trade("513880", "2021-03-04", "2021-03-23", -0.48),
    ]

    pairs = module.select_target_trade_pairs(
        current,
        candidate,
        (("512100", "2019-09-30"), ("513880", "2021-03-04")),
    )

    assert pairs[0].candidate.sell_date == "2019-10-17"
    assert pairs[0].return_delta_pct == pytest.approx(2.23)
    assert pairs[1].return_delta_pct == pytest.approx(0.0)


def test_target_attribution_rejects_missing_predeclared_entry():
    module = _module()

    with pytest.raises(ValueError, match="missing target entry"):
        module.select_target_trade_pairs(
            [],
            [],
            (("512100", "2019-09-30"),),
        )
