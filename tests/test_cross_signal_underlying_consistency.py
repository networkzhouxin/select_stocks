# -*- coding: utf-8 -*-
"""Tests for the one pre-registered underlying-direction observation."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _quality_row(code="513100", buy_date="2020-01-06", return_pct=5.0):
    from cross_signal_strategy.research.trade_quality_ledger import TradeQualityRow

    return TradeQualityRow(
        code=code,
        buy_date=buy_date,
        sell_date="2020-01-20",
        sell_reason="signal_sell",
        market_group="qdii",
        realized_return_pct=return_pct,
        holding_trade_days=10,
        holding_mfe=0.08,
        holding_mae=-0.03,
        entry_mfe={5: 0.05, 10: 0.08},
        entry_mae={5: -0.02, 10: -0.03},
        first_profitable_close_offset=1,
        first_atr_barrier="up_first",
        post_sell_returns={5: 0.01, 10: 0.02},
    )


def _index_frame(latest_close=101.0, code="513100", source_id="NDX"):
    return pd.DataFrame({
        "etf_code": [code, code],
        "source_id": [source_id, source_id],
        "session_date": ["2020-01-02", "2020-01-03"],
        "available_at": [
            "2020-01-03T06:15:00+08:00",
            "2020-01-04T06:15:00+08:00",
        ],
        "close": [100.0, latest_close],
        "is_final": [True, True],
    })


def test_report_uses_exact_positive_sign_and_carries_trade_quality_metrics():
    from cross_signal_strategy.research.underlying_consistency import (
        build_underlying_consistency_report,
    )

    rows = [
        _quality_row(return_pct=5.0),
        _quality_row(code="159915", return_pct=99.0),
    ]
    report = build_underlying_consistency_report(
        rows,
        history_lookup=lambda code, decision_at: _index_frame(),
    )

    assert report.targeted_trades == 1
    assert report.covered_trades == 1
    observation = report.observations[0]
    assert observation.group == "confirmed"
    assert observation.underlying_return == pytest.approx(0.01)
    assert observation.realized_return_pct == pytest.approx(5.0)
    assert observation.holding_mfe == pytest.approx(0.08)
    assert observation.holding_mae == pytest.approx(-0.03)
    assert observation.first_atr_barrier == "up_first"


@pytest.mark.parametrize(("latest_close", "group"), [(100.0, "unconfirmed"), (99.0, "unconfirmed")])
def test_zero_and_negative_underlying_returns_are_unconfirmed(latest_close, group):
    from cross_signal_strategy.research.underlying_consistency import (
        build_underlying_consistency_report,
    )

    report = build_underlying_consistency_report(
        [_quality_row()],
        history_lookup=lambda code, decision_at: _index_frame(latest_close=latest_close),
    )

    assert report.observations[0].group == group


def test_missing_point_in_time_history_reduces_coverage_without_guessing():
    from cross_signal_strategy.research.underlying_consistency import (
        build_underlying_consistency_report,
    )

    report = build_underlying_consistency_report(
        [_quality_row()],
        history_lookup=lambda code, decision_at: pd.DataFrame(),
    )

    assert report.targeted_trades == 1
    assert report.covered_trades == 0
    assert report.missing_trades == 1
    assert report.observations == ()
    assert report.gate.passed is False
    assert any("coverage" in reason for reason in report.gate.reasons)


def test_observation_rejects_validation_period_trade_rows():
    from cross_signal_strategy.research.underlying_consistency import (
        build_underlying_consistency_report,
    )

    with pytest.raises(ValueError, match="2019-2021 training window"):
        build_underlying_consistency_report(
            [_quality_row(buy_date="2022-01-04")],
            history_lookup=lambda code, decision_at: _index_frame(),
        )


def _stats(count, win_rate, mean_return):
    from cross_signal_strategy.research.underlying_consistency import DirectionStats

    return DirectionStats(
        count=count,
        win_rate=win_rate,
        mean_return_pct=mean_return,
        mean_holding_mfe=0.08,
        mean_holding_mae=-0.03,
        up_first_rate=0.60,
        down_first_rate=0.25,
    )


def test_candidate_gate_is_locked_to_coverage_annual_and_cross_etf_consistency():
    from cross_signal_strategy.research.underlying_consistency import (
        evaluate_underlying_candidate_gate,
    )

    confirmed = _stats(20, 0.65, 4.0)
    unconfirmed = _stats(15, 0.45, 1.0)
    by_year = {
        year: {"confirmed": _stats(6, 0.67, 3.0), "unconfirmed": _stats(5, 0.40, 1.0)}
        for year in (2019, 2020, 2021)
    }
    by_code = {
        code: {"confirmed": _stats(3, 0.67, 3.0), "unconfirmed": _stats(2, 0.50, 1.0)}
        for code in ("513100", "513500", "513050")
    }

    passed = evaluate_underlying_candidate_gate(
        targeted_trades=37,
        covered_trades=35,
        aggregate={"confirmed": confirmed, "unconfirmed": unconfirmed},
        by_year=by_year,
        by_code=by_code,
    )
    failed = evaluate_underlying_candidate_gate(
        targeted_trades=37,
        covered_trades=35,
        aggregate={"confirmed": confirmed, "unconfirmed": unconfirmed},
        by_year={
            **by_year,
            2021: {"confirmed": _stats(6, 0.40, 0.5), "unconfirmed": _stats(5, 0.60, 2.0)},
        },
        by_code=by_code,
    )

    assert passed.passed is True
    assert failed.passed is False
    assert any("2021" in reason for reason in failed.reasons)


def test_candidate_gate_requires_three_cross_etf_comparisons():
    from cross_signal_strategy.research.underlying_consistency import (
        evaluate_underlying_candidate_gate,
    )

    decision = evaluate_underlying_candidate_gate(
        targeted_trades=37,
        covered_trades=35,
        aggregate={"confirmed": _stats(20, 0.65, 4.0), "unconfirmed": _stats(15, 0.45, 1.0)},
        by_year={
            year: {"confirmed": _stats(6, 0.67, 3.0), "unconfirmed": _stats(5, 0.40, 1.0)}
            for year in (2019, 2020, 2021)
        },
        by_code={
            code: {"confirmed": _stats(3, 0.67, 3.0), "unconfirmed": _stats(2, 0.50, 1.0)}
            for code in ("513100", "513500")
        },
    )

    assert decision.passed is False
    assert any("three ETF" in reason for reason in decision.reasons)
