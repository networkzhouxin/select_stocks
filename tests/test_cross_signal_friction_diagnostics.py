# -*- coding: utf-8 -*-
"""Tests for cross-signal training friction decomposition."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSourceAdapter:
    def __init__(self):
        self.calls = []

    def score(self, code, current_date, return_reason=False):
        self.calls.append((str(code), str(current_date), bool(return_reason)))
        if code == "MISSING":
            return (None, "no_data") if return_reason else None
        score = {"code": code, "buy_score": int(str(current_date)[-2:])}
        return (score, None) if return_reason else score


def baseline_report(end_value, max_drawdown=0.05, buy_count=10, sell_count=9):
    from cross_signal_strategy.research.baseline_report import BaselineReport

    return BaselineReport(
        start_date="2019-01-02",
        end_date="2021-12-31",
        trading_days=730,
        start_value=20000.0,
        end_value=float(end_value),
        total_return=float(end_value) / 20000.0 - 1.0,
        annualized_return=0.0,
        max_drawdown=max_drawdown,
        daily_win_rate=0.5,
        annualized_volatility=0.1,
        sharpe_ratio=1.0,
        sortino_ratio=1.5,
        buy_count=buy_count,
        sell_count=sell_count,
        closed_trade_count=sell_count,
        win_rate=0.5,
        profit_loss_ratio=2.0,
        average_exposure=0.7,
    )


def test_precomputed_signal_adapter_calls_source_once_and_returns_defensive_copies():
    from cross_signal_strategy.research.friction_diagnostics import PrecomputedSignalAdapter

    source = FakeSourceAdapter()
    cached = PrecomputedSignalAdapter.from_source(
        source,
        trade_dates=["2019-01-02", "2019-01-03"],
        codes=["AAA", "MISSING"],
    )

    first, first_reason = cached.score("AAA", "2019-01-02", return_reason=True)
    first["buy_score"] = 999
    second, second_reason = cached.score("AAA", "2019-01-02", return_reason=True)
    missing, missing_reason = cached.score("MISSING", "2019-01-03", return_reason=True)

    assert len(source.calls) == 4
    assert first_reason is None
    assert second_reason is None
    assert second["buy_score"] == 2
    assert missing is None
    assert missing_reason == "no_data"


def test_precomputed_signal_adapter_rejects_dates_outside_training_window():
    from cross_signal_strategy.research.friction_diagnostics import PrecomputedSignalAdapter

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        PrecomputedSignalAdapter.from_source(
            FakeSourceAdapter(),
            trade_dates=["2022-01-04"],
            codes=["AAA"],
        )


def test_friction_decomposition_identifies_each_component_and_interaction():
    from cross_signal_strategy.research.friction_diagnostics import build_friction_decomposition

    reports = {
        "baseline": baseline_report(40000.0, max_drawdown=0.05),
        "commission_rate_x2": baseline_report(39000.0, max_drawdown=0.051),
        "minimum_commission_x2": baseline_report(38000.0, max_drawdown=0.052),
        "slippage_x2": baseline_report(36000.0, max_drawdown=0.060, buy_count=9),
        "all_x2": baseline_report(34000.0, max_drawdown=0.065, buy_count=9),
    }

    result = build_friction_decomposition(reports)

    assert result.scenarios["commission_rate_x2"].return_delta == pytest.approx(-0.05)
    assert result.scenarios["minimum_commission_x2"].return_delta == pytest.approx(-0.10)
    assert result.scenarios["slippage_x2"].return_delta == pytest.approx(-0.20)
    assert result.scenarios["all_x2"].return_delta == pytest.approx(-0.30)
    assert result.dominant_component == "slippage_x2"
    assert result.component_return_delta_sum == pytest.approx(-0.35)
    assert result.interaction_return_delta == pytest.approx(0.05)
    assert result.scenarios["slippage_x2"].buy_count_delta == -1
    assert result.scenarios["all_x2"].max_drawdown_delta == pytest.approx(0.015)


def test_friction_decomposition_requires_every_locked_scenario():
    from cross_signal_strategy.research.friction_diagnostics import build_friction_decomposition

    with pytest.raises(ValueError, match="Missing friction scenarios"):
        build_friction_decomposition({"baseline": baseline_report(40000.0)})
