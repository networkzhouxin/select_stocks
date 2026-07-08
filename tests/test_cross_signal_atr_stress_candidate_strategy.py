# -*- coding: utf-8 -*-
"""Tests for the temporary JoinQuant ATR-stress candidate strategy."""

import importlib.util
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_atr_stress_candidate",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atr_stress_candidate_declares_broad_training_candidate_params():
    strategy = load_candidate_strategy()

    params = strategy.get_default_params()

    assert strategy.STRATEGY_VERSION == "cross-v0.3.1-atr-stress-candidate"
    assert params["portfolio_atr_stress_lookback_days"] == 15
    assert params["portfolio_atr_stress_min_stops"] == 3
    assert params["portfolio_atr_stress_buy_scale"] == 0.50


def test_atr_stress_candidate_scales_new_buys_after_clustered_atr_stops():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    trade_days = [
        "2020-02-13",
        "2020-02-17",
        "2020-02-19",
        "2020-03-02",
        "2020-03-03",
        "2020-03-05",
    ]
    stop_dates = ["2020-02-17", "2020-02-19", "2020-03-02"]
    score = {"code": "513100.XSHG", "volume_score": 6}

    target = strategy.calc_stress_adjusted_buy_target_value(
        12000.0,
        score,
        params,
        current_date="2020-03-05",
        atr_stop_history=stop_dates,
        trade_days=trade_days,
    )

    assert target == pytest.approx(1900.0)


def test_atr_stress_candidate_ignores_old_atr_stops_outside_lookback():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    trade_days = ["2020-01-%02d" % day for day in range(2, 24)]
    stop_dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    score = {"code": "513100.XSHG", "volume_score": 6}

    target = strategy.calc_stress_adjusted_buy_target_value(
        12000.0,
        score,
        params,
        current_date="2020-01-23",
        atr_stop_history=stop_dates,
        trade_days=trade_days,
    )

    assert target == pytest.approx(3800.0)


def test_atr_stress_candidate_records_filled_atr_stop_dates(monkeypatch):
    strategy = load_candidate_strategy()

    class Position(object):
        total_amount = 1000

    context = types.SimpleNamespace(
        current_dt=types.SimpleNamespace(date=lambda: "2020-03-02"),
        portfolio=types.SimpleNamespace(positions={"513100.XSHG": Position()}),
    )
    strategy.g = types.SimpleNamespace(
        atr_stop_history=[],
        highest_since_buy={},
        entry_atr={},
        buy_date={},
        last_scores={},
    )

    def fake_order_target(code, amount):
        del context.portfolio.positions[code]

    monkeypatch.setattr(strategy, "order_target", fake_order_target, raising=False)

    strategy.execute_sell("513100.XSHG", context, "atr_stop 3.906<=4.000")

    assert strategy.g.atr_stop_history == ["2020-03-02"]
