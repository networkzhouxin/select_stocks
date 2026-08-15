# -*- coding: utf-8 -*-
"""Tests for the isolated profit-tiered ATR tightening candidate."""

import importlib.util
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_profit_tier_candidate",
        ROOT / "cross_signal_strategy" / "archive" / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_profit_tier_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mainline_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_mainline_for_profit_tier_test",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profit_tier_candidate_declares_frozen_tier_params():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()

    assert strategy.STRATEGY_VERSION == "cross-v0.3.3-profit-tier-candidate"
    assert params["profit_tier_low"] == 0.05
    assert params["profit_tier_high"] == 0.15
    assert params["profit_tier_factor_low"] == 0.8
    assert params["profit_tier_factor_high"] == 0.6
    # 主线其余参数必须原样携带, 包括冻结的 ATR-stress 三键。
    main_params = load_mainline_strategy().get_default_params()
    assert {
        key: params[key] for key in main_params
    } == main_params


def test_profit_tier_candidate_matches_mainline_without_profit():
    candidate = load_candidate_strategy()
    mainline = load_mainline_strategy()

    assert candidate.calc_stop_price(10.0, 0.2, 8.0) == pytest.approx(
        mainline.calc_stop_price(10.0, 0.2, 8.0))


def test_profit_tier_candidate_tightens_above_five_percent_profit():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 0.36  # 2.5xATR/highest = 9% > 5% floor, both tiers bind

    baseline = strategy.calc_stop_price(highest, atr, 9.0, params, profit_pct=0.02)
    mid = strategy.calc_stop_price(highest, atr, 9.0, params, profit_pct=0.10)
    high = strategy.calc_stop_price(highest, atr, 9.0, params, profit_pct=0.20)

    assert baseline == pytest.approx(highest * (1 - 0.09))
    assert mid == pytest.approx(highest * (1 - 0.09 * 0.8))
    assert high == pytest.approx(highest * (1 - 0.09 * 0.6))


def test_profit_tier_candidate_floor_still_dominates_low_volatility():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 0.14  # 2.5xATR/highest = 3.5% < 5% floor

    tight = strategy.calc_stop_price(highest, atr, 9.0, params, profit_pct=0.20)

    assert tight == pytest.approx(highest * (1 - 0.05))


def test_profit_tier_candidate_does_not_change_zero_profit_stop():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 0.3

    flat = strategy.calc_stop_price(highest, atr, 9.5, params, profit_pct=0.0)
    negative = strategy.calc_stop_price(highest, atr, 11.0, params, profit_pct=-0.05)

    assert flat == pytest.approx(highest * (1 - 0.075))
    assert negative == pytest.approx(highest * (1 - 0.075))
