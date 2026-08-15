# -*- coding: utf-8 -*-
"""Tests for the isolated gold-specific stop candidate."""

import importlib.util
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_gold_stop_candidate",
        ROOT / "cross_signal_strategy" / "archive" / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_gold_stop_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mainline_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_mainline_for_gold_stop_test",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gold_stop_candidate_declares_frozen_gold_params():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()

    assert strategy.STRATEGY_VERSION == "cross-v0.3.3-gold-stop-candidate"
    assert params["gold_stop_floor"] == 0.03
    assert params["gold_trailing_atr_mult"] == 2.0
    # 主线其余参数必须原样携带, 包括冻结的 ATR-stress 三键。
    main_params = load_mainline_strategy().get_default_params()
    assert {
        key: params[key] for key in main_params
    } == main_params


def test_gold_stop_candidate_only_changes_gold_stop():
    strategy = load_candidate_strategy()
    mainline = load_mainline_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 0.2

    gold = strategy.calc_stop_price("518880.XSHG", highest, atr, 9.0, params)
    other = strategy.calc_stop_price("513100.XSHG", highest, atr, 9.0, params)

    # 黄金: 2.0xATR/highest = 4% -> 被 3% 地板钳制 -> 距峰值 4% 无地板影响?
    # 4% > 3% floor, 所以黄金止损 = 10*(1-0.04) = 9.6
    assert gold == pytest.approx(highest * (1 - 0.04))
    # 其他品种与主线完全一致: 2.5xATR/highest = 5% -> 恰好等于地板 -> 9.5
    assert other == pytest.approx(
        mainline.calc_stop_price(highest, atr, 9.0, mainline.get_default_params()))


def test_gold_stop_candidate_uses_three_percent_floor_for_gold():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 0.1  # 2.0xATR/highest = 2% < 3% floor

    gold = strategy.calc_stop_price("518880.SZ", highest, atr, 9.0, params)

    assert gold == pytest.approx(highest * (1 - 0.03))


def test_gold_stop_candidate_keeps_cap_for_gold():
    strategy = load_candidate_strategy()
    params = strategy.get_default_params()
    highest = 10.0
    atr = 1.0  # 2.0xATR/highest = 20% > 15% cap

    gold = strategy.calc_stop_price("518880.SS", highest, atr, 9.0, params)

    assert gold == pytest.approx(highest * (1 - 0.15))
