# -*- coding: utf-8 -*-
"""Tests for the temporary JoinQuant ATR-2.0 candidate strategy."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_atr2_candidate",
        ROOT / "cross_signal_strategy" / "archive" / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_atr2_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atr2_candidate_only_tightens_trailing_atr_multiplier():
    strategy = load_candidate_strategy()

    params = strategy.get_default_params()

    assert strategy.STRATEGY_VERSION == "cross-v0.3.1-atr2-candidate"
    assert params["trailing_atr_mult"] == 2.0
    assert params["stop_floor"] == 0.05
    assert params["stop_cap"] == 0.15
    assert params["base_ratio"] == 0.95
    assert params["min_signal_hold_days"] == 5
