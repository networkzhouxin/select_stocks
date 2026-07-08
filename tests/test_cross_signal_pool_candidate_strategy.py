# -*- coding: utf-8 -*-
"""Tests for the temporary JoinQuant ETF-pool candidate strategy."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_pool_candidate",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf_pool_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pool_candidate_removes_training_drag_symbols_only():
    strategy = load_candidate_strategy()

    pool = strategy.get_default_etf_pool()

    assert strategy.STRATEGY_VERSION.endswith("-pool-candidate")
    assert "510300.XSHG" not in pool
    assert "510880.XSHG" not in pool
    assert "159920.XSHE" not in pool
    assert pool == [
        "159915.XSHE",
        "512100.XSHG",
        "159928.XSHE",
        "513100.XSHG",
        "513500.XSHG",
        "513880.XSHG",
        "513050.XSHG",
        "518880.XSHG",
        "159985.XSHE",
    ]
