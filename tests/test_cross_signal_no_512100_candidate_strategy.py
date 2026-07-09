# -*- coding: utf-8 -*-
"""Tests for the temporary JoinQuant no-512100 pool candidate."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_no_512100_candidate",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_512100_candidate_only_removes_512100_from_mainline_pool():
    main_spec = importlib.util.spec_from_file_location(
        "cross_signal_mainline_for_no_512100_test",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
    )
    mainline = importlib.util.module_from_spec(main_spec)
    main_spec.loader.exec_module(mainline)
    candidate = load_candidate_strategy()

    main_pool = mainline.get_default_etf_pool()
    candidate_pool = candidate.get_default_etf_pool()

    assert candidate.STRATEGY_VERSION == "cross-v0.3.1-no-512100-candidate"
    assert "512100.XSHG" in main_pool
    assert "512100.XSHG" not in candidate_pool
    assert candidate_pool == [code for code in main_pool if code != "512100.XSHG"]
    assert candidate.get_default_params() == mainline.get_default_params()
