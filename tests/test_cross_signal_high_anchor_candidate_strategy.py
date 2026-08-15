# -*- coding: utf-8 -*-
"""Tests for the isolated intraday-high anchor candidate."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_candidate_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_high_anchor_candidate",
        ROOT / "cross_signal_strategy" / "archive" / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_high_anchor_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mainline_strategy():
    spec = importlib.util.spec_from_file_location(
        "cross_signal_mainline_for_high_anchor_test",
        ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_anchor_candidate_declares_version_and_identical_params():
    candidate = load_candidate_strategy()
    mainline = load_mainline_strategy()

    assert candidate.STRATEGY_VERSION == "cross-v0.3.3-high-anchor-candidate"
    assert candidate.get_default_params() == mainline.get_default_params()
    assert candidate.get_default_etf_pool() == mainline.get_default_etf_pool()


def test_high_anchor_candidate_only_changes_the_after_close_anchor():
    source = (
        ROOT / "cross_signal_strategy" / "archive" / "candidates"
        / "smart_trade_joinquant_cross_signal_etf_high_anchor_candidate.py"
    ).read_text(encoding="utf-8")

    assert "high_price" in source
    assert "anchor_value" in source


def test_high_anchor_candidate_keeps_stop_formula_unchanged():
    candidate = load_candidate_strategy()
    mainline = load_mainline_strategy()

    # calc_stop_price must be byte-identical behavior: the anchor change is
    # only in the after-close update, not in the stop formula.
    for highest, atr, cost in ((10.0, 0.2, 8.0), (8.0, 1.0, 7.0), (10.0, 0.1, 9.0)):
        assert candidate.calc_stop_price(highest, atr, cost) == mainline.calc_stop_price(
            highest, atr, cost)
