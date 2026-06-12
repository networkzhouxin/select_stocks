# -*- coding: utf-8 -*-
"""Regression tests for deterministic rotation ordering."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

spec = importlib.util.spec_from_file_location(
    "jq_multifactor", ROOT / "smart_trade_joinquant_multifactor_etf.py")
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)
strategy.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)


def test_removable_positions_use_rank_tiebreak_for_equal_scores():
    removable = [
        ("159915.XSHE", 70.9),
        ("512100.XSHG", 70.9),
        ("513050.XSHG", 80.1),
    ]
    rank_map = {
        "513050.XSHG": 0,
        "159920.XSHE": 1,
        "513100.XSHG": 2,
        "159915.XSHE": 3,
        "512100.XSHG": 4,
    }

    ordered = strategy.sort_removable_positions(removable, rank_map)

    assert ordered[0] == ("512100.XSHG", 70.9)


def test_buy_codes_use_score_then_rank_then_code():
    codes = ["BBB", "AAA", "CCC"]
    sig_map = {
        "AAA": {"final_score": 80.0},
        "BBB": {"final_score": 80.0},
        "CCC": {"final_score": 79.0},
    }
    rank_map = {"AAA": 1, "BBB": 0, "CCC": 2}

    ordered = strategy.sort_buy_codes(codes, sig_map, rank_map)

    assert ordered == ["BBB", "AAA", "CCC"]


def test_high_rsi_shadow_helpers_define_trigger_and_return():
    assert strategy.should_record_high_rsi_shadow(85.0, 85.0)
    assert not strategy.should_record_high_rsi_shadow(84.9, 85.0)
    assert strategy.calc_shadow_return(110.0, 100.0) == 10.0


def test_buy_overheat_filter_uses_rsi_without_ma20_distance():
    sig = {"ma20": 100.0, "rsi": 75.1}

    assert strategy.is_overheated_for_buy("TEST", sig, 101.0)


if __name__ == "__main__":
    for test in (
        test_removable_positions_use_rank_tiebreak_for_equal_scores,
        test_buy_codes_use_score_then_rank_then_code,
        test_high_rsi_shadow_helpers_define_trigger_and_return,
        test_buy_overheat_filter_uses_rsi_without_ma20_distance,
    ):
        test()
