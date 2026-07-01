# -*- coding: utf-8 -*-
"""Regression tests for deterministic rotation ordering."""

import importlib.util
import pathlib
import sys
import types
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

spec = importlib.util.spec_from_file_location(
    "jq_multifactor", ROOT / "smart_trade_joinquant_multifactor_etf.py")
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)
strategy.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)

ptrade_spec = importlib.util.spec_from_file_location(
    "ptrade_multifactor", ROOT / "smart_trade_ptrade_multifactor_etf.py")
ptrade_strategy = importlib.util.module_from_spec(ptrade_spec)
ptrade_spec.loader.exec_module(ptrade_strategy)
ptrade_strategy.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)


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


def test_buy_overheat_filter_uses_rsi_without_ma20_distance():
    sig = {"ma20": 100.0, "rsi": 75.1}

    assert strategy.is_overheated_for_buy("TEST", sig, 101.0)


def test_joinquant_tiers_use_high_base_ratio_experiment():
    tiers = strategy.get_default_capital_tiers()

    assert tiers["micro"]["base_ratio"] == 0.75
    assert tiers["small"]["base_ratio"] == 0.75
    assert tiers["medium"]["base_ratio"] == 0.75
    assert tiers["large"]["base_ratio"] == 0.75
    assert all(cfg["max_hold"] == 3 for cfg in tiers.values())


def test_joinquant_profit_floor_uses_fixed_tiers():
    params = {
        "profit_floor_enabled": True,
        "profit_floor_tiers": [(0.15, 0.08), (0.10, 0.05)],
    }

    assert strategy.calc_profit_floor_price(100.0, 109.9, params) is None
    assert round(strategy.calc_profit_floor_price(100.0, 110.0, params), 2) == 105.0
    assert round(strategy.calc_profit_floor_price(100.0, 116.0, params), 2) == 108.0


def test_ptrade_tiers_match_joinquant_high_base_ratio():
    tiers = ptrade_strategy._get_default_capital_tiers()

    assert tiers["micro"]["base_ratio"] == 0.75
    assert tiers["small"]["base_ratio"] == 0.75
    assert tiers["medium"]["base_ratio"] == 0.75
    assert tiers["large"]["base_ratio"] == 0.75
    assert all(cfg["max_hold"] == 3 for cfg in tiers.values())


def test_ptrade_profit_floor_matches_joinquant_fixed_tiers():
    params = {
        "profit_floor_enabled": True,
        "profit_floor_tiers": [(0.15, 0.08), (0.10, 0.05)],
    }

    assert ptrade_strategy._calc_profit_floor_price(100.0, 109.9, params) is None
    assert round(ptrade_strategy._calc_profit_floor_price(100.0, 110.0, params), 2) == 105.0
    assert round(ptrade_strategy._calc_profit_floor_price(100.0, 116.0, params), 2) == 108.0


def test_ptrade_stop_price_applies_profit_floor():
    ptrade_strategy.g = types.SimpleNamespace(
        params={
            "atr_period": 14,
            "trailing_atr_mult": 2.5,
            "trailing_atr_mult_high_vol": 2.0,
            "high_vol_threshold": 0.30,
            "stop_floor": 0.05,
            "stop_cap": 0.15,
            "profit_floor_enabled": True,
            "profit_floor_tiers": [(0.15, 0.08), (0.10, 0.05)],
        },
        code_stop_params={},
    )

    stop_price = ptrade_strategy._calc_stop_price(
        "TEST.SS", highest=110.0, atr_val=100.0,
        profit_pct=0.10, entry_cost=100.0)

    assert round(stop_price, 2) == 105.0


def test_ptrade_buy_overheat_filter_uses_rsi_without_ma20_distance():
    sig = {"ma20": 100.0, "rsi": 75.1}

    assert ptrade_strategy._is_overheated_for_buy("TEST", sig, 101.0)


def test_ptrade_sorting_matches_joinquant_stable_tiebreaks():
    removable = [
        ("159915.SZ", 70.9),
        ("512100.SS", 70.9),
        ("513050.SS", 80.1),
    ]
    rank_map = {
        "513050.SS": 0,
        "159920.SZ": 1,
        "513100.SS": 2,
        "159915.SZ": 3,
        "512100.SS": 4,
    }

    assert ptrade_strategy._sort_removable_positions(removable, rank_map)[0] == (
        "512100.SS", 70.9)

    codes = ["BBB", "AAA", "CCC"]
    sig_map = {
        "AAA": {"final_score": 80.0},
        "BBB": {"final_score": 80.0},
        "CCC": {"final_score": 79.0},
    }
    rank_map = {"AAA": 1, "BBB": 0, "CCC": 2}

    assert ptrade_strategy._sort_buy_codes(codes, sig_map, rank_map) == [
        "BBB", "AAA", "CCC"]


def test_ptrade_tracks_all_paused_pool_codes_for_1035_recheck():
    paused_codes = {"AAA", "CCC"}

    tracked = ptrade_strategy._find_paused_pool_codes(
        ["AAA", "BBB", "CCC", "DDD"],
        paused_codes.__contains__,
    )

    assert tracked == {"AAA", "CCC"}


def test_ptrade_detects_same_day_buy_for_sell_guard():
    buy_date = {"AAA": date(2026, 6, 16)}

    assert ptrade_strategy._bought_today(buy_date, "AAA", date(2026, 6, 16))
    assert not ptrade_strategy._bought_today(buy_date, "AAA", date(2026, 6, 17))
    assert not ptrade_strategy._bought_today(buy_date, "BBB", date(2026, 6, 16))


if __name__ == "__main__":
    for test in (
        test_removable_positions_use_rank_tiebreak_for_equal_scores,
        test_buy_codes_use_score_then_rank_then_code,
        test_buy_overheat_filter_uses_rsi_without_ma20_distance,
        test_joinquant_tiers_use_high_base_ratio_experiment,
        test_joinquant_profit_floor_uses_fixed_tiers,
        test_ptrade_tiers_match_joinquant_high_base_ratio,
        test_ptrade_profit_floor_matches_joinquant_fixed_tiers,
        test_ptrade_stop_price_applies_profit_floor,
        test_ptrade_buy_overheat_filter_uses_rsi_without_ma20_distance,
        test_ptrade_sorting_matches_joinquant_stable_tiebreaks,
        test_ptrade_tracks_all_paused_pool_codes_for_1035_recheck,
        test_ptrade_detects_same_day_buy_for_sell_guard,
    ):
        test()
