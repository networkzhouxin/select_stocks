# -*- coding: utf-8 -*-
"""Tests for local walkforward profit-floor alignment."""

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
WALKFORWARD = ROOT / "walkforward"
sys.path.insert(0, str(WALKFORWARD))

spec = importlib.util.spec_from_file_location("wf_engine", WALKFORWARD / "engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def test_walkforward_tiers_match_joinquant_base_ratio():
    assert engine.TIER_CFG["micro"]["base_ratio"] == 0.75
    assert engine.TIER_CFG["small"]["base_ratio"] == 0.75
    assert engine.TIER_CFG["medium"]["base_ratio"] == 0.75
    assert engine.TIER_CFG["large"]["base_ratio"] == 0.75


def test_walkforward_default_profit_floor_uses_fixed_tiers():
    assert engine.DEFAULT_PARAMS["profit_floor_tiers"] == [(0.15, 0.08), (0.10, 0.05)]


def test_walkforward_fixed_profit_floor_uses_tiers():
    params = dict(engine.DEFAULT_PARAMS)
    params.update({
        "profit_floor_enabled": True,
        "profit_floor_tiers": [(0.15, 0.08), (0.10, 0.05)],
    })

    assert engine.calc_profit_floor_price(100.0, 109.9, params) is None
    assert round(engine.calc_profit_floor_price(100.0, 110.0, params), 2) == 105.0
    assert round(engine.calc_profit_floor_price(100.0, 116.0, params), 2) == 108.0


def test_walkforward_fixed_profit_floor_raises_stop_price():
    params = dict(engine.DEFAULT_PARAMS)
    params.update({
        "profit_floor_enabled": True,
        "profit_floor_tiers": [(0.15, 0.08), (0.10, 0.05)],
    })

    stop = engine.calc_stop_price(
        highest=110.0, atr_val=100.0, params=params,
        profit_pct=0.10, entry_cost=100.0)

    assert round(stop, 2) == 105.0


if __name__ == "__main__":
    for test in (
        test_walkforward_tiers_match_joinquant_base_ratio,
        test_walkforward_default_profit_floor_uses_fixed_tiers,
        test_walkforward_fixed_profit_floor_uses_tiers,
        test_walkforward_fixed_profit_floor_raises_stop_price,
    ):
        test()
