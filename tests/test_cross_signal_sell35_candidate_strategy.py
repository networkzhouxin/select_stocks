# -*- coding: utf-8 -*-
"""Tests for the sell-threshold confirmation candidate."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def test_sell35_candidate_only_raises_normal_signal_sell_threshold():
    from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as mainline
    from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf_sell35_candidate as candidate

    main_params = mainline.get_default_params()
    candidate_params = candidate.get_default_params()

    assert candidate.STRATEGY_VERSION == "cross-v0.3.2-sell35-candidate"
    assert main_params["sell_threshold"] == 30
    assert candidate_params["sell_threshold"] == 35

    unchanged = set(main_params) - {"sell_threshold"}
    assert {key: candidate_params[key] for key in unchanged} == {
        key: main_params[key] for key in unchanged
    }
    assert candidate.get_default_etf_pool() == mainline.get_default_etf_pool()


def test_sell35_candidate_keeps_atr_stop_unconditional_but_blocks_weak_signal_sell():
    from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf_sell35_candidate as candidate

    weak_signal = {
        "sell_score": 34,
        "close_below_ma20": True,
        "adx": 15.0,
        "plus_di": 10.0,
        "minus_di": 20.0,
        "ma20_slope_non_negative": False,
    }
    confirmed_signal = dict(weak_signal, sell_score=35)

    assert candidate.should_force_sell(weak_signal, atr_stop_triggered=True)
    assert not candidate.should_force_sell(weak_signal, atr_stop_triggered=False)
    assert candidate.should_force_sell(confirmed_signal, atr_stop_triggered=False)
