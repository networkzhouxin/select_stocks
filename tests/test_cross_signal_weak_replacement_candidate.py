# -*- coding: utf-8 -*-
"""Tests for replacement-aware weak signal-sell protection candidate."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def score(code, buy_score=35, sell_score=34, **overrides):
    item = {
        "code": code,
        "buy_allowed": True,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
    }
    item.update(overrides)
    return item


def test_weak_replacement_candidate_protects_only_weak_sell_without_replacement():
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate as candidate,
    )

    current = score("AAA", buy_score=35, sell_score=34)
    no_buy_replacement = score("BBB", buy_score=59, sell_score=0)

    assert candidate.STRATEGY_VERSION == "cross-v0.3.2-weak-replacement-candidate"
    assert candidate.should_protect_weak_no_replacement_signal_sell(
        "AAA",
        current,
        [current, no_buy_replacement],
        held_codes=["AAA"],
        params={"buy_threshold": 60, "sell_threshold": 30},
    )


def test_weak_replacement_candidate_does_not_protect_when_replacement_exists():
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate as candidate,
    )

    current = score("AAA", buy_score=35, sell_score=34)
    replacement = score("BBB", buy_score=60, sell_score=0)

    assert not candidate.should_protect_weak_no_replacement_signal_sell(
        "AAA",
        current,
        [current, replacement],
        held_codes=["AAA"],
        params={"buy_threshold": 60, "sell_threshold": 30},
    )


def test_weak_replacement_candidate_does_not_protect_stronger_or_unsupported_sells():
    from cross_signal_strategy import (
        smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate as candidate,
    )

    strong_sell = score("AAA", buy_score=35, sell_score=35)
    weak_no_buy_support = score("AAA", buy_score=34, sell_score=34)

    assert not candidate.should_protect_weak_no_replacement_signal_sell(
        "AAA",
        strong_sell,
        [strong_sell],
        held_codes=["AAA"],
        params={"buy_threshold": 60, "sell_threshold": 30},
    )
    assert not candidate.should_protect_weak_no_replacement_signal_sell(
        "AAA",
        weak_no_buy_support,
        [weak_no_buy_support],
        held_codes=["AAA"],
        params={"buy_threshold": 60, "sell_threshold": 30},
    )
