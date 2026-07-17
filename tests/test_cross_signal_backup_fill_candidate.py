# -*- coding: utf-8 -*-
"""Tests for the training-only backup cross-signal fill candidate."""

import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        item = self.scores.get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        return (dict(item), None) if return_reason else dict(item)


def candidate(code, buy_score, **overrides):
    item = {
        "code": code,
        "buy_allowed": True,
        "buy_score": buy_score,
        "sell_score": 0,
        "reversal_score": 12,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": False,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "volume_score": 0,
        "trend_score": 5,
        "atr": 0.1,
    }
    item.update(overrides)
    return item


def test_backup_filter_keeps_only_50_59_scores_that_pass_all_mainline_filters():
    from cross_signal_strategy.archive.candidates.backup_fill_candidate import filter_backup_buy_candidates

    valid = candidate("VALID", 55)
    primary = candidate("PRIMARY", 60)
    weak = candidate("WEAK", 49)
    overheat = candidate("HOT", 55, buy_allowed=False)
    blocked = candidate(
        "BLOCKED",
        55,
        macd_cross_up=True,
        kdj_k_cross_up=False,
        volume_score=4,
        trend_score=10,
    )

    kept = filter_backup_buy_candidates(
        [valid, primary, weak, overheat, blocked],
        held_codes=[],
    )

    assert kept == [valid]


def test_backup_planner_fills_only_slots_left_by_primary_candidates():
    from cross_signal_strategy.archive.candidates.backup_fill_candidate import BackupFillOrderPlanner
    from cross_signal_strategy.local.local_backtester import LocalBroker

    scores = {
        "PRIMARY": candidate("PRIMARY", 70),
        "BACKUP": candidate("BACKUP", 55),
        "WEAK": candidate("WEAK", 49),
    }
    planner = BackupFillOrderPlanner(
        FakeSignalAdapter(scores),
        etf_pool=["PRIMARY", "BACKUP", "WEAK"],
    )

    orders = planner.plan_orders("2019-01-02", None, LocalBroker(20000.0))

    assert [(order["code"], order["reason"]) for order in orders] == [
        ("PRIMARY", "buy_signal"),
        ("BACKUP", "backup_buy_signal"),
    ]


def test_backup_planner_does_not_displace_three_primary_candidates():
    from cross_signal_strategy.archive.candidates.backup_fill_candidate import BackupFillOrderPlanner
    from cross_signal_strategy.local.local_backtester import LocalBroker

    scores = {
        "P1": candidate("P1", 80),
        "P2": candidate("P2", 75),
        "P3": candidate("P3", 70),
        "BACKUP": candidate("BACKUP", 55),
    }
    planner = BackupFillOrderPlanner(
        FakeSignalAdapter(scores),
        etf_pool=["P1", "P2", "P3", "BACKUP"],
    )

    orders = planner.plan_orders("2019-01-02", None, LocalBroker(20000.0))

    assert [order["code"] for order in orders] == ["P1", "P2", "P3"]
    assert all(order["reason"] == "buy_signal" for order in orders)
