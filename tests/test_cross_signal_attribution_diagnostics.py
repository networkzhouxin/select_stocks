# -*- coding: utf-8 -*-
"""Tests for ETF-level attribution diagnostics."""

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def closed_trade(code, buy_date, sell_date, reason, pnl, entry_score=None):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason=reason,
        amount=100,
        buy_price=10.0,
        sell_price=11.0,
        pnl=pnl,
        return_pct=10.0,
        entry_score=entry_score or {},
    )


def test_etf_attribution_summarizes_trade_quality_by_code():
    from cross_signal_strategy.research.attribution_diagnostics import build_etf_attribution

    trades = [
        closed_trade("AAA", "2021-01-01", "2021-01-06", "atr_stop", 100.0),
        closed_trade("AAA", "2021-01-07", "2021-01-11", "signal_sell", -40.0),
        closed_trade("BBB", "2021-01-04", "2021-01-05", "signal_sell", 30.0),
    ]

    report = build_etf_attribution(
        trades,
        trade_dates=[
            "2021-01-01",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
        ],
    )

    aaa = report.by_code["AAA"]
    assert aaa.closed_trades == 2
    assert aaa.wins == 1
    assert aaa.losses == 1
    assert aaa.realized_pnl == pytest.approx(60.0)
    assert aaa.win_rate == pytest.approx(0.5)
    assert aaa.profit_loss_ratio == pytest.approx(100.0 / 40.0)
    assert aaa.atr_stop_rate == pytest.approx(0.5)
    assert aaa.signal_sell_rate == pytest.approx(0.5)
    assert aaa.average_holding_days == pytest.approx((3 + 2) / 2)

    assert report.total_realized_pnl == pytest.approx(90.0)
    assert report.by_code["BBB"].average_holding_days == pytest.approx(1.0)


def test_entry_signal_tags_are_stable_and_group_signal_sources():
    from cross_signal_strategy.research.attribution_diagnostics import (
        entry_combo_key,
        entry_signal_tags,
    )

    score = {
        "rsi6_cross_rsi12_up": True,
        "macd_cross_up": True,
        "kdj_j_cross_up": True,
        "location_score": 17,
        "trend_score": 20,
        "volume_score": 6,
    }

    assert entry_signal_tags(score) == (
        "kdj_up",
        "low_location",
        "macd_up",
        "rsi_up",
        "strong_trend",
        "volume_confirmed",
    )
    assert entry_combo_key(score) == (
        "kdj_up+low_location+macd_up+rsi_up+strong_trend+volume_confirmed"
    )


def test_entry_signal_combo_summary_aggregates_closed_trade_quality():
    from cross_signal_strategy.research.attribution_diagnostics import (
        summarize_entry_signal_combos,
    )

    trades = [
        closed_trade(
            "AAA",
            "2021-01-01",
            "2021-01-06",
            "atr_stop",
            100.0,
            entry_score={"rsi6_cross_rsi12_up": True, "volume_score": 6},
        ),
        closed_trade(
            "BBB",
            "2021-01-04",
            "2021-01-08",
            "signal_sell",
            -25.0,
            entry_score={"rsi6_cross_rsi24_up": True, "volume_score": 6},
        ),
        closed_trade(
            "CCC",
            "2021-01-05",
            "2021-01-11",
            "signal_sell",
            40.0,
            entry_score={"macd_cross_up": True, "trend_score": 12},
        ),
    ]

    summary = summarize_entry_signal_combos(trades)

    rsi_volume = summary["rsi_up+volume_confirmed"]
    assert rsi_volume.closed_trades == 2
    assert rsi_volume.wins == 1
    assert rsi_volume.losses == 1
    assert rsi_volume.realized_pnl == pytest.approx(75.0)
    assert rsi_volume.win_rate == pytest.approx(0.5)
    assert rsi_volume.profit_loss_ratio == pytest.approx(100.0 / 25.0)

    macd_trend = summary["macd_up+trend_support"]
    assert macd_trend.closed_trades == 1
    assert macd_trend.realized_pnl == pytest.approx(40.0)
    assert macd_trend.average_pnl == pytest.approx(40.0)


def test_entry_bucket_labels_group_buy_context():
    from cross_signal_strategy.research.attribution_diagnostics import entry_bucket_labels

    labels = entry_bucket_labels(
        closed_trade(
            "159915",
            "2021-01-01",
            "2021-01-06",
            "signal_sell",
            10.0,
            entry_score={
                "buy_score": 73,
                "sell_score": 12,
                "rsi6": 81,
                "location_score": 17,
                "trend_score": 9,
                "volume_score": 0,
                "close": 12.0,
                "ma20": 10.0,
                "upper": 12.8,
                "middle": 10.0,
                "lower": 7.2,
            },
        )
    )

    assert labels["etf_class"] == "a_share"
    assert labels["buy_score_band"] == "70+"
    assert labels["rsi6_band"] == "overheated"
    assert labels["location_bucket"] == "low_or_mid"
    assert labels["trend_bucket"] == "mild_trend"
    assert labels["volume_bucket"] == "no_volume"
    assert labels["sell_conflict"] == "no_sell_conflict"
    assert labels["ma20_distance"] == "far_above_ma20"
    assert labels["boll_position"] == "upper_half"


def test_entry_bucket_summary_aggregates_each_dimension():
    from cross_signal_strategy.research.attribution_diagnostics import summarize_entry_buckets

    trades = [
        closed_trade(
            "159915",
            "2021-01-01",
            "2021-01-06",
            "signal_sell",
            100.0,
            entry_score={"buy_score": 72, "trend_score": 20, "volume_score": 4},
        ),
        closed_trade(
            "513100",
            "2021-01-04",
            "2021-01-08",
            "atr_stop",
            -30.0,
            entry_score={"buy_score": 58, "trend_score": -9, "volume_score": 0},
        ),
    ]

    summary = summarize_entry_buckets(trades)

    assert summary["buy_score_band"]["70+"].closed_trades == 1
    assert summary["buy_score_band"]["70+"].realized_pnl == pytest.approx(100.0)
    assert summary["buy_score_band"]["below_60"].realized_pnl == pytest.approx(-30.0)
    assert summary["etf_class"]["a_share"].closed_trades == 1
    assert summary["etf_class"]["cross_market"].closed_trades == 1
    assert summary["volume_bucket"]["no_volume"].losses == 1
