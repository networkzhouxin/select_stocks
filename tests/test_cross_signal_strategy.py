# -*- coding: utf-8 -*-
"""Tests for the isolated cross-signal ETF strategy helpers."""

import importlib.util
import pathlib
import sys
import types
from datetime import date

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

spec = importlib.util.spec_from_file_location(
    "cross_signal",
    ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


def test_formal_joinquant_source_has_no_stale_release_labels():
    source = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_joinquant_cross_signal_etf.py"
    ).read_text(encoding="utf-8")

    assert "Strategy v0.1 for JoinQuant" not in source
    assert "[cross-v0.1]" not in source


def test_joinquant_exposes_stable_release_fingerprint():
    fingerprint = strategy.business_config_fingerprint()

    assert strategy.DEPLOYMENT_BUILD_ID == "20260720.1"
    assert len(fingerprint) == 12
    assert all(ch in "0123456789abcdef" for ch in fingerprint)


def test_recent_cross_detection_uses_last_three_days():
    fast = strategy.pd.Series([4.0, 4.0, 4.0, 4.0, 7.0])
    slow = strategy.pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])

    assert strategy.crossed_above_recent(fast, slow, window=3)
    assert not strategy.crossed_below_recent(fast, slow, window=3)


def test_cross_detection_matches_logged_diff_semantics():
    fast = strategy.pd.Series([40.0, 41.0, 42.0, 45.9, 48.1])
    slow = strategy.pd.Series([42.0, 42.0, 42.0, 50.5, 46.2])

    assert strategy.crossed_above_by_diff_recent(fast, slow, window=3)
    assert strategy.crossed_above_recent(fast, slow, window=3)


def test_cross_detection_uses_position_not_series_index_alignment():
    fast = strategy.pd.Series([40.0, 41.0, 42.0, 45.9, 48.1], index=[10, 11, 12, 13, 14])
    slow = strategy.pd.Series([42.0, 42.0, 42.0, 50.5, 46.2], index=[20, 21, 22, 23, 24])

    assert strategy.crossed_above_by_diff_recent(fast, slow, window=3)
    assert strategy.crossed_above_recent(fast, slow, window=3)


def test_cross_detection_ignores_jqdata_any_global_pollution():
    fast = strategy.pd.Series([40.0, 41.0, 42.0, 45.9, 48.1])
    slow = strategy.pd.Series([42.0, 42.0, 42.0, 50.5, 46.2])

    strategy.any = lambda values: True
    try:
        assert strategy.crossed_above_recent(fast, slow, window=3)
        assert "diff_cross_self_check=True expected=True" in strategy.format_self_check()
    finally:
        del strategy.any


def test_recent_cross_detection_uses_latest_cross_direction():
    fast = strategy.pd.Series([9.0, 11.0, 9.0, 9.0])
    slow = strategy.pd.Series([10.0, 10.0, 10.0, 10.0])

    assert not strategy.crossed_above_recent(fast, slow, window=3)
    assert strategy.crossed_below_recent(fast, slow, window=3)


def test_rsi_handles_one_way_and_flat_series_boundaries():
    rising = strategy.pd.Series([float(i) for i in range(1, 31)])
    falling = strategy.pd.Series([float(i) for i in range(30, 0, -1)])
    flat = strategy.pd.Series([10.0] * 30)

    assert strategy.calc_rsi(rising, 6).iloc[-1] == 100.0
    assert strategy.calc_rsi(falling, 6).iloc[-1] == 0.0
    assert strategy.calc_rsi(flat, 6).iloc[-1] == 50.0


def test_dmi_adx_identifies_directional_uptrend():
    close = strategy.pd.Series([float(i) for i in range(1, 41)])
    high = close + 0.5
    low = close - 0.5

    plus_di, minus_di, adx = strategy.calc_dmi_adx(high, low, close, 14)

    assert plus_di.iloc[-1] > minus_di.iloc[-1]
    assert adx.iloc[-1] >= 25.0


def test_buy_score_matches_v01_components():
    snapshot = {
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": True,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "ma5_gt_ma10": True,
        "ma10_gt_ma20": True,
        "ma20_slope_non_negative": True,
        "close_gt_ma60": True,
        "downside_continuation": False,
        "volume_above_vol20_and_up": True,
        "vol5_gt_vol20": True,
        "rsi6": 62.0,
    }

    score = strategy.score_buy_snapshot(snapshot)

    assert score["reversal_score"] == 45
    assert score["location_score"] == 25
    assert score["trend_score"] == 20
    assert score["volume_score"] == 10
    assert score["buy_score"] == 100
    assert score["buy_allowed"]


def test_overheated_buy_is_blocked_without_rewriting_score():
    snapshot = {
        "rsi6": 85.0,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
    }

    score = strategy.score_buy_snapshot(snapshot)

    assert score["buy_score"] == 45
    assert not score["buy_allowed"]


def test_sell_score_and_force_threshold():
    snapshot = {
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "far_above_ma20_and_rsi6_down": True,
        "close_below_falling_ma10": True,
        "fell_back_inside_boll": True,
    }

    score = strategy.score_sell_snapshot(snapshot)
    score.update(snapshot)

    assert score["sell_score"] == 69
    assert strategy.should_force_sell(score, atr_stop_triggered=False)


def test_signal_sell_requires_structure_confirmation():
    snapshot = {
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
    }

    score = strategy.score_sell_snapshot(snapshot)
    score.update(snapshot)

    assert score["sell_score"] == 45
    assert not strategy.should_force_sell(score, atr_stop_triggered=False)


def test_signal_sell_confirmed_by_ma20_break():
    snapshot = {
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "close_below_ma20": True,
        "close_below_boll_mid": True,
    }

    score = strategy.score_sell_snapshot(snapshot)
    score.update(snapshot)

    assert strategy.should_force_sell(score, atr_stop_triggered=False)


def test_strong_adx_uptrend_blocks_nonsevere_signal_sell():
    snapshot = {
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "close_below_boll_mid": True,
        "close_below_ma20": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "adx": 32.0,
        "plus_di": 31.0,
        "minus_di": 16.0,
        "ma20_slope_non_negative": True,
    }

    score = strategy.score_sell_snapshot(snapshot)
    score.update(snapshot)

    assert score["sell_score"] == 45
    assert not strategy.should_force_sell(score, atr_stop_triggered=False)


def test_strong_adx_uptrend_does_not_block_severe_structure_sell():
    snapshot = {
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
        "kdj_j_cross_down": True,
        "close_below_boll_mid": True,
        "close_below_ma20": True,
        "downside_continuation": True,
        "adx": 32.0,
        "plus_di": 31.0,
        "minus_di": 16.0,
        "ma20_slope_non_negative": True,
    }

    score = strategy.score_sell_snapshot(snapshot)
    score.update(snapshot)

    assert strategy.should_force_sell(score, atr_stop_triggered=False)


def test_below_falling_ma10_accepts_rising_close_under_meaningfully_falling_ma10():
    close = strategy.pd.Series([100.0] * 50 + [120.0] + [100.0] * 8 + [90.0, 95.0])
    high = close + 1.0
    low = close - 1.0
    frame = strategy.pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "volume": [1000.0] * len(close),
    })

    snapshot = strategy.build_signal_snapshot(frame, strategy.get_default_params())

    assert snapshot["close"] < snapshot["ma10"]
    assert snapshot["ma10"] < close.rolling(10).mean().iloc[-2]
    assert snapshot["close"] > close.iloc[-2]
    assert snapshot["close_below_falling_ma10"]


def test_below_falling_ma10_matches_joinquant_exact_less_than_comparison():
    close = strategy.pd.Series([100.0] * 59 + [99.999999995])
    high = close + 1.0
    low = close - 1.0
    frame = strategy.pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "volume": [1000.0] * len(close),
    })

    snapshot = strategy.build_signal_snapshot(frame, strategy.get_default_params())

    assert snapshot["close"] < snapshot["ma10"]
    ma10_prev = close.rolling(10).mean().iloc[-2]
    assert 0 < ma10_prev - snapshot["ma10"] < 1e-9
    assert snapshot["close_below_falling_ma10"]


def test_below_falling_ma10_accepts_flat_weak_close():
    close = strategy.pd.Series([100.0] * 50 + [120.0] + [100.0] * 8 + [95.0, 95.0])
    high = close + 1.0
    low = close - 1.0
    frame = strategy.pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "volume": [1000.0] * len(close),
    })

    snapshot = strategy.build_signal_snapshot(frame, strategy.get_default_params())

    assert snapshot["close"] == close.iloc[-2]
    assert snapshot["close"] < snapshot["ma10"]
    assert snapshot["ma10"] < close.rolling(10).mean().iloc[-2]
    assert snapshot["close_below_falling_ma10"]


def test_atr_stop_sells_without_signal_confirmation():
    assert strategy.should_force_sell({"sell_score": 0}, atr_stop_triggered=True)


def test_risk_warning_does_not_change_mainline_stop_price():
    params = strategy.get_default_params()

    normal = strategy.calc_stop_price(100.0, 2.0, 100.0, params)

    assert normal == 95.0


def test_check_atr_stops_ignores_archived_risk_tightened_state():
    class Position(object):
        total_amount = 100
        avg_cost = 100.0

    class Portfolio(object):
        positions = {"TEST": Position()}

    class Context(object):
        portfolio = Portfolio()

    class CurrentItem(object):
        paused = False
        last_price = 96.5

    old_g = strategy.g if hasattr(strategy, "g") else None
    strategy.g = types.SimpleNamespace(
        params=strategy.get_default_params(),
        highest_since_buy={"TEST": 100.0},
        entry_atr={"TEST": 2.0},
        risk_tightened={},
    )
    try:
        assert strategy.check_atr_stops(Context(), {"TEST": CurrentItem()}) == []

        strategy.g.risk_tightened["TEST"] = True

        assert strategy.check_atr_stops(Context(), {"TEST": CurrentItem()}) == []
    finally:
        if old_g is None:
            del strategy.g
        else:
            strategy.g = old_g


def test_sell_state_is_kept_when_order_does_not_change_position():
    class Position(object):
        total_amount = 7900

    class Portfolio(object):
        positions = {"513880.XSHG": Position()}

    class Context(object):
        portfolio = Portfolio()

    old_g = strategy.g if hasattr(strategy, "g") else None
    strategy.g = types.SimpleNamespace(
        highest_since_buy={"513880.XSHG": 1.092},
        entry_atr={"513880.XSHG": 0.0071},
        buy_date={"513880.XSHG": date(2019, 10, 18)},
        last_scores={"513880.XSHG": {"sell_score": 45}},
    )
    try:
        strategy.sync_sell_state_after_order("513880.XSHG", Context())

        assert strategy.g.highest_since_buy["513880.XSHG"] == 1.092
        assert strategy.g.entry_atr["513880.XSHG"] == 0.0071
        assert strategy.g.buy_date["513880.XSHG"] == date(2019, 10, 18)
        assert strategy.g.last_scores["513880.XSHG"] == {"sell_score": 45}
    finally:
        if old_g is None:
            del strategy.g
        else:
            strategy.g = old_g


def test_sell_state_is_cleared_only_after_position_is_flat():
    class Portfolio(object):
        positions = {}

    class Context(object):
        portfolio = Portfolio()

    old_g = strategy.g if hasattr(strategy, "g") else None
    strategy.g = types.SimpleNamespace(
        highest_since_buy={"513880.XSHG": 1.092},
        entry_atr={"513880.XSHG": 0.0071},
        buy_date={"513880.XSHG": date(2019, 10, 18)},
        last_scores={"513880.XSHG": {"sell_score": 45}},
    )
    try:
        strategy.sync_sell_state_after_order("513880.XSHG", Context())

        assert "513880.XSHG" not in strategy.g.highest_since_buy
        assert "513880.XSHG" not in strategy.g.entry_atr
        assert "513880.XSHG" not in strategy.g.buy_date
        assert "513880.XSHG" not in strategy.g.last_scores
    finally:
        if old_g is None:
            del strategy.g
        else:
            strategy.g = old_g


def test_has_position_does_not_probe_missing_position_with_get():
    class Positions(dict):
        def get(self, key, default=None):
            raise AssertionError("positions.get() probes missing JoinQuant positions")

    class Position(object):
        total_amount = 100

    class Portfolio(object):
        positions = Positions({"513880.XSHG": Position()})

    class Context(object):
        portfolio = Portfolio()

    assert strategy.has_position(Context(), "513880.XSHG") is True
    assert strategy.has_position(Context(), "159928.XSHE") is False


def test_buy_state_is_written_only_after_position_exists():
    class EmptyPortfolio(object):
        positions = {}

    class EmptyContext(object):
        portfolio = EmptyPortfolio()

    class Position(object):
        total_amount = 100

    class FilledPortfolio(object):
        positions = {"513880.XSHG": Position()}

    class FilledContext(object):
        portfolio = FilledPortfolio()

    old_g = strategy.g if hasattr(strategy, "g") else None
    strategy.g = types.SimpleNamespace(
        highest_since_buy={},
        entry_atr={},
        buy_date={},
    )
    try:
        strategy.sync_buy_state_after_order(
            "513880.XSHG", EmptyContext(), date(2019, 10, 18), 1.063, 0.0071)

        assert strategy.g.highest_since_buy == {}
        assert strategy.g.entry_atr == {}
        assert strategy.g.buy_date == {}

        strategy.sync_buy_state_after_order(
            "513880.XSHG", FilledContext(), date(2019, 10, 18), 1.063, 0.0071)

        assert strategy.g.highest_since_buy["513880.XSHG"] == 1.063
        assert strategy.g.entry_atr["513880.XSHG"] == 0.0071
        assert strategy.g.buy_date["513880.XSHG"] == date(2019, 10, 18)
    finally:
        if old_g is None:
            del strategy.g
        else:
            strategy.g = old_g


def test_buy_score_ignores_mixed_rsi_group_direction():
    snapshot = {
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi12_down": True,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "rsi6": 50.0,
    }

    score = strategy.score_buy_snapshot(snapshot)

    assert score["reversal_score"] == 16
    assert score["buy_score"] == 16


def test_sell_score_ignores_mixed_rsi_group_direction():
    snapshot = {
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi12_down": True,
        "macd_cross_down": True,
        "kdj_k_cross_down": True,
    }

    score = strategy.score_sell_snapshot(snapshot)

    assert score["sell_reversal_score"] == 16
    assert score["sell_score"] == 16


def test_buy_score_does_not_add_widening_positive_confirmations_without_cross():
    snapshot = {
        "rsi6": 60.0,
        "rsi6_prev": 55.0,
        "rsi12": 50.0,
        "rsi12_prev": 48.0,
        "rsi24": 45.0,
        "rsi24_prev": 43.0,
        "dif": 0.20,
        "dif_prev": 0.10,
        "dea": 0.10,
        "dea_prev": 0.05,
        "k": 60.0,
        "k_prev": 55.0,
        "d": 50.0,
        "d_prev": 50.0,
        "j": 70.0,
        "j_prev": 60.0,
    }

    score = strategy.score_buy_snapshot(snapshot)

    assert score["reversal_score"] == 0
    assert score["buy_score"] == 0


def test_buy_score_does_not_double_count_confirmations_after_strict_cross():
    snapshot = {
        "rsi6": 60.0,
        "rsi6_prev": 55.0,
        "rsi12": 50.0,
        "rsi12_prev": 48.0,
        "rsi24": 45.0,
        "rsi24_prev": 43.0,
        "dif": 0.20,
        "dif_prev": 0.10,
        "dea": 0.10,
        "dea_prev": 0.05,
        "k": 60.0,
        "k_prev": 55.0,
        "d": 50.0,
        "d_prev": 50.0,
        "j": 70.0,
        "j_prev": 60.0,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
    }

    score = strategy.score_buy_snapshot(snapshot)

    assert score["reversal_score"] == 45
    assert score["buy_score"] == 45


def test_candidate_order_is_score_reversal_then_code():
    candidates = [
        {"code": "BBB", "buy_score": 70, "reversal_score": 20},
        {"code": "AAA", "buy_score": 70, "reversal_score": 25},
        {"code": "CCC", "buy_score": 75, "reversal_score": 10},
    ]

    ordered = strategy.sort_candidates(candidates)

    assert [c["code"] for c in ordered] == ["CCC", "AAA", "BBB"]


def test_default_params_evaluate_signals_every_trading_weekday():
    params = strategy.get_default_params()

    assert params["rebalance_weekdays"] == [0, 1, 2, 3, 4]


def test_default_params_use_training_selected_broad_base_ratio():
    params = strategy.get_default_params()

    assert params["base_ratio"] == 0.95


def test_default_params_use_half_size_for_a_share_zero_volume_buys():
    params = strategy.get_default_params()

    assert params["a_share_zero_volume_buy_scale"] == 0.50


def test_default_etf_pool_uses_joinquant_confirmed_training_candidate():
    assert strategy.STRATEGY_VERSION == "cross-v0.3.2"
    assert strategy.get_default_etf_pool() == [
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


def test_buy_position_scale_halves_only_a_share_zero_volume_candidates():
    params = strategy.get_default_params()

    assert strategy.buy_position_scale({"code": "510300", "volume_score": 0}, params) == 0.50
    assert strategy.buy_position_scale({"code": "159915", "volume_score": 0}, params) == 0.50
    assert strategy.buy_position_scale({"code": "513100", "volume_score": 0}, params) == 1.0
    assert strategy.buy_position_scale({"code": "518880", "volume_score": 0}, params) == 1.0
    assert strategy.buy_position_scale({"code": "510300", "volume_score": 6}, params) == 1.0


def test_buy_target_value_applies_position_scale_after_base_allocation():
    params = strategy.get_default_params()
    score = {"code": "510300", "volume_score": 0}

    target = strategy.calc_buy_target_value(20000.0, score, params)

    assert target == pytest.approx(20000.0 * 0.95 / 3 * 0.50)


def test_buy_candidates_exclude_force_sell_conflicts():
    scores = [
        {
            "code": "CONFLICT",
            "buy_allowed": True,
            "buy_score": 75,
            "sell_score": 30,
        },
        {
            "code": "OK",
            "buy_allowed": True,
            "buy_score": 65,
            "sell_score": 18,
            "close_near_ma20": True,
            "close_far_above_ma20": False,
        },
        {
            "code": "LOW",
            "buy_allowed": True,
            "buy_score": 59,
            "sell_score": 0,
            "close_near_ma20": True,
            "close_far_above_ma20": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert [c["code"] for c in candidates] == ["OK"]


def test_buy_candidates_require_low_position_for_new_entries():
    scores = [
        {
            "code": "HIGH_POSITION",
            "buy_allowed": True,
            "buy_score": 72,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": True,
        },
        {
            "code": "LOW_POSITION",
            "buy_allowed": True,
            "buy_score": 65,
            "sell_score": 0,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert [c["code"] for c in candidates] == ["LOW_POSITION"]


def test_buy_candidates_accept_ma20_repair_position_for_new_entries():
    scores = [
        {
            "code": "MA20_REPAIR",
            "buy_allowed": True,
            "buy_score": 62,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": True,
            "close_far_above_ma20": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert [c["code"] for c in candidates] == ["MA20_REPAIR"]


def test_buy_candidates_block_validated_macd_rsi_volume_combo_without_kdj():
    scores = [
        {
            "code": "BLOCKED_COMBO",
            "buy_allowed": True,
            "buy_score": 70,
            "sell_score": 0,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": False,
            "rsi6_cross_rsi12_up": True,
            "rsi6_cross_rsi24_up": False,
            "macd_cross_up": True,
            "kdj_k_cross_up": False,
            "kdj_j_cross_up": False,
            "trend_score": 12,
            "volume_score": 6,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert candidates == []
    assert strategy.is_blocked_entry_combo(scores[0])


def test_buy_candidates_keep_validated_combo_when_kdj_or_strong_trend_confirms():
    with_kdj = {
        "code": "WITH_KDJ",
        "buy_allowed": True,
        "buy_score": 70,
        "sell_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "rsi6_cross_rsi12_up": True,
        "macd_cross_up": True,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "trend_score": 12,
        "volume_score": 6,
    }
    strong_trend = dict(with_kdj, code="STRONG_TREND", kdj_k_cross_up=False, trend_score=20)

    candidates = strategy.filter_buy_candidates(
        [with_kdj, strong_trend],
        held_codes=[],
        params=strategy.get_default_params(),
    )

    assert [c["code"] for c in candidates] == ["WITH_KDJ", "STRONG_TREND"]
    assert not strategy.is_blocked_entry_combo(with_kdj)
    assert not strategy.is_blocked_entry_combo(strong_trend)


def test_archived_weak_buy_candidate_does_not_relax_buy_threshold():
    scores = [
        {
            "code": "LOW_QUALITY_REVERSAL",
            "buy_allowed": True,
            "buy_score": 57,
            "reversal_score": 35,
            "sell_score": 0,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": False,
        },
        {
            "code": "MA20_ONLY_WEAK",
            "buy_allowed": True,
            "buy_score": 57,
            "reversal_score": 35,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": True,
            "close_far_above_ma20": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert candidates == []


def test_weak_buy_candidate_rejects_high_position_low_reversal_and_sell_conflict():
    scores = [
        {
            "code": "HIGH_WEAK",
            "buy_allowed": True,
            "buy_score": 58,
            "reversal_score": 35,
            "sell_score": 0,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_far_above_ma20": True,
        },
        {
            "code": "LOW_BUT_LOW_REVERSAL",
            "buy_allowed": True,
            "buy_score": 58,
            "reversal_score": 24,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": True,
            "close_far_above_ma20": False,
        },
        {
            "code": "LOW_BUT_SELL_CONFLICT",
            "buy_allowed": True,
            "buy_score": 58,
            "reversal_score": 35,
            "sell_score": 30,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_far_above_ma20": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert candidates == []


def test_archived_adx_buy_uptrend_rule_does_not_relax_entry_position():
    scores = [
        {
            "code": "TREND_ENTRY",
            "buy_allowed": True,
            "buy_score": 64,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": False,
            "close_gt_ma20": True,
            "adx": 31.0,
            "plus_di": 34.0,
            "minus_di": 15.0,
            "ma20_slope_non_negative": True,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert candidates == []


def test_archived_adx_buy_downtrend_rule_does_not_override_ma20_repair():
    scores = [
        {
            "code": "WEAK_REPAIR",
            "buy_allowed": True,
            "buy_score": 64,
            "sell_score": 0,
            "close_between_boll_lower_mid": False,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": True,
            "close_far_above_ma20": False,
            "adx": 31.0,
            "plus_di": 14.0,
            "minus_di": 35.0,
            "ma20_slope_non_negative": False,
        },
        {
            "code": "LOW_REVERSAL",
            "buy_allowed": True,
            "buy_score": 64,
            "sell_score": 0,
            "close_between_boll_lower_mid": True,
            "close_cross_boll_mid_up": False,
            "close_near_ma20": False,
            "close_far_above_ma20": False,
            "adx": 31.0,
            "plus_di": 14.0,
            "minus_di": 35.0,
            "ma20_slope_non_negative": False,
        },
    ]

    candidates = strategy.filter_buy_candidates(scores, held_codes=[], params=strategy.get_default_params())

    assert [c["code"] for c in candidates] == ["WEAK_REPAIR", "LOW_REVERSAL"]


def test_same_day_buy_blocks_signal_sell():
    assert not strategy.can_sell_by_signal(date(2026, 7, 2), date(2026, 7, 2))
    assert strategy.can_sell_by_signal(date(2026, 7, 1), date(2026, 7, 2))


def test_default_params_use_one_week_min_signal_hold():
    params = strategy.get_default_params()

    assert params["min_signal_hold_days"] == 5


def test_signal_sell_requires_minimum_trading_day_hold():
    trade_days = [
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 3),
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
    ]

    assert not strategy.can_sell_by_signal(
        date(2026, 7, 1),
        date(2026, 7, 7),
        min_hold_days=5,
        trade_days=trade_days,
    )
    assert strategy.can_sell_by_signal(
        date(2026, 7, 1),
        date(2026, 7, 8),
        min_hold_days=5,
        trade_days=trade_days,
    )


def test_score_skip_reason_reports_short_data_and_nan_fields():
    short_df = strategy.pd.DataFrame({"close": [1.0], "volume": [100.0]})
    assert strategy.score_skip_reason(short_df, None, ["rsi6"], min_len=3) == "short_data:1<3"

    df = strategy.pd.DataFrame({"close": [1.0, 1.1, 1.2], "volume": [100.0, 100.0, 100.0]})
    snapshot = {"rsi6": strategy.np.nan, "atr": 0.1}
    assert strategy.score_skip_reason(df, snapshot, ["rsi6", "atr"], min_len=3) == "nan_fields:rsi6"


def test_summarize_cross_signal_candidates_lists_rev_positive_only():
    scores = [
        {"code": "NO", "buy_score": 41, "reversal_score": 0},
        {"code": "LOW", "buy_score": 35, "reversal_score": 12},
        {"code": "HIGH", "buy_score": 50, "reversal_score": 24},
    ]

    summary = strategy.summarize_cross_signal_candidates(scores, limit=2)

    assert summary["count"] == 2
    assert [item["code"] for item in summary["items"]] == ["HIGH", "LOW"]


def test_summarize_loose_reversal_candidates_uses_turning_signals_only():
    scores = [
        {
            "code": "FLAT",
            "buy_score": 41,
            "reversal_score": 0,
            "rsi6": 50.0,
            "rsi6_prev": 50.0,
            "dif": 0.1,
            "dif_prev": 0.1,
            "k": 30.0,
            "k_prev": 30.0,
            "j": 20.0,
            "j_prev": 20.0,
        },
        {
            "code": "ONE",
            "buy_score": 35,
            "reversal_score": 0,
            "rsi6": 51.0,
            "rsi6_prev": 49.0,
            "dif": 0.1,
            "dif_prev": 0.1,
            "k": 30.0,
            "k_prev": 30.0,
            "j": 20.0,
            "j_prev": 20.0,
        },
        {
            "code": "THREE",
            "buy_score": 38,
            "reversal_score": 0,
            "rsi6": 51.0,
            "rsi6_prev": 49.0,
            "dif": 0.12,
            "dif_prev": 0.10,
            "k": 31.0,
            "k_prev": 30.0,
            "j": 22.0,
            "j_prev": 20.0,
        },
    ]

    summary = strategy.summarize_loose_reversal_candidates(scores, limit=2)

    assert summary["count"] == 2
    assert [item["code"] for item in summary["items"]] == ["THREE", "ONE"]
    assert summary["items"][0]["loose_reversal_count"] == 3
    assert summary["items"][0]["rsi6_delta"] == 2.0
    assert summary["items"][0]["dif_delta"] == 0.02
    assert summary["items"][0]["kdj_turn_up"]


def test_format_indicator_values_includes_all_visual_lines():
    item = {
        "rsi6": 50.1,
        "rsi6_prev": 49.0,
        "rsi12": 48.2,
        "rsi12_prev": 48.0,
        "rsi24": 46.3,
        "rsi24_prev": 46.8,
        "dif": 0.1234,
        "dif_prev": 0.1100,
        "dea": 0.1000,
        "dea_prev": 0.1050,
        "macd_hist": 0.0468,
        "k": 60.1,
        "k_prev": 56.0,
        "d": 55.2,
        "d_prev": 54.0,
        "j": 70.3,
        "j_prev": 62.0,
        "boll_upper": 11.0,
        "boll_mid": 10.0,
        "boll_lower": 9.0,
        "ma5": 10.5,
        "ma10": 10.2,
        "ma20": 10.0,
        "ma60": 9.5,
        "vol5": 1200.0,
        "vol20": 1000.0,
        "atr": 0.35,
        "plus_di": 28.0,
        "minus_di": 15.0,
        "adx": 31.0,
    }

    text = strategy.format_indicator_values(item)

    assert "RSI[6/12/24]=50.1/48.2/46.3" in text
    assert "MACD[DIF/DEA/HIST]=0.1234/0.1000/0.0468" in text
    assert "KDJ[K/D/J]=60.1/55.2/70.3" in text
    assert "BOLL[U/M/L]=11.000/10.000/9.000" in text
    assert "MA[5/10/20/60]=10.500/10.200/10.000/9.500" in text
    assert "VOL[5/20]=1200/1000" in text
    assert "ATR14=0.3500" in text
    assert "DMI[+DI/-DI/ADX]=28.0/15.0/31.0" in text
    assert "RSI_DIFF[6-12/6-24]=1.9/3.8(prev 1.0/2.2)" in text
    assert "MACD_DIFF[DIF-DEA]=0.0234(prev 0.0050)" in text
    assert "KDJ_DIFF[K-D/J-D]=4.9/15.1(prev 2.0/8.0)" in text


def test_format_cross_flags_shows_rsi_and_kdj_detail():
    item = {
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": True,
        "rsi6_cross_rsi12_down": True,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": True,
        "macd_cross_down": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": False,
        "kdj_k_cross_down": False,
        "kdj_j_cross_down": True,
    }

    text = strategy.format_cross_flags(item)

    assert "RSI12_UP=False" in text
    assert "RSI24_UP=True" in text
    assert "RSI12_DOWN=True" in text
    assert "RSI24_DOWN=False" in text
    assert "MACD_UP=True" in text
    assert "MACD_DOWN=False" in text
    assert "KDJ_K_UP=True" in text
    assert "KDJ_J_UP=False" in text
    assert "KDJ_K_DOWN=False" in text
    assert "KDJ_J_DOWN=True" in text


def test_format_self_check_reports_version_and_diff_cross_status():
    text = strategy.format_self_check()

    assert "[cross-v0.3.2] positional-diff-cross enabled" in text
    assert "diff_cross_self_check=True expected=True" in text
    assert "self_rev=12" in text


if __name__ == "__main__":
    for test in [
        test_recent_cross_detection_uses_last_three_days,
        test_cross_detection_matches_logged_diff_semantics,
        test_cross_detection_uses_position_not_series_index_alignment,
        test_cross_detection_ignores_jqdata_any_global_pollution,
        test_recent_cross_detection_uses_latest_cross_direction,
        test_rsi_handles_one_way_and_flat_series_boundaries,
        test_dmi_adx_identifies_directional_uptrend,
        test_buy_score_matches_v01_components,
        test_overheated_buy_is_blocked_without_rewriting_score,
        test_sell_score_and_force_threshold,
        test_signal_sell_requires_structure_confirmation,
        test_signal_sell_confirmed_by_ma20_break,
        test_strong_adx_uptrend_blocks_nonsevere_signal_sell,
        test_strong_adx_uptrend_does_not_block_severe_structure_sell,
        test_atr_stop_sells_without_signal_confirmation,
        test_risk_warning_does_not_change_mainline_stop_price,
        test_check_atr_stops_ignores_archived_risk_tightened_state,
        test_sell_state_is_kept_when_order_does_not_change_position,
        test_sell_state_is_cleared_only_after_position_is_flat,
        test_buy_state_is_written_only_after_position_exists,
        test_buy_score_ignores_mixed_rsi_group_direction,
        test_sell_score_ignores_mixed_rsi_group_direction,
        test_buy_score_does_not_add_widening_positive_confirmations_without_cross,
        test_buy_score_does_not_double_count_confirmations_after_strict_cross,
        test_candidate_order_is_score_reversal_then_code,
        test_default_params_evaluate_signals_every_trading_weekday,
        test_buy_candidates_exclude_force_sell_conflicts,
        test_buy_candidates_require_low_position_for_new_entries,
        test_buy_candidates_accept_ma20_repair_position_for_new_entries,
        test_buy_candidates_block_validated_macd_rsi_volume_combo_without_kdj,
        test_buy_candidates_keep_validated_combo_when_kdj_or_strong_trend_confirms,
        test_archived_weak_buy_candidate_does_not_relax_buy_threshold,
        test_weak_buy_candidate_rejects_high_position_low_reversal_and_sell_conflict,
        test_archived_adx_buy_uptrend_rule_does_not_relax_entry_position,
        test_archived_adx_buy_downtrend_rule_does_not_override_ma20_repair,
        test_same_day_buy_blocks_signal_sell,
        test_score_skip_reason_reports_short_data_and_nan_fields,
        test_summarize_cross_signal_candidates_lists_rev_positive_only,
        test_summarize_loose_reversal_candidates_uses_turning_signals_only,
        test_format_indicator_values_includes_all_visual_lines,
        test_format_cross_flags_shows_rsi_and_kdj_detail,
        test_format_self_check_reports_version_and_diff_cross_status,
    ]:
        test()
