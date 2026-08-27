import ast
import copy
import importlib.util
import inspect
import json
import pathlib
import sys
import textwrap
import types
from datetime import date

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
STRATEGY_PATH = (
    ROOT
    / "resonance_reversal_strategy"
    / "smart_trade_joinquant_resonance_reversal_etf.py"
)
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))
spec = importlib.util.spec_from_file_location("resonance_strategy", STRATEGY_PATH)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


EXPECTED_POOL = [
    "510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE",
    "510880.XSHG", "513100.XSHG", "513500.XSHG", "159920.XSHE",
    "513880.XSHG", "513050.XSHG", "518880.XSHG", "159985.XSHE",
]


def fake_position(amount):
    return types.SimpleNamespace(total_amount=amount)


def fake_context(previous_date="2021-01-05", current_date="2021-01-06",
                 positions=None, total_value=20000.0, available_cash=20000.0):
    return types.SimpleNamespace(
        previous_date=previous_date,
        current_dt=pd.Timestamp(current_date),
        portfolio=types.SimpleNamespace(
            positions={} if positions is None else positions,
            total_value=total_value,
            available_cash=available_cash,
        ),
    )


def current_record(price=10.0, paused=False):
    return types.SimpleNamespace(last_price=price, paused=paused)


def test_default_contract_is_frozen():
    assert strategy.STRATEGY_VERSION == "resonance-v0.1.0"
    assert strategy.get_default_etf_pool() == EXPECTED_POOL
    params = strategy.get_default_params()
    assert params["lookback_days"] == 120
    assert params["max_holdings"] == 3
    assert params["target_exposure"] == pytest.approx(0.95)
    assert params["resonance_window"] == 2
    assert params["rsi_period"] == 14
    assert params["kdj"] == (9, 3, 3)
    assert params["boll"] == (20, 2.0)
    assert params["atr_period"] == 14
    assert params["atr_multiplier"] == pytest.approx(2.5)
    assert params["stop_floor"] == pytest.approx(0.05)
    assert params["stop_cap"] == pytest.approx(0.15)


def test_initialize_enables_future_guard_and_fixed_schedules(monkeypatch):
    calls = []
    monkeypatch.setattr(strategy, "set_option", lambda k, v: calls.append(("option", k, v)), raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda code: calls.append(("benchmark", code)), raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: ("slippage", value), raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda value, type=None: calls.append(("slippage", value, type)), raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kw: kw, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda value, type=None: calls.append(("cost", value, type)), raising=False)
    monkeypatch.setattr(strategy, "run_daily", lambda fn, time, reference_security=None: calls.append(("daily", fn.__name__, time)), raising=False)
    monkeypatch.setattr(strategy, "log", types.SimpleNamespace(info=lambda *args: None), raising=False)
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    assert ("option", "use_real_price", True) in calls
    assert ("option", "avoid_future_data", True) in calls
    assert ("benchmark", "000300.XSHG") in calls
    assert ("daily", "do_trading", "09:35") in calls
    assert ("daily", "after_close", "15:30") in calls


def test_ensure_runtime_state_initializes_required_state(monkeypatch):
    runtime = types.SimpleNamespace()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.ensure_runtime_state()

    assert runtime.params["lookback_days"] == 120
    assert runtime.etf_pool == EXPECTED_POOL
    assert runtime.position_states == {}
    assert runtime.processed_resonance_ids == {}
    assert runtime.observation_events == {}
    assert runtime.sold_today == set()
    assert runtime.daily_attempted_buys == set()


def test_do_trading_initializes_runtime_state(monkeypatch):
    runtime = types.SimpleNamespace()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: [])
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *args: set(), raising=False)
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *args: {}, raising=False)
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: set(), raising=False)
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: [], raising=False)

    strategy.do_trading(fake_context())

    assert runtime.etf_pool == EXPECTED_POOL
    assert runtime.position_states == {}


def test_after_close_initializes_runtime_state(monkeypatch):
    runtime = types.SimpleNamespace()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)

    strategy.after_close(fake_context())

    assert runtime.etf_pool == EXPECTED_POOL
    assert runtime.observation_events == {}


import numpy as np
import pandas as pd


def make_ohlcv_frame(rows):
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(np.linspace(10.0, 20.0, rows), index=index)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(1, rows + 1, dtype=float) * 1000.0,
    }, index=index)


def test_rsi_wilder_edges_and_turn_values():
    rising = pd.Series(range(1, 40), dtype=float)
    falling = pd.Series(range(40, 1, -1), dtype=float)
    flat = pd.Series([10.0] * 40)
    assert strategy.calc_rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert strategy.calc_rsi(falling, 14).iloc[-1] == pytest.approx(0.0)
    assert strategy.calc_rsi(flat, 14).iloc[-1] == pytest.approx(50.0)
    assert strategy.calc_rsi(pd.Series([np.nan] * 20), 14).isna().all()


def test_boll_uses_population_std_and_atr_uses_simple_mean():
    close = pd.Series(np.arange(1.0, 31.0))
    mid, upper, lower = strategy.calc_bollinger(close, 20, 2.0)
    window = close.iloc[-20:]
    assert mid.iloc[-1] == pytest.approx(window.mean())
    assert upper.iloc[-1] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert lower.iloc[-1] == pytest.approx(window.mean() - 2 * window.std(ddof=0))

    high = pd.Series([11, 13, 12, 15, 16, 16, 18, 19, 18, 20, 21, 22, 23, 24, 25], dtype=float)
    low = high - 2
    close2 = high - 1
    atr = strategy.calc_atr(high, low, close2, 14)
    prev_close = close2.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    assert atr.iloc[-1] == pytest.approx(tr.iloc[-14:].mean())


def test_kdj_uses_rolling_rsv_and_flat_range_defaults_to_neutral():
    high = pd.Series([10, 12, 13, 14], dtype=float)
    low = pd.Series([8, 9, 10, 11], dtype=float)
    close = pd.Series([9, 11, 11, 13], dtype=float)
    k, d, j = strategy.calc_kdj(high, low, close, n=3, m1=2, m2=2)
    assert k.iloc[2] == pytest.approx(60.0)
    assert d.iloc[2] == pytest.approx(60.0)
    assert j.iloc[2] == pytest.approx(60.0)
    assert k.iloc[3] == pytest.approx(70.0)
    assert d.iloc[3] == pytest.approx(65.0)
    assert j.iloc[3] == pytest.approx(80.0)

    flat = pd.Series([10.0] * 5)
    flat_k, flat_d, flat_j = strategy.calc_kdj(flat, flat, flat, n=3)
    assert flat_k.iloc[-1] == pytest.approx(50.0)
    assert flat_d.iloc[-1] == pytest.approx(50.0)
    assert flat_j.iloc[-1] == pytest.approx(50.0)


def test_true_range_uses_high_low_and_previous_close_gaps():
    high = pd.Series([10.0, 13.0, 12.0])
    low = pd.Series([8.0, 9.0, 7.0])
    close = pd.Series([9.0, 10.0, 8.0])
    assert strategy.true_range(high, low, close).tolist() == pytest.approx([2.0, 4.0, 5.0])


def test_dmi_adx_rising_series_is_directional_and_flat_series_is_undefined():
    close = pd.Series(np.arange(1.0, 40.0))
    high = close + 1.0
    low = close - 1.0
    plus_di, minus_di, adx = strategy.calc_dmi_adx(high, low, close, period=14)
    assert plus_di.iloc[-1] == pytest.approx(47.00806581288051)
    assert minus_di.iloc[-1] == pytest.approx(0.0)
    assert adx.iloc[-1] == pytest.approx(100.0)

    flat = pd.Series([10.0] * 40)
    flat_plus, flat_minus, flat_adx = strategy.calc_dmi_adx(flat, flat, flat, period=14)
    assert flat_plus.iloc[14:].isna().all()
    assert flat_minus.iloc[14:].isna().all()
    assert flat_adx.iloc[14:].isna().all()


def test_indicator_frame_separates_trade_and_observation_columns():
    frame = strategy.build_indicator_frame(make_ohlcv_frame(140), strategy.get_default_params())
    assert set(strategy.TRADE_INDICATOR_COLUMNS) == {
        "rsi14", "k", "d", "j", "kd_diff", "boll_mid",
        "boll_upper", "boll_lower", "atr14",
    }
    assert set(strategy.OBSERVATION_COLUMNS) == {
        "rsi6", "rsi12", "rsi24", "plus_di", "minus_di", "adx14",
        "volume", "volume_ma5", "volume_ma20", "volume_ratio",
        "boll_width", "boll_mid_slope",
    }
    assert set(strategy.TRADE_INDICATOR_COLUMNS).isdisjoint(strategy.OBSERVATION_COLUMNS)


def test_indicator_frame_populates_trade_and_observation_values():
    index = pd.bdate_range("2020-01-01", periods=31)
    close = pd.Series(list(range(1, 31)) + [29.0], index=index)
    price_frame = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": pd.Series(np.arange(1, 32, dtype=float) * 100.0, index=index),
    }, index=index)

    frame = strategy.build_indicator_frame(price_frame, strategy.get_default_params())

    assert set(strategy.TRADE_INDICATOR_COLUMNS) <= set(frame.columns)
    assert set(strategy.OBSERVATION_COLUMNS) <= set(frame.columns)
    assert frame["rsi6"].iloc[-1] == pytest.approx(83.33333333333333)
    assert frame["rsi12"].iloc[-1] == pytest.approx(91.66666666666667)
    assert frame["rsi24"].iloc[-1] == pytest.approx(95.83333333333333)
    assert frame["volume_ma5"].iloc[-1] == pytest.approx(2900.0)
    assert frame["volume_ma20"].iloc[-1] == pytest.approx(2150.0)
    assert frame["volume_ratio"].iloc[-1] == pytest.approx(3100.0 / 2150.0)
    assert frame["boll_width"].iloc[-1] == pytest.approx(1.0497286790354372)
    assert frame["boll_mid_slope"].iloc[-1] == pytest.approx(0.9)


def make_event(indicator, direction, event_date, expires_date,
               reference_extreme=None):
    return strategy.make_turn_event(
        indicator=indicator,
        direction=direction,
        event_date=event_date,
        expires_date=expires_date,
        trigger_values={"fixture": True},
        reference_extreme=reference_extreme,
    )


def relative_indicator_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.to_datetime([
            "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
        ][:len(rows)]),
    )


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        ({"rsi14": 28.0}, {"rsi14": 29.0}, strategy.TurnDirection.BUY_TURN),
        ({"rsi14": 72.0}, {"rsi14": 71.0}, strategy.TurnDirection.SELL_TURN),
        ({"rsi14": 31.0}, {"rsi14": 32.0}, strategy.TurnDirection.NEUTRAL),
    ],
)
def test_rsi_event_does_not_require_threshold_cross(previous, current, expected):
    assert strategy.detect_rsi_direction(
        previous, current, strategy.get_default_params(),
    ) is expected


def test_kdj_buy_turn_can_precede_formal_golden_cross():
    previous = {"k": 15.0, "d": 20.0, "j": 5.0, "kd_diff": -5.0}
    current = {"k": 17.0, "d": 20.0, "j": 11.0, "kd_diff": -3.0}

    assert current["k"] < current["d"]
    assert strategy.detect_kdj_direction(
        previous, current, strategy.get_default_params(),
    ) is strategy.TurnDirection.BUY_TURN


def test_boll_touch_without_return_inside_is_neutral():
    previous = {
        "low": 9.0, "high": 10.0, "close": 9.2,
        "boll_lower": 9.3, "boll_upper": 11.0,
    }
    current = {
        "low": 8.8, "high": 9.5, "close": 9.0,
        "boll_lower": 9.1, "boll_upper": 10.8,
    }

    assert strategy.detect_boll_direction(
        previous, current,
    ) is strategy.TurnDirection.NEUTRAL


@pytest.mark.parametrize(
    "values,expected",
    [
        ((45.0, 40.0, 41.0), strategy.TurnDirection.BUY_TURN),
        ((55.0, 60.0, 59.0), strategy.TurnDirection.SELL_TURN),
        ((40.0, 40.0, 40.0), strategy.TurnDirection.NEUTRAL),
        ((np.nan, 40.0, 41.0), strategy.TurnDirection.NEUTRAL),
    ],
)
def test_relative_rsi_uses_local_turn_without_fixed_threshold(values, expected):
    older, middle, current = ({"rsi14": value} for value in values)
    assert strategy.detect_relative_rsi_direction(
        older, middle, current,
    ) is expected


@pytest.mark.parametrize(
    "older,middle,current,expected",
    [
        (
            {"j": 45.0, "kd_diff": -1.0},
            {"j": 40.0, "kd_diff": -2.0},
            {"j": 41.0, "kd_diff": -1.5},
            strategy.TurnDirection.BUY_TURN,
        ),
        (
            {"j": 55.0, "kd_diff": 2.0},
            {"j": 60.0, "kd_diff": 3.0},
            {"j": 59.0, "kd_diff": 2.5},
            strategy.TurnDirection.SELL_TURN,
        ),
        (
            {"j": 45.0, "kd_diff": -1.0},
            {"j": 40.0, "kd_diff": -2.0},
            {"j": 41.0, "kd_diff": -2.5},
            strategy.TurnDirection.NEUTRAL,
        ),
    ],
)
def test_relative_kdj_requires_j_and_kd_diff_to_turn_together(
    older, middle, current, expected):
    assert strategy.detect_relative_kdj_direction(
        older, middle, current,
    ) is expected


def test_relative_boll_uses_percent_b_turn_without_touching_band():
    older = {
        "close": 9.6, "low": 9.4, "high": 9.8,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    middle = {
        "close": 9.0, "low": 8.8, "high": 9.3,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    current = {
        "close": 9.2, "low": 8.8, "high": 9.5,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }

    assert middle["low"] > middle["boll_lower"]
    assert strategy.detect_relative_boll_direction(
        older, middle, current,
    ) is strategy.TurnDirection.BUY_TURN


def test_relative_boll_rejects_zero_width_nonfinite_and_new_extreme():
    valid = {
        "close": 9.0, "low": 8.8, "high": 9.3,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    zero_width = dict(valid, boll_upper=8.0, boll_lower=8.0)
    nonfinite = dict(valid, close=np.inf)
    lower_low = dict(valid, close=9.2, low=8.7)

    assert strategy._boll_percent_b(zero_width) is None
    assert strategy._boll_percent_b(nonfinite) is None
    assert strategy.detect_relative_boll_direction(
        dict(valid, close=9.6), valid, lower_low,
    ) is strategy.TurnDirection.NEUTRAL


def test_relative_event_book_detects_t2_and_t1_from_last_four_complete_bars():
    rows = [
        {"rsi14": 50.0, "j": 50.0, "kd_diff": 0.0,
         "close": 9.8, "low": 9.6, "high": 10.0,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 45.0, "j": 45.0, "kd_diff": -1.0,
         "close": 9.4, "low": 9.2, "high": 9.7,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 46.0, "j": 40.0, "kd_diff": -2.0,
         "close": 9.0, "low": 8.8, "high": 9.3,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 47.0, "j": 41.0, "kd_diff": -1.5,
         "close": 9.2, "low": 8.8, "high": 9.5,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
    ]
    frame = relative_indicator_frame(rows)

    book = strategy.collect_latest_relative_events(
        frame, date(2021, 1, 8), date(2021, 1, 11),
    )

    assert book is not strategy.empty_event_book()
    assert book["active"]["RSI"]["event_date"] == date(2021, 1, 7)
    assert book["active"]["RSI"]["expires_date"] == date(2021, 1, 8)
    assert book["active"]["KDJ"]["event_date"] == date(2021, 1, 8)
    assert book["active"]["KDJ"]["expires_date"] == date(2021, 1, 11)
    assert book["active"]["BOLL"]["event_date"] == date(2021, 1, 8)
    assert book["active"]["BOLL"]["expires_date"] == date(2021, 1, 11)
    assert all(
        event["event_mode"] == "RELATIVE"
        for event in book["active"].values()
    )
    assert book["active"]["BOLL"]["reference_extreme"] == pytest.approx(8.8)

    strategy.expire_events(book, date(2021, 1, 11))
    assert "RSI" not in book["active"]
    assert "KDJ" in book["active"]
    assert "BOLL" in book["active"]

    strategy.expire_events(book, date(2021, 1, 12))
    assert "KDJ" not in book["active"]
    assert "BOLL" not in book["active"]
    assert [event["invalid_reason"] for event in book["invalidated"]] == [
        "EVENT_EXPIRED", "EVENT_EXPIRED", "EVENT_EXPIRED",
    ]


def test_relative_opposite_event_replaces_only_relative_book():
    relative_book = strategy.empty_event_book()
    hard_book = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-07",
    )
    strategy.apply_event(relative_book, strategy.make_relative_turn_event(
        "RSI", strategy.TurnDirection.BUY_TURN,
        "2021-01-07", "2021-01-08", {"fixture": "buy"},
    ))
    strategy.apply_event(relative_book, strategy.make_relative_turn_event(
        "RSI", strategy.TurnDirection.SELL_TURN,
        "2021-01-08", "2021-01-11", {"fixture": "sell"},
    ))

    assert relative_book["active"]["RSI"]["direction"] is (
        strategy.TurnDirection.SELL_TURN
    )
    assert relative_book["invalidated"][-1]["invalid_reason"] == (
        "REPLACED_BY_OPPOSITE_EVENT"
    )
    assert hard_book["active"]["RSI"]["direction"] is (
        strategy.TurnDirection.BUY_TURN
    )


def test_relative_boll_invalidates_on_new_extreme_without_band_requirement():
    book = strategy.empty_event_book()
    strategy.apply_event(book, strategy.make_relative_turn_event(
        "BOLL", strategy.TurnDirection.BUY_TURN,
        "2021-01-07", "2021-01-08", {"fixture": True},
        reference_extreme=8.8,
    ))

    strategy.invalidate_relative_boll_structure(
        book, {"low": 8.7, "high": 9.5},
    )

    assert "BOLL" not in book["active"]
    assert book["invalidated"][-1]["invalid_reason"] == (
        "NEW_LOWER_LOW_AFTER_RELATIVE_TURN"
    )


def test_opposite_event_replaces_old_event_with_auditable_reason():
    book = strategy.empty_event_book()
    strategy.apply_event(book, make_event(
        "RSI", strategy.TurnDirection.BUY_TURN, "2021-01-04", "2021-01-05",
    ))
    strategy.apply_event(book, make_event(
        "RSI", strategy.TurnDirection.SELL_TURN, "2021-01-05", "2021-01-06",
    ))

    assert book["active"]["RSI"]["direction"] is strategy.TurnDirection.SELL_TURN
    assert len(book["invalidated"]) == 1
    assert book["invalidated"][0]["invalid_reason"] == "REPLACED_BY_OPPOSITE_EVENT"


def test_boll_lower_band_new_low_invalidates_old_buy_event():
    book = strategy.empty_event_book()
    strategy.apply_event(book, make_event(
        "BOLL", strategy.TurnDirection.BUY_TURN, "2021-01-04", "2021-01-05",
        reference_extreme=9.0,
    ))

    invalidated = strategy.invalidate_boll_structure(book, {
        "date": "2021-01-05", "close": 8.8, "low": 8.7,
        "boll_lower": 8.9, "high": 9.1, "boll_upper": 10.5,
    })

    assert "BOLL" not in book["active"]
    assert invalidated["invalid_reason"] == "NEW_LOWER_LOW_OUTSIDE_LOWER_BAND"


def test_boll_upper_band_new_high_invalidates_old_sell_event():
    book = strategy.empty_event_book()
    strategy.apply_event(book, make_event(
        "BOLL", strategy.TurnDirection.SELL_TURN, "2021-01-04", "2021-01-05",
        reference_extreme=11.0,
    ))

    invalidated = strategy.invalidate_boll_structure(book, {
        "date": "2021-01-05", "close": 11.2, "low": 10.9,
        "boll_lower": 9.5, "high": 11.4, "boll_upper": 11.1,
    })

    assert "BOLL" not in book["active"]
    assert invalidated["invalid_reason"] == "NEW_HIGHER_HIGH_OUTSIDE_UPPER_BAND"


def test_expired_event_is_invalidated_only_after_its_calendar_expiry_date():
    book = strategy.empty_event_book()
    strategy.apply_event(book, make_event(
        "KDJ", strategy.TurnDirection.BUY_TURN,
        pd.Timestamp("2021-01-08"), pd.Timestamp("2021-01-11"),
    ))

    strategy.expire_events(book, pd.Timestamp("2021-01-11"))
    assert "KDJ" in book["active"]

    strategy.expire_events(book, pd.Timestamp("2021-01-12"))
    assert book["invalidated"][0]["invalid_reason"] == "EVENT_EXPIRED"


def test_collect_latest_events_uses_trading_sessions_for_event_expiry():
    index = pd.to_datetime(["2021-01-07", "2021-01-08", "2021-01-11"])
    frame = pd.DataFrame({
        "open": [10.0, 10.0, 10.0], "high": [10.5, 10.5, 10.5],
        "low": [9.5, 9.5, 9.5], "close": [10.0, 10.0, 10.0],
        "rsi14": [28.0, 29.0, 28.0],
        "k": [50.0, 50.0, 50.0], "d": [50.0, 50.0, 50.0],
        "j": [50.0, 50.0, 50.0], "kd_diff": [0.0, 0.0, 0.0],
        "boll_lower": [9.0, 9.0, 9.0], "boll_upper": [11.0, 11.0, 11.0],
    }, index=index)

    book = strategy.collect_latest_events(
        frame, date(2021, 1, 11), date(2021, 1, 12),
    )

    assert book["active"]["RSI"]["event_date"] == date(2021, 1, 8)
    assert book["active"]["RSI"]["expires_date"] == date(2021, 1, 11)


def event_book_for_directions(boll, rsi, kdj, event_date, expires_date=None):
    if expires_date is None:
        expires_date = "2021-01-06"
    active = {}
    for indicator, direction_name in (("BOLL", boll), ("RSI", rsi), ("KDJ", kdj)):
        if direction_name == "NEUTRAL":
            continue
        active[indicator] = make_event(
            indicator,
            strategy.TurnDirection[direction_name],
            event_date,
            expires_date,
        )
    return {"active": active, "invalidated": []}


def relative_event_book_for_directions(
        boll, rsi, kdj, event_date, expires_date=None):
    if expires_date is None:
        expires_date = event_date
    active = {}
    for indicator, direction_name in (
            ("BOLL", boll), ("RSI", rsi), ("KDJ", kdj)):
        enum_direction = strategy.TurnDirection[direction_name]
        if enum_direction is strategy.TurnDirection.NEUTRAL:
            continue
        active[indicator] = strategy.make_relative_turn_event(
            indicator, enum_direction, event_date, expires_date,
            {"fixture": indicator},
            reference_extreme=(8.8 if indicator == "BOLL" else None),
        )
    return {"active": active, "invalidated": []}


def test_relative_branch_a_requires_hard_boll_and_relative_oscillator():
    hard = event_book_for_directions(
        "BUY_TURN", "NEUTRAL", "NEUTRAL", "2021-01-08",
        expires_date="2021-01-08",
    )
    relative = relative_event_book_for_directions(
        "NEUTRAL", "BUY_TURN", "NEUTRAL", "2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["branch"] == "HARD_BOLL_SOFT_OSC"
    assert observation["supporters"] == ("BOLL", "RSI")
    assert observation["hard_or_relative_source_by_indicator"] == {
        "BOLL": "HARD", "RSI": "RELATIVE",
    }
    assert observation["relative_observation_id"].startswith("RELATIVE:")


def test_relative_branch_b_requires_all_three_relative_indicators():
    hard = strategy.empty_event_book()
    relative = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", "2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["branch"] == "SOFT_ALL_THREE"
    assert observation["supporters"] == ("BOLL", "KDJ", "RSI")


@pytest.mark.parametrize(
    "hard_directions,relative_directions",
    [
        (("BUY_TURN", "SELL_TURN", "NEUTRAL"),
         ("NEUTRAL", "BUY_TURN", "NEUTRAL")),
        (("NEUTRAL", "NEUTRAL", "NEUTRAL"),
         ("BUY_TURN", "BUY_TURN", "SELL_TURN")),
        (("NEUTRAL", "NEUTRAL", "NEUTRAL"),
         ("BUY_TURN", "BUY_TURN", "NEUTRAL")),
    ],
)
def test_relative_candidate_rejects_opposite_or_incomplete_support(
        hard_directions, relative_directions):
    hard = event_book_for_directions(
        *hard_directions, event_date="2021-01-08",
        expires_date="2021-01-08",
    )
    relative = relative_event_book_for_directions(
        *relative_directions, event_date="2021-01-08",
        expires_date="2021-01-08",
    )
    if hard_directions == ("BUY_TURN", "SELL_TURN", "NEUTRAL"):
        support_only = event_book_for_directions(
            "BUY_TURN", "NEUTRAL", "NEUTRAL", event_date="2021-01-08",
            expires_date="2021-01-08",
        )
        assert strategy.build_relative_resonance_observation(
            "510300.XSHG", strategy.TurnDirection.BUY_TURN,
            support_only, relative, date(2021, 1, 8), 10.0,
        ) is not None
    assert strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    ) is None


def test_existing_complete_hard_resonance_suppresses_relative_candidate():
    hard = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-08",
        expires_date="2021-01-08",
    )
    relative = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", "2021-01-08",
    )
    assert strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    ) is None


def test_expired_support_event_does_not_generate_relative_candidate():
    hard = event_book_for_directions(
        "BUY_TURN", "NEUTRAL", "NEUTRAL", "2021-01-06",
        expires_date="2021-01-07",
    )
    relative = relative_event_book_for_directions(
        "NEUTRAL", "BUY_TURN", "NEUTRAL", "2021-01-08",
    )

    assert strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    ) is None


def test_expired_opposite_event_does_not_veto_relative_candidate():
    hard = event_book_for_directions(
        "SELL_TURN", "NEUTRAL", "NEUTRAL", "2021-01-06",
        expires_date="2021-01-07",
    )
    relative = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", "2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["branch"] == "SOFT_ALL_THREE"


def test_relative_candidate_accepts_t_minus_two_support_and_signal_expiry_boundary():
    hard = event_book_for_directions(
        "BUY_TURN", "NEUTRAL", "NEUTRAL", "2021-01-06",
        expires_date="2021-01-08",
    )
    relative = relative_event_book_for_directions(
        "NEUTRAL", "BUY_TURN", "NEUTRAL", "2021-01-08",
        expires_date="2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["supporter_event_dates"] == {
        "BOLL": date(2021, 1, 6), "RSI": date(2021, 1, 8),
    }
    assert observation["expires_date"] == date(2021, 1, 8)


def test_relative_fingerprint_is_deterministic_and_formal_fingerprints_are_frozen():
    params = strategy.get_default_params()
    self_check = strategy.run_event_logic_self_check(params)

    assert strategy._value_fingerprint(params) == "e1227fbd8b4a884e"
    assert strategy._value_fingerprint(
        strategy.get_default_etf_pool(),
    ) == "9123995edeb1ed84"
    assert strategy.business_config_fingerprint(
        params, strategy.get_default_etf_pool(),
    ) == "88fdf95966ea0368"
    assert strategy.event_logic_fingerprint(
        params, self_check,
    ) == "1c0b8a22f48c97c3"
    first = strategy.relative_observation_fingerprint()
    second = strategy.relative_observation_fingerprint()
    assert first == second
    assert len(first) == 16
    json.dumps(strategy.relative_observation_logic_contract(), sort_keys=True)


@pytest.mark.parametrize(
    "boll,rsi,kdj,buy_allowed,sell_allowed",
    [
        ("BUY_TURN", "BUY_TURN", "NEUTRAL", True, False),
        ("BUY_TURN", "NEUTRAL", "BUY_TURN", True, False),
        ("BUY_TURN", "BUY_TURN", "BUY_TURN", True, False),
        ("BUY_TURN", "BUY_TURN", "SELL_TURN", False, False),
        ("BUY_TURN", "SELL_TURN", "BUY_TURN", False, False),
        ("SELL_TURN", "SELL_TURN", "NEUTRAL", False, True),
        ("SELL_TURN", "NEUTRAL", "SELL_TURN", False, True),
        ("SELL_TURN", "SELL_TURN", "SELL_TURN", False, True),
        ("SELL_TURN", "SELL_TURN", "BUY_TURN", False, False),
        ("SELL_TURN", "BUY_TURN", "SELL_TURN", False, False),
        ("NEUTRAL", "BUY_TURN", "BUY_TURN", False, False),
    ],
)
def test_complete_resonance_truth_table(
        boll, rsi, kdj, buy_allowed, sell_allowed):
    events = event_book_for_directions(boll, rsi, kdj, "2021-01-05")

    buy = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN, events, "2021-01-05",
    )
    sell = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.SELL_TURN, events, "2021-01-05",
    )

    assert (buy is not None) is buy_allowed
    assert (sell is not None) is sell_allowed


def test_two_old_events_cannot_resonate():
    old_events = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-04",
    )

    assert strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        old_events, "2021-01-05",
    ) is None


def test_candidate_sort_prefers_support_count_then_boll_freshness_then_code():
    decisions = [
        {"code": "513100.XSHG", "support_count": 2, "boll_age": 0},
        {"code": "159915.XSHE", "support_count": 3, "boll_age": 1},
        {"code": "510300.XSHG", "support_count": 2, "boll_age": 0},
        {"code": "512100.XSHG", "support_count": 2, "boll_age": 1},
    ]

    assert [item["code"] for item in strategy.sort_buy_decisions(decisions)] == [
        "159915.XSHE", "510300.XSHG", "513100.XSHG", "512100.XSHG",
    ]


def test_observation_values_do_not_change_resonance_or_priority():
    ordinary = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-05",
    )
    changed_observations = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-05",
    )
    changed_observations["active"]["BOLL"]["trigger_values"] = {
        "rsi6": 1.0, "rsi12": 99.0, "rsi24": 50.0, "adx14": 100.0,
        "volume_ratio": 999.0, "boll_width": 0.001,
    }
    changed_observations["active"]["RSI"]["trigger_values"] = {
        "rsi6": 99.0, "rsi12": 1.0, "rsi24": 50.0, "adx14": 0.0,
        "volume_ratio": 0.001, "boll_width": 99.0,
    }

    ordinary_decision = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        ordinary, "2021-01-05",
    )
    changed_decision = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        changed_observations, "2021-01-05",
    )

    assert ordinary_decision["resonance_id"] == changed_decision["resonance_id"]
    assert ordinary_decision["support_count"] == changed_decision["support_count"]
    assert ordinary_decision["boll_age"] == changed_decision["boll_age"]
    assert strategy.sort_buy_decisions([changed_decision, ordinary_decision]) == [
        changed_decision, ordinary_decision,
    ]


def test_processed_id_is_retained_for_same_support_events_until_expiry():
    events = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-05",
    )
    first = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN, events, "2021-01-05",
    )
    repeated = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN, events, "2021-01-05",
    )
    processed = {}

    strategy.mark_resonance_processed(processed, first)
    strategy.mark_resonance_processed(processed, repeated)

    assert first["resonance_id"] == repeated["resonance_id"]
    assert strategy.prune_processed_resonance_ids(processed, "2021-01-06") == {
        first["resonance_id"]: date(2021, 1, 6),
    }


def test_processed_id_is_pruned_only_after_expiry():
    processed = {"expired": "2021-01-05", "still_active": "2021-01-06"}

    assert strategy.prune_processed_resonance_ids(processed, "2021-01-06") == {
        "still_active": date(2021, 1, 6),
    }
    assert strategy.prune_processed_resonance_ids(processed, "2021-01-07") == {}


def test_new_supporter_dates_create_new_resonance_id_after_old_expiry():
    old_events = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-05",
    )
    new_events = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-07",
    )
    old_decision = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        old_events, "2021-01-05",
    )
    new_decision = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        new_events, "2021-01-07",
    )
    processed = {}
    strategy.mark_resonance_processed(processed, old_decision)

    assert old_decision["resonance_id"] != new_decision["resonance_id"]
    assert new_decision["resonance_id"] not in strategy.prune_processed_resonance_ids(
        processed, "2021-01-07",
    )


@pytest.mark.parametrize(
    "total,cash,expected",
    [
        (20000.0, 20000.0, 20000.0 * 0.95 / 3),
        (30000.0, 4000.0, 2500.0),
        (30000.0, 1000.0, 0.0),
    ],
)
def test_buy_target_adapts_to_current_assets_and_preserves_cash(
        total, cash, expected):
    assert strategy.calc_buy_target_value(
        total, cash, strategy.get_default_params(),
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "anchor,entry_atr,expected_pct",
    [(100.0, 1.0, 0.05), (100.0, 4.0, 0.10), (100.0, 10.0, 0.15)],
)
def test_atr_stop_clamps_percentage(anchor, entry_atr, expected_pct):
    result = strategy.calc_stop_state(
        anchor, entry_atr, strategy.get_default_params(),
    )
    assert result["stop_pct"] == pytest.approx(expected_pct)
    assert result["stop_price"] == pytest.approx(anchor * (1 - expected_pct))


def test_atr_stop_rejects_nan_highest_close_anchor():
    assert strategy.calc_stop_state(
        float("nan"), 4.0, strategy.get_default_params(),
    ) is None


def test_highest_anchor_only_moves_up_on_close_and_entry_atr_stays_frozen():
    state = strategy.make_position_state("2021-01-05", 2.0, 100.0)

    strategy.update_highest_close_anchor(state, 105.0)
    strategy.update_highest_close_anchor(state, 102.0)

    assert state["highest_close_anchor"] == pytest.approx(105.0)
    assert state["entry_atr"] == pytest.approx(2.0)


def test_signal_sell_is_next_trade_day_only_but_atr_has_no_hold_lock():
    assert not strategy.can_signal_sell("2021-01-05", "2021-01-05")
    assert strategy.can_signal_sell("2021-01-05", "2021-01-06")


def test_daily_state_resets_only_when_decision_date_changes_and_prunes_ids(
        monkeypatch):
    runtime = types.SimpleNamespace(
        state_date=date(2021, 1, 5),
        sold_today={"510300.XSHG"},
        daily_attempted_buys={"159915.XSHE"},
        processed_resonance_ids={
            "expired": date(2021, 1, 5),
            "active": date(2021, 1, 6),
        },
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.reset_daily_state("2021-01-05", "2021-01-06")

    assert runtime.sold_today == {"510300.XSHG"}
    assert runtime.daily_attempted_buys == {"159915.XSHE"}
    assert runtime.processed_resonance_ids == {"active": date(2021, 1, 6)}

    strategy.reset_daily_state("2021-01-06", "2021-01-06")

    assert runtime.state_date == date(2021, 1, 6)
    assert runtime.sold_today == set()
    assert runtime.daily_attempted_buys == set()
    assert runtime.daily_retried_exits == set()


def test_daily_state_same_day_string_does_not_erase_anti_repeat_sets(
        monkeypatch):
    runtime = types.SimpleNamespace(
        state_date="2021-01-05",
        sold_today={"510300.XSHG"},
        daily_attempted_buys={"159915.XSHE"},
        daily_retried_exits={"512100.XSHG"},
        processed_resonance_ids={},
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.reset_daily_state(date(2021, 1, 5), date(2021, 1, 5))

    assert runtime.state_date == date(2021, 1, 5)
    assert runtime.sold_today == {"510300.XSHG"}
    assert runtime.daily_attempted_buys == {"159915.XSHE"}
    assert runtime.daily_retried_exits == {"512100.XSHG"}


@pytest.mark.parametrize("persisted_state_date", [
    "2021-01-05",
    pd.Timestamp("2021-01-05"),
])
def test_daily_state_new_calendar_day_clears_every_anti_repeat_set(
        monkeypatch, persisted_state_date):
    runtime = types.SimpleNamespace(
        state_date=persisted_state_date,
        sold_today={"510300.XSHG"},
        daily_attempted_buys={"159915.XSHE"},
        daily_retried_exits={"512100.XSHG"},
        processed_resonance_ids={},
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.reset_daily_state(date(2021, 1, 6), date(2021, 1, 5))

    assert runtime.state_date == date(2021, 1, 6)
    assert runtime.sold_today == set()
    assert runtime.daily_attempted_buys == set()
    assert runtime.daily_retried_exits == set()


@pytest.mark.parametrize(
    "side,before,after,target,tradability,order_amount,filled,expected",
    [
        ("BUY", 0, 0, 100, "PAUSED", None, None, "PAUSED"),
        ("BUY", 0, 0, 100, "UNKNOWN", None, None, "UNKNOWN"),
        ("BUY", 0, 100, 100, "TRADEABLE", 100, 100, "FILLED"),
        ("BUY", 0, 50, 100, "TRADEABLE", 100, 50, "PARTIAL"),
        ("BUY", 0, 50, 100, "TRADEABLE", 100, 0, "PARTIAL"),
        ("BUY", 0, 0, 100, "TRADEABLE", 100, 100, "NOT_FILLED"),
        ("BUY", 0, 0, 100, "TRADEABLE", 100, 0, "NOT_FILLED"),
        ("SELL", 100, 0, 0, "TRADEABLE", -100, -100, "FILLED"),
        ("SELL", 100, 40, 0, "TRADEABLE", -100, -60, "PARTIAL"),
        ("SELL", 100, 40, 0, "TRADEABLE", -100, 0, "PARTIAL"),
        ("SELL", 100, 100, 0, "TRADEABLE", -100, -100, "NOT_FILLED"),
        ("SELL", 100, 100, 0, "TRADEABLE", -100, 0, "NOT_FILLED"),
    ],
)
def test_order_outcome_truth_table(
        side, before, after, target, tradability, order_amount, filled,
        expected):
    order = None if order_amount is None else types.SimpleNamespace(
        amount=order_amount, filled=filled,
    )

    outcome = strategy.classify_order_outcome(
        strategy.OrderSide[side], before, after, target,
        strategy.Tradability[tradability], order,
    )

    assert outcome is strategy.OrderOutcome[expected]


def test_partial_buy_establishes_frozen_risk_state_and_consumes_daily_attempt(
        monkeypatch):
    runtime = types.SimpleNamespace(
        position_states={}, daily_attempted_buys=set(),
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    outcome = strategy.sync_buy_state_after_order(
        "510300.XSHG", strategy.OrderOutcome.PARTIAL,
        before_amount=0, after_amount=50, decision_date="2021-01-05",
        entry_atr=2.5, entry_price=10.2,
    )

    assert outcome is strategy.OrderOutcome.PARTIAL
    assert runtime.daily_attempted_buys == {"510300.XSHG"}
    assert runtime.position_states["510300.XSHG"] == {
        "buy_date": "2021-01-05",
        "entry_atr": 2.5,
        "highest_close_anchor": 10.2,
        "pending_exit": None,
    }


def test_not_filled_buy_consumes_daily_attempt_without_creating_position_state(
        monkeypatch):
    runtime = types.SimpleNamespace(
        position_states={}, daily_attempted_buys=set(),
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    outcome = strategy.sync_buy_state_after_order(
        "510300.XSHG", strategy.OrderOutcome.NOT_FILLED,
        before_amount=0, after_amount=0, decision_date="2021-01-05",
        entry_atr=2.5, entry_price=10.2,
    )

    assert outcome is strategy.OrderOutcome.NOT_FILLED
    assert runtime.daily_attempted_buys == {"510300.XSHG"}
    assert runtime.position_states == {}


@pytest.mark.parametrize(
    "outcome,actual_amount",
    [
        (strategy.OrderOutcome.PARTIAL, 40),
        (strategy.OrderOutcome.NOT_FILLED, 100),
    ],
)
def test_partial_or_not_filled_sell_preserves_risk_state_and_sets_pending_exit(
        monkeypatch, outcome, actual_amount):
    position_state = {
        "buy_date": "2021-01-05",
        "entry_atr": 2.0,
        "highest_close_anchor": 105.0,
        "pending_exit": None,
    }
    runtime = types.SimpleNamespace(
        position_states={"510300.XSHG": position_state}, sold_today=set(),
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    returned = strategy.sync_sell_state_after_order(
        "510300.XSHG", outcome, strategy.ExitReason.SIGNAL_EXIT,
        decision_date="2021-01-06", trigger_value=102.0,
        actual_amount=actual_amount,
    )

    assert returned is outcome
    assert runtime.sold_today == set()
    assert position_state == {
        "buy_date": "2021-01-05",
        "entry_atr": 2.0,
        "highest_close_anchor": 105.0,
        "pending_exit": {
            "created_date": "2021-01-06",
            "reason": strategy.ExitReason.SIGNAL_EXIT,
            "trigger_value": 102.0,
            "remaining_amount": actual_amount,
        },
    }


def test_pending_exit_can_upgrade_to_atr_but_cannot_downgrade_to_signal():
    state = strategy.make_position_state("2021-01-05", 2.0, 100.0)
    strategy.set_pending_exit(
        state, strategy.ExitReason.SIGNAL_EXIT, "2021-01-06", 99.0, 100,
    )

    upgraded = strategy.set_pending_exit(
        state, strategy.ExitReason.ATR_EXIT, "2021-01-07", 95.0, 80,
    )
    retained = strategy.set_pending_exit(
        state, strategy.ExitReason.SIGNAL_EXIT, "2021-01-08", 98.0, 80,
    )

    expected = {
        "created_date": "2021-01-07",
        "reason": strategy.ExitReason.ATR_EXIT,
        "trigger_value": 95.0,
        "remaining_amount": 80,
    }
    assert upgraded == expected
    assert retained == expected
    assert state["pending_exit"] == expected


def test_filled_sell_clears_state_and_marks_sold_today_only_when_flat(
        monkeypatch):
    runtime = types.SimpleNamespace(
        position_states={
            "510300.XSHG": strategy.make_position_state(
                "2021-01-05", 2.0, 100.0,
            ),
        },
        sold_today=set(),
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    outcome = strategy.sync_sell_state_after_order(
        "510300.XSHG", strategy.OrderOutcome.FILLED,
        strategy.ExitReason.ATR_EXIT, decision_date="2021-01-06",
        trigger_value=95.0, actual_amount=0,
    )

    assert outcome is strategy.OrderOutcome.FILLED
    assert runtime.position_states == {}
    assert runtime.sold_today == {"510300.XSHG"}


def test_retry_pending_exits_dispatches_only_existing_pending_state(
        monkeypatch):
    position_state = strategy.make_position_state("2021-01-05", 2.0, 100.0)
    position_state["pending_exit"] = {
        "created_date": "2021-01-06",
        "reason": strategy.ExitReason.ATR_EXIT,
        "trigger_value": 95.0,
        "remaining_amount": 40,
    }
    runtime = types.SimpleNamespace(
        position_states={
            "510300.XSHG": position_state,
            "159915.XSHE": strategy.make_position_state(
                "2021-01-05", 1.0, 5.0,
            ),
        },
        observation_events=types.MappingProxyType({}),
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    dispatched = []

    def fake_submit_sell(context, code, reason, trigger_value):
        dispatched.append((context, code, reason, trigger_value))
        return strategy.OrderOutcome.PARTIAL

    monkeypatch.setattr(strategy, "submit_sell", fake_submit_sell, raising=False)
    context = types.SimpleNamespace(current_dt="2021-01-07 09:35")

    results = strategy.retry_pending_exits(context, current_data=object())

    assert results == [("510300.XSHG", strategy.OrderOutcome.PARTIAL)]
    assert dispatched == [(
        context, "510300.XSHG", strategy.ExitReason.ATR_EXIT, 95.0,
    )]
    assert position_state["pending_exit"]["remaining_amount"] == 40


def test_partial_exit_keeps_all_position_risk_state(monkeypatch):
    position_state = {
        "buy_date": "2021-01-05",
        "entry_atr": 2.0,
        "highest_close_anchor": 105.0,
        "pending_exit": "ATR_EXIT",
    }
    runtime = types.SimpleNamespace(position_states={"510300.XSHG": position_state})
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    assert not strategy.clear_position_state_if_flat("510300.XSHG", actual_amount=100)
    assert runtime.position_states["510300.XSHG"] == position_state


def test_flat_position_clears_only_its_risk_state(monkeypatch):
    runtime = types.SimpleNamespace(position_states={"510300.XSHG": {"entry_atr": 2.0}})
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    assert strategy.clear_position_state_if_flat("510300.XSHG", actual_amount=0)
    assert runtime.position_states == {}


def resonance_snapshot(code, direction="BUY_TURN", signal_date="2021-01-05",
                       support_count=2):
    kdj = direction if support_count == 3 else "NEUTRAL"
    book = event_book_for_directions(
        direction, direction, kdj, signal_date,
    )
    return {
        "code": code,
        "valid": True,
        "signal_date": signal_date,
        "close": 10.0,
        "entry_atr": 1.0,
        "event_book": book,
        "trade_values": {"atr14": 1.0},
        "observation_values": {"rsi6": 20.0},
    }


def runtime_state(max_holdings=3, position_states=None, processed=None,
                  sold=None, attempted=None, retried=None):
    params = strategy.get_default_params()
    params["max_holdings"] = max_holdings
    return types.SimpleNamespace(
        params=params,
        etf_pool=list(EXPECTED_POOL),
        position_states={} if position_states is None else position_states,
        processed_resonance_ids={} if processed is None else processed,
        observation_events={},
        sold_today=set() if sold is None else sold,
        daily_attempted_buys=set() if attempted is None else attempted,
        daily_retried_exits=set() if retried is None else retried,
    )


def test_signal_loader_is_strictly_t_minus_one(monkeypatch):
    calls = []
    monkeypatch.setattr(
        strategy, "get_price",
        lambda code, **kw: calls.append((code, kw)) or make_ohlcv_frame(120),
        raising=False,
    )

    strategy.load_signal_price_frame("510300.XSHG", "2021-01-05", 120)

    assert calls == [("510300.XSHG", {
        "end_date": date(2021, 1, 5),
        "count": 120,
        "frequency": "daily",
        "fields": ["open", "high", "low", "close", "volume"],
        "skip_paused": True,
        "fq": "pre",
        "panel": False,
    })]


def test_signal_loader_propagates_future_data_error(monkeypatch):
    class FutureDataError(RuntimeError):
        """Local sentinel proving the loader preserves platform errors."""

    expected = FutureDataError("future boundary")

    def reject_future_access(*args, **kwargs):
        raise expected

    monkeypatch.setattr(strategy, "get_price", reject_future_access, raising=False)

    with pytest.raises(FutureDataError) as raised:
        strategy.load_signal_price_frame("510300.XSHG", "2021-01-05", 120)

    assert raised.value is expected


def test_build_signal_snapshot_rejects_insufficient_data_before_indicators(
        monkeypatch):
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: make_ohlcv_frame(119),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_indicator_frame",
        lambda *args: pytest.fail("insufficient data must not build indicators"),
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", "2021-01-05", strategy.get_default_params(),
        "2021-01-06",
    )

    assert snapshot == {
        "code": "510300.XSHG",
        "valid": False,
        "reason": "INSUFFICIENT_DATA",
    }


def test_build_signal_snapshot_keeps_observations_out_of_event_builder(
        monkeypatch):
    frame = make_ohlcv_frame(120)
    signal_timestamp = frame.index[-1]
    signal_date = signal_timestamp.date()
    decision_date = (signal_timestamp + pd.offsets.BDay(1)).date()
    captured = []
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: frame, raising=False,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_events",
        lambda indicators, date, next_date: captured.append(
            (indicators, date, next_date)
        ) or strategy.empty_event_book(),
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", signal_date, strategy.get_default_params(),
        decision_date,
    )

    assert snapshot["valid"] is True
    assert snapshot["signal_date"] == signal_date
    assert snapshot["close"] == pytest.approx(20.0)
    assert set(snapshot["trade_values"]) == set(strategy.TRADE_INDICATOR_COLUMNS)
    assert set(snapshot["observation_values"]) == set(strategy.OBSERVATION_COLUMNS)
    assert captured[0][1:] == (signal_date, decision_date)


@pytest.mark.parametrize("invalid_atr", [np.nan, np.inf, -np.inf, 0.0])
def test_invalid_current_atr_blocks_only_new_buy_not_existing_signal_exit(
        monkeypatch, invalid_atr):
    held, candidate = "510300.XSHG", "159915.XSHE"
    params = strategy.get_default_params()
    price_frame = make_ohlcv_frame(params["lookback_days"])
    indicators = strategy.build_indicator_frame(price_frame, params)
    indicators.loc[indicators.index[-1], "atr14"] = invalid_atr
    signal_date = "2021-01-05"
    event_books = iter([
        event_book_for_directions(
            "SELL_TURN", "SELL_TURN", "NEUTRAL", signal_date,
        ),
        event_book_for_directions(
            "BUY_TURN", "BUY_TURN", "NEUTRAL", signal_date,
        ),
    ])
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: price_frame,
    )
    monkeypatch.setattr(
        strategy, "build_indicator_frame", lambda *args: indicators.copy(),
    )
    monkeypatch.setattr(
        strategy, "collect_latest_events", lambda *args: next(event_books),
    )

    sell_snapshot = strategy.build_signal_snapshot(
        held, signal_date, params, "2021-01-06",
    )
    buy_snapshot = strategy.build_signal_snapshot(
        candidate, signal_date, params, "2021-01-06",
    )

    assert sell_snapshot["valid"] is True
    assert buy_snapshot["valid"] is True
    assert sell_snapshot["entry_atr"] is None
    assert buy_snapshot["entry_atr"] is None

    state = strategy.make_position_state(
        pd.Timestamp("2021-01-04").date(), 1.0, 10.0,
    )
    runtime = runtime_state(max_holdings=2, position_states={held: state})
    context = fake_context(positions={held: fake_position(100)})
    current_data = {
        held: current_record(10.0), candidate: current_record(5.0),
    }
    submitted_sells = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda context_arg, code, reason, trigger: submitted_sells.append(
            (code, reason, trigger)
        ) or strategy.OrderOutcome.FILLED,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("invalid current ATR must block a new buy"),
    )

    sell_attempts = strategy.run_signal_exits(
        context, current_data, {held: sell_snapshot},
    )
    buy_results = strategy.run_signal_buys(
        context, current_data, {candidate: buy_snapshot},
    )

    assert sell_attempts == {held}
    assert submitted_sells == [
        (held, strategy.ExitReason.SIGNAL_EXIT, pytest.approx(20.0)),
    ]
    assert buy_results == []
    assert candidate not in runtime.daily_attempted_buys


def test_build_signal_snapshots_uses_known_decision_date_without_calendar_lookup(
        monkeypatch):
    snapshot_calls = []

    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pytest.fail("09:35 must not query a future calendar"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshot",
        lambda code, prev, params, next_date: snapshot_calls.append(
            (code, prev, next_date)
        ) or {"code": code, "valid": False},
        raising=False,
    )

    snapshots = strategy.build_signal_snapshots(
        date(2021, 1, 8), strategy.get_default_params(), date(2021, 1, 11),
    )

    assert list(snapshots) == EXPECTED_POOL
    assert snapshot_calls == [
        (code, date(2021, 1, 8), date(2021, 1, 11))
        for code in EXPECTED_POOL
    ]


def test_full_0935_path_never_requests_a_future_trade_date(monkeypatch):
    class FutureDataError(RuntimeError):
        """Local sentinel for forbidden future calendar access."""

    current_date = date(2021, 1, 11)
    snapshot_calls = []

    def reject_future_calendar(**kwargs):
        end_date = pd.Timestamp(kwargs["end_date"]).date()
        if end_date > current_date:
            raise FutureDataError("future calendar request: %s" % end_date)
        pytest.fail("09:35 must not query the trade calendar at all")

    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days", reject_future_calendar, raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshot",
        lambda code, signal_date, params, decision_date: snapshot_calls.append(
            (code, signal_date, decision_date)
        ) or {"code": code, "valid": False, "reason": "INSUFFICIENT_DATA"},
        raising=False,
    )

    strategy.do_trading(fake_context(
        previous_date=date(2021, 1, 8), current_date=current_date,
    ))

    assert snapshot_calls == [
        (code, date(2021, 1, 8), current_date)
        for code in EXPECTED_POOL
    ]


def test_do_trading_stage_order_has_no_broad_early_return(monkeypatch):
    order = []
    monkeypatch.setattr(
        strategy, "reset_daily_state", lambda *args: order.append("reset"),
    )
    monkeypatch.setattr(
        strategy, "retry_pending_exits", lambda *args: order.append("pending"),
    )
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: order.append("atr"), raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshots",
        lambda *args: order.append("signals") or {}, raising=False,
    )
    monkeypatch.setattr(
        strategy, "run_signal_exits",
        lambda *args: order.append("signal_sells"), raising=False,
    )
    monkeypatch.setattr(
        strategy, "run_signal_buys", lambda *args: order.append("buys"),
        raising=False,
    )
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)

    strategy.do_trading(fake_context())

    assert order == [
        "reset", "pending", "atr", "signals", "signal_sells", "buys",
    ]


def test_atr_before_insufficient_signal_data_still_runs(monkeypatch):
    order = []
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "reset_daily_state", lambda *args: None)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: [])
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: order.append("atr") or set(),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshots",
        lambda *args: order.append("insufficient") or {
            "510300.XSHG": {"code": "510300.XSHG", "valid": False},
        },
        raising=False,
    )
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: set(), raising=False)
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: [], raising=False)

    strategy.do_trading(fake_context())

    assert order == ["atr", "insufficient"]


@pytest.mark.parametrize(
    "record,expected_tradability,expected_price",
    [
        (None, "UNKNOWN", None),
        (types.SimpleNamespace(paused=True, last_price=10.0), "PAUSED", 10.0),
        (types.SimpleNamespace(paused=False, last_price=10.0), "TRADEABLE", 10.0),
        (types.SimpleNamespace(paused=False, last_price=0.0), "TRADEABLE", None),
        (types.SimpleNamespace(paused=False, last_price=float("nan")), "TRADEABLE", None),
    ],
)
def test_platform_position_tradability_and_execution_price_boundaries(
        record, expected_tradability, expected_price):
    code = "510300.XSHG"
    current_data = {} if record is None else {code: record}
    context = fake_context(positions={
        code: fake_position(100), "159915.XSHE": fake_position(0),
    })

    assert strategy.get_actual_positions(context) == {code: context.portfolio.positions[code]}
    assert strategy.get_actual_amount(context, code) == 100
    assert strategy.get_actual_amount(context, "159915.XSHE") == 0
    assert strategy.get_tradability(current_data, code) is strategy.Tradability[
        expected_tradability
    ]
    if expected_price is None:
        assert strategy.get_execution_price(current_data, code) is None
    else:
        assert strategy.get_execution_price(current_data, code) == pytest.approx(
            expected_price
        )


def test_lazy_current_data_is_loaded_by_subscript_for_status_and_price():
    code = "510300.XSHG"
    record = current_record(10.5, paused=False)

    class LazyCurrentData:
        def __init__(self):
            self.loaded = {}
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            self.loaded[key] = record
            return self.loaded[key]

        def get(self, key, default=None):
            pytest.fail("JoinQuant current_data must be loaded through subscription")

    current_data = LazyCurrentData()

    assert strategy.get_tradability(
        current_data, code,
    ) is strategy.Tradability.TRADEABLE
    assert strategy.get_execution_price(current_data, code) == pytest.approx(10.5)
    assert current_data.accesses == [code, code]


@pytest.mark.parametrize("missing_error", [KeyError("missing"), IndexError(), TypeError()])
def test_lazy_current_data_narrow_missing_failures_map_to_unknown(missing_error):
    class MissingCurrentData:
        def __getitem__(self, key):
            raise missing_error

    current_data = MissingCurrentData()

    assert strategy.get_tradability(
        current_data, "510300.XSHG",
    ) is strategy.Tradability.UNKNOWN
    assert strategy.get_execution_price(current_data, "510300.XSHG") is None


def test_lazy_current_data_unrelated_error_propagates():
    class FutureDataError(TypeError):
        """Local sentinel for unrelated quote access failures."""

    expected = FutureDataError("quote boundary")

    class ExplodingCurrentData:
        def __getitem__(self, key):
            raise expected

    with pytest.raises(FutureDataError) as raised:
        strategy.get_tradability(ExplodingCurrentData(), "510300.XSHG")
    assert raised.value is expected

    with pytest.raises(FutureDataError) as raised:
        strategy.get_execution_price(ExplodingCurrentData(), "510300.XSHG")
    assert raised.value is expected


def test_nonfinite_risk_inputs_and_execution_prices_are_rejected():
    code = "510300.XSHG"
    for price in (np.inf, -np.inf, np.nan):
        assert strategy.get_execution_price(
            {code: current_record(price)}, code,
        ) is None

    params = strategy.get_default_params()
    for anchor, entry_atr in (
        (np.inf, 1.0), (-np.inf, 1.0),
        (100.0, np.inf), (100.0, -np.inf),
    ):
        assert strategy.calc_stop_state(anchor, entry_atr, params) is None

    state = strategy.make_position_state("2021-01-05", 1.0, 100.0)
    for closing_price in (np.inf, -np.inf, np.nan):
        strategy.update_highest_close_anchor(state, closing_price)
    assert state["highest_close_anchor"] == pytest.approx(100.0)


def test_after_close_cleans_only_actual_flats_and_updates_held_close_anchor(
        monkeypatch):
    held = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 1.0, 10.0)
    flat = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 2.0, 20.0)
    runtime = runtime_state(position_states={
        "510300.XSHG": held, "159915.XSHE": flat,
    })
    context = fake_context(positions={"510300.XSHG": fake_position(100)})
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_current_data",
        lambda: {"510300.XSHG": current_record(12.0)}, raising=False,
    )

    strategy.after_close(context)

    assert runtime.position_states == {"510300.XSHG": held}
    assert held["highest_close_anchor"] == pytest.approx(12.0)
    assert held["entry_atr"] == pytest.approx(1.0)


def test_submit_buy_uses_current_account_values_and_actual_partial_fill(
        monkeypatch):
    code = "510300.XSHG"
    runtime = runtime_state()
    context = fake_context(total_value=30000.0, available_cash=4000.0)
    current_data = {code: current_record(11.0)}
    target_values = []

    def partial_order(order_code, target_value):
        target_values.append((order_code, target_value))
        context.portfolio.positions[code] = fake_position(50)
        return types.SimpleNamespace(amount=100, filled=50)

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
    monkeypatch.setattr(strategy, "order_target_value", partial_order, raising=False)
    snapshot = resonance_snapshot(code)
    snapshot["entry_atr"] = 2.5
    decision = strategy.build_resonance_decision(
        code, strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )

    outcome = strategy.submit_buy(context, code, snapshot, decision)

    assert outcome is strategy.OrderOutcome.PARTIAL
    assert target_values[0][0] == code
    assert target_values[0][1] == pytest.approx(2500.0)
    assert runtime.position_states[code] == {
        "buy_date": pd.Timestamp("2021-01-06").date(),
        "entry_atr": 2.5,
        "highest_close_anchor": 11.0,
        "pending_exit": None,
    }


def test_invalid_quote_buy_consumes_slot_without_order_or_backfill(monkeypatch):
    first, second = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    current_data = {
        first: current_record(float("nan")), second: current_record(10.0),
    }
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: pytest.fail("invalid quote must not submit an order"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, current_data,
        {first: resonance_snapshot(first), second: resonance_snapshot(second)},
    )

    assert results == [(first, strategy.OrderOutcome.NOT_FILLED)]
    assert runtime.daily_attempted_buys == {first}
    assert len(runtime.processed_resonance_ids) == 1


def test_current_price_changes_entry_anchor_but_not_target_formula(monkeypatch):
    targets = []
    anchors = []
    code = "510300.XSHG"
    snapshot = resonance_snapshot(code)
    decision = strategy.build_resonance_decision(
        code, strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )

    for price in (10.0, 20.0):
        runtime = runtime_state()
        context = fake_context()
        current_data = {code: current_record(price)}

        def filled_order(order_code, target_value):
            targets.append(target_value)
            context.portfolio.positions[code] = fake_position(100)
            return types.SimpleNamespace(amount=100, filled=100)

        monkeypatch.setattr(strategy, "g", runtime, raising=False)
        monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
        monkeypatch.setattr(strategy, "order_target_value", filled_order, raising=False)

        strategy.submit_buy(context, code, snapshot, decision)
        anchors.append(runtime.position_states[code]["highest_close_anchor"])

    assert targets == [pytest.approx(20000.0 * 0.95 / 3)] * 2
    assert anchors == [10.0, 20.0]


def test_submit_sell_preserves_exit_reason_and_clears_only_actual_zero(
        monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 1.0, 10.0)
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    current_data = {code: current_record(9.0)}
    orders = []

    def filled_sell(order_code, target_amount):
        orders.append((order_code, target_amount))
        context.portfolio.positions.pop(code)
        return types.SimpleNamespace(amount=-100, filled=-100)

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
    monkeypatch.setattr(strategy, "order_target", filled_sell, raising=False)

    outcome = strategy.submit_sell(
        context, code, strategy.ExitReason.ATR_EXIT, 9.5,
    )

    assert outcome is strategy.OrderOutcome.FILLED
    assert orders == [(code, 0)]
    assert runtime.position_states == {}
    assert runtime.sold_today == {code}


def test_atr_exit_marks_sold_and_blocks_same_day_resonance_rebuy(monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 2.0, 100.0)
    runtime = runtime_state(max_holdings=1, position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    current_data = {code: current_record(90.0)}

    def filled_sell(order_code, target_amount):
        context.portfolio.positions.pop(code)
        return types.SimpleNamespace(amount=-100, filled=-100)

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
    monkeypatch.setattr(strategy, "order_target", filled_sell, raising=False)
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: pytest.fail("ATR-sold code must not be bought back today"),
        raising=False,
    )

    attempted = strategy.run_atr_exits(context, current_data)
    buy_results = strategy.run_signal_buys(
        context, current_data, {code: resonance_snapshot(code)},
    )

    assert attempted == {code}
    assert runtime.sold_today == {code}
    assert buy_results == []


def test_pending_retry_then_atr_upgrades_reason_without_duplicate_sell(
        monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 2.0, 100.0)
    strategy.set_pending_exit(
        state, strategy.ExitReason.SIGNAL_EXIT,
        pd.Timestamp("2021-01-06").date(), 99.0, 100,
    )
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    current_data = {code: current_record(90.0)}
    sell_calls = []

    def pending_retry(*args):
        sell_calls.append(args[1:])
        return strategy.OrderOutcome.NOT_FILLED

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "submit_sell", pending_retry, raising=False)

    retry_results = strategy.retry_pending_exits(context, current_data)
    atr_attempts = strategy.run_atr_exits(context, current_data)

    assert retry_results == [(code, strategy.OrderOutcome.NOT_FILLED)]
    assert atr_attempts == set()
    assert len(sell_calls) == 1
    assert state["pending_exit"] == {
        "created_date": pd.Timestamp("2021-01-06").date(),
        "reason": strategy.ExitReason.ATR_EXIT,
        "trigger_value": 95.0,
        "remaining_amount": 100,
    }


def test_pending_retry_code_cannot_receive_second_signal_sell(monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 1.0, 10.0)
    runtime = runtime_state(
        position_states={code: state}, retried={code},
    )
    context = fake_context(positions={code: fake_position(100)})
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda *args: pytest.fail("retried pending exit cannot sell twice"),
        raising=False,
    )

    attempted = strategy.run_signal_exits(
        context, {code: current_record(10.0)},
        {code: resonance_snapshot(code, direction="SELL_TURN")},
    )

    assert attempted == set()


def test_paused_signal_exit_freezes_pending_and_retries_first_next_session(
        monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(date(2021, 1, 4), 1.0, 10.0)
    runtime = runtime_state(position_states={code: state})
    positions = {code: fake_position(100)}
    paused_context = fake_context(
        previous_date=date(2021, 1, 5), current_date=date(2021, 1, 6),
        positions=positions,
    )
    snapshot = resonance_snapshot(
        code, direction="SELL_TURN", signal_date=date(2021, 1, 5),
    )
    real_submit_sell = strategy.submit_sell
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda *args: pytest.fail("paused signal exit must not submit an order"),
        raising=False,
    )

    attempted = strategy.run_signal_exits(
        paused_context, {code: current_record(10.0, paused=True)},
        {code: snapshot},
    )

    assert attempted == set()
    assert runtime.processed_resonance_ids == {}
    assert state["pending_exit"] == {
        "created_date": date(2021, 1, 6),
        "reason": strategy.ExitReason.SIGNAL_EXIT,
        "trigger_value": 10.0,
        "remaining_amount": 100,
    }

    tradeable_context = fake_context(
        previous_date=date(2021, 1, 6), current_date=date(2021, 1, 7),
        positions=positions,
    )
    orders = []

    def fill_pending_exit(order_code, target_amount):
        orders.append((order_code, target_amount))
        positions.pop(code)
        return types.SimpleNamespace(amount=-100, filled=-100)

    monkeypatch.setattr(
        strategy, "get_current_data",
        lambda: {code: current_record(10.0, paused=False)}, raising=False,
    )
    monkeypatch.setattr(strategy, "order_target", fill_pending_exit, raising=False)
    monkeypatch.setattr(strategy, "submit_sell", real_submit_sell, raising=False)

    retry_results = strategy.retry_pending_exits(
        tradeable_context, {code: current_record(10.0, paused=False)},
    )

    assert retry_results == [(code, strategy.OrderOutcome.FILLED)]
    assert orders == [(code, 0)]
    assert runtime.position_states == {}
    assert runtime.sold_today == {code}


def test_atr_pending_exit_overrides_signal_without_second_sell(monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 2.0, 100.0)
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    current_data = {code: current_record(90.0)}
    sell_calls = []

    def not_filled_atr(context_arg, order_code, reason, trigger_value):
        sell_calls.append((order_code, reason, trigger_value))
        if len(sell_calls) > 1:
            pytest.fail("ATR pending exit must override the ordinary signal sell")
        return strategy.sync_sell_state_after_order(
            order_code, strategy.OrderOutcome.NOT_FILLED, reason,
            context_arg.current_dt.date(), trigger_value, actual_amount=100,
        )

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "submit_sell", not_filled_atr, raising=False)

    atr_attempts = strategy.run_atr_exits(context, current_data)
    signal_attempts = strategy.run_signal_exits(
        context, current_data,
        {code: resonance_snapshot(code, direction="SELL_TURN")},
    )

    assert atr_attempts == {code}
    assert signal_attempts == set()
    assert sell_calls == [(code, strategy.ExitReason.ATR_EXIT, 95.0)]


def test_ordinary_sell_rereads_actual_positions_before_buy_slots(monkeypatch):
    held, candidate = "510300.XSHG", "159915.XSHE"
    state = strategy.make_position_state(pd.Timestamp("2021-01-05").date(), 1.0, 10.0)
    runtime = runtime_state(max_holdings=1, position_states={held: state})
    context = fake_context(positions={held: fake_position(100)})
    current_data = {
        held: current_record(10.0), candidate: current_record(5.0),
    }
    sold = []
    bought = []

    def sell_and_refresh(context_arg, code, reason, trigger_value):
        sold.append((code, reason, trigger_value))
        context_arg.portfolio.positions.pop(code)
        return strategy.OrderOutcome.FILLED

    def buy_after_refresh(context_arg, code, snapshot, decision):
        bought.append(code)
        return strategy.OrderOutcome.FILLED

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "submit_sell", sell_and_refresh, raising=False)
    monkeypatch.setattr(strategy, "submit_buy", buy_after_refresh, raising=False)

    snapshots = {
        held: resonance_snapshot(held, direction="SELL_TURN"),
        candidate: resonance_snapshot(candidate),
    }
    sell_attempts = strategy.run_signal_exits(context, current_data, snapshots)
    buy_results = strategy.run_signal_buys(context, current_data, snapshots)

    assert sell_attempts == {held}
    assert sold == [(held, strategy.ExitReason.SIGNAL_EXIT, 10.0)]
    assert buy_results == [(candidate, strategy.OrderOutcome.FILLED)]
    assert bought == [candidate]


def test_paused_buy_candidate_does_not_consume_slot_and_backfills(monkeypatch):
    paused, backfill = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    current_data = {
        paused: current_record(5.0, paused=True),
        backfill: current_record(10.0),
    }
    submitted = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda context_arg, code, snapshot, decision: submitted.append(code)
        or strategy.OrderOutcome.FILLED,
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, current_data,
        {paused: resonance_snapshot(paused), backfill: resonance_snapshot(backfill)},
    )

    assert results == [
        (paused, strategy.OrderOutcome.PAUSED),
        (backfill, strategy.OrderOutcome.FILLED),
    ]
    assert submitted == [backfill]
    assert runtime.daily_attempted_buys == {backfill}
    assert len(runtime.processed_resonance_ids) == 1


def test_refreshed_paused_buy_rolls_back_marks_and_backfills(monkeypatch):
    refreshed_paused, backfill = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    initial_current_data = {
        refreshed_paused: current_record(5.0),
        backfill: current_record(10.0),
    }
    refreshed_current_data = {
        refreshed_paused: current_record(5.0, paused=True),
        backfill: current_record(10.0),
    }
    submitted = []

    def fill_backfill(code, target_value):
        submitted.append(code)
        context.portfolio.positions[code] = fake_position(100)
        return types.SimpleNamespace(amount=100, filled=100)

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_current_data", lambda: refreshed_current_data,
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target_value", fill_backfill, raising=False,
    )
    snapshots = {
        refreshed_paused: resonance_snapshot(refreshed_paused),
        backfill: resonance_snapshot(backfill),
    }
    backfill_decision = strategy.build_resonance_decision(
        backfill, strategy.TurnDirection.BUY_TURN,
        snapshots[backfill]["event_book"], snapshots[backfill]["signal_date"],
    )

    results = strategy.run_signal_buys(
        context, initial_current_data, snapshots,
    )

    assert results == [
        (refreshed_paused, strategy.OrderOutcome.PAUSED),
        (backfill, strategy.OrderOutcome.FILLED),
    ]
    assert submitted == [backfill]
    assert runtime.daily_attempted_buys == {backfill}
    assert runtime.processed_resonance_ids == {
        backfill_decision["resonance_id"]: backfill_decision["expires_date"],
    }
    assert set(runtime.position_states) == {backfill}


def test_unknown_buy_candidate_consumes_slot_without_backfill(monkeypatch):
    unknown, lower = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    current_data = {lower: current_record(10.0)}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("unknown first candidate consumes the only slot"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, current_data,
        {unknown: resonance_snapshot(unknown), lower: resonance_snapshot(lower)},
    )

    assert results == [(unknown, strategy.OrderOutcome.UNKNOWN)]
    assert runtime.daily_attempted_buys == {unknown}
    assert len(runtime.processed_resonance_ids) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        strategy.OrderOutcome.NOT_FILLED,
        strategy.OrderOutcome.PARTIAL,
        strategy.OrderOutcome.FILLED,
    ],
)
def test_nonpaused_buy_attempt_outcomes_consume_one_slot_without_chasing(
        monkeypatch, outcome):
    first, lower = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    current_data = {first: current_record(5.0), lower: current_record(10.0)}
    submitted = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda context_arg, code, snapshot, decision: submitted.append(code) or outcome,
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, current_data,
        {first: resonance_snapshot(first), lower: resonance_snapshot(lower)},
    )

    assert results == [(first, outcome)]
    assert submitted == [first]
    assert runtime.daily_attempted_buys == {first}
    assert len(runtime.processed_resonance_ids) == 1


def test_full_portfolio_never_replaces_or_tops_up_held_resonance(monkeypatch):
    held_codes = ["510300.XSHG", "159915.XSHE", "512100.XSHG"]
    positions = {code: fake_position(100) for code in held_codes}
    runtime = runtime_state(max_holdings=3)
    context = fake_context(positions=positions)
    snapshots = {
        held_codes[0]: resonance_snapshot(held_codes[0], support_count=3),
        "513100.XSHG": resonance_snapshot("513100.XSHG", support_count=3),
    }
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("full portfolio must not replace or top up"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context,
        {code: current_record(10.0) for code in snapshots},
        snapshots,
    )

    assert results == []
    assert runtime.daily_attempted_buys == set()
    assert runtime.processed_resonance_ids == {}


def test_held_sold_attempted_and_processed_candidates_are_skipped(monkeypatch):
    held = "159915.XSHE"
    sold = "510300.XSHG"
    attempted = "510880.XSHG"
    processed = "512100.XSHG"
    eligible = "513100.XSHG"
    snapshots = {
        code: resonance_snapshot(code)
        for code in (held, sold, attempted, processed, eligible)
    }
    processed_decision = strategy.build_resonance_decision(
        processed, strategy.TurnDirection.BUY_TURN,
        snapshots[processed]["event_book"], snapshots[processed]["signal_date"],
    )
    runtime = runtime_state(
        max_holdings=3,
        processed={
            processed_decision["resonance_id"]: processed_decision["expires_date"],
        },
        sold={sold}, attempted={attempted},
    )
    context = fake_context(positions={held: fake_position(100)})
    submitted = []
    registrations = []
    logs = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda decision, event_date, event_close: registrations.append(
            decision["code"]
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda decision, accepted, reason: logs.append(
            (decision["code"], accepted, reason)
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda context_arg, code, snapshot, decision: submitted.append(code)
        or strategy.OrderOutcome.FILLED,
        raising=False,
    )

    results = strategy.run_signal_buys(
        context,
        {code: current_record(10.0) for code in snapshots},
        snapshots,
    )

    assert results == [(eligible, strategy.OrderOutcome.FILLED)]
    assert submitted == [eligible]
    assert sorted(registrations) == sorted(snapshots)
    assert (held, False, "HELD_NO_ADD") in logs
    assert (sold, False, "SOLD_TODAY") in logs
    assert (attempted, False, "ALREADY_ATTEMPTED_TODAY") in logs
    assert (processed, False, "RESONANCE_ALREADY_PROCESSED") in logs


def test_processed_resonance_id_prevents_duplicate_buy_submission(monkeypatch):
    code = "510300.XSHG"
    snapshot = resonance_snapshot(code)
    decision = strategy.build_resonance_decision(
        code, strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )
    runtime = runtime_state(processed={
        decision["resonance_id"]: decision["expires_date"],
    })
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("processed resonance must not resubmit"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        fake_context(), {code: current_record(10.0)}, {code: snapshot},
    )

    assert results == []


def test_current_quote_does_not_change_frozen_buy_candidate_order(monkeypatch):
    first, second = "159915.XSHE", "510300.XSHG"
    snapshots = {
        first: resonance_snapshot(first, support_count=3),
        second: resonance_snapshot(second, support_count=2),
    }
    observed_orders = []

    for prices in ((1.0, 100.0), (100.0, 1.0)):
        runtime = runtime_state(max_holdings=2)
        context = fake_context()
        submitted = []
        current_data = {
            first: current_record(prices[0]), second: current_record(prices[1]),
        }
        monkeypatch.setattr(strategy, "g", runtime, raising=False)
        monkeypatch.setattr(
            strategy, "submit_buy",
            lambda context_arg, code, snapshot, decision: submitted.append(code)
            or strategy.OrderOutcome.NOT_FILLED,
            raising=False,
        )

        strategy.run_signal_buys(context, current_data, snapshots)
        observed_orders.append(submitted)

    assert observed_orders == [[first, second], [first, second]]


def test_decision_payload_cannot_receive_observation_fields():
    assert list(inspect.signature(
        strategy.build_resonance_decision,
    ).parameters) == [
        "code", "direction", "event_book", "signal_date",
    ]


def test_strategy_ast_has_no_cross_signal_import_dependency():
    tree = ast.parse(STRATEGY_PATH.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert all(
        "cross_signal_strategy" not in module
        and "smart_trade_joinquant_cross_signal_etf" not in module
        for module in imported_modules
    )


def test_observation_outcomes_are_retrospective_and_one_shot():
    record = strategy.make_observation_event(
        resonance_id="abc",
        code="510300.XSHG",
        event_date="2021-01-05",
        event_close=10.0,
        horizons=(1, 3, 5),
    )

    assert "due_dates" not in record
    assert record["event_date"] == date(2021, 1, 5)
    assert strategy.due_observation_horizons(record, 0) == []
    assert strategy.due_observation_horizons(record, 1) == [1]
    record["outcomes"][1] = {"return": 0.01}
    assert strategy.due_observation_horizons(record, 1) == []
    assert record["event_close"] == pytest.approx(10.0)


def test_register_observation_event_never_prefetches_future_trading_sessions(
        monkeypatch):
    code = "510300.XSHG"
    snapshot = resonance_snapshot(code)
    decision = strategy.build_resonance_decision(
        code, strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )
    runtime = runtime_state()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pytest.fail("event registration must not read future dates"),
        raising=False,
    )

    strategy.register_observation_event(
        decision, pd.Timestamp("2021-01-05").date(), 10.0,
    )
    strategy.register_observation_event(
        decision, pd.Timestamp("2021-01-05").date(), 10.0,
    )

    record = runtime.observation_events[decision["resonance_id"]]
    assert record["event_date"] == date(2021, 1, 5)
    assert record["horizons"] == (1, 3, 5)
    assert "due_dates" not in record
    assert runtime.position_states == {}
    assert runtime.processed_resonance_ids == {}


def test_observation_registration_propagates_future_data_error(monkeypatch):
    class FutureDataError(RuntimeError):
        """Local sentinel for forbidden future observation access."""

    expected = FutureDataError("future observation access")
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda *args: (_ for _ in ()).throw(expected), raising=False,
    )

    with pytest.raises(FutureDataError) as raised:
        strategy.try_register_observation_event(
            {"resonance_id": "abc", "code": "510300.XSHG"},
            date(2021, 1, 5), 10.0,
        )

    assert raised.value is expected


def test_record_due_observation_outcomes_is_close_only_read_projection(
        monkeypatch):
    code = "510300.XSHG"
    closing_date = pd.Timestamp("2021-01-06").date()
    record = strategy.make_observation_event(
        resonance_id="abc",
        code=code,
        event_date=pd.Timestamp("2021-01-05").date(),
        event_close=10.0,
        horizons=(1, 3, 5),
    )
    runtime = runtime_state()
    runtime.observation_events = {"abc": record}
    context = fake_context(current_date="2021-01-06")
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "order_target",
        lambda *args: pytest.fail("close observation must never sell"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: pytest.fail("close observation must never buy"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "log", types.SimpleNamespace(info=lambda *args: None),
        raising=False,
    )
    calendar_calls = []

    def retrospective_calendar(**kwargs):
        calendar_calls.append(kwargs)
        assert kwargs == {
            "start_date": date(2021, 1, 5),
            "end_date": closing_date,
        }
        return pd.DatetimeIndex(["2021-01-05", "2021-01-06"])

    monkeypatch.setattr(
        strategy, "get_trade_days", retrospective_calendar, raising=False,
    )

    monkeypatch.setattr(
        strategy, "get_current_data", lambda: {code: current_record(11.0)},
        raising=False,
    )

    strategy.after_close(context)

    assert record["outcomes"] == {
        1: {
            "status": "RECORDED",
            "closing_date": closing_date,
            "closing_price": 11.0,
            "return": pytest.approx(0.1),
        },
    }
    assert "abc" in runtime.observation_events
    assert len(calendar_calls) == 1
    assert runtime.position_states == {}
    assert runtime.processed_resonance_ids == {}


def test_due_observation_missing_price_is_terminal_and_record_is_cleaned(
        monkeypatch):
    due_date = pd.Timestamp("2021-01-06").date()
    record = strategy.make_observation_event(
        "missing", "510300.XSHG", pd.Timestamp("2021-01-05").date(),
        10.0, horizons=(1,),
    )
    runtime = runtime_state()
    runtime.observation_events = {"missing": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pd.DatetimeIndex(["2021-01-05", "2021-01-06"]),
        raising=False,
    )

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-06"), {},
    )

    assert record["outcomes"][1] == {
        "status": "PRICE_UNAVAILABLE",
        "closing_date": due_date,
        "closing_price": None,
        "return": None,
    }
    assert runtime.observation_events == {}


def test_fully_terminal_observation_is_cleaned_when_no_horizon_is_newly_due(
        monkeypatch):
    record = strategy.make_observation_event(
        "terminal", "510300.XSHG", date(2021, 1, 5), 10.0,
        horizons=(1, 3, 5),
    )
    record["outcomes"] = {
        1: {"status": "RECORDED", "return": 0.01},
        3: {"status": "HORIZON_MISSED", "return": None},
        5: {"status": "PRICE_UNAVAILABLE", "return": None},
    }
    runtime = runtime_state()
    runtime.observation_events = {"terminal": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pd.DatetimeIndex([
            "2021-01-05", "2021-01-06", "2021-01-07",
            "2021-01-08", "2021-01-11", "2021-01-12",
        ]),
        raising=False,
    )

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-12"), {},
    )

    assert runtime.observation_events == {}


def test_after_close_prunes_terminal_observations_before_market_access(
        monkeypatch):
    record = strategy.make_observation_event(
        "terminal-first", "510300.XSHG", date(2021, 1, 5), 10.0,
        horizons=(1, 3, 5),
    )
    record["outcomes"] = {
        1: {"status": "RECORDED", "return": 0.01},
        3: {"status": "HORIZON_MISSED", "return": None},
        5: {"status": "PRICE_UNAVAILABLE", "return": None},
    }
    runtime = runtime_state()
    runtime.observation_events = {"terminal-first": record}
    expected = RuntimeError("market access after terminal prune")
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    def market_access_after_prune():
        assert runtime.observation_events == {}
        raise expected

    monkeypatch.setattr(
        strategy, "get_current_data", market_access_after_prune, raising=False,
    )
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pytest.fail(
            "terminal observation must not reach calendar access"
        ),
        raising=False,
    )

    with pytest.raises(RuntimeError) as raised:
        strategy.after_close(fake_context(current_date="2021-01-12"))

    assert raised.value is expected
    assert runtime.observation_events == {}


def test_legacy_due_dates_migrate_without_reading_values_and_stay_cleanable(
        monkeypatch):
    code = "510300.XSHG"
    record = {
        "resonance_id": "legacy",
        "code": code,
        "event_date": date(2021, 1, 5),
        "event_close": 10.0,
        "due_dates": {
            5: date(2099, 1, 12),
            1: date(2099, 1, 6),
            3: date(2099, 1, 8),
        },
        "outcomes": {1: 0.1},
    }
    runtime = runtime_state()
    runtime.observation_events = {"legacy": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    calendar_calls = []

    def retrospective_calendar(**kwargs):
        calendar_calls.append(kwargs)
        end_date = kwargs["end_date"]
        return pd.bdate_range("2021-01-05", end_date)

    monkeypatch.setattr(
        strategy, "get_trade_days", retrospective_calendar, raising=False,
    )

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-08"),
        {code: current_record(12.0)},
    )

    assert record["horizons"] == (1, 3, 5)
    assert "due_dates" not in record
    assert record["outcomes"][1]["status"] == "RECORDED"
    assert record["outcomes"][1]["return"] == pytest.approx(0.1)
    assert record["outcomes"][3]["return"] == pytest.approx(0.2)
    assert "legacy" in runtime.observation_events

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-12"),
        {code: current_record(13.0)},
    )

    assert record["outcomes"][1]["return"] == pytest.approx(0.1)
    assert record["outcomes"][5]["return"] == pytest.approx(0.3)
    assert runtime.observation_events == {}
    assert calendar_calls == [
        {"start_date": date(2021, 1, 5), "end_date": date(2021, 1, 8)},
        {"start_date": date(2021, 1, 5), "end_date": date(2021, 1, 12)},
    ]


def test_legacy_string_horizon_keys_migrate_without_reading_due_date_values():
    class KeysOnlyDueDates(dict):
        def __getitem__(self, key):
            pytest.fail("migration must not read a legacy due-date value")

        def items(self):
            pytest.fail("migration must not iterate legacy due-date values")

        def values(self):
            pytest.fail("migration must not iterate legacy due-date values")

    due_dates = KeysOnlyDueDates({
        "5": object(), "1": object(), "3": object(),
    })
    record = {
        "due_dates": due_dates,
        "outcomes": {
            "1": 0.1,
            "3": {"return": 0.2},
            "5": {"status": "PRICE_UNAVAILABLE", "return": None},
        },
    }

    strategy._normalize_observation_record(record)

    assert record["horizons"] == (1, 3, 5)
    assert "due_dates" not in record
    assert set(record["outcomes"]) == {1, 3, 5}
    assert record["outcomes"][1]["status"] == "RECORDED"
    assert record["outcomes"][1]["return"] == pytest.approx(0.1)
    assert record["outcomes"][3] == {
        "status": "RECORDED", "return": 0.2,
    }
    assert strategy._observation_record_is_terminal(record)


def test_mixed_horizon_keys_synchronize_outcomes_without_completing_pending():
    record = {
        "due_dates": {
            1: date(2099, 1, 6),
            "3": date(2099, 1, 8),
            5: date(2099, 1, 12),
        },
        "outcomes": {
            "1": {"status": "PENDING"},
            3: {"status": "RECORDED", "return": 0.2},
        },
    }

    strategy._normalize_observation_record(record)

    assert record["horizons"] == (1, 3, 5)
    assert set(record["outcomes"]) == {1, 3}
    assert record["outcomes"][1] == {"status": "PENDING"}
    assert strategy.due_observation_horizons(record, 3) == [1]
    assert not strategy._observation_record_is_terminal(record)


@pytest.mark.parametrize("record", [
    {
        "due_dates": {
            1: date(2099, 1, 6),
            "1": date(2099, 1, 7),
            3: date(2099, 1, 8),
        },
        "outcomes": {},
    },
    {
        "horizons": (1, "1", 3),
        "outcomes": {},
    },
    {
        "horizons": (1, 3, 5),
        "outcomes": {1: 0.1, "1": 0.2},
    },
    {
        "due_dates": {"2": date(2099, 1, 7)},
        "outcomes": {},
    },
    {
        "horizons": (1, 3, 5),
        "outcomes": {"2": 0.2},
    },
])
def test_invalid_or_duplicate_horizon_keys_leave_record_unchanged(record):
    before = copy.deepcopy(record)

    with pytest.raises(ValueError):
        strategy._normalize_observation_record(record)

    assert record == before


def test_duplicate_horizon_migration_failure_is_retryable_without_data_loss():
    due_dates = {
        1: date(2099, 1, 6),
        "1": date(2099, 1, 7),
        "3": date(2099, 1, 8),
    }
    record = {"due_dates": due_dates, "outcomes": {"1": 0.1}}
    before = copy.deepcopy(record)

    with pytest.raises(ValueError):
        strategy._normalize_observation_record(record)

    assert record == before
    due_dates.pop("1")
    strategy._normalize_observation_record(record)

    assert "due_dates" not in record
    assert record["horizons"] == (1, 3)
    assert record["outcomes"][1]["status"] == "RECORDED"
    assert record["outcomes"][1]["return"] == pytest.approx(0.1)


@pytest.mark.parametrize("existing_outcome", [
    {"status": "PENDING"},
    {"status": "UNKNOWN"},
    {},
    True,
    "UNRECOGNIZED",
    {"status": "PENDING", "return": 0.5},
])
def test_elapsed_nonterminal_outcome_is_replaced_by_retrospective_result(
        monkeypatch, existing_outcome):
    code = "510300.XSHG"
    record = strategy.make_observation_event(
        "unfinished", code, date(2021, 1, 5), 10.0, horizons=(1,),
    )
    record["outcomes"] = {1: existing_outcome}
    runtime = runtime_state()
    runtime.observation_events = {"unfinished": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pd.DatetimeIndex(["2021-01-05", "2021-01-06"]),
        raising=False,
    )

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-06"),
        {code: current_record(11.0)},
    )

    assert record["outcomes"][1] == {
        "status": "RECORDED",
        "closing_date": date(2021, 1, 6),
        "closing_price": 11.0,
        "return": pytest.approx(0.1),
    }
    assert runtime.observation_events == {}


def test_overdue_observation_is_missed_without_using_later_price(monkeypatch):
    due_date = pd.Timestamp("2021-01-06").date()
    record = strategy.make_observation_event(
        "overdue", "510300.XSHG", pd.Timestamp("2021-01-05").date(),
        10.0, horizons=(1,),
    )
    runtime = runtime_state()
    runtime.observation_events = {"overdue": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: pd.DatetimeIndex([
            "2021-01-05", "2021-01-06", "2021-01-07",
        ]),
        raising=False,
    )

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-07"),
        {"510300.XSHG": current_record(12.0)},
    )

    assert record["outcomes"][1] == {
        "status": "HORIZON_MISSED",
        "closing_date": due_date,
        "closing_price": None,
        "return": None,
    }
    assert runtime.observation_events == {}


def test_observation_calendar_future_data_error_propagates(monkeypatch):
    class FutureDataError(RuntimeError):
        """Local sentinel for forbidden future calendar access."""

    expected = FutureDataError("future calendar blocked")
    record = strategy.make_observation_event(
        "future", "510300.XSHG", date(2021, 1, 5), 10.0,
        horizons=(1,),
    )
    runtime = runtime_state()
    runtime.observation_events = {"future": record}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: (_ for _ in ()).throw(expected), raising=False,
    )
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)

    with pytest.raises(FutureDataError) as raised:
        strategy.after_close(fake_context(current_date="2021-01-06"))

    assert raised.value is expected


def test_structured_logger_propagates_type_error_future_boundary(monkeypatch):
    class FutureDataError(TypeError):
        """Local sentinel proving logger isolation cannot hide future data."""

    expected = FutureDataError("future log payload")
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(
            info=lambda *args: (_ for _ in ()).throw(expected)
        ),
        raising=False,
    )

    with pytest.raises(FutureDataError) as raised:
        strategy._emit_structured_log("future_boundary", {"value": 1})

    assert raised.value is expected


def test_structured_logger_propagates_future_boundary_named_in_base_mro(
        monkeypatch):
    class FutureDataError(TypeError):
        """Local platform-equivalent boundary type."""

    class WrappedFutureDataError(FutureDataError):
        """A platform subclass whose concrete name is not FutureDataError."""

    expected = WrappedFutureDataError("wrapped future log payload")
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(
            info=lambda *args: (_ for _ in ()).throw(expected)
        ),
        raising=False,
    )

    with pytest.raises(WrappedFutureDataError) as raised:
        strategy._emit_structured_log("future_boundary", {"value": 1})

    assert raised.value is expected


def test_structured_logger_still_isolates_ordinary_runtime_error(monkeypatch):
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(
            info=lambda *args: (_ for _ in ()).throw(
                RuntimeError("logger unavailable")
            )
        ),
        raising=False,
    )

    assert strategy._emit_structured_log("ordinary_failure", {}) is None


def test_structured_logging_contract_contains_required_audit_fields(
        monkeypatch):
    messages = []
    runtime = runtime_state()
    runtime.state_date = pd.Timestamp("2021-01-06").date()

    def capture(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr(
        strategy, "log", types.SimpleNamespace(info=capture), raising=False,
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    snapshot = resonance_snapshot("510300.XSHG", support_count=3)
    snapshot.update({
        "decision_date": "2021-01-06",
        "trade_values": {
            "rsi14": 28.0, "k": 18.0, "d": 19.0, "j": 16.0,
            "kd_diff": -1.0, "boll_mid": 10.0, "boll_upper": 11.0,
            "boll_lower": 9.0, "atr14": 0.5,
        },
        "observation_values": {
            "rsi6": 30.0, "rsi12": 31.0, "rsi24": 32.0,
            "plus_di": 20.0, "minus_di": 25.0, "adx14": np.nan,
            "volume": 1000.0, "volume_ma5": 900.0,
            "volume_ma20": 800.0, "volume_ratio": 1.25,
            "boll_width": 0.2, "boll_mid_slope": 0.1,
        },
    })
    decision = strategy.build_resonance_decision(
        snapshot["code"], strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )

    strategy.log_signal_snapshot(snapshot)
    strategy.log_resonance_decision(decision, True, "BUY_CANDIDATE_SORTED:1")
    strategy.log_order_transition(
        snapshot["code"], strategy.OrderSide.BUY,
        strategy.OrderOutcome.PARTIAL, 0, 100, 200,
        {"reason": strategy.ExitReason.SIGNAL_EXIT},
    )

    payloads = [json.loads(message) for message in messages]
    signal_payload = next(item for item in payloads if item["event"] == "signal_snapshot")
    assert signal_payload["version"] == strategy.STRATEGY_VERSION
    assert signal_payload["build"] == strategy.DEPLOYMENT_BUILD_ID
    assert signal_payload["parameter_fingerprint"]
    assert signal_payload["pool_fingerprint"]
    assert signal_payload["decision_date"] == "2021-01-06"
    assert signal_payload["signal_date"] == "2021-01-05"
    assert set(signal_payload["trade_values"]) == set(strategy.TRADE_INDICATOR_COLUMNS)
    assert set(signal_payload["observation_values"]) == set(strategy.OBSERVATION_COLUMNS)
    assert signal_payload["observation_values"]["adx14"] is None
    assert "active_events" in signal_payload
    assert "invalidated_events" in signal_payload

    decision_payload = next(
        item for item in payloads if item["event"] == "resonance_decision"
    )
    assert decision_payload["accepted"] is True
    assert decision_payload["reason"] == "BUY_CANDIDATE_SORTED:1"
    assert decision_payload["decision_date"] == "2021-01-06"
    assert decision_payload["signal_date"] == "2021-01-05"
    assert decision_payload["supporters"] == ["BOLL", "RSI", "KDJ"]
    assert decision_payload["support_count"] == 3
    assert decision_payload["boll_age"] == 0

    order_payload = next(item for item in payloads if item["event"] == "order_transition")
    assert order_payload["side"] == "BUY"
    assert order_payload["outcome"] == "PARTIAL"
    assert order_payload["before_amount"] == 0
    assert order_payload["after_amount"] == 100
    assert order_payload["requested_target"] == 200
    assert order_payload["pending_exit"]["reason"] == "SIGNAL_EXIT"

    held_state = strategy.make_position_state(
        pd.Timestamp("2021-01-05").date(), 0.5, 10.0,
    )
    runtime = runtime_state(position_states={"510300.XSHG": held_state})
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    strategy.log_portfolio_summary(fake_context(
        positions={"510300.XSHG": fake_position(100)},
        total_value=21000.0,
        available_cash=5000.0,
    ))
    portfolio_payload = json.loads(messages[-1])
    assert portfolio_payload["event"] == "portfolio_summary"
    assert portfolio_payload["total_value"] == pytest.approx(21000.0)
    assert portfolio_payload["available_cash"] == pytest.approx(5000.0)
    assert portfolio_payload["positions"] == {"510300.XSHG": 100}
    assert portfolio_payload["highest_close_anchors"] == {
        "510300.XSHG": 10.0,
    }


def test_initialize_emits_version_and_separate_configuration_fingerprints(
        monkeypatch):
    messages = []
    monkeypatch.setattr(strategy, "set_option", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: value, raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "run_daily", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda message, *args: messages.append(
            message % args if args else message
        )),
        raising=False,
    )
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert payload["event"] == "strategy_initialized"
    assert payload["version"] == strategy.STRATEGY_VERSION
    assert payload["build"] == strategy.DEPLOYMENT_BUILD_ID
    assert payload["parameter_fingerprint"]
    assert payload["pool_fingerprint"]


def test_after_close_runs_observations_anchor_cleanup_and_summary_without_orders(
        monkeypatch):
    runtime = runtime_state()
    context = fake_context(current_date="2021-01-06")
    order = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "record_due_observation_outcomes",
        lambda *args: order.append("observations"), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_portfolio_summary",
        lambda *args: order.append("summary"), raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target",
        lambda *args: pytest.fail("15:30 must not submit sell orders"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: pytest.fail("15:30 must not submit buy orders"),
        raising=False,
    )

    strategy.after_close(context)

    assert order == ["observations", "summary"]


def test_buy_attempt_logs_transition_and_registers_retrospective_event(
        monkeypatch):
    code = "510300.XSHG"
    snapshot = resonance_snapshot(code)
    runtime = runtime_state(max_holdings=1)
    context = fake_context()
    registrations = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: strategy.OrderOutcome.NOT_FILLED, raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_resonance_decision", lambda *args: None, raising=False,
    )
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda decision, event_date, event_close: registrations.append(
            (decision["resonance_id"], event_date, event_close)
        ),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, {code: current_record(10.0)}, {code: snapshot},
    )

    assert results == [(code, strategy.OrderOutcome.NOT_FILLED)]
    decision = strategy.build_resonance_decision(
        code, strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"], snapshot["signal_date"],
    )
    assert registrations == [(
        decision["resonance_id"], snapshot["signal_date"], snapshot["close"],
    )]


def test_submit_orders_log_actual_transition_after_state_sync(monkeypatch):
    buy_code = "510300.XSHG"
    sell_code = "159915.XSHE"
    sell_state = strategy.make_position_state(
        pd.Timestamp("2021-01-04").date(), 1.0, 10.0,
    )
    runtime = runtime_state(position_states={sell_code: sell_state})
    context = fake_context(positions={sell_code: fake_position(100)})
    current_data = {
        buy_code: current_record(10.0), sell_code: current_record(10.0),
    }
    transitions = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: current_data, raising=False)
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: types.SimpleNamespace(amount=100, filled=0), raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target",
        lambda *args: types.SimpleNamespace(amount=-100, filled=0), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_order_transition",
        lambda *args: transitions.append(args), raising=False,
    )
    buy_snapshot = resonance_snapshot(buy_code)
    buy_decision = strategy.build_resonance_decision(
        buy_code, strategy.TurnDirection.BUY_TURN,
        buy_snapshot["event_book"], buy_snapshot["signal_date"],
    )

    strategy.submit_buy(context, buy_code, buy_snapshot, buy_decision)
    strategy.submit_sell(
        context, sell_code, strategy.ExitReason.SIGNAL_EXIT, 9.0,
    )

    assert transitions[0][:5] == (
        buy_code, strategy.OrderSide.BUY, strategy.OrderOutcome.NOT_FILLED,
        0, 0,
    )
    assert transitions[0][6] is None
    assert transitions[1][:5] == (
        sell_code, strategy.OrderSide.SELL, strategy.OrderOutcome.NOT_FILLED,
        100, 100,
    )
    assert transitions[1][5] == 0
    assert transitions[1][6]["reason"] is strategy.ExitReason.SIGNAL_EXIT


def test_buy_rejection_logs_third_indicator_conflict_and_stale_support(
        monkeypatch):
    conflict_code = "510300.XSHG"
    stale_code = "159915.XSHE"
    conflict = resonance_snapshot(conflict_code)
    conflict["event_book"] = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "SELL_TURN", "2021-01-05",
    )
    stale = resonance_snapshot(stale_code)
    stale["event_book"] = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-04",
    )
    reasons = []
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda decision, accepted, reason: reasons.append(
            (decision["code"], accepted, reason)
        ),
        raising=False,
    )

    decisions = strategy.collect_buy_decisions(
        {conflict_code: conflict, stale_code: stale}, {},
    )

    assert decisions == []
    assert reasons == [
        (conflict_code, False, "THIRD_INDICATOR_CONFLICT"),
        (stale_code, False, "NO_FRESH_SUPPORTER"),
    ]


def test_empty_no_event_pool_emits_no_resonance_rejection_logs(monkeypatch):
    snapshots = {
        code: {
            "code": code,
            "valid": True,
            "signal_date": date(2021, 1, 5),
            "close": 10.0,
            "entry_atr": 1.0,
            "event_book": strategy.empty_event_book(),
        }
        for code in EXPECTED_POOL
    }
    logs = []
    registrations = []
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda *args: logs.append(args), raising=False,
    )
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda *args: registrations.append(args), raising=False,
    )

    buy = strategy.collect_complete_resonance_decisions(
        snapshots, strategy.TurnDirection.BUY_TURN,
    )
    sell = strategy.collect_complete_resonance_decisions(
        snapshots, strategy.TurnDirection.SELL_TURN,
    )

    assert buy == {}
    assert sell == {}
    assert logs == []
    assert registrations == []


def test_atr_check_log_is_observation_only_and_contains_frozen_risk_state(
        monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(
        pd.Timestamp("2021-01-05").date(), 1.0, 10.0,
    )
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    payloads = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: payloads.append((event, payload)), raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda *args: pytest.fail("price above stop must not sell"),
        raising=False,
    )

    result = strategy.run_atr_exits(
        context, {code: current_record(9.9)},
    )

    assert result == set()
    assert payloads == [("atr_check", {
        "code": code,
        "entry_atr": 1.0,
        "highest_close_anchor": 10.0,
        "stop_price": pytest.approx(8.5),
        "stop_pct": pytest.approx(0.15),
        "current_price": 9.9,
        "triggered": False,
        "pending_exit": None,
    })]


def test_full_portfolio_still_records_complete_resonance_without_order(
        monkeypatch):
    code = "513100.XSHG"
    runtime = runtime_state(max_holdings=3)
    context = fake_context(positions={
        "510300.XSHG": fake_position(100),
        "159915.XSHE": fake_position(100),
        "512100.XSHG": fake_position(100),
    })
    registrations = []
    logs = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda decision, event_date, event_close: registrations.append(
            decision["resonance_id"]
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda decision, accepted, reason: logs.append((accepted, reason)),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("full portfolio must not order"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context, {code: current_record()}, {code: resonance_snapshot(code)},
    )

    assert results == []
    assert len(registrations) == 1
    assert (False, "PORTFOLIO_FULL") in logs
    assert runtime.daily_attempted_buys == set()
    assert runtime.processed_resonance_ids == {}


def test_held_and_paused_resonances_are_observed_without_changing_trade_state(
        monkeypatch):
    held, paused = "510300.XSHG", "159915.XSHE"
    runtime = runtime_state(max_holdings=3)
    context = fake_context(positions={held: fake_position(100)})
    snapshots = {
        held: resonance_snapshot(held), paused: resonance_snapshot(paused),
    }
    registrations = []
    logs = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda decision, event_date, event_close: registrations.append(
            decision["code"]
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda decision, accepted, reason: logs.append(
            (decision["code"], accepted, reason)
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda *args: pytest.fail("held/paused resonance must not order"),
        raising=False,
    )

    results = strategy.run_signal_buys(
        context,
        {held: current_record(), paused: current_record(paused=True)},
        snapshots,
    )

    assert results == [(paused, strategy.OrderOutcome.PAUSED)]
    assert sorted(registrations) == sorted([held, paused])
    assert (held, False, "HELD_NO_ADD") in logs
    assert (paused, False, "PAUSED_BACKFILL") in logs
    assert runtime.daily_attempted_buys == set()
    assert runtime.processed_resonance_ids == {}


def test_observation_registration_failure_cannot_interrupt_remaining_candidates(
        monkeypatch):
    first, second = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=2)
    context = fake_context()
    submitted = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda *args: (_ for _ in ()).throw(RuntimeError("calendar unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda context_arg, code, snapshot, decision: submitted.append(code)
        or strategy.OrderOutcome.NOT_FILLED,
        raising=False,
    )

    results = strategy.run_signal_buys(
        context,
        {first: current_record(), second: current_record()},
        {first: resonance_snapshot(first), second: resonance_snapshot(second)},
    )

    assert submitted == [first, second]
    assert results == [
        (first, strategy.OrderOutcome.NOT_FILLED),
        (second, strategy.OrderOutcome.NOT_FILLED),
    ]


def test_logger_failure_cannot_change_candidate_orders(monkeypatch):
    first, second = "159915.XSHE", "510300.XSHG"
    runtime = runtime_state(max_holdings=2)
    submitted = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda *args: (_ for _ in ()).throw(
            RuntimeError("logger unavailable")
        )),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "register_observation_event", lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_buy",
        lambda context_arg, code, snapshot, decision: submitted.append(code)
        or strategy.OrderOutcome.NOT_FILLED,
        raising=False,
    )

    strategy.run_signal_buys(
        fake_context(),
        {first: current_record(), second: current_record()},
        {first: resonance_snapshot(first), second: resonance_snapshot(second)},
    )

    assert submitted == [first, second]


@pytest.mark.parametrize(
    "previous_diff,current_diff,expected",
    [
        (-0.5, 0.2, "GOLDEN_CROSS"),
        (0.5, -0.2, "DEATH_CROSS"),
        (-0.5, -0.1, "NONE"),
    ],
)
def test_kdj_formal_cross_is_observation_only(
        previous_diff, current_diff, expected):
    assert strategy.detect_kdj_formal_cross(
        {"kd_diff": previous_diff}, {"kd_diff": current_diff},
    ) == expected


def test_build_signal_snapshot_projects_kdj_cross_without_event_dependency(
        monkeypatch):
    params = strategy.get_default_params()
    price_frame = make_ohlcv_frame(params["lookback_days"])
    indicators = strategy.build_indicator_frame(price_frame, params)
    indicators.loc[indicators.index[-2], "kd_diff"] = -0.5
    indicators.loc[indicators.index[-1], "kd_diff"] = 0.2
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: price_frame,
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_indicator_frame", lambda *args: indicators,
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_events",
        lambda *args: strategy.empty_event_book(), raising=False,
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", indicators.index[-1], params,
        indicators.index[-1] + pd.offsets.BDay(1),
    )

    assert snapshot["kdj_cross"] == "GOLDEN_CROSS"
    assert snapshot["event_book"] == strategy.empty_event_book()


def test_do_trading_logs_formal_kdj_cross_without_turn_event(monkeypatch):
    code = "510300.XSHG"
    runtime = runtime_state()
    snapshot = {
        "code": code,
        "valid": True,
        "signal_date": "2021-01-05",
        "close": 10.0,
        "entry_atr": 1.0,
        "event_book": strategy.empty_event_book(),
        "trade_values": {},
        "observation_values": {},
        "kdj_cross": "GOLDEN_CROSS",
    }
    logged = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: [])
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *args: set())
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *args: {code: snapshot})
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: set())
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: [])
    monkeypatch.setattr(
        strategy, "log_signal_snapshot",
        lambda value: logged.append(value), raising=False,
    )

    strategy.do_trading(fake_context())

    assert len(logged) == 1
    assert logged[0]["code"] == code
    assert logged[0]["kdj_cross"] == "GOLDEN_CROSS"


def test_kdj_cross_field_cannot_change_candidate_order_or_orders(monkeypatch):
    first, second = "159915.XSHE", "510300.XSHG"
    observed = []
    for first_cross, second_cross in (
            ("GOLDEN_CROSS", "DEATH_CROSS"),
            ("NONE", "GOLDEN_CROSS")):
        runtime = runtime_state(max_holdings=2)
        submitted = []
        snapshots = {
            first: dict(resonance_snapshot(first, support_count=3),
                        kdj_cross=first_cross),
            second: dict(resonance_snapshot(second, support_count=2),
                         kdj_cross=second_cross),
        }
        monkeypatch.setattr(strategy, "g", runtime, raising=False)
        monkeypatch.setattr(
            strategy, "register_observation_event", lambda *args: None,
            raising=False,
        )
        monkeypatch.setattr(
            strategy, "submit_buy",
            lambda context_arg, code, snapshot, decision: submitted.append(code)
            or strategy.OrderOutcome.NOT_FILLED,
            raising=False,
        )

        strategy.run_signal_buys(
            fake_context(),
            {first: current_record(), second: current_record()}, snapshots,
        )
        observed.append(submitted)

    assert observed == [[first, second], [first, second]]


def test_unheld_complete_sell_resonance_is_recorded_without_order(monkeypatch):
    code = "510300.XSHG"
    runtime = runtime_state()
    snapshot = resonance_snapshot(code, direction="SELL_TURN")
    registrations = []
    logs = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "register_observation_event",
        lambda decision, event_date, event_close: registrations.append(
            decision["resonance_id"]
        ), raising=False,
    )
    monkeypatch.setattr(
        strategy, "log_resonance_decision",
        lambda decision, accepted, reason: logs.append((accepted, reason)),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda *args: pytest.fail("unheld sell resonance is record-only"),
        raising=False,
    )

    result = strategy.run_signal_exits(
        fake_context(), {code: current_record()}, {code: snapshot},
    )

    assert result == set()
    assert len(registrations) == 1
    assert (False, "UNHELD_RECORD_ONLY") in logs


def _event_diagnostic_frame(previous_overrides=None, current_overrides=None):
    base = {
        "low": 9.5,
        "high": 10.5,
        "close": 10.0,
        "boll_lower": 9.0,
        "boll_upper": 11.0,
        "rsi14": 50.0,
        "k": 50.0,
        "d": 50.0,
        "j": 50.0,
        "kd_diff": 0.0,
    }
    previous = dict(base)
    current = dict(base)
    previous.update(previous_overrides or {})
    current.update(current_overrides or {})
    return pd.DataFrame(
        [previous, current],
        index=pd.to_datetime(["2021-01-08", "2021-01-11"]),
    )


def test_diagnostic_build_id_is_bumped():
    assert strategy.DEPLOYMENT_BUILD_ID == "20260827.4"


def test_relative_observation_build_and_formal_fingerprints_are_separated(
        monkeypatch):
    messages = []
    _install_initialize_platform_stubs(monkeypatch, messages, [])
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert strategy.DEPLOYMENT_BUILD_ID == "20260827.4"
    assert payload["build"] == "20260827.4"
    assert payload["parameter_fingerprint"] == "e1227fbd8b4a884e"
    assert payload["pool_fingerprint"] == "9123995edeb1ed84"
    assert payload["event_logic_fingerprint"] == "1c0b8a22f48c97c3"
    assert payload["relative_observation_fingerprint"] == (
        strategy.relative_observation_fingerprint()
    )


def test_signal_snapshot_builds_separate_relative_event_book(monkeypatch):
    frame = make_ohlcv_frame(120)
    relative_book = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", frame.index[-1].date(),
    )
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: frame, raising=False,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_relative_events",
        lambda *args: relative_book, raising=False,
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", frame.index[-1], strategy.get_default_params(),
        frame.index[-1] + pd.offsets.BDay(1),
    )

    assert snapshot["relative_event_book"] is relative_book
    assert snapshot["event_book"] is not relative_book


def test_do_trading_runs_relative_stage_without_skipping_formal_pipeline(
        monkeypatch):
    calls = []
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "retry_pending_exits", lambda *args: calls.append("retry"),
    )
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: calls.append("atr"),
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshots",
        lambda *args: calls.append("snapshots") or {},
    )
    monkeypatch.setattr(
        strategy, "run_relative_observation_stage",
        lambda snapshots: calls.append("relative"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_exits", lambda *args: calls.append("exits"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_buys", lambda *args: calls.append("buys"),
    )

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "snapshots", "relative", "exits", "buys"]


def test_relative_stage_isolates_ordinary_error_but_propagates_future_error(
        monkeypatch):
    logs = []
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: logs.append((event, payload)),
    )
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations",
        lambda snapshots: (_ for _ in ()).throw(RuntimeError("ordinary")),
    )
    assert strategy.run_relative_observation_stage({}) is None
    assert logs[-1][0] == "relative_observation_pipeline"

    class FutureDataError(RuntimeError):
        pass

    expected = FutureDataError("future")
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations",
        lambda snapshots: (_ for _ in ()).throw(expected),
    )
    with pytest.raises(FutureDataError) as raised:
        strategy.run_relative_observation_stage({})
    assert raised.value is expected


def test_relative_stage_returns_none_when_snapshot_input_is_missing(
        monkeypatch):
    logs = []
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: logs.append((event, payload)),
    )

    assert strategy.run_relative_observation_stage(None) is None
    assert logs[-1][0] == "relative_observation_pipeline"


def test_relative_registration_isolates_ordinary_error_and_rethrows_future(
        monkeypatch):
    observation = {
        "relative_observation_id": "RELATIVE:fixture",
        "code": "510300.XSHG",
    }
    monkeypatch.setattr(
        strategy, "register_relative_observation_event",
        lambda value: (_ for _ in ()).throw(RuntimeError("ordinary")),
    )
    assert strategy.try_register_relative_observation_event(observation) is False

    class FutureDataError(RuntimeError):
        pass

    expected = FutureDataError("future registration")
    monkeypatch.setattr(
        strategy, "register_relative_observation_event",
        lambda value: (_ for _ in ()).throw(expected),
    )
    with pytest.raises(FutureDataError) as raised:
        strategy.try_register_relative_observation_event(observation)
    assert raised.value is expected


def test_relative_stage_registers_once_for_repeated_observation_and_empty(
        monkeypatch):
    observation = {
        "relative_observation_id": "RELATIVE:fixture",
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": "SOFT_ALL_THREE",
        "code": "510300.XSHG",
        "direction": strategy.TurnDirection.BUY_TURN,
        "signal_date": date(2021, 1, 5),
        "supporters": ("BOLL", "KDJ", "RSI"),
        "supporter_event_dates": {
            "BOLL": date(2021, 1, 5),
            "KDJ": date(2021, 1, 5),
            "RSI": date(2021, 1, 5),
        },
        "hard_or_relative_source_by_indicator": {
            "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
        },
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }
    runtime = runtime_state()
    logged = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations",
        lambda snapshots: [observation],
    )
    monkeypatch.setattr(
        strategy, "log_relative_resonance_observation",
        lambda value: logged.append(value["relative_observation_id"]),
    )

    assert strategy.run_relative_observation_stage({}) is None
    strategy.ensure_runtime_state()
    assert strategy.run_relative_observation_stage({}) is None
    assert set(runtime.observation_events) == {"RELATIVE:fixture"}
    assert logged == ["RELATIVE:fixture"]

    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations", lambda snapshots: [],
    )
    assert strategy.run_relative_observation_stage({}) is None
    assert set(runtime.observation_events) == {"RELATIVE:fixture"}


def test_relative_outcome_adds_direction_adjusted_return_without_orders(
        monkeypatch):
    observation = {
        "relative_observation_id": "RELATIVE:fixture",
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": "SOFT_ALL_THREE",
        "code": "510300.XSHG",
        "direction": strategy.TurnDirection.SELL_TURN,
        "signal_date": date(2021, 1, 5),
        "supporters": ("BOLL", "KDJ", "RSI"),
        "supporter_event_dates": {
            "BOLL": date(2021, 1, 5),
            "KDJ": date(2021, 1, 5),
            "RSI": date(2021, 1, 5),
        },
        "hard_or_relative_source_by_indicator": {
            "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
        },
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }
    runtime = runtime_state()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: [date(2021, 1, 5), date(2021, 1, 6)],
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target", lambda *args: pytest.fail("no sell order"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target_value", lambda *args: pytest.fail("no buy order"),
        raising=False,
    )
    assert strategy.register_relative_observation_event(observation) is True

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-06"),
        {"510300.XSHG": current_record(price=9.0)},
    )

    outcome = runtime.observation_events["RELATIVE:fixture"]["outcomes"][1]
    assert outcome["return"] == pytest.approx(-0.1)
    assert outcome["direction_adjusted_return"] == pytest.approx(0.1)


def test_formal_observation_outcome_log_has_no_relative_contract_fields(
        monkeypatch):
    runtime = runtime_state()
    logs = []
    decision = {
        "resonance_id": "formal-fixture",
        "code": "510300.XSHG",
    }
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: [date(2021, 1, 5), date(2021, 1, 6)],
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: logs.append((event, payload)),
    )
    strategy.register_observation_event(decision, date(2021, 1, 5), 10.0)

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-06"),
        {"510300.XSHG": current_record(price=11.0)},
    )

    payload = [payload for event, payload in logs
               if event == "observation_outcome"][0]
    forbidden = {
        "relative_observation_id", "observation_kind", "branch",
        "direction", "supporters", "build", "relative_observation_fingerprint",
    }
    assert forbidden.isdisjoint(payload)


def test_trading_functions_have_no_relative_observation_dependency():
    forbidden = {
        "relative_event_book", "relative_observation_id",
        "relative_observation", "relative_resonance",
    }
    for function in (
        strategy.run_atr_exits,
        strategy.collect_complete_resonance_decisions,
        strategy.collect_buy_decisions,
        strategy.sort_buy_decisions,
        strategy.run_signal_exits,
        strategy.run_signal_buys,
        strategy.submit_buy,
        strategy.submit_sell,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        strings = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert forbidden.isdisjoint(names | strings), function.__name__


def test_relative_snapshot_runtime_error_keeps_formal_trading_pipeline(
        monkeypatch):
    code = "510300.XSHG"
    calls = []
    formal_snapshots = []
    formal_book = strategy.empty_event_book()
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_default_etf_pool", lambda: [code])
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: make_ohlcv_frame(120),
    )
    monkeypatch.setattr(
        strategy, "collect_latest_events", lambda *args: formal_book,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_relative_events",
        lambda *args: (_ for _ in ()).throw(RuntimeError("relative snapshot")),
    )
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "retry_pending_exits", lambda *args: calls.append("retry"),
    )
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: calls.append("atr"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_exits",
        lambda context, current_data, snapshots: calls.append("exits")
        or formal_snapshots.append(snapshots),
    )
    monkeypatch.setattr(
        strategy, "run_signal_buys",
        lambda context, current_data, snapshots: calls.append("buys")
        or snapshots,
    )

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "exits", "buys"]
    assert formal_snapshots[0][code]["event_book"] is formal_book
    assert formal_snapshots[0][code]["relative_event_book"] == (
        strategy.empty_event_book()
    )


def test_relative_snapshot_future_data_error_is_rethrown_unchanged(
        monkeypatch):
    class FutureDataError(RuntimeError):
        pass

    expected = FutureDataError("future relative snapshot")
    frame = make_ohlcv_frame(120)
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: frame,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_relative_events",
        lambda *args: (_ for _ in ()).throw(expected),
    )

    with pytest.raises(FutureDataError) as raised:
        strategy.build_signal_snapshot(
            "510300.XSHG", frame.index[-1], strategy.get_default_params(),
            frame.index[-1] + pd.offsets.BDay(1),
        )

    assert raised.value is expected


def test_relative_observation_log_runtime_error_keeps_formal_trading_pipeline(
        monkeypatch):
    code = "510300.XSHG"
    calls = []
    observation = {
        "relative_observation_id": "RELATIVE:log-failure",
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": "SOFT_ALL_THREE",
        "code": code,
        "direction": strategy.TurnDirection.BUY_TURN,
        "signal_date": date(2021, 1, 5),
        "supporters": ("BOLL", "KDJ", "RSI"),
        "supporter_event_dates": {},
        "hard_or_relative_source_by_indicator": {},
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }
    snapshot = dict(resonance_snapshot(code), relative_event_book={})
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: calls.append("retry"))
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *args: calls.append("atr"))
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *args: {code: snapshot})
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations", lambda snapshots: [observation],
    )
    monkeypatch.setattr(
        strategy, "log_relative_resonance_observation",
        lambda value: (_ for _ in ()).throw(RuntimeError("relative log")),
    )
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: calls.append("exits"))
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: calls.append("buys"))

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "exits", "buys"]


def test_malformed_relative_candidate_keeps_formal_trading_pipeline(
        monkeypatch):
    code = "510300.XSHG"
    calls = []
    logs = []
    snapshot = dict(resonance_snapshot(code), relative_event_book={})
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: calls.append("retry"))
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *args: calls.append("atr"))
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *args: {code: snapshot})
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations", lambda snapshots: [object()],
    )
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: logs.append((event, payload)),
    )
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: calls.append("exits"))
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: calls.append("buys"))

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "exits", "buys"]
    assert logs[-1] == ("relative_observation_registration", {
        "relative_observation_id": None,
        "code": None,
        "reason": "RELATIVE_OBSERVATION_REGISTRATION_FAILED",
        "error_type": "TypeError",
    })


def test_relative_signal_snapshot_log_failure_keeps_formal_pipeline(
        monkeypatch):
    code = "510300.XSHG"
    calls = []
    snapshot = resonance_snapshot(code)
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "retry_pending_exits", lambda *args: calls.append("retry"),
    )
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: calls.append("atr"),
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshots", lambda *args: {code: snapshot},
    )
    monkeypatch.setattr(
        strategy, "relative_observation_fingerprint",
        lambda: (_ for _ in ()).throw(RuntimeError("relative fingerprint")),
    )
    monkeypatch.setattr(
        strategy, "run_relative_observation_stage", lambda *args: None,
    )
    monkeypatch.setattr(
        strategy, "run_signal_exits", lambda *args: calls.append("exits"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_buys", lambda *args: calls.append("buys"),
    )

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "exits", "buys"]


def test_relative_sidecar_states_leave_real_formal_execution_equivalent(
        monkeypatch):
    code = "510300.XSHG"
    observation = {
        "relative_observation_id": "RELATIVE:metamorphic",
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": "SOFT_ALL_THREE",
        "code": code,
        "direction": strategy.TurnDirection.BUY_TURN,
        "signal_date": date(2021, 1, 5),
        "supporters": ("BOLL", "KDJ", "RSI"),
        "supporter_event_dates": {},
        "hard_or_relative_source_by_indicator": {},
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }

    def run_with_relative_collector(collector):
        runtime = runtime_state(max_holdings=1)
        context = fake_context()
        orders = []
        with monkeypatch.context() as local:
            local.setattr(strategy, "g", runtime, raising=False)
            local.setattr(
                strategy, "get_current_data",
                lambda: {code: current_record(10.0)}, raising=False,
            )
            local.setattr(
                strategy, "build_signal_snapshots",
                lambda *args: {code: resonance_snapshot(code)},
            )
            local.setattr(
                strategy, "collect_relative_resonance_observations", collector,
            )
            local.setattr(
                strategy, "order_target_value",
                lambda order_code, target_value: orders.append(
                    (order_code, target_value)
                ) or context.portfolio.positions.__setitem__(
                    order_code, fake_position(100)
                ) or types.SimpleNamespace(amount=100),
                raising=False,
            )

            strategy.do_trading(context)

        return {
            "orders": orders,
            "processed": copy.deepcopy(runtime.processed_resonance_ids),
            "sold_today": set(runtime.sold_today),
            "attempted": set(runtime.daily_attempted_buys),
            "positions": {
                item: position.total_amount
                for item, position in context.portfolio.positions.items()
            },
            "position_states": copy.deepcopy(runtime.position_states),
        }

    outcomes = [
        run_with_relative_collector(lambda snapshots: [observation]),
        run_with_relative_collector(lambda snapshots: []),
        run_with_relative_collector(
            lambda snapshots: (_ for _ in ()).throw(RuntimeError("sidecar")),
        ),
    ]

    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0]["orders"]


def test_logged_kdj_values_flow_through_snapshot_trace_and_event_book(
        monkeypatch):
    params = strategy.get_default_params()
    price_frame = make_ohlcv_frame(params["lookback_days"])
    indicators = strategy.build_indicator_frame(price_frame, params).iloc[-2:].copy()
    previous_index, current_index = indicators.index
    indicators.loc[previous_index, [
        "k", "d", "j", "kd_diff", "rsi14", "low", "high", "close",
        "boll_lower", "boll_upper",
    ]] = [92.72, 91.23, 95.70, 1.49, 50.0, 9.5, 10.5, 10.0, 9.0, 11.0]
    indicators.loc[current_index, [
        "k", "d", "j", "kd_diff", "rsi14", "low", "high", "close",
        "boll_lower", "boll_upper",
    ]] = [87.34, 89.93, 82.15, -2.59, 50.0, 9.5, 10.5, 10.0, 9.0, 11.0]
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: price_frame,
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "build_indicator_frame", lambda *args: indicators.copy(),
        raising=False,
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", current_index, params,
        current_index + pd.offsets.BDay(1),
    )

    trace = snapshot["event_detection_trace"]
    assert trace["previous_date"] == previous_index.date()
    assert trace["current_date"] == current_index.date()
    assert trace["kdj"]["previous"] == {
        "k": 92.72, "d": 91.23, "j": 95.70, "kd_diff": 1.49,
    }
    assert trace["kdj"]["current"] == {
        "k": 87.34, "d": 89.93, "j": 82.15, "kd_diff": -2.59,
    }
    assert trace["kdj"]["sell_extreme"] is True
    assert trace["kdj"]["j_falling"] is True
    assert trace["kdj"]["kd_diff_falling"] is True
    assert trace["kdj"]["direction"] is strategy.TurnDirection.SELL_TURN
    assert trace["kdj"]["formal_cross"] == "DEATH_CROSS"
    assert snapshot["event_book"]["active"]["KDJ"]["direction"] is (
        strategy.TurnDirection.SELL_TURN
    )


@pytest.mark.parametrize(
    "previous,current,direction,touch_key,inside_key,close_key",
    [
        (
            {"low": 8.8, "close": 9.0, "boll_lower": 9.0},
            {"low": 9.2, "close": 9.5, "boll_lower": 9.1},
            strategy.TurnDirection.BUY_TURN,
            "lower_touch", "returned_inside_lower", "close_rising",
        ),
        (
            {"high": 11.2, "close": 11.0, "boll_upper": 11.0},
            {"high": 10.8, "close": 10.5, "boll_upper": 10.9},
            strategy.TurnDirection.SELL_TURN,
            "upper_touch", "returned_inside_upper", "close_falling",
        ),
    ],
)
def test_boll_turn_trace_matches_event_collection(
        previous, current, direction, touch_key, inside_key, close_key):
    params = strategy.get_default_params()
    frame = _event_diagnostic_frame(previous, current)

    trace = strategy.build_event_detection_trace(frame, params)
    event_book = strategy.collect_latest_events(
        frame, frame.index[-1], frame.index[-1] + pd.offsets.BDay(1),
    )

    assert trace["boll"][touch_key] is True
    assert trace["boll"][inside_key] is True
    assert trace["boll"][close_key] is True
    assert trace["boll"]["direction"] is direction
    assert event_book["active"]["BOLL"]["direction"] is direction


def test_runtime_event_logic_self_check_covers_literal_turn_contract():
    result = strategy.run_event_logic_self_check(strategy.get_default_params())

    assert result["passed"] is True
    assert set(result["cases"]) == {
        "kdj_buy_before_cross",
        "kdj_sell_high",
        "boll_buy_return_inside",
        "boll_sell_return_inside",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["kdj_buy_before_cross"]["actual_direction"] == (
        "BUY_TURN"
    )
    assert result["cases"]["kdj_buy_before_cross"]["formal_cross"] == "NONE"
    assert result["cases"]["kdj_sell_high"]["actual_direction"] == (
        "SELL_TURN"
    )
    assert result["cases"]["boll_buy_return_inside"]["actual_direction"] == (
        "BUY_TURN"
    )
    assert result["cases"]["boll_sell_return_inside"]["actual_direction"] == (
        "SELL_TURN"
    )
    assert all("inputs" in case for case in result["cases"].values())
    assert all("expected_direction" in case for case in result["cases"].values())


def test_event_logic_fingerprint_is_deterministic_and_contract_sensitive():
    params = strategy.get_default_params()
    self_check = strategy.run_event_logic_self_check(params)
    fingerprint = strategy.event_logic_fingerprint(params, self_check)
    reordered_params = dict(reversed(list(params.items())))
    reordered_check = {
        "cases": dict(reversed(list(self_check["cases"].items()))),
        "passed": self_check["passed"],
    }

    assert strategy.event_logic_fingerprint(
        reordered_params, reordered_check,
    ) == fingerprint
    assert len(fingerprint) == 16

    changed_threshold = copy.deepcopy(params)
    changed_threshold["kdj_high"] += 1
    assert strategy.event_logic_fingerprint(
        changed_threshold, self_check,
    ) != fingerprint

    changed_result = copy.deepcopy(self_check)
    changed_result["cases"]["kdj_sell_high"]["passed"] = False
    changed_result["passed"] = False
    assert strategy.event_logic_fingerprint(
        params, changed_result,
    ) != fingerprint


def test_event_logic_fingerprint_ignores_unexpected_detector_object_identity(
        monkeypatch):
    instances = []

    def unexpected_direction(*args):
        value = object()
        instances.append(value)
        return value

    monkeypatch.setattr(
        strategy, "detect_boll_direction", unexpected_direction,
    )
    params = strategy.get_default_params()

    first_check = strategy.run_event_logic_self_check(params)
    second_check = strategy.run_event_logic_self_check(params)

    assert first_check == second_check
    assert first_check["cases"]["boll_buy_return_inside"][
        "actual_direction"
    ] == "INVALID:object"
    assert strategy.event_logic_fingerprint(params, first_check) == (
        strategy.event_logic_fingerprint(params, second_check)
    )


def test_initialize_logs_runtime_self_check_and_event_logic_fingerprint(
        monkeypatch):
    messages = []
    monkeypatch.setattr(strategy, "set_option", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: value, raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "run_daily", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda message, *args: messages.append(
            message % args if args else message
        )),
        raising=False,
    )
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert payload["event"] == "strategy_initialized"
    assert payload["event_logic_self_check"]["passed"] is True
    assert payload["event_logic_fingerprint"] == strategy.event_logic_fingerprint(
        strategy.get_default_params(),
        strategy.run_event_logic_self_check(strategy.get_default_params()),
    )
    buy_case = payload["event_logic_self_check"]["cases"][
        "kdj_buy_before_cross"
    ]
    assert buy_case["inputs"]["previous"]["kd_diff"] < 0
    assert buy_case["inputs"]["current"]["kd_diff"] < 0


def test_failed_event_logic_self_check_is_logged_without_blocking_initialize(
        monkeypatch):
    messages = []
    scheduled = []
    failed_check = {
        "cases": {
            "synthetic_failure": {
                "inputs": {"previous": {}, "current": {}},
                "actual_direction": "NEUTRAL",
                "expected_direction": "BUY_TURN",
                "passed": False,
            },
        },
        "passed": False,
    }
    monkeypatch.setattr(strategy, "set_option", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: value, raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        strategy, "run_daily",
        lambda function, **kwargs: scheduled.append(function.__name__),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "run_event_logic_self_check", lambda params: failed_check,
    )
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda message, *args: messages.append(
            message % args if args else message
        )),
        raising=False,
    )
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert scheduled == ["do_trading", "after_close"]
    assert payload["event_logic_self_check"] == failed_check
    assert payload["event_logic_self_check"]["passed"] is False
    assert payload["event_logic_fingerprint"]


def test_signal_snapshot_log_contains_trace_and_event_logic_fingerprint(
        monkeypatch):
    messages = []
    runtime = runtime_state()
    frame = _event_diagnostic_frame(
        {"low": 8.8, "close": 9.0, "boll_lower": 9.0},
        {"low": 9.2, "close": 9.5, "boll_lower": 9.1},
    )
    snapshot = resonance_snapshot("510300.XSHG")
    snapshot.update({
        "decision_date": "2021-01-12",
        "event_detection_trace": strategy.build_event_detection_trace(
            frame, runtime.params,
        ),
    })
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda message, *args: messages.append(
            message % args if args else message
        )),
        raising=False,
    )

    strategy.log_signal_snapshot(snapshot)

    payload = json.loads(messages[-1])
    assert payload["event_detection_trace"]["boll"]["direction"] == "BUY_TURN"
    assert payload["event_detection_trace"]["previous_date"] == "2021-01-08"
    assert payload["event_detection_trace"]["current_date"] == "2021-01-11"
    assert payload["event_logic_fingerprint"] == strategy.event_logic_fingerprint(
        runtime.params,
        strategy.run_event_logic_self_check(runtime.params),
    )


def test_do_trading_logs_boll_touch_trace_without_event_or_cross(monkeypatch):
    code = "510300.XSHG"
    runtime = runtime_state()
    snapshot = {
        "code": code,
        "valid": True,
        "signal_date": "2021-01-05",
        "close": 10.0,
        "entry_atr": 1.0,
        "event_book": strategy.empty_event_book(),
        "trade_values": {},
        "observation_values": {},
        "kdj_cross": "NONE",
        "event_detection_trace": {
            "kdj": {"direction": strategy.TurnDirection.NEUTRAL},
            "boll": {
                "direction": strategy.TurnDirection.NEUTRAL,
                "lower_touch": True,
                "upper_touch": False,
            },
        },
    }
    logged = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *args: [])
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *args: set())
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *args: {code: snapshot})
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *args: set())
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *args: [])
    monkeypatch.setattr(
        strategy, "log_signal_snapshot", lambda value: logged.append(value),
        raising=False,
    )

    strategy.do_trading(fake_context())

    assert [item["code"] for item in logged] == [code]


def test_event_detection_trace_cannot_change_resonance_or_submitted_orders(
        monkeypatch):
    code = "510300.XSHG"
    observations = []
    traces = [
        {
            "kdj": {"direction": strategy.TurnDirection.SELL_TURN},
            "boll": {
                "direction": strategy.TurnDirection.SELL_TURN,
                "lower_touch": False,
                "upper_touch": True,
            },
        },
        {
            "kdj": {"direction": strategy.TurnDirection.NEUTRAL},
            "boll": {
                "direction": strategy.TurnDirection.NEUTRAL,
                "lower_touch": True,
                "upper_touch": False,
            },
        },
    ]
    for trace in traces:
        runtime = runtime_state(max_holdings=1)
        submitted = []
        snapshot = dict(
            resonance_snapshot(code), event_detection_trace=trace,
        )
        decision = strategy.build_resonance_decision(
            code, strategy.TurnDirection.BUY_TURN,
            snapshot["event_book"], snapshot["signal_date"],
        )
        monkeypatch.setattr(strategy, "g", runtime, raising=False)
        monkeypatch.setattr(
            strategy, "register_observation_event", lambda *args: None,
            raising=False,
        )
        monkeypatch.setattr(
            strategy, "submit_buy",
            lambda context_arg, candidate, snapshot_arg, decision_arg:
            submitted.append(candidate) or strategy.OrderOutcome.NOT_FILLED,
            raising=False,
        )

        strategy.run_signal_buys(
            fake_context(), {code: current_record()}, {code: snapshot},
        )
        observations.append((decision, submitted))

    assert observations[0][0] == observations[1][0]
    assert observations[0][1] == observations[1][1] == [code]


def _poison_strategy_reducers(monkeypatch):
    def joinquant_like_reducer(values):
        return (value for value in values)

    monkeypatch.setattr(strategy, "all", joinquant_like_reducer, raising=False)
    monkeypatch.setattr(strategy, "any", joinquant_like_reducer, raising=False)


def _install_initialize_platform_stubs(monkeypatch, messages, scheduled):
    monkeypatch.setattr(strategy, "set_option", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda *args: None, raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: value, raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        strategy, "run_daily",
        lambda function, **kwargs: scheduled.append(function.__name__),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "log",
        types.SimpleNamespace(info=lambda message, *args: messages.append(
            message % args if args else message
        )),
        raising=False,
    )
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)


def test_poisoned_reducers_keep_initialize_self_check_boolean_and_serializable(
        monkeypatch):
    messages = []
    scheduled = []
    _poison_strategy_reducers(monkeypatch)
    _install_initialize_platform_stubs(monkeypatch, messages, scheduled)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert scheduled == ["do_trading", "after_close"]
    assert payload["event_logic_self_check"]["passed"] is True
    assert isinstance(payload["event_logic_self_check"]["passed"], bool)
    assert isinstance(payload["event_logic_fingerprint"], str)


def test_poisoned_any_preserves_logged_kdj_sell_trace_and_active_event(
        monkeypatch):
    _poison_strategy_reducers(monkeypatch)
    params = strategy.get_default_params()
    frame = _event_diagnostic_frame(
        {"k": 92.72, "d": 91.23, "j": 95.70, "kd_diff": 1.49},
        {"k": 87.34, "d": 89.93, "j": 82.15, "kd_diff": -2.59},
    )

    trace = strategy.build_event_detection_trace(frame, params)
    event_book = strategy.collect_latest_events(
        frame, frame.index[-1], frame.index[-1] + pd.offsets.BDay(1),
    )

    assert trace["kdj"]["direction"] is strategy.TurnDirection.SELL_TURN
    assert event_book["active"]["KDJ"]["direction"] is (
        strategy.TurnDirection.SELL_TURN
    )


def test_poisoned_any_preserves_boll_buy_trace_and_active_event(monkeypatch):
    _poison_strategy_reducers(monkeypatch)
    params = strategy.get_default_params()
    frame = _event_diagnostic_frame(
        {"low": 8.8, "close": 9.0, "boll_lower": 9.0},
        {"low": 9.2, "close": 9.5, "boll_lower": 9.1},
    )

    trace = strategy.build_event_detection_trace(frame, params)
    event_book = strategy.collect_latest_events(
        frame, frame.index[-1], frame.index[-1] + pd.offsets.BDay(1),
    )

    assert trace["boll"]["direction"] is strategy.TurnDirection.BUY_TURN
    assert event_book["active"]["BOLL"]["direction"] is (
        strategy.TurnDirection.BUY_TURN
    )


def test_poisoned_any_preserves_resonance_conflict_and_freshness(monkeypatch):
    _poison_strategy_reducers(monkeypatch)
    direction = strategy.TurnDirection.BUY_TURN
    accepted = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-05",
    )
    conflicted = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "SELL_TURN", "2021-01-05",
    )
    stale = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-04",
    )

    assert strategy.build_resonance_decision(
        "510300.XSHG", direction, accepted, "2021-01-05",
    ) is not None
    assert strategy.resonance_rejection_reason(
        direction, accepted, "2021-01-05",
    ) == "RESONANCE_REJECTED"
    assert strategy.build_resonance_decision(
        "510300.XSHG", direction, conflicted, "2021-01-05",
    ) is None
    assert strategy.resonance_rejection_reason(
        direction, conflicted, "2021-01-05",
    ) == "THIRD_INDICATOR_CONFLICT"
    assert strategy.build_resonance_decision(
        "510300.XSHG", direction, stale, "2021-01-05",
    ) is None
    assert strategy.resonance_rejection_reason(
        direction, stale, "2021-01-05",
    ) == "NO_FRESH_SUPPORTER"


def test_poisoned_all_preserves_observation_terminal_booleans(monkeypatch):
    _poison_strategy_reducers(monkeypatch)
    terminal = {
        "horizons": (1, 3, 5),
        "outcomes": {
            1: {"status": "RECORDED", "return": 0.1},
            3: {"status": "HORIZON_MISSED", "return": None},
            5: {"status": "PRICE_UNAVAILABLE", "return": None},
        },
    }
    pending = copy.deepcopy(terminal)
    pending["outcomes"][3] = {"status": "PENDING"}

    assert strategy._observation_record_is_terminal(terminal) is True
    assert strategy._observation_record_is_terminal(pending) is False


def test_poisoned_any_preserves_future_data_error_classification(monkeypatch):
    class FutureDataError(RuntimeError):
        pass

    _poison_strategy_reducers(monkeypatch)

    assert strategy._is_future_data_error(FutureDataError("future")) is True
    assert strategy._is_future_data_error(RuntimeError("ordinary")) is False


def test_failed_self_check_stays_diagnostic_under_poisoned_reducers(
        monkeypatch):
    messages = []
    scheduled = []
    _poison_strategy_reducers(monkeypatch)
    _install_initialize_platform_stubs(monkeypatch, messages, scheduled)
    monkeypatch.setattr(
        strategy, "detect_kdj_direction",
        lambda *args: strategy.TurnDirection.NEUTRAL,
    )

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert scheduled == ["do_trading", "after_close"]
    assert payload["event_logic_self_check"]["passed"] is False
    assert isinstance(payload["event_logic_self_check"]["passed"], bool)
    assert isinstance(payload["event_logic_fingerprint"], str)
