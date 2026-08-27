import importlib.util
import pathlib
import sys
import types

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

    strategy.do_trading(types.SimpleNamespace())

    assert runtime.etf_pool == EXPECTED_POOL
    assert runtime.position_states == {}


def test_after_close_initializes_runtime_state(monkeypatch):
    runtime = types.SimpleNamespace()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.after_close(types.SimpleNamespace())

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
