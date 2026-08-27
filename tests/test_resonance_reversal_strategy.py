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
