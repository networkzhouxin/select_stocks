# -*- coding: utf-8 -*-
"""Tests for the observation-only resonance labels on formal Cross entries."""

from datetime import date, datetime
import importlib.util
import pathlib
import sys
import types

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMAL_PATH = (
    ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py"
)
OBSERVATION_PATH = (
    ROOT
    / "cross_signal_strategy"
    / "smart_trade_joinquant_cross_signal_etf_resonance_observation.py"
)
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_module("cross_signal_formal_for_observation_test", FORMAL_PATH)
observation = (
    _load_module("cross_signal_resonance_observation_test", OBSERVATION_PATH)
    if OBSERVATION_PATH.exists()
    else None
)


def test_standalone_observation_strategy_exists():
    """Catches omission of the separately runnable JoinQuant observation file."""
    assert observation is not None


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_observation_copy_preserves_formal_business_configuration():
    """Catches accidental changes to formal parameters, pool, or business identity."""
    assert observation.STRATEGY_VERSION == formal.STRATEGY_VERSION
    assert observation.DEPLOYMENT_BUILD_ID == formal.DEPLOYMENT_BUILD_ID
    assert observation.get_default_params() == formal.get_default_params()
    assert observation.get_default_etf_pool() == formal.get_default_etf_pool()
    assert observation.business_config_fingerprint() == formal.business_config_fingerprint()


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_reference_indicator_formulas_use_resonance_boundaries():
    """Catches reuse of Cross's different RSI/BOLL edge semantics."""
    rising = pd.Series([float(value) for value in range(1, 31)])
    falling = pd.Series([float(value) for value in range(30, 0, -1)])
    flat = pd.Series([10.0] * 30)

    assert observation.observation_calc_rsi(rising, 14).iloc[-1] == 100.0
    assert observation.observation_calc_rsi(falling, 14).iloc[-1] == 0.0
    assert observation.observation_calc_rsi(flat, 14).iloc[-1] == 50.0

    closes = pd.Series([float(value) for value in range(1, 21)])
    mid, upper, lower = observation.observation_calc_bollinger(closes, 20, 2.0)
    assert mid.iloc[-1] == pytest.approx(10.5)
    assert upper.iloc[-1] == pytest.approx(22.032562594670797)
    assert lower.iloc[-1] == pytest.approx(-1.0325625946707966)

    constant = pd.Series([5.0] * 12)
    k, d, j = observation.observation_calc_kdj(
        constant, constant, constant, 9, 3, 3
    )
    assert k.iloc[-1] == pytest.approx(50.0)
    assert d.iloc[-1] == pytest.approx(50.0)
    assert j.iloc[-1] == pytest.approx(50.0)


def _indicator_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
    )


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_prior_boll_turn_and_fresh_oscillators_form_complete_three_support():
    """Catches loss of the two-session event window or fresh-support requirement."""
    frame = _indicator_frame([
        {
            "high": 10.0, "low": 8.0, "close": 8.0,
            "rsi14": 50.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.5, "low": 8.5, "close": 10.0,
            "rsi14": 28.0, "k": 15.0, "d": 18.0, "j": -1.0,
            "kd_diff": -3.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.2, "low": 9.5, "close": 9.8,
            "rsi14": 35.0, "k": 25.0, "d": 20.0, "j": 35.0,
            "kd_diff": 5.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
    ])

    result = observation.build_resonance_entry_observation_from_indicators(
        frame, "2021-01-06"
    )

    assert result["status"] == "OK"
    assert result["result"] == "COMPLETE_3"
    assert result["is_complete"] is True
    assert result["support_count"] == 3
    assert result["supporters"] == ("BOLL", "RSI", "KDJ")
    assert result["fresh_supporters"] == ("RSI", "KDJ")
    assert result["events"]["BOLL"] == {
        "direction": "BUY", "event_date": "2021-01-05", "age": 1,
    }


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_opposite_oscillator_is_reported_as_conflict():
    """Catches score-like compensation that ignores an active opposite oscillator."""
    frame = _indicator_frame([
        {
            "high": 10.0, "low": 8.0, "close": 8.0,
            "rsi14": 50.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.5, "low": 8.5, "close": 10.0,
            "rsi14": 75.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.2, "low": 9.5, "close": 9.8,
            "rsi14": 65.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
    ])

    result = observation.build_resonance_entry_observation_from_indicators(
        frame, "2021-01-06"
    )

    assert result["result"] == "THIRD_INDICATOR_CONFLICT"
    assert result["is_complete"] is False
    assert result["events"]["RSI"]["direction"] == "SELL"


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_stale_support_is_not_reported_as_complete():
    """Catches acceptance when every active supporter came from T-2."""
    frame = _indicator_frame([
        {
            "high": 10.0, "low": 8.0, "close": 8.0,
            "rsi14": 20.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.5, "low": 8.5, "close": 10.0,
            "rsi14": 25.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
        {
            "high": 10.2, "low": 9.5, "close": 9.8,
            "rsi14": 25.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 9.0, "boll_upper": 12.0,
        },
    ])

    result = observation.build_resonance_entry_observation_from_indicators(
        frame, "2021-01-06"
    )

    assert result["result"] == "NO_FRESH_SUPPORTER"
    assert result["is_complete"] is False
    assert result["support_count"] == 2
    assert result["fresh_supporters"] == ()


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_safe_observation_builder_contains_diagnostic_failure(monkeypatch):
    """Catches propagation of an observation failure into formal scoring."""
    def fail(_frame, _signal_date):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(observation, "build_resonance_entry_observation", fail)

    result = observation.safe_build_resonance_entry_observation(None, "2021-01-06")

    assert result == {
        "status": "ERROR",
        "error_type": "RuntimeError",
        "result": "UNAVAILABLE",
        "is_complete": False,
    }


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_successful_observation_log_is_self_identifying_and_parseable(monkeypatch):
    """Catches logs that omit the frozen rule identity or core grouping fields."""
    messages = []
    monkeypatch.setattr(
        observation,
        "log",
        types.SimpleNamespace(
            info=lambda message: messages.append(str(message)),
            warning=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    score = {
        "code": "513100.XSHG",
        "resonance_entry_observation": {
            "status": "OK",
            "result": "COMPLETE_2_RSI",
            "is_complete": True,
            "support_count": 2,
            "supporters": ("BOLL", "RSI"),
            "fresh_supporters": ("RSI",),
            "events": {
                "BOLL": {
                    "direction": "BUY", "event_date": "2021-01-05", "age": 1,
                },
                "RSI": {
                    "direction": "BUY", "event_date": "2021-01-06", "age": 0,
                },
            },
            "values": {
                "rsi14": 35.0, "k": 25.0, "d": 20.0, "j": 35.0,
                "boll_lower": 9.0, "boll_mid": 10.0, "boll_upper": 11.0,
            },
        },
    }

    assert observation.emit_resonance_entry_observation(
        score, date(2021, 1, 7), date(2021, 1, 6)
    ) is True

    assert len(messages) == 1
    message = messages[0]
    for token in (
        "[resonance-entry-observation]",
        "schema=1",
        "build=20260902.2",
        "rule=HARD_BOLL_RSI14_KDJ_W2",
        "code=513100.XSHG",
        "execution_date=2021-01-07",
        "signal_date=2021-01-06",
        "result=COMPLETE_2_RSI",
        "support_count=2",
        "supporters=BOLL+RSI",
        "fresh=RSI",
        "BOLL=BUY@1:2021-01-05",
        "RSI=BUY@0:2021-01-06",
        "KDJ=NONE",
    ):
        assert token in message


def _full_buy_score(code, resonance_observation=None):
    score = {
        "code": code,
        "buy_allowed": True,
        "buy_score": 70,
        "reversal_score": 35,
        "location_score": 15,
        "trend_score": 20,
        "volume_score": 0,
        "sell_score": 0,
        "close": 2.0,
        "atr": 0.05,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
    }
    for key in (
        "rsi6", "rsi12", "rsi24", "rsi6_prev", "rsi12_prev", "rsi24_prev",
        "dif", "dea", "macd_hist", "dif_prev", "dea_prev", "macd_hist_prev",
        "k", "d", "j", "k_prev", "d_prev", "j_prev", "boll_upper",
        "boll_mid", "boll_lower", "ma5", "ma10", "ma20", "ma60", "vol5",
        "vol20", "plus_di", "minus_di", "adx",
    ):
        score[key] = 1.0
    for key in (
        "rsi6_cross_rsi12_up", "rsi6_cross_rsi24_up", "macd_cross_up",
        "kdj_k_cross_up", "kdj_j_cross_up", "rsi6_cross_rsi12_down",
        "rsi6_cross_rsi24_down", "macd_cross_down", "kdj_k_cross_down",
        "kdj_j_cross_down",
    ):
        score[key] = False
    if resonance_observation is not None:
        score["resonance_entry_observation"] = resonance_observation
    return score


def _run_single_filled_buy(module, monkeypatch, logger):
    code = "513100.XSHG"

    class Position(object):
        total_amount = 100
        avg_cost = 2.0

    class Portfolio(object):
        def __init__(self):
            self.positions = {}
            self.total_value = 20000.0
            self.available_cash = 20000.0

    class Context(object):
        def __init__(self):
            self.current_dt = datetime(2021, 1, 7, 9, 35)
            self.portfolio = Portfolio()

    class CurrentItem(object):
        paused = False
        last_price = 2.0

    context = Context()
    order_calls = []
    score = _full_buy_score(
        code,
        {
            "status": "OK", "result": "COMPLETE_2_RSI", "is_complete": True,
            "support_count": 2, "supporters": ("BOLL", "RSI"),
            "fresh_supporters": ("RSI",), "events": {},
        },
    )

    def order_target_value(order_code, target_value):
        order_calls.append((order_code, target_value))
        context.portfolio.positions[order_code] = Position()

    monkeypatch.setattr(
        module,
        "g",
        types.SimpleNamespace(
            params=module.get_default_params(), etf_pool=[code],
            highest_since_buy={}, entry_atr={}, buy_date={}, last_scores={},
            sold_today=set(), sold_guard_date=None, atr_stop_history=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module, "get_prev_trade_date", lambda _context: date(2021, 1, 6))
    monkeypatch.setattr(module, "get_current_data", lambda: {code: CurrentItem()}, raising=False)
    monkeypatch.setattr(
        module,
        "get_trade_days",
        lambda **_kwargs: [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "calc_cross_signal_score",
        lambda _code, _date, return_reason=False: (dict(score), None),
    )
    monkeypatch.setattr(module, "order_target_value", order_target_value, raising=False)

    module.do_trading(context)
    return order_calls, context


@pytest.mark.skipif(observation is None, reason="observation strategy not created yet")
def test_observation_log_failure_does_not_change_filled_buy_path(monkeypatch):
    """Catches an observation logger exception blocking or changing an order."""
    class QuietLogger(object):
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class FailingObservationLogger(object):
        def __init__(self):
            self.warnings = []

        def info(self, message, *_args, **_kwargs):
            if str(message).startswith("[resonance-entry-observation]"):
                raise RuntimeError("log sink failed")

        def warning(self, message, *_args, **_kwargs):
            self.warnings.append(str(message))

    formal_orders, formal_context = _run_single_filled_buy(
        formal, monkeypatch, QuietLogger()
    )
    failing_logger = FailingObservationLogger()
    observed_orders, observed_context = _run_single_filled_buy(
        observation, monkeypatch, failing_logger
    )

    assert observed_orders == formal_orders
    assert sorted(observed_context.portfolio.positions) == sorted(
        formal_context.portfolio.positions
    )
    assert any(
        message.startswith("[resonance-entry-observation-error]")
        for message in failing_logger.warnings
    )

