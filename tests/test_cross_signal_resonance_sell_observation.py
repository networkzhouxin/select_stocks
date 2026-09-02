# -*- coding: utf-8 -*-
"""Tests for observation-only resonance labels on filled Cross signal exits."""

from datetime import date, datetime
import importlib.util
import pathlib
import sys
import types

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMAL_PATH = ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py"
OBSERVATION_PATH = (
    ROOT / "cross_signal_strategy"
    / "smart_trade_joinquant_cross_signal_etf_resonance_observation.py"
)
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_module("cross_signal_formal_for_sell_observation_test", FORMAL_PATH)
observation = _load_module(
    "cross_signal_resonance_sell_observation_test", OBSERVATION_PATH
)


def _indicator_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
    )


def test_sell_direction_uses_complete_three_way_reversal():
    """Catches reusing BUY supporters when classifying a SELL resonance."""
    frame = _indicator_frame([
        {
            "high": 10.0, "low": 9.0, "close": 9.5,
            "rsi14": 50.0, "k": 50.0, "d": 50.0, "j": 50.0,
            "kd_diff": 0.0, "boll_lower": 8.0, "boll_upper": 11.0,
        },
        {
            "high": 11.6, "low": 10.5, "close": 11.5,
            "rsi14": 75.0, "k": 85.0, "d": 82.0, "j": 101.0,
            "kd_diff": 3.0, "boll_lower": 8.0, "boll_upper": 11.0,
        },
        {
            "high": 11.0, "low": 10.0, "close": 10.5,
            "rsi14": 65.0, "k": 70.0, "d": 75.0, "j": 60.0,
            "kd_diff": -5.0, "boll_lower": 8.0, "boll_upper": 11.0,
        },
    ])

    result = observation.build_resonance_sell_observation_from_indicators(
        frame, "2021-01-06"
    )

    assert result["status"] == "OK"
    assert result["direction"] == "SELL"
    assert result["result"] == "COMPLETE_3"
    assert result["is_complete"] is True
    assert result["supporters"] == ("BOLL", "RSI", "KDJ")
    assert result["fresh_supporters"] == ("BOLL", "RSI", "KDJ")
    assert result["events"]["BOLL"]["direction"] == "SELL"


def test_successful_sell_observation_log_is_self_identifying(monkeypatch):
    """Catches a sell observation log that cannot be joined to its filled exit."""
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
        "sell_score": 45,
        "resonance_sell_observation": {
            "status": "OK",
            "direction": "SELL",
            "result": "COMPLETE_2_RSI",
            "is_complete": True,
            "support_count": 2,
            "supporters": ("BOLL", "RSI"),
            "fresh_supporters": ("RSI",),
            "events": {
                "BOLL": {
                    "direction": "SELL", "event_date": "2021-01-05", "age": 1,
                },
                "RSI": {
                    "direction": "SELL", "event_date": "2021-01-06", "age": 0,
                },
            },
            "values": {
                "rsi14": 65.0, "k": 70.0, "d": 75.0, "j": 60.0,
                "boll_lower": 8.0, "boll_mid": 9.5, "boll_upper": 11.0,
            },
        },
    }

    assert observation.emit_resonance_sell_observation(
        score, date(2021, 1, 7), date(2021, 1, 6)
    ) is True

    assert len(messages) == 1
    message = messages[0]
    for token in (
        "[resonance-sell-observation]",
        "schema=1",
        "build=20260902.2",
        "rule=HARD_BOLL_RSI14_KDJ_W2",
        "code=513100.XSHG",
        "execution_date=2021-01-07",
        "signal_date=2021-01-06",
        "exit_reason=sell_score",
        "sell_score=45",
        "direction=SELL",
        "result=COMPLETE_2_RSI",
        "supporters=BOLL+RSI",
        "fresh=RSI",
        "BOLL=SELL@1:2021-01-05",
        "RSI=SELL@0:2021-01-06",
        "KDJ=NONE",
    ):
        assert token in message


def _full_sell_score(code):
    score = {
        "code": code,
        "buy_allowed": False,
        "buy_score": 0,
        "reversal_score": 0,
        "location_score": 0,
        "trend_score": 0,
        "volume_score": 0,
        "sell_score": 45,
        "close": 2.0,
        "atr": 0.05,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": True,
        "resonance_sell_observation": {
            "status": "OK", "direction": "SELL",
            "result": "COMPLETE_2_RSI", "is_complete": True,
            "support_count": 2, "supporters": ("BOLL", "RSI"),
            "fresh_supporters": ("RSI",), "events": {}, "values": {},
        },
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
    return score


def _run_single_filled_signal_sell(
        module, monkeypatch, logger, current_price=2.0, fill_order=True):
    code = "513100.XSHG"

    class Position(object):
        total_amount = 100
        avg_cost = 2.0

    class Portfolio(object):
        def __init__(self):
            self.positions = {code: Position()}
            self.total_value = 20000.0
            self.available_cash = 10000.0

    class Context(object):
        def __init__(self):
            self.current_dt = datetime(2021, 1, 7, 9, 35)
            self.portfolio = Portfolio()

    class CurrentItem(object):
        paused = False
        last_price = current_price

    context = Context()
    order_calls = []
    score = _full_sell_score(code)

    def order_target(order_code, target_amount):
        order_calls.append((order_code, target_amount))
        if fill_order:
            context.portfolio.positions.pop(order_code, None)

    monkeypatch.setattr(
        module,
        "g",
        types.SimpleNamespace(
            params=module.get_default_params(), etf_pool=[code],
            highest_since_buy={code: 2.1}, entry_atr={code: 0.01},
            buy_date={code: date(2020, 12, 31)}, last_scores={},
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
        lambda **_kwargs: [
            date(2020, 12, 31), date(2021, 1, 1), date(2021, 1, 4),
            date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7),
        ],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "calc_cross_signal_score",
        lambda _code, _date, return_reason=False: (dict(score), None),
    )
    monkeypatch.setattr(module, "order_target", order_target, raising=False)
    monkeypatch.setattr(module, "order_target_value", lambda *_args: None, raising=False)

    module.do_trading(context)
    return order_calls, context


def test_sell_observation_log_failure_does_not_change_filled_sell_path(monkeypatch):
    """Catches an observation exception blocking a completed signal exit."""
    class QuietLogger(object):
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class FailingSellObservationLogger(object):
        def __init__(self):
            self.warnings = []

        def info(self, message, *_args, **_kwargs):
            if str(message).startswith("[resonance-sell-observation]"):
                raise RuntimeError("log sink failed")

        def warning(self, message, *_args, **_kwargs):
            self.warnings.append(str(message))

    formal_orders, formal_context = _run_single_filled_signal_sell(
        formal, monkeypatch, QuietLogger()
    )
    failing_logger = FailingSellObservationLogger()
    observed_orders, observed_context = _run_single_filled_signal_sell(
        observation, monkeypatch, failing_logger
    )

    assert observed_orders == formal_orders == [("513100.XSHG", 0)]
    assert observed_context.portfolio.positions == formal_context.portfolio.positions == {}
    assert any(
        message.startswith("[resonance-sell-observation-error]")
        for message in failing_logger.warnings
    )

def test_atr_exit_does_not_emit_sell_resonance_observation(monkeypatch):
    """Keeps the new signal-exit observer outside the existing ATR path."""
    class CollectingLogger(object):
        def __init__(self):
            self.messages = []

        def info(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    logger = CollectingLogger()
    orders, context = _run_single_filled_signal_sell(
        observation, monkeypatch, logger, current_price=1.9
    )

    assert orders == [("513100.XSHG", 0)]
    assert context.portfolio.positions == {}
    assert not any(
        message.startswith("[resonance-sell-observation]")
        for message in logger.messages
    )

def test_unfilled_signal_sell_does_not_emit_sell_resonance_observation(monkeypatch):
    """Logs only after the position state proves the sell has filled."""
    class CollectingLogger(object):
        def __init__(self):
            self.messages = []

        def info(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    logger = CollectingLogger()
    orders, context = _run_single_filled_signal_sell(
        observation, monkeypatch, logger, fill_order=False
    )

    assert orders == [("513100.XSHG", 0)]
    assert "513100.XSHG" in context.portfolio.positions
    assert not any(
        message.startswith("[resonance-sell-observation]")
        for message in logger.messages
    )
