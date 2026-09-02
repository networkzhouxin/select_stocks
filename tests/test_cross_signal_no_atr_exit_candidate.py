# -*- coding: utf-8 -*-
"""Tests for the isolated Cross candidate with observation-only ATR exits."""

from datetime import date, datetime
import importlib.util
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    ROOT / "cross_signal_strategy"
    / "smart_trade_joinquant_cross_signal_etf_no_atr_exit_candidate.py"
)
FORMAL_PATH = (
    ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py"
)
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def _load_module(name, path):
    assert path.is_file(), "candidate implementation is missing: %s" % path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_candidate():
    return _load_module("cross_signal_no_atr_exit_candidate_test", CANDIDATE_PATH)


def _load_formal():
    return _load_module("cross_signal_formal_for_no_atr_exit_test", FORMAL_PATH)


def _full_signal_sell_score(code):
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


def _run_one_position(
        module, monkeypatch, current_price, current_dt, score=None):
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
            self.current_dt = current_dt
            self.portfolio = Portfolio()

    class CurrentItem(object):
        paused = False
        last_price = current_price

    class CollectingLogger(object):
        def __init__(self):
            self.messages = []

        def info(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    context = Context()
    logger = CollectingLogger()
    order_calls = []

    def order_target(order_code, target_amount):
        order_calls.append((order_code, target_amount))
        context.portfolio.positions.pop(order_code, None)

    monkeypatch.setattr(
        module,
        "g",
        types.SimpleNamespace(
            params=module.get_default_params(),
            etf_pool=[code],
            highest_since_buy={code: 2.1},
            entry_atr={code: 0.01},
            buy_date={code: date(2020, 12, 31)},
            last_scores={},
            sold_today=set(),
            sold_guard_date=None,
            atr_stop_history=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(
        module, "get_prev_trade_date", lambda _context: date(2021, 1, 8)
    )
    monkeypatch.setattr(
        module, "get_current_data", lambda: {code: CurrentItem()}, raising=False
    )
    monkeypatch.setattr(
        module,
        "get_trade_days",
        lambda **_kwargs: [
            date(2020, 12, 31), date(2021, 1, 4), date(2021, 1, 5),
            date(2021, 1, 6), date(2021, 1, 7), date(2021, 1, 8),
        ],
        raising=False,
    )
    if score is not None:
        monkeypatch.setattr(
            module,
            "calc_cross_signal_score",
            lambda _code, _date, return_reason=False: (dict(score), None),
        )
    monkeypatch.setattr(module, "order_target", order_target, raising=False)
    monkeypatch.setattr(
        module, "order_target_value", lambda *_args: None, raising=False
    )

    module.do_trading(context)
    return code, context, logger.messages, order_calls


def test_candidate_keeps_formal_business_configuration():
    candidate = _load_candidate()
    formal = _load_formal()

    assert candidate.STRATEGY_VERSION == "cross-v0.3.3-no-atr-exit-candidate"
    assert candidate.DEPLOYMENT_BUILD_ID == "20260902.3-candidate"
    assert candidate.ATR_EXIT_POLICY == "OBSERVE_ONLY"
    assert candidate.get_default_params() == formal.get_default_params()
    assert candidate.get_default_etf_pool() == formal.get_default_etf_pool()


def test_atr_hit_at_exact_stop_is_observed_without_order_or_state_cleanup(
        monkeypatch):
    candidate = _load_candidate()
    params = candidate.get_default_params()
    exact_stop = candidate.calc_stop_price(2.1, 0.01, 2.0, params)

    code, context, messages, orders = _run_one_position(
        candidate,
        monkeypatch,
        current_price=exact_stop,
        current_dt=datetime(2021, 1, 9, 9, 35),
    )

    assert orders == []
    assert code in context.portfolio.positions
    assert candidate.g.atr_stop_history == []
    assert any(
        message.startswith("[atr-exit-observation]")
        and "policy=OBSERVE_ONLY" in message
        and "action=hold" in message
        and "sell_signal_only=True" in message
        for message in messages
    )


def test_original_signal_sell_still_closes_the_position(monkeypatch):
    candidate = _load_candidate()
    score = _full_signal_sell_score("513100.XSHG")

    code, context, _messages, orders = _run_one_position(
        candidate,
        monkeypatch,
        current_price=2.2,
        current_dt=datetime(2021, 1, 8, 9, 35),
        score=score,
    )

    assert orders == [(code, 0)]
    assert context.portfolio.positions == {}
    assert candidate.g.atr_stop_history == []

def test_execute_policy_retains_formal_atr_exit_path(monkeypatch):
    """Catches a policy label that does not actually control ATR execution."""
    candidate = _load_candidate()
    monkeypatch.setattr(candidate, "ATR_EXIT_POLICY", "EXECUTE")
    params = candidate.get_default_params()
    exact_stop = candidate.calc_stop_price(2.1, 0.01, 2.0, params)

    code, context, _messages, orders = _run_one_position(
        candidate,
        monkeypatch,
        current_price=exact_stop,
        current_dt=datetime(2021, 1, 9, 9, 35),
    )

    assert orders == [(code, 0)]
    assert context.portfolio.positions == {}
    assert candidate.g.atr_stop_history == [date(2021, 1, 9)]

def test_unknown_atr_exit_policy_fails_fast(monkeypatch):
    """Catches a misspelled policy silently disabling every ATR action."""
    candidate = _load_candidate()
    monkeypatch.setattr(candidate, "ATR_EXIT_POLICY", "UNKNOWN")
    params = candidate.get_default_params()
    exact_stop = candidate.calc_stop_price(2.1, 0.01, 2.0, params)

    with pytest.raises(ValueError, match="unsupported ATR_EXIT_POLICY: UNKNOWN"):
        _run_one_position(
            candidate,
            monkeypatch,
            current_price=exact_stop,
            current_dt=datetime(2021, 1, 9, 9, 35),
        )
