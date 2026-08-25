# -*- coding: utf-8 -*-
"""Behavior tests for the independent KRBA RSI-turn JoinQuant candidate."""

from __future__ import annotations

import __future__
import builtins
import pathlib
import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def _snapshot(**overrides):
    value = {
        "k_prev": 19.0,
        "d_prev": 20.0,
        "k": 22.0,
        "d": 21.0,
        "rsi6_2ago": 32.0,
        "rsi6_prev": 28.0,
        "rsi6": 29.0,
        "close_prev": 9.8,
        "low": 9.7,
        "close": 9.9,
        "boll_lower": 9.8,
        "boll_mid": 10.5,
        "boll_upper": 11.2,
        "atr": 0.2,
    }
    value.update(overrides)
    return value


def candidate_module():
    from cross_signal_strategy import (
        smart_trade_joinquant_kdj_rsi_boll_atr_scheme_a_candidate,
    )

    return smart_trade_joinquant_kdj_rsi_boll_atr_scheme_a_candidate


def test_joinquant_upload_avoids_python37_only_annotation_and_dataclass_features():
    path = (
        ROOT
        / "cross_signal_strategy"
        / "smart_trade_joinquant_kdj_rsi_boll_atr_scheme_a_candidate.py"
    )
    source = path.read_text(encoding="utf-8")
    compiled = compile(source, str(path), "exec", dont_inherit=True)
    assert compiled.co_flags & __future__.annotations.compiler_flag == 0

    real_import = builtins.__import__

    def import_without_dataclasses(name, *args, **kwargs):
        if name == "dataclasses":
            raise AssertionError("JoinQuant upload imported Python 3.7-only dataclasses")
        return real_import(name, *args, **kwargs)

    namespace = {"__name__": "joinquant_candidate_compatibility_probe"}
    original_import = builtins.__import__
    try:
        builtins.__import__ = import_without_dataclasses
        exec(compiled, namespace)
    finally:
        builtins.__import__ = original_import

    state = namespace["PositionSignalState"]("2020-01-02", 10.0, 0.2, 10.0)
    assert state.entry_atr == pytest.approx(0.2)


def test_original_kdj_channel_keeps_all_frozen_entry_conditions():
    candidate = candidate_module()

    assert candidate.classify_entry_channel(_snapshot()) == "kdj_cross"


@pytest.mark.parametrize(
    "broken",
    [
        {"k_prev": 21.0},
        {"k": 20.0},
        {"rsi6": 30.01},
        {"low": 9.81},
        {"close": 9.8},
    ],
)
def test_original_kdj_channel_rejects_each_broken_condition(broken):
    candidate = candidate_module()

    snapshot = _snapshot(
        rsi6_2ago=20.0,
        rsi6_prev=21.0,
        close_prev=10.0,
        **broken,
    )
    assert candidate.classify_entry_channel(snapshot) is None


def test_rsi_low_turn_channel_enters_without_kdj_cross_or_boll_reclaim():
    candidate = candidate_module()
    snapshot = _snapshot(
        k_prev=22.0,
        d_prev=20.0,
        k=19.0,
        d=21.0,
        rsi6_2ago=32.0,
        rsi6_prev=28.0,
        rsi6=29.0,
        close_prev=9.6,
        low=10.1,
        close=9.7,
        boll_lower=9.8,
    )

    assert candidate.classify_entry_channel(snapshot) == "rsi_low_turn"


@pytest.mark.parametrize(
    "broken",
    [
        {"rsi6_2ago": 28.0},
        {"rsi6": 28.0},
        {"rsi6_prev": 30.01},
        {"close": 9.6},
        {"rsi6": float("nan")},
    ],
)
def test_rsi_low_turn_channel_rejects_equalities_high_trough_and_invalid_data(broken):
    candidate = candidate_module()
    values = {
        "k_prev": 22.0,
        "d_prev": 20.0,
        "k": 19.0,
        "d": 21.0,
        "close_prev": 9.6,
        "low": 10.1,
        "close": 9.7,
        "boll_lower": 9.8,
    }
    values.update(broken)
    snapshot = _snapshot(**values)

    assert candidate.classify_entry_channel(snapshot) is None


def test_same_etf_matching_both_channels_is_attributed_once_to_kdj():
    candidate = candidate_module()

    assert candidate.classify_entry_channel(_snapshot()) == "kdj_cross"


def test_signal_loader_ends_at_explicit_t1_and_maps_three_rsi_dates(monkeypatch):
    candidate = candidate_module()
    dates = pd.bdate_range("2019-12-23", periods=30)
    close = pd.Series(
        [10.0 + ((index % 7) - 3) * 0.08 + index * 0.01 for index in range(30)],
        index=dates,
    )
    frame = pd.DataFrame(
        {
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=dates,
    )
    calls = []

    def fake_get_price(code, **kwargs):
        calls.append((code, kwargs))
        return frame.copy()

    monkeypatch.setattr(candidate, "get_price", fake_get_price, raising=False)

    snapshot, reason = candidate.load_signal_snapshot(
        "513100.XSHG", dates[-1].date(), return_reason=True
    )

    assert reason is None
    assert calls == [
        (
            "513100.XSHG",
            {
                "end_date": dates[-1].date(),
                "count": 120,
                "frequency": "daily",
                "fields": ["high", "low", "close", "volume"],
                "skip_paused": True,
                "fq": "pre",
                "panel": False,
            },
        )
    ]
    assert snapshot["signal_date"] == dates[-1].date().isoformat()
    assert snapshot["max_data_date"] == dates[-1].date().isoformat()

    from cross_signal_strategy.research.rsi_low_turn_shadow import calculate_rsi6

    expected_rsi = calculate_rsi6(close.reset_index(drop=True))
    assert snapshot["rsi6_2ago"] == pytest.approx(expected_rsi.iloc[-3])
    assert snapshot["rsi6_prev"] == pytest.approx(expected_rsi.iloc[-2])
    assert snapshot["rsi6"] == pytest.approx(expected_rsi.iloc[-1])


def test_signal_loader_fails_closed_when_last_completed_bar_is_older_than_t1(monkeypatch):
    candidate = candidate_module()
    dates = pd.bdate_range("2019-12-02", periods=30)
    close = pd.Series([10.0 + index * 0.01 for index in range(30)], index=dates)
    frame = pd.DataFrame(
        {
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=dates,
    )
    requested_t1 = (dates[-1] + pd.offsets.BDay(1)).date()
    monkeypatch.setattr(candidate, "get_price", lambda *args, **kwargs: frame, raising=False)

    snapshot, reason = candidate.load_signal_snapshot(
        "513100.XSHG", requested_t1, return_reason=True
    )

    assert snapshot is None
    assert reason == "stale_signal_date"


def test_signal_loader_fails_closed_when_joinquant_data_request_raises(monkeypatch):
    candidate = candidate_module()
    monkeypatch.setattr(
        candidate,
        "get_price",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("data unavailable")),
        raising=False,
    )

    snapshot, reason = candidate.load_signal_snapshot(
        "513100.XSHG", pd.Timestamp("2020-01-08").date(), return_reason=True
    )

    assert snapshot is None
    assert reason == "exception:RuntimeError"


def test_buy_queue_prioritizes_kdj_then_uses_frozen_order_for_rsi_channel():
    candidate = candidate_module()
    rsi_first = _snapshot(
        code="159915.XSHE",
        k_prev=22.0,
        d_prev=20.0,
        k=19.0,
        d=21.0,
        low=10.1,
        close_prev=9.6,
        close=9.7,
        boll_lower=9.8,
    )
    rsi_later = dict(rsi_first, code="513050.XSHG")
    weaker_kdj = _snapshot(code="513500.XSHG", k=21.5, d=21.0)
    stronger_kdj = _snapshot(code="518880.XSHG", k=23.0, d=21.0)

    queue = candidate.build_buy_queue(
        [rsi_later, weaker_kdj, rsi_first, stronger_kdj],
        excluded_codes=set(),
    )

    assert [(item["code"], item["entry_channel"]) for item in queue] == [
        ("518880.XSHG", "kdj_cross"),
        ("513500.XSHG", "kdj_cross"),
        ("159915.XSHE", "rsi_low_turn"),
        ("513050.XSHG", "rsi_low_turn"),
    ]


def test_atr_exit_is_unconditional_and_precedes_boll_target():
    candidate = candidate_module()
    state = candidate.PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=11.0,
        mean_reached=True,
        upper_reached=True,
    )

    reason = candidate.choose_exit_reason(
        state,
        _snapshot(),
        current_price=10.44,
        hold_days=1,
        code="513100.XSHG",
    )

    assert reason == "atr_stop"


def test_boll_exits_require_five_sessions_and_preserve_frozen_order():
    candidate = candidate_module()
    upper = candidate.PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=10.5,
        upper_reached=True,
    )
    mean_weak = candidate.PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=10.5,
        mean_reached=True,
    )
    death_cross = _snapshot(k_prev=25.0, d_prev=24.0, k=20.0, d=22.0)

    assert candidate.choose_exit_reason(
        upper, _snapshot(), 10.4, 4, "513100.XSHG"
    ) is None
    assert candidate.choose_exit_reason(
        upper, _snapshot(), 10.4, 5, "513100.XSHG"
    ) == "boll_upper_target"
    assert candidate.choose_exit_reason(
        mean_weak, death_cross, 10.4, 5, "513100.XSHG"
    ) == "mean_reached_weakness"


def test_position_state_updates_only_from_completed_t1_close_and_bands():
    candidate = candidate_module()
    state = candidate.PositionSignalState(
        entry_date="2020-01-02",
        entry_price=10.0,
        entry_atr=0.2,
        highest_close=10.0,
    )

    candidate.update_state_from_t1(
        state,
        _snapshot(close=10.6, boll_mid=10.5, boll_upper=11.0),
    )

    assert state.highest_close == pytest.approx(10.6)
    assert state.mean_reached is True
    assert state.upper_reached is False


def test_initialize_registers_only_0935_main_and_1450_atr_callbacks(monkeypatch):
    candidate = candidate_module()
    scheduled = []
    options = []
    monkeypatch.setattr(candidate, "g", SimpleNamespace(), raising=False)
    monkeypatch.setattr(candidate, "set_benchmark", lambda value: None, raising=False)
    monkeypatch.setattr(
        candidate, "set_option", lambda name, value: options.append((name, value)), raising=False
    )
    monkeypatch.setattr(candidate, "set_slippage", lambda value: None, raising=False)
    monkeypatch.setattr(candidate, "PriceRelatedSlippage", lambda value: value, raising=False)
    monkeypatch.setattr(candidate, "set_order_cost", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(candidate, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(
        candidate,
        "run_daily",
        lambda function, time: scheduled.append((function.__name__, time)),
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.initialize(SimpleNamespace())

    assert ("avoid_future_data", True) in options
    assert scheduled == [("do_trading", "09:35"), ("check_atr_1450", "14:50")]
    assert candidate.g.position_states == {}
    assert candidate.g.sold_today == set()


def test_1450_callback_checks_only_atr_and_never_loads_signals(monkeypatch):
    candidate = candidate_module()
    candidate.g = SimpleNamespace(
        params=candidate.get_default_params(),
        position_states={
            "513100.XSHG": candidate.PositionSignalState(
                "2020-01-02", 10.0, 0.2, 11.0, upper_reached=True
            ),
            "513500.XSHG": candidate.PositionSignalState(
                "2020-01-02", 10.0, 0.2, 10.2, upper_reached=True
            ),
        },
        sold_today=set(),
        sold_guard_date="2020-01-06",
    )
    positions = {
        "513100.XSHG": SimpleNamespace(total_amount=100),
        "513500.XSHG": SimpleNamespace(total_amount=100),
    }
    context = SimpleNamespace(
        current_dt=pd.Timestamp("2020-01-06 14:50:00"),
        portfolio=SimpleNamespace(positions=positions),
    )
    quotes = {
        "513100.XSHG": SimpleNamespace(last_price=10.44, paused=False),
        "513500.XSHG": SimpleNamespace(last_price=10.0, paused=False),
    }
    orders = []

    def fill_sell(code, amount):
        orders.append((code, amount))
        positions.pop(code, None)
        return SimpleNamespace(filled=100, status="filled")

    monkeypatch.setattr(candidate, "get_current_data", lambda: quotes, raising=False)
    monkeypatch.setattr(
        candidate,
        "load_signal_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("14:50 loaded signals")),
    )
    monkeypatch.setattr(candidate, "order_target", fill_sell, raising=False)
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.check_atr_1450(context)

    assert orders == [("513100.XSHG", 0)]
    assert candidate.g.sold_today == {"513100.XSHG"}


@pytest.mark.parametrize(
    "order,before_amount,after_amount,expected",
    [
        (None, 100, 100, "rejected"),
        (SimpleNamespace(filled=0, status="canceled"), 100, 100, "rejected"),
        (SimpleNamespace(filled=0, status="held"), 100, 100, "pending"),
        (SimpleNamespace(filled=40, status="held"), 100, 60, "partial_pending"),
        (SimpleNamespace(filled=40, status="canceled"), 100, 60, "partial"),
        (SimpleNamespace(filled=100, status="filled"), 100, 100, "full"),
        (SimpleNamespace(filled=0, status="filled"), 100, 0, "full"),
    ],
)
def test_sell_submission_classifies_rejected_pending_partial_and_full(
    order, before_amount, after_amount, expected
):
    candidate = candidate_module()

    assert candidate.classify_sell_submission(
        order, before_amount, after_amount
    ) == expected


def test_sell_submission_uses_builtin_any_when_jqdata_shadows_global(monkeypatch):
    candidate = candidate_module()
    monkeypatch.setattr(candidate, "any", lambda values: False, raising=False)

    outcome = candidate.classify_sell_submission(
        SimpleNamespace(filled=0, status="held"),
        before_amount=100,
        after_amount=100,
    )

    assert outcome == "pending"


def test_rejected_0935_atr_sell_remains_eligible_for_1450_retry(monkeypatch):
    candidate = candidate_module()
    code = "513100.XSHG"
    candidate.g = SimpleNamespace(
        params=candidate.get_default_params(),
        etf_pool=candidate.get_default_etf_pool(),
        position_states={
            code: candidate.PositionSignalState("2020-01-02", 10.0, 0.2, 11.0)
        },
        last_snapshots={},
        sold_today=set(),
        pending_sells=set(),
        sold_guard_date=None,
    )
    positions = {code: SimpleNamespace(total_amount=100, avg_cost=10.0)}
    context = SimpleNamespace(
        current_dt=pd.Timestamp("2020-01-09 09:35:00"),
        portfolio=SimpleNamespace(
            positions=positions,
            total_value=20000.0,
            available_cash=10000.0,
        ),
    )
    quotes = {
        pool_code: SimpleNamespace(
            last_price=10.44 if pool_code == code else 10.0,
            paused=False,
        )
        for pool_code in candidate.get_default_etf_pool()
    }
    recent_days = [pd.Timestamp("2020-01-08").date(), pd.Timestamp("2020-01-09").date()]
    held_days = [pd.Timestamp("2020-01-02").date(), pd.Timestamp("2020-01-09").date()]
    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: recent_days if "count" in kwargs else held_days,
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "load_signal_snapshot",
        lambda *args, **kwargs: (None, "missing") if kwargs.get("return_reason") else None,
    )
    monkeypatch.setattr(candidate, "get_current_data", lambda: quotes, raising=False)
    attempts = []

    def fake_sell(order_code, amount):
        attempts.append((order_code, amount))
        if len(attempts) == 1:
            return SimpleNamespace(filled=0, status="canceled")
        positions.pop(order_code, None)
        return SimpleNamespace(filled=100, status="filled")

    monkeypatch.setattr(candidate, "order_target", fake_sell, raising=False)
    monkeypatch.setattr(candidate, "order_target_value", lambda *args: None, raising=False)
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.do_trading(context)
    assert candidate.g.sold_today == set()
    assert candidate.g.pending_sells == set()

    context.current_dt = pd.Timestamp("2020-01-09 14:50:00")
    candidate.check_atr_1450(context)

    assert attempts == [(code, 0), (code, 0)]
    assert candidate.g.sold_today == {code}
    assert code not in candidate.g.position_states


def test_1450_does_not_duplicate_an_active_pending_sell(monkeypatch):
    candidate = candidate_module()
    code = "513100.XSHG"
    candidate.g = SimpleNamespace(
        params=candidate.get_default_params(),
        position_states={
            code: candidate.PositionSignalState("2020-01-02", 10.0, 0.2, 11.0)
        },
        sold_today=set(),
        pending_sells={code},
        sold_guard_date="2020-01-09",
    )
    context = SimpleNamespace(
        current_dt=pd.Timestamp("2020-01-09 14:50:00"),
        portfolio=SimpleNamespace(
            positions={code: SimpleNamespace(total_amount=100, avg_cost=10.0)}
        ),
    )
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(last_price=10.44, paused=False)},
        raising=False,
    )
    attempts = []
    monkeypatch.setattr(
        candidate,
        "order_target",
        lambda order_code, amount: attempts.append((order_code, amount)),
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.check_atr_1450(context)

    assert attempts == []


def test_0935_plan_sells_before_filling_new_slot_and_never_rebuys_same_day():
    candidate = candidate_module()
    held_code = "513100.XSHG"
    held_state = candidate.PositionSignalState(
        "2020-01-02", 10.0, 0.2, 10.8, upper_reached=True
    )
    kdj = _snapshot(code="518880.XSHG")
    sold_candidate = _snapshot(code="513500.XSHG", k=30.0, d=21.0)
    rsi = _snapshot(
        code="159915.XSHE",
        k_prev=22.0,
        d_prev=20.0,
        k=19.0,
        d=21.0,
        low=10.1,
        close_prev=9.6,
        close=9.7,
        boll_lower=9.8,
    )
    held_snapshot = _snapshot(
        code=held_code,
        close=11.2,
        boll_upper=11.0,
        entry_channel=None,
    )
    trade_days = [
        pd.Timestamp("2020-01-02").date(),
        pd.Timestamp("2020-01-03").date(),
        pd.Timestamp("2020-01-06").date(),
        pd.Timestamp("2020-01-07").date(),
        pd.Timestamp("2020-01-08").date(),
        pd.Timestamp("2020-01-09").date(),
    ]

    plans = candidate.plan_0935_orders(
        snapshots=[rsi, held_snapshot, sold_candidate, kdj],
        held_codes=[held_code],
        position_states={held_code: held_state},
        current_prices={held_code: 11.1},
        today="2020-01-09",
        trade_days=trade_days,
        total_value=20000.0,
        sold_today={"513500.XSHG"},
    )

    assert plans == [
        {"code": held_code, "target_value": 0.0, "reason": "boll_upper_target"},
        {
            "code": "518880.XSHG",
            "target_value": pytest.approx(20000.0 * 0.95 / 3),
            "reason": "kdj_cross",
            "entry_atr": pytest.approx(0.2),
        },
        {
            "code": "159915.XSHE",
            "target_value": pytest.approx(20000.0 * 0.95 / 3),
            "reason": "rsi_low_turn",
            "entry_atr": pytest.approx(0.2),
        },
    ]


def test_0935_atr_stop_survives_missing_t1_indicator_snapshot():
    candidate = candidate_module()
    code = "513100.XSHG"
    state = candidate.PositionSignalState("2020-01-02", 10.0, 0.2, 11.0)

    plans = candidate.plan_0935_orders(
        snapshots=[],
        held_codes=[code],
        position_states={code: state},
        current_prices={code: 10.44},
        today="2020-01-09",
        trade_days=["2020-01-02", "2020-01-09"],
        total_value=20000.0,
        sold_today=set(),
    )

    assert plans == [{"code": code, "target_value": 0.0, "reason": "atr_stop"}]


def test_0935_callback_loads_t1_signals_and_freezes_filled_entry_state(monkeypatch):
    candidate = candidate_module()
    candidate.g = SimpleNamespace(
        params=candidate.get_default_params(),
        etf_pool=candidate.get_default_etf_pool(),
        position_states={},
        last_snapshots={},
        sold_today=set(),
        sold_guard_date=None,
    )
    positions = {}
    context = SimpleNamespace(
        current_dt=pd.Timestamp("2020-01-09 09:35:00"),
        portfolio=SimpleNamespace(
            positions=positions,
            total_value=20000.0,
            available_cash=20000.0,
        ),
    )
    signal_dates = []
    eligible = _snapshot(code="159915.XSHE")

    def fake_load(code, signal_date, return_reason=False):
        signal_dates.append((code, signal_date))
        value = dict(eligible, code=code) if code == "159915.XSHE" else None
        result = (value, None if value is not None else "not_eligible")
        return result if return_reason else value

    quotes = {
        code: SimpleNamespace(last_price=10.0, paused=False)
        for code in candidate.get_default_etf_pool()
    }
    orders = []

    def fake_order_target_value(code, target_value):
        orders.append((code, target_value))
        positions[code] = SimpleNamespace(total_amount=600, avg_cost=10.01)
        return SimpleNamespace(filled=600)

    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: [pd.Timestamp("2020-01-08").date(), pd.Timestamp("2020-01-09").date()],
        raising=False,
    )
    monkeypatch.setattr(candidate, "load_signal_snapshot", fake_load)
    monkeypatch.setattr(candidate, "get_current_data", lambda: quotes, raising=False)
    monkeypatch.setattr(candidate, "order_target_value", fake_order_target_value, raising=False)
    monkeypatch.setattr(candidate, "order_target", lambda *args: None, raising=False)
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.do_trading(context)

    assert signal_dates == [
        (code, pd.Timestamp("2020-01-08").date())
        for code in candidate.get_default_etf_pool()
    ]
    assert orders == [("159915.XSHE", pytest.approx(20000.0 * 0.95 / 3))]
    state = candidate.g.position_states["159915.XSHE"]
    assert state.entry_date == "2020-01-09"
    assert state.entry_price == pytest.approx(10.01)
    assert state.entry_atr == pytest.approx(0.2)


def test_0935_callback_executes_planned_sell_before_replacement_buy(monkeypatch):
    candidate = candidate_module()
    held_code = "513100.XSHG"
    candidate.g = SimpleNamespace(
        params=candidate.get_default_params(),
        etf_pool=candidate.get_default_etf_pool(),
        position_states={
            held_code: candidate.PositionSignalState(
                "2020-01-02", 10.0, 0.2, 10.8, upper_reached=True
            )
        },
        last_snapshots={},
        sold_today=set(),
        sold_guard_date=None,
    )
    positions = {held_code: SimpleNamespace(total_amount=100, avg_cost=10.0)}
    context = SimpleNamespace(
        current_dt=pd.Timestamp("2020-01-09 09:35:00"),
        portfolio=SimpleNamespace(
            positions=positions,
            total_value=20000.0,
            available_cash=10000.0,
        ),
    )
    buy_snapshot = _snapshot(code="159915.XSHE")
    held_snapshot = _snapshot(code=held_code, close=11.2, boll_upper=11.0)

    def fake_load(code, signal_date, return_reason=False):
        value = {"159915.XSHE": buy_snapshot, held_code: held_snapshot}.get(code)
        result = (value, None if value is not None else "not_eligible")
        return result if return_reason else value

    recent_days = [pd.Timestamp("2020-01-08").date(), pd.Timestamp("2020-01-09").date()]
    held_days = [
        pd.Timestamp("2020-01-02").date(),
        pd.Timestamp("2020-01-03").date(),
        pd.Timestamp("2020-01-06").date(),
        pd.Timestamp("2020-01-07").date(),
        pd.Timestamp("2020-01-08").date(),
        pd.Timestamp("2020-01-09").date(),
    ]
    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: recent_days if "count" in kwargs else held_days,
        raising=False,
    )
    monkeypatch.setattr(candidate, "load_signal_snapshot", fake_load)
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {
            code: SimpleNamespace(last_price=11.1 if code == held_code else 10.0, paused=False)
            for code in candidate.get_default_etf_pool()
        },
        raising=False,
    )
    orders = []

    def fake_sell(code, amount):
        orders.append(("sell", code, amount))
        positions.pop(code, None)

    def fake_buy(code, target):
        orders.append(("buy", code, target))
        positions[code] = SimpleNamespace(total_amount=600, avg_cost=10.01)
        return SimpleNamespace(filled=600)

    monkeypatch.setattr(candidate, "order_target", fake_sell, raising=False)
    monkeypatch.setattr(candidate, "order_target_value", fake_buy, raising=False)
    monkeypatch.setattr(
        candidate,
        "log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        raising=False,
    )

    candidate.do_trading(context)

    assert orders[0] == ("sell", held_code, 0)
    assert orders[1][0:2] == ("buy", "159915.XSHE")
    assert held_code not in candidate.g.position_states
    assert held_code in candidate.g.sold_today
