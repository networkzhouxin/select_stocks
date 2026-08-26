# -*- coding: utf-8 -*-
"""Tests for the standalone weekly-trend/daily-pullback JoinQuant candidate."""

from __future__ import annotations

import __future__
import builtins
from datetime import date
from datetime import datetime
import pathlib
import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

CANDIDATE_PATH = (
    ROOT
    / "cross_signal_strategy"
    / "smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py"
)

from cross_signal_strategy import (  # noqa: E402
    smart_trade_joinquant_weekly_trend_daily_pullback_candidate as candidate,
)
from cross_signal_strategy.research import (  # noqa: E402
    weekly_trend_daily_pullback_candidate as research,
)


def _daily_frame(periods=150, end="2021-03-09"):
    dates = pd.bdate_range(end=end, periods=periods)
    closes = pd.Series(
        [10.0 + index * 0.02 + ((index % 9) - 4) * 0.03 for index in range(periods)],
        index=dates,
        dtype=float,
    )
    return pd.DataFrame(
        {
            "open": closes - 0.03,
            "high": closes + 0.10,
            "low": closes - 0.10,
            "close": closes,
            "volume": [1000.0] * periods,
        },
        index=dates,
    )


def test_joinquant_candidate_compiles_without_future_annotations_or_dataclasses():
    source = CANDIDATE_PATH.read_text(encoding="utf-8")
    compiled = compile(source, str(CANDIDATE_PATH), "exec", dont_inherit=True)
    assert compiled.co_flags & __future__.annotations.compiler_flag == 0

    real_import = builtins.__import__

    def import_without_dataclasses(name, *args, **kwargs):
        if name == "dataclasses":
            raise AssertionError("JoinQuant upload imported dataclasses")
        return real_import(name, *args, **kwargs)

    namespace = {"__name__": "weekly_pullback_joinquant_probe"}
    original_import = builtins.__import__
    try:
        builtins.__import__ = import_without_dataclasses
        exec(compiled, namespace)
    finally:
        builtins.__import__ = original_import

    state = namespace["PositionSignalState"]("2021-03-08", 10.0, 0.2, 10.0)
    assert state.entry_atr == pytest.approx(0.2)


def test_default_params_and_pool_are_frozen_for_the_independent_candidate():
    assert candidate.STRATEGY_VERSION == (
        "weekly-trend-pullback-v0.1-joinquant-candidate"
    )
    assert candidate.DEPLOYMENT_BUILD_ID == "20260826.1-candidate"
    assert candidate.LOOKBACK == 180
    assert candidate.get_default_params() == {
        "lookback": 180,
        "max_hold": 3,
        "base_ratio": 0.95,
        "min_signal_hold_days": 5,
        "atr_multiplier": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
    }
    assert candidate.get_default_etf_pool() == [
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


def test_loader_requests_daily_bars_ending_at_explicit_t1(monkeypatch):
    frame = _daily_frame()
    calls = []

    def fake_get_price(code, **kwargs):
        calls.append((code, kwargs))
        return frame.copy()

    monkeypatch.setattr(candidate, "get_price", fake_get_price, raising=False)

    snapshot, reason = candidate.load_signal_snapshot(
        "513100.XSHG",
        signal_date=date(2021, 3, 9),
        decision_date=date(2021, 3, 10),
        return_reason=True,
    )

    assert reason is None
    assert calls == [
        (
            "513100.XSHG",
            {
                "end_date": date(2021, 3, 9),
                "count": 180,
                "frequency": "daily",
                "fields": ["open", "high", "low", "close", "volume"],
                "skip_paused": True,
                "fq": "pre",
                "panel": False,
            },
        )
    ]
    assert snapshot["signal_date"] == "2021-03-09"
    assert snapshot["max_data_date"] == "2021-03-09"
    assert snapshot["weekly_period_end"] == "2021-03-07"
    assert snapshot["weekly_last_trade_date"] == "2021-03-05"


def test_loader_fails_closed_when_last_daily_bar_is_older_than_t1():
    daily_frame = _daily_frame()

    result, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        daily_frame.loc[:"2021-03-08"],
    )

    assert result is None
    assert reason == "stale_signal_date"


def test_snapshot_excludes_partial_decision_week_from_weekly_context():
    frame = _daily_frame()
    current_week_close = float(frame.loc[pd.Timestamp("2021-03-09"), "close"])

    snapshot, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        frame,
    )

    assert reason is None
    assert snapshot["weekly_last_trade_date"] == "2021-03-05"
    assert snapshot["weekly_close"] != pytest.approx(current_week_close)


def test_snapshot_requires_21_completed_weeks():
    dates = pd.date_range(end="2021-03-05", periods=20, freq="W-FRI")
    closes = pd.Series(range(1, 21), index=dates, dtype=float)
    frame = pd.DataFrame(
        {
            "open": closes - 0.1,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=dates,
    )
    current_week = frame.iloc[[-1]].copy()
    current_week.index = pd.DatetimeIndex(["2021-03-09"])
    current_week.loc[:, "close"] = 99.0
    frame = pd.concat([frame, current_week]).sort_index()

    snapshot, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        frame,
    )

    assert snapshot is None
    assert reason == "insufficient_weekly_history"


def test_snapshot_fails_closed_for_malformed_data_and_zero_recent_volume():
    malformed = _daily_frame().drop(columns=["open"])
    result, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        malformed,
    )
    assert result is None
    assert reason == "missing_daily_data"

    zero_volume = _daily_frame()
    zero_volume.loc[zero_volume.index[-5:], "volume"] = 0.0
    result, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        zero_volume,
    )
    assert result is None
    assert reason == "zero_recent_volume"


def test_joinquant_predicates_match_research_predicates():
    snapshot, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        _daily_frame(),
    )
    assert reason is None

    assert candidate.is_daily_entry_eligible(snapshot) == (
        research.is_daily_entry_eligible(snapshot)
    )
    assert candidate.is_entry_eligible(snapshot) == research.is_entry_eligible(
        snapshot
    )
    joinquant_state = candidate.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 11.0
    )
    research_state = research.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 11.0
    )
    assert candidate.calc_frozen_atr_stop(
        joinquant_state, "513100.XSHG"
    ) == pytest.approx(
        research.calc_frozen_atr_stop(research_state, "513100.XSHG")
    )
    assert candidate.choose_exit_reason(
        joinquant_state,
        snapshot,
        10.80,
        8,
        "513100.XSHG",
    ) == research.choose_exit_reason(
        research_state,
        snapshot,
        10.80,
        8,
        "513100.XSHG",
    )


def _eligible_snapshot(code="159915.XSHE", **overrides):
    values = {
        "code": code,
        "signal_date": "2021-03-05",
        "max_data_date": "2021-03-05",
        "weekly_period_end": "2021-02-28",
        "weekly_last_trade_date": "2021-02-26",
        "weekly_close": 11.0,
        "weekly_ma20": 10.0,
        "weekly_ma20_prev": 9.0,
        "close": 10.0,
        "boll_lower": 9.7,
        "boll_mid": 10.0,
        "boll_upper": 10.3,
        "k_prev": 19.0,
        "d_prev": 20.0,
        "k": 22.0,
        "d": 20.0,
        "rsi6_prev": 39.0,
        "rsi6": 40.0,
        "atr": 0.2,
    }
    values.update(overrides)
    return values


class _LogCapture:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warning_messages.append(str(message))


def _runtime_state(**overrides):
    values = {
        "params": candidate.get_default_params(),
        "etf_pool": [],
        "position_states": {},
        "last_snapshots": {},
        "sold_today": set(),
        "pending_sells": set(),
        "sold_guard_date": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(positions=None, cash=20000.0, total_value=20000.0, minute=35):
    return SimpleNamespace(
        current_dt=datetime(2021, 3, 8, 9 if minute == 35 else 14, minute),
        portfolio=SimpleNamespace(
            positions=dict(positions or {}),
            available_cash=float(cash),
            total_value=float(total_value),
        ),
    )


def test_initialize_registers_only_0935_and_1450(monkeypatch):
    scheduled = []
    options = []
    monkeypatch.setattr(candidate, "g", SimpleNamespace(), raising=False)
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)
    monkeypatch.setattr(candidate, "set_benchmark", lambda value: None, raising=False)
    monkeypatch.setattr(
        candidate,
        "set_option",
        lambda name, value: options.append((name, value)),
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "PriceRelatedSlippage",
        lambda value: ("slippage", value),
        raising=False,
    )
    monkeypatch.setattr(candidate, "set_slippage", lambda value: None, raising=False)
    monkeypatch.setattr(candidate, "OrderCost", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(
        candidate,
        "set_order_cost",
        lambda value, type=None: None,
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "run_daily",
        lambda function, time: scheduled.append((function.__name__, time)),
        raising=False,
    )

    candidate.initialize(SimpleNamespace())

    assert scheduled == [("do_trading", "09:35"), ("check_atr_1450", "14:50")]
    assert ("avoid_future_data", True) in options
    assert candidate.g.position_states == {}
    assert candidate.g.pending_sells == set()


def test_joinquant_plan_0935_keeps_sell_first_fixed_slot_and_sold_guard():
    held = "513100.XSHG"
    new = "159915.XSHE"
    sold_today = "518880.XSHG"
    snapshots = [
        _eligible_snapshot(
            held,
            weekly_close=9.0,
            weekly_ma20=10.0,
            weekly_ma20_prev=11.0,
        ),
        _eligible_snapshot(new),
        _eligible_snapshot(sold_today, weekly_close=12.0),
    ]
    states = {
        held: candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    }

    plans = candidate.plan_0935_orders(
        snapshots=snapshots,
        held_codes=[held],
        position_states=states,
        current_prices={held: 10.8, new: 10.0, sold_today: 10.0},
        today="2021-03-08",
        trade_days=[date(2021, 3, 1), date(2021, 3, 8)],
        total_value=20000.0,
        sold_today={sold_today},
        params=candidate.get_default_params(),
    )

    assert plans[0] == {
        "code": held,
        "target_value": 0.0,
        "reason": "weekly_trend_break",
    }
    assert plans[1] == {
        "code": new,
        "target_value": pytest.approx(20000.0 * 0.95 / 3),
        "reason": "weekly_pullback_entry",
        "entry_atr": pytest.approx(0.2),
    }
    assert sold_today not in [plan["code"] for plan in plans[1:]]


def test_plan_0935_refreshes_highest_close_from_t1_before_atr_check():
    code = "513100.XSHG"
    state = candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 10.0)
    snapshot = _eligible_snapshot(code, close=11.0, boll_mid=11.2)

    plans = candidate.plan_0935_orders(
        snapshots=[snapshot],
        held_codes=[code],
        position_states={code: state},
        current_prices={code: 10.44},
        today="2021-03-08",
        trade_days=[date(2021, 3, 1), date(2021, 3, 8)],
        total_value=20000.0,
        params=candidate.get_default_params(),
    )

    assert state.highest_close == pytest.approx(11.0)
    assert plans == [
        {"code": code, "target_value": 0.0, "reason": "atr_stop"}
    ]


def test_1450_never_loads_daily_or_weekly_signals(monkeypatch):
    code = "513100.XSHG"
    context = _context(
        positions={code: SimpleNamespace(total_amount=100, avg_cost=10.0)},
        minute=50,
    )
    state = candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(position_states={code: state}, sold_guard_date="2021-03-08"),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)

    def exploding_loader(*args, **kwargs):
        raise AssertionError("14:50 loaded ordinary signals")

    orders = []
    monkeypatch.setattr(candidate, "load_signal_snapshot", exploding_loader)
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(paused=False, last_price=10.44)},
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "order_target",
        lambda order_code, target: orders.append((order_code, target))
        or SimpleNamespace(filled=100, status="filled"),
        raising=False,
    )

    candidate.check_atr_1450(context)

    assert orders == [(code, 0)]


def test_0935_executes_all_planned_sells_before_any_buy(monkeypatch):
    held = "513100.XSHG"
    new = "159915.XSHE"
    held_position = SimpleNamespace(total_amount=100, avg_cost=10.0)
    context = _context(positions={held: held_position})
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(
            etf_pool=[held, new],
            position_states={
                held: candidate.PositionSignalState(
                    "2021-03-01", 10.0, 0.2, 11.0
                )
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)
    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: [date(2021, 3, 5), date(2021, 3, 8)],
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "load_signal_snapshot",
        lambda code, signal_date, decision_date, return_reason=False: (
            _eligible_snapshot(code),
            None,
        ),
    )
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {
            held: SimpleNamespace(paused=False, last_price=10.8),
            new: SimpleNamespace(paused=False, last_price=10.0),
        },
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "plan_0935_orders",
        lambda **kwargs: [
            {"code": held, "target_value": 0.0, "reason": "weekly_trend_break"},
            {
                "code": new,
                "target_value": 20000.0 * 0.95 / 3,
                "reason": "weekly_pullback_entry",
                "entry_atr": 0.2,
            },
        ],
    )
    orders = []

    def sell(code, target):
        orders.append(("sell", code, target))
        context.portfolio.positions.pop(code, None)
        return SimpleNamespace(filled=100, status="filled")

    def buy(code, target):
        orders.append(("buy", code, target))
        context.portfolio.positions[code] = SimpleNamespace(
            total_amount=600,
            avg_cost=10.01,
        )
        return SimpleNamespace(filled=600, status="filled")

    monkeypatch.setattr(candidate, "order_target", sell, raising=False)
    monkeypatch.setattr(candidate, "order_target_value", buy, raising=False)

    candidate.do_trading(context)

    assert [item[0] for item in orders] == ["sell", "buy"]


@pytest.mark.parametrize(
    "order,before,after,expected",
    [
        (None, 100, 100, "rejected"),
        (SimpleNamespace(filled=0, status="held"), 100, 100, "pending"),
        (
            SimpleNamespace(filled=40, status="held"),
            100,
            60,
            "partial_pending",
        ),
        (SimpleNamespace(filled=40, status="canceled"), 100, 60, "partial"),
        (SimpleNamespace(filled=100, status="filled"), 100, 0, "full"),
    ],
)
def test_sell_submission_state_machine(order, before, after, expected):
    assert candidate.classify_sell_submission(order, before, after) == expected


def test_sell_submission_uses_builtin_any_if_jqdata_shadows_global(monkeypatch):
    monkeypatch.setattr(
        candidate,
        "any",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shadowed jqdata.any was called")
        ),
        raising=False,
    )
    order = SimpleNamespace(filled=0, status="held")

    assert candidate.classify_sell_submission(order, 100, 100) == "pending"


def test_rejected_0935_atr_sell_can_retry_at_1450(monkeypatch):
    code = "513100.XSHG"
    context = _context(
        positions={code: SimpleNamespace(total_amount=100, avg_cost=10.0)},
        minute=50,
    )
    state = candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(
            position_states={code: state},
            pending_sells=set(),
            sold_today=set(),
            sold_guard_date="2021-03-08",
        ),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)

    assert candidate._record_sell_submission(code, None, 100, 100) == "rejected"
    assert code not in candidate.g.pending_sells

    attempts = []
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(paused=False, last_price=10.44)},
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "order_target",
        lambda order_code, target: attempts.append((order_code, target))
        or SimpleNamespace(filled=100, status="filled"),
        raising=False,
    )

    candidate.check_atr_1450(context)

    assert attempts == [(code, 0)]


def test_active_pending_sell_is_not_duplicated_at_1450(monkeypatch):
    code = "513100.XSHG"
    context = _context(
        positions={code: SimpleNamespace(total_amount=100, avg_cost=10.0)},
        minute=50,
    )
    state = candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(
            position_states={code: state},
            pending_sells=set(),
            sold_today=set(),
            sold_guard_date="2021-03-08",
        ),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)
    order = SimpleNamespace(filled=0, status="held")
    assert candidate._record_sell_submission(code, order, 100, 100) == "pending"

    attempts = []
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(paused=False, last_price=10.44)},
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "order_target",
        lambda order_code, target: attempts.append((order_code, target)),
        raising=False,
    )

    candidate.check_atr_1450(context)

    assert attempts == []


def test_buy_state_is_created_only_after_a_confirmed_fill(monkeypatch):
    code = "159915.XSHE"
    context = _context()
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(etf_pool=[code]),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", _LogCapture(), raising=False)
    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: [date(2021, 3, 5), date(2021, 3, 8)],
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "load_signal_snapshot",
        lambda code, signal_date, decision_date, return_reason=False: (
            _eligible_snapshot(code),
            None,
        ),
    )
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(paused=False, last_price=10.0)},
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "order_target_value",
        lambda code, value: None,
        raising=False,
    )

    candidate.do_trading(context)
    assert candidate.g.position_states == {}

    def filled_buy(order_code, value):
        context.portfolio.positions[order_code] = SimpleNamespace(
            total_amount=600,
            avg_cost=10.01,
        )
        return SimpleNamespace(filled=600, status="filled")

    monkeypatch.setattr(candidate, "order_target_value", filled_buy)
    candidate.do_trading(context)

    assert candidate.g.position_states[code].entry_atr == pytest.approx(0.2)


def test_0935_snapshot_log_contains_exact_causal_fields(monkeypatch):
    code = "159915.XSHE"
    log_capture = _LogCapture()
    context = _context()
    monkeypatch.setattr(
        candidate,
        "g",
        _runtime_state(etf_pool=[code]),
        raising=False,
    )
    monkeypatch.setattr(candidate, "log", log_capture, raising=False)
    monkeypatch.setattr(
        candidate,
        "get_trade_days",
        lambda **kwargs: [date(2021, 3, 5), date(2021, 3, 8)],
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "load_signal_snapshot",
        lambda code, signal_date, decision_date, return_reason=False: (
            _eligible_snapshot(code),
            None,
        ),
    )
    monkeypatch.setattr(
        candidate,
        "get_current_data",
        lambda: {code: SimpleNamespace(paused=False, last_price=10.0)},
        raising=False,
    )
    monkeypatch.setattr(
        candidate,
        "order_target_value",
        lambda code, value: None,
        raising=False,
    )

    candidate.do_trading(context)

    snapshot_logs = [
        message for message in log_capture.info_messages if "[09:35 snapshot]" in message
    ]
    assert len(snapshot_logs) == 1
    message = snapshot_logs[0]
    for field in (
        "decision=2021-03-08",
        "signal=2021-03-05",
        "week_end=2021-02-28",
        "w_close=",
        "w_ma20=",
        "w_ma20_prev=",
        "boll=",
        "kd=",
        "rsi=",
        "atr=",
        "eligible=",
    ):
        assert field in message
