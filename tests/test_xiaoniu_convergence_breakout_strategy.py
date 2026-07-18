# -*- coding: utf-8 -*-
"""Tests for the frozen Xiaoniu V4 convergence-breakout pattern."""

import ast
import importlib.util
import inspect
import pathlib
import sys
import types
from datetime import date, datetime

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

spec = importlib.util.spec_from_file_location(
    "xiaoniu_v4",
    ROOT / "xiaoniustock" / "xiaoniustock_joinquant_v4.py",
)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


def make_valid_bars():
    size = 67
    close = np.full(size, 100.0)
    open_ = np.full(size, 99.8)
    high = np.full(size, 100.5)
    low = np.full(size, 99.5)
    volume = np.full(size, 100.0)

    breakout = 65
    close[breakout] = 104.0
    open_[breakout] = 100.0
    high[breakout] = 105.0
    low[breakout] = 99.8
    volume[breakout] = 180.0

    confirmation = 66
    close[confirmation] = 103.5
    open_[confirmation] = 102.5
    high[confirmation] = 104.0
    low[confirmation] = 102.0
    volume[confirmation] = 90.0

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def test_detects_valid_convergence_breakout_pullback():
    signal = strategy.detect_convergence_breakout(make_valid_bars())

    assert signal is not None
    assert signal["breakout_index"] == 65
    assert signal["confirmation_index"] == 66
    assert signal["breakout_level"] == 100.0
    assert signal["volume_ratio"] == 1.8
    assert signal["stop_price"] == 99.0


def test_rejects_when_moving_averages_do_not_converge():
    bars = make_valid_bars()
    rising_close = np.linspace(70.0, 100.0, 65)
    bars["close"][:65] = rising_close
    bars["open"][:65] = rising_close - 0.2
    bars["high"][:65] = rising_close + 0.5
    bars["low"][:65] = rising_close - 0.5

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_when_breakout_does_not_clear_prior_high():
    bars = make_valid_bars()
    bars["close"][65] = 100.0

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_when_breakout_volume_is_below_threshold():
    bars = make_valid_bars()
    bars["volume"][65] = 149.0

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_confirmation_after_five_sessions():
    size = 72
    close = np.full(size, 100.0)
    open_ = np.full(size, 99.8)
    high = np.full(size, 100.5)
    low = np.full(size, 99.5)
    volume = np.full(size, 100.0)

    close[65], open_[65], high[65], low[65], volume[65] = (
        104.0, 100.0, 105.0, 99.8, 180.0)
    close[66:71] = 98.0
    open_[66:71] = 99.0
    high[66:71] = 99.5
    low[66:71] = 97.5
    close[71], open_[71], high[71], low[71], volume[71] = (
        103.5, 102.5, 104.0, 102.0, 90.0)
    bars = {
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_confirmation_close_below_breakout_support():
    bars = make_valid_bars()
    bars["open"][-1] = 99.5
    bars["close"][-1] = 99.9

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_confirmation_low_beyond_one_percent_tolerance():
    bars = make_valid_bars()
    bars["low"][-1] = 98.9

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_confirmation_without_volume_contraction():
    bars = make_valid_bars()
    bars["volume"][-1] = 180.0

    assert strategy.detect_convergence_breakout(bars) is None


def test_rejects_bearish_confirmation_candle():
    bars = make_valid_bars()
    bars["open"][-1] = 104.0

    assert strategy.detect_convergence_breakout(bars) is None


def test_invalid_nan_or_zero_volume_is_rejected():
    nan_bars = make_valid_bars()
    nan_bars["close"][10] = np.nan
    zero_volume_bars = make_valid_bars()
    zero_volume_bars["volume"][10] = 0.0
    scalar_bars = {name: 1.0 for name in (
        "open", "high", "low", "close", "volume")}

    assert strategy.detect_convergence_breakout(nan_bars) is None
    assert strategy.detect_convergence_breakout(zero_volume_bars) is None
    assert strategy.detect_convergence_breakout(scalar_bars) is None


def test_rank_candidates_uses_strength_volume_then_code():
    candidates = [
        {"code": "BBB", "confirmation_close": 103.0,
         "breakout_level": 100.0, "volume_ratio": 2.0},
        {"code": "CCC", "confirmation_close": 104.0,
         "breakout_level": 100.0, "volume_ratio": 1.6},
        {"code": "DDD", "confirmation_close": 103.0,
         "breakout_level": 100.0, "volume_ratio": 2.2},
        {"code": "AAA", "confirmation_close": 103.0,
         "breakout_level": 100.0, "volume_ratio": 2.0},
    ]

    ranked = strategy.rank_candidates(candidates)

    assert [item["code"] for item in ranked] == ["CCC", "DDD", "AAA", "BBB"]


def test_calculate_order_shares_uses_one_percent_risk_budget():
    shares = strategy.calculate_order_shares(
        total_value=100000.0,
        available_cash=100000.0,
        execution_price=10.0,
        stop_price=9.5,
    )

    assert shares == 2000


def test_calculate_order_shares_respects_position_and_cash_caps():
    position_capped = strategy.calculate_order_shares(
        total_value=100000.0,
        available_cash=100000.0,
        execution_price=10.0,
        stop_price=9.9,
    )
    cash_capped = strategy.calculate_order_shares(
        total_value=100000.0,
        available_cash=1500.0,
        execution_price=10.0,
        stop_price=9.5,
    )
    unaffordable = strategy.calculate_order_shares(
        total_value=100000.0,
        available_cash=999.0,
        execution_price=10.0,
        stop_price=9.5,
    )

    assert position_capped == 3000
    assert cash_capped == 100
    assert unaffordable == 0


def test_calculate_order_shares_rejects_invalid_stop_distance():
    assert strategy.calculate_order_shares(100000, 100000, 10.0, 10.0) == 0
    assert strategy.calculate_order_shares(100000, 100000, 10.0, 10.1) == 0
    assert strategy.calculate_order_shares(100000, 100000, np.nan, 9.5) == 0


def test_market_gate_requires_price_above_ma60_and_non_declining_ma():
    healthy = np.full(65, 100.0)
    below_ma = np.full(65, 100.0)
    below_ma[-1] = 80.0
    declining_ma = np.full(65, 110.0)
    declining_ma[-5:] = [100.0, 100.0, 100.0, 100.0, 120.0]

    assert strategy.market_gate_is_open(healthy)
    assert not strategy.market_gate_is_open(below_ma)
    assert not strategy.market_gate_is_open(declining_ma)


def test_exit_reasons_are_shape_ma20_or_time_only():
    assert strategy.should_exit_position(95.0, 96.0, 97.0, 5) == "shape_stop"
    assert strategy.should_exit_position(98.0, 99.0, 97.0, 5) == "ma20"
    assert strategy.should_exit_position(100.0, 99.0, 97.0, 20) == "time"
    assert strategy.should_exit_position(100.0, 99.0, 97.0, 19) is None
    assert strategy.should_exit_position(97.0, 97.0, 97.0, 19) is None


def test_initialize_schedules_snapshot_execution_and_close_tasks(monkeypatch):
    scheduled = []
    strategy.g = types.SimpleNamespace()
    monkeypatch.setattr(strategy, "set_benchmark", lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(strategy, "set_option", lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: value,
                        raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kwargs: kwargs,
                        raising=False)
    monkeypatch.setattr(
        strategy,
        "run_daily",
        lambda func, time: scheduled.append((func.__name__, time)),
        raising=False,
    )

    strategy.initialize(types.SimpleNamespace())

    assert scheduled == [
        ("prepare_daily_state", "09:30"),
        ("execute_trades", "09:35"),
        ("record_close_state", "15:00"),
        ("after_close", "15:30"),
    ]


def test_previous_trade_date_is_explicitly_resolved(monkeypatch):
    captured = {}

    def fake_trade_days(**kwargs):
        captured.update(kwargs)
        return [date(2026, 7, 17), date(2026, 7, 20)]

    monkeypatch.setattr(strategy, "get_trade_days", fake_trade_days, raising=False)
    context = types.SimpleNamespace(current_dt=datetime(2026, 7, 20, 9, 30))

    assert strategy.get_prev_trade_date(context) == date(2026, 7, 17)
    assert captured == {"end_date": date(2026, 7, 20), "count": 2}


def test_daily_history_loader_ends_at_previous_date(monkeypatch):
    captured = {}
    frame = {
        "open": np.full(90, 100.0),
        "high": np.full(90, 101.0),
        "low": np.full(90, 99.0),
        "close": np.full(90, 100.0),
        "volume": np.full(90, 100.0),
    }

    def fake_get_price(code, **kwargs):
        captured["code"] = code
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(strategy, "get_price", fake_get_price, raising=False)
    prev_date = date(2026, 7, 17)

    bars = strategy.load_daily_bars("600000.XSHG", prev_date, count=90)

    assert len(bars["close"]) == 90
    assert captured["end_date"] == prev_date
    assert captured["frequency"] == "daily"
    assert captured["skip_paused"] is True


def test_point_in_time_universe_uses_previous_date(monkeypatch):
    calls = []

    def fake_index_stocks(index, date=None):
        calls.append((index, date))
        if index == "000300.XSHG":
            return ["600000.XSHG", "300001.XSHE"]
        return ["000001.XSHE", "600000.XSHG"]

    monkeypatch.setattr(strategy, "get_index_stocks", fake_index_stocks,
                        raising=False)
    monkeypatch.setattr(strategy, "get_extras", lambda *args, **kwargs: None,
                        raising=False)
    prev_date = date(2026, 7, 17)

    universe = strategy.get_point_in_time_universe(prev_date)

    assert universe == ["000001.XSHE", "600000.XSHG"]
    assert calls == [
        ("000300.XSHG", prev_date),
        ("000905.XSHG", prev_date),
    ]


def test_prepare_snapshot_does_not_read_current_execution_quotes(monkeypatch):
    strategy.g = types.SimpleNamespace(
        managed_codes=set(),
        stop_prices={},
        entry_dates={},
    )
    context = types.SimpleNamespace(
        current_dt=datetime(2026, 7, 20, 9, 30),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(strategy, "get_prev_trade_date",
                        lambda context: date(2026, 7, 17), raising=False)
    monkeypatch.setattr(strategy, "get_market_closes",
                        lambda prev_date: np.full(65, 100.0), raising=False)
    monkeypatch.setattr(strategy, "get_point_in_time_universe",
                        lambda prev_date: ["600000.XSHG"], raising=False)
    monkeypatch.setattr(
        strategy,
        "scan_candidates",
        lambda universe, prev_date: [{
            "code": "600000.XSHG", "confirmation_close": 103.0,
            "breakout_level": 100.0, "volume_ratio": 1.8,
            "stop_price": 99.0,
        }],
        raising=False,
    )
    monkeypatch.setattr(strategy, "prepare_exit_reasons",
                        lambda context, prev_date: {}, raising=False)
    monkeypatch.setattr(
        strategy,
        "get_current_data",
        lambda: (_ for _ in ()).throw(AssertionError("current quote read at 09:30")),
        raising=False,
    )

    strategy.prepare_daily_state(context)

    assert strategy.g.prepared_for == date(2026, 7, 20)
    assert [item["code"] for item in strategy.g.candidates] == ["600000.XSHG"]
    assert "get_current_data" not in inspect.getsource(strategy.prepare_daily_state)


def test_limit_up_and_excessive_execution_gap_are_rejected():
    normal = types.SimpleNamespace(
        paused=False, is_st=False, name="正常股份",
        last_price=10.0, high_limit=11.0,
    )
    limit_up = types.SimpleNamespace(
        paused=False, is_st=False, name="正常股份",
        last_price=11.0, high_limit=11.0,
    )
    chased = types.SimpleNamespace(
        paused=False, is_st=False, name="正常股份",
        last_price=10.31, high_limit=11.0,
    )

    assert strategy.execution_quote_is_buyable(normal, confirmation_close=10.0)
    assert not strategy.execution_quote_is_buyable(limit_up, confirmation_close=10.0)
    assert not strategy.execution_quote_is_buyable(chased, confirmation_close=10.0)


def test_sold_code_cannot_reenter_on_same_day():
    assert strategy.can_open_candidate("AAA", {"BBB"}, {"CCC"})
    assert not strategy.can_open_candidate("AAA", {"AAA"}, set())
    assert not strategy.can_open_candidate("AAA", set(), {"AAA"})


def test_execution_requires_a_snapshot_prepared_for_today(monkeypatch):
    strategy.g = types.SimpleNamespace(
        prepared_for=date(2026, 7, 19),
        candidates=[],
        exit_reasons={},
        sold_today=set(),
    )
    context = types.SimpleNamespace(current_dt=datetime(2026, 7, 20, 9, 35))
    quote_reads = []
    monkeypatch.setattr(strategy, "get_current_data",
                        lambda: quote_reads.append(True), raising=False)

    strategy.execute_trades(context)

    assert quote_reads == []


def test_exit_snapshot_uses_previous_close_and_stored_shape_stop(monkeypatch):
    prev_date = date(2026, 7, 17)
    closes = np.full(65, 100.0)
    closes[-1] = 95.0
    calls = []
    strategy.g = types.SimpleNamespace(
        managed_codes={"600000.XSHG"},
        stop_prices={"600000.XSHG": 97.0},
        entry_dates={"600000.XSHG": date(2026, 7, 1)},
    )
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"600000.XSHG": types.SimpleNamespace(total_amount=100)}
        )
    )

    def fake_load(code, requested_date, count):
        calls.append((code, requested_date, count))
        return {"close": closes, "last_date": requested_date}

    monkeypatch.setattr(strategy, "load_daily_bars", fake_load)
    monkeypatch.setattr(strategy, "_holding_days", lambda *args: 5)

    reasons = strategy.prepare_exit_reasons(context, prev_date)

    assert reasons == {"600000.XSHG": "shape_stop"}
    assert calls[0][1] == prev_date


def test_execution_sells_before_buys_and_blocks_same_day_reentry(monkeypatch):
    today = date(2026, 7, 20)
    code = "600000.XSHG"
    quote = types.SimpleNamespace(
        paused=False, is_st=False, name="正常股份",
        last_price=10.0, high_limit=11.0,
    )
    strategy.g = types.SimpleNamespace(
        prepared_for=today,
        market_open=True,
        candidates=[{
            "code": code, "confirmation_close": 10.0,
            "breakout_level": 9.8, "volume_ratio": 1.8,
            "stop_price": 9.5,
        }],
        exit_reasons={code: "shape_stop"},
        sold_today=set(),
        managed_codes={code},
        stop_prices={code: 9.5},
        entry_dates={code: date(2026, 7, 1)},
    )
    context = types.SimpleNamespace(
        current_dt=datetime(2026, 7, 20, 9, 35),
        portfolio=types.SimpleNamespace(
            positions={code: types.SimpleNamespace(total_amount=100)},
            available_cash=100000.0,
            total_value=100000.0,
        ),
    )
    events = []
    monkeypatch.setattr(strategy, "get_current_data", lambda: {code: quote},
                        raising=False)
    monkeypatch.setattr(strategy, "order_target",
                        lambda order_code, amount: events.append(
                            ("sell", order_code, amount)) or object(),
                        raising=False)
    monkeypatch.setattr(strategy, "order",
                        lambda order_code, amount: events.append(
                            ("buy", order_code, amount)) or object(),
                        raising=False)

    strategy.execute_trades(context)

    assert events == [("sell", code, 0)]
    assert code in strategy.g.sold_today


def test_every_daily_history_call_explicitly_ends_at_previous_date():
    source = pathlib.Path(strategy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_price"
    ]

    assert len(calls) == 3
    for call in calls:
        end_date = next(
            (keyword.value for keyword in call.keywords
             if keyword.arg == "end_date"),
            None,
        )
        assert isinstance(end_date, ast.Name)
        assert end_date.id == "prev_date"


def test_platform_lookup_supports_joinquant_index_only_containers():
    class IndexOnly:
        def __getitem__(self, key):
            if key == "AAA":
                return "quote"
            raise KeyError(key)

    assert strategy.mapping_get(IndexOnly(), "AAA") == "quote"
    assert strategy.mapping_get(IndexOnly(), "MISSING") is None


def make_joinquant_batch_frame():
    dates = pd.date_range("2020-01-02", periods=67, freq="B")
    rows = []
    for code in ("000001.XSHE", "600000.XSHG"):
        for index, timestamp in enumerate(dates):
            paused = code == "000001.XSHE" and index == 10
            rows.append({
                "time": timestamp,
                "code": code,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 0.0 if paused else 1000.0,
                "paused": paused,
            })
    return pd.DataFrame(rows)


def test_batch_loader_preserves_shared_axis_and_requests_paused_field(monkeypatch):
    captured = {}

    def fake_get_price(codes, **kwargs):
        captured["codes"] = codes
        captured.update(kwargs)
        return make_joinquant_batch_frame()

    monkeypatch.setattr(strategy, "get_price", fake_get_price, raising=False)
    codes = ["000001.XSHE", "600000.XSHG"]

    histories = strategy._load_daily_bars_batch(
        codes, date(2020, 4, 3))

    assert captured["skip_paused"] is False
    assert "paused" in captured["fields"]
    assert set(histories) == set(codes)


def test_batch_loader_removes_filled_paused_rows_per_security(monkeypatch):
    monkeypatch.setattr(
        strategy,
        "get_price",
        lambda *args, **kwargs: make_joinquant_batch_frame(),
        raising=False,
    )

    histories = strategy._load_daily_bars_batch(
        ["000001.XSHE", "600000.XSHG"], date(2020, 4, 3))

    assert len(histories["000001.XSHE"]["close"]) == 66
    assert np.all(histories["000001.XSHE"]["volume"] > 0)
    assert len(histories["600000.XSHG"]["close"]) == 67


def test_scan_candidates_records_loaded_stale_and_signal_counts(monkeypatch):
    prev_date = date(2020, 4, 3)
    valid = make_valid_bars()
    valid["last_date"] = prev_date
    stale = make_valid_bars()
    stale["last_date"] = date(2020, 4, 2)
    no_signal = make_valid_bars()
    no_signal["volume"][65] = 100.0
    no_signal["last_date"] = prev_date
    histories = {
        "AAA": valid,
        "BBB": stale,
        "CCC": no_signal,
    }
    strategy.g = types.SimpleNamespace()
    monkeypatch.setattr(
        strategy,
        "_load_daily_bars_batch",
        lambda codes, requested_date: {
            code: histories[code] for code in codes
        },
    )

    candidates = strategy.scan_candidates(["AAA", "BBB", "CCC"], prev_date)

    assert [candidate["code"] for candidate in candidates] == ["AAA"]
    assert strategy.g.last_scan_stats == {
        "universe": 3,
        "loaded": 3,
        "stale": 1,
        "evaluated": 2,
        "signals": 1,
    }
