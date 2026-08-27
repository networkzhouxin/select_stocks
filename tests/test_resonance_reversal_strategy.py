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
        frame, pd.Timestamp("2021-01-11"), pd.Timestamp("2021-01-12"),
    )

    assert book["active"]["RSI"]["event_date"] == pd.Timestamp("2021-01-08")
    assert book["active"]["RSI"]["expires_date"] == pd.Timestamp("2021-01-11")


def event_book_for_directions(boll, rsi, kdj, event_date):
    active = {}
    for indicator, direction_name in (("BOLL", boll), ("RSI", rsi), ("KDJ", kdj)):
        if direction_name == "NEUTRAL":
            continue
        active[indicator] = make_event(
            indicator,
            strategy.TurnDirection[direction_name],
            event_date,
            "2021-01-06",
        )
    return {"active": active, "invalidated": []}


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
        first["resonance_id"]: "2021-01-06",
    }


def test_processed_id_is_pruned_only_after_expiry():
    processed = {"expired": "2021-01-05", "still_active": "2021-01-06"}

    assert strategy.prune_processed_resonance_ids(processed, "2021-01-06") == {
        "still_active": "2021-01-06",
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
        state_date="2021-01-05",
        sold_today={"510300.XSHG"},
        daily_attempted_buys={"159915.XSHE"},
        processed_resonance_ids={"expired": "2021-01-05", "active": "2021-01-06"},
    )
    monkeypatch.setattr(strategy, "g", runtime, raising=False)

    strategy.reset_daily_state("2021-01-05", "2021-01-06")

    assert runtime.sold_today == {"510300.XSHG"}
    assert runtime.daily_attempted_buys == {"159915.XSHE"}
    assert runtime.processed_resonance_ids == {"active": "2021-01-06"}

    strategy.reset_daily_state("2021-01-06", "2021-01-06")

    assert runtime.state_date == "2021-01-06"
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
        ("BUY", 0, 0, 100, "TRADEABLE", 100, 100, "PARTIAL"),
        ("BUY", 0, 0, 100, "TRADEABLE", 100, 0, "NOT_FILLED"),
        ("SELL", 100, 0, 0, "TRADEABLE", -100, -100, "FILLED"),
        ("SELL", 100, 40, 0, "TRADEABLE", -100, -60, "PARTIAL"),
        ("SELL", 100, 100, 0, "TRADEABLE", -100, -100, "PARTIAL"),
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
        "end_date": "2021-01-05",
        "count": 120,
        "frequency": "daily",
        "fields": ["open", "high", "low", "close", "volume"],
        "skip_paused": True,
        "fq": "pre",
        "panel": False,
    })]


def test_signal_loader_propagates_future_data_error(monkeypatch):
    class FutureDataError(RuntimeError):
        pass

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
    signal_date = frame.index[-1]
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
        signal_date + pd.offsets.BDay(1),
    )

    assert snapshot["valid"] is True
    assert snapshot["signal_date"] == signal_date
    assert snapshot["close"] == pytest.approx(20.0)
    assert set(snapshot["trade_values"]) == set(strategy.TRADE_INDICATOR_COLUMNS)
    assert set(snapshot["observation_values"]) == set(strategy.OBSERVATION_COLUMNS)
    assert captured[0][1:] == (signal_date, signal_date + pd.offsets.BDay(1))


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


def test_build_signal_snapshots_uses_trading_calendar_for_next_session(
        monkeypatch):
    calendar_calls = []
    snapshot_calls = []
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kw: calendar_calls.append(kw) or [
            pd.Timestamp("2021-01-08"), pd.Timestamp("2021-01-11"),
        ],
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
        pd.Timestamp("2021-01-08"), strategy.get_default_params(),
    )

    assert calendar_calls == [{
        "start_date": pd.Timestamp("2021-01-08"), "count": 2,
    }]
    assert list(snapshots) == EXPECTED_POOL
    assert snapshot_calls == [
        (code, pd.Timestamp("2021-01-08"), pd.Timestamp("2021-01-11"))
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
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
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
