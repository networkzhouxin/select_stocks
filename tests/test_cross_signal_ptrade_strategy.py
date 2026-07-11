# -*- coding: utf-8 -*-
"""Parity and live-safety tests for the PTrade cross-signal strategy."""

from datetime import date, datetime
import importlib.util
from pathlib import Path
import pickle
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.log = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    return module


jq = load_module(
    "cross_signal_joinquant_for_ptrade_parity",
    ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf.py",
)
pt = load_module(
    "cross_signal_ptrade",
    ROOT / "cross_signal_strategy" / "smart_trade_ptrade_cross_signal_etf.py",
)


def make_g(**overrides):
    values = {
        "params": pt.get_default_params(),
        "etf_pool": pt.get_default_etf_pool(),
        "highest_since_buy": {},
        "entry_atr": {},
        "buy_date": {},
        "last_scores": {},
        "sold_today": {},
        "paused_pool_codes": set(),
        "unverified_positions": set(),
        "execution_date": None,
        "deferred_scores": [],
        "deferred_signal_date": None,
        "__last_snapshot": {},
        "__pending_orders": {},
        "__pending_sells": {},
        "__order_state_unknown": False,
        "__is_live": True,
        "__data": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_buy_score(code="513100.SS"):
    return {
        "code": code,
        "buy_allowed": True,
        "buy_score": 70,
        "sell_score": 0,
        "reversal_score": 35,
        "location_score": 17,
        "trend_score": 20,
        "volume_score": 0,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "rsi6": 50,
        "atr": 0.05,
    }


def test_ptrade_business_configuration_matches_frozen_joinquant_mainline():
    assert pt.STRATEGY_VERSION == jq.STRATEGY_VERSION == "cross-v0.3.2"
    assert pt.get_default_params() == jq.get_default_params()
    assert pt.get_default_etf_pool() == [
        "159915.SZ",
        "512100.SS",
        "159928.SZ",
        "513100.SS",
        "513500.SS",
        "513880.SS",
        "513050.SS",
        "518880.SS",
        "159985.SZ",
    ]


def test_before_trading_start_relocks_frozen_business_config_after_restore(monkeypatch):
    today = date(2026, 7, 13)
    stale_params = pt.get_default_params()
    stale_params["buy_threshold"] = 999
    pt.g = make_g(
        params=stale_params,
        etf_pool=["510300.SS"],
        execution_date=today,
        __is_live=False,
    )
    monkeypatch.setattr(pt, "_restore_live_state", lambda: True, raising=False)

    context = types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 8, 30))
    pt.before_trading_start(context, data=None)

    assert pt.g.params == pt.get_default_params()
    assert pt.g.etf_pool == pt.get_default_etf_pool()


def test_explicit_live_state_round_trip_excludes_business_configuration(tmp_path):
    state_path = tmp_path / "cross-signal-state.pkl"
    today = date(2026, 7, 13)
    signal_date = date(2026, 7, 10)
    pt.g = make_g(
        highest_since_buy={"513100.SS": 2.5},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": signal_date},
        last_scores={"513100.SS": make_buy_score()},
        sold_today={"159915.SZ": True},
        paused_pool_codes={"513880.SS"},
        unverified_positions={"159985.SZ"},
        execution_date=today,
        deferred_scores=[make_buy_score()],
        deferred_signal_date=signal_date,
    )

    assert pt._persist_live_state(path=state_path) is True

    pt.g.highest_since_buy = {}
    pt.g.entry_atr = {}
    pt.g.buy_date = {}
    pt.g.last_scores = {}
    pt.g.sold_today = {}
    pt.g.paused_pool_codes = set()
    pt.g.unverified_positions = set()
    pt.g.execution_date = None
    pt.g.deferred_scores = []
    pt.g.deferred_signal_date = None
    pt.g.params = {"buy_threshold": 777}
    pt.g.etf_pool = ["510300.SS"]

    assert pt._restore_live_state(path=state_path) is True
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.entry_atr == {"513100.SS": 0.05}
    assert pt.g.buy_date == {"513100.SS": signal_date}
    assert pt.g.sold_today == {"159915.SZ": True}
    assert pt.g.paused_pool_codes == {"513880.SS"}
    assert pt.g.unverified_positions == {"159985.SZ"}
    assert pt.g.execution_date == today
    assert pt.g.deferred_signal_date == signal_date
    assert pt.g.params == {"buy_threshold": 777}
    assert pt.g.etf_pool == ["510300.SS"]


def test_automatic_live_state_path_is_isolated_by_account_and_trade(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(pt, "get_user_name", lambda: "account-a", raising=False)
    monkeypatch.setattr(pt, "get_trade_name", lambda: "simulation", raising=False)

    simulation_path = pt._live_state_path()
    monkeypatch.setattr(pt, "get_trade_name", lambda: "live", raising=False)
    live_path = pt._live_state_path()

    assert simulation_path != live_path
    assert state_parent(simulation_path) == state_parent(live_path) == str(tmp_path)
    assert "account-a" not in simulation_path


def test_automatic_live_state_path_fails_closed_without_instance_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pt, "get_research_path", lambda: str(tmp_path), raising=False)
    monkeypatch.delattr(pt, "get_user_name", raising=False)
    monkeypatch.delattr(pt, "get_trade_name", raising=False)

    assert pt._live_state_path() is None


def state_parent(path):
    return str(Path(path).parent)


def test_malformed_live_state_is_rejected_without_partial_restore(tmp_path):
    state_path = tmp_path / "malformed-state.pkl"
    payload = {
        "strategy_version": pt.STRATEGY_VERSION,
        "state": {
            "highest_since_buy": {"513100.SS": 9.9},
            "entry_atr": ["not", "a", "mapping"],
        },
    }
    state_path.write_bytes(pickle.dumps(payload))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 2.5},
        entry_atr={"513100.SS": 0.05},
    )

    assert pt._restore_live_state(path=state_path) is False
    assert pt.g.highest_since_buy == {"513100.SS": 2.5}
    assert pt.g.entry_atr == {"513100.SS": 0.05}


def test_live_schedule_wrappers_checkpoint_state(monkeypatch):
    calls = []
    monkeypatch.setattr(pt, "do_trading", lambda context: calls.append("trade"))
    monkeypatch.setattr(pt, "halt_recover", lambda context: calls.append("recover"))
    monkeypatch.setattr(pt, "after_close", lambda context: calls.append("close"))
    monkeypatch.setattr(
        pt, "_persist_live_state", lambda: calls.append("persist") or True,
        raising=False,
    )
    pt.g = make_g()
    context = types.SimpleNamespace()

    pt._do_trading_wrapper(context)
    pt._halt_recover_wrapper(context)
    pt._after_close_wrapper(context)

    assert calls == [
        "trade", "persist",
        "recover", "persist",
        "close", "persist",
    ]


def test_order_and_trade_callbacks_checkpoint_state(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        pt, "_persist_live_state", lambda: persisted.append(True) or True,
        raising=False,
    )
    pt.g = make_g()

    pt.on_order_response(types.SimpleNamespace(), [])
    pt.on_trade_response(types.SimpleNamespace(), [])

    assert len(persisted) == 2


def test_ptrade_scoring_and_stop_math_match_joinquant_mainline():
    snapshot = {
        "rsi6": 61.0,
        "rsi6_cross_rsi12_up": True,
        "rsi6_cross_rsi24_up": True,
        "macd_cross_up": False,
        "kdj_k_cross_up": True,
        "kdj_j_cross_up": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": True,
        "close_far_above_ma20": False,
        "ma5_gt_ma10": True,
        "ma10_gt_ma20": False,
        "ma20_slope_non_negative": True,
        "close_gt_ma60": True,
        "downside_continuation": False,
        "volume_above_vol20_and_up": False,
        "vol5_gt_vol20": True,
    }

    assert pt.score_buy_snapshot(snapshot) == jq.score_buy_snapshot(snapshot)
    assert pt.calc_stop_price(10.0, 0.2, 8.0) == jq.calc_stop_price(10.0, 0.2, 8.0)


def test_ptrade_normalizes_callback_codes_to_universe_format():
    assert pt.normalize_code("513100") == "513100.SS"
    assert pt.normalize_code("159915") == "159915.SZ"
    assert pt.normalize_code("513100.XSHG") == "513100.SS"
    assert pt.normalize_code("159915.XSHE") == "159915.SZ"


def test_live_price_fails_closed_when_snapshot_is_unavailable(monkeypatch):
    pt.g = make_g()
    history_called = []
    monkeypatch.setattr(pt, "get_snapshot", lambda code: {}, raising=False)
    monkeypatch.setattr(
        pt, "get_history", lambda *args, **kwargs: history_called.append(True), raising=False
    )

    assert pt.get_current_price("513100.SS") is None
    assert history_called == []


def test_prev_trade_date_does_not_guess_weekdays_when_apis_fail(monkeypatch):
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35))
    )
    monkeypatch.setattr(
        pt, "get_trade_days", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()),
        raising=False,
    )
    monkeypatch.setattr(
        pt, "get_all_trades_days", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()),
        raising=False,
    )

    assert pt.get_prev_trade_date(context) is None


def test_signal_sell_requires_verified_trading_calendar_for_five_day_hold():
    buy_date = date(2026, 7, 10)
    today = date(2026, 7, 15)

    assert not pt.can_sell_with_verified_calendar(
        buy_date, today, min_hold_days=5, trade_days=None
    )
    assert pt.can_sell_with_verified_calendar(
        buy_date,
        date(2026, 7, 17),
        min_hold_days=5,
        trade_days=[
            date(2026, 7, 10),
            date(2026, 7, 13),
            date(2026, 7, 14),
            date(2026, 7, 15),
            date(2026, 7, 16),
            date(2026, 7, 17),
        ],
    )


def test_daily_signal_loader_uses_pre_adjusted_data_ending_at_t_minus_one(monkeypatch):
    captured = {}

    def fake_get_price(code, **kwargs):
        captured.update({"code": code, **kwargs})
        return pt.pd.DataFrame({
            "open": [1.0, 1.1],
            "close": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "volume": [100, 200],
        })

    monkeypatch.setattr(pt, "get_price", fake_get_price, raising=False)
    frame = pt.get_price_data("513100.SS", date(2021, 12, 30), 120)

    assert len(frame) == 2
    assert captured["end_date"] == "2021-12-30"
    assert captured["frequency"] == "1d"
    assert captured["fq"] == "pre"


def test_sell_submission_keeps_state_until_full_fill(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={"513100.SS": position}))
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
    )
    orders = []
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(pt, "get_sell_limit_price", lambda code, price: 1.0)
    monkeypatch.setattr(
        pt,
        "order_target",
        lambda code, amount, limit_price=None: (
            orders.append((code, amount, limit_price)) or "sell-order-1"
        ),
        raising=False,
    )

    assert pt.execute_sell("513100.SS", context, "test")
    assert orders == [("513100.SS", 0, 1.0)]
    assert "513100.SS" in pt.g.highest_since_buy
    assert pt.g.__pending_sells["513100.SS"]["requested_qty"] == 500
    assert pt.g.__pending_sells["513100.SS"]["order_id"] == "sell-order-1"


def test_sell_submission_failure_does_not_create_guard(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(highest_since_buy={"513100.SS": 1.2})
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(pt, "get_sell_limit_price", lambda code, price: 1.0)
    monkeypatch.setattr(pt, "order_target", lambda *args, **kwargs: None, raising=False)

    assert not pt.execute_sell("513100.SS", context, "test")
    assert pt.g.__pending_sells == {}
    assert pt.g.sold_today == {}


def test_partial_buy_callbacks_accumulate_before_pending_is_cleared():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )
    context = types.SimpleNamespace()

    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 200,
        "business_price": 1.10,
        "order_id": "buy-order-1",
    })
    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 200
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.10)

    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 300,
        "business_price": 1.20,
        "order_id": "buy-order-1",
    })
    assert "513100.SS" not in pt.g.__pending_orders
    assert pt.g.highest_since_buy["513100.SS"] == pytest.approx(1.16)


def test_cancel_trade_push_is_not_counted_as_a_fill():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "business_price": 1.10,
        "order_id": "buy-order-1",
        "real_type": "2",
    })

    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 0
    assert "513100.SS" not in pt.g.highest_since_buy


def test_delayed_callback_for_old_order_does_not_touch_current_guard():
    pt.g = make_g(
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-new",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 500,
        "business_price": 1.15,
        "order_id": "sell-order-old",
        "real_type": "0",
    })

    assert pt.g.__pending_sells["513100.SS"]["filled_qty"] == 0


def test_full_sell_callback_clears_strategy_state():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 0, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 500,
        "business_price": 1.15,
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.highest_since_buy
    assert "513100.SS" not in pt.g.entry_atr
    assert "513100.SS" not in pt.g.buy_date


def test_live_pause_check_fails_closed_when_status_is_unknown(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_stock_status", lambda *args, **kwargs: {}, raising=False)

    assert pt.is_paused("513100.SS") is True


def test_live_pause_check_fails_closed_when_status_value_is_none(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(
        pt,
        "get_stock_status",
        lambda *args, **kwargs: {"513100.SS": None},
        raising=False,
    )

    assert pt.is_paused("513100.SS") is True


def test_live_pause_check_refreshes_status_instead_of_trusting_stale_snapshot(monkeypatch):
    pt.g = make_g(__last_snapshot={"513100.SS": {"trade_status": "HALT"}})
    monkeypatch.setattr(
        pt,
        "get_stock_status",
        lambda *args, **kwargs: {"513100.SS": False},
        raising=False,
    )

    assert pt.is_paused("513100.SS") is False


def test_partial_sell_callbacks_keep_state_until_cumulative_full_fill():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        last_scores={"513100.SS": {"buy_score": 60}},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 0, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 200,
        "business_price": 1.15,
        "order_id": "sell-order-1",
    })
    assert pt.g.__pending_sells["513100.SS"]["filled_qty"] == 200
    assert "513100.SS" in pt.g.highest_since_buy

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "business_amount": 300,
        "business_price": 1.14,
        "order_id": "sell-order-1",
    })
    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.highest_since_buy


def test_rejected_sell_releases_retry_guard_without_clearing_position_state():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 0, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_order_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "status": "9",
        "business_amount": 0,
        "error_info": "rejected",
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.sold_today
    assert "513100.SS" in pt.g.highest_since_buy


def test_partial_cancelled_sell_keeps_risk_state_for_remaining_position():
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": date(2026, 6, 1)},
        sold_today={"513100.SS": True},
        __pending_sells={
            "513100.SS": {
                "requested_qty": 500, "filled_qty": 200, "order_id": "sell-order-1"
            }
        },
    )

    pt.on_order_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "2",
        "status": "5",
        "business_amount": 200,
        "error_info": "partial cancel",
        "order_id": "sell-order-1",
    })

    assert "513100.SS" not in pt.g.__pending_sells
    assert "513100.SS" not in pt.g.sold_today
    assert "513100.SS" in pt.g.highest_since_buy
    assert "513100.SS" in pt.g.entry_atr


def test_before_trading_clears_expired_day_order_guards(monkeypatch):
    pt.g = make_g(
        __pending_orders={"513100.SS": {"requested_qty": 100}},
        __pending_sells={"159915.SZ": {"requested_qty": 100}},
        sold_today={"159915.SZ": True},
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)

    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    pt.before_trading_start(context, data={})

    assert pt.g.__pending_orders == {}
    assert pt.g.__pending_sells == {}
    assert pt.g.sold_today == {}


def test_before_trading_rebuilds_guards_from_broker_open_orders(monkeypatch):
    pt.g = make_g(last_scores={"513100.SS": {"atr": 0.05}})
    open_orders = [
        types.SimpleNamespace(
            id="buy-order-open", symbol="513100.XSHG", amount=500, filled=200
        ),
        types.SimpleNamespace(
            id="sell-order-open", symbol="159915.XSHE", amount=-300, filled=100
        ),
    ]
    monkeypatch.setattr(pt, "get_open_orders", lambda: open_orders, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )

    pt.before_trading_start(context, data={})

    assert pt.g.__pending_orders["513100.SS"]["requested_qty"] == 500
    assert pt.g.__pending_orders["513100.SS"]["filled_qty"] == 200
    assert pt.g.__pending_orders["513100.SS"]["order_id"] == "buy-order-open"
    assert pt.g.__pending_sells["159915.SZ"]["requested_qty"] == 300
    assert pt.g.__pending_sells["159915.SZ"]["filled_qty"] == 100
    assert pt.g.__pending_sells["159915.SZ"]["order_id"] == "sell-order-open"
    assert pt.g.sold_today["159915.SZ"] is True
    assert pt.g.__order_state_unknown is False


def test_before_trading_fails_closed_on_duplicate_symbol_open_orders(monkeypatch):
    pt.g = make_g()
    open_orders = [
        types.SimpleNamespace(
            id="buy-order-1", symbol="513100.XSHG", amount=500, filled=0
        ),
        types.SimpleNamespace(
            id="buy-order-2", symbol="513100.XSHG", amount=500, filled=0
        ),
    ]
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: open_orders, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.__order_state_unknown is True
    assert pt.g.__pending_orders == {}
    assert pt.g.__pending_sells == {}


@pytest.mark.parametrize(
    ("amount", "filled"),
    [(float("nan"), 0), (500, float("nan")), (500, float("inf")), (500, 600)],
)
def test_before_trading_fails_closed_on_invalid_open_order_quantities(
    monkeypatch, amount, filled
):
    pt.g = make_g()
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: [
            types.SimpleNamespace(
                id="buy-order-1",
                symbol="513100.XSHG",
                amount=amount,
                filled=filled,
            )
        ],
        raising=False,
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.__order_state_unknown is True
    assert pt.g.__pending_orders == {}


def test_recovered_partial_buy_without_synced_cost_remains_unverified(monkeypatch):
    pt.g = make_g(last_scores={"513100.SS": {"atr": 0.05}})
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: [
            types.SimpleNamespace(
                id="buy-order-1",
                symbol="513100.XSHG",
                amount=500,
                filled=200,
            )
        ],
        raising=False,
    )

    pt.before_trading_start(context, data={})
    pt.on_trade_response(context, {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 300,
        "business_price": 1.20,
        "order_id": "buy-order-1",
        "real_type": "0",
    })

    assert "513100.SS" not in pt.g.__pending_orders
    assert "513100.SS" in pt.g.unverified_positions
    assert "513100.SS" not in pt.g.highest_since_buy


def test_nonfinite_buy_fill_price_keeps_position_unverified():
    pt.g = make_g(
        __pending_orders={
            "513100.SS": {
                "requested_qty": 500,
                "filled_qty": 0,
                "filled_value": 0.0,
                "fill_value_complete": True,
                "atr": 0.05,
                "buy_date": date(2026, 7, 10),
                "order_id": "buy-order-1",
            }
        }
    )

    pt.on_trade_response(types.SimpleNamespace(), {
        "stock_code": "513100",
        "entrust_bs": "1",
        "business_amount": 500,
        "business_price": float("nan"),
        "order_id": "buy-order-1",
        "real_type": "0",
    })

    assert "513100.SS" not in pt.g.__pending_orders
    assert "513100.SS" in pt.g.unverified_positions
    assert "513100.SS" not in pt.g.highest_since_buy


def test_before_trading_marks_order_state_unknown_when_reconciliation_fails(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(
        pt,
        "get_open_orders",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        raising=False,
    )
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(types.SimpleNamespace(), data={})

    assert pt.g.__order_state_unknown is True


def test_before_trading_fails_closed_when_open_order_response_is_none(monkeypatch):
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_open_orders", lambda: None, raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(types.SimpleNamespace(), data={})

    assert pt.g.__order_state_unknown is True


def test_trading_aborts_when_broker_order_state_is_unknown(monkeypatch):
    pt.g = make_g(__order_state_unknown=True)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35))
    )
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(
        pt,
        "check_atr_stops",
        lambda context: (_ for _ in ()).throw(AssertionError("must not evaluate")),
    )

    pt.do_trading(context)


def test_buy_execution_waits_for_submitted_sells_to_finish(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(
            positions={"159915.SZ": position}, portfolio_value=20000, cash=500
        )
    )
    pt.g = make_g(
        __pending_sells={"159915.SZ": {"requested_qty": 500, "filled_qty": 0}}
    )
    orders = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: orders.append((args, kwargs)) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 0
    assert orders == []


def test_buy_execution_uses_confirmed_cash_and_creates_fill_guard(monkeypatch):
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000)
    )
    pt.g = make_g()
    orders = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(
        pt,
        "order",
        lambda *args, **kwargs: orders.append((args, kwargs)) or "buy-order-1",
        raising=False,
    )

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 1
    assert orders == [(('513100.SS', 3100), {'limit_price': 2.0})]
    assert pt.g.__pending_orders["513100.SS"]["requested_qty"] == 3100
    assert pt.g.__pending_orders["513100.SS"]["order_id"] == "buy-order-1"


def test_buy_submission_failure_does_not_create_guard(monkeypatch):
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={}, portfolio_value=20000, cash=20000)
    )
    pt.g = make_g()
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 2.0)
    monkeypatch.setattr(pt, "order", lambda *args, **kwargs: None, raising=False)

    assert pt.execute_buy_candidates(
        context, [make_buy_score()], date(2026, 7, 13)
    ) == 0
    assert pt.g.__pending_orders == {}


def test_halt_recovery_only_merges_resumed_scores_and_executes_deferred_buys(monkeypatch):
    pt.g = make_g(
        paused_pool_codes={"513100.SS"},
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=[make_buy_score("159985.SZ")],
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    scored = []
    executed = []
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(
        pt,
        "calc_cross_signal_score",
        lambda code, end_date, return_reason=False: (
            scored.append((code, end_date)) or make_buy_score(code),
            None,
        ),
    )
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda context, scores, today: executed.append([s["code"] for s in scores]) or 0,
    )
    monkeypatch.setattr(
        pt,
        "do_trading",
        lambda context: (_ for _ in ()).throw(AssertionError("no second full pass")),
    )

    pt.halt_recover(context)

    assert scored == [("513100.SS", date(2026, 7, 10))]
    assert executed == [["159985.SZ", "513100.SS"]]
    assert pt.g.paused_pool_codes == set()


def test_new_trading_day_clears_date_scoped_deferred_state(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 10),
        deferred_signal_date=date(2026, 7, 9),
        deferred_scores=[make_buy_score()],
        paused_pool_codes={"513100.SS"},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.execution_date == date(2026, 7, 13)
    assert pt.g.deferred_signal_date is None
    assert pt.g.deferred_scores == []
    assert pt.g.paused_pool_codes == set()


def test_intraday_restart_preserves_same_day_deferred_and_halt_state(monkeypatch):
    deferred = [make_buy_score()]
    pt.g = make_g(
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=deferred,
        paused_pool_codes={"513100.SS"},
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 40)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    monkeypatch.setattr(pt, "get_open_orders", lambda: [], raising=False)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: None)

    pt.before_trading_start(context, data={})

    assert pt.g.deferred_scores == deferred
    assert pt.g.deferred_signal_date == date(2026, 7, 10)
    assert pt.g.paused_pool_codes == {"513100.SS"}


def test_halt_recovery_rejects_stale_deferred_scores(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 10),
        deferred_signal_date=date(2026, 7, 9),
        deferred_scores=[make_buy_score()],
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35))
    )
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stale scores must not execute")
        ),
    )

    pt.halt_recover(context)


def test_halt_recovery_reconciles_open_orders_before_deferred_buy(monkeypatch):
    pt.g = make_g(
        execution_date=date(2026, 7, 13),
        deferred_signal_date=date(2026, 7, 10),
        deferred_scores=[make_buy_score()],
        __pending_sells={
            "159915.SZ": {
                "requested_qty": 500,
                "filled_qty": 0,
                "order_id": "sell-order-1",
            }
        },
    )
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 10, 35)),
        portfolio=types.SimpleNamespace(positions={}),
    )
    calls = []
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))

    def reconcile(context):
        calls.append("reconcile")
        pt.g.__pending_sells = {}
        return True

    monkeypatch.setattr(pt, "_reconcile_open_orders", reconcile)
    monkeypatch.setattr(pt, "recover_live_state", lambda context: calls.append("recover"))
    monkeypatch.setattr(
        pt,
        "execute_buy_candidates",
        lambda context, scores, today: calls.append("buy") or 0,
    )

    pt.halt_recover(context)

    assert calls == ["reconcile", "recover", "buy"]


def test_live_recovery_does_not_invent_missing_entry_risk_state(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 0)),
        portfolio=types.SimpleNamespace(positions={"513100.SS": position}),
    )
    pt.g = make_g()
    monkeypatch.setattr(pt, "get_prev_trade_date", lambda context: date(2026, 7, 10))
    monkeypatch.setattr(pt, "get_current_price", lambda code: 1.1)
    monkeypatch.setattr(
        pt,
        "get_price_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not synthesize entry ATR")
        ),
    )

    pt.recover_live_state(context)

    assert pt.g.unverified_positions == {"513100.SS"}
    assert "513100.SS" not in pt.g.buy_date
    assert "513100.SS" not in pt.g.highest_since_buy
    assert "513100.SS" not in pt.g.entry_atr


def test_unverified_position_is_excluded_from_atr_stop_execution(monkeypatch):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=0.8)
    context = types.SimpleNamespace(
        blotter=types.SimpleNamespace(current_dt=datetime(2026, 7, 13, 9, 35)),
        portfolio=types.SimpleNamespace(positions={"513100.SS": position}),
    )
    pt.g = make_g(
        highest_since_buy={"513100.SS": 1.2},
        entry_atr={"513100.SS": 0.05},
        buy_date={"513100.SS": None},
        unverified_positions={"513100.SS"},
    )
    monkeypatch.setattr(pt, "is_paused", lambda code: False)
    monkeypatch.setattr(pt, "get_current_price", lambda code: 0.8)

    assert pt.check_atr_stops(context) == []


@pytest.mark.parametrize(
    ("atr", "highest"),
    [("not-a-number", 1.2), (0.05, float("inf")), (float("nan"), 1.2)],
)
def test_live_recovery_rejects_malformed_or_nonfinite_risk_state(atr, highest):
    position = types.SimpleNamespace(amount=500, cost_basis=1.0, last_sale_price=1.1)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={"513100.SS": position})
    )
    pt.g = make_g(
        buy_date={"513100.SS": date(2026, 6, 1)},
        highest_since_buy={"513100.SS": highest},
        entry_atr={"513100.SS": atr},
    )

    pt.recover_live_state(context)

    assert pt.g.unverified_positions == {"513100.SS"}


def test_initialize_live_schedules_only_cross_signal_tasks(monkeypatch):
    scheduled = []
    platform_parameters = []
    pt.g = types.SimpleNamespace()
    monkeypatch.setattr(pt, "set_benchmark", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pt, "set_commission", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pt, "set_slippage", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pt, "set_universe", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        pt,
        "set_parameters",
        lambda **kwargs: platform_parameters.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(pt, "is_trade", lambda: True, raising=False)
    monkeypatch.setattr(
        pt,
        "run_daily",
        lambda context, func, time: scheduled.append((func.__name__, time)),
        raising=False,
    )

    pt.initialize(types.SimpleNamespace())

    assert scheduled == [
        ("_do_trading_wrapper", "09:35"),
        ("_halt_recover_wrapper", "10:35"),
        ("_after_close_wrapper", "15:30"),
    ]
    assert pt.g.params == jq.get_default_params()
    assert not hasattr(pt.g, "base_weights")
    assert platform_parameters == [{
        "receive_cancel_response": "1",
        "not_restart_trade": "0",
        "server_restart_not_do_before": "0",
    }]


def test_ptrade_deployment_notes_pin_frozen_version_and_live_schedule():
    notes = (
        ROOT / "cross_signal_strategy" / "docs" / "ptrade_deployment.md"
    ).read_text(encoding="utf-8")

    assert "cross-v0.3.2" in notes
    assert "09:35" in notes
    assert "10:35" in notes
    assert "15:30" in notes
    assert "JoinQuant" in notes
    assert "PTrade" in notes
    assert "configuration lock" in notes
    assert "explicit state checkpoint" in notes
