from jqdata import *

import hashlib
import json
import math
from datetime import timedelta
from enum import Enum


STRATEGY_VERSION = "resonance-v0.1.0"
DEPLOYMENT_BUILD_ID = "20260827.1"
BENCHMARK = "000300.XSHG"


class TurnDirection(Enum):
    BUY_TURN = "BUY_TURN"
    SELL_TURN = "SELL_TURN"
    NEUTRAL = "NEUTRAL"


OPPOSITE = {
    TurnDirection.BUY_TURN: TurnDirection.SELL_TURN,
    TurnDirection.SELL_TURN: TurnDirection.BUY_TURN,
}


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExitReason(Enum):
    ATR_EXIT = "ATR_EXIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"


class Tradability(Enum):
    TRADEABLE = "TRADEABLE"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


class OrderOutcome(Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    NOT_FILLED = "NOT_FILLED"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


EXIT_PRIORITY = {
    ExitReason.SIGNAL_EXIT: 1,
    ExitReason.ATR_EXIT: 2,
}


def classify_order_outcome(side, before_amount, after_amount, target_amount,
                           tradability, order):
    if tradability is Tradability.PAUSED:
        return OrderOutcome.PAUSED
    if tradability is Tradability.UNKNOWN:
        return OrderOutcome.UNKNOWN
    if side is OrderSide.SELL and after_amount == 0:
        return OrderOutcome.FILLED
    if (side is OrderSide.BUY and target_amount is not None
            and after_amount >= target_amount):
        return OrderOutcome.FILLED
    filled = (
        abs(getattr(order, "filled", 0) or 0)
        if order is not None else 0
    )
    if after_amount != before_amount or filled > 0:
        return OrderOutcome.PARTIAL
    return OrderOutcome.NOT_FILLED


def get_default_etf_pool():
    return [
        "510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE",
        "510880.XSHG", "513100.XSHG", "513500.XSHG", "159920.XSHE",
        "513880.XSHG", "513050.XSHG", "518880.XSHG", "159985.XSHE",
    ]


def get_default_params():
    return {
        "lookback_days": 120,
        "max_holdings": 3,
        "target_exposure": 0.95,
        "resonance_window": 2,
        "rsi_period": 14,
        "observation_rsi_periods": (6, 12, 24),
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "kdj": (9, 3, 3),
        "kdj_low": 20.0,
        "kdj_high": 80.0,
        "j_low": 0.0,
        "j_high": 100.0,
        "boll": (20, 2.0),
        "atr_period": 14,
        "atr_multiplier": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
    }


def business_config_fingerprint(params=None, etf_pool=None):
    payload = {
        "params": params or get_default_params(),
        "etf_pool": etf_pool or get_default_etf_pool(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _value_fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _json_ready(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _emit_structured_log(event, payload):
    logger = globals().get("log")
    if logger is None:
        return
    body = {"event": event}
    body.update(payload)
    try:
        logger.info(json.dumps(
            _json_ready(body), ensure_ascii=False, sort_keys=True, default=str,
            allow_nan=False,
        ))
    except Exception:
        return


def _runtime_params_and_pool():
    runtime = globals().get("g")
    params = getattr(runtime, "params", get_default_params())
    etf_pool = getattr(runtime, "etf_pool", get_default_etf_pool())
    return params, etf_pool


def log_signal_snapshot(snapshot):
    params, etf_pool = _runtime_params_and_pool()
    event_book = snapshot.get("event_book") or empty_event_book()
    _emit_structured_log("signal_snapshot", {
        "version": STRATEGY_VERSION,
        "build": DEPLOYMENT_BUILD_ID,
        "parameter_fingerprint": _value_fingerprint(params),
        "pool_fingerprint": _value_fingerprint(etf_pool),
        "code": snapshot.get("code"),
        "decision_date": snapshot.get("decision_date"),
        "signal_date": snapshot.get("signal_date"),
        "valid": snapshot.get("valid"),
        "reason": snapshot.get("reason"),
        "trade_values": dict(snapshot.get("trade_values") or {}),
        "observation_values": dict(snapshot.get("observation_values") or {}),
        "kdj_cross": snapshot.get("kdj_cross", "NONE"),
        "active_events": dict(event_book.get("active") or {}),
        "invalidated_events": list(event_book.get("invalidated") or []),
    })


def log_resonance_decision(decision, accepted, reason):
    decision = dict(decision or {})
    runtime = globals().get("g")
    _emit_structured_log("resonance_decision", {
        "code": decision.get("code"),
        "direction": decision.get("direction"),
        "decision_date": getattr(runtime, "state_date", None),
        "signal_date": decision.get("signal_date"),
        "accepted": bool(accepted),
        "reason": reason,
        "supporters": list(decision.get("supporters") or []),
        "support_count": decision.get("support_count", 0),
        "boll_age": decision.get("boll_age"),
        "resonance_id": decision.get("resonance_id"),
        "expires_date": decision.get("expires_date"),
    })


def log_order_transition(code, side, outcome, before_amount, after_amount,
                         requested_target, pending_exit):
    _emit_structured_log("order_transition", {
        "code": code,
        "side": side,
        "outcome": outcome,
        "before_amount": before_amount,
        "after_amount": after_amount,
        "requested_target": requested_target,
        "pending_exit": dict(pending_exit) if pending_exit is not None else None,
    })


def log_portfolio_summary(context):
    positions = {
        code: get_actual_amount(context, code)
        for code in context.portfolio.positions
        if get_actual_amount(context, code) > 0
    }
    anchors = {
        code: state.get("highest_close_anchor")
        for code, state in g.position_states.items()
        if code in positions
    }
    _emit_structured_log("portfolio_summary", {
        "closing_date": context.current_dt.date(),
        "total_value": context.portfolio.total_value,
        "available_cash": context.portfolio.available_cash,
        "positions": positions,
        "highest_close_anchors": anchors,
    })


def make_observation_event(resonance_id, code, event_date, event_close,
                           due_dates):
    return {
        "resonance_id": resonance_id,
        "code": code,
        "event_date": event_date,
        "event_close": float(event_close),
        "due_dates": dict(due_dates),
        "outcomes": {},
    }


def due_observation_horizons(record, closing_date):
    return sorted(
        horizon
        for horizon, due_date in record["due_dates"].items()
        if due_date <= closing_date and horizon not in record["outcomes"]
    )


def _calendar_date(value):
    if hasattr(value, "date"):
        return value.date()
    return value


def get_following_trade_days(anchor_date, required_count):
    if required_count <= 0:
        return []
    anchor = pd.Timestamp(anchor_date)
    start_date = anchor + timedelta(days=1)
    span_days = max(14, required_count * 3)
    for _ in range(9):
        end_date = anchor + timedelta(days=span_days)
        trade_days = get_trade_days(
            start_date=start_date, end_date=end_date,
        )
        following = [
            day for day in trade_days
            if pd.Timestamp(day) > anchor
        ]
        if len(following) >= required_count:
            return following[:required_count]
        span_days *= 2
    raise ValueError(
        "unable to resolve %s following trading sessions after %s"
        % (required_count, anchor_date)
    )


def register_observation_event(decision, event_date, event_close):
    if decision is None or not is_finite_positive(event_close):
        return
    resonance_id = decision["resonance_id"]
    if resonance_id in g.observation_events:
        return
    trade_days = get_following_trade_days(event_date, 5)
    due_dates = {
        horizon: _calendar_date(trade_days[horizon - 1])
        for horizon in (1, 3, 5)
    }
    g.observation_events[resonance_id] = make_observation_event(
        resonance_id, decision["code"], _calendar_date(event_date),
        event_close, due_dates,
    )


def try_register_observation_event(decision, event_date, event_close):
    try:
        register_observation_event(decision, event_date, event_close)
    except Exception as error:
        _emit_structured_log("observation_registration", {
            "resonance_id": (
                decision.get("resonance_id") if decision is not None else None
            ),
            "code": decision.get("code") if decision is not None else None,
            "reason": "OBSERVATION_REGISTRATION_FAILED",
            "error_type": type(error).__name__,
        })
        return False
    return True


def record_due_observation_outcomes(context, current_data):
    closing_date = context.current_dt.date()
    for resonance_id, record in list(g.observation_events.items()):
        due_horizons = due_observation_horizons(record, closing_date)
        if not due_horizons:
            continue
        for horizon in due_horizons:
            due_date = record["due_dates"][horizon]
            if due_date < closing_date:
                outcome = {
                    "status": "HORIZON_MISSED",
                    "closing_date": due_date,
                    "closing_price": None,
                    "return": None,
                }
            else:
                closing_price = get_execution_price(
                    current_data, record["code"],
                )
                if closing_price is None:
                    outcome = {
                        "status": "PRICE_UNAVAILABLE",
                        "closing_date": due_date,
                        "closing_price": None,
                        "return": None,
                    }
                else:
                    outcome = {
                        "status": "RECORDED",
                        "closing_date": due_date,
                        "closing_price": closing_price,
                        "return": closing_price / record["event_close"] - 1.0,
                    }
            record["outcomes"][horizon] = outcome
            _emit_structured_log("observation_outcome", {
                "resonance_id": resonance_id,
                "code": record["code"],
                "event_date": record["event_date"],
                "horizon": horizon,
                "outcome": outcome,
            })
        if all(
                horizon in record["outcomes"]
                for horizon in record["due_dates"]):
            g.observation_events.pop(resonance_id, None)


def ensure_runtime_state():
    if not hasattr(g, "params"):
        g.params = get_default_params()
    if not hasattr(g, "etf_pool"):
        g.etf_pool = get_default_etf_pool()
    if not hasattr(g, "position_states"):
        g.position_states = {}
    if not hasattr(g, "processed_resonance_ids"):
        g.processed_resonance_ids = {}
    if not hasattr(g, "observation_events"):
        g.observation_events = {}
    if not hasattr(g, "sold_today"):
        g.sold_today = set()
    if not hasattr(g, "daily_attempted_buys"):
        g.daily_attempted_buys = set()
    if not hasattr(g, "daily_retried_exits"):
        g.daily_retried_exits = set()


def do_trading(context):
    ensure_runtime_state()
    decision_date = context.current_dt.date()
    signal_date = context.previous_date
    reset_daily_state(decision_date, signal_date)
    current_data = get_current_data()
    retry_pending_exits(context, current_data)
    run_atr_exits(context, current_data)
    snapshots = build_signal_snapshots(signal_date, g.params)
    held_codes = set(get_actual_positions(context))
    for snapshot in snapshots.values():
        event_book = snapshot.get("event_book") or {}
        if (snapshot.get("code") in held_codes
                or event_book.get("active")
                or event_book.get("invalidated")
                or snapshot.get("kdj_cross", "NONE") != "NONE"):
            log_signal_snapshot(dict(snapshot, decision_date=decision_date))
    run_signal_exits(context, current_data, snapshots)
    run_signal_buys(context, current_data, snapshots)


def after_close(context):
    ensure_runtime_state()
    current_data = get_current_data()
    for code, state in list(g.position_states.items()):
        actual_amount = get_actual_amount(context, code)
        if actual_amount == 0:
            clear_position_state_if_flat(code, actual_amount)
            continue
        closing_price = get_execution_price(current_data, code)
        update_highest_close_anchor(state, closing_price)
    record_due_observation_outcomes(context, current_data)
    log_portfolio_summary(context)


def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark(BENCHMARK)
    set_slippage(PriceRelatedSlippage(0.001), type="fund")
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5,
    ), type="fund")
    run_daily(do_trading, time="09:35", reference_security=BENCHMARK)
    run_daily(after_close, time="15:30", reference_security=BENCHMARK)
    ensure_runtime_state()
    _emit_structured_log("strategy_initialized", {
        "version": STRATEGY_VERSION,
        "build": DEPLOYMENT_BUILD_ID,
        "parameter_fingerprint": _value_fingerprint(g.params),
        "pool_fingerprint": _value_fingerprint(g.etf_pool),
        "etf_pool": list(g.etf_pool),
    })


import numpy as np
import pandas as pd


def is_finite_positive(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0)


def load_signal_price_frame(code, prev_date, lookback_days):
    return get_price(
        code,
        end_date=prev_date,
        count=lookback_days,
        frequency="daily",
        fields=["open", "high", "low", "close", "volume"],
        skip_paused=True,
        fq="pre",
        panel=False,
    )


def get_next_trade_date(signal_date):
    return get_following_trade_days(signal_date, 1)[0]


def build_signal_snapshot(code, prev_date, params, next_trade_date):
    price_frame = load_signal_price_frame(
        code, prev_date, params["lookback_days"],
    )
    if price_frame is None or len(price_frame) < params["lookback_days"]:
        return {"code": code, "valid": False, "reason": "INSUFFICIENT_DATA"}

    indicators = build_indicator_frame(price_frame, params)
    latest = indicators.iloc[-1]
    previous = indicators.iloc[-2]
    required = list(TRADE_INDICATOR_COLUMNS)
    signal_required = [name for name in required if name != "atr14"]
    if latest[signal_required].isna().any():
        return {
            "code": code,
            "valid": False,
            "reason": "INVALID_TRADE_INDICATORS",
        }

    entry_atr = (
        float(latest["atr14"])
        if is_finite_positive(latest["atr14"])
        else None
    )

    event_book = collect_latest_events(
        indicators, prev_date, next_trade_date,
    )
    return {
        "code": code,
        "valid": True,
        "signal_date": prev_date,
        "close": float(latest["close"]),
        "entry_atr": entry_atr,
        "event_book": event_book,
        "trade_values": latest[required].to_dict(),
        "observation_values": latest[list(OBSERVATION_COLUMNS)].to_dict(),
        "kdj_cross": detect_kdj_formal_cross(previous, latest),
    }


def build_signal_snapshots(prev_date, params):
    next_trade_date = get_next_trade_date(prev_date)
    snapshots = {}
    for code in get_default_etf_pool():
        snapshots[code] = build_signal_snapshot(
            code, prev_date, params, next_trade_date,
        )
    return snapshots


def calc_buy_target_value(total_value, available_cash, params):
    standard_target = (
        total_value * params["target_exposure"] / params["max_holdings"]
    )
    cash_reserve = total_value * (1.0 - params["target_exposure"])
    return min(standard_target, max(0.0, available_cash - cash_reserve))


def calc_stop_state(highest_close_anchor, entry_atr, params):
    if (not is_finite_positive(highest_close_anchor)
            or not is_finite_positive(entry_atr)):
        return None
    raw_pct = params["atr_multiplier"] * entry_atr / highest_close_anchor
    stop_pct = min(params["stop_cap"], max(params["stop_floor"], raw_pct))
    return {
        "raw_pct": raw_pct,
        "stop_pct": stop_pct,
        "stop_price": highest_close_anchor * (1.0 - stop_pct),
    }


def get_actual_amount(context, code):
    position = context.portfolio.positions.get(code)
    if position is None:
        return 0
    return max(0, int(getattr(position, "total_amount", 0) or 0))


def get_actual_positions(context):
    return {
        code: position
        for code, position in context.portfolio.positions.items()
        if get_actual_amount(context, code) > 0
    }


def get_tradability(current_data, code):
    record = current_data.get(code) if current_data is not None else None
    if record is None:
        return Tradability.UNKNOWN
    paused = getattr(record, "paused", None)
    if paused is True:
        return Tradability.PAUSED
    if paused is False:
        return Tradability.TRADEABLE
    return Tradability.UNKNOWN


def get_execution_price(current_data, code):
    record = current_data.get(code) if current_data is not None else None
    if record is None:
        return None
    price = getattr(record, "last_price", None)
    if not is_finite_positive(price):
        return None
    return float(price)


def make_position_state(buy_date, entry_atr, entry_price):
    return {
        "buy_date": buy_date,
        "entry_atr": float(entry_atr),
        "highest_close_anchor": float(entry_price),
        "pending_exit": None,
    }


def set_pending_exit(position_state, reason, created_date, trigger_value,
                     remaining_amount):
    existing = position_state.get("pending_exit")
    if (existing is not None
            and EXIT_PRIORITY[existing["reason"]] > EXIT_PRIORITY[reason]):
        return existing
    position_state["pending_exit"] = {
        "created_date": created_date,
        "reason": reason,
        "trigger_value": trigger_value,
        "remaining_amount": remaining_amount,
    }
    return position_state["pending_exit"]


def sync_buy_state_after_order(code, outcome, before_amount, after_amount,
                               decision_date, entry_atr, entry_price):
    g.daily_attempted_buys.add(code)
    if after_amount > before_amount:
        g.position_states[code] = make_position_state(
            decision_date, entry_atr, entry_price,
        )
    return outcome


def sync_sell_state_after_order(code, outcome, reason, decision_date,
                                trigger_value, actual_amount):
    state = g.position_states.get(code)
    if actual_amount == 0:
        g.position_states.pop(code, None)
        g.sold_today.add(code)
        return outcome
    if state is not None:
        set_pending_exit(
            state, reason, decision_date, trigger_value, actual_amount,
        )
    return outcome


def submit_buy(context, code, snapshot, decision):
    current_data = get_current_data()
    tradability = get_tradability(current_data, code)
    before_amount = get_actual_amount(context, code)
    execution_price = get_execution_price(current_data, code)
    order = None
    target_value = None
    target_amount = None

    if (tradability is Tradability.TRADEABLE
            and execution_price is not None
            and is_finite_positive(snapshot.get("entry_atr"))):
        target_value = calc_buy_target_value(
            context.portfolio.total_value,
            context.portfolio.available_cash,
            g.params,
        )
        if target_value > 0:
            order = order_target_value(code, target_value)
            ordered_amount = getattr(order, "amount", 0) if order is not None else 0
            if ordered_amount and ordered_amount > 0:
                target_amount = before_amount + int(ordered_amount)

    after_amount = get_actual_amount(context, code)
    outcome = classify_order_outcome(
        OrderSide.BUY, before_amount, after_amount, target_amount,
        tradability, order,
    )
    result = sync_buy_state_after_order(
        code, outcome, before_amount, after_amount,
        context.current_dt.date(), snapshot["entry_atr"], execution_price,
    )
    state = g.position_states.get(code)
    log_order_transition(
        code, OrderSide.BUY, result, before_amount, after_amount,
        target_value, state.get("pending_exit") if state is not None else None,
    )
    return result


def submit_sell(context, code, reason, trigger_value):
    current_data = get_current_data()
    tradability = get_tradability(current_data, code)
    before_amount = get_actual_amount(context, code)
    order = None
    if before_amount > 0 and tradability is Tradability.TRADEABLE:
        order = order_target(code, 0)
    after_amount = get_actual_amount(context, code)
    outcome = classify_order_outcome(
        OrderSide.SELL, before_amount, after_amount, 0,
        tradability, order,
    )
    result = sync_sell_state_after_order(
        code, outcome, reason, context.current_dt.date(),
        trigger_value, after_amount,
    )
    state = g.position_states.get(code)
    log_order_transition(
        code, OrderSide.SELL, result, before_amount, after_amount, 0,
        state.get("pending_exit") if state is not None else None,
    )
    return result


def retry_pending_exits(context, current_data):
    if not hasattr(g, "daily_retried_exits"):
        g.daily_retried_exits = set()
    results = []
    for code, state in list(g.position_states.items()):
        pending_exit = state.get("pending_exit")
        if pending_exit is None:
            continue
        g.daily_retried_exits.add(code)
        outcome = submit_sell(
            context, code, pending_exit["reason"],
            pending_exit["trigger_value"],
        )
        results.append((code, outcome))
    return results


def update_highest_close_anchor(position_state, closing_price):
    if is_finite_positive(closing_price):
        position_state["highest_close_anchor"] = max(
            position_state["highest_close_anchor"], float(closing_price),
        )


def can_signal_sell(buy_date, decision_date):
    return buy_date < decision_date


def reset_daily_state(decision_date, signal_date):
    ensure_runtime_state()
    if getattr(g, "state_date", None) != decision_date:
        g.state_date = decision_date
        g.sold_today = set()
        g.daily_attempted_buys = set()
        g.daily_retried_exits = set()
    g.processed_resonance_ids = prune_processed_resonance_ids(
        g.processed_resonance_ids, signal_date,
    )


def clear_position_state_if_flat(code, actual_amount):
    if actual_amount == 0:
        g.position_states.pop(code, None)
        return True
    return False


TRADE_INDICATOR_COLUMNS = (
    "rsi14", "k", "d", "j", "kd_diff", "boll_mid",
    "boll_upper", "boll_lower", "atr14",
)
OBSERVATION_COLUMNS = (
    "rsi6", "rsi12", "rsi24", "plus_di", "minus_di", "adx14",
    "volume", "volume_ma5", "volume_ma20", "volume_ratio",
    "boll_width", "boll_mid_slope",
)
INDICATORS = ("BOLL", "RSI", "KDJ")


def calc_rsi(close, period):
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    rolling_high = high.rolling(n, min_periods=n).max()
    rolling_low = low.rolling(n, min_periods=n).min()
    spread = rolling_high - rolling_low
    rsv = 100.0 * (close - rolling_low) / spread.replace(0, np.nan)
    rsv = rsv.mask(spread == 0, 50.0)
    k = rsv.ewm(alpha=1.0 / m1, adjust=False, min_periods=1).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False, min_periods=1).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def calc_bollinger(close, period=20, std_mult=2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def calc_atr(high, low, close, period=14):
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def calc_dmi_adx(high, low, close, period=14):
    tr = true_range(high, low, close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0,
    ), index=high.index)
    minus_dm = pd.Series(np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0,
    ), index=high.index)
    atr_rma = tr.ewm(alpha=1.0 / period, adjust=False,
                     min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=1.0 / period, adjust=False,
                min_periods=period).mean()
    return plus_di, minus_di, adx


def build_indicator_frame(price_frame, params):
    frame = price_frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    frame["rsi14"] = calc_rsi(frame["close"], params["rsi_period"])
    for period in params["observation_rsi_periods"]:
        frame["rsi%s" % period] = calc_rsi(frame["close"], period)
    k, d, j = calc_kdj(frame["high"], frame["low"], frame["close"], *params["kdj"])
    frame["k"], frame["d"], frame["j"] = k, d, j
    frame["kd_diff"] = k - d
    mid, upper, lower = calc_bollinger(frame["close"], *params["boll"])
    frame["boll_mid"], frame["boll_upper"], frame["boll_lower"] = mid, upper, lower
    frame["atr14"] = calc_atr(
        frame["high"], frame["low"], frame["close"], params["atr_period"],
    )
    plus_di, minus_di, adx = calc_dmi_adx(
        frame["high"], frame["low"], frame["close"], 14,
    )
    frame["plus_di"], frame["minus_di"], frame["adx14"] = plus_di, minus_di, adx
    frame["volume_ma5"] = frame["volume"].rolling(5, min_periods=5).mean()
    frame["volume_ma20"] = frame["volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume_ma20"].replace(0, np.nan)
    frame["boll_width"] = (upper - lower) / mid.replace(0, np.nan)
    frame["boll_mid_slope"] = mid.diff()
    return frame


def detect_rsi_direction(previous, current, params):
    prev_rsi, curr_rsi = previous["rsi14"], current["rsi14"]
    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return TurnDirection.NEUTRAL
    if prev_rsi <= params["rsi_low"] and curr_rsi > prev_rsi:
        return TurnDirection.BUY_TURN
    if prev_rsi >= params["rsi_high"] and curr_rsi < prev_rsi:
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_kdj_direction(previous, current, params):
    required = ("k", "d", "j", "kd_diff")
    if any(pd.isna(previous[name]) or pd.isna(current[name]) for name in required):
        return TurnDirection.NEUTRAL
    buy_extreme = (min(previous["k"], previous["d"]) <= params["kdj_low"] or
                   previous["j"] <= params["j_low"])
    sell_extreme = (max(previous["k"], previous["d"]) >= params["kdj_high"] or
                    previous["j"] >= params["j_high"])
    if (buy_extreme and current["j"] > previous["j"] and
            current["kd_diff"] > previous["kd_diff"]):
        return TurnDirection.BUY_TURN
    if (sell_extreme and current["j"] < previous["j"] and
            current["kd_diff"] < previous["kd_diff"]):
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_kdj_formal_cross(previous, current):
    previous_diff = previous.get("kd_diff")
    current_diff = current.get("kd_diff")
    if pd.isna(previous_diff) or pd.isna(current_diff):
        return "NONE"
    if previous_diff <= 0 and current_diff > 0:
        return "GOLDEN_CROSS"
    if previous_diff >= 0 and current_diff < 0:
        return "DEATH_CROSS"
    return "NONE"


def detect_boll_direction(previous, current):
    fields = ("low", "high", "close", "boll_lower", "boll_upper")
    values = [previous.get(name) for name in fields]
    values += [current.get(name) for name in fields]
    if any(pd.isna(value) for value in values):
        return TurnDirection.NEUTRAL
    touched_lower = (previous["low"] <= previous["boll_lower"] or
                     previous["close"] <= previous["boll_lower"] or
                     current["low"] <= current["boll_lower"] or
                     current["close"] <= current["boll_lower"])
    touched_upper = (previous["high"] >= previous["boll_upper"] or
                     previous["close"] >= previous["boll_upper"] or
                     current["high"] >= current["boll_upper"] or
                     current["close"] >= current["boll_upper"])
    if (touched_lower and current["close"] > current["boll_lower"] and
            current["close"] > previous["close"]):
        return TurnDirection.BUY_TURN
    if (touched_upper and current["close"] < current["boll_upper"] and
            current["close"] < previous["close"]):
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def make_turn_event(indicator, direction, event_date, expires_date,
                    trigger_values, reference_extreme=None):
    return {
        "indicator": indicator,
        "direction": direction,
        "event_date": event_date,
        "expires_date": expires_date,
        "trigger_values": dict(trigger_values),
        "reference_extreme": reference_extreme,
        "invalid_reason": None,
    }


def empty_event_book():
    return {"active": {}, "invalidated": []}


def invalidate_event(book, indicator, reason):
    event = book["active"].pop(indicator, None)
    if event is not None:
        event = dict(event)
        event["invalid_reason"] = reason
        book["invalidated"].append(event)
    return event


def apply_event(book, event):
    old = book["active"].get(event["indicator"])
    if old is not None and old["direction"] is not event["direction"]:
        invalidate_event(book, event["indicator"], "REPLACED_BY_OPPOSITE_EVENT")
    book["active"][event["indicator"]] = event


def expire_events(book, signal_date):
    for indicator, event in list(book["active"].items()):
        if event["expires_date"] < signal_date:
            invalidate_event(book, indicator, "EVENT_EXPIRED")


def invalidate_boll_structure(book, latest_row):
    event = book["active"].get("BOLL")
    if event is None:
        return None
    if event["direction"] is TurnDirection.BUY_TURN:
        broken = (latest_row["close"] <= latest_row["boll_lower"] and
                  latest_row["low"] < event["reference_extreme"])
        if broken:
            return invalidate_event(
                book, "BOLL", "NEW_LOWER_LOW_OUTSIDE_LOWER_BAND",
            )
    if event["direction"] is TurnDirection.SELL_TURN:
        broken = (latest_row["close"] >= latest_row["boll_upper"] and
                  latest_row["high"] > event["reference_extreme"])
        if broken:
            return invalidate_event(
                book, "BOLL", "NEW_HIGHER_HIGH_OUTSIDE_UPPER_BAND",
            )
    return None


def _trigger_values(indicator, previous, current):
    fields_by_indicator = {
        "BOLL": ("low", "high", "close", "boll_lower", "boll_upper"),
        "RSI": ("rsi14",),
        "KDJ": ("k", "d", "j", "kd_diff"),
    }
    fields = fields_by_indicator[indicator]
    return {
        "previous": {name: previous[name] for name in fields},
        "current": {name: current[name] for name in fields},
    }


def _make_detected_event(indicator, direction, previous, current,
                         event_date, expires_date):
    reference_extreme = None
    if indicator == "BOLL":
        reference_extreme = (
            current["low"] if direction is TurnDirection.BUY_TURN
            else current["high"]
        )
    return make_turn_event(
        indicator=indicator,
        direction=direction,
        event_date=event_date,
        expires_date=expires_date,
        trigger_values=_trigger_values(indicator, previous, current),
        reference_extreme=reference_extreme,
    )


def collect_latest_events(indicator_frame, signal_date, next_trade_date):
    params = get_default_params()
    complete_frame = indicator_frame.loc[indicator_frame.index <= signal_date]
    book = empty_event_book()
    first_event_position = max(1, len(complete_frame) - 2)
    for position in range(first_event_position, len(complete_frame)):
        previous = complete_frame.iloc[position - 1]
        current = complete_frame.iloc[position]
        event_date = complete_frame.index[position]
        expires_date = (
            complete_frame.index[position + 1]
            if position + 1 < len(complete_frame) else next_trade_date
        )
        expire_events(book, event_date)
        directions = {
            "BOLL": detect_boll_direction(previous, current),
            "RSI": detect_rsi_direction(previous, current, params),
            "KDJ": detect_kdj_direction(previous, current, params),
        }
        for indicator in INDICATORS:
            direction = directions[indicator]
            if direction is not TurnDirection.NEUTRAL:
                apply_event(book, _make_detected_event(
                    indicator, direction, previous, current,
                    event_date, expires_date,
                ))
        invalidate_boll_structure(book, current)
    expire_events(book, signal_date)
    return book


def active_direction(event_book, indicator):
    event = event_book["active"].get(indicator)
    return event["direction"] if event is not None else TurnDirection.NEUTRAL


def build_resonance_id(code, direction, supporters):
    parts = [direction.value, code]
    parts.extend(
        "%s:%s" % (event["indicator"], event["event_date"])
        for event in sorted(supporters, key=lambda item: item["indicator"])
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def build_resonance_decision(code, direction, event_book, signal_date):
    boll = event_book["active"].get("BOLL")
    if boll is None or boll["direction"] is not direction:
        return None

    oscillators = [
        event_book["active"].get("RSI"),
        event_book["active"].get("KDJ"),
    ]
    if any(event is not None and event["direction"] is OPPOSITE[direction]
           for event in oscillators):
        return None

    supporters = [boll] + [
        event for event in oscillators
        if event is not None and event["direction"] is direction
    ]
    if len(supporters) < 2:
        return None
    if not any(event["event_date"] == signal_date for event in supporters):
        return None

    return {
        "code": code,
        "direction": direction,
        "signal_date": signal_date,
        "supporters": tuple(event["indicator"] for event in supporters),
        "support_count": len(supporters),
        "boll_age": 0 if boll["event_date"] == signal_date else 1,
        "resonance_id": build_resonance_id(code, direction, supporters),
        "expires_date": min(event["expires_date"] for event in supporters),
    }


def resonance_rejection_reason(direction, event_book, signal_date):
    boll = event_book["active"].get("BOLL")
    if boll is None or boll["direction"] is not direction:
        return "BOLL_NOT_SUPPORTING"
    oscillators = [
        event_book["active"].get("RSI"),
        event_book["active"].get("KDJ"),
    ]
    if any(
            event is not None and event["direction"] is OPPOSITE[direction]
            for event in oscillators):
        return "THIRD_INDICATOR_CONFLICT"
    supporters = [boll] + [
        event for event in oscillators
        if event is not None and event["direction"] is direction
    ]
    if len(supporters) < 2:
        return "INSUFFICIENT_SUPPORT"
    if not any(event["event_date"] == signal_date for event in supporters):
        return "NO_FRESH_SUPPORTER"
    return "RESONANCE_REJECTED"


def sort_buy_decisions(decisions):
    return sorted(decisions, key=lambda item: (
        -item["support_count"], item["boll_age"], item["code"],
    ))


def prune_processed_resonance_ids(processed, signal_date):
    return {
        resonance_id: expires_date
        for resonance_id, expires_date in processed.items()
        if expires_date >= signal_date
    }


def mark_resonance_processed(processed, decision):
    processed[decision["resonance_id"]] = decision["expires_date"]


def collect_complete_resonance_decisions(snapshots, direction):
    decisions = {}
    for code, snapshot in snapshots.items():
        if not snapshot.get("valid"):
            log_resonance_decision({
                "code": code, "direction": direction,
                "signal_date": snapshot.get("signal_date"),
            }, False, snapshot.get("reason", "INVALID_SIGNAL_SNAPSHOT"))
            continue
        decision = build_resonance_decision(
            code, direction,
            snapshot["event_book"], snapshot["signal_date"],
        )
        if decision is not None:
            decisions[code] = decision
            log_resonance_decision(decision, True, "COMPLETE_RESONANCE")
            try_register_observation_event(
                decision, snapshot["signal_date"], snapshot["close"],
            )
        else:
            log_resonance_decision({
                "code": code, "direction": direction,
                "signal_date": snapshot.get("signal_date"),
            }, False, resonance_rejection_reason(
                direction, snapshot["event_book"],
                snapshot["signal_date"],
            ))
    return decisions


def collect_buy_decisions(snapshots, actual_positions):
    complete = collect_complete_resonance_decisions(
        snapshots, TurnDirection.BUY_TURN,
    )
    decisions = []
    for code, decision in complete.items():
        if not is_finite_positive(snapshots[code].get("entry_atr")):
            log_resonance_decision(decision, False, "INVALID_ENTRY_ATR")
            continue
        decisions.append(decision)
    return decisions


def run_atr_exits(context, current_data):
    attempted = set()
    decision_date = context.current_dt.date()
    retried_codes = getattr(g, "daily_retried_exits", set())
    for code in get_actual_positions(context):
        if code in g.sold_today:
            continue
        state = g.position_states.get(code)
        if state is None:
            continue
        stop_state = calc_stop_state(
            state["highest_close_anchor"], state["entry_atr"], g.params,
        )
        execution_price = get_execution_price(current_data, code)
        triggered = bool(
            stop_state is not None
            and execution_price is not None
            and execution_price <= stop_state["stop_price"]
        )
        _emit_structured_log("atr_check", {
            "code": code,
            "entry_atr": state["entry_atr"],
            "highest_close_anchor": state["highest_close_anchor"],
            "stop_price": (
                stop_state["stop_price"] if stop_state is not None else None
            ),
            "stop_pct": (
                stop_state["stop_pct"] if stop_state is not None else None
            ),
            "current_price": execution_price,
            "triggered": triggered,
            "pending_exit": state.get("pending_exit"),
        })
        if (stop_state is None or execution_price is None
                or execution_price > stop_state["stop_price"]):
            continue
        if code in retried_codes:
            set_pending_exit(
                state, ExitReason.ATR_EXIT, decision_date,
                stop_state["stop_price"], get_actual_amount(context, code),
            )
            continue
        submit_sell(
            context, code, ExitReason.ATR_EXIT, stop_state["stop_price"],
        )
        attempted.add(code)
    return attempted


def run_signal_exits(context, current_data, snapshots):
    attempted = set()
    decision_date = context.current_dt.date()
    retried_codes = getattr(g, "daily_retried_exits", set())
    actual_positions = get_actual_positions(context)
    sell_decisions = collect_complete_resonance_decisions(
        snapshots, TurnDirection.SELL_TURN,
    )
    for code, decision in sell_decisions.items():
        if code not in actual_positions:
            log_resonance_decision(decision, False, "UNHELD_RECORD_ONLY")
    for code in actual_positions:
        decision = sell_decisions.get(code)
        if code in g.sold_today:
            if decision is not None:
                log_resonance_decision(decision, False, "SOLD_TODAY")
            continue
        if code in retried_codes:
            if decision is not None:
                log_resonance_decision(decision, False, "PENDING_RETRIED_TODAY")
            continue
        state = g.position_states.get(code)
        if state is None:
            continue
        if not can_signal_sell(state["buy_date"], decision_date):
            if decision is not None:
                log_resonance_decision(decision, False, "MINIMUM_HOLD_DAY")
            continue
        pending_exit = state.get("pending_exit")
        if (pending_exit is not None
                and pending_exit["reason"] is ExitReason.ATR_EXIT):
            if decision is not None:
                log_resonance_decision(decision, False, "ATR_PENDING_PRIORITY")
            continue
        if decision is None:
            continue
        snapshot = snapshots[code]
        if decision["resonance_id"] in g.processed_resonance_ids:
            log_resonance_decision(
                decision, False, "RESONANCE_ALREADY_PROCESSED",
            )
            continue
        tradability = get_tradability(current_data, code)
        if tradability is Tradability.PAUSED:
            log_resonance_decision(decision, False, "PAUSED")
            continue
        mark_resonance_processed(g.processed_resonance_ids, decision)
        log_resonance_decision(decision, True, "SIGNAL_EXIT_ATTEMPT")
        submit_sell(
            context, code, ExitReason.SIGNAL_EXIT, snapshot["close"],
        )
        attempted.add(code)
    return attempted


def run_signal_buys(context, current_data, snapshots):
    actual_positions = get_actual_positions(context)
    decisions = collect_buy_decisions(snapshots, actual_positions)
    sorted_decisions = sort_buy_decisions(decisions)
    for rank, decision in enumerate(sorted_decisions, start=1):
        log_resonance_decision(
            decision, True, "BUY_CANDIDATE_SORTED:%s" % rank,
        )
    remaining_slots = max(
        0, g.params["max_holdings"] - len(actual_positions),
    )
    if remaining_slots == 0:
        for decision in sorted_decisions:
            log_resonance_decision(decision, False, "PORTFOLIO_FULL")
        return []

    results = []
    for decision in sorted_decisions:
        if remaining_slots == 0:
            log_resonance_decision(decision, False, "PORTFOLIO_FULL")
            continue
        code = decision["code"]
        if code in actual_positions:
            log_resonance_decision(decision, False, "HELD_NO_ADD")
            continue
        if code in g.sold_today:
            log_resonance_decision(decision, False, "SOLD_TODAY")
            continue
        if code in g.daily_attempted_buys:
            log_resonance_decision(
                decision, False, "ALREADY_ATTEMPTED_TODAY",
            )
            continue
        if decision["resonance_id"] in g.processed_resonance_ids:
            log_resonance_decision(
                decision, False, "RESONANCE_ALREADY_PROCESSED",
            )
            continue
        tradability = get_tradability(current_data, code)
        if tradability is Tradability.PAUSED:
            log_resonance_decision(decision, False, "PAUSED_BACKFILL")
            results.append((code, OrderOutcome.PAUSED))
            continue
        mark_resonance_processed(g.processed_resonance_ids, decision)
        g.daily_attempted_buys.add(code)
        if tradability is Tradability.UNKNOWN:
            log_resonance_decision(
                decision, False, "UNKNOWN_TRADABILITY_ATTEMPT_CONSUMED",
            )
            results.append((code, OrderOutcome.UNKNOWN))
            remaining_slots -= 1
            continue
        outcome = submit_buy(context, code, snapshots[code], decision)
        results.append((code, outcome))
        if outcome is OrderOutcome.PAUSED:
            log_resonance_decision(
                decision, False, "REFRESHED_PAUSED_BACKFILL",
            )
            g.processed_resonance_ids.pop(decision["resonance_id"], None)
            g.daily_attempted_buys.discard(code)
            continue
        remaining_slots -= 1
    return results
