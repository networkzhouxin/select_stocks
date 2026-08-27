from jqdata import *

import hashlib
import json
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


def do_trading(context):
    ensure_runtime_state()


def after_close(context):
    ensure_runtime_state()


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
    log.info("version=%s build=%s fingerprint=%s pool=%s",
             STRATEGY_VERSION, DEPLOYMENT_BUILD_ID,
             business_config_fingerprint(), get_default_etf_pool())


import numpy as np
import pandas as pd


def calc_buy_target_value(total_value, available_cash, params):
    standard_target = (
        total_value * params["target_exposure"] / params["max_holdings"]
    )
    cash_reserve = total_value * (1.0 - params["target_exposure"])
    return min(standard_target, max(0.0, available_cash - cash_reserve))


def calc_stop_state(highest_close_anchor, entry_atr, params):
    if (highest_close_anchor <= 0 or pd.isna(highest_close_anchor)
            or entry_atr <= 0 or pd.isna(entry_atr)):
        return None
    raw_pct = params["atr_multiplier"] * entry_atr / highest_close_anchor
    stop_pct = min(params["stop_cap"], max(params["stop_floor"], raw_pct))
    return {
        "raw_pct": raw_pct,
        "stop_pct": stop_pct,
        "stop_price": highest_close_anchor * (1.0 - stop_pct),
    }


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


def retry_pending_exits(context, current_data):
    results = []
    for code, state in list(g.position_states.items()):
        pending_exit = state.get("pending_exit")
        if pending_exit is None:
            continue
        outcome = submit_sell(
            context, code, pending_exit["reason"],
            pending_exit["trigger_value"],
        )
        results.append((code, outcome))
    return results


def update_highest_close_anchor(position_state, closing_price):
    if closing_price is not None and closing_price > 0:
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
        "supporters": tuple(event["indicator"] for event in supporters),
        "support_count": len(supporters),
        "boll_age": 0 if boll["event_date"] == signal_date else 1,
        "resonance_id": build_resonance_id(code, direction, supporters),
        "expires_date": min(event["expires_date"] for event in supporters),
    }


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
