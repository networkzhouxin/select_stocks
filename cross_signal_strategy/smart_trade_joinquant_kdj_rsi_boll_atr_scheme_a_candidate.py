# -*- coding: utf-8 -*-
"""Independent JoinQuant candidate for KRBA scheme A.

All indicator decisions use completed T-1-or-earlier daily bars.  The formal
``cross-v0.3.3`` JoinQuant and PTrade strategies do not import this module.
"""

import builtins as _builtins
import math

import numpy as np
import pandas as pd
from jqdata import *


STRATEGY_VERSION = "krba-rsi-turn-v0.1-joinquant-candidate"
DEPLOYMENT_BUILD_ID = "20260826.1-candidate"
LOOKBACK = 120


def get_default_params():
    return {
        "lookback": LOOKBACK,
        "max_hold": 3,
        "base_ratio": 0.95,
        "min_signal_hold_days": 5,
        "atr_multiplier": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
    }


def get_default_etf_pool():
    return [
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


def initialize(context):
    del context
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    g.params = get_default_params()
    g.etf_pool = get_default_etf_pool()
    g.position_states = {}
    g.last_snapshots = {}
    g.sold_today = set()
    g.pending_sells = set()
    g.sold_guard_date = None
    run_daily(do_trading, time="09:35")
    run_daily(check_atr_1450, time="14:50")
    log.info(
        "[%s] initialized build=%s T-1 signals; 14:50 ATR-only"
        % (STRATEGY_VERSION, DEPLOYMENT_BUILD_ID)
    )


def do_trading(context):
    today = context.current_dt.date()
    today_text = today.isoformat()
    _reset_sold_today(today_text)
    for code in list(g.position_states):
        if code not in context.portfolio.positions:
            g.position_states.pop(code, None)
    previous_days = list(get_trade_days(end_date=today, count=2))
    if len(previous_days) < 2:
        log.warning("[09:35 blocked] previous trade date unavailable")
        return
    signal_date = previous_days[-2]
    snapshots = []
    for code in g.etf_pool:
        snapshot, reason = load_signal_snapshot(code, signal_date, return_reason=True)
        if snapshot is None:
            log.info("[signal skipped] %s reason=%s" % (code, reason))
            continue
        snapshots.append(snapshot)
    g.last_snapshots = {item["code"]: item for item in snapshots}

    current_data = get_current_data()
    held_codes = list(context.portfolio.positions)
    current_prices = {}
    for code in set(g.etf_pool) | set(held_codes):
        quote = current_data[code]
        if getattr(quote, "paused", False):
            continue
        price = _number(getattr(quote, "last_price", None))
        if math.isfinite(price) and price > 0:
            current_prices[code] = price

    if g.position_states:
        earliest = min(state.entry_date for state in g.position_states.values())
        trade_days = list(get_trade_days(start_date=earliest, end_date=today))
    else:
        trade_days = previous_days
    plans = plan_0935_orders(
        snapshots=snapshots,
        held_codes=held_codes,
        position_states=g.position_states,
        current_prices=current_prices,
        today=today_text,
        trade_days=trade_days,
        total_value=context.portfolio.total_value,
        sold_today=g.sold_today,
        params=g.params,
    )
    sell_plans = [plan for plan in plans if plan["target_value"] <= 0]
    buy_plans = [plan for plan in plans if plan["target_value"] > 0]

    for plan in sell_plans:
        code = plan["code"]
        if code in g.sold_today or code not in context.portfolio.positions:
            continue
        if code not in current_prices:
            log.warning("[09:35 sell blocked] %s invalid or paused quote" % code)
            continue
        log.info(
            "[09:35 sell] %s reason=%s price=%.3f"
            % (code, plan["reason"], current_prices[code])
        )
        before_amount = _position_amount(context, code)
        order = order_target(code, 0)
        after_amount = _position_amount(context, code)
        outcome = _record_sell_submission(
            code, order, before_amount, after_amount
        )
        log.info("[09:35 sell result] %s outcome=%s" % (code, outcome))

    for plan in buy_plans:
        code = plan["code"]
        if code in g.sold_today or code in context.portfolio.positions:
            continue
        if len(context.portfolio.positions) >= int(g.params["max_hold"]):
            break
        if code not in current_prices:
            log.warning("[09:35 buy blocked] %s invalid or paused quote" % code)
            continue
        log.info(
            "[09:35 buy] %s channel=%s price=%.3f target=%.2f"
            % (code, plan["reason"], current_prices[code], plan["target_value"])
        )
        order = order_target_value(code, plan["target_value"])
        position = context.portfolio.positions.get(code)
        filled = _number(getattr(order, "filled", 0)) if order is not None else 0.0
        if position is None and (not math.isfinite(filled) or filled <= 0):
            log.warning("[09:35 buy not filled] %s" % code)
            continue
        entry_price = _number(getattr(position, "avg_cost", None))
        if not math.isfinite(entry_price) or entry_price <= 0:
            entry_price = current_prices[code]
        g.position_states[code] = PositionSignalState(
            entry_date=today_text,
            entry_price=entry_price,
            entry_atr=float(plan["entry_atr"]),
            highest_close=entry_price,
        )


def check_atr_1450(context):
    today = context.current_dt.date().isoformat()
    _reset_sold_today(today)
    current_data = get_current_data()
    for code in list(context.portfolio.positions):
        if code in getattr(g, "pending_sells", set()):
            continue
        state = g.position_states.get(code)
        if state is None:
            log.warning("[14:50 ATR blocked] %s missing position state" % code)
            continue
        quote = current_data[code]
        if getattr(quote, "paused", False):
            log.info("[14:50 ATR blocked] %s paused" % code)
            continue
        price = _number(getattr(quote, "last_price", None))
        if not math.isfinite(price) or price <= 0:
            log.warning("[14:50 ATR blocked] %s invalid price" % code)
            continue
        stop = calc_frozen_atr_stop(state, code)
        if round(price, 3) <= round(stop, 3):
            log.info(
                "[14:50 ATR sell] %s price=%.3f stop=%.3f" % (code, price, stop)
            )
            before_amount = _position_amount(context, code)
            order = order_target(code, 0)
            after_amount = _position_amount(context, code)
            outcome = _record_sell_submission(
                code, order, before_amount, after_amount
            )
            log.info("[14:50 ATR result] %s outcome=%s" % (code, outcome))


def _reset_sold_today(today):
    if getattr(g, "sold_guard_date", None) != str(today):
        g.sold_guard_date = str(today)
        g.sold_today = set()
        g.pending_sells = set()


def classify_sell_submission(order, before_amount, after_amount):
    before = max(0.0, _number(before_amount))
    after = max(0.0, _number(after_amount))
    filled = _number(getattr(order, "filled", 0)) if order is not None else 0.0
    if not math.isfinite(filled) or filled < 0:
        filled = 0.0
    if after <= 0 or (before > 0 and filled >= before):
        return "full"
    status = str(getattr(order, "status", "")).lower() if order is not None else ""
    active = _builtins.any(
        token in status for token in ("held", "pending", "open", "new")
    )
    partial = filled > 0 or (math.isfinite(before) and after < before)
    if partial:
        return "partial_pending" if active else "partial"
    if active:
        return "pending"
    return "rejected"


def _position_amount(context, code):
    position = context.portfolio.positions.get(code)
    if position is None:
        return 0.0
    value = getattr(position, "total_amount", getattr(position, "amount", 0))
    result = _number(value)
    return result if math.isfinite(result) and result > 0 else 0.0


def _record_sell_submission(code, order, before_amount, after_amount):
    outcome = classify_sell_submission(order, before_amount, after_amount)
    pending = getattr(g, "pending_sells", set())
    g.pending_sells = pending
    if outcome == "full":
        g.sold_today.add(code)
        pending.discard(code)
        g.position_states.pop(code, None)
    elif outcome == "partial_pending":
        g.sold_today.add(code)
        pending.add(code)
    elif outcome == "partial":
        g.sold_today.add(code)
        pending.discard(code)
    elif outcome == "pending":
        pending.add(code)
    else:
        pending.discard(code)
    return outcome


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def calc_rsi(close, period=6):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    result[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return result


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d


def calc_bollinger(close, period=20, std_mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def calc_atr(high, low, close, period=14):
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def load_signal_snapshot(code, signal_date, return_reason=False):
    try:
        frame = get_price(
            code,
            end_date=signal_date,
            count=LOOKBACK,
            frequency="daily",
            fields=["high", "low", "close", "volume"],
            skip_paused=True,
            fq="pre",
            panel=False,
        )
        result, reason = _snapshot_from_daily_frame(code, signal_date, frame)
    except Exception as exc:
        result, reason = None, "exception:%s" % exc.__class__.__name__
    if return_reason:
        return result, reason
    return result


def _snapshot_from_daily_frame(code, signal_date, frame):
    required = {"high", "low", "close", "volume"}
    if frame is None or not required.issubset(frame.columns):
        return None, "missing_daily_data"
    work = frame.copy()
    work.index = pd.to_datetime(work.index)
    cutoff = pd.Timestamp(signal_date)
    work = work.loc[work.index.normalize() <= cutoff.normalize()].sort_index()
    if work.empty or work.index[-1].date() != cutoff.date():
        return None, "stale_signal_date"
    if len(work) < 20:
        return None, "insufficient_history"
    if float(pd.to_numeric(work["volume"], errors="coerce").iloc[-5:].sum()) <= 0:
        return None, "zero_recent_volume"

    high = pd.to_numeric(work["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(work["low"], errors="coerce").reset_index(drop=True)
    close = pd.to_numeric(work["close"], errors="coerce").reset_index(drop=True)
    rsi6 = calc_rsi(close, 6)
    k, d = calc_kdj(high, low, close, 9, 3, 3)
    upper, mid, lower = calc_bollinger(close, 20, 2.0)
    atr = calc_atr(high, low, close, 14)
    values = {
        "k_prev": k.iloc[-2],
        "d_prev": d.iloc[-2],
        "k": k.iloc[-1],
        "d": d.iloc[-1],
        "rsi6_2ago": rsi6.iloc[-3],
        "rsi6_prev": rsi6.iloc[-2],
        "rsi6": rsi6.iloc[-1],
        "close_prev": close.iloc[-2],
        "low": low.iloc[-1],
        "close": close.iloc[-1],
        "boll_lower": lower.iloc[-1],
        "boll_mid": mid.iloc[-1],
        "boll_upper": upper.iloc[-1],
        "atr": atr.iloc[-1],
    }
    if not all(math.isfinite(_number(value)) for value in values.values()):
        return None, "invalid_indicator"
    last_date = work.index[-1].date().isoformat()
    result = dict(values)
    result.update(
        {
            "code": code,
            "signal_date": last_date,
            "max_data_date": last_date,
        }
    )
    result["entry_channel"] = classify_entry_channel(result)
    return result, None


def is_original_kdj_entry(snapshot):
    values = tuple(
        _number(snapshot.get(name))
        for name in (
            "k_prev",
            "d_prev",
            "k",
            "d",
            "rsi6",
            "low",
            "close",
            "boll_lower",
        )
    )
    if not all(math.isfinite(value) for value in values):
        return False
    k_prev, d_prev, k, d, rsi6, low, close, lower = values
    return bool(
        k_prev <= d_prev
        and k > d
        and rsi6 <= 30.0
        and low <= lower
        and close > lower
    )


def is_rsi_low_turn_entry(snapshot):
    values = tuple(
        _number(snapshot.get(name))
        for name in ("rsi6_2ago", "rsi6_prev", "rsi6", "close_prev", "close")
    )
    if not all(math.isfinite(value) for value in values):
        return False
    rsi_2ago, rsi_prev, rsi_now, close_prev, close_now = values
    return bool(
        rsi_2ago > rsi_prev
        and rsi_now > rsi_prev
        and rsi_prev <= 30.0
        and close_now > close_prev
    )


def classify_entry_channel(snapshot):
    if is_original_kdj_entry(snapshot):
        return "kdj_cross"
    if is_rsi_low_turn_entry(snapshot):
        return "rsi_low_turn"
    return None


def build_buy_queue(snapshots, excluded_codes=None, etf_pool=None):
    excluded = set(excluded_codes or ())
    pool = list(etf_pool or get_default_etf_pool())
    pool_rank = {code: index for index, code in enumerate(pool)}
    by_channel = {"kdj_cross": [], "rsi_low_turn": []}
    seen = set()
    for snapshot in snapshots:
        code = snapshot.get("code")
        if not code or code in excluded or code in seen:
            continue
        channel = classify_entry_channel(snapshot)
        if channel is None:
            continue
        item = dict(snapshot)
        item["entry_channel"] = channel
        by_channel[channel].append(item)
        seen.add(code)
    by_channel["kdj_cross"].sort(
        key=lambda item: (
            -(_number(item.get("k")) - _number(item.get("d"))),
            _number(item.get("rsi6")),
            pool_rank.get(item["code"], len(pool)),
        )
    )
    by_channel["rsi_low_turn"].sort(
        key=lambda item: pool_rank.get(item["code"], len(pool))
    )
    return by_channel["kdj_cross"] + by_channel["rsi_low_turn"]


class PositionSignalState:
    def __init__(
        self,
        entry_date,
        entry_price,
        entry_atr,
        highest_close,
        mean_reached=False,
        upper_reached=False,
    ):
        self.entry_date = str(entry_date)
        self.entry_price = float(entry_price)
        self.entry_atr = float(entry_atr)
        self.highest_close = float(highest_close)
        self.mean_reached = bool(mean_reached)
        self.upper_reached = bool(upper_reached)


def calc_frozen_atr_stop(state, code="", multiplier=2.5, floor=0.05, cap=0.15):
    if str(code).split(".")[0] == "518880":
        floor = 0.03
    if state.highest_close <= 0 or state.entry_atr <= 0:
        return state.entry_price * (1.0 - cap)
    distance = multiplier * state.entry_atr / state.highest_close
    distance = max(floor, min(cap, distance))
    return state.highest_close * (1.0 - distance)


def update_state_from_t1(state, snapshot):
    close = _number(snapshot.get("close"))
    mid = _number(snapshot.get("boll_mid"))
    upper = _number(snapshot.get("boll_upper"))
    if not all(math.isfinite(value) for value in (close, mid, upper)):
        return
    state.highest_close = max(state.highest_close, close)
    state.mean_reached = bool(state.mean_reached or close >= mid)
    state.upper_reached = bool(state.upper_reached or close >= upper)


def _kd_cross_down(snapshot):
    values = tuple(
        _number(snapshot.get(name))
        for name in ("k_prev", "d_prev", "k", "d")
    )
    return bool(
        all(math.isfinite(value) for value in values)
        and values[0] >= values[1]
        and values[2] < values[3]
    )


def choose_exit_reason(state, snapshot, current_price, hold_days, code=""):
    price = _number(current_price)
    if math.isfinite(price) and price > 0:
        if round(price, 3) <= round(calc_frozen_atr_stop(state, code), 3):
            return "atr_stop"
    if int(hold_days) < 5:
        return None
    if state.upper_reached:
        return "boll_upper_target"
    close = _number(snapshot.get("close"))
    mid = _number(snapshot.get("boll_mid"))
    if state.mean_reached and (
        _kd_cross_down(snapshot)
        or (math.isfinite(close) and math.isfinite(mid) and close < mid)
    ):
        return "mean_reached_weakness"
    return None


def trading_days_between(start_date, end_date, trade_days):
    days = [str(day) for day in trade_days]
    start = str(start_date)
    end = str(end_date)
    try:
        return days.index(end) - days.index(start)
    except ValueError:
        return 0


def plan_0935_orders(
    snapshots,
    held_codes,
    position_states,
    current_prices,
    today,
    trade_days,
    total_value,
    sold_today=None,
    params=None,
):
    p = params or get_default_params()
    held = list(held_codes)
    sold = set(sold_today or ())
    snapshot_map = {item["code"]: item for item in snapshots}
    plans = []
    planned_sells = set()
    for code in held:
        if code in sold:
            continue
        state = position_states.get(code)
        snapshot = snapshot_map.get(code)
        if state is None:
            continue
        if snapshot is not None:
            update_state_from_t1(state, snapshot)
        price = _number(current_prices.get(code))
        if math.isfinite(price) and price > 0:
            stop = calc_frozen_atr_stop(state, code)
            if round(price, 3) <= round(stop, 3):
                plans.append(
                    {"code": code, "target_value": 0.0, "reason": "atr_stop"}
                )
                planned_sells.add(code)
                continue
        if snapshot is None:
            continue
        hold_days = trading_days_between(state.entry_date, today, trade_days)
        reason = choose_exit_reason(
            state,
            snapshot,
            price,
            hold_days,
            code,
        )
        if reason is not None:
            plans.append({"code": code, "target_value": 0.0, "reason": reason})
            planned_sells.add(code)

    held_after_sell = [code for code in held if code not in planned_sells]
    slots = int(p["max_hold"]) - len(held_after_sell)
    if slots <= 0:
        return plans
    exclusions = set(held) | sold | planned_sells
    queue = build_buy_queue(snapshots, exclusions, get_default_etf_pool())
    target = float(total_value) * float(p["base_ratio"]) / int(p["max_hold"])
    for item in queue[:slots]:
        plans.append(
            {
                "code": item["code"],
                "target_value": target,
                "reason": item["entry_channel"],
                "entry_atr": float(item["atr"]),
            }
        )
    return plans
