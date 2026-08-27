# -*- coding: utf-8 -*-
"""Standalone JoinQuant candidate for completed-week trend plus daily pullback.

All ordinary indicators use an explicit T-1-or-earlier daily frame.  The
decision week's partial weekly bar is excluded before weekly aggregation.
This file intentionally does not import the formal cross-signal strategies.
"""

import builtins as _builtins
import math

import numpy as np
import pandas as pd
from jqdata import *


STRATEGY_VERSION = "weekly-trend-pullback-v0.1-joinquant-candidate"
DEPLOYMENT_BUILD_ID = "20260827.1-candidate"
LOOKBACK = 180


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


class PositionSignalState(object):
    def __init__(self, entry_date, entry_price, entry_atr, highest_close):
        self.entry_date = str(entry_date)
        self.entry_price = float(entry_price)
        self.entry_atr = float(entry_atr)
        self.highest_close = float(highest_close)


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
        "[%s] initialized build=%s exact T-1 daily/completed-week signals; "
        "14:50 ATR-only; pandas=%s"
        % (STRATEGY_VERSION, DEPLOYMENT_BUILD_ID, pd.__version__)
    )


def do_trading(context):
    today = context.current_dt.date()
    today_text = today.isoformat()
    _reset_sold_today(today_text)
    for code in list(g.position_states):
        if code not in context.portfolio.positions:
            g.position_states.pop(code, None)

    trade_calendar = list(get_trade_days(end_date=today, count=2))
    if len(trade_calendar) < 2:
        log.warning("[09:35 blocked] previous trade date unavailable")
        return
    signal_date = trade_calendar[-2]

    snapshots = []
    for code in g.etf_pool:
        snapshot, reason = load_signal_snapshot(
            code,
            signal_date,
            today,
            return_reason=True,
        )
        if snapshot is None:
            log.info(
                "[09:35 snapshot blocked] %s decision=%s signal=%s reason=%s"
                % (code, today_text, _date_text(signal_date), reason)
            )
            continue
        snapshots.append(snapshot)
        log.info(
            "[09:35 snapshot] %s decision=%s signal=%s week_end=%s "
            "week_last=%s w_close=%.6f w_ma20=%.6f w_ma20_prev=%.6f "
            "boll=%.6f/%.6f/%.6f kd=%.6f/%.6f->%.6f/%.6f "
            "rsi=%.6f->%.6f atr=%.6f eligible=%s"
            % (
                code,
                today_text,
                snapshot["signal_date"],
                snapshot["weekly_period_end"],
                snapshot["weekly_last_trade_date"],
                snapshot["weekly_close"],
                snapshot["weekly_ma20"],
                snapshot["weekly_ma20_prev"],
                snapshot["boll_lower"],
                snapshot["boll_mid"],
                snapshot["boll_upper"],
                snapshot["k_prev"],
                snapshot["d_prev"],
                snapshot["k"],
                snapshot["d"],
                snapshot["rsi6_prev"],
                snapshot["rsi6"],
                snapshot["atr"],
                is_entry_eligible(snapshot),
            )
        )
    g.last_snapshots = dict((item["code"], item) for item in snapshots)

    current_data = get_current_data()
    held_codes = list(context.portfolio.positions)
    current_prices = {}
    price_codes = list(g.etf_pool) + [
        code for code in held_codes if code not in g.etf_pool
    ]
    for code in price_codes:
        quote = current_data[code]
        if getattr(quote, "paused", False):
            continue
        price = _number(getattr(quote, "last_price", None))
        if math.isfinite(price) and price > 0.0:
            current_prices[code] = price

    if g.position_states:
        earliest = min(state.entry_date for state in g.position_states.values())
        trade_days = list(get_trade_days(start_date=earliest, end_date=today))
    else:
        trade_days = trade_calendar
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
    sell_plans = [plan for plan in plans if plan["target_value"] <= 0.0]
    buy_plans = [plan for plan in plans if plan["target_value"] > 0.0]

    for plan in sell_plans:
        code = plan["code"]
        if code in g.sold_today or code not in context.portfolio.positions:
            continue
        if code not in current_prices:
            log.warning(
                "[09:35 sell blocked] %s reason=%s invalid_or_paused_price"
                % (code, plan["reason"])
            )
            continue
        log.info(
            "[09:35 sell] %s reason=%s price=%.6f target=0"
            % (code, plan["reason"], current_prices[code])
        )
        before_amount = _position_amount(context, code)
        order = order_target(code, 0)
        after_amount = _position_amount(context, code)
        outcome = _record_sell_submission(
            code,
            order,
            before_amount,
            after_amount,
        )
        log.info(
            "[09:35 sell result] %s reason=%s outcome=%s"
            % (code, plan["reason"], outcome)
        )

    for plan in buy_plans:
        code = plan["code"]
        if code in g.sold_today or code in context.portfolio.positions:
            continue
        if len(context.portfolio.positions) >= int(g.params["max_hold"]):
            break
        if code not in current_prices:
            log.warning(
                "[09:35 buy blocked] %s reason=%s invalid_or_paused_price"
                % (code, plan["reason"])
            )
            continue
        log.info(
            "[09:35 buy] %s reason=%s price=%.6f target=%.2f entry_atr=%.6f"
            % (
                code,
                plan["reason"],
                current_prices[code],
                plan["target_value"],
                plan["entry_atr"],
            )
        )
        order = order_target_value(code, plan["target_value"])
        position = context.portfolio.positions.get(code)
        filled = _number(getattr(order, "filled", 0)) if order is not None else 0.0
        if position is None and (not math.isfinite(filled) or filled <= 0.0):
            log.warning("[09:35 buy not filled] %s" % code)
            continue
        entry_price = _number(getattr(position, "avg_cost", None))
        if not math.isfinite(entry_price) or entry_price <= 0.0:
            entry_price = current_prices[code]
        g.position_states[code] = PositionSignalState(
            today_text,
            entry_price,
            plan["entry_atr"],
            entry_price,
        )
        log.info(
            "[09:35 buy result] %s outcome=filled entry_price=%.6f entry_atr=%.6f"
            % (code, entry_price, plan["entry_atr"])
        )


def check_atr_1450(context):
    today_text = context.current_dt.date().isoformat()
    _reset_sold_today(today_text)
    current_data = get_current_data()
    for code in list(context.portfolio.positions):
        if code in g.pending_sells or code in g.sold_today:
            continue
        state = g.position_states.get(code)
        if state is None:
            log.warning("[14:50 ATR blocked] %s missing_position_state" % code)
            continue
        quote = current_data[code]
        if getattr(quote, "paused", False):
            log.info("[14:50 ATR blocked] %s paused" % code)
            continue
        price = _number(getattr(quote, "last_price", None))
        stop = calc_frozen_atr_stop(state, code)
        if (
            not math.isfinite(price)
            or price <= 0.0
            or not math.isfinite(stop)
        ):
            log.warning("[14:50 ATR blocked] %s invalid_price_or_state" % code)
            continue
        log.info(
            "[14:50 ATR] %s entry_price=%.6f entry_atr=%.6f "
            "highest_close=%.6f stop=%.6f price=%.6f triggered=%s"
            % (
                code,
                state.entry_price,
                state.entry_atr,
                state.highest_close,
                stop,
                price,
                price <= stop,
            )
        )
        if price > stop:
            continue
        before_amount = _position_amount(context, code)
        order = order_target(code, 0)
        after_amount = _position_amount(context, code)
        outcome = _record_sell_submission(
            code,
            order,
            before_amount,
            after_amount,
        )
        log.info("[14:50 ATR result] %s outcome=%s" % (code, outcome))


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
    k_value = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d_value = k_value.ewm(com=m2 - 1, adjust=False).mean()
    return k_value, d_value


def calc_bollinger(close, period=20, std_mult=2.0):
    middle = close.rolling(period).mean()
    deviation = close.rolling(period).std()
    return (
        middle + std_mult * deviation,
        middle,
        middle - std_mult * deviation,
    )


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


def aggregate_completed_weeks(frame, decision_date):
    required = set(["open", "high", "low", "close"])
    if frame is None or not required.issubset(set(frame.columns)):
        raise ValueError("missing weekly OHLC data")
    work = frame.copy()
    if "date" in work.columns:
        dates = pd.to_datetime(work["date"], errors="coerce")
    else:
        dates = pd.Series(
            pd.to_datetime(work.index, errors="coerce"),
            index=work.index,
        )
    if dates.isna().any():
        raise ValueError("invalid weekly dates")
    decision = pd.Timestamp(decision_date).normalize()
    if pd.isna(decision):
        raise ValueError("invalid decision date")
    current_monday = decision - pd.Timedelta(days=decision.weekday())
    completed = dates < current_monday
    work = work.loc[completed].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "last_trade_date"]
        )
    work["date"] = dates.loc[completed]
    work = work.sort_values("date", kind="mergesort")
    work["week"] = work["date"].dt.to_period("W-SUN")
    grouped = work.groupby("week", sort=True)
    weeks = grouped.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    weeks["last_trade_date"] = grouped["date"].last()
    return weeks[["open", "high", "low", "close", "last_trade_date"]]


def _build_weekly_context(frame, decision_date):
    try:
        weeks = aggregate_completed_weeks(frame, decision_date)
    except Exception as exc:
        detail = " ".join(str(exc).split())[:160]
        reason = "invalid_weekly_data:%s" % exc.__class__.__name__
        if detail:
            reason = "%s:%s" % (reason, detail)
        return None, reason
    if len(weeks) < 21:
        return None, "insufficient_weekly_history"
    closes = pd.to_numeric(weeks["close"], errors="coerce")
    moving = closes.rolling(20).mean()
    values = {
        "weekly_close": _number(closes.iloc[-1]),
        "weekly_ma20": _number(moving.iloc[-1]),
        "weekly_ma20_prev": _number(moving.iloc[-2]),
        "weekly_period_end": weeks.index[-1].end_time.date().isoformat(),
        "weekly_last_trade_date": pd.Timestamp(
            weeks.iloc[-1]["last_trade_date"]
        ).date().isoformat(),
    }
    for key in ("weekly_close", "weekly_ma20", "weekly_ma20_prev"):
        if not math.isfinite(_number(values.get(key))):
            return None, "invalid_weekly_indicator"
    return values, None


def load_signal_snapshot(
    code,
    signal_date,
    decision_date,
    return_reason=False,
):
    try:
        frame = get_price(
            code,
            end_date=signal_date,
            count=LOOKBACK,
            frequency="daily",
            fields=["open", "high", "low", "close", "volume"],
            skip_paused=True,
            fq="pre",
            panel=False,
        )
        result, reason = _snapshot_from_daily_frame(
            code,
            signal_date,
            decision_date,
            frame,
        )
    except Exception as exc:
        result, reason = None, "exception:%s" % exc.__class__.__name__
    if return_reason:
        return result, reason
    return result


def _snapshot_from_daily_frame(code, signal_date, decision_date, frame):
    required = set(["open", "high", "low", "close", "volume"])
    if frame is None or not required.issubset(set(frame.columns)):
        return None, "missing_daily_data"
    try:
        work = frame.copy()
        work.index = pd.to_datetime(work.index, errors="coerce")
        if work.index.isna().any():
            return None, "invalid_daily_date"
        cutoff = pd.Timestamp(signal_date).normalize()
        decision = pd.Timestamp(decision_date).normalize()
        if pd.isna(cutoff) or pd.isna(decision) or cutoff >= decision:
            return None, "invalid_signal_boundary"
        work = work.loc[work.index.normalize() <= cutoff].sort_index()
        if work.empty or work.index[-1].date() != cutoff.date():
            return None, "stale_signal_date"
        if len(work) < 20:
            return None, "insufficient_history"
        recent_volume = pd.to_numeric(
            work["volume"], errors="coerce"
        ).iloc[-5:]
        if not math.isfinite(_number(recent_volume.sum())) or recent_volume.sum() <= 0:
            return None, "zero_recent_volume"

        weekly, weekly_reason = _build_weekly_context(work, decision)
        if weekly is None:
            return None, weekly_reason

        high = pd.to_numeric(work["high"], errors="coerce").reset_index(drop=True)
        low = pd.to_numeric(work["low"], errors="coerce").reset_index(drop=True)
        close = pd.to_numeric(work["close"], errors="coerce").reset_index(drop=True)
        rsi6 = calc_rsi(close, 6)
        k_value, d_value = calc_kdj(high, low, close, 9, 3, 3)
        upper, middle, lower = calc_bollinger(close, 20, 2.0)
        atr = calc_atr(high, low, close, 14)
        values = {
            "k_prev": k_value.iloc[-2],
            "d_prev": d_value.iloc[-2],
            "k": k_value.iloc[-1],
            "d": d_value.iloc[-1],
            "rsi6_prev": rsi6.iloc[-2],
            "rsi6": rsi6.iloc[-1],
            "close": close.iloc[-1],
            "boll_lower": lower.iloc[-1],
            "boll_mid": middle.iloc[-1],
            "boll_upper": upper.iloc[-1],
            "atr": atr.iloc[-1],
        }
        if not all(math.isfinite(_number(value)) for value in values.values()):
            return None, "invalid_indicator"
        if (
            _number(values["close"]) <= 0.0
            or _number(values["boll_lower"]) <= 0.0
            or _number(values["boll_mid"]) <= 0.0
            or _number(values["atr"]) <= 0.0
        ):
            return None, "invalid_indicator"
        result = dict(values)
        result.update(weekly)
        last_date = work.index[-1].date().isoformat()
        result.update(
            {
                "code": str(code),
                "signal_date": last_date,
                "max_data_date": last_date,
            }
        )
        return result, None
    except Exception as exc:
        return None, "exception:%s" % exc.__class__.__name__


def _weekly_values(snapshot):
    if not isinstance(snapshot, dict):
        return None
    values = tuple(
        _number(snapshot.get(key))
        for key in ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def weekly_entry_allowed(snapshot):
    values = _weekly_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close > weekly_ma20 > weekly_ma20_prev


def weekly_trend_broken(snapshot):
    values = _weekly_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close < weekly_ma20 < weekly_ma20_prev


def is_daily_entry_eligible(snapshot):
    if not isinstance(snapshot, dict):
        return False
    keys = (
        "close",
        "boll_lower",
        "boll_mid",
        "k_prev",
        "d_prev",
        "k",
        "d",
        "rsi6_prev",
        "rsi6",
    )
    values = dict((key, _number(snapshot.get(key))) for key in keys)
    if not all(math.isfinite(value) for value in values.values()):
        return False
    return bool(
        values["close"] > values["boll_lower"]
        and values["close"] <= values["boll_mid"]
        and values["k_prev"] <= values["d_prev"]
        and values["k"] > values["d"]
        and values["rsi6"] > values["rsi6_prev"]
        and values["rsi6"] <= 50.0
    )


def is_entry_eligible(snapshot):
    return weekly_entry_allowed(snapshot) and is_daily_entry_eligible(snapshot)


def calc_frozen_atr_stop(
    state,
    code="",
    multiplier=2.5,
    floor=0.05,
    cap=0.15,
):
    entry_price = _number(getattr(state, "entry_price", None))
    entry_atr = _number(getattr(state, "entry_atr", None))
    highest_close = _number(getattr(state, "highest_close", None))
    multiplier_value = _number(multiplier)
    floor_value = 0.03 if str(code).split(".")[0] == "518880" else _number(floor)
    cap_value = _number(cap)
    values = (
        entry_price,
        entry_atr,
        highest_close,
        multiplier_value,
        floor_value,
        cap_value,
    )
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    if (
        entry_price <= 0.0
        or entry_atr <= 0.0
        or highest_close <= 0.0
        or multiplier_value <= 0.0
        or floor_value < 0.0
        or cap_value <= 0.0
        or floor_value > cap_value
    ):
        return float("nan")
    distance = multiplier_value * entry_atr / entry_price
    distance = max(floor_value, min(cap_value, distance))
    return highest_close * (1.0 - distance)


def _kd_cross_down(snapshot):
    if not isinstance(snapshot, dict):
        return False
    values = tuple(
        _number(snapshot.get(key)) for key in ("k_prev", "d_prev", "k", "d")
    )
    if not all(math.isfinite(value) for value in values):
        return False
    k_prev, d_prev, k_value, d_value = values
    return k_prev >= d_prev and k_value < d_value


def choose_exit_reason(state, snapshot, current_price, hold_days, code=""):
    price = _number(current_price)
    stop = calc_frozen_atr_stop(state, code)
    if (
        math.isfinite(price)
        and price > 0.0
        and math.isfinite(stop)
        and price <= stop
    ):
        return "atr_stop"
    if weekly_trend_broken(snapshot):
        return "weekly_trend_break"
    try:
        held_sessions = int(hold_days)
    except (TypeError, ValueError, OverflowError):
        held_sessions = 0
    if held_sessions < 5 or not isinstance(snapshot, dict):
        return None
    close = _number(snapshot.get("close"))
    middle = _number(snapshot.get("boll_mid"))
    if (
        math.isfinite(close)
        and math.isfinite(middle)
        and close < middle
        and _kd_cross_down(snapshot)
    ):
        return "daily_pullback_failure"
    return None


def _date_text(value):
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _hold_days(entry_date, today, trade_days):
    days = [_date_text(value) for value in trade_days]
    try:
        return days.index(_date_text(today)) - days.index(_date_text(entry_date))
    except ValueError:
        return 0


def _update_highest_close_from_t1(state, close):
    close_value = _number(close)
    highest = _number(getattr(state, "highest_close", None))
    if math.isfinite(close_value) and close_value > 0.0:
        if not math.isfinite(highest) or highest <= 0.0 or close_value > highest:
            state.highest_close = close_value


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
    params = dict(params or get_default_params())
    sold = set(sold_today or set())
    held = list(held_codes)
    snapshot_map = dict((item["code"], item) for item in snapshots)
    plans = []
    planned_sells = set()

    for code in held:
        state = position_states.get(code)
        if state is None:
            continue
        snapshot = snapshot_map.get(code)
        if snapshot is not None:
            _update_highest_close_from_t1(state, snapshot.get("close"))
        reason = choose_exit_reason(
            state,
            snapshot if snapshot is not None else {},
            current_prices.get(code),
            _hold_days(state.entry_date, today, trade_days),
            code,
        )
        if reason is None:
            continue
        plans.append({"code": code, "target_value": 0.0, "reason": reason})
        planned_sells.add(code)

    held_after_sells = [code for code in held if code not in planned_sells]
    max_hold = int(params.get("max_hold", 3))
    slots = max_hold - len(held_after_sells)
    if slots <= 0:
        return plans

    pool_rank = dict((item["code"], rank) for rank, item in enumerate(snapshots))
    eligible = []
    for item in snapshots:
        code = item["code"]
        if code in held or code in sold or not is_entry_eligible(item):
            continue
        entry_atr = _number(item.get("atr"))
        weekly_close = _number(item.get("weekly_close"))
        weekly_ma20 = _number(item.get("weekly_ma20"))
        k_value = _number(item.get("k"))
        d_value = _number(item.get("d"))
        if (
            not math.isfinite(entry_atr)
            or entry_atr <= 0.0
            or not math.isfinite(weekly_close)
            or not math.isfinite(weekly_ma20)
            or weekly_ma20 == 0.0
            or not math.isfinite(k_value)
            or not math.isfinite(d_value)
        ):
            continue
        eligible.append(item)
    eligible.sort(
        key=lambda item: (
            -(
                _number(item["weekly_close"])
                / _number(item["weekly_ma20"])
                - 1.0
            ),
            -(_number(item["k"]) - _number(item["d"])),
            pool_rank[item["code"]],
        )
    )

    base_ratio = _number(params.get("base_ratio", 0.95))
    portfolio_value = _number(total_value)
    if (
        not math.isfinite(base_ratio)
        or base_ratio <= 0.0
        or not math.isfinite(portfolio_value)
        or portfolio_value <= 0.0
        or max_hold <= 0
    ):
        return plans
    target = portfolio_value * base_ratio / float(max_hold)
    for item in eligible[:slots]:
        plans.append(
            {
                "code": item["code"],
                "target_value": target,
                "reason": "weekly_pullback_entry",
                "entry_atr": _number(item["atr"]),
            }
        )
    return plans


def classify_sell_submission(order, before_amount, after_amount):
    before = max(0.0, _number(before_amount))
    after = max(0.0, _number(after_amount))
    filled = _number(getattr(order, "filled", 0)) if order is not None else 0.0
    if not math.isfinite(filled) or filled < 0.0:
        filled = 0.0
    if after <= 0.0 or (before > 0.0 and filled >= before):
        return "full"
    status = str(getattr(order, "status", "")).lower() if order is not None else ""
    active = _builtins.any(
        token in status for token in ("held", "pending", "open", "new")
    )
    partial = filled > 0.0 or (math.isfinite(before) and after < before)
    if partial:
        return "partial_pending" if active else "partial"
    if active:
        return "pending"
    return "rejected"


def _position_amount(context, code):
    position = context.portfolio.positions.get(code)
    if position is None:
        return 0.0
    amount = getattr(position, "total_amount", getattr(position, "amount", 0))
    value = _number(amount)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _record_sell_submission(code, order, before_amount, after_amount):
    outcome = classify_sell_submission(order, before_amount, after_amount)
    if not hasattr(g, "pending_sells"):
        g.pending_sells = set()
    if not hasattr(g, "sold_today"):
        g.sold_today = set()
    if outcome == "full":
        g.sold_today.add(code)
        g.pending_sells.discard(code)
        g.position_states.pop(code, None)
    elif outcome == "partial_pending":
        g.sold_today.add(code)
        g.pending_sells.add(code)
    elif outcome == "partial":
        g.sold_today.add(code)
        g.pending_sells.discard(code)
    elif outcome == "pending":
        g.pending_sells.add(code)
    else:
        g.pending_sells.discard(code)
    return outcome


def _reset_sold_today(today):
    if getattr(g, "sold_guard_date", None) != str(today):
        g.sold_guard_date = str(today)
        g.sold_today = set()
        g.pending_sells = set()
