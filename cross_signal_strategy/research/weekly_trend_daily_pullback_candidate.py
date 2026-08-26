# -*- coding: utf-8 -*-
"""Platform-neutral weekly-trend/daily-pullback research candidate primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pandas as pd


VERSION = "weekly-trend-pullback-v0.1-research-candidate"

_OHLC_COLUMNS = ("open", "high", "low", "close")
_WEEKLY_COLUMNS = ("open", "high", "low", "close", "last_trade_date")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_snapshot_values(snapshot):
    if not isinstance(snapshot, dict):
        return None
    keys = ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    values = tuple(_number(snapshot.get(key)) for key in keys)
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def aggregate_completed_weeks(frame, decision_date):
    """Aggregate daily OHLC into calendar weeks strictly before decision week."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in _OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))

    work = frame.copy()
    raw_dates = work["date"] if "date" in work.columns else work.index
    dates = pd.Series(pd.to_datetime(raw_dates, errors="coerce"), index=work.index)
    if dates.isna().any():
        raise ValueError("date values must be valid timestamps")

    decision = pd.Timestamp(decision_date).normalize()
    if pd.isna(decision):
        raise ValueError("decision_date must be a valid timestamp")
    current_monday = decision - pd.Timedelta(days=decision.weekday())

    completed_mask = dates < current_monday
    work = work.loc[completed_mask].copy()
    if work.empty:
        return pd.DataFrame(columns=_WEEKLY_COLUMNS)

    work["date"] = dates.loc[completed_mask]
    work = work.sort_values("date", kind="stable")
    work["week"] = work["date"].dt.to_period("W-SUN")
    return work.groupby("week", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        last_trade_date=("date", "last"),
    )


def build_weekly_context(frame, decision_date):
    """Build the most recent completed-week MA20 context with explicit errors."""

    weeks = aggregate_completed_weeks(frame, decision_date)
    if len(weeks) < 21:
        return None, "insufficient_weekly_history"

    closes = pd.to_numeric(weeks["close"], errors="coerce")
    ma20 = closes.rolling(20).mean()
    values = {
        "weekly_close": _number(closes.iloc[-1]),
        "weekly_ma20": _number(ma20.iloc[-1]),
        "weekly_ma20_prev": _number(ma20.iloc[-2]),
        "weekly_period_end": weeks.index[-1].end_time.date().isoformat(),
        "weekly_last_trade_date": (
            pd.Timestamp(weeks.iloc[-1]["last_trade_date"]).date().isoformat()
        ),
    }
    numeric_keys = ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    if not all(math.isfinite(values[key]) for key in numeric_keys):
        return None, "invalid_weekly_indicator"
    return values, None


def weekly_entry_allowed(snapshot):
    """Return whether the completed-week close and MA20 are strictly rising."""

    values = _finite_snapshot_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close > weekly_ma20 > weekly_ma20_prev


def weekly_trend_broken(snapshot):
    """Return whether both completed-week close and MA20 strictly trend down."""

    values = _finite_snapshot_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close < weekly_ma20 < weekly_ma20_prev


def is_daily_entry_eligible(snapshot):
    """Return whether every frozen T-1 daily pullback condition is satisfied."""

    if not isinstance(snapshot, Mapping):
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
    values = {key: _number(snapshot.get(key)) for key in keys}
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
    """Combine the completed-week gate with all frozen daily entry conditions."""

    return weekly_entry_allowed(snapshot) and is_daily_entry_eligible(snapshot)


def build_buy_queue(snapshots, excluded_codes, etf_pool):
    """Filter eligible snapshots and apply the frozen stable ranking tuple."""

    pool_rank = {str(code): rank for rank, code in enumerate(etf_pool)}
    excluded = {str(code) for code in (excluded_codes or ())}
    eligible = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        code = str(snapshot.get("code", ""))
        if code not in pool_rank or code in excluded or not is_entry_eligible(snapshot):
            continue
        weekly_close = _number(snapshot.get("weekly_close"))
        weekly_ma20 = _number(snapshot.get("weekly_ma20"))
        k_value = _number(snapshot.get("k"))
        d_value = _number(snapshot.get("d"))
        if weekly_ma20 == 0.0:
            continue
        item = dict(snapshot)
        item["code"] = code
        item["weekly_strength"] = weekly_close / weekly_ma20 - 1.0
        item["kd_spread"] = k_value - d_value
        eligible.append(item)
    return sorted(
        eligible,
        key=lambda item: (
            -item["weekly_strength"],
            -item["kd_spread"],
            pool_rank[item["code"]],
        ),
    )


@dataclass
class PositionSignalState:
    entry_date: str
    entry_price: float
    entry_atr: float
    highest_close: float


def calc_frozen_atr_stop(
    state,
    code="",
    multiplier=2.5,
    floor=0.05,
    cap=0.15,
):
    """Calculate a close-anchored trailing stop from the frozen entry ATR."""

    entry_price = _number(getattr(state, "entry_price", math.nan))
    entry_atr = _number(getattr(state, "entry_atr", math.nan))
    highest_close = _number(getattr(state, "highest_close", math.nan))
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
        return math.nan
    if (
        entry_price <= 0.0
        or entry_atr <= 0.0
        or highest_close <= 0.0
        or multiplier_value <= 0.0
        or floor_value < 0.0
        or cap_value <= 0.0
        or floor_value > cap_value
    ):
        return math.nan
    distance = multiplier_value * entry_atr / entry_price
    distance = max(floor_value, min(cap_value, distance))
    return highest_close * (1.0 - distance)


def update_highest_close_from_t1(state, close):
    """Mutate highest_close only with a valid completed T-1 close."""

    close_value = _number(close)
    previous = _number(getattr(state, "highest_close", math.nan))
    if math.isfinite(close_value) and close_value > 0.0:
        if not math.isfinite(previous) or previous <= 0.0 or close_value > previous:
            state.highest_close = close_value
    return state


def _kd_cross_down(snapshot):
    if not isinstance(snapshot, Mapping):
        return False
    keys = ("k_prev", "d_prev", "k", "d")
    values = {key: _number(snapshot.get(key)) for key in keys}
    return bool(
        all(math.isfinite(value) for value in values.values())
        and values["k_prev"] >= values["d_prev"]
        and values["k"] < values["d"]
    )


def choose_exit_reason(state, snapshot, current_price, hold_days, code=""):
    """Choose exactly one exit in ATR, weekly-break, daily-failure priority."""

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
    if held_sessions < 5 or not isinstance(snapshot, Mapping):
        return None
    close = _number(snapshot.get("close"))
    boll_mid = _number(snapshot.get("boll_mid"))
    if (
        math.isfinite(close)
        and math.isfinite(boll_mid)
        and close < boll_mid
        and _kd_cross_down(snapshot)
    ):
        return "daily_pullback_failure"
    return None
