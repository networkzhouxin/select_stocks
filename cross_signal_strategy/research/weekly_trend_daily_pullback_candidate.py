# -*- coding: utf-8 -*-
"""Platform-neutral weekly-trend/daily-pullback research candidate primitives."""

from __future__ import annotations

import math

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
