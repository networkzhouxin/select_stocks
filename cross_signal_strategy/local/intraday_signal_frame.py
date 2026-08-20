# -*- coding: utf-8 -*-
"""Build a causal provisional T-day bar for the fixed 14:45 candidate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DAILY_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
MINUTE_COLUMNS = {
    "date",
    "time",
    "prev_close",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


@dataclass(frozen=True)
class IntradayFrameAudit:
    decision_time: str
    data_cutoff: str
    last_minute: str
    minute_count: int
    partial_volume: bool = True


@dataclass(frozen=True)
class IntradaySignalFrame:
    frame: pd.DataFrame
    audit: IntradayFrameAudit


def build_intraday_signal_frame(
    t1_daily_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    trade_date: str,
    decision_time: str = "14:45",
) -> IntradaySignalFrame:
    """Append one T-day bar made only from interval-start labels before 14:45."""
    if str(decision_time)[:5] != "14:45":
        raise ValueError("Only the pre-registered 14:45 decision is allowed")
    _require_columns(t1_daily_frame, DAILY_COLUMNS)
    _require_columns(minute_frame, MINUTE_COLUMNS)

    trade_day = pd.Timestamp(trade_date).normalize()
    daily = t1_daily_frame.copy()
    daily_dates = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    if daily.empty or (daily_dates >= trade_day).any():
        raise ValueError("T-1 frame must contain completed dates before T only")
    if daily_dates.duplicated().any() or not daily_dates.is_monotonic_increasing:
        raise ValueError("T-1 dates must be unique and ordered")

    minutes = minute_frame.copy()
    trade_date_text = trade_day.strftime("%Y-%m-%d")
    if set(minutes["date"].astype(str)) != {trade_date_text}:
        raise ValueError("Minute frame must contain exactly one requested trade date")
    timestamps = pd.to_datetime(
        minutes["date"].astype(str) + " " + minutes["time"].astype(str),
        errors="raise",
    )
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and ordered")

    cutoff = trade_day + pd.Timedelta(hours=14, minutes=45)
    visible = minutes.loc[timestamps < cutoff].copy()
    if visible.empty:
        raise ValueError("No completed minute before 14:45")

    numeric = visible[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    prices = numeric[["open", "high", "low", "close"]]
    invalid = (
        numeric.isna().any().any()
        or (prices <= 0).any().any()
        or (numeric["volume"] < 0).any()
        or (numeric["high"] < prices[["open", "close", "low"]].max(axis=1)).any()
        or (numeric["low"] > prices[["open", "close", "high"]].min(axis=1)).any()
    )
    if invalid:
        raise ValueError("Invalid point-in-time OHLCV")

    previous_closes = pd.to_numeric(visible["prev_close"], errors="coerce")
    daily_close = float(daily.iloc[-1]["close"])
    if previous_closes.isna().any() or any(
        round(float(value), 3) != round(daily_close, 3)
        for value in previous_closes
    ):
        raise ValueError("Daily/minute adjustment boundary mismatch")

    partial = {
        "date": trade_date_text,
        "open": float(numeric.iloc[0]["open"]),
        "high": float(numeric["high"].max()),
        "low": float(numeric["low"].min()),
        "close": float(numeric.iloc[-1]["close"]),
        "volume": float(numeric["volume"].sum()),
    }
    combined = pd.concat([daily, pd.DataFrame([partial])], ignore_index=True)
    return IntradaySignalFrame(
        frame=combined,
        audit=IntradayFrameAudit(
            decision_time="14:45",
            data_cutoff="14:44",
            last_minute=str(visible.iloc[-1]["time"])[:5],
            minute_count=len(visible),
        ),
    )


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
