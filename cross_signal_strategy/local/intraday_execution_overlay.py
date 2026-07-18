# -*- coding: utf-8 -*-
"""Causal minute execution rules for the single ordinary-buy overlay."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TRAINING_START = pd.Timestamp("2019-01-01")
TRAINING_END = pd.Timestamp("2021-12-31")
ARRIVAL_TIME = "09:35"
DECISION_INTERVAL_MINUTES = 5
DECISION_CYCLES = 6
FALLBACK_TIME = "10:05"


@dataclass(frozen=True)
class BuyExecution:
    filled: bool
    route: str
    fill_time: str | None
    raw_price: float
    reason: str = ""


def choose_buy_execution(
    minute_frame: pd.DataFrame,
    trade_date: str,
    arrival_price: float,
) -> BuyExecution:
    """Choose one causal fill without using the 09:35 bar after the decision."""
    date = pd.Timestamp(trade_date)
    if date < TRAINING_START or date > TRAINING_END:
        raise ValueError("Execution date is outside 2019-2021 training window")
    if float(arrival_price) <= 0.0:
        return BuyExecution(False, "unfilled", None, 0.0, "invalid arrival price")

    required = {"date", "time", "low", "close"}
    missing = sorted(required.difference(minute_frame.columns))
    if missing:
        raise ValueError("Minute frame missing columns: %s" % ", ".join(missing))

    date_text = date.strftime("%Y-%m-%d")
    rows = minute_frame[minute_frame["date"].astype(str) == date_text].copy()
    if rows.empty:
        return BuyExecution(False, "unfilled", None, 0.0, "no rows for trade date")
    rows["_time"] = rows["time"].astype(str).str.slice(0, 5)
    rows = rows.sort_values("_time", kind="stable")

    passive_rows = rows[
        (rows["_time"] > ARRIVAL_TIME) & (rows["_time"] < FALLBACK_TIME)
    ]
    for _, row in passive_rows.iterrows():
        if not _bar_has_executable_trade(row):
            continue
        low = _numeric(row.get("low"))
        if low > 0.0 and low < float(arrival_price):
            return BuyExecution(
                True,
                "passive_limit",
                str(row["_time"]),
                float(arrival_price),
            )

    fallback_rows = rows[rows["_time"] >= FALLBACK_TIME]
    for _, row in fallback_rows.iterrows():
        if not _bar_has_executable_trade(row):
            continue
        price = _numeric(row.get("close"))
        if price > 0.0:
            return BuyExecution(
                True,
                "market_fallback",
                str(row["_time"]),
                price,
            )

    return BuyExecution(False, "unfilled", None, 0.0, "no executable fallback minute")


def _bar_has_executable_trade(row) -> bool:
    return _numeric(row.get("volume")) > 0.0 or _numeric(row.get("num_trades")) > 0.0


def _numeric(value) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return numeric if pd.notna(numeric) else 0.0
