"""Order-free observer for the exact RSI(6) low-turn signal."""

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import math
from typing import Mapping

import numpy as np
import pandas as pd


VERSION = "rsi-low-turn-shadow-v0.1"


@dataclass(frozen=True)
class RsiTurnInput:
    code: str
    arrival_dt: datetime
    signal_date: date
    r2: float
    r1: float
    r0: float
    c1: float
    c0: float
    entry_open: float | None
    price_proved: bool
    price_reason: str | None = None
    background: Mapping[str, float] = field(default_factory=dict)
    source_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignalDecision:
    item: RsiTurnInput
    event_id: str
    signal_detected: bool
    valid_event: bool
    reasons: tuple[str, ...]


def calculate_rsi6(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 6, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1.0 / 6, min_periods=6).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    result[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return result


def event_id_for(item: RsiTurnInput) -> str:
    raw = "|".join(
        [VERSION, item.code, item.arrival_dt.date().isoformat(), item.signal_date.isoformat()]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def detect_rsi_low_turn(item: RsiTurnInput) -> SignalDecision:
    reasons = []
    if not all(math.isfinite(float(value)) for value in (item.r2, item.r1, item.r0, item.c1, item.c0)):
        reasons.append("non_finite_signal_value")
    else:
        if not item.r2 > item.r1:
            reasons.append("rsi_not_falling_into_trough")
        if not item.r0 > item.r1:
            reasons.append("rsi_not_turning_up")
        if not item.r1 <= 30.0:
            reasons.append("rsi_trough_not_oversold")
        if not item.c0 > item.c1:
            reasons.append("close_not_turning_up")
    signal = not reasons
    valid = signal and item.price_proved
    if signal and not item.price_proved:
        reasons.append(item.price_reason or "price_unproved")
    return SignalDecision(item, event_id_for(item), signal, valid, tuple(reasons))
