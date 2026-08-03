"""Frozen common configuration for the isolated Binance Spot M1 foundation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum


class StageName(StrEnum):
    """Named historical evaluation stages."""

    TRAINING = "training"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


def _require_utc(value: datetime, name: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be exact datetime")
    if value.tzinfo is not timezone.utc:
        raise ValueError(f"{name} must use timezone.utc")


@dataclass(frozen=True, slots=True)
class StageWindow:
    """A half-open historical stage window with a fixed bar warm-up."""

    name: StageName
    start: datetime
    end: datetime
    warmup_bars: int = 540

    def __post_init__(self) -> None:
        if type(self.name) is not StageName:
            raise TypeError("name must be StageName")
        _require_utc(self.start, "start")
        _require_utc(self.end, "end")
        if self.end <= self.start:
            raise ValueError("stage end must be after start")
        if type(self.warmup_bars) is not int:
            raise TypeError("warmup_bars must be exact int")
        if self.warmup_bars != 540:
            raise ValueError("warmup_bars must equal 540")

    def admits(self, open_time: datetime, close_time: datetime) -> bool:
        _require_utc(open_time, "open_time")
        _require_utc(close_time, "close_time")
        if close_time < open_time:
            raise ValueError("close_time must not precede open_time")
        return self.start <= open_time < self.end and close_time < self.end


SYMBOLS = ("BTCUSDT", "ETHUSDT")
BAR_INTERVAL = "4h"
FORMAL_STARTING_BALANCE = Decimal("500.00")
STAGE_WINDOWS = (
    StageWindow(
        StageName.TRAINING,
        datetime(2018, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 1, 1, tzinfo=timezone.utc),
    ),
    StageWindow(
        StageName.VALIDATION,
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    ),
    StageWindow(
        StageName.HOLDOUT,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    ),
)
SLIPPAGE_RATE = Decimal("0.0005")
COMMISSION_RATE = Decimal("0.001")
RISK_BUDGET = Decimal("0.01")
ALLOCATION_CAP = Decimal("0.30")
ATR_MULTIPLIER = Decimal("2.5")
STOP_FLOOR = Decimal("0.05")
STOP_CAP = Decimal("0.15")
DESIGN_REVISION_SHA256 = "18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0"


__all__ = (
    "ALLOCATION_CAP",
    "ATR_MULTIPLIER",
    "BAR_INTERVAL",
    "COMMISSION_RATE",
    "DESIGN_REVISION_SHA256",
    "FORMAL_STARTING_BALANCE",
    "RISK_BUDGET",
    "SLIPPAGE_RATE",
    "STOP_CAP",
    "STOP_FLOOR",
    "STAGE_WINDOWS",
    "SYMBOLS",
    "StageName",
    "StageWindow",
)
