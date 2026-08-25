"""Fixed-horizon, order-free labels for the RSI low-turn shadow observer."""

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Mapping, Protocol

from cross_signal_strategy.local.local_backtester import LocalBroker


HORIZONS = (1, 3, 5, 10)
SLOT_CAPITAL = 20000.0 * 0.95 / 3.0


@dataclass(frozen=True)
class Friction:
    commission_rate: float
    min_commission: float
    slippage_rate: float


@dataclass(frozen=True)
class RoundTripResult:
    amount: int
    buy_exec_price: float
    sell_exec_price: float
    buy_commission: float
    sell_commission: float
    net_pnl: float
    net_return: float


@dataclass(frozen=True)
class FutureSnapshot:
    horizon: int
    status: str
    exit_open: float | None
    mfe: float | None
    mae: float | None
    available_at: datetime | None


@dataclass(frozen=True)
class MaturedLabel:
    event_id: str
    horizon: int
    status: str
    exit_price: float | None
    nominal: RoundTripResult | None
    doubled: RoundTripResult | None
    mfe: float | None
    mae: float | None


@dataclass(frozen=True)
class EventOutcomeRecord:
    event_id: str
    code: str
    arrival_date: date
    labels: Mapping[int, MaturedLabel]


class FuturePriceSource(Protocol):
    def snapshot_for(
        self, event: Mapping[str, object], horizon: int, as_of: datetime,
    ) -> FutureSnapshot:
        """Return only prices already executable and available at ``as_of``."""


NOMINAL_FRICTION = Friction(0.0003, 5.0, 0.001)
DOUBLED_FRICTION = Friction(0.0006, 10.0, 0.002)


def calculate_round_trip(
    code: str, entry_open: float, exit_open: float, friction: Friction,
) -> RoundTripResult:
    """Apply the existing LocalBroker model to one fixed-horizon shadow trade."""
    broker = LocalBroker(
        SLOT_CAPITAL,
        commission_rate=friction.commission_rate,
        min_commission=friction.min_commission,
        slippage_rate=friction.slippage_rate,
    )
    buy = broker.order_target_value(code, SLOT_CAPITAL, entry_open, "shadow_entry")
    if not buy.filled:
        raise ValueError(f"shadow entry not executable: {buy.reason}")
    sell = broker.order_target_value(code, 0.0, exit_open, "shadow_exit")
    if not sell.filled:
        raise ValueError(f"shadow exit not executable: {sell.reason}")
    pnl = broker.cash - SLOT_CAPITAL
    return RoundTripResult(
        buy.amount_delta,
        buy.exec_price,
        sell.exec_price,
        buy.commission,
        sell.commission,
        pnl,
        pnl / SLOT_CAPITAL,
    )


def mature_event_labels(
    event: Mapping[str, object], source: FuturePriceSource, as_of: datetime,
) -> tuple[MaturedLabel, ...]:
    """Return one append-ready label per frozen horizon without price substitution."""
    event_id, code, entry_open = _event_identity(event)
    labels = []
    for horizon in HORIZONS:
        snapshot = source.snapshot_for(event, horizon, as_of)
        if snapshot.horizon != horizon:
            raise ValueError("future source returned a snapshot for the wrong horizon")
        if snapshot.status != "matured" or not _positive_finite(snapshot.exit_open):
            status = (
                snapshot.status
                if snapshot.status != "matured"
                else "pending_missing_executable_price"
            )
            labels.append(MaturedLabel(
                event_id, horizon, status, None, None, None,
                snapshot.mfe, snapshot.mae,
            ))
            continue
        exit_price = float(snapshot.exit_open)
        labels.append(MaturedLabel(
            event_id,
            horizon,
            "matured",
            exit_price,
            calculate_round_trip(code, entry_open, exit_price, NOMINAL_FRICTION),
            calculate_round_trip(code, entry_open, exit_price, DOUBLED_FRICTION),
            snapshot.mfe,
            snapshot.mae,
        ))
    return tuple(labels)


def _event_identity(event: Mapping[str, object]) -> tuple[str, str, float]:
    event_id = event.get("event_id")
    code = event.get("code")
    entry_open = event.get("entry_open")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    if not isinstance(code, str) or not code:
        raise ValueError("event code must be a non-empty string")
    if not _positive_finite(entry_open):
        raise ValueError("event entry_open must be a positive finite number")
    return event_id, code, float(entry_open)


def _positive_finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0
