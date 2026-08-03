"""Immutable domain contracts for the Binance Spot M1 foundation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import re

from binance_spot_strategy.protocols import Q18


_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}\Z")


class Symbol(str, Enum):
    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"


class BarInterval(str, Enum):
    FOUR_HOURS = "4h"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderState(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"


class ScopeType(str, Enum):
    TRAINING_STAGE = "training_stage"
    RESERVED_STAGE = "reserved_stage"
    PAPER_EPOCH = "paper_epoch"


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} has invalid identifier grammar")
    return value


def _require_enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be {enum_type.__name__}")


def _require_canonical_utc(value: object, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is not timezone.utc:
        raise ValueError(f"{name} must use timezone.utc")
    if value.microsecond % 1000 != 0:
        raise ValueError(f"{name} must be millisecond-aligned")
    return value


def _require_fingerprint(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    if _FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase 64-hex")
    return value


def _require_decimal(value: object, name: str) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def _require_q18(value: object, name: str) -> Q18:
    if type(value) is not Q18:
        raise TypeError(f"{name} must be Q18")
    return value


def canonical_utc_timestamp(value: datetime) -> str:
    """Format a canonical UTC datetime at second or millisecond precision."""

    checked = _require_canonical_utc(value, "value")
    timespec = "seconds" if checked.microsecond == 0 else "milliseconds"
    return checked.isoformat(timespec=timespec).removesuffix("+00:00") + "Z"


@dataclass(frozen=True, slots=True)
class TrainingStageId:
    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "TrainingStageId.value")


@dataclass(frozen=True, slots=True)
class StageActivationId:
    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "StageActivationId.value")


@dataclass(frozen=True, slots=True)
class PaperObservationEpochId:
    value: str

    def __post_init__(self) -> None:
        _require_identifier(self.value, "PaperObservationEpochId.value")


ScopeId = TrainingStageId | StageActivationId | PaperObservationEpochId


@dataclass(frozen=True, slots=True)
class DecisionKey:
    run_id: str
    scope_type: ScopeType
    scope_id: ScopeId
    candidate_strategy_fingerprint: str
    run_fingerprint: str
    bar_close_time: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id")
        _require_enum(self.scope_type, ScopeType, "scope_type")
        required_scope_id_type = {
            ScopeType.TRAINING_STAGE: TrainingStageId,
            ScopeType.RESERVED_STAGE: StageActivationId,
            ScopeType.PAPER_EPOCH: PaperObservationEpochId,
        }[self.scope_type]
        if type(self.scope_id) is not required_scope_id_type:
            raise TypeError(
                f"scope_id must be {required_scope_id_type.__name__} "
                f"for {self.scope_type.value}"
            )
        _require_fingerprint(
            self.candidate_strategy_fingerprint,
            "candidate_strategy_fingerprint",
        )
        _require_fingerprint(self.run_fingerprint, "run_fingerprint")
        _require_canonical_utc(self.bar_close_time, "bar_close_time")


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: Symbol
    interval: BarInterval
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        _require_enum(self.symbol, Symbol, "symbol")
        _require_enum(self.interval, BarInterval, "interval")
        _require_canonical_utc(self.open_time, "open_time")
        _require_canonical_utc(self.close_time, "close_time")
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        for name in ("open", "high", "low", "close"):
            if _require_decimal(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _require_decimal(self.volume, "volume") < 0:
            raise ValueError("volume must be nonnegative")
        if not (
            self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
        ):
            raise ValueError("OHLC values violate the high-low envelope")

    @property
    def identity(self) -> tuple[Symbol, BarInterval, datetime]:
        return (self.symbol, self.interval, self.open_time)


@dataclass(frozen=True, slots=True)
class SignalIntent:
    signal_intent_id: str
    symbol: Symbol
    side: Side
    bar_close_time: datetime
    reason_code: str
    risk_atr: Q18 | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.signal_intent_id, "signal_intent_id")
        _require_enum(self.symbol, Symbol, "symbol")
        _require_enum(self.side, Side, "side")
        _require_canonical_utc(self.bar_close_time, "bar_close_time")
        _require_identifier(self.reason_code, "reason_code")
        if self.side is Side.BUY:
            if self.risk_atr is None:
                raise ValueError("buy risk_atr must be present")
            atr = _require_q18(self.risk_atr, "risk_atr")
            if atr.value <= 0:
                raise ValueError("buy risk_atr must be positive")
        elif self.risk_atr is not None:
            raise ValueError("sell risk_atr must be None")


@dataclass(frozen=True, slots=True)
class DecisionPlan:
    decision_key: DecisionKey
    execution_group_id: str
    sell: SignalIntent | None = None
    buy: SignalIntent | None = None

    def __post_init__(self) -> None:
        if type(self.decision_key) is not DecisionKey:
            raise TypeError("decision_key must be DecisionKey")
        _require_identifier(self.execution_group_id, "execution_group_id")
        if self.sell is not None:
            if type(self.sell) is not SignalIntent:
                raise TypeError("sell must be SignalIntent or None")
            if self.sell.side is not Side.SELL:
                raise ValueError("sell slot requires a sell intent")
            if self.sell.bar_close_time != self.decision_key.bar_close_time:
                raise ValueError("sell bar_close_time must match decision key")
        if self.buy is not None:
            if type(self.buy) is not SignalIntent:
                raise TypeError("buy must be SignalIntent or None")
            if self.buy.side is not Side.BUY:
                raise ValueError("buy slot requires a buy intent")
            if self.buy.bar_close_time != self.decision_key.bar_close_time:
                raise ValueError("buy bar_close_time must match decision key")
        if (
            self.sell is not None
            and self.buy is not None
            and self.sell.symbol is self.buy.symbol
        ):
            raise ValueError("replacement buy must use a different symbol")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    execution_group_id: str
    symbol: Symbol
    side: Side
    state: OrderState
    quantity: Q18
    depends_on_intent_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.intent_id, "intent_id")
        _require_identifier(self.execution_group_id, "execution_group_id")
        _require_enum(self.symbol, Symbol, "symbol")
        _require_enum(self.side, Side, "side")
        _require_enum(self.state, OrderState, "state")
        quantity = _require_q18(self.quantity, "quantity")
        if quantity.value <= 0:
            raise ValueError("quantity must be positive")
        if self.depends_on_intent_id is not None:
            _require_identifier(
                self.depends_on_intent_id,
                "depends_on_intent_id",
            )


@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    execution_group_id: str
    order_intents: tuple[OrderIntent, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.execution_group_id, "execution_group_id")
        if type(self.order_intents) is not tuple:
            raise TypeError("order_intents must be tuple")
        if not all(
            type(intent) is OrderIntent for intent in self.order_intents
        ):
            raise TypeError("order_intents must contain only OrderIntent")
        if not all(
            intent.execution_group_id == self.execution_group_id
            for intent in self.order_intents
        ):
            raise ValueError("all order intents must match the group ID")
        if len(self.order_intents) == 1:
            standalone = self.order_intents[0]
            if (
                standalone.state is not OrderState.PENDING
                or standalone.depends_on_intent_id is not None
            ):
                raise ValueError("standalone order must be pending and independent")
            return
        if len(self.order_intents) == 2:
            parent, child = self.order_intents
            if not (
                parent.side is Side.SELL
                and parent.state is OrderState.PENDING
                and parent.intent_id != child.intent_id
                and parent.depends_on_intent_id is None
                and child.side is Side.BUY
                and child.state is OrderState.BLOCKED
                and child.depends_on_intent_id == parent.intent_id
            ):
                raise ValueError("replacement orders must be sell parent then buy child")
            return
        raise ValueError("execution group must contain one or two order intents")


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    intent_id: str
    symbol: Symbol
    side: Side
    quantity: Q18
    price: Q18
    commission: Q18
    event_time: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.fill_id, "fill_id")
        _require_identifier(self.intent_id, "intent_id")
        _require_enum(self.symbol, Symbol, "symbol")
        _require_enum(self.side, Side, "side")
        if _require_q18(self.quantity, "quantity").value <= 0:
            raise ValueError("quantity must be positive")
        if _require_q18(self.price, "price").value <= 0:
            raise ValueError("price must be positive")
        if _require_q18(self.commission, "commission").value < 0:
            raise ValueError("commission must be nonnegative")
        _require_canonical_utc(self.event_time, "event_time")


@dataclass(frozen=True, slots=True)
class Position:
    symbol: Symbol
    quantity: Q18
    average_cost: Q18

    def __post_init__(self) -> None:
        _require_enum(self.symbol, Symbol, "symbol")
        if _require_q18(self.quantity, "quantity").value <= 0:
            raise ValueError("quantity must be positive")
        if _require_q18(self.average_cost, "average_cost").value <= 0:
            raise ValueError("average_cost must be positive")


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    event_time: datetime
    cash: Q18
    position_value: Q18
    total_equity: Q18

    def __post_init__(self) -> None:
        _require_canonical_utc(self.event_time, "event_time")
        for name in ("cash", "position_value", "total_equity"):
            if _require_q18(getattr(self, name), name).value < 0:
                raise ValueError(f"{name} must be nonnegative")


__all__ = (
    "Bar",
    "BarInterval",
    "DecisionKey",
    "DecisionPlan",
    "EquitySnapshot",
    "ExecutionGroup",
    "Fill",
    "OrderIntent",
    "OrderState",
    "PaperObservationEpochId",
    "Position",
    "ScopeType",
    "Side",
    "SignalIntent",
    "StageActivationId",
    "Symbol",
    "TrainingStageId",
    "canonical_utc_timestamp",
)
