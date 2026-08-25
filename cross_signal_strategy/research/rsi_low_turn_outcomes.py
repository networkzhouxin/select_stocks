"""Fixed-horizon, order-free labels for the RSI low-turn shadow observer."""

from dataclasses import dataclass
from datetime import date, datetime
import calendar
import math
from statistics import median
from typing import Iterable, Mapping, Protocol

from cross_signal_strategy.local.local_backtester import LocalBroker
from cross_signal_strategy.research.rsi_low_turn_shadow import VERSION


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


@dataclass(frozen=True)
class GateDecision:
    status: str
    reasons: tuple[str, ...]
    metrics: Mapping[str, float | int]


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
    execute_target = getattr(broker, "order_target_value")
    buy = execute_target(code, SLOT_CAPITAL, entry_open, "shadow_entry")
    if not buy.filled:
        raise ValueError(f"shadow entry not executable: {buy.reason}")
    sell = execute_target(code, 0.0, exit_open, "shadow_exit")
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
        if (
            snapshot.status != "matured"
            or not _positive_finite(snapshot.exit_open)
            or not _available_by(snapshot.available_at, as_of)
        ):
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


def _available_by(available_at: object, as_of: datetime) -> bool:
    if not isinstance(available_at, datetime) or available_at.tzinfo is None:
        return False
    if available_at.utcoffset() is None:
        return False
    try:
        return available_at <= as_of
    except TypeError:
        return False


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial success rate."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def evaluate_gate(records: Iterable[EventOutcomeRecord]) -> GateDecision:
    """Apply the pre-registered evidence gate to mature shadow outcomes."""
    materialized = tuple(records)
    _validate_records(materialized)
    five_day = _matured_returns(materialized, 5, "doubled")
    ten_day = _matured_returns(materialized, 10, "doubled")
    five_returns = tuple(item[1] for item in five_day)
    ten_returns = tuple(item[1] for item in ten_day)
    metrics = _gate_metrics(five_day, ten_returns)

    if len(five_day) < 50:
        return GateDecision(
            "accumulating", ("fewer_than_50_matured_five_day_events",), metrics,
        )

    reasons = []
    if metrics["elapsed_calendar_months"] < 6:
        reasons.append("observation_span_under_six_months")
    if metrics["distinct_etf_count"] < 5:
        reasons.append("fewer_than_five_etfs")
    if metrics["max_single_etf_share"] > 0.40:
        reasons.append("single_etf_share_over_40_percent")
    if metrics["five_day_wilson_lower"] <= 0.50:
        reasons.append("five_day_wilson_lower_not_above_50_percent")
    if metrics["five_day_double_mean"] <= 0.0:
        reasons.append("five_day_double_mean_not_positive")
    if metrics["five_day_double_median"] <= 0.0:
        reasons.append("five_day_double_median_not_positive")
    if not ten_returns or metrics["ten_day_double_mean"] < 0.0:
        reasons.append("ten_day_double_mean_negative")
    if not ten_returns or metrics["ten_day_double_median"] < 0.0:
        reasons.append("ten_day_double_median_negative")
    if metrics["leave_top_winner_out_mean"] <= 0.0:
        reasons.append("leave_top_winner_out_mean_not_positive")
    return GateDecision("pass" if not reasons else "stop", tuple(reasons), metrics)


def build_summary(
    records: Iterable[EventOutcomeRecord],
    collection_start: date,
    generated_at: datetime,
    version: str = VERSION,
) -> dict[str, object]:
    """Build a JSON-ready, order-free summary of frozen shadow evidence."""
    if not isinstance(collection_start, date):
        raise TypeError("collection_start must be a date")
    if not isinstance(generated_at, datetime) or generated_at.tzinfo is None:
        raise ValueError("generated_at must be a timezone-aware datetime")
    if generated_at.utcoffset() is None:
        raise ValueError("generated_at must be a timezone-aware datetime")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty string")

    materialized = tuple(records)
    _validate_records(materialized)
    decision = evaluate_gate(materialized)
    five_day = _matured_returns(materialized, 5, "doubled")
    dates = [item[0].arrival_date for item in five_day]
    distribution = _etf_distribution(item[0] for item in five_day)
    return {
        "version": version,
        "collection_start": collection_start.isoformat(),
        "generated_at": generated_at.isoformat(),
        "counts": {
            "event_records": len(materialized),
            "matured_five_day_events": len(five_day),
            "matured_ten_day_events": len(_matured_returns(materialized, 10, "doubled")),
        },
        "date_span": {
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None,
            "elapsed_calendar_months": _completed_elapsed_calendar_months(dates),
        },
        "etf_distribution": distribution,
        "return_metrics": {
            str(horizon): {
                "nominal": _return_metrics(_matured_returns(materialized, horizon, "nominal")),
                "doubled": _return_metrics(_matured_returns(materialized, horizon, "doubled")),
            }
            for horizon in HORIZONS
        },
        "wilson_interval": {
            "lower": decision.metrics["five_day_wilson_lower"],
            "upper": decision.metrics["five_day_wilson_upper"],
        },
        "leave_top_winner_out_mean": decision.metrics["leave_top_winner_out_mean"],
        "status": decision.status,
        "reasons": list(decision.reasons),
    }


def _matured_returns(
    records: Iterable[EventOutcomeRecord], horizon: int, friction: str,
) -> tuple[tuple[EventOutcomeRecord, float], ...]:
    outcomes = []
    for record in records:
        label = record.labels.get(horizon)
        result = getattr(label, friction, None) if label is not None else None
        if label is not None and label.status == "matured" and result is not None:
            outcomes.append((record, result.net_return))
    return tuple(outcomes)


def _validate_records(records: Iterable[EventOutcomeRecord]) -> None:
    event_ids = set()
    for record in records:
        if record.event_id in event_ids:
            raise ValueError("duplicate event_id")
        event_ids.add(record.event_id)
        for horizon, label in record.labels.items():
            if label.horizon != horizon:
                raise ValueError("label horizon does not match mapping key")
            if label.event_id != record.event_id:
                raise ValueError("label event_id does not match record event_id")
            if label.status == "matured":
                _validate_matured_return(label.nominal)
                _validate_matured_return(label.doubled)


def _validate_matured_return(result: RoundTripResult | None) -> None:
    if result is None:
        return
    try:
        finite = math.isfinite(float(result.net_return))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("non-finite matured net_return")


def _gate_metrics(
    five_day: tuple[tuple[EventOutcomeRecord, float], ...], ten_returns: tuple[float, ...],
) -> dict[str, float | int]:
    five_returns = tuple(item[1] for item in five_day)
    wins = sum(value > 0.0 for value in five_returns)
    lower, upper = wilson_interval(wins, len(five_returns)) if five_returns else (0.0, 0.0)
    return {
        "matured_five_day_events": len(five_returns),
        "matured_ten_day_events": len(ten_returns),
        "elapsed_calendar_months": _completed_elapsed_calendar_months(
            [item[0].arrival_date for item in five_day]
        ),
        "distinct_etf_count": len(_etf_distribution(item[0] for item in five_day)),
        "max_single_etf_share": _max_etf_share(item[0] for item in five_day),
        "five_day_double_wins": wins,
        "five_day_wilson_lower": lower,
        "five_day_wilson_upper": upper,
        "five_day_double_mean": _mean_or_zero(five_returns),
        "five_day_double_median": _median_or_zero(five_returns),
        "ten_day_double_mean": _mean_or_zero(ten_returns),
        "ten_day_double_median": _median_or_zero(ten_returns),
        "leave_top_winner_out_mean": _leave_top_winner_out_mean(five_returns),
    }


def _return_metrics(
    outcomes: tuple[tuple[EventOutcomeRecord, float], ...],
) -> dict[str, float | int]:
    values = tuple(item[1] for item in outcomes)
    return {
        "count": len(values),
        "wins": sum(value > 0.0 for value in values),
        "mean": _mean_or_zero(values),
        "median": _median_or_zero(values),
    }


def _etf_distribution(records: Iterable[EventOutcomeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.code] = counts.get(record.code, 0) + 1
    return dict(sorted(counts.items()))


def _max_etf_share(records: Iterable[EventOutcomeRecord]) -> float:
    distribution = _etf_distribution(records)
    total = sum(distribution.values())
    return max(distribution.values()) / total if total else 0.0


def _completed_elapsed_calendar_months(dates: Iterable[date]) -> int:
    values = tuple(dates)
    if not values:
        return 0
    first, last = min(values), max(values)
    months = (last.year - first.year) * 12 + last.month - first.month
    if last < _add_calendar_months(first, months):
        months -= 1
    return months


def _add_calendar_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _mean_or_zero(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median_or_zero(values: tuple[float, ...]) -> float:
    return float(median(values)) if values else 0.0


def _leave_top_winner_out_mean(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    top = max(values)
    return (sum(values) - top) / (len(values) - 1)
