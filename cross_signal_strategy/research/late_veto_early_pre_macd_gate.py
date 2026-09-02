# -*- coding: utf-8 -*-
"""Exact paired-result gate for the frozen stacked JoinQuant candidate."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Mapping


class GateKind(str, Enum):
    TRAINING_NOMINAL = "training_nominal"
    TRAINING_DOUBLE_FRICTION = "training_double_friction"
    VALIDATION_2022_2023 = "validation_2022_2023"
    VALIDATION_2024_LATEST = "validation_2024_latest"
    VALIDATION_2015_2018 = "validation_2015_2018"
    VALIDATION_2010_2014 = "validation_2010_2014"
    FULL_PERIOD = "full_period"


@dataclass(frozen=True)
class RunConfig:
    strategy_version: str
    build_id: str
    fingerprint: str
    start_date: str
    end_date: str
    initial_cash: Decimal
    frequency: str
    execution_time: str
    commission_rate: Decimal
    minimum_commission: Decimal
    slippage_rate: Decimal
    friction_profile: str
    log_sha256: str
    trade_export_sha256: str


@dataclass(frozen=True)
class RunMetrics:
    total_return: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    closed_trade_count: int
    profit_loss_ratio: Decimal | None
    annual_returns: Mapping[int, Decimal]
    positive_to_negative_round_trips: int | None
    early_fills: int
    early_fill_years: tuple[int, ...]


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    metrics: RunMetrics


@dataclass(frozen=True)
class PairedRun:
    kind: GateKind
    baseline: RunResult
    candidate: RunResult


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


FORMAL_IDENTITY = ("cross-v0.3.3", "20260822.2", "77e44d93d255")
CANDIDATE_IDENTITY = (
    "cross-v0.3.3-late-veto-early-pre-macd-candidate",
    "20260822.3-candidate",
    "f6b08195dd3d",
)
EXPECTED_WINDOWS = {
    GateKind.TRAINING_NOMINAL: ("2019-01-01", "2021-12-31", "nominal"),
    GateKind.TRAINING_DOUBLE_FRICTION: (
        "2019-01-01", "2021-12-31", "double"
    ),
    GateKind.VALIDATION_2022_2023: (
        "2022-01-01", "2023-12-31", "nominal"
    ),
    GateKind.VALIDATION_2015_2018: (
        "2015-01-01", "2018-12-31", "nominal"
    ),
    GateKind.VALIDATION_2010_2014: (
        "2010-01-01", "2014-12-31", "nominal"
    ),
}
EXPECTED_COSTS = {
    "nominal": (Decimal("0.0003"), Decimal("5"), Decimal("0.001")),
    "double": (Decimal("0.0006"), Decimal("10"), Decimal("0.002")),
}


def load_paired_run(path: str | Path) -> PairedRun:
    """Load one paired run without converting metrics through binary floats."""
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_float=Decimal,
        parse_int=Decimal,
    )
    pair = PairedRun(
        kind=GateKind(str(payload["kind"])),
        baseline=_load_run(payload["baseline"]),
        candidate=_load_run(payload["candidate"]),
    )
    _validate_pair(pair)
    return pair


def evaluate_pair(pair: PairedRun) -> GateDecision:
    """Apply the frozen nondegradation gate without numeric tolerances."""
    baseline = pair.baseline.metrics
    candidate = pair.candidate.metrics
    reasons = []
    if candidate.total_return < baseline.total_return:
        reasons.append("candidate return is lower than paired baseline")
    if candidate.max_drawdown > baseline.max_drawdown:
        reasons.append("candidate drawdown is higher than paired baseline")
    required_delta = (
        Decimal("0.03")
        if pair.kind in {GateKind.TRAINING_NOMINAL, GateKind.FULL_PERIOD}
        else Decimal("0")
    )
    if candidate.win_rate - baseline.win_rate < required_delta:
        reasons.append("candidate win-rate improvement is below required delta")
    if pair.kind is GateKind.TRAINING_NOMINAL:
        reasons.extend(_training_reasons(baseline, candidate))
    return GateDecision(passed=not reasons, reasons=tuple(reasons))


def _load_run(payload: Mapping[str, object]) -> RunResult:
    config = payload["config"]
    metrics = payload["metrics"]
    profit_loss_ratio = metrics.get("profit_loss_ratio")
    positive_to_negative = metrics.get("positive_to_negative_round_trips")
    annual = {
        int(year): _decimal(value)
        for year, value in metrics.get("annual_returns", {}).items()
    }
    return RunResult(
        config=RunConfig(
            strategy_version=str(config["strategy_version"]),
            build_id=str(config["build_id"]),
            fingerprint=str(config["fingerprint"]),
            start_date=str(config["start_date"]),
            end_date=str(config["end_date"]),
            initial_cash=_decimal(config["initial_cash"]),
            frequency=str(config["frequency"]),
            execution_time=str(config["execution_time"]),
            commission_rate=_decimal(config["commission_rate"]),
            minimum_commission=_decimal(config["minimum_commission"]),
            slippage_rate=_decimal(config["slippage_rate"]),
            friction_profile=str(config["friction_profile"]),
            log_sha256=str(config["log_sha256"]),
            trade_export_sha256=str(config["trade_export_sha256"]),
        ),
        metrics=RunMetrics(
            total_return=_decimal(metrics["total_return"]),
            max_drawdown=_decimal(metrics["max_drawdown"]),
            win_rate=_decimal(metrics["win_rate"]),
            closed_trade_count=_nonnegative_int(
                metrics["closed_trade_count"], "closed_trade_count"
            ),
            profit_loss_ratio=(
                None if profit_loss_ratio is None else _decimal(profit_loss_ratio)
            ),
            annual_returns=MappingProxyType(annual),
            positive_to_negative_round_trips=(
                None
                if positive_to_negative is None
                else _nonnegative_int(
                    positive_to_negative, "positive_to_negative_round_trips"
                )
            ),
            early_fills=_nonnegative_int(metrics.get("early_fills", 0), "early_fills"),
            early_fill_years=tuple(
                _nonnegative_int(year, "early_fill_year")
                for year in metrics.get("early_fill_years", ())
            ),
        ),
    )


def _training_reasons(
    baseline: RunMetrics,
    candidate: RunMetrics,
) -> list[str]:
    reasons = []
    if candidate.win_rate <= Decimal("0.558"):
        reasons.append("candidate win rate does not beat standalone late-veto")
    if candidate.early_fills < 3:
        reasons.append("candidate has fewer than three early fills")
    if len(set(candidate.early_fill_years)) < 2:
        reasons.append("candidate early fills do not span two years")
    if candidate.profit_loss_ratio is None or candidate.profit_loss_ratio < Decimal("3"):
        reasons.append("candidate profit/loss ratio is below 3")
    baseline_round_trips = baseline.positive_to_negative_round_trips
    candidate_round_trips = candidate.positive_to_negative_round_trips
    if baseline_round_trips is None or candidate_round_trips is None:
        reasons.append("positive-to-negative round-trip evidence is missing")
    elif candidate_round_trips > baseline_round_trips:
        reasons.append("candidate positive-to-negative round trips increased")
    for year in (2019, 2020, 2021):
        annual_return = candidate.annual_returns.get(year)
        if annual_return is None or annual_return <= 0:
            reasons.append("candidate annual return is not positive for %d" % year)
    return reasons


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _nonnegative_int(value: object, field: str) -> int:
    number = _decimal(value)
    if (
        not number.is_finite()
        or number < 0
        or number != number.to_integral_value()
    ):
        raise ValueError("%s must be a nonnegative integer" % field)
    return int(number)


def _validate_pair(pair: PairedRun) -> None:
    _validate_identity(pair.baseline.config, FORMAL_IDENTITY, "baseline")
    _validate_identity(pair.candidate.config, CANDIDATE_IDENTITY, "candidate")
    _validate_hashes(pair.baseline.config, "baseline")
    _validate_hashes(pair.candidate.config, "candidate")
    _validate_metrics(pair.baseline.metrics, "baseline")
    _validate_metrics(pair.candidate.metrics, "candidate")

    paired_fields = (
        "start_date",
        "end_date",
        "initial_cash",
        "frequency",
        "execution_time",
        "commission_rate",
        "minimum_commission",
        "slippage_rate",
        "friction_profile",
    )
    for field in paired_fields:
        if getattr(pair.baseline.config, field) != getattr(pair.candidate.config, field):
            raise ValueError("paired run configuration differs for %s" % field)

    config = pair.baseline.config
    if config.initial_cash != Decimal("20000"):
        raise ValueError("initial_cash must remain 20000")
    if config.frequency != "daily":
        raise ValueError("frequency must remain daily")
    if config.execution_time != "09:35":
        raise ValueError("execution_time must remain 09:35")
    _validate_window(pair.kind, config)
    expected_costs = EXPECTED_COSTS[config.friction_profile]
    actual_costs = (
        config.commission_rate,
        config.minimum_commission,
        config.slippage_rate,
    )
    if actual_costs != expected_costs:
        raise ValueError("cost values do not match friction_profile")


def _validate_identity(
    config: RunConfig,
    expected: tuple[str, str, str],
    label: str,
) -> None:
    actual = (config.strategy_version, config.build_id, config.fingerprint)
    if actual != expected:
        raise ValueError("%s strategy identity does not match frozen source" % label)


def _validate_hashes(config: RunConfig, label: str) -> None:
    for field in ("log_sha256", "trade_export_sha256"):
        value = getattr(config, field)
        if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("%s %s must be a complete SHA-256" % (label, field))


def _validate_metrics(metrics: RunMetrics, label: str) -> None:
    if not metrics.total_return.is_finite():
        raise ValueError("%s total_return must be finite" % label)
    if (
        metrics.profit_loss_ratio is not None
        and not metrics.profit_loss_ratio.is_finite()
    ):
        raise ValueError("%s profit_loss_ratio must be finite" % label)
    for year, value in metrics.annual_returns.items():
        if not value.is_finite():
            raise ValueError("%s annual return for %d must be finite" % (label, year))
    for field in ("win_rate", "max_drawdown"):
        value = getattr(metrics, field)
        if not value.is_finite() or not Decimal("0") <= value <= Decimal("1"):
            raise ValueError("%s %s must be between 0 and 1" % (label, field))


def _validate_window(kind: GateKind, config: RunConfig) -> None:
    if kind in EXPECTED_WINDOWS:
        expected = EXPECTED_WINDOWS[kind]
        actual = (config.start_date, config.end_date, config.friction_profile)
        if actual != expected:
            raise ValueError("run window or friction profile does not match gate kind")
        return
    if kind is GateKind.VALIDATION_2024_LATEST:
        if config.start_date != "2024-01-01" or config.friction_profile != "nominal":
            raise ValueError("2024-latest validation must start 2024-01-01 nominal")
        return
    if kind is GateKind.FULL_PERIOD:
        if config.start_date > "2010-01-01" or config.friction_profile != "nominal":
            raise ValueError("full-period run must start by 2010-01-01 and be nominal")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("INVALID: expected exactly one paired-result JSON path")
        return 2
    try:
        pair = load_paired_run(args[0])
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        print("INVALID: %s" % exc)
        return 2
    decision = evaluate_pair(pair)
    if decision.passed:
        print("PASS")
        return 0
    print("FAIL")
    for reason in decision.reasons:
        print("- %s" % reason)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
