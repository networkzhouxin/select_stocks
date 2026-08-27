# -*- coding: utf-8 -*-
"""Frozen observation for MACD-late, BOLL-upper official buy events."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.research.order_path_diagnostics import (
    OrderPathEvent,
    parse_joinquant_filled_order_events,
)


MINIMUM_TOTAL_EVENTS = 3
MINIMUM_DISTINCT_YEARS = 2
_PRIOR_CROSS_AGES = {1, 2}


@dataclass(frozen=True)
class LateMacdBollEvent:
    date: str
    code: str
    signal_date: str
    close: float
    boll_upper: float
    rsi_cross_age: int
    kdj_cross_age: int
    macd_cross_age: int


@dataclass(frozen=True)
class LateMacdBollObservation:
    total_filled_buys: int
    matched_events: tuple[LateMacdBollEvent, ...]
    distinct_years: tuple[int, ...]
    gate_passed: bool


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _active_age(snapshot: dict[str, Any], flags: Sequence[str]) -> int | None:
    ages = []
    for flag in flags:
        if not bool(snapshot.get(flag)):
            continue
        age = snapshot.get(f"{flag}_age")
        if isinstance(age, bool) or age not in (0, 1, 2):
            continue
        ages.append(int(age))
    eligible = [age for age in ages if age in _PRIOR_CROSS_AGES]
    return min(eligible) if eligible else None


def _rsi_direction_is_up(snapshot: dict[str, Any]) -> bool:
    up = bool(
        snapshot.get("rsi6_cross_rsi12_up")
        or snapshot.get("rsi6_cross_rsi24_up")
    )
    down = bool(
        snapshot.get("rsi6_cross_rsi12_down")
        or snapshot.get("rsi6_cross_rsi24_down")
    )
    return up and not down


def _matched_ages(snapshot: dict[str, Any]) -> tuple[int, int, int] | None:
    if not bool(snapshot.get("macd_cross_up")):
        return None
    macd_age = snapshot.get("macd_cross_up_age")
    if isinstance(macd_age, bool) or macd_age != 0:
        return None
    if not _rsi_direction_is_up(snapshot):
        return None
    rsi_age = _active_age(snapshot, (
        "rsi6_cross_rsi12_up",
        "rsi6_cross_rsi24_up",
    ))
    kdj_age = _active_age(snapshot, (
        "kdj_k_cross_up",
        "kdj_j_cross_up",
    ))
    if rsi_age is None or kdj_age is None:
        return None
    return rsi_age, kdj_age, 0


def is_late_macd_boll_upper_entry(snapshot: dict[str, Any]) -> bool:
    """Return whether one T-1 snapshot matches the exact frozen veto shape."""
    ages = _matched_ages(snapshot)
    close = _finite_float(snapshot.get("close"))
    upper = _finite_float(snapshot.get("boll_upper"))
    return bool(
        ages is not None
        and close is not None
        and upper is not None
        and close >= upper
    )


def decode_joinquant_log_bytes(raw: bytes) -> str:
    """Preserve ASCII order records when a JoinQuant log mixes encodings."""
    return bytes(raw).decode("utf-8", errors="replace")


def _assert_causal_snapshot(event: OrderPathEvent, snapshot: dict[str, Any]) -> None:
    signal_date = str(snapshot.get("signal_date", ""))
    max_data_date = str(snapshot.get("max_data_date", ""))
    if not signal_date or not max_data_date:
        raise ValueError("snapshot must disclose signal_date and max_data_date")
    if max_data_date > signal_date or signal_date >= event.date:
        raise ValueError(
            "future data in observation snapshot: event=%s signal=%s max=%s"
            % (event.date, signal_date, max_data_date)
        )


def observe_official_filled_buys(
    events: Sequence[OrderPathEvent],
    score_provider: Callable[[str, str], dict[str, Any] | None],
) -> LateMacdBollObservation:
    """Score only official filled buys and apply the pre-registered sample gate."""
    buys = [event for event in events if event.side == "BUY"]
    matched = []
    for event in buys:
        snapshot = score_provider(event.code, event.date)
        if snapshot is None:
            continue
        _assert_causal_snapshot(event, snapshot)
        if not is_late_macd_boll_upper_entry(snapshot):
            continue
        rsi_age, kdj_age, macd_age = _matched_ages(snapshot)  # type: ignore[misc]
        matched.append(LateMacdBollEvent(
            date=event.date,
            code=event.code,
            signal_date=str(snapshot["signal_date"]),
            close=float(snapshot["close"]),
            boll_upper=float(snapshot["boll_upper"]),
            rsi_cross_age=rsi_age,
            kdj_cross_age=kdj_age,
            macd_cross_age=macd_age,
        ))
    years = tuple(sorted({int(event.date[:4]) for event in matched}))
    return LateMacdBollObservation(
        total_filled_buys=len(buys),
        matched_events=tuple(matched),
        distinct_years=years,
        gate_passed=(
            len(matched) >= MINIMUM_TOTAL_EVENTS
            and len(years) >= MINIMUM_DISTINCT_YEARS
        ),
    )


def run_official_buy_observation(
    log_path: str | Path,
    training_root: str | Path = APPROVED_TRAINING_ROOT,
    warmup_root: str | Path = APPROVED_WARMUP_ROOT,
) -> LateMacdBollObservation:
    """Run the frozen read-only observation against an official JoinQuant log."""
    from cross_signal_strategy.local_training_run import build_training_signal_adapter

    log_text = decode_joinquant_log_bytes(Path(log_path).read_bytes())
    events = parse_joinquant_filled_order_events(log_text)
    loader = CrossSignalTrainingDataLoader(training_root)
    adapter = build_training_signal_adapter(loader, warmup_root=warmup_root)
    return observe_official_filled_buys(
        events,
        lambda code, date: adapter.score(code, date),
    )


def format_observation(result: LateMacdBollObservation) -> str:
    lines = [
        "LATE_MACD_BOLL_UPPER_OBSERVATION",
        "official_filled_buys=%d" % result.total_filled_buys,
        "matched_events=%d" % len(result.matched_events),
        "distinct_years=%s" % ",".join(str(year) for year in result.distinct_years),
        "gate=%s" % ("PASS" if result.gate_passed else "STOP"),
    ]
    for event in result.matched_events:
        lines.append(
            "%s %s signal=%s close=%.6f upper=%.6f rsi_age=%d kdj_age=%d macd_age=%d"
            % (
                event.date,
                event.code,
                event.signal_date,
                event.close,
                event.boll_upper,
                event.rsi_cross_age,
                event.kdj_cross_age,
                event.macd_cross_age,
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    args = parser.parse_args()
    print(format_observation(run_official_buy_observation(args.log_path)))


if __name__ == "__main__":
    main()
