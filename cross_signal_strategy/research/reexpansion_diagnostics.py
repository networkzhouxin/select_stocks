# -*- coding: utf-8 -*-
"""Training-only same-side contraction/re-expansion diagnostics.

This module studies a fixed shape that is deliberately kept separate from a
true cross:

1. fast and slow remain on the same side for three observations;
2. their gap contracts for one observation;
3. the gap then expands again while the fast line moves in that direction.

Future prices are used only as offline outcome labels. They never participate
in event detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cross_signal_strategy.local.local_adjustment import (
    default_training_adjustment_factors,
    default_training_daily_corrections,
)
from cross_signal_strategy.local.local_data_loader import (
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_signal_adapter import strategy
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
WARMUP_START = "2018-01-01"
FORWARD_HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON = 5
MIN_TOTAL_OBSERVATIONS = 30
MIN_ANNUAL_OBSERVATIONS = 5
VALID_DIRECTIONS = {"bullish", "bearish"}
VALID_GROUPS = {"novel_reexpansion", "existing_cross"}


def classify_same_side_reexpansion(fast, slow) -> str:
    """Classify the latest fixed three-point same-side shape."""
    fast_values = np.asarray(getattr(fast, "values", fast), dtype=float)
    slow_values = np.asarray(getattr(slow, "values", slow), dtype=float)
    if len(fast_values) < 3 or len(slow_values) < 3:
        return "no_data"
    fast_tail = fast_values[-3:]
    slow_tail = slow_values[-3:]
    if not np.isfinite(fast_tail).all() or not np.isfinite(slow_tail).all():
        return "no_data"

    gap_older, gap_contracted, gap_latest = fast_tail - slow_tail
    if (
        gap_older > gap_contracted > 0
        and gap_latest > gap_contracted
        and fast_tail[-1] > fast_tail[-2]
    ):
        return "bullish"
    if (
        gap_older < gap_contracted < 0
        and gap_latest < gap_contracted
        and fast_tail[-1] < fast_tail[-2]
    ):
        return "bearish"
    return "none"


def build_reexpansion_flags(
    frame: pd.DataFrame,
    signal_date: str,
    params: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Calculate fixed re-expansion labels using data through signal_date."""
    required = {"date", "close", "high", "low"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "re-expansion frame missing columns: %s" % ", ".join(missing)
        )
    visible = frame.copy()
    visible["_date"] = pd.to_datetime(visible["date"], errors="coerce")
    if visible["_date"].isna().any():
        raise ValueError("re-expansion frame contains invalid dates")
    signal_ts = pd.Timestamp(signal_date)
    if not visible.empty and visible["_date"].max() > signal_ts:
        raise ValueError("re-expansion frame contains data after signal_date")
    visible = visible.loc[visible["_date"] <= signal_ts].sort_values("_date")
    if visible.empty or visible["_date"].max() != signal_ts:
        raise ValueError("re-expansion frame does not end on signal_date")

    p = dict(params or strategy.get_default_params())
    close = pd.to_numeric(visible["close"], errors="coerce")
    high = pd.to_numeric(visible["high"], errors="coerce")
    low = pd.to_numeric(visible["low"], errors="coerce")
    rsi6 = strategy.calc_rsi(close, int(p["rsi_fast"]))
    rsi12 = strategy.calc_rsi(close, int(p["rsi_mid"]))
    rsi24 = strategy.calc_rsi(close, int(p["rsi_slow"]))
    dif, dea, _ = strategy.calc_macd(
        close,
        int(p["macd_fast"]),
        int(p["macd_slow"]),
        int(p["macd_signal"]),
    )
    k, d, j = strategy.calc_kdj(
        high,
        low,
        close,
        int(p["kdj_n"]),
        int(p["kdj_m1"]),
        int(p["kdj_m2"]),
    )
    return {
        "rsi6_rsi12": classify_same_side_reexpansion(rsi6, rsi12),
        "rsi6_rsi24": classify_same_side_reexpansion(rsi6, rsi24),
        "macd": classify_same_side_reexpansion(dif, dea),
        "kdj_k": classify_same_side_reexpansion(k, d),
        "kdj_j": classify_same_side_reexpansion(j, d),
    }


@dataclass(frozen=True)
class ReexpansionObservation:
    code: str
    execution_date: str
    signal_date: str
    direction: str
    group: str
    indicators: tuple[str, ...]
    forward_returns: Mapping[int, float]


@dataclass(frozen=True)
class ReexpansionStats:
    observations: int = 0
    average_return: float = 0.0
    median_return: float = 0.0
    directional_success_rate: float = 0.0


@dataclass(frozen=True)
class ReexpansionGateDecision:
    passed: bool
    primary_horizon: int = PRIMARY_HORIZON
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReexpansionReport:
    event_count: int
    by_direction_group_horizon: Dict[str, ReexpansionStats]
    by_year_direction_group_horizon: Dict[str, ReexpansionStats]
    by_indicator_direction_group_horizon: Dict[str, ReexpansionStats]
    gate: ReexpansionGateDecision


def build_reexpansion_report(
    observations: Iterable[ReexpansionObservation],
) -> ReexpansionReport:
    """Aggregate the fixed event study and evaluate its pre-registered gate."""
    items = list(observations)
    _assert_observation_dates(items)
    for item in items:
        if item.direction not in VALID_DIRECTIONS:
            raise ValueError("unsupported re-expansion direction: %s" % item.direction)
        if item.group not in VALID_GROUPS:
            raise ValueError("unsupported re-expansion group: %s" % item.group)

    by_group: Dict[str, ReexpansionStats] = {}
    by_year: Dict[str, ReexpansionStats] = {}
    by_indicator: Dict[str, ReexpansionStats] = {}
    for direction in sorted(VALID_DIRECTIONS):
        for group in sorted(VALID_GROUPS):
            selected = [
                item for item in items
                if item.direction == direction and item.group == group
            ]
            for horizon in FORWARD_HORIZONS:
                key = "%s:%s:%d" % (direction, group, horizon)
                by_group[key] = _stats(selected, horizon, direction)
                for year in (2019, 2020, 2021):
                    annual = [
                        item for item in selected
                        if item.execution_date.startswith(str(year))
                    ]
                    annual_key = "%d:%s" % (year, key)
                    by_year[annual_key] = _stats(annual, horizon, direction)

    indicators = sorted({
        indicator
        for item in items
        for indicator in item.indicators
    })
    for indicator in indicators:
        for direction in sorted(VALID_DIRECTIONS):
            for group in sorted(VALID_GROUPS):
                selected = [
                    item for item in items
                    if indicator in item.indicators
                    and item.direction == direction
                    and item.group == group
                ]
                for horizon in FORWARD_HORIZONS:
                    key = "%s:%s:%s:%d" % (
                        indicator,
                        direction,
                        group,
                        horizon,
                    )
                    by_indicator[key] = _stats(selected, horizon, direction)

    gate = evaluate_reexpansion_gate(by_group, by_year)
    return ReexpansionReport(
        event_count=len(items),
        by_direction_group_horizon=by_group,
        by_year_direction_group_horizon=by_year,
        by_indicator_direction_group_horizon=by_indicator,
        gate=gate,
    )


def evaluate_reexpansion_gate(
    by_group: Mapping[str, ReexpansionStats],
    by_year: Mapping[str, ReexpansionStats],
) -> ReexpansionGateDecision:
    """Require sample size and five-day dominance in both directions."""
    reasons = []
    horizon = PRIMARY_HORIZON
    for direction in ("bullish", "bearish"):
        novel_key = "%s:novel_reexpansion:%d" % (direction, horizon)
        cross_key = "%s:existing_cross:%d" % (direction, horizon)
        novel = by_group.get(novel_key, ReexpansionStats())
        cross = by_group.get(cross_key, ReexpansionStats())
        for group_name, stats in (
            ("novel_reexpansion", novel),
            ("existing_cross", cross),
        ):
            if stats.observations < MIN_TOTAL_OBSERVATIONS:
                reasons.append(
                    "%s %s has fewer than %d five-day observations"
                    % (direction, group_name, MIN_TOTAL_OBSERVATIONS)
                )
        _append_comparison_reasons(reasons, direction, "aggregate", novel, cross)

        for year in (2019, 2020, 2021):
            annual_novel = by_year.get(
                "%d:%s" % (year, novel_key),
                ReexpansionStats(),
            )
            annual_cross = by_year.get(
                "%d:%s" % (year, cross_key),
                ReexpansionStats(),
            )
            for group_name, stats in (
                ("novel_reexpansion", annual_novel),
                ("existing_cross", annual_cross),
            ):
                if stats.observations < MIN_ANNUAL_OBSERVATIONS:
                    reasons.append(
                        "%d %s %s has fewer than %d five-day observations"
                        % (
                            year,
                            direction,
                            group_name,
                            MIN_ANNUAL_OBSERVATIONS,
                        )
                    )
            _append_comparison_reasons(
                reasons,
                direction,
                str(year),
                annual_novel,
                annual_cross,
            )
    return ReexpansionGateDecision(
        passed=not reasons,
        primary_horizon=horizon,
        reasons=tuple(reasons),
    )


def run_training_reexpansion_observation(
    loader=None,
) -> ReexpansionReport:
    """Run the fixed event study on approved 2019-2021 data only."""
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    params = strategy.get_default_params()
    pool = [str(code).split(".")[0] for code in strategy.get_default_etf_pool()]
    date_index = {date: index for index, date in enumerate(trade_dates)}
    candidates = []

    for execution_date in trade_dates:
        for code in pool:
            score = adapter.score(code, execution_date)
            if score is None:
                continue
            frame, signal_date = adapter.load_signal_frame(code, execution_date)
            flags = build_reexpansion_flags(frame, signal_date, params)
            for direction in ("bullish", "bearish"):
                pattern_indicators = tuple(sorted(
                    name for name, label in flags.items() if label == direction
                ))
                cross_indicators = _active_cross_indicators(score, direction)
                if cross_indicators:
                    candidates.append((
                        code,
                        execution_date,
                        signal_date,
                        direction,
                        "existing_cross",
                        cross_indicators,
                    ))
                elif pattern_indicators:
                    candidates.append((
                        code,
                        execution_date,
                        signal_date,
                        direction,
                        "novel_reexpansion",
                        pattern_indicators,
                    ))

    candidates = _collapse_consecutive_events(candidates, date_index)
    observations = []
    factors = default_training_adjustment_factors()
    corrections = default_training_daily_corrections()
    for code, execution_date, signal_date, direction, group, indicators in candidates:
        try:
            minute = loader.get_minute_bar(code, execution_date, "09:35")
            entry_price = float(minute["close"])
            entry_volume = float(minute.get("volume", 0) or 0)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if entry_price <= 0 or entry_volume <= 0:
            continue

        forward_returns = {}
        execution_index = date_index[execution_date]
        for horizon in FORWARD_HORIZONS:
            future_index = execution_index + horizon
            if future_index >= len(trade_dates):
                continue
            future_date = trade_dates[future_index]
            try:
                forward_returns[horizon] = _forward_adjusted_return(
                    loader=loader,
                    factors=factors,
                    corrections=corrections,
                    code=code,
                    execution_date=execution_date,
                    future_date=future_date,
                    entry_price=entry_price,
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
        observations.append(ReexpansionObservation(
            code=code,
            execution_date=execution_date,
            signal_date=str(signal_date),
            direction=direction,
            group=group,
            indicators=tuple(indicators),
            forward_returns=forward_returns,
        ))
    return build_reexpansion_report(observations)


def format_reexpansion_report(report: ReexpansionReport) -> str:
    lines = [
        "Cross-signal same-side re-expansion observation (2019-2021)",
        "events=%d primary_horizon=%dd gate=%s"
        % (
            report.event_count,
            report.gate.primary_horizon,
            "PASS" if report.gate.passed else "FAIL",
        ),
    ]
    for key, stats in report.by_direction_group_horizon.items():
        lines.append(
            "GROUP %s n=%d avg=%.4f median=%.4f directional_win=%.4f"
            % (
                key,
                stats.observations,
                stats.average_return,
                stats.median_return,
                stats.directional_success_rate,
            )
        )
    for key, stats in report.by_year_direction_group_horizon.items():
        if key.endswith(":%d" % PRIMARY_HORIZON):
            lines.append(
                "YEAR %s n=%d avg=%.4f median=%.4f directional_win=%.4f"
                % (
                    key,
                    stats.observations,
                    stats.average_return,
                    stats.median_return,
                    stats.directional_success_rate,
                )
            )
    for key, stats in report.by_indicator_direction_group_horizon.items():
        if key.endswith(":%d" % PRIMARY_HORIZON):
            lines.append(
                "INDICATOR %s n=%d avg=%.4f median=%.4f directional_win=%.4f"
                % (
                    key,
                    stats.observations,
                    stats.average_return,
                    stats.median_return,
                    stats.directional_success_rate,
                )
            )
    for reason in report.gate.reasons:
        lines.append("GATE_FAIL %s" % reason)
    return "\n".join(lines)


def _active_cross_indicators(
    score: Mapping[str, object],
    direction: str,
) -> tuple[str, ...]:
    suffix = "up" if direction == "bullish" else "down"
    fields = {
        "rsi6_rsi12": "rsi6_cross_rsi12_%s" % suffix,
        "rsi6_rsi24": "rsi6_cross_rsi24_%s" % suffix,
        "macd": "macd_cross_%s" % suffix,
        "kdj_k": "kdj_k_cross_%s" % suffix,
        "kdj_j": "kdj_j_cross_%s" % suffix,
    }
    return tuple(sorted(
        name for name, field in fields.items() if bool(score.get(field))
    ))


def _collapse_consecutive_events(
    candidates: Sequence[tuple],
    date_index: Mapping[str, int],
) -> list[tuple]:
    result = []
    last_index: Dict[tuple[str, str, str], int] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            date_index.get(item[1], 10**9),
            item[0],
            item[3],
            item[4],
        ),
    ):
        code, execution_date, _, direction, group, _ = candidate
        index = date_index.get(execution_date)
        if index is None:
            continue
        key = (code, direction, group)
        previous = last_index.get(key)
        if previous is None or index > previous + 1:
            result.append(candidate)
        last_index[key] = index
    return result


def _forward_adjusted_return(
    *,
    loader,
    factors,
    corrections,
    code: str,
    execution_date: str,
    future_date: str,
    entry_price: float,
) -> float:
    frame = loader.load_daily_frame(code, future_date)
    frame = corrections.apply_daily_frame(frame, code)
    rows = frame.loc[frame["date"].astype(str) == str(future_date)]
    if rows.empty:
        raise KeyError("No daily close for %s %s" % (code, future_date))
    future_close = float(rows.iloc[-1]["close"])
    if not isfinite(future_close) or future_close <= 0:
        raise ValueError("Invalid future close for %s %s" % (code, future_date))
    comparable = pd.DataFrame({
        "date": [execution_date, future_date],
        "close": [float(entry_price), future_close],
    })
    adjusted = factors.adjust_daily_frame(
        comparable,
        code,
        current_date=future_date,
    )
    adjusted_entry = float(adjusted.iloc[0]["close"])
    adjusted_future = float(adjusted.iloc[1]["close"])
    if adjusted_entry <= 0:
        raise ValueError("Invalid adjusted entry price")
    return adjusted_future / adjusted_entry - 1.0


def _stats(
    observations: Sequence[ReexpansionObservation],
    horizon: int,
    direction: str,
) -> ReexpansionStats:
    values = [
        float(item.forward_returns[horizon])
        for item in observations
        if horizon in item.forward_returns
        and isfinite(float(item.forward_returns[horizon]))
    ]
    if direction == "bullish":
        success = sum(1 for value in values if value > 0)
    else:
        success = sum(1 for value in values if value < 0)
    return ReexpansionStats(
        observations=len(values),
        average_return=sum(values) / len(values) if values else 0.0,
        median_return=float(median(values)) if values else 0.0,
        directional_success_rate=success / len(values) if values else 0.0,
    )


def _append_comparison_reasons(
    reasons: list[str],
    direction: str,
    scope: str,
    novel: ReexpansionStats,
    cross: ReexpansionStats,
) -> None:
    if direction == "bullish":
        if novel.average_return <= cross.average_return:
            reasons.append(
                "%s %s novel average return does not beat existing crosses"
                % (scope, direction)
            )
    elif novel.average_return >= cross.average_return:
        reasons.append(
            "%s %s novel average return is not below existing crosses"
            % (scope, direction)
        )
    if novel.directional_success_rate <= cross.directional_success_rate:
        reasons.append(
            "%s %s novel directional win rate does not beat existing crosses"
            % (scope, direction)
        )


def _assert_observation_dates(
    observations: Iterable[ReexpansionObservation],
) -> None:
    for item in observations:
        execution_date = str(item.execution_date)
        signal_date = str(item.signal_date)
        if execution_date < TRAINING_START or execution_date > TRAINING_END:
            raise ValueError(
                "Re-expansion observation contains dates outside 2019-2021 training window"
            )
        if signal_date < WARMUP_START or signal_date > TRAINING_END:
            raise ValueError(
                "Re-expansion signal date is outside approved 2018 warm-up and 2019-2021 training data"
            )
        if signal_date >= execution_date:
            raise ValueError(
                "Re-expansion signal date must be earlier than execution date"
            )


if __name__ == "__main__":
    print(format_reexpansion_report(run_training_reexpansion_observation()))
