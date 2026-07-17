# -*- coding: utf-8 -*-
"""Training-only capital-utilization diagnostics for cross-signal replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.research.friction_diagnostics import PrecomputedSignalAdapter
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local.local_order_planner import (
    LocalCrossSignalOrderPlanner,
    strategy,
)
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"


@dataclass(frozen=True)
class ForwardReturnStats:
    observations: int = 0
    average_return: float = 0.0
    median_return: float = 0.0
    win_rate: float = 0.0


@dataclass(frozen=True)
class ShadowReasonStats:
    candidate_days: int = 0
    episodes: int = 0
    score_bands: Dict[str, int] = field(default_factory=dict)
    forward: Dict[int, ForwardReturnStats] = field(default_factory=dict)
    forward_by_score_band: Dict[str, Dict[int, ForwardReturnStats]] = field(default_factory=dict)


@dataclass(frozen=True)
class CapitalUtilizationReport:
    trading_days: int
    average_exposure: float
    average_cash_ratio: float
    position_count_days: Dict[int, int]
    exposure_by_position_count: Dict[int, float]
    vacant_slot_days: int
    total_vacant_slots: int
    vacant_slot_reasons: Dict[str, int]
    shadow_by_reason: Dict[str, ShadowReasonStats]


@dataclass(frozen=True)
class _ShadowCandidate:
    date: str
    code: str
    buy_score: float


@dataclass
class CapitalDiagnosticPlanner(LocalCrossSignalOrderPlanner):
    daily_scores: Dict[str, list[dict]] = field(default_factory=dict)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        self.daily_scores[str(current_date)] = [
            dict(score) for score in self.last_scores.values()
        ]
        return orders


def candidate_rejection_reason(score: Mapping[str, object], params=None) -> str:
    p = params or strategy.get_default_params()
    if not score.get("buy_allowed"):
        return "overheat"
    if float(score.get("buy_score", 0) or 0) < float(p["buy_threshold"]):
        return "below_buy_threshold"
    if float(score.get("sell_score", 0) or 0) >= float(p["sell_threshold"]):
        return "sell_conflict"
    if not strategy.has_new_buy_position(score, p):
        return "location_filter"
    if strategy.is_blocked_entry_combo(score):
        return "blocked_entry_combo"
    return "eligible_unfilled"


def build_capital_utilization_report(
    results: Iterable[object],
    daily_scores: Mapping[str, Sequence[Mapping[str, object]]],
    trade_dates: Sequence[str],
    entry_price_lookup,
    close_price_lookup,
    horizons: Sequence[int] = (5, 10, 20),
    max_hold: int = 3,
    params=None,
) -> CapitalUtilizationReport:
    days = list(results)
    dates = [str(date) for date in trade_dates]
    _assert_training_dates([str(day.date) for day in days] + dates + list(daily_scores.keys()))
    date_index = {date: index for index, date in enumerate(dates)}

    position_count_days: Dict[int, int] = {}
    exposure_sums: Dict[int, float] = {}
    exposure_observations: Dict[int, int] = {}
    exposure_ratios = []
    cash_ratios = []
    vacant_slot_days = 0
    total_vacant_slots = 0
    vacant_reasons: Dict[str, int] = {}
    shadow_records: Dict[str, list[_ShadowCandidate]] = {}

    for day in days:
        day_date = str(day.date)
        positions = getattr(day, "positions", {})
        position_count = len(positions)
        position_count_days[position_count] = position_count_days.get(position_count, 0) + 1

        total_value = float(day.total_value)
        marks = getattr(day, "marks", {})
        exposure_value = sum(
            int(position.amount) * float(marks.get(code, position.avg_cost))
            for code, position in positions.items()
        )
        exposure_ratio = exposure_value / total_value if total_value > 0 else 0.0
        cash_ratio = float(day.cash) / total_value if total_value > 0 else 0.0
        exposure_ratios.append(exposure_ratio)
        cash_ratios.append(cash_ratio)
        exposure_sums[position_count] = exposure_sums.get(position_count, 0.0) + exposure_ratio
        exposure_observations[position_count] = exposure_observations.get(position_count, 0) + 1

        vacant_slots = max(0, int(max_hold) - position_count)
        if vacant_slots <= 0:
            continue
        vacant_slot_days += 1
        total_vacant_slots += vacant_slots

        bought_codes = {
            str(order.code).split(".")[0]
            for order in getattr(day, "orders", [])
            if getattr(order, "filled", False) and int(order.amount_delta) > 0
        }
        held_codes = {str(code).split(".")[0] for code in positions}
        shadow_pool = []
        for raw_score in daily_scores.get(day_date, []):
            item = dict(raw_score)
            code = str(item.get("code", "")).split(".")[0]
            if not code or code in held_codes or code in bought_codes:
                continue
            if float(item.get("reversal_score", 0) or 0) <= 0:
                continue
            item["code"] = code
            shadow_pool.append(item)

        selected = strategy.sort_candidates(shadow_pool)[:vacant_slots]
        for item in selected:
            reason = candidate_rejection_reason(item, params=params)
            vacant_reasons[reason] = vacant_reasons.get(reason, 0) + 1
            shadow_records.setdefault(reason, []).append(
                _ShadowCandidate(
                    date=day_date,
                    code=item["code"],
                    buy_score=float(item.get("buy_score", 0) or 0),
                )
            )

        missing_slots = vacant_slots - len(selected)
        if missing_slots > 0:
            vacant_reasons["no_reversal_candidate"] = (
                vacant_reasons.get("no_reversal_candidate", 0) + missing_slots
            )

    shadow_by_reason = {
        reason: _shadow_reason_stats(
            records,
            dates,
            date_index,
            entry_price_lookup,
            close_price_lookup,
            horizons,
        )
        for reason, records in sorted(shadow_records.items())
    }
    exposure_by_count = {
        count: exposure_sums[count] / exposure_observations[count]
        for count in sorted(exposure_sums)
    }
    return CapitalUtilizationReport(
        trading_days=len(days),
        average_exposure=sum(exposure_ratios) / len(exposure_ratios) if exposure_ratios else 0.0,
        average_cash_ratio=sum(cash_ratios) / len(cash_ratios) if cash_ratios else 0.0,
        position_count_days=dict(sorted(position_count_days.items())),
        exposure_by_position_count=exposure_by_count,
        vacant_slot_days=vacant_slot_days,
        total_vacant_slots=total_vacant_slots,
        vacant_slot_reasons=dict(sorted(vacant_reasons.items())),
        shadow_by_reason=shadow_by_reason,
    )


def run_training_capital_utilization(
    loader=None,
    initial_cash: float = 20000.0,
    horizons: Sequence[int] = (5, 10, 20),
) -> CapitalUtilizationReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source_adapter = build_training_signal_adapter(loader)
    initial_planner = LocalCrossSignalOrderPlanner(source_adapter, trade_dates=trade_dates)
    cached_adapter = PrecomputedSignalAdapter.from_source(
        source_adapter,
        trade_dates=trade_dates,
        codes=initial_planner.etf_pool,
    )
    planner = CapitalDiagnosticPlanner(
        cached_adapter,
        etf_pool=initial_planner.etf_pool,
        trade_dates=trade_dates,
    )
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)

    def entry_price(code: str, date: str) -> float:
        return float(loader.get_minute_bar(code, date, "09:35")["close"])

    def close_price(code: str, date: str) -> float:
        frame = loader.load_daily_frame(code, date)
        rows = frame[frame["date"].astype(str) == str(date)]
        if rows.empty:
            raise KeyError("No daily close for %s %s" % (code, date))
        return float(rows.iloc[0]["close"])

    return build_capital_utilization_report(
        results=results,
        daily_scores=planner.daily_scores,
        trade_dates=trade_dates,
        entry_price_lookup=entry_price,
        close_price_lookup=close_price,
        horizons=horizons,
        max_hold=int(planner.params["max_hold"]),
        params=planner.params,
    )


def _shadow_reason_stats(
    records: Sequence[_ShadowCandidate],
    trade_dates: Sequence[str],
    date_index: Mapping[str, int],
    entry_price_lookup,
    close_price_lookup,
    horizons: Sequence[int],
) -> ShadowReasonStats:
    episodes = _collapse_shadow_episodes(records, date_index)
    returns = _collect_forward_returns(
        episodes,
        trade_dates,
        date_index,
        entry_price_lookup,
        close_price_lookup,
        horizons,
    )
    by_band: Dict[str, list[_ShadowCandidate]] = {}
    for candidate in episodes:
        by_band.setdefault(_score_band(candidate.buy_score), []).append(candidate)

    return ShadowReasonStats(
        candidate_days=len(records),
        episodes=len(episodes),
        score_bands=_score_bands(episodes),
        forward={
            horizon: _forward_return_stats(values)
            for horizon, values in sorted(returns.items())
        },
        forward_by_score_band={
            band: {
                horizon: _forward_return_stats(values)
                for horizon, values in sorted(
                    _collect_forward_returns(
                        items,
                        trade_dates,
                        date_index,
                        entry_price_lookup,
                        close_price_lookup,
                        horizons,
                    ).items()
                )
            }
            for band, items in sorted(by_band.items())
        },
    )


def _collect_forward_returns(
    records: Sequence[_ShadowCandidate],
    trade_dates: Sequence[str],
    date_index: Mapping[str, int],
    entry_price_lookup,
    close_price_lookup,
    horizons: Sequence[int],
) -> Dict[int, list[float]]:
    returns: Dict[int, list[float]] = {int(horizon): [] for horizon in horizons}
    for candidate in records:
        entry_date = candidate.date
        code = candidate.code
        try:
            entry_price = float(entry_price_lookup(code, entry_date))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if entry_price <= 0 or entry_date not in date_index:
            continue
        entry_index = date_index[entry_date]
        for horizon in returns:
            future_index = entry_index + horizon
            if future_index >= len(trade_dates):
                continue
            future_date = trade_dates[future_index]
            try:
                future_price = float(close_price_lookup(code, future_date))
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if future_price > 0:
                returns[horizon].append(future_price / entry_price - 1.0)
    return returns


def _collapse_shadow_episodes(
    records: Sequence[_ShadowCandidate],
    date_index: Mapping[str, int],
) -> list[_ShadowCandidate]:
    episodes = []
    last_index_by_code: Dict[str, int] = {}
    ordered = sorted(records, key=lambda item: (date_index.get(item.date, 10**9), item.code))
    for candidate in ordered:
        current_index = date_index.get(candidate.date)
        if current_index is None:
            continue
        previous_index = last_index_by_code.get(candidate.code)
        if previous_index is None or current_index > previous_index + 1:
            episodes.append(candidate)
        last_index_by_code[candidate.code] = current_index
    return episodes


def _score_bands(records: Sequence[_ShadowCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in records:
        band = _score_band(candidate.buy_score)
        counts[band] = counts.get(band, 0) + 1
    return dict(sorted(counts.items()))


def _score_band(score: float) -> str:
    if score >= 50:
        return "50-59"
    if score >= 40:
        return "40-49"
    if score >= 30:
        return "30-39"
    if score >= 20:
        return "20-29"
    return "below_20"


def _forward_return_stats(values: Sequence[float]) -> ForwardReturnStats:
    return ForwardReturnStats(
        observations=len(values),
        average_return=sum(values) / len(values) if values else 0.0,
        median_return=float(median(values)) if values else 0.0,
        win_rate=sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    )


def _assert_training_dates(dates: Iterable[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Capital diagnostics contain dates outside 2019-2021 training window")


def format_capital_utilization(report: CapitalUtilizationReport) -> str:
    lines = [
        "Cross-signal training capital utilization (2019-2021)",
        "days={} exposure={:.3f} cash={:.3f} vacant_days={} vacant_slots={}".format(
            report.trading_days,
            report.average_exposure,
            report.average_cash_ratio,
            report.vacant_slot_days,
            report.total_vacant_slots,
        ),
        "position_count_days=%s" % report.position_count_days,
        "vacant_slot_reasons=%s" % report.vacant_slot_reasons,
    ]
    for reason, stats in report.shadow_by_reason.items():
        parts = []
        for horizon, forward in stats.forward.items():
            parts.append(
                "{}d n={} avg={:.2%} median={:.2%} win={:.2%}".format(
                    horizon,
                    forward.observations,
                    forward.average_return,
                    forward.median_return,
                    forward.win_rate,
                )
            )
        lines.append(
            "{} candidate_days={} episodes={} bands={} {}".format(
                reason,
                stats.candidate_days,
                stats.episodes,
                stats.score_bands,
                " | ".join(parts),
            )
        )
        for band, band_forward in stats.forward_by_score_band.items():
            band_parts = []
            for horizon, forward in band_forward.items():
                band_parts.append(
                    "{}d n={} avg={:.2%} median={:.2%} win={:.2%}".format(
                        horizon,
                        forward.observations,
                        forward.average_return,
                        forward.median_return,
                        forward.win_rate,
                    )
                )
            lines.append("  band={} {}".format(band, " | ".join(band_parts)))
    return "\n".join(lines)


def main() -> None:
    print(format_capital_utilization(run_training_capital_utilization()))


if __name__ == "__main__":
    main()
