# -*- coding: utf-8 -*-
"""Isolated training candidate for unified KDJ extreme-zone score bonuses.

The adapter reads only the official causal T-1 snapshot.  It adds five points
to the final buy score when K <= 20 unless downside continuation is active,
and five points to the final sell score when K >= 80.  No cross is required.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)


OVERSOLD_K_MAX = 20.0
OVERBOUGHT_K_MIN = 80.0
EXTREME_ZONE_POINTS = 5.0
BUY_THRESHOLD = 60.0
SELL_THRESHOLD = 30.0
TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
DOUBLE_FRICTION = {
    "commission_rate": 0.0006,
    "min_commission": 10.0,
    "slippage_rate": 0.002,
}


@dataclass(frozen=True)
class ExtremeZoneScoreAdapter:
    """Decorate an official score while preserving its T-1 evidence."""

    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        base_result = self.source.score(
            code,
            current_date,
            return_reason=return_reason,
        )
        if return_reason:
            base_score, reason = base_result
            if base_score is None:
                return None, reason
            return self._adjust(base_score), reason
        if base_result is None:
            return None
        return self._adjust(base_result)

    def _adjust(self, base_score: Mapping[str, object]) -> dict:
        result = deepcopy(dict(base_score))
        k_value = _finite_float(result.get("k"))
        buy_bonus = (
            EXTREME_ZONE_POINTS
            if k_value is not None
            and k_value <= OVERSOLD_K_MAX
            and not bool(result.get("downside_continuation"))
            else 0.0
        )
        sell_bonus = (
            EXTREME_ZONE_POINTS
            if k_value is not None and k_value >= OVERBOUGHT_K_MIN
            else 0.0
        )
        official_buy = float(result.get("buy_score", 0.0) or 0.0)
        official_sell = float(result.get("sell_score", 0.0) or 0.0)
        result.update(
            {
                "official_buy_score": official_buy,
                "official_sell_score": official_sell,
                "buy_extreme_zone_score": buy_bonus,
                "sell_extreme_zone_score": sell_bonus,
                "buy_score": max(0.0, official_buy + buy_bonus),
                "sell_score": max(0.0, official_sell + sell_bonus),
            }
        )
        return result


@dataclass(frozen=True)
class ExtremeZonePerformance:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    win_rate: float
    profit_loss_ratio: float | None
    buy_count: int
    sell_count: int
    annual_returns: Dict[int, float]


@dataclass(frozen=True)
class ExtremeZoneGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtremeZoneComparisonReport:
    baseline_report: object
    candidate_report: object
    baseline_double_friction_report: object
    candidate_double_friction_report: object
    baseline: ExtremeZonePerformance
    candidate: ExtremeZonePerformance
    baseline_double_friction: ExtremeZonePerformance
    candidate_double_friction: ExtremeZonePerformance
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    gate: ExtremeZoneGateDecision


@dataclass(frozen=True)
class ExtremeZoneBindingSummary:
    oversold_bonus_events: int
    overbought_bonus_events: int
    buy_threshold_crossings: int
    sell_threshold_crossings: int
    buy_crossings_by_code: Dict[str, int]
    sell_crossings_by_code: Dict[str, int]


def summarize_extreme_zone_bindings(
    official_scores: Sequence[Mapping[str, object]],
) -> ExtremeZoneBindingSummary:
    """Count when the fixed five points can actually cross a score gate."""

    oversold_events = 0
    overbought_events = 0
    buy_crossings: Dict[str, int] = {}
    sell_crossings: Dict[str, int] = {}
    for score in official_scores:
        k_value = _finite_float(score.get("k"))
        if k_value is None:
            continue
        code = str(score.get("code", "")).split(".")[0] or "UNKNOWN"
        buy_bonus = (
            EXTREME_ZONE_POINTS
            if k_value <= OVERSOLD_K_MAX
            and not bool(score.get("downside_continuation"))
            else 0.0
        )
        sell_bonus = EXTREME_ZONE_POINTS if k_value >= OVERBOUGHT_K_MIN else 0.0
        if buy_bonus:
            oversold_events += 1
            buy_score = float(score.get("buy_score", 0.0) or 0.0)
            if buy_score < BUY_THRESHOLD <= buy_score + buy_bonus:
                buy_crossings[code] = buy_crossings.get(code, 0) + 1
        if sell_bonus:
            overbought_events += 1
            sell_score = float(score.get("sell_score", 0.0) or 0.0)
            if sell_score < SELL_THRESHOLD <= sell_score + sell_bonus:
                sell_crossings[code] = sell_crossings.get(code, 0) + 1
    return ExtremeZoneBindingSummary(
        oversold_bonus_events=oversold_events,
        overbought_bonus_events=overbought_events,
        buy_threshold_crossings=sum(buy_crossings.values()),
        sell_threshold_crossings=sum(sell_crossings.values()),
        buy_crossings_by_code=dict(sorted(buy_crossings.items())),
        sell_crossings_by_code=dict(sorted(sell_crossings.items())),
    )


def evaluate_extreme_zone_gate(
    baseline: ExtremeZonePerformance,
    candidate: ExtremeZonePerformance,
    changed_days_by_year: Mapping[int, int],
) -> ExtremeZoneGateDecision:
    """Accuracy-first gate frozen before the training A/B is inspected."""

    reasons = []
    for year in (2019, 2020, 2021):
        if int(changed_days_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed filled-order day" % year)
    if candidate.win_rate <= baseline.win_rate:
        reasons.append("candidate win rate does not improve")
    if candidate.total_return < baseline.total_return * 0.95:
        reasons.append("candidate retains less than 95% of baseline return")
    if candidate.max_drawdown > baseline.max_drawdown + 0.005:
        reasons.append("candidate maximum drawdown worsens by more than 0.5pp")
    if not _retains_ratio(candidate.sharpe_ratio, baseline.sharpe_ratio, 0.95):
        reasons.append("candidate Sharpe ratio worsens by more than 5%")
    if not _retains_ratio(candidate.sortino_ratio, baseline.sortino_ratio, 0.95):
        reasons.append("candidate Sortino ratio worsens by more than 5%")
    if not _retains_ratio(
        candidate.profit_loss_ratio,
        baseline.profit_loss_ratio,
        0.95,
    ):
        reasons.append("candidate profit/loss ratio worsens by more than 5%")
    for year in (2019, 2020, 2021):
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif baseline_return > 0 and candidate_return <= 0:
            reasons.append("%d candidate annual return turns non-positive" % year)
    return ExtremeZoneGateDecision(not reasons, tuple(reasons))


def run_extreme_zone_training_ab(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> ExtremeZoneComparisonReport:
    """Run the frozen candidate and official baseline on approved training data."""

    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local.local_order_planner import (
        LocalCrossSignalOrderPlanner,
        strategy,
    )
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.baseline_report import build_baseline_report
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )

    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    warmup = Path(warmup_root).expanduser().resolve()
    if warmup != Path(APPROVED_WARMUP_ROOT).expanduser().resolve():
        raise ValueError("Use approved warm-up data root only: %s" % APPROVED_WARMUP_ROOT)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)
    params = strategy.get_default_params()
    pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    official_source = build_training_signal_adapter(loader, warmup_root=warmup)
    cached = PrecomputedSignalAdapter.from_source(
        official_source,
        trade_dates=trade_dates,
        codes=pool,
    )

    baseline_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    candidate_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            ExtremeZoneScoreAdapter(cached),
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    baseline_stress_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=DOUBLE_FRICTION,
    )
    candidate_stress_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            ExtremeZoneScoreAdapter(cached),
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=DOUBLE_FRICTION,
    )

    baseline_report = build_baseline_report(baseline_days, initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash)
    baseline_stress_report = build_baseline_report(
        baseline_stress_days, initial_cash
    )
    candidate_stress_report = build_baseline_report(
        candidate_stress_days, initial_cash
    )
    baseline = _performance(baseline_report, baseline_days, initial_cash)
    candidate = _performance(candidate_report, candidate_days, initial_cash)
    baseline_stress = _performance(
        baseline_stress_report, baseline_stress_days, initial_cash
    )
    candidate_stress = _performance(
        candidate_stress_report, candidate_stress_days, initial_cash
    )
    changed_days_by_year = _changed_days_by_year(baseline_days, candidate_days)
    gate = evaluate_extreme_zone_gate(
        baseline,
        candidate,
        changed_days_by_year,
    )
    return ExtremeZoneComparisonReport(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_double_friction_report=baseline_stress_report,
        candidate_double_friction_report=candidate_stress_report,
        baseline=baseline,
        candidate=candidate,
        baseline_double_friction=baseline_stress,
        candidate_double_friction=candidate_stress,
        changed_order_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        gate=gate,
    )


def format_extreme_zone_comparison(report: ExtremeZoneComparisonReport) -> str:
    lines = [
        "KDJ extreme-zone score candidate (2019-2021; local screen only)",
        "rule=K<=20 buy+5 unless downside_continuation; K>=80 sell+5; no cross required",
        _performance_line("BASELINE", report.baseline),
        _performance_line("CANDIDATE", report.candidate),
        _performance_line("BASELINE_X2_FRICTION", report.baseline_double_friction),
        _performance_line("CANDIDATE_X2_FRICTION", report.candidate_double_friction),
        "CHANGED days=%d by_year=%s"
        % (report.changed_order_days, report.changed_days_by_year),
        "GATE=%s" % ("PASS" if report.gate.passed else "REJECT"),
    ]
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.append("authority=local_screen_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def _run_replay(
    loader,
    planner,
    trade_dates: Sequence[str],
    initial_cash: float,
    broker_kwargs: Mapping[str, object] | None = None,
):
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine

    engine = LocalBacktestEngine(
        loader=loader,
        initial_cash=initial_cash,
        execution_time="09:35",
        broker_kwargs=broker_kwargs,
    )
    return engine.run(trade_dates, planner.plan_orders)


def _performance(report, days: Sequence[object], initial_cash: float):
    return ExtremeZonePerformance(
        total_return=float(report.total_return),
        annualized_return=float(report.annualized_return),
        max_drawdown=float(report.max_drawdown),
        sharpe_ratio=report.sharpe_ratio,
        sortino_ratio=report.sortino_ratio,
        win_rate=float(report.win_rate),
        profit_loss_ratio=report.profit_loss_ratio,
        buy_count=int(report.buy_count),
        sell_count=int(report.sell_count),
        annual_returns=_annual_returns(days, initial_cash),
    )


def _annual_returns(days: Sequence[object], initial_cash: float) -> Dict[int, float]:
    grouped: Dict[int, list[object]] = {}
    for day in days:
        grouped.setdefault(int(str(day.date)[:4]), []).append(day)
    annual = {}
    start_value = float(initial_cash)
    for year, year_days in sorted(grouped.items()):
        end_value = float(year_days[-1].total_value)
        annual[year] = end_value / start_value - 1.0 if start_value > 0 else 0.0
        start_value = end_value
    return annual


def _changed_days_by_year(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
) -> Dict[int, int]:
    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    if baseline_dates != candidate_dates:
        raise ValueError("A/B comparison requires identical trading dates")
    changed: Dict[int, int] = {}
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        if _filled_order_signature(baseline_day) == _filled_order_signature(candidate_day):
            continue
        year = int(str(baseline_day.date)[:4])
        changed[year] = changed.get(year, 0) + 1
    return changed


def _filled_order_signature(day: object) -> tuple[tuple[str, str, str], ...]:
    signature = []
    for order in getattr(day, "orders", []):
        if not getattr(order, "filled", False):
            continue
        amount = int(getattr(order, "amount_delta", 0))
        side = "buy" if amount > 0 else "sell" if amount < 0 else "flat"
        signature.append(
            (
                str(order.code).split(".")[0],
                side,
                str(getattr(order, "reason", "")),
            )
        )
    return tuple(sorted(signature))


def _assert_approved_loader(loader: object) -> None:
    root = getattr(loader, "root", None)
    if root is None or (
        Path(root).expanduser().resolve()
        != Path(APPROVED_TRAINING_ROOT).expanduser().resolve()
    ):
        raise ValueError("Use approved training data root only: %s" % APPROVED_TRAINING_ROOT)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("A/B comparison contains dates outside 2019-2021 training window")


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _retains_ratio(
    candidate: float | None,
    baseline: float | None,
    fraction: float,
) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline) * fraction


def _performance_line(label: str, item: ExtremeZonePerformance) -> str:
    return (
        "%s return=%.2f%% annualized=%.2f%% dd=%.2f%% sharpe=%s sortino=%s "
        "win_rate=%.2f%% pl=%s buys=%d sells=%d annual=%s"
        % (
            label,
            item.total_return * 100.0,
            item.annualized_return * 100.0,
            item.max_drawdown * 100.0,
            _optional_ratio(item.sharpe_ratio),
            _optional_ratio(item.sortino_ratio),
            item.win_rate * 100.0,
            _optional_ratio(item.profit_loss_ratio),
            item.buy_count,
            item.sell_count,
            {year: round(value, 6) for year, value in item.annual_returns.items()},
        )
    )


def _optional_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % float(value)


def main() -> None:
    print(format_extreme_zone_comparison(run_extreme_zone_training_ab()))


if __name__ == "__main__":
    main()
