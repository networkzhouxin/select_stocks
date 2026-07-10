# -*- coding: utf-8 -*-
"""Isolated local comparison of official and reversal-first candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.baseline_report import BaselineReport, build_baseline_report
from cross_signal_strategy.friction_diagnostics import PrecomputedSignalAdapter
from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"


def reversal_first_sort(candidates: Iterable[Mapping[str, object]]) -> list[dict]:
    return sorted(
        [dict(item) for item in candidates],
        key=lambda item: (
            -float(item.get("reversal_score", 0) or 0),
            -float(item.get("buy_score", 0) or 0),
            str(item.get("code", "")),
        ),
    )


@dataclass
class ReversalFirstOrderPlanner(LocalCrossSignalOrderPlanner):
    def _score_pool(self, current_date: str) -> list[dict]:
        return reversal_first_sort(super()._score_pool(current_date))


@dataclass(frozen=True)
class RankingPerformance:
    total_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    annual_returns: Dict[int, float]


@dataclass(frozen=True)
class RankingGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangedBuyDecision:
    date: str
    baseline_codes: tuple[str, ...]
    candidate_codes: tuple[str, ...]


@dataclass(frozen=True)
class RankingComparisonReport:
    baseline: BaselineReport
    candidate: BaselineReport
    baseline_performance: RankingPerformance
    candidate_performance: RankingPerformance
    changed_buy_days: int
    changed_days_by_year: Dict[int, int]
    changed_decisions: tuple[ChangedBuyDecision, ...]
    gate: RankingGateDecision


def evaluate_ranking_gate(
    baseline: RankingPerformance,
    candidate: RankingPerformance,
    changed_days_by_year: Mapping[int, int],
) -> RankingGateDecision:
    reasons = []
    changed_total = sum(int(value) for value in changed_days_by_year.values())
    if changed_total < 10:
        reasons.append("candidate changes fewer than 10 buy decision days")
    for year in (2019, 2020, 2021):
        if int(changed_days_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed buy decision day" % year)

    if candidate.total_return <= baseline.total_return:
        reasons.append("candidate total return does not improve")
    if candidate.max_drawdown > baseline.max_drawdown:
        reasons.append("candidate max drawdown worsens")
    if not _ratio_not_worse(candidate.sharpe_ratio, baseline.sharpe_ratio):
        reasons.append("candidate Sharpe ratio worsens")
    if not _ratio_not_worse(candidate.sortino_ratio, baseline.sortino_ratio):
        reasons.append("candidate Sortino ratio worsens")
    for year in (2019, 2020, 2021):
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_return < baseline_return:
            reasons.append("%d candidate annual return worsens" % year)
    return RankingGateDecision(passed=not reasons, reasons=tuple(reasons))


def build_ranking_comparison(
    baseline_results: Iterable[object],
    candidate_results: Iterable[object],
    initial_cash: float = 20000.0,
) -> RankingComparisonReport:
    baseline_days = list(baseline_results)
    candidate_days = list(candidate_results)
    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    _assert_training_dates(baseline_dates + candidate_dates)
    if baseline_dates != candidate_dates:
        raise ValueError("ranking comparison requires identical trading dates")

    baseline_report = build_baseline_report(baseline_days, initial_cash=initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash=initial_cash)
    changed_days_by_year: Dict[int, int] = {}
    changed_decisions = []
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        baseline_buys = _filled_buy_codes(baseline_day)
        candidate_buys = _filled_buy_codes(candidate_day)
        if baseline_buys == candidate_buys:
            continue
        year = int(str(baseline_day.date)[:4])
        changed_days_by_year[year] = changed_days_by_year.get(year, 0) + 1
        changed_decisions.append(
            ChangedBuyDecision(
                date=str(baseline_day.date),
                baseline_codes=tuple(sorted(baseline_buys)),
                candidate_codes=tuple(sorted(candidate_buys)),
            )
        )

    baseline_performance = _performance(
        baseline_report,
        baseline_days,
        initial_cash,
    )
    candidate_performance = _performance(
        candidate_report,
        candidate_days,
        initial_cash,
    )
    gate = evaluate_ranking_gate(
        baseline_performance,
        candidate_performance,
        changed_days_by_year,
    )
    return RankingComparisonReport(
        baseline=baseline_report,
        candidate=candidate_report,
        baseline_performance=baseline_performance,
        candidate_performance=candidate_performance,
        changed_buy_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        changed_decisions=tuple(changed_decisions),
        gate=gate,
    )


def run_training_ranking_comparison(
    loader=None,
    initial_cash: float = 20000.0,
) -> RankingComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    seed_planner = LocalCrossSignalOrderPlanner(source, trade_dates=trade_dates)
    cached = PrecomputedSignalAdapter.from_source(
        source,
        trade_dates=trade_dates,
        codes=seed_planner.etf_pool,
    )
    baseline_planner = LocalCrossSignalOrderPlanner(
        cached,
        etf_pool=seed_planner.etf_pool,
        trade_dates=trade_dates,
    )
    candidate_planner = ReversalFirstOrderPlanner(
        cached,
        etf_pool=seed_planner.etf_pool,
        trade_dates=trade_dates,
    )
    baseline_engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    candidate_engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    baseline_results = baseline_engine.run(trade_dates, baseline_planner.plan_orders)
    candidate_results = candidate_engine.run(trade_dates, candidate_planner.plan_orders)
    return build_ranking_comparison(
        baseline_results,
        candidate_results,
        initial_cash=initial_cash,
    )


def _performance(
    report: BaselineReport,
    days: Sequence[object],
    initial_cash: float,
) -> RankingPerformance:
    return RankingPerformance(
        total_return=float(report.total_return),
        max_drawdown=float(report.max_drawdown),
        sharpe_ratio=report.sharpe_ratio,
        sortino_ratio=report.sortino_ratio,
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


def _filled_buy_codes(day) -> set[str]:
    return {
        str(order.code).split(".")[0]
        for order in getattr(day, "orders", [])
        if getattr(order, "filled", False) and int(getattr(order, "amount_delta", 0)) > 0
    }


def _ratio_not_worse(candidate: float | None, baseline: float | None) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Ranking comparison contains dates outside 2019-2021 training window")


def format_ranking_comparison(report: RankingComparisonReport) -> str:
    baseline = report.baseline_performance
    candidate = report.candidate_performance
    lines = [
        "Cross-signal ranking comparison (2019-2021)",
        "BASE return={:.2%} dd={:.2%} sharpe={} sortino={} buys={} sells={}".format(
            baseline.total_return,
            baseline.max_drawdown,
            _format_ratio(baseline.sharpe_ratio),
            _format_ratio(baseline.sortino_ratio),
            report.baseline.buy_count,
            report.baseline.sell_count,
        ),
        "REVERSAL_FIRST return={:.2%} dd={:.2%} sharpe={} sortino={} buys={} sells={}".format(
            candidate.total_return,
            candidate.max_drawdown,
            _format_ratio(candidate.sharpe_ratio),
            _format_ratio(candidate.sortino_ratio),
            report.candidate.buy_count,
            report.candidate.sell_count,
        ),
        "CHANGED buy_days={} by_year={}".format(
            report.changed_buy_days,
            report.changed_days_by_year,
        ),
        "ANNUAL base={} candidate={}".format(
            {year: round(value, 6) for year, value in baseline.annual_returns.items()},
            {year: round(value, 6) for year, value in candidate.annual_returns.items()},
        ),
        "GATE passed=%s" % report.gate.passed,
    ]
    lines.extend(
        "CHANGED_DETAIL {} base={} candidate={}".format(
            item.date,
            list(item.baseline_codes),
            list(item.candidate_codes),
        )
        for item in report.changed_decisions
    )
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def main() -> None:
    print(format_ranking_comparison(run_training_ranking_comparison()))


if __name__ == "__main__":
    main()
