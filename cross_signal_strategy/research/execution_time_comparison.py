# -*- coding: utf-8 -*-
"""Training-only single-variable comparison of 09:35 versus 10:00 execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.baseline_report import BaselineReport, build_baseline_report


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
BASELINE_TIME = "09:35"
CANDIDATE_TIME = "10:00"
EXECUTION_TIMES = (BASELINE_TIME, CANDIDATE_TIME)
QDII_CODES = frozenset({"513050", "513100", "513500", "513880"})


@dataclass(frozen=True)
class ExecutionTimePerformance:
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
class ExecutionPriceStats:
    matched_orders: int
    average_signed_improvement: float | None
    matched_by_year: Dict[int, int]
    average_by_year: Dict[int, float]
    matched_by_group: Dict[str, int]
    average_by_group: Dict[str, float]


@dataclass(frozen=True)
class ExecutionTimeGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionTimeComparisonReport:
    baseline_time: str
    candidate_time: str
    baseline_report: BaselineReport
    candidate_report: BaselineReport
    baseline_performance: ExecutionTimePerformance
    candidate_performance: ExecutionTimePerformance
    price_stats: ExecutionPriceStats
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    gate: ExecutionTimeGateDecision


def evaluate_execution_time_gate(
    baseline: ExecutionTimePerformance,
    candidate: ExecutionTimePerformance,
    price_stats: ExecutionPriceStats,
) -> ExecutionTimeGateDecision:
    """Apply the market-structure and strict non-degradation gate fixed in advance."""
    reasons = []
    if candidate.total_return <= baseline.total_return:
        reasons.append("candidate total return does not improve")
    if candidate.annualized_return <= baseline.annualized_return:
        reasons.append("candidate annualized return does not improve")
    if candidate.max_drawdown > baseline.max_drawdown:
        reasons.append("candidate maximum drawdown worsens")
    if not _ratio_not_worse(candidate.sharpe_ratio, baseline.sharpe_ratio):
        reasons.append("candidate Sharpe ratio worsens")
    if not _ratio_not_worse(candidate.sortino_ratio, baseline.sortino_ratio):
        reasons.append("candidate Sortino ratio worsens")
    if candidate.win_rate < baseline.win_rate:
        reasons.append("candidate win rate worsens")
    if not _ratio_not_worse(candidate.profit_loss_ratio, baseline.profit_loss_ratio):
        reasons.append("candidate profit/loss ratio worsens")

    for year in (2019, 2020, 2021):
        if int(price_stats.matched_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no matched filled order" % year)
        if float(price_stats.average_by_year.get(year, 0.0)) <= 0.0:
            reasons.append("%d average execution price does not improve" % year)
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_return < baseline_return:
            reasons.append("%d candidate annual return worsens" % year)

    for group in ("qdii", "non_qdii"):
        if int(price_stats.matched_by_group.get(group, 0)) <= 0:
            reasons.append("%s has no matched filled order" % group)
        if float(price_stats.average_by_group.get(group, 0.0)) <= 0.0:
            reasons.append("%s average execution price does not improve" % group)
    return ExecutionTimeGateDecision(passed=not reasons, reasons=tuple(reasons))


def build_execution_time_comparison(
    results_by_time: Mapping[str, Iterable[object]],
    initial_cash: float = 20000.0,
) -> ExecutionTimeComparisonReport:
    if set(results_by_time) != set(EXECUTION_TIMES):
        raise ValueError("Execution-time comparison requires exactly %s" % (EXECUTION_TIMES,))

    baseline_days = list(results_by_time[BASELINE_TIME])
    candidate_days = list(results_by_time[CANDIDATE_TIME])
    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    _assert_training_dates(baseline_dates + candidate_dates)
    if baseline_dates != candidate_dates:
        raise ValueError("Execution-time comparison requires identical trading dates")

    baseline_report = build_baseline_report(baseline_days, initial_cash=initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash=initial_cash)
    baseline_performance = _performance(baseline_report, baseline_days, initial_cash)
    candidate_performance = _performance(candidate_report, candidate_days, initial_cash)
    price_stats = _execution_price_stats(baseline_days, candidate_days)

    changed_by_year: Dict[int, int] = {}
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        if _filled_order_signature(baseline_day) == _filled_order_signature(candidate_day):
            continue
        year = int(str(baseline_day.date)[:4])
        changed_by_year[year] = changed_by_year.get(year, 0) + 1

    return ExecutionTimeComparisonReport(
        baseline_time=BASELINE_TIME,
        candidate_time=CANDIDATE_TIME,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_performance=baseline_performance,
        candidate_performance=candidate_performance,
        price_stats=price_stats,
        changed_order_days=sum(changed_by_year.values()),
        changed_days_by_year=dict(sorted(changed_by_year.items())),
        gate=evaluate_execution_time_gate(
            baseline_performance,
            candidate_performance,
            price_stats,
        ),
    )


def run_training_execution_time_comparison(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
) -> ExecutionTimeComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    results_by_time = {}
    for execution_time in EXECUTION_TIMES:
        adapter = build_training_signal_adapter(loader, warmup_root=warmup_root)
        planner = LocalCrossSignalOrderPlanner(adapter, trade_dates=trade_dates)
        engine = LocalBacktestEngine(
            loader=loader,
            initial_cash=initial_cash,
            execution_time=execution_time,
        )
        results_by_time[execution_time] = engine.run(trade_dates, planner.plan_orders)
    return build_execution_time_comparison(results_by_time, initial_cash=initial_cash)


def _performance(
    report: BaselineReport,
    days: Sequence[object],
    initial_cash: float,
) -> ExecutionTimePerformance:
    return ExecutionTimePerformance(
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


def _execution_price_stats(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
) -> ExecutionPriceStats:
    baseline_orders = _indexed_filled_orders(baseline_days)
    candidate_orders = _indexed_filled_orders(candidate_days)
    matched_keys = sorted(set(baseline_orders).intersection(candidate_orders))

    observations = []
    for key in matched_keys:
        baseline = baseline_orders[key]
        candidate = candidate_orders[key]
        baseline_price = float(baseline.exec_price)
        candidate_price = float(candidate.exec_price)
        if baseline_price <= 0.0:
            continue
        side = key[2]
        improvement = (
            (baseline_price - candidate_price) / baseline_price
            if side == "buy"
            else (candidate_price - baseline_price) / baseline_price
        )
        observations.append((int(key[0][:4]), _etf_group(key[1]), improvement))

    by_year: Dict[int, list[float]] = {}
    by_group: Dict[str, list[float]] = {}
    for year, group, improvement in observations:
        by_year.setdefault(year, []).append(improvement)
        by_group.setdefault(group, []).append(improvement)
    values = [item[2] for item in observations]
    return ExecutionPriceStats(
        matched_orders=len(observations),
        average_signed_improvement=_mean(values),
        matched_by_year={key: len(items) for key, items in sorted(by_year.items())},
        average_by_year={key: _mean(items) or 0.0 for key, items in sorted(by_year.items())},
        matched_by_group={key: len(items) for key, items in sorted(by_group.items())},
        average_by_group={key: _mean(items) or 0.0 for key, items in sorted(by_group.items())},
    )


def _indexed_filled_orders(days: Sequence[object]) -> Dict[tuple[str, str, str, str, int], object]:
    indexed = {}
    counts: Dict[tuple[str, str, str, str], int] = {}
    for day in days:
        date = str(day.date)
        for order in getattr(day, "orders", []):
            if not getattr(order, "filled", False):
                continue
            amount = int(getattr(order, "amount_delta", 0))
            side = "buy" if amount > 0 else "sell" if amount < 0 else "flat"
            base = (date, str(order.code).split(".")[0], side, str(order.reason))
            occurrence = counts.get(base, 0)
            counts[base] = occurrence + 1
            indexed[base + (occurrence,)] = order
    return indexed


def _filled_order_signature(day) -> tuple[tuple[str, str, str], ...]:
    signature = []
    for order in getattr(day, "orders", []):
        if not getattr(order, "filled", False):
            continue
        amount = int(getattr(order, "amount_delta", 0))
        side = "buy" if amount > 0 else "sell" if amount < 0 else "flat"
        signature.append((str(order.code).split(".")[0], side, str(order.reason)))
    return tuple(sorted(signature))


def _etf_group(code: str) -> str:
    return "qdii" if str(code).split(".")[0] in QDII_CODES else "non_qdii"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio_not_worse(candidate: float | None, baseline: float | None) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Execution-time comparison contains dates outside 2019-2021 training window")


def format_execution_time_comparison(report: ExecutionTimeComparisonReport) -> str:
    lines = [
        "Cross-signal execution-time comparison (2019-2021; 2018 warm-up only)",
    ]
    for label, item in (
        (report.baseline_time, report.baseline_performance),
        (report.candidate_time, report.candidate_performance),
    ):
        lines.append(
            "TIME {} return={:.2%} annualized={:.2%} dd={:.2%} sharpe={} "
            "sortino={} win_rate={:.2%} pl_ratio={} buys={} sells={} annual={}".format(
                label,
                item.total_return,
                item.annualized_return,
                item.max_drawdown,
                _format_ratio(item.sharpe_ratio),
                _format_ratio(item.sortino_ratio),
                item.win_rate,
                _format_ratio(item.profit_loss_ratio),
                item.buy_count,
                item.sell_count,
                {year: round(value, 6) for year, value in item.annual_returns.items()},
            )
        )
    stats = report.price_stats
    lines.append(
        "PRICE matched={} avg_improvement={} matched_by_year={} avg_by_year={} "
        "matched_by_group={} avg_by_group={}".format(
            stats.matched_orders,
            _format_ratio(stats.average_signed_improvement),
            stats.matched_by_year,
            {key: round(value, 6) for key, value in stats.average_by_year.items()},
            stats.matched_by_group,
            {key: round(value, 6) for key, value in stats.average_by_group.items()},
        )
    )
    lines.append(
        "PATH changed_days={} changed_by_year={} gate={}".format(
            report.changed_order_days,
            report.changed_days_by_year,
            report.gate.passed,
        )
    )
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def main() -> None:
    print(format_execution_time_comparison(run_training_execution_time_comparison()))


if __name__ == "__main__":
    main()
