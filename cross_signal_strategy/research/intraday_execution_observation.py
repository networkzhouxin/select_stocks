# -*- coding: utf-8 -*-
"""Training-only counterfactual for the pre-registered minute buy overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from cross_signal_strategy.local.intraday_execution_overlay import (
    ARRIVAL_TIME,
    choose_buy_execution,
)
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine, LocalBroker
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
QDII_CODES = frozenset({"513050", "513100", "513500", "513880"})


@dataclass(frozen=True)
class ExecutionObservation:
    date: str
    code: str
    amount: int
    route: str
    fill_time: str
    baseline_exec_price: float
    candidate_exec_price: float
    signed_improvement: float


@dataclass(frozen=True)
class ExecutionGroupStats:
    count: int = 0
    average_signed_improvement: float = 0.0


@dataclass(frozen=True)
class IntradayExecutionGate:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntradayExecutionReport:
    eligible_buy_count: int
    matched_buy_count: int
    limit_fill_count: int
    fallback_fill_count: int
    observations: tuple[ExecutionObservation, ...]
    overall: ExecutionGroupStats
    by_year: Dict[int, ExecutionGroupStats]
    by_group: Dict[str, ExecutionGroupStats]
    gate: IntradayExecutionGate


def build_intraday_execution_report(
    baseline_days: Iterable[object],
    loader,
) -> IntradayExecutionReport:
    """Compare fixed baseline buy amounts with conservative candidate prices."""
    days = list(baseline_days)
    _assert_training_dates(str(day.date) for day in days)
    pricing = LocalBroker(initial_cash=0.0)
    eligible = 0
    observations = []

    for day in days:
        date = str(day.date)
        for order in getattr(day, "orders", []):
            if not (
                getattr(order, "filled", False)
                and int(getattr(order, "amount_delta", 0)) > 0
                and str(getattr(order, "reason", "")) == "buy_signal"
            ):
                continue
            eligible += 1
            code = str(order.code).split(".")[0]
            try:
                arrival_bar = loader.get_minute_bar(code, date, ARRIVAL_TIME)
                arrival_price = float(arrival_bar["close"])
                execution = choose_buy_execution(
                    loader.load_minute_frame(code, date),
                    date,
                    arrival_price,
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if not execution.filled or execution.fill_time is None:
                continue

            candidate_price = (
                pricing._round_price(execution.raw_price)
                if execution.route == "passive_limit"
                else pricing._buy_exec_price(execution.raw_price)
            )
            baseline_price = float(order.exec_price)
            if baseline_price <= 0.0 or candidate_price <= 0.0:
                continue
            observations.append(ExecutionObservation(
                date=date,
                code=code,
                amount=int(order.amount_delta),
                route=execution.route,
                fill_time=execution.fill_time,
                baseline_exec_price=baseline_price,
                candidate_exec_price=candidate_price,
                signed_improvement=(baseline_price - candidate_price) / baseline_price,
            ))

    frozen = tuple(observations)
    overall = _stats(frozen)
    by_year = {
        year: _stats(item for item in frozen if int(item.date[:4]) == year)
        for year in (2019, 2020, 2021)
    }
    by_group = {
        group: _stats(
            item for item in frozen
            if ("qdii" if item.code in QDII_CODES else "non_qdii") == group
        )
        for group in ("qdii", "non_qdii")
    }
    gate = evaluate_intraday_execution_gate(
        eligible_buy_count=eligible,
        matched_buy_count=len(frozen),
        overall=overall,
        by_year=by_year,
        by_group=by_group,
    )
    return IntradayExecutionReport(
        eligible_buy_count=eligible,
        matched_buy_count=len(frozen),
        limit_fill_count=sum(item.route == "passive_limit" for item in frozen),
        fallback_fill_count=sum(item.route == "market_fallback" for item in frozen),
        observations=frozen,
        overall=overall,
        by_year=by_year,
        by_group=by_group,
        gate=gate,
    )


def evaluate_intraday_execution_gate(
    eligible_buy_count: int,
    matched_buy_count: int,
    overall: ExecutionGroupStats,
    by_year: Dict[int, ExecutionGroupStats],
    by_group: Dict[str, ExecutionGroupStats],
) -> IntradayExecutionGate:
    reasons = []
    if int(matched_buy_count) != int(eligible_buy_count):
        reasons.append("matched buy count does not equal eligible buy count")
    if overall.count <= 0 or overall.average_signed_improvement <= 0.0:
        reasons.append("overall average execution price does not improve")
    for year in (2019, 2020, 2021):
        stats = by_year.get(year, ExecutionGroupStats())
        if stats.count <= 0:
            reasons.append("%d has no matched ordinary buy" % year)
        elif stats.average_signed_improvement <= 0.0:
            reasons.append("%d average execution price does not improve" % year)
    for group in ("qdii", "non_qdii"):
        stats = by_group.get(group, ExecutionGroupStats())
        if stats.count <= 0:
            reasons.append("%s has no matched ordinary buy" % group)
        elif stats.average_signed_improvement <= 0.0:
            reasons.append("%s average execution price does not improve" % group)
    return IntradayExecutionGate(not reasons, tuple(reasons))


def run_training_intraday_execution_observation(
    loader=None,
    initial_cash: float = 20000.0,
) -> IntradayExecutionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = LocalCrossSignalOrderPlanner(adapter, trade_dates=trade_dates)
    baseline = LocalBacktestEngine(loader=loader, initial_cash=initial_cash).run(
        trade_dates,
        planner.plan_orders,
    )
    return build_intraday_execution_report(baseline, loader)


def format_intraday_execution_report(report: IntradayExecutionReport) -> str:
    lines = [
        "Cross-signal minute execution overlay v1 (2019-2021)",
        "eligible=%d matched=%d limit=%d fallback=%d"
        % (
            report.eligible_buy_count,
            report.matched_buy_count,
            report.limit_fill_count,
            report.fallback_fill_count,
        ),
        "overall count=%d avg_improvement=%.6f"
        % (report.overall.count, report.overall.average_signed_improvement),
    ]
    for year, stats in sorted(report.by_year.items()):
        lines.append(
            "year=%d count=%d avg_improvement=%.6f"
            % (year, stats.count, stats.average_signed_improvement)
        )
    for group, stats in sorted(report.by_group.items()):
        lines.append(
            "group=%s count=%d avg_improvement=%.6f"
            % (group, stats.count, stats.average_signed_improvement)
        )
    lines.append("gate_passed=%s" % report.gate.passed)
    lines.extend("gate_reason=%s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _stats(items: Iterable[ExecutionObservation]) -> ExecutionGroupStats:
    values = [float(item.signed_improvement) for item in items]
    return ExecutionGroupStats(
        count=len(values),
        average_signed_improvement=(sum(values) / len(values) if values else 0.0),
    )


def _assert_training_dates(dates: Iterable[str]) -> None:
    for date in dates:
        if str(date) < TRAINING_START or str(date) > TRAINING_END:
            raise ValueError("Intraday execution contains dates outside 2019-2021 training window")


def main() -> None:
    print(format_intraday_execution_report(run_training_intraday_execution_observation()))


if __name__ == "__main__":
    main()
