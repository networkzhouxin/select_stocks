# -*- coding: utf-8 -*-
"""Isolated MACD(6,13,5) single-variable training comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.research.baseline_report import BaselineReport, build_baseline_report
from cross_signal_strategy.local.local_adjustment import (
    default_training_adjustment_factors,
    default_training_daily_corrections,
)
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner, strategy
from cross_signal_strategy.local.local_signal_adapter import LocalSignalAdapter
from cross_signal_strategy.local_training_run import get_training_trade_dates


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
CANDIDATE_VERSION = "cross-v0.3.2-macd-6-13-5-candidate"


def candidate_params() -> dict:
    params = strategy.get_default_params()
    params.update({
        "macd_fast": 6,
        "macd_slow": 13,
        "macd_signal": 5,
    })
    return params


@dataclass(frozen=True)
class MacdPerformance:
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
class MacdGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangedOrderDecision:
    date: str
    baseline_orders: tuple[tuple[str, str, str], ...]
    candidate_orders: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class MacdComparisonReport:
    baseline: BaselineReport
    candidate: BaselineReport
    baseline_performance: MacdPerformance
    candidate_performance: MacdPerformance
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    changed_decisions: tuple[ChangedOrderDecision, ...]
    gate: MacdGateDecision


def evaluate_macd_gate(
    baseline: MacdPerformance,
    candidate: MacdPerformance,
    changed_days_by_year: Mapping[int, int],
) -> MacdGateDecision:
    """Apply the gate fixed before observing the candidate backtest."""
    reasons = []
    for year in (2019, 2020, 2021):
        if int(changed_days_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed filled-order day" % year)

    if candidate.total_return <= baseline.total_return:
        reasons.append("candidate total return does not improve")
    if candidate.annualized_return <= baseline.annualized_return:
        reasons.append("candidate annualized return does not improve")
    if candidate.max_drawdown > baseline.max_drawdown:
        reasons.append("candidate max drawdown worsens")
    if not _ratio_not_worse(candidate.sharpe_ratio, baseline.sharpe_ratio):
        reasons.append("candidate Sharpe ratio worsens")
    if not _ratio_not_worse(candidate.sortino_ratio, baseline.sortino_ratio):
        reasons.append("candidate Sortino ratio worsens")
    if candidate.win_rate < baseline.win_rate:
        reasons.append("candidate win rate worsens")
    if not _ratio_not_worse(candidate.profit_loss_ratio, baseline.profit_loss_ratio):
        reasons.append("candidate profit/loss ratio worsens")

    for year in (2019, 2020, 2021):
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_return < baseline_return:
            reasons.append("%d candidate annual return worsens" % year)
    return MacdGateDecision(passed=not reasons, reasons=tuple(reasons))


def build_macd_comparison(
    baseline_results: Iterable[object],
    candidate_results: Iterable[object],
    initial_cash: float = 20000.0,
) -> MacdComparisonReport:
    baseline_days = list(baseline_results)
    candidate_days = list(candidate_results)
    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    _assert_training_dates(baseline_dates + candidate_dates)
    if baseline_dates != candidate_dates:
        raise ValueError("MACD comparison requires identical trading dates")

    baseline_report = build_baseline_report(baseline_days, initial_cash=initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash=initial_cash)
    changed_days_by_year: Dict[int, int] = {}
    changed_decisions = []
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        baseline_orders = _filled_order_signature(baseline_day)
        candidate_orders = _filled_order_signature(candidate_day)
        if baseline_orders == candidate_orders:
            continue
        year = int(str(baseline_day.date)[:4])
        changed_days_by_year[year] = changed_days_by_year.get(year, 0) + 1
        changed_decisions.append(ChangedOrderDecision(
            date=str(baseline_day.date),
            baseline_orders=baseline_orders,
            candidate_orders=candidate_orders,
        ))

    baseline_performance = _performance(baseline_report, baseline_days, initial_cash)
    candidate_performance = _performance(candidate_report, candidate_days, initial_cash)
    gate = evaluate_macd_gate(
        baseline_performance,
        candidate_performance,
        changed_days_by_year,
    )
    return MacdComparisonReport(
        baseline=baseline_report,
        candidate=candidate_report,
        baseline_performance=baseline_performance,
        candidate_performance=candidate_performance,
        changed_order_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        changed_decisions=tuple(changed_decisions),
        gate=gate,
    )


def run_training_macd_comparison(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
) -> MacdComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    baseline_params = strategy.get_default_params()
    variant_params = candidate_params()
    baseline_adapter = _training_adapter(loader, baseline_params, warmup_root)
    candidate_adapter = _training_adapter(loader, variant_params, warmup_root)
    etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    baseline_planner = LocalCrossSignalOrderPlanner(
        baseline_adapter,
        etf_pool=etf_pool,
        params=baseline_params,
        trade_dates=trade_dates,
    )
    candidate_planner = LocalCrossSignalOrderPlanner(
        candidate_adapter,
        etf_pool=etf_pool,
        params=variant_params,
        trade_dates=trade_dates,
    )
    baseline_engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    candidate_engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    baseline_results = baseline_engine.run(trade_dates, baseline_planner.plan_orders)
    candidate_results = candidate_engine.run(trade_dates, candidate_planner.plan_orders)
    return build_macd_comparison(
        baseline_results,
        candidate_results,
        initial_cash=initial_cash,
    )


def _training_adapter(loader, params: dict, warmup_root) -> LocalSignalAdapter:
    return LocalSignalAdapter(
        loader,
        params=params,
        warmup_root=warmup_root,
        adjustment_factors=default_training_adjustment_factors(),
        daily_corrections=default_training_daily_corrections(),
    )


def _performance(
    report: BaselineReport,
    days: Sequence[object],
    initial_cash: float,
) -> MacdPerformance:
    return MacdPerformance(
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


def _filled_order_signature(day) -> tuple[tuple[str, str, str], ...]:
    signature = []
    for order in getattr(day, "orders", []):
        if not getattr(order, "filled", False):
            continue
        amount = int(getattr(order, "amount_delta", 0))
        side = "buy" if amount > 0 else "sell" if amount < 0 else "flat"
        signature.append((
            str(order.code).split(".")[0],
            side,
            str(getattr(order, "reason", "")),
        ))
    return tuple(sorted(signature))


def _ratio_not_worse(candidate: float | None, baseline: float | None) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("MACD comparison contains dates outside 2019-2021 training window")


def format_macd_comparison(report: MacdComparisonReport) -> str:
    baseline = report.baseline_performance
    candidate = report.candidate_performance
    lines = [
        "Cross-signal MACD parameter comparison (2019-2021; 2018 warm-up only)",
        "BASE MACD(12,26,9) " + _format_performance(baseline),
        "CAND MACD(6,13,5) " + _format_performance(candidate),
        "ANNUAL base={} candidate={}".format(
            {year: round(value, 6) for year, value in baseline.annual_returns.items()},
            {year: round(value, 6) for year, value in candidate.annual_returns.items()},
        ),
        "CHANGED order_days={} by_year={}".format(
            report.changed_order_days,
            report.changed_days_by_year,
        ),
        "GATE passed=%s" % report.gate.passed,
    ]
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.extend(
        "CHANGED_DETAIL {} base={} candidate={}".format(
            item.date,
            list(item.baseline_orders),
            list(item.candidate_orders),
        )
        for item in report.changed_decisions[:40]
    )
    if len(report.changed_decisions) > 40:
        lines.append("CHANGED_DETAIL omitted=%d" % (len(report.changed_decisions) - 40))
    return "\n".join(lines)


def _format_performance(item: MacdPerformance) -> str:
    return (
        "return={:.2%} annualized={:.2%} dd={:.2%} sharpe={} sortino={} "
        "win_rate={:.2%} pl_ratio={} buys={} sells={}"
    ).format(
        item.total_return,
        item.annualized_return,
        item.max_drawdown,
        _format_ratio(item.sharpe_ratio),
        _format_ratio(item.sortino_ratio),
        item.win_rate,
        _format_ratio(item.profit_loss_ratio),
        item.buy_count,
        item.sell_count,
    )


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def main() -> None:
    print(format_macd_comparison(run_training_macd_comparison()))


if __name__ == "__main__":
    main()
