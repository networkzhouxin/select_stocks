# -*- coding: utf-8 -*-
"""Training-only single-variable comparison for cross_window=1/2/3/4."""

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
WINDOWS = (1, 2, 3, 4)
BASELINE_WINDOW = 3


def params_for_window(window: int) -> dict:
    window = int(window)
    if window not in WINDOWS:
        raise ValueError("cross_window must be one of %s" % (WINDOWS,))
    params = strategy.get_default_params()
    params["cross_window"] = window
    return params


@dataclass(frozen=True)
class CrossWindowPerformance:
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
class CrossWindowGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangedOrderDecision:
    date: str
    baseline_orders: tuple[tuple[str, str, str], ...]
    candidate_orders: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class CrossWindowVariantResult:
    window: int
    report: BaselineReport
    performance: CrossWindowPerformance
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    changed_decisions: tuple[ChangedOrderDecision, ...]
    gate: CrossWindowGateDecision | None


@dataclass(frozen=True)
class CrossWindowComparisonReport:
    baseline_window: int
    variants: Dict[int, CrossWindowVariantResult]


def evaluate_window_gate(
    baseline: CrossWindowPerformance,
    candidate: CrossWindowPerformance,
    changed_days_by_year: Mapping[int, int],
) -> CrossWindowGateDecision:
    """Apply the strict dominance gate fixed before running the comparison."""
    reasons = []
    for year in (2019, 2020, 2021):
        if int(changed_days_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed filled-order day" % year)

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
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_return < baseline_return:
            reasons.append("%d candidate annual return worsens" % year)
    return CrossWindowGateDecision(passed=not reasons, reasons=tuple(reasons))


def build_cross_window_comparison(
    results_by_window: Mapping[int, Iterable[object]],
    initial_cash: float = 20000.0,
) -> CrossWindowComparisonReport:
    if set(results_by_window) != set(WINDOWS):
        raise ValueError("Cross-window comparison requires exactly windows %s" % (WINDOWS,))

    days_by_window = {
        window: list(results_by_window[window])
        for window in WINDOWS
    }
    baseline_days = days_by_window[BASELINE_WINDOW]
    baseline_dates = [str(day.date) for day in baseline_days]
    _assert_training_dates(baseline_dates)

    reports = {}
    performances = {}
    for window in WINDOWS:
        days = days_by_window[window]
        dates = [str(day.date) for day in days]
        _assert_training_dates(dates)
        if dates != baseline_dates:
            raise ValueError("Cross-window comparison requires identical trading dates")
        report = build_baseline_report(days, initial_cash=initial_cash)
        reports[window] = report
        performances[window] = _performance(report, days, initial_cash)

    baseline_performance = performances[BASELINE_WINDOW]
    variants = {}
    for window in WINDOWS:
        changed_days_by_year: Dict[int, int] = {}
        changed_decisions = []
        for baseline_day, candidate_day in zip(baseline_days, days_by_window[window]):
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
        gate = None
        if window != BASELINE_WINDOW:
            gate = evaluate_window_gate(
                baseline_performance,
                performances[window],
                changed_days_by_year,
            )
        variants[window] = CrossWindowVariantResult(
            window=window,
            report=reports[window],
            performance=performances[window],
            changed_order_days=sum(changed_days_by_year.values()),
            changed_days_by_year=dict(sorted(changed_days_by_year.items())),
            changed_decisions=tuple(changed_decisions),
            gate=gate,
        )
    return CrossWindowComparisonReport(
        baseline_window=BASELINE_WINDOW,
        variants=variants,
    )


def run_training_cross_window_comparison(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
) -> CrossWindowComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    results_by_window = {}
    for window in WINDOWS:
        params = params_for_window(window)
        adapter = _training_adapter(loader, params, warmup_root)
        planner = LocalCrossSignalOrderPlanner(
            adapter,
            etf_pool=etf_pool,
            params=params,
            trade_dates=trade_dates,
        )
        engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
        results_by_window[window] = engine.run(trade_dates, planner.plan_orders)
    return build_cross_window_comparison(results_by_window, initial_cash=initial_cash)


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
) -> CrossWindowPerformance:
    return CrossWindowPerformance(
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
        raise ValueError("Cross-window comparison contains dates outside 2019-2021 training window")


def format_cross_window_comparison(report: CrossWindowComparisonReport) -> str:
    lines = [
        "Cross-signal cross-window comparison (2019-2021; 2018 warm-up only)",
        "BASELINE window=%d" % report.baseline_window,
    ]
    for window, variant in report.variants.items():
        item = variant.performance
        lines.append(
            "WINDOW {} return={:.2%} annualized={:.2%} dd={:.2%} sharpe={} "
            "sortino={} win_rate={:.2%} pl_ratio={} buys={} sells={} annual={} "
            "changed_days={} changed_by_year={} gate={}".format(
                window,
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
                variant.changed_order_days,
                variant.changed_days_by_year,
                "baseline" if variant.gate is None else variant.gate.passed,
            )
        )
        if variant.gate is not None:
            lines.extend(
                "WINDOW_GATE_REASON {} {}".format(window, reason)
                for reason in variant.gate.reasons
            )
    return "\n".join(lines)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def main() -> None:
    print(format_cross_window_comparison(run_training_cross_window_comparison()))


if __name__ == "__main__":
    main()
