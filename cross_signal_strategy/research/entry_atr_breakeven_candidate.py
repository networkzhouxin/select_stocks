# -*- coding: utf-8 -*-
"""Isolated training candidate: recover initial risk after one entry ATR.

The rule activates only from the stored highest closing price, so a T-day
decision can use at most T-1 close state. The official planner is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from cross_signal_strategy.local.local_adjustment import (
    default_training_adjustment_factors,
    default_training_daily_corrections,
)
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import (
    LocalCrossSignalOrderPlanner,
    strategy,
)
from cross_signal_strategy.local.local_signal_adapter import LocalSignalAdapter
from cross_signal_strategy.local_training_run import get_training_trade_dates
from cross_signal_strategy.research.baseline_report import BaselineReport, build_baseline_report


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
BREAK_EVEN_ACTIVATION_ATR = 1.0
BREAK_EVEN_FLOOR_RETURN = 0.0


@dataclass
class EntryAtrBreakevenPlanner(LocalCrossSignalOrderPlanner):
    """Candidate planner preserving the official stop until activation."""

    def _atr_stop_codes(self, broker, current_prices: Mapping[str, float]) -> set:
        stopped = set()
        for code, pos in broker.positions.items():
            if code not in current_prices:
                continue
            highest = self.highest_since_buy.get(code)
            atr_val = self.entry_atr.get(code)
            price = float(current_prices[code])
            if highest is None or atr_val is None or price <= 0:
                continue
            stop_price = strategy.calc_stop_price(
                float(highest), float(atr_val), float(pos.avg_cost), self.params
            )
            activation_price = float(pos.avg_cost) + BREAK_EVEN_ACTIVATION_ATR * float(atr_val)
            if float(highest) >= activation_price:
                break_even_floor = float(pos.avg_cost) * (1.0 + BREAK_EVEN_FLOOR_RETURN)
                stop_price = max(stop_price, break_even_floor)
            if round(price, 3) <= round(stop_price, 3):
                stopped.add(code)
        return stopped


@dataclass(frozen=True)
class EntryAtrBreakevenPerformance:
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
class EntryAtrBreakevenGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryAtrBreakevenComparisonReport:
    baseline_report: BaselineReport
    candidate_report: BaselineReport
    baseline: EntryAtrBreakevenPerformance
    candidate: EntryAtrBreakevenPerformance
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    gate: EntryAtrBreakevenGateDecision


def evaluate_entry_atr_breakeven_gate(
    baseline: EntryAtrBreakevenPerformance,
    candidate: EntryAtrBreakevenPerformance,
    changed_days_by_year: Mapping[int, int],
) -> EntryAtrBreakevenGateDecision:
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
        base_year = baseline.annual_returns.get(year)
        candidate_year = candidate.annual_returns.get(year)
        if base_year is None or candidate_year is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_year < base_year:
            reasons.append("%d candidate annual return worsens" % year)
    return EntryAtrBreakevenGateDecision(not reasons, tuple(reasons))


def run_training_entry_atr_breakeven_comparison(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
) -> EntryAtrBreakevenComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)
    params = strategy.get_default_params()
    etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]

    baseline_adapter = _training_adapter(loader, params, warmup_root)
    baseline_planner = LocalCrossSignalOrderPlanner(
        baseline_adapter, etf_pool=etf_pool, params=dict(params), trade_dates=trade_dates
    )
    baseline_days = LocalBacktestEngine(loader, initial_cash).run(
        trade_dates, baseline_planner.plan_orders
    )

    candidate_adapter = _training_adapter(loader, params, warmup_root)
    candidate_planner = EntryAtrBreakevenPlanner(
        candidate_adapter, etf_pool=etf_pool, params=dict(params), trade_dates=trade_dates
    )
    candidate_days = LocalBacktestEngine(loader, initial_cash).run(
        trade_dates, candidate_planner.plan_orders
    )

    baseline_report = build_baseline_report(baseline_days, initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash)
    baseline = _performance(baseline_report, baseline_days, initial_cash)
    candidate = _performance(candidate_report, candidate_days, initial_cash)
    changed_days_by_year: Dict[int, int] = {}
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        if _filled_order_signature(baseline_day) == _filled_order_signature(candidate_day):
            continue
        year = int(str(baseline_day.date)[:4])
        changed_days_by_year[year] = changed_days_by_year.get(year, 0) + 1
    gate = evaluate_entry_atr_breakeven_gate(baseline, candidate, changed_days_by_year)
    return EntryAtrBreakevenComparisonReport(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline=baseline,
        candidate=candidate,
        changed_order_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        gate=gate,
    )


def _training_adapter(loader, params: dict, warmup_root) -> LocalSignalAdapter:
    return LocalSignalAdapter(
        loader,
        params=dict(params),
        warmup_root=warmup_root,
        adjustment_factors=default_training_adjustment_factors(),
        daily_corrections=default_training_daily_corrections(),
    )


def _performance(
    report: BaselineReport,
    days: Sequence[object],
    initial_cash: float,
) -> EntryAtrBreakevenPerformance:
    return EntryAtrBreakevenPerformance(
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
            str(order.code).split(".")[0], side, str(getattr(order, "reason", ""))
        ))
    return tuple(sorted(signature))


def _ratio_not_worse(candidate: float | None, baseline: float | None) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Comparison contains dates outside 2019-2021 training window")


def format_entry_atr_breakeven_comparison(
    report: EntryAtrBreakevenComparisonReport,
) -> str:
    lines = ["Entry-ATR break-even comparison (2019-2021; 2018 warm-up only)"]
    for label, item in (("BASELINE", report.baseline), ("CANDIDATE", report.candidate)):
        lines.append(
            "%s return=%.2f%% annualized=%.2f%% dd=%.2f%% sharpe=%s sortino=%s "
            "win_rate=%.2f%% pl_ratio=%s buys=%d sells=%d annual=%s"
            % (
                label,
                item.total_return * 100.0,
                item.annualized_return * 100.0,
                item.max_drawdown * 100.0,
                _format_ratio(item.sharpe_ratio),
                _format_ratio(item.sortino_ratio),
                item.win_rate * 100.0,
                _format_ratio(item.profit_loss_ratio),
                item.buy_count,
                item.sell_count,
                {year: round(value, 6) for year, value in item.annual_returns.items()},
            )
        )
    lines.append(
        "CHANGED days=%d by_year=%s gate=%s"
        % (report.changed_order_days, report.changed_days_by_year, report.gate.passed)
    )
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def main() -> None:
    print(format_entry_atr_breakeven_comparison(
        run_training_entry_atr_breakeven_comparison()
    ))


if __name__ == "__main__":
    main()
