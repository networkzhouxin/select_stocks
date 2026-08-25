# -*- coding: utf-8 -*-
"""Sole frozen 2019-2021 local comparison for the KRBA candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import types

sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as formal
from cross_signal_strategy.local.krba_backtester import KRBABacktestEngine
from cross_signal_strategy.local.local_adjustment import (
    default_training_adjustment_factors,
    default_training_daily_corrections,
)
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.baseline_report import build_baseline_report
from cross_signal_strategy.research.kdj_rsi_boll_atr_candidate import (
    KRBAOrderPlanner,
    KRBASignalAdapter,
    VERSION,
)


INITIAL_CASH = 20000.0
DOUBLE_FRICTION = {
    "commission_rate": 0.0006,
    "min_commission": 10.0,
    "slippage_rate": 0.002,
}


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrainingComparison:
    baseline: object
    candidate: object
    baseline_x2: object
    candidate_x2: object
    candidate_annual_returns: dict[int, float]
    candidate_trades_by_year: dict[int, int]
    max_single_profit_share: float
    gate: GateDecision
    candidate_days: tuple[object, ...]


def evaluate_gate(
    baseline,
    candidate,
    baseline_x2,
    candidate_x2,
    candidate_annual_returns,
    candidate_trades_by_year,
    max_single_profit_share,
) -> GateDecision:
    reasons = []
    if int(candidate.closed_trade_count) < 50:
        reasons.append("fewer than 50 closed trades")
    if float(candidate.win_rate) + 1e-12 < float(baseline.win_rate) + 0.05:
        reasons.append("win rate gain is below 5 percentage points")
    if float(candidate.total_return) + 1e-12 < float(baseline.total_return) * 0.80:
        reasons.append("total return retains less than 80%")
    baseline_pl = baseline.profit_loss_ratio
    candidate_pl = candidate.profit_loss_ratio
    if baseline_pl is not None and (
        candidate_pl is None
        or float(candidate_pl) + 1e-12 < float(baseline_pl) * 0.70
    ):
        reasons.append("profit/loss ratio retains less than 70%")
    if float(candidate.max_drawdown) > float(baseline.max_drawdown) + 0.01 + 1e-12:
        reasons.append("maximum drawdown worsens by more than 1 percentage point")
    for year in (2019, 2020, 2021):
        if year not in candidate_annual_returns:
            reasons.append(f"missing {year} annual return")
        if int(candidate_trades_by_year.get(year, 0)) <= 0:
            reasons.append(f"no closed trade in {year}")
    if float(max_single_profit_share) > 0.40 + 1e-12:
        reasons.append("largest winner contributes more than 40% of gross profit")
    if float(candidate_x2.total_return) + 1e-12 < float(baseline_x2.total_return) * 0.75:
        reasons.append("doubled-friction return retains less than 75%")
    if float(candidate_x2.max_drawdown) > float(baseline_x2.max_drawdown) + 0.01 + 1e-12:
        reasons.append("doubled-friction drawdown worsens by more than 1 percentage point")
    return GateDecision(not reasons, tuple(reasons))


def _annual_returns(days, initial_cash):
    result = {}
    start = float(initial_cash)
    for year in (2019, 2020, 2021):
        rows = [day for day in days if int(str(day.date)[:4]) == year]
        if not rows:
            continue
        end = float(rows[-1].total_value)
        result[year] = end / start - 1.0
        start = end
    return result


def _closed_trade_pnls(days):
    open_lots = {}
    result = []
    by_year = {2019: 0, 2020: 0, 2021: 0}
    for day in days:
        for order in day.orders:
            if not order.filled:
                continue
            code = str(order.code)
            if order.amount_delta > 0:
                open_lots[code] = (
                    int(order.amount_delta),
                    float(order.exec_price),
                    float(order.commission),
                )
            elif order.amount_delta < 0:
                lot = open_lots.pop(code, None)
                if lot is None:
                    continue
                amount, cost, buy_fee = lot
                sell_amount = min(amount, abs(int(order.amount_delta)))
                pnl = (
                    sell_amount * (float(order.exec_price) - cost)
                    - buy_fee
                    - float(order.commission)
                )
                result.append(pnl)
                by_year[int(str(day.date)[:4])] += 1
    return result, by_year


def _max_single_profit_share(pnls):
    winners = [float(value) for value in pnls if float(value) > 0]
    total = sum(winners)
    return max(winners) / total if total > 0 else 0.0


def _run_baseline(loader, dates, broker_kwargs=None):
    adapter = build_training_signal_adapter(loader)
    planner = LocalCrossSignalOrderPlanner(adapter, trade_dates=dates)
    engine = LocalBacktestEngine(
        loader, INITIAL_CASH, broker_kwargs=broker_kwargs
    )
    return engine.run(dates, planner.plan_orders)


def _run_candidate(loader, dates, broker_kwargs=None):
    adapter = KRBASignalAdapter(
        loader,
        warmup_root=APPROVED_WARMUP_ROOT,
        adjustment_factors=default_training_adjustment_factors(),
        daily_corrections=default_training_daily_corrections(),
    )
    pool = [str(code).split(".")[0] for code in formal.get_default_etf_pool()]
    planner = KRBAOrderPlanner(adapter, pool, trade_dates=dates)
    engine = KRBABacktestEngine(
        loader, INITIAL_CASH, broker_kwargs=broker_kwargs
    )
    return engine.run(dates, planner)


def run_locked_training_comparison(loader=None) -> TrainingComparison:
    loader = loader or CrossSignalTrainingDataLoader(APPROVED_TRAINING_ROOT)
    dates = get_training_trade_dates(loader)
    if not dates or dates[0] < "2019-01-01" or dates[-1] > "2021-12-31":
        raise ValueError("KRBA replay must stay inside 2019-2021")
    baseline_days = _run_baseline(loader, dates)
    candidate_days = _run_candidate(loader, dates)
    baseline_x2_days = _run_baseline(loader, dates, DOUBLE_FRICTION)
    candidate_x2_days = _run_candidate(loader, dates, DOUBLE_FRICTION)
    baseline = build_baseline_report(baseline_days, INITIAL_CASH)
    candidate = build_baseline_report(candidate_days, INITIAL_CASH)
    baseline_x2 = build_baseline_report(baseline_x2_days, INITIAL_CASH)
    candidate_x2 = build_baseline_report(candidate_x2_days, INITIAL_CASH)
    pnls, trades_by_year = _closed_trade_pnls(candidate_days)
    annual = _annual_returns(candidate_days, INITIAL_CASH)
    share = _max_single_profit_share(pnls)
    gate = evaluate_gate(
        baseline,
        candidate,
        baseline_x2,
        candidate_x2,
        annual,
        trades_by_year,
        share,
    )
    return TrainingComparison(
        baseline,
        candidate,
        baseline_x2,
        candidate_x2,
        annual,
        trades_by_year,
        share,
        gate,
        tuple(candidate_days),
    )


def _format_metrics(label, report):
    return (
        f"{label}: return={report.total_return:.4%} "
        f"annualized={report.annualized_return:.4%} "
        f"max_dd={report.max_drawdown:.4%} "
        f"win_rate={report.win_rate:.4%} "
        f"pl_ratio={report.profit_loss_ratio} "
        f"buys={report.buy_count} sells={report.sell_count} "
        f"closed={report.closed_trade_count}"
    )


def main():
    report = run_locked_training_comparison()
    print(f"version={VERSION}")
    print(_format_metrics("BASELINE", report.baseline))
    print(_format_metrics("CANDIDATE", report.candidate))
    print(_format_metrics("BASELINE_X2", report.baseline_x2))
    print(_format_metrics("CANDIDATE_X2", report.candidate_x2))
    print(f"annual_returns={report.candidate_annual_returns}")
    print(f"closed_trades_by_year={report.candidate_trades_by_year}")
    print(f"max_single_profit_share={report.max_single_profit_share:.4%}")
    print(f"gate={'PASS' if report.gate.passed else 'FAIL'}")
    for reason in report.gate.reasons:
        print(f"gate_reason={reason}")
    return 0 if report.gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
