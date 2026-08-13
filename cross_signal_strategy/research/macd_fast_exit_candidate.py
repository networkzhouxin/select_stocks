# -*- coding: utf-8 -*-
"""Isolated training candidate: sell on a recent MACD death cross.

The candidate changes only the ordinary exit trigger.  After the frozen
five-trading-day minimum hold, a recent MACD death cross may exit without the
official sell-score, price-structure confirmation, or ADX protection.  ATR
stops and the complete official signal-sell path remain available.
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
CANDIDATE_MIN_SIGNAL_HOLD_DAYS = 5


def candidate_rule_contract() -> dict:
    """Expose the frozen rule boundary used by this single-variable test."""

    params = strategy.get_default_params()
    return {
        "min_signal_hold_days": int(params["min_signal_hold_days"]),
        "cross_window": int(params["cross_window"]),
        "buy_logic_changed": False,
        "official_signal_sell_preserved": True,
        "atr_stop_preserved": True,
    }


@dataclass
class MacdFastExitPlanner(LocalCrossSignalOrderPlanner):
    """Official local planner plus one optional MACD fast-exit channel."""

    macd_fast_exit_enabled: bool = True

    def plan_orders(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        current_prices: Mapping[str, float] | None = None,
    ) -> list[Mapping[str, float]]:
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}
        self.last_scores = score_map

        orders: list[Mapping[str, float]] = []
        sold_codes = self._atr_stop_codes(broker, current_prices or {})
        for code in sorted(sold_codes):
            orders.append({"code": code, "target_value": 0.0, "reason": "atr_stop"})

        for code in list(broker.positions.keys()):
            if code in sold_codes:
                continue
            score = score_map.get(code)
            if score is None:
                continue
            if not strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params.get(
                    "min_signal_hold_days", CANDIDATE_MIN_SIGNAL_HOLD_DAYS
                ),
                trade_days=self.trade_dates,
            ):
                continue
            if self.macd_fast_exit_enabled and score.get("macd_cross_down"):
                orders.append({
                    "code": code,
                    "target_value": 0.0,
                    "reason": "macd_fast_exit",
                })
                sold_codes.add(code)
            elif strategy.should_force_sell(
                score, atr_stop_triggered=False, params=self.params
            ):
                orders.append({
                    "code": code,
                    "target_value": 0.0,
                    "reason": "signal_sell",
                })
                sold_codes.add(code)

        held_after_sell = [
            code for code in broker.positions.keys() if code not in sold_codes
        ]
        slots = int(self.params["max_hold"]) - len(held_after_sell)
        if slots <= 0:
            return orders

        # Match the formal JoinQuant/PTrade rule: a filled same-day sell may
        # release a slot, but that ETF itself cannot be bought back that day.
        buy_exclusions = set(held_after_sell) | set(sold_codes)
        candidates = [
            item
            for item in strategy.filter_buy_candidates(
                scores, buy_exclusions, self.params
            )
            if not self._is_in_atr_stop_cooldown(item["code"], current_date)
        ]
        total_value = self._total_value(broker, current_prices or {})
        for score in candidates[:slots]:
            orders.append({
                "code": score["code"],
                "target_value": self._scaled_buy_target_value(
                    total_value, score, current_date
                ),
                "reason": "buy_signal",
            })
        return orders


@dataclass(frozen=True)
class MacdFastExitPerformance:
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
class MacdFastExitGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MacdFastExitComparisonReport:
    baseline_report: BaselineReport
    candidate_report: BaselineReport
    baseline: MacdFastExitPerformance
    candidate: MacdFastExitPerformance
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    gate: MacdFastExitGateDecision


def evaluate_macd_fast_exit_gate(
    baseline: MacdFastExitPerformance,
    candidate: MacdFastExitPerformance,
    changed_days_by_year: Mapping[int, int],
) -> MacdFastExitGateDecision:
    """Reject a faster exit unless it broadly dominates the frozen baseline."""

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
    if candidate.win_rate <= baseline.win_rate:
        reasons.append("candidate win rate does not improve")
    if not _ratio_not_worse(candidate.profit_loss_ratio, baseline.profit_loss_ratio):
        reasons.append("candidate profit/loss ratio worsens")
    for year in (2019, 2020, 2021):
        base_year = baseline.annual_returns.get(year)
        candidate_year = candidate.annual_returns.get(year)
        if base_year is None or candidate_year is None:
            reasons.append("%d annual return is missing" % year)
        elif candidate_year < base_year:
            reasons.append("%d candidate annual return worsens" % year)
    return MacdFastExitGateDecision(not reasons, tuple(reasons))


def run_training_macd_fast_exit_comparison(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
) -> MacdFastExitComparisonReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)
    params = strategy.get_default_params()
    contract = candidate_rule_contract()
    if contract["min_signal_hold_days"] != CANDIDATE_MIN_SIGNAL_HOLD_DAYS:
        raise ValueError("Candidate requires the frozen five-trading-day minimum hold")
    etf_pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]

    baseline_planner = MacdFastExitPlanner(
        _training_adapter(loader, params, warmup_root),
        etf_pool=etf_pool,
        params=dict(params),
        trade_dates=trade_dates,
        macd_fast_exit_enabled=False,
    )
    baseline_days = LocalBacktestEngine(loader, initial_cash).run(
        trade_dates, baseline_planner.plan_orders
    )

    candidate_planner = MacdFastExitPlanner(
        _training_adapter(loader, params, warmup_root),
        etf_pool=etf_pool,
        params=dict(params),
        trade_dates=trade_dates,
        macd_fast_exit_enabled=True,
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
    gate = evaluate_macd_fast_exit_gate(baseline, candidate, changed_days_by_year)
    return MacdFastExitComparisonReport(
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
) -> MacdFastExitPerformance:
    return MacdFastExitPerformance(
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
        raise ValueError("Comparison contains dates outside 2019-2021 training window")


def format_macd_fast_exit_comparison(report: MacdFastExitComparisonReport) -> str:
    lines = ["MACD-fast-exit comparison (2019-2021; 2018 warm-up only)"]
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
    print(format_macd_fast_exit_comparison(run_training_macd_fast_exit_comparison()))


if __name__ == "__main__":
    main()
