# -*- coding: utf-8 -*-
"""Training-only backup cross-signal slot-fill candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from cross_signal_strategy.research.baseline_report import BaselineReport, build_baseline_report
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


BACKUP_BUY_THRESHOLD = 50


def filter_backup_buy_candidates(
    scores: Iterable[Mapping[str, object]],
    held_codes: Iterable[str],
    params=None,
) -> list[dict]:
    p = params or strategy.get_default_params()
    held = {str(code).split(".")[0] for code in held_codes}
    primary_threshold = float(p["buy_threshold"])
    candidates = []
    for raw_score in scores:
        score = dict(raw_score)
        code = str(score.get("code", "")).split(".")[0]
        score["code"] = code
        buy_score = float(score.get("buy_score", 0) or 0)
        if not code or code in held:
            continue
        if float(score.get("reversal_score", 0) or 0) <= 0:
            continue
        if not score.get("buy_allowed"):
            continue
        if buy_score < BACKUP_BUY_THRESHOLD or buy_score >= primary_threshold:
            continue
        if float(score.get("sell_score", 0) or 0) >= float(p["sell_threshold"]):
            continue
        if not strategy.has_new_buy_position(score, p):
            continue
        if strategy.is_blocked_entry_combo(score):
            continue
        candidates.append(score)
    return strategy.sort_candidates(candidates)


@dataclass
class BackupFillOrderPlanner(LocalCrossSignalOrderPlanner):
    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = list(
            super().plan_orders(
                current_date,
                previous_date,
                broker,
                current_prices=current_prices,
            )
        )
        sold_codes = {
            str(order["code"]).split(".")[0]
            for order in orders
            if float(order.get("target_value", 0.0)) == 0.0
        }
        primary_buy_codes = {
            str(order["code"]).split(".")[0]
            for order in orders
            if order.get("reason") == "buy_signal"
        }
        held_after_sell = {
            str(code).split(".")[0]
            for code in broker.positions
            if str(code).split(".")[0] not in sold_codes
        }
        occupied = held_after_sell | primary_buy_codes
        slots = int(self.params["max_hold"]) - len(occupied)
        if slots <= 0:
            return orders

        force_stopped = {
            str(order["code"]).split(".")[0]
            for order in orders
            if order.get("reason") == "atr_stop"
        }
        backups = [
            item
            for item in filter_backup_buy_candidates(
                self.last_scores.values(),
                held_codes=occupied,
                params=self.params,
            )
            if item["code"] not in force_stopped
            and not self._is_in_atr_stop_cooldown(item["code"], current_date)
        ]
        total_value = self._total_value(broker, current_prices or {})
        for score in backups[:slots]:
            orders.append(
                {
                    "code": score["code"],
                    "target_value": self._scaled_buy_target_value(
                        total_value,
                        score,
                        current_date,
                    ),
                    "reason": "backup_buy_signal",
                }
            )
        return orders


@dataclass(frozen=True)
class BackupFillComparison:
    baseline: BaselineReport
    candidate: BaselineReport
    backup_buy_count: int


def run_training_backup_fill_comparison(
    loader=None,
    initial_cash: float = 20000.0,
) -> BackupFillComparison:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source_adapter = build_training_signal_adapter(loader)
    initial_planner = LocalCrossSignalOrderPlanner(source_adapter, trade_dates=trade_dates)
    cached_adapter = PrecomputedSignalAdapter.from_source(
        source_adapter,
        trade_dates=trade_dates,
        codes=initial_planner.etf_pool,
    )
    baseline = _run_planner(
        loader,
        LocalCrossSignalOrderPlanner(
            cached_adapter,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    candidate = _run_planner(
        loader,
        BackupFillOrderPlanner(
            cached_adapter,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    candidate_report, candidate_results = candidate
    backup_count = sum(
        1
        for day in candidate_results
        for order in day.orders
        if getattr(order, "filled", False)
        and getattr(order, "reason", "") == "backup_buy_signal"
    )
    return BackupFillComparison(
        baseline=baseline[0],
        candidate=candidate_report,
        backup_buy_count=backup_count,
    )


def _run_planner(loader, planner, trade_dates: Sequence[str], initial_cash: float):
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    return build_baseline_report(results, initial_cash=initial_cash), results


def format_backup_fill_comparison(comparison: BackupFillComparison) -> str:
    baseline = comparison.baseline
    candidate = comparison.candidate
    return "\n".join(
        [
            "Cross-signal backup-fill local training comparison (2019-2021)",
            "baseline return={:.2%} max_drawdown={:.2%} buys={} sells={} exposure={:.3f}".format(
                baseline.total_return,
                baseline.max_drawdown,
                baseline.buy_count,
                baseline.sell_count,
                baseline.average_exposure,
            ),
            "candidate return={:.2%} max_drawdown={:.2%} buys={} sells={} exposure={:.3f}".format(
                candidate.total_return,
                candidate.max_drawdown,
                candidate.buy_count,
                candidate.sell_count,
                candidate.average_exposure,
            ),
            "backup_buys={} return_delta={:.2%} drawdown_delta={:.2%}".format(
                comparison.backup_buy_count,
                candidate.total_return - baseline.total_return,
                candidate.max_drawdown - baseline.max_drawdown,
            ),
        ]
    )


def main() -> None:
    print(format_backup_fill_comparison(run_training_backup_fill_comparison()))


if __name__ == "__main__":
    main()
