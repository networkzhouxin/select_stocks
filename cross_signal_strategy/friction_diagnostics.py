# -*- coding: utf-8 -*-
"""Training-only transaction-friction decomposition for cross-signal replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.baseline_report import BaselineReport, build_baseline_report
from cross_signal_strategy.local_backtester import LocalBacktestEngine, LocalBroker
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
COMPONENT_SCENARIOS = (
    "commission_rate_x2",
    "minimum_commission_x2",
    "slippage_x2",
)
REQUIRED_SCENARIOS = ("baseline",) + COMPONENT_SCENARIOS + ("all_x2",)


@dataclass(frozen=True)
class FrictionScenarioConfig:
    commission_rate: float
    min_commission: float
    slippage_rate: float


LOCKED_SCENARIOS = {
    "baseline": FrictionScenarioConfig(0.0003, 5.0, 0.001),
    "commission_rate_x2": FrictionScenarioConfig(0.0006, 5.0, 0.001),
    "minimum_commission_x2": FrictionScenarioConfig(0.0003, 10.0, 0.001),
    "slippage_x2": FrictionScenarioConfig(0.0003, 5.0, 0.002),
    "all_x2": FrictionScenarioConfig(0.0006, 10.0, 0.002),
}


@dataclass(frozen=True)
class FrictionScenarioResult:
    name: str
    total_return: float
    max_drawdown: float
    end_value: float
    buy_count: int
    sell_count: int
    return_delta: float
    max_drawdown_delta: float
    end_value_delta: float
    buy_count_delta: int
    sell_count_delta: int


@dataclass(frozen=True)
class FrictionDecompositionReport:
    scenarios: Dict[str, FrictionScenarioResult]
    dominant_component: str
    component_return_delta_sum: float
    interaction_return_delta: float


class PrecomputedSignalAdapter:
    """Immutable-by-interface T-1 signal cache shared by friction scenarios."""

    def __init__(self, cache: Mapping[tuple[str, str], tuple[dict | None, str | None]]):
        self._cache = {
            (str(date), str(code).split(".")[0]): (
                dict(score) if score is not None else None,
                reason,
            )
            for (date, code), (score, reason) in cache.items()
        }

    @classmethod
    def from_source(
        cls,
        source,
        trade_dates: Sequence[str],
        codes: Iterable[str],
    ) -> "PrecomputedSignalAdapter":
        dates = [str(date) for date in trade_dates]
        if any(date < TRAINING_START or date > TRAINING_END for date in dates):
            raise ValueError("Signal cache contains dates outside 2019-2021 training window")

        normalized_codes = [str(code).split(".")[0] for code in codes]
        cache: Dict[tuple[str, str], tuple[dict | None, str | None]] = {}
        for date in dates:
            for code in normalized_codes:
                score, reason = source.score(code, date, return_reason=True)
                cache[(date, code)] = (
                    dict(score) if score is not None else None,
                    str(reason) if reason is not None else None,
                )
        return cls(cache)

    def score(self, code, current_date, return_reason=False):
        key = (str(current_date), str(code).split(".")[0])
        score, reason = self._cache.get(key, (None, "not_precomputed"))
        copied = dict(score) if score is not None else None
        return (copied, reason) if return_reason else copied


def build_friction_decomposition(
    reports: Mapping[str, BaselineReport],
) -> FrictionDecompositionReport:
    missing = [name for name in REQUIRED_SCENARIOS if name not in reports]
    if missing:
        raise ValueError("Missing friction scenarios: %s" % ", ".join(missing))

    baseline = reports["baseline"]
    scenarios: Dict[str, FrictionScenarioResult] = {}
    for name in REQUIRED_SCENARIOS:
        report = reports[name]
        scenarios[name] = FrictionScenarioResult(
            name=name,
            total_return=report.total_return,
            max_drawdown=report.max_drawdown,
            end_value=report.end_value,
            buy_count=report.buy_count,
            sell_count=report.sell_count,
            return_delta=report.total_return - baseline.total_return,
            max_drawdown_delta=report.max_drawdown - baseline.max_drawdown,
            end_value_delta=report.end_value - baseline.end_value,
            buy_count_delta=report.buy_count - baseline.buy_count,
            sell_count_delta=report.sell_count - baseline.sell_count,
        )

    dominant = min(
        COMPONENT_SCENARIOS,
        key=lambda name: scenarios[name].return_delta,
    )
    component_sum = sum(scenarios[name].return_delta for name in COMPONENT_SCENARIOS)
    interaction = scenarios["all_x2"].return_delta - component_sum
    return FrictionDecompositionReport(
        scenarios=scenarios,
        dominant_component=dominant,
        component_return_delta_sum=component_sum,
        interaction_return_delta=interaction,
    )


def run_training_friction_decomposition(
    loader=None,
    initial_cash: float = 20000.0,
) -> FrictionDecompositionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source_adapter = build_training_signal_adapter(loader)
    pool = LocalCrossSignalOrderPlanner(source_adapter, trade_dates=trade_dates).etf_pool
    cached_adapter = PrecomputedSignalAdapter.from_source(
        source_adapter,
        trade_dates=trade_dates,
        codes=pool,
    )

    reports = {
        name: _run_scenario(
            loader=loader,
            signal_adapter=cached_adapter,
            trade_dates=trade_dates,
            etf_pool=pool,
            initial_cash=initial_cash,
            config=config,
        )
        for name, config in LOCKED_SCENARIOS.items()
    }
    return build_friction_decomposition(reports)


def _run_scenario(
    loader,
    signal_adapter,
    trade_dates: Sequence[str],
    etf_pool: Sequence[str],
    initial_cash: float,
    config: FrictionScenarioConfig,
) -> BaselineReport:
    planner = LocalCrossSignalOrderPlanner(
        signal_adapter,
        etf_pool=etf_pool,
        trade_dates=list(trade_dates),
    )
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    engine.broker = LocalBroker(
        initial_cash=initial_cash,
        commission_rate=config.commission_rate,
        min_commission=config.min_commission,
        slippage_rate=config.slippage_rate,
    )
    results = engine.run(trade_dates, planner.plan_orders)
    return build_baseline_report(results, initial_cash=initial_cash)


def format_friction_decomposition(report: FrictionDecompositionReport) -> str:
    lines = ["Cross-signal training friction decomposition (2019-2021)"]
    for name in REQUIRED_SCENARIOS:
        item = report.scenarios[name]
        lines.append(
            "{} return={:.2%} delta={:.2%} max_drawdown={:.2%} buys={} sells={}".format(
                name,
                item.total_return,
                item.return_delta,
                item.max_drawdown,
                item.buy_count,
                item.sell_count,
            )
        )
    lines.append("dominant_component=%s" % report.dominant_component)
    lines.append(
        "component_delta_sum={:.2%} interaction={:.2%}".format(
            report.component_return_delta_sum,
            report.interaction_return_delta,
        )
    )
    return "\n".join(lines)


def main() -> None:
    print(format_friction_decomposition(run_training_friction_decomposition()))


if __name__ == "__main__":
    main()
