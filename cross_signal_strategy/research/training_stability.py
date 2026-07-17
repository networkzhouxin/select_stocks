# -*- coding: utf-8 -*-
"""Training-only stability diagnostics for the cross-signal strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Iterable, Mapping, Sequence

from cross_signal_strategy.research.baseline_report import BaselineReport, build_baseline_report
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine, LocalBroker
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.trade_diagnostics import (
    ClosedTradeDiagnostic,
    DiagnosticOrderPlanner,
    build_closed_trade_diagnostics,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"


@dataclass(frozen=True)
class TradeGroupStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_holding_days: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None


@dataclass(frozen=True)
class AnnualStabilityStats:
    year: int
    start_value: float
    end_value: float
    total_return: float
    max_drawdown: float
    daily_win_rate: float
    average_exposure: float
    buy_count: int
    sell_count: int
    trade_stats: TradeGroupStats


@dataclass(frozen=True)
class ContributionConcentration:
    gross_profit: float = 0.0
    profitable_trade_count: int = 0
    largest_trade_profit_share: float = 0.0
    top_three_trade_profit_share: float = 0.0
    largest_code_profit_share: float = 0.0


@dataclass(frozen=True)
class HoldingPeriodStats:
    average_days: float = 0.0
    median_days: float = 0.0
    buckets: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FrictionStressStats:
    baseline_return: float
    stressed_return: float
    return_delta: float
    baseline_max_drawdown: float
    stressed_max_drawdown: float
    end_value_delta: float


@dataclass(frozen=True)
class TrainingStabilityReport:
    baseline: BaselineReport
    annual: Dict[int, AnnualStabilityStats]
    concentration: ContributionConcentration
    exit_reasons: Dict[str, TradeGroupStats]
    holding_periods: HoldingPeriodStats
    entry_regimes: Dict[str, TradeGroupStats]
    volatility_cutoff: float | None = None
    friction_stress: FrictionStressStats | None = None


def build_training_stability_report(
    results: Iterable[object],
    trades: Iterable[ClosedTradeDiagnostic],
    initial_cash: float = 20000.0,
    stressed_baseline: BaselineReport | None = None,
) -> TrainingStabilityReport:
    days = list(results)
    closed_trades = list(trades)
    _assert_training_window(days, closed_trades)

    baseline = build_baseline_report(days, initial_cash=initial_cash)
    trade_dates = [str(day.date) for day in days]
    holding_days = {
        _trade_key(trade): _holding_days(trade, trade_dates)
        for trade in closed_trades
    }
    volatility_cutoff = _volatility_cutoff(closed_trades)

    return TrainingStabilityReport(
        baseline=baseline,
        annual=_annual_stats(days, closed_trades, holding_days, initial_cash),
        concentration=_concentration(closed_trades),
        exit_reasons=_group_trade_stats(
            closed_trades,
            holding_days,
            key_fn=lambda trade: str(trade.sell_reason or "unknown"),
        ),
        holding_periods=_holding_period_stats(list(holding_days.values())),
        entry_regimes=_entry_regime_stats(closed_trades, holding_days, volatility_cutoff),
        volatility_cutoff=volatility_cutoff,
        friction_stress=_friction_stress(baseline, stressed_baseline),
    )


def run_training_stability_diagnostics(
    loader=None,
    initial_cash: float = 20000.0,
    friction_multiplier: float = 2.0,
) -> TrainingStabilityReport:
    loader = loader or CrossSignalTrainingDataLoader()
    baseline_results, trades = _run_training_path(loader, initial_cash)
    stressed_results, _ = _run_training_path(
        loader,
        initial_cash,
        commission_rate=0.0003 * friction_multiplier,
        min_commission=5.0 * friction_multiplier,
        slippage_rate=0.001 * friction_multiplier,
    )
    stressed_baseline = build_baseline_report(stressed_results, initial_cash=initial_cash)
    return build_training_stability_report(
        baseline_results,
        trades,
        initial_cash=initial_cash,
        stressed_baseline=stressed_baseline,
    )


def _run_training_path(
    loader,
    initial_cash: float,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
    slippage_rate: float = 0.001,
):
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    engine.broker = LocalBroker(
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        min_commission=min_commission,
        slippage_rate=slippage_rate,
    )
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return results, trades


def _assert_training_window(days: Sequence[object], trades: Sequence[ClosedTradeDiagnostic]) -> None:
    dates = [str(day.date) for day in days]
    dates.extend(str(trade.buy_date) for trade in trades)
    dates.extend(str(trade.sell_date) for trade in trades)
    if any(date < TRAINING_START or date > TRAINING_END for date in dates):
        raise ValueError("Stability diagnostics contain dates outside 2019-2021 training window")


def _annual_stats(
    days: Sequence[object],
    trades: Sequence[ClosedTradeDiagnostic],
    holding_days: Mapping[tuple, int],
    initial_cash: float,
) -> Dict[int, AnnualStabilityStats]:
    grouped_days: Dict[int, list] = {}
    for day in days:
        grouped_days.setdefault(int(str(day.date)[:4]), []).append(day)

    annual: Dict[int, AnnualStabilityStats] = {}
    start_value = float(initial_cash)
    for year, year_days in sorted(grouped_days.items()):
        year_report = build_baseline_report(year_days, initial_cash=start_value)
        values = [start_value] + [float(day.total_value) for day in year_days]
        year_trades = [trade for trade in trades if int(str(trade.sell_date)[:4]) == year]
        annual[year] = AnnualStabilityStats(
            year=year,
            start_value=start_value,
            end_value=float(year_days[-1].total_value),
            total_return=float(year_days[-1].total_value) / start_value - 1.0,
            max_drawdown=_max_drawdown(values),
            daily_win_rate=year_report.daily_win_rate,
            average_exposure=year_report.average_exposure,
            buy_count=year_report.buy_count,
            sell_count=year_report.sell_count,
            trade_stats=_trade_stats(year_trades, holding_days),
        )
        start_value = float(year_days[-1].total_value)
    return annual


def _concentration(trades: Sequence[ClosedTradeDiagnostic]) -> ContributionConcentration:
    profits = sorted((float(trade.pnl) for trade in trades if trade.pnl > 0), reverse=True)
    gross_profit = sum(profits)
    code_profits: Dict[str, float] = {}
    for trade in trades:
        if trade.pnl <= 0:
            continue
        code = str(trade.code).split(".")[0]
        code_profits[code] = code_profits.get(code, 0.0) + float(trade.pnl)
    if gross_profit <= 0:
        return ContributionConcentration()
    return ContributionConcentration(
        gross_profit=gross_profit,
        profitable_trade_count=len(profits),
        largest_trade_profit_share=profits[0] / gross_profit,
        top_three_trade_profit_share=sum(profits[:3]) / gross_profit,
        largest_code_profit_share=max(code_profits.values(), default=0.0) / gross_profit,
    )


def _entry_regime_stats(
    trades: Sequence[ClosedTradeDiagnostic],
    holding_days: Mapping[tuple, int],
    volatility_cutoff: float | None,
) -> Dict[str, TradeGroupStats]:
    grouped: Dict[str, list] = {}
    for trade in trades:
        trend = _trend_regime(trade.entry_score)
        volatility = _volatility_regime(trade.entry_score, volatility_cutoff)
        grouped.setdefault(f"trend:{trend}", []).append(trade)
        grouped.setdefault(f"volatility:{volatility}", []).append(trade)
    return {
        key: _trade_stats(items, holding_days)
        for key, items in sorted(grouped.items())
    }


def _trend_regime(score: Mapping[str, object]) -> str:
    trend = _numeric(score.get("trend_score"))
    if trend >= 20:
        return "strong_up"
    if trend > 0:
        return "mild_up"
    if trend < 0:
        return "down"
    return "sideways"


def _volatility_regime(score: Mapping[str, object], cutoff: float | None) -> str:
    atr = _numeric_or_none(score.get("atr"))
    close = _numeric_or_none(score.get("close"))
    if atr is None or close is None or close <= 0 or cutoff is None:
        return "unknown"
    return "high" if atr / close > cutoff else "normal"


def _volatility_cutoff(trades: Sequence[ClosedTradeDiagnostic]) -> float | None:
    ratios = []
    for trade in trades:
        atr = _numeric_or_none(trade.entry_score.get("atr"))
        close = _numeric_or_none(trade.entry_score.get("close"))
        if atr is not None and close is not None and close > 0:
            ratios.append(atr / close)
    return float(median(ratios)) if ratios else None


def _group_trade_stats(trades, holding_days, key_fn) -> Dict[str, TradeGroupStats]:
    grouped: Dict[str, list] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {
        key: _trade_stats(items, holding_days)
        for key, items in sorted(grouped.items())
    }


def _trade_stats(
    trades: Sequence[ClosedTradeDiagnostic],
    holding_days: Mapping[tuple, int],
) -> TradeGroupStats:
    gross_profit = sum(float(trade.pnl) for trade in trades if trade.pnl > 0)
    gross_loss = sum(abs(float(trade.pnl)) for trade in trades if trade.pnl < 0)
    durations = [holding_days.get(_trade_key(trade), 0) for trade in trades]
    return TradeGroupStats(
        closed_trades=len(trades),
        wins=sum(1 for trade in trades if trade.pnl > 0),
        losses=sum(1 for trade in trades if trade.pnl < 0),
        realized_pnl=sum(float(trade.pnl) for trade in trades),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        average_holding_days=sum(durations) / len(durations) if durations else 0.0,
    )


def _holding_period_stats(durations: Sequence[int]) -> HoldingPeriodStats:
    buckets = {"0-4": 0, "5-9": 0, "10-19": 0, "20+": 0}
    for duration in durations:
        if duration < 5:
            buckets["0-4"] += 1
        elif duration < 10:
            buckets["5-9"] += 1
        elif duration < 20:
            buckets["10-19"] += 1
        else:
            buckets["20+"] += 1
    return HoldingPeriodStats(
        average_days=sum(durations) / len(durations) if durations else 0.0,
        median_days=float(median(durations)) if durations else 0.0,
        buckets=buckets,
    )


def _friction_stress(
    baseline: BaselineReport,
    stressed: BaselineReport | None,
) -> FrictionStressStats | None:
    if stressed is None:
        return None
    return FrictionStressStats(
        baseline_return=baseline.total_return,
        stressed_return=stressed.total_return,
        return_delta=stressed.total_return - baseline.total_return,
        baseline_max_drawdown=baseline.max_drawdown,
        stressed_max_drawdown=stressed.max_drawdown,
        end_value_delta=stressed.end_value - baseline.end_value,
    )


def _holding_days(trade: ClosedTradeDiagnostic, trade_dates: Sequence[str]) -> int:
    date_index = {str(date): index for index, date in enumerate(trade_dates)}
    buy_date = str(trade.buy_date)
    sell_date = str(trade.sell_date)
    if buy_date in date_index and sell_date in date_index:
        return max(0, date_index[sell_date] - date_index[buy_date])
    return 0


def _trade_key(trade: ClosedTradeDiagnostic) -> tuple:
    return (
        str(trade.code),
        str(trade.buy_date),
        str(trade.sell_date),
        str(trade.sell_reason),
        float(trade.pnl),
    )


def _max_drawdown(values: Sequence[float]) -> float:
    peak = None
    drawdown = 0.0
    for value in values:
        peak = float(value) if peak is None else max(peak, float(value))
        if peak > 0:
            drawdown = max(drawdown, (peak - float(value)) / peak)
    return drawdown


def _numeric(value: object) -> float:
    number = _numeric_or_none(value)
    return number if number is not None else 0.0


def _numeric_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_training_stability_report(report: TrainingStabilityReport) -> str:
    lines = [
        "Cross-signal training stability report (2019-2021)",
        "baseline return={:.2%} max_drawdown={:.2%} exposure={:.3f}".format(
            report.baseline.total_return,
            report.baseline.max_drawdown,
            report.baseline.average_exposure,
        ),
    ]
    for year, stats in report.annual.items():
        lines.append(
            "{} return={:.2%} max_drawdown={:.2%} trade_pnl={:.2f} exposure={:.3f}".format(
                year,
                stats.total_return,
                stats.max_drawdown,
                stats.trade_stats.realized_pnl,
                stats.average_exposure,
            )
        )
    concentration = report.concentration
    lines.append(
        "profit concentration top1={:.2%} top3={:.2%} top_code={:.2%}".format(
            concentration.largest_trade_profit_share,
            concentration.top_three_trade_profit_share,
            concentration.largest_code_profit_share,
        )
    )
    if report.friction_stress is not None:
        lines.append(
            "2x friction return={:.2%} delta={:.2%} max_drawdown={:.2%}".format(
                report.friction_stress.stressed_return,
                report.friction_stress.return_delta,
                report.friction_stress.stressed_max_drawdown,
            )
        )
    return "\n".join(lines)


def main() -> None:
    print(format_training_stability_report(run_training_stability_diagnostics()))


if __name__ == "__main__":
    main()
