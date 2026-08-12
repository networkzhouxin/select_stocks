# -*- coding: utf-8 -*-
"""Ex-post profit-giveback diagnostics for the frozen 2019-2021 replay.

The forward path values consumed here are labels for research reporting only.
They never enter signal calculation, order planning, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_data_loader import TRAIN_END, TRAIN_START
from cross_signal_strategy.research.trade_diagnostics import run_training_trade_diagnostics
from cross_signal_strategy.research.trade_quality_ledger import (
    TradeQualityRow,
    build_trade_quality_ledger,
)
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader


FIXED_PEAK_BANDS = (0.02, 0.03, 0.04)


@dataclass(frozen=True)
class ProfitGivebackObservation:
    code: str
    buy_date: str
    sell_date: str
    peak_return: float
    realized_return: float
    giveback_from_peak: float
    entry_atr_pct: float
    reached_one_entry_atr: bool
    round_trip_to_non_profit: bool


@dataclass(frozen=True)
class ProfitGivebackStats:
    count: int
    reached_count: int
    round_trip_count: int
    reached_rate: float
    round_trip_rate: float
    mean_peak_return: float
    mean_realized_return: float
    mean_giveback: float


@dataclass(frozen=True)
class ProfitGivebackReport:
    all_trades: ProfitGivebackStats
    fixed_peak_bands: Mapping[float, ProfitGivebackStats]
    by_year: Mapping[int, ProfitGivebackStats]
    observations: tuple[ProfitGivebackObservation, ...]


def build_profit_giveback_observations(
    rows: Iterable[TradeQualityRow],
) -> list[ProfitGivebackObservation]:
    observations = []
    for row in rows:
        buy_date = pd.Timestamp(row.buy_date)
        sell_date = pd.Timestamp(row.sell_date)
        if buy_date < TRAIN_START or sell_date > TRAIN_END or sell_date < buy_date:
            raise ValueError("Trade dates must stay inside the training window")
        peak_return = float(row.holding_mfe)
        realized_return = float(row.realized_return_pct) / 100.0
        entry_atr_pct = float(row.entry_atr_pct)
        reached_one_atr = entry_atr_pct > 0.0 and peak_return >= entry_atr_pct
        observations.append(ProfitGivebackObservation(
            code=str(row.code).split(".")[0],
            buy_date=str(row.buy_date),
            sell_date=str(row.sell_date),
            peak_return=peak_return,
            realized_return=realized_return,
            giveback_from_peak=peak_return - realized_return,
            entry_atr_pct=entry_atr_pct,
            reached_one_entry_atr=reached_one_atr,
            round_trip_to_non_profit=reached_one_atr and realized_return <= 0.0,
        ))
    return observations


def build_profit_giveback_report(
    rows: Iterable[TradeQualityRow],
) -> ProfitGivebackReport:
    observations = build_profit_giveback_observations(rows)
    by_year: Dict[int, list[ProfitGivebackObservation]] = {}
    for item in observations:
        by_year.setdefault(pd.Timestamp(item.buy_date).year, []).append(item)
    return ProfitGivebackReport(
        all_trades=_summarize(observations),
        fixed_peak_bands={
            band: _summarize(observations, peak_band=band)
            for band in FIXED_PEAK_BANDS
        },
        by_year={year: _summarize(items) for year, items in sorted(by_year.items())},
        observations=tuple(observations),
    )


def run_training_profit_giveback_report(loader=None) -> ProfitGivebackReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trades = run_training_trade_diagnostics(loader=loader)
    rows = build_trade_quality_ledger(trades, loader)
    return build_profit_giveback_report(rows)


def _summarize(
    observations: Sequence[ProfitGivebackObservation],
    peak_band: float | None = None,
) -> ProfitGivebackStats:
    count = len(observations)
    if peak_band is None:
        reached = [item for item in observations if item.reached_one_entry_atr]
    else:
        reached = [item for item in observations if item.peak_return >= float(peak_band)]
    round_trips = [item for item in reached if item.realized_return <= 0.0]
    return ProfitGivebackStats(
        count=count,
        reached_count=len(reached),
        round_trip_count=len(round_trips),
        reached_rate=len(reached) / count if count else 0.0,
        round_trip_rate=len(round_trips) / len(reached) if reached else 0.0,
        mean_peak_return=_mean([item.peak_return for item in observations]),
        mean_realized_return=_mean([item.realized_return for item in observations]),
        mean_giveback=_mean([item.giveback_from_peak for item in observations]),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_profit_giveback_report(report: ProfitGivebackReport) -> str:
    lines = [
        "Cross-signal profit giveback diagnostic (2019-2021; ex-post labels only)",
        _format_stats("ONE_ENTRY_ATR", report.all_trades),
    ]
    lines.extend(
        _format_stats("PEAK_%dPCT" % round(band * 100), stats)
        for band, stats in report.fixed_peak_bands.items()
    )
    lines.extend(
        _format_stats("YEAR_%d_ONE_ENTRY_ATR" % year, stats)
        for year, stats in report.by_year.items()
    )
    return "\n".join(lines)


def _format_stats(label: str, stats: ProfitGivebackStats) -> str:
    return (
        "%s count=%d reached=%d reached_rate=%.2f%% round_trip=%d "
        "round_trip_rate=%.2f%% mean_peak=%.2f%% mean_realized=%.2f%% mean_giveback=%.2f%%"
        % (
            label,
            stats.count,
            stats.reached_count,
            stats.reached_rate * 100.0,
            stats.round_trip_count,
            stats.round_trip_rate * 100.0,
            stats.mean_peak_return * 100.0,
            stats.mean_realized_return * 100.0,
            stats.mean_giveback * 100.0,
        )
    )


def main() -> None:
    print(format_profit_giveback_report(run_training_profit_giveback_report()))


if __name__ == "__main__":
    main()
