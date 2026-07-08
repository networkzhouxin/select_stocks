# -*- coding: utf-8 -*-
"""Post-sell diagnostics for cross-signal local training replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.trade_diagnostics import (
    ClosedTradeDiagnostic,
    run_training_trade_diagnostics,
)


DEFAULT_FORWARD_HORIZONS = (3, 5, 10, 20)


@dataclass(frozen=True)
class PostSellDiagnostic:
    code: str
    sell_date: str
    sell_reason: str
    sell_price: float
    forward_returns: Mapping[int, float | None]


@dataclass(frozen=True)
class ForwardReturnSummary:
    count: int
    mean_return: float | None
    positive_rate: float | None


def post_sell_returns(
    trade: ClosedTradeDiagnostic,
    loader,
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> PostSellDiagnostic:
    code = str(trade.code).split(".")[0]
    frame = loader.load_daily_frame(code, trade.sell_date)
    rows = frame[["date", "close"]].copy()
    rows["date"] = rows["date"].astype(str)
    rows = rows.sort_values("date").reset_index(drop=True)
    sell_date = str(trade.sell_date)
    sell_rows = rows.index[rows["date"] == sell_date].tolist()
    if not sell_rows:
        raise KeyError(f"No sell date row for {code} {sell_date}")
    sell_idx = sell_rows[0]
    sell_close = float(rows.loc[sell_idx, "close"])

    returns: Dict[int, float | None] = {}
    for horizon in horizons:
        target_idx = sell_idx + int(horizon)
        if target_idx >= len(rows):
            returns[int(horizon)] = None
            continue
        target_close = float(rows.loc[target_idx, "close"])
        returns[int(horizon)] = target_close / sell_close - 1.0 if sell_close > 0 else None
    return PostSellDiagnostic(
        code=code,
        sell_date=sell_date,
        sell_reason=str(trade.sell_reason),
        sell_price=float(trade.sell_price),
        forward_returns=returns,
    )


def summarize_post_sell_returns(
    diagnostics: Iterable[PostSellDiagnostic],
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> Dict[str, Dict[int, ForwardReturnSummary]]:
    grouped: Dict[str, Dict[int, list[float]]] = {}
    for diagnostic in diagnostics:
        reason = str(diagnostic.sell_reason)
        reason_group = grouped.setdefault(reason, {int(h): [] for h in horizons})
        for horizon in horizons:
            value = diagnostic.forward_returns.get(int(horizon))
            if value is not None:
                reason_group[int(horizon)].append(float(value))

    summary: Dict[str, Dict[int, ForwardReturnSummary]] = {}
    for reason, by_horizon in grouped.items():
        summary[reason] = {}
        for horizon, values in by_horizon.items():
            if not values:
                summary[reason][horizon] = ForwardReturnSummary(0, None, None)
                continue
            summary[reason][horizon] = ForwardReturnSummary(
                count=len(values),
                mean_return=sum(values) / len(values),
                positive_rate=sum(1 for value in values if value > 0) / len(values),
            )
    return summary


def run_training_post_sell_diagnostics(
    loader=None,
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> list[PostSellDiagnostic]:
    loader = loader or CrossSignalTrainingDataLoader()
    trades = run_training_trade_diagnostics(loader=loader)
    return [
        post_sell_returns(trade, loader, horizons=horizons)
        for trade in trades
        if trade.sell_reason
    ]
