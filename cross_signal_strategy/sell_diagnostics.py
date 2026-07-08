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


@dataclass(frozen=True)
class SellFlyDiagnostic:
    code: str
    sell_date: str
    sell_reason: str
    horizon: int
    forward_return: float | None
    missed_pnl: float | None
    is_sell_fly: bool
    exit_features: Mapping[str, object]


@dataclass(frozen=True)
class SellFlyFeatureSummary:
    count: int
    sell_fly_count: int
    average_forward_return: float | None
    average_missed_pnl: float | None

    @property
    def sell_fly_rate(self) -> float:
        return self.sell_fly_count / self.count if self.count else 0.0


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


def sell_fly_diagnostic(
    trade: ClosedTradeDiagnostic,
    loader,
    horizon: int = 5,
    min_forward_return: float = 0.03,
) -> SellFlyDiagnostic:
    post_sell = post_sell_returns(trade, loader, horizons=(horizon,))
    forward_return = post_sell.forward_returns.get(int(horizon))
    missed_pnl = None
    if forward_return is not None:
        missed_pnl = float(trade.amount) * float(trade.sell_price) * float(forward_return)
    return SellFlyDiagnostic(
        code=str(trade.code).split(".")[0],
        sell_date=str(trade.sell_date),
        sell_reason=str(trade.sell_reason),
        horizon=int(horizon),
        forward_return=forward_return,
        missed_pnl=missed_pnl,
        is_sell_fly=(
            str(trade.sell_reason) == "signal_sell" and
            forward_return is not None and
            forward_return >= float(min_forward_return)
        ),
        exit_features=dict(getattr(trade, "exit_score", {}) or {}),
    )


def summarize_sell_fly_by_feature(
    diagnostics: Iterable[SellFlyDiagnostic],
    feature_name: str,
) -> Dict[object, SellFlyFeatureSummary]:
    grouped: Dict[object, list[SellFlyDiagnostic]] = {}
    for diagnostic in diagnostics:
        key = diagnostic.exit_features.get(feature_name)
        grouped.setdefault(key, []).append(diagnostic)

    summary: Dict[object, SellFlyFeatureSummary] = {}
    for key, items in grouped.items():
        forward_values = [
            float(item.forward_return)
            for item in items
            if item.forward_return is not None
        ]
        missed_values = [
            float(item.missed_pnl)
            for item in items
            if item.missed_pnl is not None
        ]
        summary[key] = SellFlyFeatureSummary(
            count=len(items),
            sell_fly_count=sum(1 for item in items if item.is_sell_fly),
            average_forward_return=(
                sum(forward_values) / len(forward_values)
                if forward_values else None
            ),
            average_missed_pnl=(
                sum(missed_values) / len(missed_values)
                if missed_values else None
            ),
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
