# -*- coding: utf-8 -*-
"""Observation-only trade quality ledger for the 2019-2021 training replay.

Forward closes in this module are ex-post diagnostic labels. They are never
returned to the signal adapter, order planner, or broker execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_data_loader import TRAIN_END, TRAIN_START
from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic


ENTRY_HORIZONS = (5, 10)
POST_SELL_HORIZONS = (5, 10)
ATR_BARRIER_HORIZON = 10
QDII_CODES = frozenset({"159920", "513050", "513100", "513500", "513880"})
_YEAR_PROBES = {
    2019: "2019-01-02",
    2020: "2020-01-02",
    2021: "2021-01-04",
}


@dataclass(frozen=True)
class TradeQualityRow:
    code: str
    buy_date: str
    sell_date: str
    sell_reason: str
    market_group: str
    realized_return_pct: float
    holding_trade_days: int
    holding_mfe: float
    holding_mae: float
    entry_mfe: Mapping[int, float | None]
    entry_mae: Mapping[int, float | None]
    first_profitable_close_offset: int | None
    first_atr_barrier: str
    post_sell_returns: Mapping[int, float | None]
    entry_atr_pct: float = 0.0


@dataclass(frozen=True)
class TradeQualitySummary:
    count: int
    win_rate: float
    mean_return_pct: float
    mean_holding_mfe: float
    mean_holding_mae: float
    quick_profit_rate: float
    up_first_rate: float
    down_first_rate: float


def _training_history(loader, code: str) -> pd.DataFrame:
    frames = []
    for year, probe_date in _YEAR_PROBES.items():
        try:
            frame = loader.load_daily_frame(code, probe_date)
        except FileNotFoundError:
            continue
        rows = frame[["date", "close"]].copy()
        rows["date"] = pd.to_datetime(rows["date"], errors="raise")
        if (rows["date"] < TRAIN_START).any() or (rows["date"] > TRAIN_END).any():
            raise ValueError("Daily history contains dates outside training window")
        frames.append(rows)
    if not frames:
        raise FileNotFoundError(f"No training daily history for {code}")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _validate_trade(trade: ClosedTradeDiagnostic) -> tuple[pd.Timestamp, pd.Timestamp]:
    buy_date = pd.Timestamp(trade.buy_date)
    sell_date = pd.Timestamp(trade.sell_date)
    if buy_date < TRAIN_START or sell_date > TRAIN_END or sell_date < buy_date:
        raise ValueError("Trade dates must stay inside the training window")
    entry_score = trade.entry_score or {}
    signal_date = entry_score.get("signal_date")
    if signal_date is not None:
        signal_day = pd.Timestamp(signal_date)
        proven_intraday = (
            signal_day == buy_date
            and str(entry_score.get("decision_time")) == "14:45"
            and str(entry_score.get("data_cutoff")) == "14:44"
        )
        if signal_day > buy_date or (signal_day == buy_date and not proven_intraday):
            raise ValueError("Entry signal date must be before buy date")
    return buy_date, sell_date


def _path_extremes(closes: Sequence[float], entry_price: float) -> tuple[float, float]:
    returns = [float(close) / entry_price - 1.0 for close in closes]
    return max(returns), min(returns)


def _first_atr_barrier(closes: Sequence[float], entry_price: float, atr: float) -> str:
    if atr <= 0:
        return "unavailable"
    upper = entry_price + atr
    lower = entry_price - atr
    for close in closes[:ATR_BARRIER_HORIZON]:
        value = float(close)
        if value >= upper:
            return "up_first"
        if value <= lower:
            return "down_first"
    return "neither" if len(closes) >= ATR_BARRIER_HORIZON else "unavailable"


def build_trade_quality_ledger(
    trades: Iterable[ClosedTradeDiagnostic],
    loader,
    entry_horizons: Sequence[int] = ENTRY_HORIZONS,
    post_sell_horizons: Sequence[int] = POST_SELL_HORIZONS,
) -> list[TradeQualityRow]:
    history_cache: Dict[str, pd.DataFrame] = {}
    ledger: list[TradeQualityRow] = []
    for trade in trades:
        buy_date, sell_date = _validate_trade(trade)
        code = str(trade.code).split(".")[0]
        if code not in history_cache:
            history_cache[code] = _training_history(loader, code)
        history = history_cache[code]
        buy_rows = history.index[history["date"] == buy_date].tolist()
        sell_rows = history.index[history["date"] == sell_date].tolist()
        if not buy_rows or not sell_rows:
            raise KeyError(f"Missing buy/sell daily row for {code}")
        buy_idx, sell_idx = buy_rows[0], sell_rows[0]
        entry_price = float(trade.buy_price)
        if entry_price <= 0:
            raise ValueError("Buy price must be positive")

        holding_closes = history.loc[buy_idx:sell_idx - 1, "close"].astype(float).tolist()
        holding_closes.append(float(trade.sell_price))
        holding_mfe, holding_mae = _path_extremes(holding_closes, entry_price)

        entry_mfe: Dict[int, float | None] = {}
        entry_mae: Dict[int, float | None] = {}
        for horizon in entry_horizons:
            end_idx = buy_idx + int(horizon) - 1
            if end_idx >= len(history):
                entry_mfe[int(horizon)] = None
                entry_mae[int(horizon)] = None
                continue
            closes = history.loc[buy_idx:end_idx, "close"].astype(float).tolist()
            entry_mfe[int(horizon)], entry_mae[int(horizon)] = _path_extremes(closes, entry_price)

        first_profitable = next(
            (offset for offset, close in enumerate(holding_closes) if float(close) > entry_price),
            None,
        )
        barrier_closes = history.loc[
            buy_idx:buy_idx + ATR_BARRIER_HORIZON - 1, "close"
        ].astype(float).tolist()
        atr = float((trade.entry_score or {}).get("atr") or 0.0)

        post_sell_returns: Dict[int, float | None] = {}
        sell_price = float(trade.sell_price)
        for horizon in post_sell_horizons:
            target_idx = sell_idx + int(horizon)
            post_sell_returns[int(horizon)] = (
                float(history.loc[target_idx, "close"]) / sell_price - 1.0
                if sell_price > 0 and target_idx < len(history)
                else None
            )

        ledger.append(TradeQualityRow(
            code=code,
            buy_date=str(trade.buy_date),
            sell_date=str(trade.sell_date),
            sell_reason=str(trade.sell_reason),
            market_group="qdii" if code in QDII_CODES else "non_qdii",
            realized_return_pct=float(trade.return_pct),
            holding_trade_days=sell_idx - buy_idx + 1,
            holding_mfe=holding_mfe,
            holding_mae=holding_mae,
            entry_mfe=entry_mfe,
            entry_mae=entry_mae,
            first_profitable_close_offset=first_profitable,
            first_atr_barrier=_first_atr_barrier(barrier_closes, entry_price, atr),
            post_sell_returns=post_sell_returns,
            entry_atr_pct=atr / entry_price if atr > 0 else 0.0,
        ))
    return ledger


def _summarize(rows: Sequence[TradeQualityRow]) -> TradeQualitySummary:
    count = len(rows)
    if not count:
        return TradeQualitySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TradeQualitySummary(
        count=count,
        win_rate=sum(row.realized_return_pct > 0 for row in rows) / count,
        mean_return_pct=sum(row.realized_return_pct for row in rows) / count,
        mean_holding_mfe=sum(row.holding_mfe for row in rows) / count,
        mean_holding_mae=sum(row.holding_mae for row in rows) / count,
        quick_profit_rate=sum(
            row.first_profitable_close_offset is not None and row.first_profitable_close_offset <= 2
            for row in rows
        ) / count,
        up_first_rate=sum(row.first_atr_barrier == "up_first" for row in rows) / count,
        down_first_rate=sum(row.first_atr_barrier == "down_first" for row in rows) / count,
    )


def summarize_trade_quality(rows: Iterable[TradeQualityRow]) -> Dict[str, TradeQualitySummary]:
    items = list(rows)
    groups: Dict[str, list[TradeQualityRow]] = {"all": items}
    for row in items:
        keys = (
            f"year:{pd.Timestamp(row.buy_date).year}",
            f"market:{row.market_group}",
            f"reason:{row.sell_reason}",
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    return {key: _summarize(group) for key, group in groups.items()}
