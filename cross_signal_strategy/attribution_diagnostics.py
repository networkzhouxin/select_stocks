# -*- coding: utf-8 -*-
"""ETF-level attribution diagnostics for cross-signal training trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import pandas as pd

from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_training_run import get_training_trade_dates
from cross_signal_strategy.trade_diagnostics import (
    ClosedTradeDiagnostic,
    run_training_trade_diagnostics,
)


@dataclass(frozen=True)
class EtfAttributionStats:
    code: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_holding_days: float = 0.0
    atr_stop_count: int = 0
    signal_sell_count: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None

    @property
    def atr_stop_rate(self) -> float:
        return self.atr_stop_count / self.closed_trades if self.closed_trades else 0.0

    @property
    def signal_sell_rate(self) -> float:
        return self.signal_sell_count / self.closed_trades if self.closed_trades else 0.0


@dataclass(frozen=True)
class EtfAttributionReport:
    by_code: Dict[str, EtfAttributionStats] = field(default_factory=dict)
    total_realized_pnl: float = 0.0


@dataclass
class _MutableStats:
    code: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    holding_days_sum: float = 0.0
    atr_stop_count: int = 0
    signal_sell_count: int = 0

    def add(self, trade: ClosedTradeDiagnostic, holding_days: int) -> None:
        self.closed_trades += 1
        self.realized_pnl += float(trade.pnl)
        self.holding_days_sum += float(holding_days)
        if trade.pnl > 0:
            self.wins += 1
            self.gross_profit += float(trade.pnl)
        elif trade.pnl < 0:
            self.losses += 1
            self.gross_loss += abs(float(trade.pnl))
        if str(trade.sell_reason) == "atr_stop":
            self.atr_stop_count += 1
        elif str(trade.sell_reason) == "signal_sell":
            self.signal_sell_count += 1

    def freeze(self) -> EtfAttributionStats:
        return EtfAttributionStats(
            code=self.code,
            closed_trades=self.closed_trades,
            wins=self.wins,
            losses=self.losses,
            realized_pnl=self.realized_pnl,
            gross_profit=self.gross_profit,
            gross_loss=self.gross_loss,
            average_holding_days=(
                self.holding_days_sum / self.closed_trades
                if self.closed_trades else 0.0
            ),
            atr_stop_count=self.atr_stop_count,
            signal_sell_count=self.signal_sell_count,
        )


def build_etf_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
    trade_dates: Sequence[str],
) -> EtfAttributionReport:
    date_index = {pd.Timestamp(day).strftime("%Y-%m-%d"): idx for idx, day in enumerate(trade_dates)}
    mutable: Dict[str, _MutableStats] = {}
    for trade in trades:
        code = str(trade.code).split(".")[0]
        stats = mutable.setdefault(code, _MutableStats(code=code))
        stats.add(trade, _holding_days(trade, date_index))

    by_code = {
        code: item.freeze()
        for code, item in sorted(
            mutable.items(),
            key=lambda entry: (-entry[1].realized_pnl, entry[0]),
        )
    }
    return EtfAttributionReport(
        by_code=by_code,
        total_realized_pnl=sum(item.realized_pnl for item in by_code.values()),
    )


def _holding_days(trade: ClosedTradeDiagnostic, date_index: Dict[str, int]) -> int:
    buy = pd.Timestamp(trade.buy_date).strftime("%Y-%m-%d")
    sell = pd.Timestamp(trade.sell_date).strftime("%Y-%m-%d")
    if buy not in date_index or sell not in date_index:
        return max(0, (pd.Timestamp(sell) - pd.Timestamp(buy)).days)
    return max(0, date_index[sell] - date_index[buy])


def run_training_etf_attribution(loader=None) -> EtfAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    trades = run_training_trade_diagnostics(loader=loader)
    return build_etf_attribution(trades, trade_dates)
