# -*- coding: utf-8 -*-
"""Baseline diagnostics for cross-signal local training replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass(frozen=True)
class CodeBaselineStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0


@dataclass(frozen=True)
class BaselineReport:
    start_date: str
    end_date: str
    trading_days: int
    start_value: float
    end_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    buy_count: int
    sell_count: int
    closed_trade_count: int
    win_rate: float
    profit_loss_ratio: float | None
    average_exposure: float
    position_count_days: Dict[int, int] = field(default_factory=dict)
    full_position_days: int = 0
    empty_days: int = 0
    by_code: Dict[str, CodeBaselineStats] = field(default_factory=dict)


@dataclass
class _OpenLot:
    amount: int
    cost: float
    commission: float


@dataclass
class _MutableCodeStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def freeze(self) -> CodeBaselineStats:
        return CodeBaselineStats(
            closed_trades=self.closed_trades,
            wins=self.wins,
            losses=self.losses,
            realized_pnl=self.realized_pnl,
            gross_profit=self.gross_profit,
            gross_loss=self.gross_loss,
        )


def build_baseline_report(
    results: Iterable[object],
    initial_cash: float,
    periods_per_year: int = 244,
) -> BaselineReport:
    days = list(results)
    if not days:
        return BaselineReport(
            start_date="",
            end_date="",
            trading_days=0,
            start_value=float(initial_cash),
            end_value=float(initial_cash),
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            buy_count=0,
            sell_count=0,
            closed_trade_count=0,
            win_rate=0.0,
            profit_loss_ratio=None,
            average_exposure=0.0,
            position_count_days={},
            full_position_days=0,
            empty_days=0,
            by_code={},
        )

    values = [float(day.total_value) for day in days]
    max_drawdown = _max_drawdown(values)
    total_return = values[-1] / float(initial_cash) - 1.0
    annualized_return = (1.0 + total_return) ** (periods_per_year / len(days)) - 1.0

    open_lots: Dict[str, _OpenLot] = {}
    stats: Dict[str, _MutableCodeStats] = {}
    buy_count = 0
    sell_count = 0

    for day in days:
        for order in getattr(day, "orders", []):
            if not getattr(order, "filled", False):
                continue
            code = str(order.code)
            amount = int(order.amount_delta)
            if amount > 0:
                buy_count += 1
                open_lots[code] = _OpenLot(
                    amount=amount,
                    cost=float(order.exec_price),
                    commission=float(order.commission),
                )
            elif amount < 0:
                sell_count += 1
                lot = open_lots.pop(code, None)
                if lot is None:
                    continue
                sell_amount = abs(amount)
                gross_pnl = sell_amount * (float(order.exec_price) - lot.cost)
                pnl = gross_pnl - lot.commission - float(order.commission)
                code_stats = stats.setdefault(code, _MutableCodeStats())
                code_stats.closed_trades += 1
                code_stats.realized_pnl += pnl
                if pnl > 0:
                    code_stats.wins += 1
                    code_stats.gross_profit += pnl
                elif pnl < 0:
                    code_stats.losses += 1
                    code_stats.gross_loss += abs(pnl)

    frozen = {code: item.freeze() for code, item in sorted(stats.items())}
    closed_trade_count = sum(item.closed_trades for item in frozen.values())
    wins = sum(item.wins for item in frozen.values())
    gross_profit = sum(item.gross_profit for item in frozen.values())
    gross_loss = sum(item.gross_loss for item in frozen.values())

    return BaselineReport(
        start_date=str(days[0].date),
        end_date=str(days[-1].date),
        trading_days=len(days),
        start_value=float(initial_cash),
        end_value=values[-1],
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        buy_count=buy_count,
        sell_count=sell_count,
        closed_trade_count=closed_trade_count,
        win_rate=wins / closed_trade_count if closed_trade_count else 0.0,
        profit_loss_ratio=gross_profit / gross_loss if gross_loss > 0 else None,
        average_exposure=_average_exposure(days),
        position_count_days=_position_count_days(days),
        full_position_days=sum(1 for day in days if len(getattr(day, "positions", {})) >= 3),
        empty_days=sum(1 for day in days if len(getattr(day, "positions", {})) == 0),
        by_code=frozen,
    )


def _max_drawdown(values: list[float]) -> float:
    peak = None
    drawdown = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            drawdown = max(drawdown, (peak - value) / peak)
    return drawdown


def _average_exposure(days: list[object]) -> float:
    total_exposure = 0.0
    total_value_sum = 0.0
    for day in days:
        positions = getattr(day, "positions", {})
        marks = getattr(day, "marks", {})
        exposure = 0.0
        for code, pos in positions.items():
            exposure += int(pos.amount) * float(marks.get(code, pos.avg_cost))
        total_value = float(day.total_value)
        total_exposure += exposure
        total_value_sum += total_value
    return total_exposure / total_value_sum if total_value_sum > 0 else 0.0


def _position_count_days(days: list[object]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for day in days:
        count = len(getattr(day, "positions", {}))
        counts[count] = counts.get(count, 0) + 1
    return dict(sorted(counts.items()))
