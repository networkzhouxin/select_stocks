# -*- coding: utf-8 -*-
"""Trade-level attribution diagnostics for cross-signal local training replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple

from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)


ScoreKey = Tuple[str, str]


@dataclass
class DiagnosticOrderPlanner(LocalCrossSignalOrderPlanner):
    """Planner variant that freezes buy-score snapshots at order-planning time."""

    entry_score_snapshots: Dict[ScoreKey, dict] = field(default_factory=dict)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = super().plan_orders(current_date, previous_date, broker, current_prices=current_prices)
        for order in orders:
            if order.get("reason") != "buy_signal":
                continue
            code = str(order["code"]).split(".")[0]
            score = self.last_scores.get(code)
            if score is not None:
                self.entry_score_snapshots[(str(current_date), code)] = dict(score)
        return orders


@dataclass(frozen=True)
class ClosedTradeDiagnostic:
    code: str
    buy_date: str
    sell_date: str
    sell_reason: str
    amount: int
    buy_price: float
    sell_price: float
    pnl: float
    return_pct: float
    entry_score: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _OpenTrade:
    date: str
    price: float
    amount: int
    commission: float
    entry_score: Mapping[str, object]


def build_closed_trade_diagnostics(
    results: Iterable[object],
    entry_score_snapshots: Mapping[ScoreKey, Mapping[str, object]],
) -> List[ClosedTradeDiagnostic]:
    open_trades: Dict[str, _OpenTrade] = {}
    closed: List[ClosedTradeDiagnostic] = []

    for day in results:
        day_date = str(day.date)
        for order in getattr(day, "orders", []):
            if not getattr(order, "filled", False):
                continue
            code = str(order.code).split(".")[0]
            amount_delta = int(order.amount_delta)
            if amount_delta > 0:
                open_trades[code] = _OpenTrade(
                    date=day_date,
                    price=float(order.exec_price),
                    amount=amount_delta,
                    commission=float(order.commission),
                    entry_score=dict(entry_score_snapshots.get((day_date, code), {})),
                )
            elif amount_delta < 0:
                open_trade = open_trades.pop(code, None)
                if open_trade is None:
                    continue
                sell_amount = abs(amount_delta)
                pnl = sell_amount * (float(order.exec_price) - open_trade.price)
                pnl -= open_trade.commission + float(order.commission)
                return_pct = (float(order.exec_price) / open_trade.price - 1.0) * 100.0
                closed.append(
                    ClosedTradeDiagnostic(
                        code=code,
                        buy_date=open_trade.date,
                        sell_date=day_date,
                        sell_reason=str(getattr(order, "reason", "")),
                        amount=sell_amount,
                        buy_price=open_trade.price,
                        sell_price=float(order.exec_price),
                        pnl=pnl,
                        return_pct=return_pct,
                        entry_score=open_trade.entry_score,
                    )
                )
    return closed


def run_training_trade_diagnostics(
    loader=None,
    initial_cash: float = 20000.0,
) -> List[ClosedTradeDiagnostic]:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    return build_closed_trade_diagnostics(results, planner.entry_score_snapshots)
