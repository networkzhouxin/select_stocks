# -*- coding: utf-8 -*-
"""Training-window local replay helpers for cross_signal_strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from cross_signal_strategy.local_adjustment import default_training_adjustment_factors
from cross_signal_strategy.local_data_loader import APPROVED_WARMUP_ROOT
from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter


@dataclass(frozen=True)
class TrainingReplaySummary:
    start_date: str
    end_date: str
    trading_days: int
    start_value: float
    end_value: float
    total_return: float
    max_drawdown: float
    buy_count: int
    sell_count: int
    max_holdings: int
    final_holdings: List[str]
    order_dates: List[str]


def get_training_trade_dates(loader, reference_code: str = "510300") -> List[str]:
    frames = [
        loader.load_daily_frame(reference_code, f"{year}-12-31")
        for year in [2019, 2020, 2021]
    ]
    dates = pd.concat(frames, ignore_index=True)["date"].astype(str).drop_duplicates()
    dates = sorted(dates.tolist())
    if dates and (dates[0] < "2019-01-01" or dates[-1] > "2021-12-31"):
        raise ValueError("Training dates outside 2019-01-01 to 2021-12-31")
    return dates


def run_training_replay(loader, initial_cash: float = 20000.0, warmup_root=APPROVED_WARMUP_ROOT) -> TrainingReplaySummary:
    trade_dates = get_training_trade_dates(loader)
    adapter = LocalSignalAdapter(
        loader,
        warmup_root=warmup_root,
        adjustment_factors=default_training_adjustment_factors(),
    )
    planner = LocalCrossSignalOrderPlanner(adapter)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)

    values = [day.total_value for day in results]
    peak = None
    max_drawdown = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    buy_count = 0
    sell_count = 0
    order_dates = []
    for day in results:
        for order in day.orders:
            if not order.filled:
                continue
            order_dates.append(day.date)
            if order.amount_delta > 0:
                buy_count += 1
            elif order.amount_delta < 0:
                sell_count += 1

    max_holdings = max((len(day.positions) for day in results), default=0)
    final_holdings = sorted(results[-1].positions.keys()) if results else []
    end_value = values[-1] if values else initial_cash
    return TrainingReplaySummary(
        start_date=trade_dates[0],
        end_date=trade_dates[-1],
        trading_days=len(trade_dates),
        start_value=float(initial_cash),
        end_value=float(end_value),
        total_return=float(end_value / initial_cash - 1.0),
        max_drawdown=float(max_drawdown),
        buy_count=buy_count,
        sell_count=sell_count,
        max_holdings=max_holdings,
        final_holdings=final_holdings,
        order_dates=order_dates,
    )
