# -*- coding: utf-8 -*-
"""被动配置 + 再平衡 本地回测引擎。

逻辑：不预测涨跌，按目标权重持有，季度或偏离阈值触发再平衡（机械高抛低吸）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from passive_allocation_strategy.local.data_loader import (
    TARGET_WEIGHTS, PassiveAllocationDataLoader,
)

REBALANCE_THRESHOLD = 0.05  # 任一资产偏离目标 > 5% 触发
COMMISSION = 0.0003  # 万三
MIN_COMMISSION = 5.0


@dataclass
class BacktestSummary:
    start_date: str
    end_date: str
    trading_days: int
    initial_cash: float
    final_value: float
    total_return: float
    max_drawdown: float
    rebalance_count: int
    annual_returns: Dict[str, float] = field(default_factory=dict)


class RebalanceEngine:
    def __init__(
        self,
        loader: Optional[PassiveAllocationDataLoader] = None,
        target_weights: Optional[dict] = None,
        threshold: float = REBALANCE_THRESHOLD,
        initial_cash: float = 100000.0,
        start: str = "2017-01-01",
        end: str = "2021-12-31",
    ):
        self.loader = loader or PassiveAllocationDataLoader()
        self.target_weights = dict(target_weights or TARGET_WEIGHTS)
        self.threshold = threshold
        self.initial_cash = initial_cash
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self._prepare()

    def _prepare(self) -> None:
        self.closes: Dict[str, Dict[str, float]] = {}
        all_dates = set()
        for code in self.target_weights:
            df = self.loader.load_daily(code)
            m = {}
            for d, c in zip(df["date"], df["close"]):
                ts = pd.Timestamp(d)
                if self.start <= ts <= self.end:
                    m[ts.strftime("%Y-%m-%d")] = float(c)
                    all_dates.add(ts.strftime("%Y-%m-%d"))
            self.closes[code] = m
        self.calendar = sorted(all_dates)
        self.date_index = {d: i for i, d in enumerate(self.calendar)}

    def run(self) -> BacktestSummary:
        self.shares: Dict[str, float] = {c: 0.0 for c in self.target_weights}
        self.cash = self.initial_cash
        self.equity: List[tuple] = []
        self.rebalance_count = 0

        self._rebalance(self.calendar[0])   # 初始建仓
        self._record(self.calendar[0])
        for i in range(1, len(self.calendar)):
            ds = self.calendar[i]
            if self._should_rebalance(ds, i):
                self._rebalance(ds)
            self._record(ds)
        return self._summary()

    def _price(self, code: str, ds: str) -> Optional[float]:
        return self.closes[code].get(ds)

    def _total_value(self, ds: str) -> float:
        total = self.cash
        for code, sh in self.shares.items():
            p = self._price(code, ds)
            if p:
                total += sh * p
        return total

    def _should_rebalance(self, ds: str, i: int) -> bool:
        d = pd.Timestamp(ds)
        if d.month in (1, 4, 7, 10):
            if pd.Timestamp(self.calendar[i - 1]).month != d.month:
                return True  # 季度首个交易日
        total = self._total_value(ds)
        if total <= 0:
            return False
        for code, target in self.target_weights.items():
            p = self._price(code, ds)
            if not p:
                continue
            current = self.shares[code] * p / total
            if abs(current - target) > self.threshold:
                return True
        return False

    def _rebalance(self, ds: str) -> None:
        total = self._total_value(ds)
        if total <= 0:
            return
        # 先卖超配（释放现金）
        for code, target in self.target_weights.items():
            p = self._price(code, ds)
            if not p:
                continue
            target_shares = total * target / p
            delta = target_shares - self.shares[code]
            if delta < 0:
                proceeds = -delta * p
                commission = max(proceeds * COMMISSION, MIN_COMMISSION)
                self.cash += proceeds - commission
                self.shares[code] = target_shares
        # 再买低配（现金不足时按可负担额度买入，避免一只永远买不满）
        for code, target in self.target_weights.items():
            p = self._price(code, ds)
            if not p:
                continue
            target_shares = total * target / p
            delta = target_shares - self.shares[code]
            if delta <= 0:
                continue
            cost = delta * p
            commission = max(cost * COMMISSION, MIN_COMMISSION)
            if cost + commission <= self.cash:
                self.cash -= cost + commission
                self.shares[code] = target_shares
            else:
                affordable = self.cash / (p * (1 + COMMISSION))
                if affordable > 0:
                    actual = affordable * p
                    self.cash -= actual + max(actual * COMMISSION, MIN_COMMISSION)
                    self.shares[code] += affordable
        self.rebalance_count += 1

    def _record(self, ds: str) -> None:
        self.equity.append((ds, self._total_value(ds)))

    def _summary(self) -> BacktestSummary:
        final_value = self.equity[-1][1]
        total_return = final_value / self.initial_cash - 1
        peak, max_dd = -float("inf"), 0.0
        for _, v in self.equity:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        by_year: Dict[str, List[float]] = {}
        for ds, v in self.equity:
            by_year.setdefault(ds[:4], []).append(v)
        annual = {}
        prev = self.initial_cash
        for y in sorted(by_year):
            annual[y] = by_year[y][-1] / prev - 1
            prev = by_year[y][-1]
        return BacktestSummary(
            start_date=self.calendar[0],
            end_date=self.calendar[-1],
            trading_days=len(self.calendar),
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return=total_return,
            max_drawdown=max_dd,
            rebalance_count=self.rebalance_count,
            annual_returns=annual,
        )
