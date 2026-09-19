# -*- coding: utf-8 -*-
"""被动配置 + 再平衡 本地回测引擎。

逻辑：不预测涨跌，按目标权重持有，季度或偏离阈值触发再平衡（机械高抛低吸）。
债 = 现金（货基/逆回购近似，年化 ~2%），不买国债 ETF（一手 1.1 万，小资金买不进且会跌）。
执行口径对齐交易规则底线：整手（100股）+ 滑点 0.1% + 万三/最低5元佣金。
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
LOT_SIZE = 100  # 一手 = 100 股，禁止分数股（对齐交易规则底线）
SLIPPAGE = 0.001  # 滑点 0.1%，对齐聚宽 PriceRelatedSlippage(0.001)

BOND_CODE = "511010"  # 债券压舱石 = 现金（货基/逆回购），不买国债 ETF
BOND_RATE = 0.02      # 货基/逆回购年化 ~2%
TRADING_DAYS = 244    # A股年交易日数


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
        bond_rate: float = BOND_RATE,
    ):
        self.loader = loader or PassiveAllocationDataLoader()
        self.target_weights = dict(target_weights or TARGET_WEIGHTS)
        self.threshold = threshold
        self.initial_cash = initial_cash
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.bond_rate = bond_rate
        self.bond_code = BOND_CODE
        self.equity_codes = [c for c in self.target_weights if c != self.bond_code]
        self._prepare()

    def _prepare(self) -> None:
        self.closes: Dict[str, Dict[str, float]] = {}
        all_dates = set()
        for code in self.equity_codes:
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

    def _daily_rate(self) -> float:
        return (1 + self.bond_rate) ** (1 / TRADING_DAYS) - 1

    def run(self) -> BacktestSummary:
        self.shares: Dict[str, float] = {c: 0.0 for c in self.equity_codes}
        self.cash = self.initial_cash
        self.equity: List[tuple] = []
        self.rebalance_count = 0
        self.net_invested: Dict[str, float] = {c: 0.0 for c in self.equity_codes}
        self.interest_accrued = 0.0
        self.attribution: List[tuple] = []   # (ds, {code: 贡献}, 累计利息)

        self._rebalance(self.calendar[0])   # 初始建仓
        self._record(self.calendar[0])
        self._record_attribution(self.calendar[0])
        for i in range(1, len(self.calendar)):
            ds = self.calendar[i]
            interest = self.cash * self._daily_rate()   # 货基每日计息
            self.cash += interest
            self.interest_accrued += interest
            if self._should_rebalance(ds, i):
                self._rebalance(ds)
            self._record(ds)
            self._record_attribution(ds)
        return self._summary()

    def _record_attribution(self, ds: str) -> None:
        contrib = {}
        for code in self.equity_codes:
            p = self._price(code, ds)
            contrib[code] = (self.shares[code] * p - self.net_invested[code]) if p else 0.0
        self.attribution.append((ds, contrib, self.interest_accrued))

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
        for code in self.equity_codes:
            p = self._price(code, ds)
            if not p:
                continue
            current = self.shares[code] * p / total
            if abs(current - self.target_weights[code]) > self.threshold:
                return True
        cash_weight = self.cash / total
        if abs(cash_weight - self.target_weights.get(self.bond_code, 0.0)) > self.threshold:
            return True
        return False

    def _delta_lot(self, code: str, total: float, p: float) -> int:
        """按比例算目标股数，与当前持仓的差向零取整到整手（差 < 1 手返回 0）。"""
        delta_float = total * self.target_weights[code] / p - self.shares[code]
        delta = int(abs(delta_float) // LOT_SIZE) * LOT_SIZE
        return -delta if delta_float < 0 else delta

    def _rebalance(self, ds: str) -> None:
        total = self._total_value(ds)
        if total <= 0:
            return
        # 先卖超配（释放现金），卖出价 = 收盘价 × (1 - 滑点)
        for code in self.equity_codes:
            p = self._price(code, ds)
            if not p:
                continue
            delta = self._delta_lot(code, total, p)
            if delta < 0:
                sell_price = p * (1 - SLIPPAGE)
                proceeds = -delta * sell_price
                commission = max(proceeds * COMMISSION, MIN_COMMISSION)
                net_proceeds = proceeds - commission
                self.cash += net_proceeds
                self.net_invested[code] -= net_proceeds
                self.shares[code] += delta
        # 再买低配（现金不足时按可负担的整手数买入），买入价 = 收盘价 × (1 + 滑点)
        for code in self.equity_codes:
            p = self._price(code, ds)
            if not p:
                continue
            delta = self._delta_lot(code, total, p)
            if delta <= 0:
                continue
            buy_price = p * (1 + SLIPPAGE)
            cost = delta * buy_price
            commission = max(cost * COMMISSION, MIN_COMMISSION)
            if cost + commission <= self.cash:
                total_cost = cost + commission
                self.cash -= total_cost
                self.net_invested[code] += total_cost
                self.shares[code] += delta
            else:
                lots = int(self.cash // (buy_price * LOT_SIZE * (1 + COMMISSION)))
                if lots > 0:
                    shares = lots * LOT_SIZE
                    actual = shares * buy_price
                    total_cost = actual + max(actual * COMMISSION, MIN_COMMISSION)
                    self.cash -= total_cost
                    self.net_invested[code] += total_cost
                    self.shares[code] += shares
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
