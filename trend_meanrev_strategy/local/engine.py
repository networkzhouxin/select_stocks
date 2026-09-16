# -*- coding: utf-8 -*-
"""趋势腿信号驱动本地回测引擎（训练期 2017-2021）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from trend_meanrev_strategy.local import indicators as ind
from trend_meanrev_strategy.local.data_loader import (
    TRAIN_END, TRAIN_START, TREND_POOL, TrendMeanrevDataLoader,
)

PARAMS = {
    "max_hold": 3,
    "base_ratio": 0.95,
    "atr_period": 14,
    "trailing_atr_mult": 2.5,
    "stop_floor": 0.05,
    "stop_cap": 0.15,
    "min_hold_days": 5,
    "cooldown_days": 5,
    "overheat_rsi": 75,
    "rsi_period": 14,
    "ma_fast": 20,
    "ma_slow": 60,
    "high_period": 20,
    "roc_period": 20,
    "commission": 0.0003,
    "min_commission": 5.0,
}


@dataclass
class Order:
    date: str
    side: str
    code: str
    price: float
    value: float
    reason: str


@dataclass
class TrainingSummary:
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
    orders: List[Order] = field(default_factory=list)


def _flag(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


class TrendLegEngine:
    def __init__(
        self,
        loader: Optional[TrendMeanrevDataLoader] = None,
        params: Optional[dict] = None,
        initial_cash: float = 20000.0,
        start=TRAIN_START,
        end=TRAIN_END,
    ):
        self.loader = loader or TrendMeanrevDataLoader()
        self.params = dict(PARAMS)
        if params:
            self.params.update(params)
        self.initial_cash = initial_cash
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self._prepare()

    # ---- 数据准备 ----

    def _prepare(self) -> None:
        p = self.params
        self.signals: Dict[str, Dict[str, dict]] = {}
        self.closes: Dict[str, Dict[str, float]] = {}
        self.minute_0935: Dict[str, Dict[str, float]] = {}
        all_dates = set()
        for code in TREND_POOL:
            daily = self.loader.load_daily(code)
            c, h, l = daily["close"], daily["high"], daily["low"]
            daily = daily.assign(
                ma_fast=ind.calc_ma(c, p["ma_fast"]),
                ma_slow=ind.calc_ma(c, p["ma_slow"]),
                atr=ind.calc_atr(h, l, c, p["atr_period"]),
                rsi=ind.calc_rsi(c, p["rsi_period"]),
                roc20=ind.calc_roc(c, p["roc_period"]),
                new_high=ind.calc_new_high(c, p["high_period"]),
                dead_cross=(ind.calc_ma(c, p["ma_fast"]) < ind.calc_ma(c, p["ma_slow"]))
                & (ind.calc_ma(c, p["ma_fast"]).shift(1) >= ind.calc_ma(c, p["ma_slow"]).shift(1)),
            )
            sig, close_map = {}, {}
            for row in daily.itertuples(index=False):
                d = row.date
                if d < self.start or d > self.end:
                    continue
                ds = d.strftime("%Y-%m-%d")
                sig[ds] = row._asdict()
                close_map[ds] = float(row.close)
                all_dates.add(d)
            self.signals[code] = sig
            self.closes[code] = close_map
            minute_map = {}
            for row in self.loader.load_minute_0935(code).itertuples(index=False):
                d = pd.Timestamp(row.date)
                if d < self.start or d > self.end:
                    continue
                minute_map[d.strftime("%Y-%m-%d")] = float(row.open)
            self.minute_0935[code] = minute_map

        self.calendar_str = sorted(d.strftime("%Y-%m-%d") for d in all_dates)
        self.date_index = {ds: i for i, ds in enumerate(self.calendar_str)}

    # ---- 主循环 ----

    def run(self) -> TrainingSummary:
        self.cash = self.initial_cash
        self.positions: Dict[str, dict] = {}
        self.orders: List[Order] = []
        self.cooldown: Dict[str, int] = {}
        self.equity: List[tuple] = []
        self.max_holdings_seen = 0
        for i in range(1, len(self.calendar_str)):
            self._trade_day(i, self.calendar_str[i], self.calendar_str[i - 1])
            self._record_equity(self.calendar_str[i])
        return self._summary()

    def _trade_day(self, i: int, ds: str, prev_ds: str) -> None:
        # 卖出阶段：ATR 止损（始终生效）→ 死叉（信号卖出，需满足最低持有）
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            held_days = i - pos["entry_index"]
            price = self.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            sig = self.signals[code].get(prev_ds)
            if held_days > 0:
                stop_price = self._calc_stop(pos, sig, price)
                if stop_price is not None and price <= stop_price:
                    self._sell(code, ds, price, "atr_stop")
                    continue
            if sig is not None and held_days >= self.params["min_hold_days"] and _flag(sig["dead_cross"]):
                self._sell(code, ds, price, "dead_cross")
                continue

        # 买入阶段：合格候选按 ROC20 降序取前 N
        qualified = []
        for code in TREND_POOL:
            if code in self.positions:
                continue
            if code in self.cooldown and i - self.cooldown[code] <= self.params["cooldown_days"]:
                continue
            sig = self.signals[code].get(prev_ds)
            if sig is None:
                continue
            if not self._buy_signal(sig):
                continue
            if not pd.isna(sig["rsi"]) and float(sig["rsi"]) > self.params["overheat_rsi"]:
                continue  # 防追高
            price = self.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            roc = float(sig["roc20"]) if not pd.isna(sig["roc20"]) else -1.0
            qualified.append((code, roc, price))
        qualified.sort(key=lambda x: (-x[1], x[0]))
        slots = self.params["max_hold"] - len(self.positions)
        for code, _roc, price in qualified[:max(0, slots)]:
            self._buy(code, ds, price)

    def _buy_signal(self, sig: dict) -> bool:
        close = sig["close"]
        ma_fast = sig["ma_fast"]
        ma_slow = sig["ma_slow"]
        for v in (close, ma_fast, ma_slow):
            if v is None or pd.isna(v):
                return False
        return _flag(sig["new_high"]) and float(close) > float(ma_fast) and float(ma_fast) > float(ma_slow)

    def _calc_stop(self, pos: dict, sig: Optional[dict], price: float) -> Optional[float]:
        if sig is None:
            return None
        atr = sig["atr"]
        if atr is None or pd.isna(atr) or float(atr) <= 0:
            return None
        highest = pos["highest"]
        if highest <= 0:
            return None
        profit = price / pos["entry_cost"] - 1 if pos["entry_cost"] > 0 else 0.0
        mult = self.params["trailing_atr_mult"]
        if profit > 0.15:
            mult *= 0.6
        elif profit > 0.05:
            mult *= 0.8
        pct = mult * float(atr) / highest
        pct = max(self.params["stop_floor"], min(self.params["stop_cap"], pct))
        return highest * (1 - pct)

    # ---- 成交 ----

    def _sell(self, code: str, ds: str, price: float, reason: str) -> None:
        pos = self.positions.pop(code)
        proceeds = pos["shares"] * price
        commission = max(proceeds * self.params["commission"], self.params["min_commission"])
        self.cash += proceeds - commission
        self.orders.append(Order(ds, "SELL", code, price, proceeds, reason))
        self.cooldown[code] = self.date_index[ds]

    def _buy(self, code: str, ds: str, price: float) -> None:
        total = self.cash + sum(p["shares"] * p["last_price"] for p in self.positions.values())
        target = total * self.params["base_ratio"] / self.params["max_hold"]
        shares = target / price
        cost = shares * price
        commission = max(cost * self.params["commission"], self.params["min_commission"])
        if cost + commission > self.cash:
            affordable = self.cash - commission
            if affordable <= 0:
                return
            shares = affordable / price
            cost = shares * price
            commission = max(cost * self.params["commission"], self.params["min_commission"])
        self.cash -= cost + commission
        self.positions[code] = {
            "shares": shares,
            "entry_cost": price,
            "entry_index": self.date_index[ds],
            "highest": price,
            "last_price": price,
        }
        self.orders.append(Order(ds, "BUY", code, price, cost, "buy"))

    # ---- 盘后与汇总 ----

    def _record_equity(self, ds: str) -> None:
        total = self.cash
        for code, pos in self.positions.items():
            close = self.closes[code].get(ds, pos["last_price"])
            total += pos["shares"] * close
            pos["last_price"] = close
            pos["highest"] = max(pos["highest"], close)
        self.equity.append((ds, total))
        self.max_holdings_seen = max(self.max_holdings_seen, len(self.positions))

    def _summary(self) -> TrainingSummary:
        end_value = self.equity[-1][1] if self.equity else self.initial_cash
        total_return = end_value / self.initial_cash - 1
        peak, max_dd = -float("inf"), 0.0
        for _, v in self.equity:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        return TrainingSummary(
            start_date=self.calendar_str[0],
            end_date=self.calendar_str[-1],
            trading_days=len(self.calendar_str),
            start_value=self.initial_cash,
            end_value=end_value,
            total_return=total_return,
            max_drawdown=max_dd,
            buy_count=sum(1 for o in self.orders if o.side == "BUY"),
            sell_count=sum(1 for o in self.orders if o.side == "SELL"),
            max_holdings=self.max_holdings_seen,
            final_holdings=sorted(self.positions.keys()),
            orders=self.orders,
        )
