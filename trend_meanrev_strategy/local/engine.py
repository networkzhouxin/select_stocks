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
    # 利润分段收紧：(利润阈值, ATR倍数缩放)，从高到低。默认 5-15%→0.8x、>15%→0.6x
    "profit_tiers": ((0.15, 0.6), (0.05, 0.8)),
    "min_hold_days": 5,
    "overheat_rsi": 75,
    "rsi_min": None,  # 买入要求 RSI >= 该值（趋势强度下限），None=关闭
    "rsi_period": 14,
    "ma_fast": 20,
    "ma_slow": 60,
    "high_period": 20,
    "roc_period": 20,
    "commission": 0.0003,
    "min_commission": 5.0,
    # 压力半仓（H1）：lookback 天内 >= min_stops 次 ATR 止损 → 新买入 × stress_buy_scale
    "stress_lookback_days": 15,
    "stress_min_stops": 0,  # 0 = 关闭
    "stress_buy_scale": 0.50,
    # 熊市过滤：510300 收盘<MA60 且 MA60 下行 → A股ETF 暂停新买入
    "bear_filter": True,
    # MA10 实验开关
    "triple_ma_buy": False,  # 买入要求 MA10>MA20>MA60 三均线排列
    "ma10_exit": False,      # 卖出加：收盘<MA10 且 MA10 下行
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
        pool=None,
    ):
        self.loader = loader or TrendMeanrevDataLoader()
        self.params = dict(PARAMS)
        if params:
            self.params.update(params)
        self.pool = list(pool) if pool is not None else list(TREND_POOL)
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
        for code in self.pool:
            daily = self.loader.load_daily(code)
            c, h, l = daily["close"], daily["high"], daily["low"]
            daily = daily.assign(
                ma10=ind.calc_ma(c, 10),
                ma10_prev=ind.calc_ma(c, 10).shift(1),
                ma_fast=ind.calc_ma(c, p["ma_fast"]),
                ma_slow=ind.calc_ma(c, p["ma_slow"]),
                ma_slow_prev=ind.calc_ma(c, p["ma_slow"]).shift(1),
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
        self.equity: List[tuple] = []
        self.atr_stop_history: List[int] = []
        self.trade_stats: List[tuple] = []
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
            if (self.params.get("ma10_exit", False) and sig is not None
                    and held_days >= self.params["min_hold_days"]):
                ma10 = sig.get("ma10")
                ma10_prev = sig.get("ma10_prev")
                if (ma10 is not None and ma10_prev is not None
                        and not pd.isna(ma10) and not pd.isna(ma10_prev)
                        and float(sig["close"]) < float(ma10) and float(ma10) < float(ma10_prev)):
                    self._sell(code, ds, price, "ma10_exit")
                    continue

        # 买入阶段：合格候选按 ROC20 降序取前 N
        bear = False
        if self.params.get("bear_filter", False):
            s510 = self.signals.get("510300", {}).get(prev_ds)
            if s510 is not None:
                bear = (s510["close"] < s510["ma_slow"]) and (s510["ma_slow"] < s510["ma_slow_prev"])
        a_share = {"510300", "159915", "512100", "159928"}
        qualified = []
        for code in self.pool:
            if code in self.positions:
                continue
            if bear and code in a_share:
                continue
            sig = self.signals[code].get(prev_ds)
            if sig is None:
                continue
            if not self._buy_signal(sig):
                continue
            if not pd.isna(sig["rsi"]) and float(sig["rsi"]) > self.params["overheat_rsi"]:
                continue  # 防追高
            rsi_min = self.params.get("rsi_min")
            if rsi_min is not None and not pd.isna(sig["rsi"]) and float(sig["rsi"]) < rsi_min:
                continue  # 趋势太弱
            price = self.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            roc = float(sig["roc20"]) if not pd.isna(sig["roc20"]) else -1.0
            qualified.append((code, roc, price))
        qualified.sort(key=lambda x: (-x[1], x[0]))
        slots = self.params["max_hold"] - len(self.positions)
        for code, roc, price in qualified[:max(0, slots)]:
            self._buy(code, ds, price, roc)

    def _buy_signal(self, sig: dict) -> bool:
        ma_fast = sig["ma_fast"]
        ma_slow = sig["ma_slow"]
        for v in (ma_fast, ma_slow):
            if v is None or pd.isna(v):
                return False
        if self.params.get("triple_ma_buy", False):
            ma10 = sig.get("ma10")
            if ma10 is None or pd.isna(ma10):
                return False
            return _flag(sig["new_high"]) and float(ma10) > float(ma_fast) > float(ma_slow)
        return _flag(sig["new_high"]) and float(ma_fast) > float(ma_slow)

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
        for thr, scale in self.params["profit_tiers"]:
            if profit > thr:
                mult *= scale
                break
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
        if reason == "atr_stop":
            self.atr_stop_history.append(self.date_index[ds])
        final_pnl = price / pos["entry_cost"] - 1 if pos["entry_cost"] > 0 else 0.0
        hold_days = self.date_index[ds] - pos["entry_index"]
        self.trade_stats.append((code, hold_days, pos.get("buy_roc", 0.0), pos["peak_pnl"], final_pnl, reason))

    def _stress_scale(self, current_idx: int) -> float:
        p = self.params
        lookback = p.get("stress_lookback_days", 0)
        min_stops = p.get("stress_min_stops", 0)
        if lookback <= 0 or min_stops <= 0:
            return 1.0
        recent = sum(1 for s in self.atr_stop_history if 0 <= current_idx - s <= lookback)
        if recent < min_stops:
            return 1.0
        return float(p.get("stress_buy_scale", 1.0))

    def _buy(self, code: str, ds: str, price: float, roc: float = 0.0) -> None:
        total = self.cash + sum(p["shares"] * p["last_price"] for p in self.positions.values())
        target = total * self.params["base_ratio"] / self.params["max_hold"]
        target *= self._stress_scale(self.date_index[ds])
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
            "peak_pnl": 0.0,
            "buy_roc": roc,
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
            pnl = close / pos["entry_cost"] - 1 if pos["entry_cost"] > 0 else 0.0
            pos["peak_pnl"] = max(pos["peak_pnl"], pnl)
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
