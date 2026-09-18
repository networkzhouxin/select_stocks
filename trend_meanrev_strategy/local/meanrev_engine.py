# -*- coding: utf-8 -*-
"""均值回归腿信号驱动本地回测引擎（训练期 2017-2021）。

买入：布尔共振（BOLL 超卖收回 + RSI/KDJ 至少一个低位拐头 + 第三指标反向否决）。
卖出：回归中轨了结 / 超买了结 / 时间止损 / ATR 止损（先到先卖，OR 逻辑）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trend_meanrev_strategy.local import indicators as ind
from trend_meanrev_strategy.local import meanrev_indicators as mi
from trend_meanrev_strategy.local.data_loader import (
    MEANREV_POOL, TRAIN_END, TRAIN_START, TrendMeanrevDataLoader,
)

PARAMS = {
    "max_hold": 2,
    "base_ratio": 0.95,
    "atr_period": 14,
    "trailing_atr_mult": 2.0,
    "stop_floor": 0.05,
    "stop_cap": 0.15,
    "min_hold_days": 5,
    "cooldown_days": 0,
    "time_stop_days": 10,
    "overheat_rsi": 75,
    "rsi_period": 14,
    "rsi_low": 30,
    "rsi_high": 70,
    "kdj_n": 9,
    "kdj_m1": 3,
    "kdj_m2": 3,
    "kdj_low": 20,
    "kdj_high": 80,
    "boll_period": 20,
    "boll_std": 2.0,
    "ma_fast": 20,
    "ma_slow": 60,
    "knife_mode": "none",
    "require_boll": True,
    "no_boll_oversold_rsi": 25.0,
    "dmi_period": 14,
    "dmi_exit_enabled": False,
    "dmi_adx_threshold": 25.0,
    "sell_mode": "mid",
    "relative_backfill_enabled": False,
    "buy_mode": "resonance",
    "divergence_sell_enabled": False,
    "divergence_window": 10,
    "volume_veto_enabled": False,
    "volume_ratio_threshold": 1.0,
    "commission": 0.0003,
    "min_commission": 5.0,
}

# 品种级止损覆盖（V2.8 验证：黄金收紧）
CODE_STOP_PARAMS = {
    "518880": {"stop_floor": 0.03, "trailing_atr_mult": 2.0},
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


class MeanrevEngine:
    def __init__(
        self,
        loader: Optional[TrendMeanrevDataLoader] = None,
        params: Optional[dict] = None,
        initial_cash: float = 20000.0,
        start=TRAIN_START,
        end=TRAIN_END,
        pool: Optional[list] = None,
    ):
        self.loader = loader or TrendMeanrevDataLoader()
        self.params = dict(PARAMS)
        if params:
            self.params.update(params)
        self.initial_cash = initial_cash
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.pool = list(pool) if pool is not None else list(MEANREV_POOL)
        self._prepare()

    # ---- 数据与事件准备 ----

    def _prepare(self) -> None:
        p = self.params
        self.signals: Dict[str, Dict[str, dict]] = {}
        self.closes: Dict[str, Dict[str, float]] = {}
        self.minute_0935: Dict[str, Dict[str, float]] = {}
        all_dates = set()
        for code in self.pool:
            daily = self.loader.load_daily(code)
            c, h, l = daily["close"], daily["high"], daily["low"]
            ma_fast = ind.calc_ma(c, p["ma_fast"])
            ma_slow = ind.calc_ma(c, p["ma_slow"])
            atr = ind.calc_atr(h, l, c, p["atr_period"])
            rsi = ind.calc_rsi(c, p["rsi_period"])
            mid, upper, lower = mi.calc_boll(c, p["boll_period"], p["boll_std"])
            k, d, j = mi.calc_kdj(h, l, c, p["kdj_n"], p["kdj_m1"], p["kdj_m2"])
            plus_di, minus_di, adx = mi.calc_dmi_adx(h, l, c, p["dmi_period"])

            touch_lower = (l <= lower) | (c <= lower)
            kd_diff = k - d
            boll_buy = (touch_lower | touch_lower.shift(1)) & (c > lower) & (c > c.shift(1))
            rsi_buy = (rsi.shift(1) <= p["rsi_low"]) & (rsi > rsi.shift(1))
            kdj_low_prev = (k.shift(1) <= p["kdj_low"]) | (d.shift(1) <= p["kdj_low"]) | (j.shift(1) <= 0)
            kdj_buy = kdj_low_prev & (j > j.shift(1)) & (kd_diff > kd_diff.shift(1))
            rsi_sell = (rsi.shift(1) >= p["rsi_high"]) & (rsi < rsi.shift(1))
            kdj_high_prev = (k.shift(1) >= p["kdj_high"]) | (d.shift(1) >= p["kdj_high"]) | (j.shift(1) >= 100)
            kdj_sell = kdj_high_prev & (j < j.shift(1)) & (kd_diff < kd_diff.shift(1))
            knife = self._calc_knife(c, ma_fast, ma_slow)

            percent_b = (c - lower) / (upper - lower).replace(0, np.nan)
            rsi_rel_buy = (rsi.shift(2) >= rsi.shift(1)) & (rsi > rsi.shift(1))
            j_rel = (j.shift(2) >= j.shift(1)) & (j > j.shift(1))
            diff_rel = (kd_diff.shift(2) >= kd_diff.shift(1)) & (kd_diff > kd_diff.shift(1))
            kdj_rel_buy = j_rel & diff_rel
            boll_rel_buy = (
                (c.shift(1) < mid.shift(1))
                & (percent_b.shift(2) >= percent_b.shift(1))
                & (percent_b > percent_b.shift(1))
                & (c > c.shift(1))
                & (l >= l.shift(1))
            )

            vol = pd.to_numeric(daily["volume"], errors="coerce")
            volume_ratio = vol / vol.rolling(20).mean()

            div_win = int(p.get("divergence_window", 10))
            divergence = (
                (c > c.rolling(div_win).max().shift(1))
                & ~(rsi > rsi.rolling(div_win).max().shift(1))
            )

            daily = daily.assign(
                ma_fast=ma_fast, ma_slow=ma_slow, atr=atr, rsi=rsi,
                mid=mid, upper=upper, lower=lower,
                plus_di=plus_di, minus_di=minus_di, adx=adx,
                boll_buy=boll_buy, rsi_buy=rsi_buy, kdj_buy=kdj_buy,
                rsi_sell=rsi_sell, kdj_sell=kdj_sell, knife=knife,
                boll_rel_buy=boll_rel_buy, rsi_rel_buy=rsi_rel_buy, kdj_rel_buy=kdj_rel_buy,
                volume_ratio=volume_ratio, divergence=divergence,
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

    def _calc_knife(self, c: pd.Series, ma_fast: pd.Series, ma_slow: pd.Series) -> pd.Series:
        """接飞刀防护：识别趋势性下跌（均值回归最怕的接飞刀场景）。"""
        mode = self.params["knife_mode"]
        if mode == "none":
            return pd.Series(False, index=c.index)
        if mode == "ma60_slope_weak":
            return (c < ma_slow) & (ma_fast < ma_fast.shift(1)) & (c < ma_fast)
        if mode == "new_low_10":
            return c < c.rolling(10).min().shift(1)
        return (c < ma_slow) & (ma_fast < ma_fast.shift(1))

    # ---- 主循环 ----

    def run(self) -> TrainingSummary:
        self.cash = self.initial_cash
        self.positions: Dict[str, dict] = {}
        self.orders: List[Order] = []
        self.cooldown: Dict[str, int] = {}
        self.equity: List[tuple] = []
        self.max_holdings_seen = 0
        for i in range(1, len(self.calendar_str)):
            prev2 = self.calendar_str[i - 2] if i >= 2 else None
            self._trade_day(i, self.calendar_str[i], self.calendar_str[i - 1], prev2)
            self._record_equity(self.calendar_str[i])
        return self._summary()

    def _trade_day(self, i: int, ds: str, prev_ds: str, prev2_ds: Optional[str]) -> None:
        # 卖出阶段（先到先卖）
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            held_days = i - pos["entry_index"]
            price = self.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            sig = self.signals[code].get(prev_ds)
            if held_days > 0 and sig is not None:
                stop_price = self._calc_stop(pos, sig)
                if stop_price is not None and price <= stop_price:
                    self._sell(code, ds, price, "atr_stop")
                    continue
            if sig is not None and held_days >= self.params["min_hold_days"]:
                if self.params["divergence_sell_enabled"]:
                    if _flag(sig.get("divergence")):
                        self._sell(code, ds, price, "divergence")
                        continue
                elif self.params["sell_mode"] == "top":
                    if self._top_signal_exit(sig):
                        self._sell(code, ds, price, "top_signal")
                        continue
                else:
                    if self._mean_revert_exit(sig):
                        self._sell(code, ds, price, "mean_revert")
                        continue
                    if self._overbought_exit(sig):
                        self._sell(code, ds, price, "overbought")
                        continue
            if (self.params["dmi_exit_enabled"] and held_days >= 2
                    and sig is not None):
                s2 = self.signals[code].get(prev2_ds) if prev2_ds else None
                if self._dmi_downtrend_exit(sig, s2):
                    self._sell(code, ds, price, "dmi_downtrend")
                    continue
            if held_days >= self.params["time_stop_days"]:
                self._sell(code, ds, price, "time_stop")
                continue

        # 买入阶段
        formal, relative = [], []
        for code in self.pool:
            if code in self.positions:
                continue
            if code in self.cooldown and i - self.cooldown[code] <= self.params["cooldown_days"]:
                continue
            s1 = self.signals[code].get(prev_ds)
            if s1 is None:
                continue
            s2 = self.signals[code].get(prev2_ds) if prev2_ds else None
            if _flag(s1.get("knife")):
                continue  # 接飞刀防护
            rsi = s1.get("rsi")
            if rsi is not None and not pd.isna(rsi) and float(rsi) > self.params["overheat_rsi"]:
                continue  # 防追高
            if self.params["volume_veto_enabled"]:
                vr = s1.get("volume_ratio")
                if vr is not None and not pd.isna(vr) and float(vr) >= self.params["volume_ratio_threshold"]:
                    continue  # 放量不买（缩量=抛压衰竭）
            price = self.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            if self.params["buy_mode"] == "relative":
                if self._relative_buy_signal(s1, s2):
                    relative.append((code, price, False))
                continue
            ok, all_multi = self._buy_resonance(s1, s2)
            if ok:
                formal.append((code, price, all_multi))
                continue
            if self.params["relative_backfill_enabled"] and self._relative_buy_signal(s1, s2):
                relative.append((code, price, False))

        formal.sort(key=lambda x: (not x[2], x[0]))
        relative.sort(key=lambda x: x[0])
        candidates = formal + relative
        slots = self.params["max_hold"] - len(self.positions)
        for code, price, _all_multi in candidates[:max(0, slots)]:
            self._buy(code, ds, price)

    # ---- 买入共振 ----

    def _buy_resonance(self, s1: dict, s2: Optional[dict]) -> Tuple[bool, bool]:
        """返回 (是否合格, 是否多指标同向)。事件窗口 2 交易日，至少一个支持事件在 T-1。

        require_boll=True：BOLL 触下轨收回必选（现状）。
        require_boll=False：新增通道，RSI/KDJ 超卖反转即可，不要求 BOLL。
        """
        def f(row, key):
            return row is not None and _flag(row.get(key))

        boll_t1, boll_t2 = f(s1, "boll_buy"), f(s2, "boll_buy")
        rsi_t1, rsi_t2 = f(s1, "rsi_buy"), f(s2, "rsi_buy")
        kdj_t1, kdj_t2 = f(s1, "kdj_buy"), f(s2, "kdj_buy")
        boll_ok = boll_t1 or boll_t2
        rsi_ok = rsi_t1 or rsi_t2
        kdj_ok = kdj_t1 or kdj_t2
        if not (rsi_ok or kdj_ok):
            return False, False
        req = self.params["require_boll"]
        if req and not boll_ok:
            return False, False
        if not req:
            rsi_val = s1.get("rsi")
            if rsi_val is None or pd.isna(rsi_val) or float(rsi_val) >= self.params["no_boll_oversold_rsi"]:
                return False, False
        fresh = (boll_t1 or rsi_t1 or kdj_t1) if req else (rsi_t1 or kdj_t1)
        if not fresh:
            return False, False
        rsi_sell_ok = f(s1, "rsi_sell") or f(s2, "rsi_sell")
        kdj_sell_ok = f(s1, "kdj_sell") or f(s2, "kdj_sell")
        if rsi_ok and not kdj_ok and kdj_sell_ok:
            return False, False
        if kdj_ok and not rsi_ok and rsi_sell_ok:
            return False, False
        all_multi = (boll_ok and rsi_ok and kdj_ok) if req else (rsi_ok and kdj_ok)
        return True, all_multi

    def _relative_buy_signal(self, s1: dict, s2: Optional[dict]) -> bool:
        """相对拐点补位（借鉴 resonance SOFT_ALL_THREE）：相对 BOLL/RSI/KDJ 三项齐全。"""
        def f(row, key):
            return row is not None and _flag(row.get(key))

        boll_ok = f(s1, "boll_rel_buy") or f(s2, "boll_rel_buy")
        rsi_ok = f(s1, "rsi_rel_buy") or f(s2, "rsi_rel_buy")
        kdj_ok = f(s1, "kdj_rel_buy") or f(s2, "kdj_rel_buy")
        if not (boll_ok and rsi_ok and kdj_ok):
            return False
        if not (f(s1, "boll_rel_buy") or f(s1, "rsi_rel_buy") or f(s1, "kdj_rel_buy")):
            return False
        return True

    # ---- 卖出了结 ----

    def _mean_revert_exit(self, sig: dict) -> bool:
        close, mid = sig.get("close"), sig.get("mid")
        if close is None or mid is None or pd.isna(close) or pd.isna(mid):
            return False
        return float(close) >= float(mid)

    def _top_signal_exit(self, sig: dict) -> bool:
        """顶部信号卖出：RSI 超买 或 KDJ 高位拐头（涨不动了就卖，不强制中轨）。"""
        rsi = sig.get("rsi")
        if rsi is not None and not pd.isna(rsi) and float(rsi) >= self.params["rsi_high"]:
            return True
        if _flag(sig.get("kdj_sell")):
            return True
        return False

    def _overbought_exit(self, sig: dict) -> bool:
        rsi = sig.get("rsi")
        if rsi is None or pd.isna(rsi):
            return False
        return float(rsi) >= self.params["rsi_high"]

    def _dmi_downtrend_exit(self, s1: dict, s2: Optional[dict]) -> bool:
        """借鉴 resonance：连续2日 -DI>+DI 且 ADX>阈值 且 ADX上升 且 close下跌。"""
        if s1 is None or s2 is None:
            return False
        vals = []
        for row in (s1, s2):
            for key in ("minus_di", "plus_di", "adx", "close"):
                v = row.get(key)
                if v is None or pd.isna(v):
                    return False
                vals.append(float(v))
        minus1, plus1, adx1, close1, minus2, plus2, adx2, close2 = vals
        th = self.params["dmi_adx_threshold"]
        return (minus1 > plus1 and adx1 > th
                and minus2 > plus2 and adx2 > th
                and adx1 > adx2
                and close1 < close2)

    def _calc_stop(self, pos: dict, sig: dict) -> Optional[float]:
        atr = sig.get("atr")
        if atr is None or pd.isna(atr) or float(atr) <= 0:
            return None
        highest = pos["highest"]
        if highest <= 0:
            return None
        sp = CODE_STOP_PARAMS.get(pos["code"], {})
        mult = sp.get("trailing_atr_mult", self.params["trailing_atr_mult"])
        floor = sp.get("stop_floor", self.params["stop_floor"])
        cap = self.params["stop_cap"]
        pct = mult * float(atr) / highest
        pct = max(floor, min(cap, pct))
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
            "code": code,
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
