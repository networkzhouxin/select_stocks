# -*- coding: utf-8 -*-
"""Frozen KRBA research-candidate primitives.

This module is isolated from the formal JoinQuant/PTrade strategies.  It uses
completed T-1 daily bars for signals and keeps the entry ATR fixed for the
whole holding period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import types
from typing import Iterable, Mapping

import pandas as pd

sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as core
from cross_signal_strategy.local.local_signal_adapter import LocalSignalAdapter


VERSION = "krba-v0.1-candidate"


def _number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def is_entry_eligible(snapshot: Mapping[str, object]) -> bool:
    """Return the frozen same-session K/D + RSI + lower-band decision."""
    k_prev = _number(snapshot.get("k_prev"))
    d_prev = _number(snapshot.get("d_prev"))
    k = _number(snapshot.get("k"))
    d = _number(snapshot.get("d"))
    rsi6 = _number(snapshot.get("rsi6"))
    low = _number(snapshot.get("low"))
    close = _number(snapshot.get("close"))
    lower = _number(snapshot.get("boll_lower"))
    values = (k_prev, d_prev, k, d, rsi6, low, close, lower)
    if not all(math.isfinite(item) for item in values):
        return False
    return bool(
        k_prev <= d_prev
        and k > d
        and rsi6 <= 30.0
        and low <= lower
        and close > lower
    )


def _kd_cross_down(snapshot: Mapping[str, object]) -> bool:
    k_prev = _number(snapshot.get("k_prev"))
    d_prev = _number(snapshot.get("d_prev"))
    k = _number(snapshot.get("k"))
    d = _number(snapshot.get("d"))
    return bool(
        all(math.isfinite(item) for item in (k_prev, d_prev, k, d))
        and k_prev >= d_prev
        and k < d
    )


@dataclass
class PositionSignalState:
    entry_date: str
    entry_price: float
    entry_atr: float
    highest_close: float
    mean_reached: bool = False
    upper_reached: bool = False


def calc_frozen_atr_stop(
    state: PositionSignalState,
    code: str = "",
    multiplier: float = 2.5,
    floor: float = 0.05,
    cap: float = 0.15,
) -> float:
    if str(code).split(".")[0] == "518880":
        floor = 0.03
    if state.highest_close <= 0 or state.entry_atr <= 0:
        return state.entry_price * (1.0 - cap)
    distance = multiplier * state.entry_atr / state.highest_close
    distance = max(floor, min(cap, distance))
    return state.highest_close * (1.0 - distance)


def choose_exit_reason(
    state: PositionSignalState,
    snapshot: Mapping[str, object],
    current_price: float,
    hold_days: int,
    code: str = "",
) -> str | None:
    price = _number(current_price)
    if math.isfinite(price) and price > 0:
        if round(price, 3) <= round(calc_frozen_atr_stop(state, code), 3):
            return "atr_stop"
    if int(hold_days) < 5:
        return None
    if state.upper_reached:
        return "boll_upper_target"
    close = _number(snapshot.get("close"))
    mid = _number(snapshot.get("boll_mid"))
    if state.mean_reached and (
        _kd_cross_down(snapshot)
        or (math.isfinite(close) and math.isfinite(mid) and close < mid)
    ):
        return "mean_reached_weakness"
    return None


@dataclass
class KRBASignalAdapter:
    loader: object
    warmup_root: object | None = None
    adjustment_factors: object | None = None
    daily_corrections: object | None = None
    _base: LocalSignalAdapter = field(init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._base = LocalSignalAdapter(
            self.loader,
            warmup_root=self.warmup_root,
            adjustment_factors=self.adjustment_factors,
            daily_corrections=self.daily_corrections,
        )

    def score(self, code: str, current_date: str, return_reason: bool = False):
        key = (str(code).split(".")[0], str(current_date))
        if key not in self._cache:
            self._cache[key] = self._score_uncached(*key)
        result, reason = self._cache[key]
        copied = dict(result) if result is not None else None
        return (copied, reason) if return_reason else copied

    def _score_uncached(self, code: str, current_date: str):
        frame, signal_date = self._base.load_signal_frame(code, current_date)
        if signal_date is None:
            return None, "no_previous_trade_date"
        required = {"date", "high", "low", "close"}
        if not required.issubset(frame.columns):
            return None, "missing_daily_columns"
        if len(frame) < 20:
            return None, "insufficient_history"
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        rsi6 = core.calc_rsi(close, 6)
        k, d, _j = core.calc_kdj(high, low, close, 9, 3, 3)
        upper, mid, lower = core.calc_bollinger(close, 20, 2.0)
        atr = core.calc_atr(high, low, close, 14)
        values = {
            "k_prev": k.iloc[-2],
            "d_prev": d.iloc[-2],
            "k": k.iloc[-1],
            "d": d.iloc[-1],
            "rsi6": rsi6.iloc[-1],
            "low": low.iloc[-1],
            "close": close.iloc[-1],
            "boll_lower": lower.iloc[-1],
            "boll_mid": mid.iloc[-1],
            "boll_upper": upper.iloc[-1],
            "atr": atr.iloc[-1],
        }
        if not all(math.isfinite(_number(value)) for value in values.values()):
            return None, "invalid_indicator"
        result = dict(values)
        result.update(
            {
                "code": code,
                "current_date": str(current_date),
                "signal_date": str(signal_date),
                "max_data_date": str(frame["date"].max()),
            }
        )
        result["entry_eligible"] = is_entry_eligible(result)
        return result, None


@dataclass
class KRBAOrderPlanner:
    signal_adapter: object
    etf_pool: Iterable[str]
    trade_dates: list[str] | None = None
    params: dict = field(default_factory=lambda: {"max_hold": 3})
    position_states: dict[str, PositionSignalState] = field(default_factory=dict)
    last_scores: dict[str, dict] = field(default_factory=dict)
    _sold_date: str | None = field(default=None, init=False, repr=False)
    _sold_codes: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.etf_pool = [str(code).split(".")[0] for code in self.etf_pool]
        if self.trade_dates is not None:
            self.trade_dates = [str(item) for item in self.trade_dates]

    def plan_orders_at(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        decision_time: str,
        current_prices: Mapping[str, float] | None = None,
    ):
        del previous_date
        current_prices = dict(current_prices or {})
        self._reset_sold_codes(current_date)
        if str(decision_time) == "14:50":
            return self._plan_1450_atr(broker, current_prices)
        if str(decision_time) != "09:35":
            raise ValueError("KRBA supports only 09:35 and 14:50")

        scores = self._score_pool(current_date)
        self.last_scores = {item["code"]: item for item in scores}
        for code in list(broker.positions):
            state = self.position_states.get(code)
            score = self.last_scores.get(code)
            if state is None or score is None:
                continue
            state.mean_reached = bool(
                state.mean_reached
                or float(score["close"]) >= float(score["boll_mid"])
            )
            state.upper_reached = bool(
                state.upper_reached
                or float(score["close"]) >= float(score["boll_upper"])
            )

        plans = []
        planned_sells = set()
        for code in list(broker.positions):
            state = self.position_states.get(code)
            score = self.last_scores.get(code)
            if state is None or score is None:
                continue
            reason = choose_exit_reason(
                state,
                score,
                current_prices.get(code, float("nan")),
                self._hold_days(state.entry_date, current_date),
                code=code,
            )
            if reason is None:
                continue
            plans.append({"code": code, "target_value": 0.0, "reason": reason})
            planned_sells.add(code)

        held_after = [code for code in broker.positions if code not in planned_sells]
        slots = int(self.params["max_hold"]) - len(held_after)
        if slots <= 0:
            return plans
        eligible = [
            item for item in scores
            if item["code"] not in broker.positions
            and item["code"] not in self._sold_codes
            and is_entry_eligible(item)
        ]
        pool_rank = {code: index for index, code in enumerate(self.etf_pool)}
        eligible.sort(
            key=lambda item: (
                -(float(item["k"]) - float(item["d"])),
                float(item["rsi6"]),
                pool_rank[item["code"]],
            )
        )
        target = self._total_value(broker, current_prices) * 0.95 / 3.0
        for item in eligible[:slots]:
            plans.append(
                {
                    "code": item["code"],
                    "target_value": target,
                    "reason": "krba_entry",
                }
            )
        return plans

    def _plan_1450_atr(self, broker, current_prices):
        plans = []
        for code in list(broker.positions):
            state = self.position_states.get(code)
            price = _number(current_prices.get(code))
            if state is None or not math.isfinite(price) or price <= 0:
                continue
            if round(price, 3) <= round(calc_frozen_atr_stop(state, code), 3):
                plans.append(
                    {"code": code, "target_value": 0.0, "reason": "atr_stop"}
                )
        return plans

    def on_orders_processed(self, current_date, decision_time, plans, results):
        del decision_time, plans
        for order in results:
            if not getattr(order, "filled", False):
                continue
            code = str(order.code).split(".")[0]
            if order.amount_delta > 0:
                score = self.last_scores.get(code)
                if score is None:
                    continue
                self.position_states[code] = PositionSignalState(
                    entry_date=str(current_date),
                    entry_price=float(order.exec_price),
                    entry_atr=float(score["atr"]),
                    highest_close=float(order.exec_price),
                )
            elif order.amount_delta < 0:
                self.position_states.pop(code, None)
                self._sold_codes.add(code)

    def on_after_close(self, current_date, marks):
        del current_date
        for code, close in marks.items():
            state = self.position_states.get(code)
            if state is not None:
                state.highest_close = max(state.highest_close, float(close))

    def _score_pool(self, current_date):
        scores = []
        for code in self.etf_pool:
            score, _reason = self.signal_adapter.score(
                code, current_date, return_reason=True
            )
            if score is None:
                continue
            item = dict(score)
            item["code"] = code
            scores.append(item)
        return scores

    def _hold_days(self, entry_date, current_date):
        if self.trade_dates is None:
            return 0
        try:
            return self.trade_dates.index(str(current_date)) - self.trade_dates.index(
                str(entry_date)
            )
        except ValueError:
            return 0

    def _total_value(self, broker, prices):
        return float(broker.cash) + sum(
            pos.amount * float(prices.get(code, pos.avg_cost))
            for code, pos in broker.positions.items()
        )

    def _reset_sold_codes(self, current_date):
        if self._sold_date != str(current_date):
            self._sold_date = str(current_date)
            self._sold_codes.clear()
