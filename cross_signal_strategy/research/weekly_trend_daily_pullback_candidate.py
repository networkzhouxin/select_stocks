# -*- coding: utf-8 -*-
"""Platform-neutral weekly-trend/daily-pullback research candidate primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping

import pandas as pd


VERSION = "weekly-trend-pullback-v0.1-research-candidate"

_OHLC_COLUMNS = ("open", "high", "low", "close")
_WEEKLY_COLUMNS = ("open", "high", "low", "close", "last_trade_date")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_snapshot_values(snapshot):
    if not isinstance(snapshot, dict):
        return None
    keys = ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    values = tuple(_number(snapshot.get(key)) for key in keys)
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def aggregate_completed_weeks(frame, decision_date):
    """Aggregate daily OHLC into calendar weeks strictly before decision week."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in _OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))

    work = frame.copy()
    raw_dates = work["date"] if "date" in work.columns else work.index
    dates = pd.Series(pd.to_datetime(raw_dates, errors="coerce"), index=work.index)
    if dates.isna().any():
        raise ValueError("date values must be valid timestamps")

    decision = pd.Timestamp(decision_date).normalize()
    if pd.isna(decision):
        raise ValueError("decision_date must be a valid timestamp")
    current_monday = decision - pd.Timedelta(days=decision.weekday())

    completed_mask = dates < current_monday
    work = work.loc[completed_mask].copy()
    if work.empty:
        return pd.DataFrame(columns=_WEEKLY_COLUMNS)

    work["date"] = dates.loc[completed_mask]
    work = work.sort_values("date", kind="stable")
    work["week"] = work["date"].dt.to_period("W-SUN")
    return work.groupby("week", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        last_trade_date=("date", "last"),
    )


def build_weekly_context(frame, decision_date):
    """Build the most recent completed-week MA20 context with explicit errors."""

    weeks = aggregate_completed_weeks(frame, decision_date)
    if len(weeks) < 21:
        return None, "insufficient_weekly_history"

    closes = pd.to_numeric(weeks["close"], errors="coerce")
    ma20 = closes.rolling(20).mean()
    values = {
        "weekly_close": _number(closes.iloc[-1]),
        "weekly_ma20": _number(ma20.iloc[-1]),
        "weekly_ma20_prev": _number(ma20.iloc[-2]),
        "weekly_period_end": weeks.index[-1].end_time.date().isoformat(),
        "weekly_last_trade_date": (
            pd.Timestamp(weeks.iloc[-1]["last_trade_date"]).date().isoformat()
        ),
    }
    numeric_keys = ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    if not all(math.isfinite(values[key]) for key in numeric_keys):
        return None, "invalid_weekly_indicator"
    return values, None


def weekly_entry_allowed(snapshot):
    """Return whether the completed-week close and MA20 are strictly rising."""

    values = _finite_snapshot_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close > weekly_ma20 > weekly_ma20_prev


def weekly_trend_broken(snapshot):
    """Return whether both completed-week close and MA20 strictly trend down."""

    values = _finite_snapshot_values(snapshot)
    if values is None:
        return False
    weekly_close, weekly_ma20, weekly_ma20_prev = values
    return weekly_close < weekly_ma20 < weekly_ma20_prev


def is_daily_entry_eligible(snapshot):
    """Return whether every frozen T-1 daily pullback condition is satisfied."""

    if not isinstance(snapshot, Mapping):
        return False
    keys = (
        "close",
        "boll_lower",
        "boll_mid",
        "k_prev",
        "d_prev",
        "k",
        "d",
        "rsi6_prev",
        "rsi6",
    )
    values = {key: _number(snapshot.get(key)) for key in keys}
    if not all(math.isfinite(value) for value in values.values()):
        return False
    return bool(
        values["close"] > values["boll_lower"]
        and values["close"] <= values["boll_mid"]
        and values["k_prev"] <= values["d_prev"]
        and values["k"] > values["d"]
        and values["rsi6"] > values["rsi6_prev"]
        and values["rsi6"] <= 50.0
    )


def is_entry_eligible(snapshot):
    """Combine the completed-week gate with all frozen daily entry conditions."""

    return weekly_entry_allowed(snapshot) and is_daily_entry_eligible(snapshot)


def build_buy_queue(snapshots, excluded_codes, etf_pool):
    """Filter eligible snapshots and apply the frozen stable ranking tuple."""

    pool_rank = {str(code): rank for rank, code in enumerate(etf_pool)}
    excluded = {str(code) for code in (excluded_codes or ())}
    eligible = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        code = str(snapshot.get("code", ""))
        if code not in pool_rank or code in excluded or not is_entry_eligible(snapshot):
            continue
        weekly_close = _number(snapshot.get("weekly_close"))
        weekly_ma20 = _number(snapshot.get("weekly_ma20"))
        k_value = _number(snapshot.get("k"))
        d_value = _number(snapshot.get("d"))
        if weekly_ma20 == 0.0:
            continue
        item = dict(snapshot)
        item["code"] = code
        item["weekly_strength"] = weekly_close / weekly_ma20 - 1.0
        item["kd_spread"] = k_value - d_value
        eligible.append(item)
    return sorted(
        eligible,
        key=lambda item: (
            -item["weekly_strength"],
            -item["kd_spread"],
            pool_rank[item["code"]],
        ),
    )


@dataclass
class PositionSignalState:
    entry_date: str
    entry_price: float
    entry_atr: float
    highest_close: float


def calc_frozen_atr_stop(
    state,
    code="",
    multiplier=2.5,
    floor=0.05,
    cap=0.15,
):
    """Calculate a close-anchored trailing stop from the frozen entry ATR."""

    entry_price = _number(getattr(state, "entry_price", math.nan))
    entry_atr = _number(getattr(state, "entry_atr", math.nan))
    highest_close = _number(getattr(state, "highest_close", math.nan))
    multiplier_value = _number(multiplier)
    floor_value = 0.03 if str(code).split(".")[0] == "518880" else _number(floor)
    cap_value = _number(cap)
    values = (
        entry_price,
        entry_atr,
        highest_close,
        multiplier_value,
        floor_value,
        cap_value,
    )
    if not all(math.isfinite(value) for value in values):
        return math.nan
    if (
        entry_price <= 0.0
        or entry_atr <= 0.0
        or highest_close <= 0.0
        or multiplier_value <= 0.0
        or floor_value < 0.0
        or cap_value <= 0.0
        or floor_value > cap_value
    ):
        return math.nan
    distance = multiplier_value * entry_atr / entry_price
    distance = max(floor_value, min(cap_value, distance))
    return highest_close * (1.0 - distance)


def update_highest_close_from_t1(state, close):
    """Mutate highest_close only with a valid completed T-1 close."""

    close_value = _number(close)
    previous = _number(getattr(state, "highest_close", math.nan))
    if math.isfinite(close_value) and close_value > 0.0:
        if not math.isfinite(previous) or previous <= 0.0 or close_value > previous:
            state.highest_close = close_value
    return state


def _kd_cross_down(snapshot):
    if not isinstance(snapshot, Mapping):
        return False
    keys = ("k_prev", "d_prev", "k", "d")
    values = {key: _number(snapshot.get(key)) for key in keys}
    return bool(
        all(math.isfinite(value) for value in values.values())
        and values["k_prev"] >= values["d_prev"]
        and values["k"] < values["d"]
    )


def choose_exit_reason(state, snapshot, current_price, hold_days, code=""):
    """Choose exactly one exit in ATR, weekly-break, daily-failure priority."""

    price = _number(current_price)
    stop = calc_frozen_atr_stop(state, code)
    if (
        math.isfinite(price)
        and price > 0.0
        and math.isfinite(stop)
        and price <= stop
    ):
        return "atr_stop"

    if weekly_trend_broken(snapshot):
        return "weekly_trend_break"

    try:
        held_sessions = int(hold_days)
    except (TypeError, ValueError, OverflowError):
        held_sessions = 0
    if held_sessions < 5 or not isinstance(snapshot, Mapping):
        return None
    close = _number(snapshot.get("close"))
    boll_mid = _number(snapshot.get("boll_mid"))
    if (
        math.isfinite(close)
        and math.isfinite(boll_mid)
        and close < boll_mid
        and _kd_cross_down(snapshot)
    ):
        return "daily_pullback_failure"
    return None


@dataclass
class TrendPullbackOrderPlanner:
    """Platform-neutral causal order planner for the frozen candidate rules."""

    signal_adapter: object
    etf_pool: Iterable[str]
    trade_dates: list[str] | None = None
    params: dict = field(
        default_factory=lambda: {"max_hold": 3, "base_ratio": 0.95}
    )
    position_states: dict[str, PositionSignalState] = field(default_factory=dict)
    sold_today: set[str] = field(default_factory=set)
    sold_today_date: str | None = None
    last_scores: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        self.etf_pool = tuple(str(code) for code in self.etf_pool)
        self.trade_dates = (
            None if self.trade_dates is None else [str(day) for day in self.trade_dates]
        )

    def plan_orders_at(
        self,
        current_date,
        previous_date,
        broker,
        decision_time,
        current_prices=None,
    ):
        del previous_date
        current_day = str(current_date)
        prices = dict(current_prices or {})
        self._reset_sold_today(current_day)
        if str(decision_time) == "14:50":
            return self._plan_1450_atr(broker, prices)
        if str(decision_time) != "09:35":
            raise ValueError("candidate supports only 09:35 and 14:50")

        scores = self._score_pool(current_day)
        self.last_scores = {item["code"]: item for item in scores}

        plans = []
        planned_sells = set()
        positions = getattr(broker, "positions", {})
        for code in list(positions):
            state = self.position_states.get(code)
            if state is None:
                continue
            snapshot = self.last_scores.get(code, {})
            reason = choose_exit_reason(
                state,
                snapshot,
                prices.get(code, math.nan),
                self._hold_days(state.entry_date, current_day),
                code=code,
            )
            if reason is None:
                continue
            plans.append({"code": code, "target_value": 0.0, "reason": reason})
            planned_sells.add(code)

        held_after_sells = [code for code in positions if code not in planned_sells]
        slots = int(self.params.get("max_hold", 3)) - len(held_after_sells)
        if slots <= 0:
            return plans

        excluded = set(positions) | set(self.sold_today)
        queue = build_buy_queue(scores, excluded, self.etf_pool)
        total_value = self._total_value(broker, prices)
        target = (
            total_value
            * _number(self.params.get("base_ratio", 0.95))
            / float(self.params.get("max_hold", 3))
        )
        if not math.isfinite(target) or target <= 0.0:
            return plans
        for item in queue:
            entry_atr = _number(item.get("atr"))
            if not math.isfinite(entry_atr) or entry_atr <= 0.0:
                continue
            plans.append(
                {
                    "code": item["code"],
                    "target_value": target,
                    "reason": "weekly_pullback_entry",
                    "entry_atr": entry_atr,
                }
            )
            if sum(1 for plan in plans if plan["target_value"] > 0.0) >= slots:
                break
        return plans

    def _plan_1450_atr(self, broker, current_prices):
        plans = []
        positions = getattr(broker, "positions", {})
        for code in list(positions):
            if code in self.sold_today:
                continue
            state = self.position_states.get(code)
            price = _number(current_prices.get(code))
            stop = calc_frozen_atr_stop(state, code) if state is not None else math.nan
            if (
                math.isfinite(price)
                and price > 0.0
                and math.isfinite(stop)
                and price <= stop
            ):
                plans.append(
                    {"code": code, "target_value": 0.0, "reason": "atr_stop"}
                )
        return plans

    def on_orders_processed(self, current_date, decision_time, plans, results):
        del decision_time
        current_day = str(current_date)
        self._reset_sold_today(current_day)
        buy_plans = {
            str(plan.get("code")): plan
            for plan in plans
            if _number(plan.get("target_value")) > 0.0
        }
        for order in results:
            if not bool(getattr(order, "filled", False)):
                continue
            code = str(getattr(order, "code", ""))
            amount_delta = _number(getattr(order, "amount_delta", 0.0))
            exec_price = _number(getattr(order, "exec_price", math.nan))
            if amount_delta > 0.0:
                plan = buy_plans.get(code)
                entry_atr = _number(plan.get("entry_atr")) if plan else math.nan
                if (
                    plan is None
                    or not math.isfinite(exec_price)
                    or exec_price <= 0.0
                    or not math.isfinite(entry_atr)
                    or entry_atr <= 0.0
                ):
                    continue
                self.position_states[code] = PositionSignalState(
                    entry_date=current_day,
                    entry_price=exec_price,
                    entry_atr=entry_atr,
                    highest_close=exec_price,
                )
            elif amount_delta < 0.0:
                self.position_states.pop(code, None)
                self.sold_today.add(code)

    def on_after_close(self, current_date, marks):
        del current_date
        for code, close in marks.items():
            state = self.position_states.get(str(code))
            if state is not None:
                update_highest_close_from_t1(state, close)

    def _score_pool(self, current_date):
        scores = []
        for code in self.etf_pool:
            score, _reason = self.signal_adapter.score(
                code,
                current_date,
                return_reason=True,
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
        reported = _number(getattr(broker, "total_value", math.nan))
        if math.isfinite(reported) and reported > 0.0:
            return reported
        cash = _number(getattr(broker, "cash", 0.0))
        total = cash if math.isfinite(cash) else 0.0
        for code, position in getattr(broker, "positions", {}).items():
            fallback = _number(getattr(position, "avg_cost", math.nan))
            price = _number(prices.get(code, fallback))
            amount = _number(getattr(position, "amount", 0.0))
            if math.isfinite(price) and math.isfinite(amount):
                total += price * amount
        return total

    def _reset_sold_today(self, current_date):
        if self.sold_today_date != str(current_date):
            self.sold_today.clear()
            self.sold_today_date = str(current_date)
