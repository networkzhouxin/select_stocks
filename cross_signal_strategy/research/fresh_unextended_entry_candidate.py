# -*- coding: utf-8 -*-
"""Frozen training-only fresh, unextended fast-entry candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_order_planner import (
    LocalCrossSignalOrderPlanner,
    strategy,
)
from cross_signal_strategy.research.baseline_report import (
    BaselineReport,
    build_baseline_report,
)


FRESH_BUY_MIN_SCORE = 50.0
FRESH_BUY_MAX_SCORE = 60.0
FRESH_MIN_REVERSAL_SCORE = 35.0
FRESH_MAX_CROSS_AGE = 1
FRESH_MAX_EXTENSION_ATR = 1.0

_BULLISH_CROSS_FIELDS = (
    "rsi6_cross_rsi12_up",
    "rsi6_cross_rsi24_up",
    "macd_cross_up",
    "kdj_k_cross_up",
    "kdj_j_cross_up",
)


@dataclass(frozen=True)
class FreshUnextendedSignalAdapter:
    """Enrich an existing causal T-1 adapter without changing its scores."""

    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        raw_score, reason = self.source.score(
            code,
            current_date,
            return_reason=True,
        )
        if raw_score is None:
            return (None, reason) if return_reason else None
        frame, signal_date = self.source.load_signal_frame(code, current_date)
        score = dict(raw_score)
        if signal_date is None:
            reason = "no_previous_trade_date"
            return (None, reason) if return_reason else None
        score["signal_date"] = str(signal_date)
        score["max_data_date"] = str(pd.Timestamp(frame["date"].max()).date())
        enriched = enrich_fresh_entry_context(score, frame)
        return (enriched, None) if return_reason else enriched


def enrich_fresh_entry_context(
    raw_score: Mapping[str, object],
    signal_frame: pd.DataFrame,
) -> dict:
    """Attach causal cross-age/extension fields used by the candidate filter."""
    score = dict(raw_score)
    signal_date = pd.Timestamp(str(score.get("signal_date", "")))
    frame = signal_frame.copy()
    dates = pd.to_datetime(frame["date"], errors="raise")
    if (dates > signal_date).any():
        raise ValueError("signal frame contains a row later than signal_date")
    if frame.empty or dates.max() != signal_date:
        raise ValueError("signal frame must end exactly on signal_date")

    active_ages, context_complete = _contributing_bullish_cross_ages(score)
    if not active_ages:
        score.update(
            {
                "fresh_entry_context_complete": context_complete,
                "fresh_entry_earliest_cross_age": None,
                "fresh_entry_earliest_cross_date": None,
                "fresh_entry_cross_close": None,
                "fresh_entry_extension_atr": None,
            }
        )
        return score

    earliest_age = max(active_ages)
    row_index = len(frame) - 1 - earliest_age
    if row_index < 0:
        raise ValueError("signal frame is too short for contributing cross age")
    cross_row = frame.iloc[row_index]
    cross_close = float(cross_row["close"])
    close = _finite_float(score.get("close"))
    atr = _finite_float(score.get("atr"))
    extension = None
    if close is not None and atr is not None and atr > 0 and cross_close > 0:
        extension = (close - cross_close) / atr
    score.update(
        {
            "fresh_entry_context_complete": context_complete,
            "fresh_entry_earliest_cross_age": earliest_age,
            "fresh_entry_earliest_cross_date": pd.Timestamp(cross_row["date"]).strftime(
                "%Y-%m-%d"
            ),
            "fresh_entry_cross_close": cross_close,
            "fresh_entry_extension_atr": extension,
        }
    )
    return score


def filter_fresh_unextended_buy_candidates(
    scores: Iterable[Mapping[str, object]],
    held_codes: Iterable[str],
    params=None,
) -> list[dict]:
    """Return only the single frozen 50-59 fresh-entry candidate family."""
    p = params or strategy.get_default_params()
    held = {str(code).split(".")[0] for code in held_codes}
    candidates = []
    for raw_score in scores:
        score = dict(raw_score)
        code = str(score.get("code", "")).split(".")[0]
        score["code"] = code
        buy_score = _finite_float(score.get("buy_score"))
        reversal_score = _finite_float(score.get("reversal_score"))
        age = _finite_float(score.get("fresh_entry_earliest_cross_age"))
        extension = _finite_float(score.get("fresh_entry_extension_atr"))
        if not code or code in held:
            continue
        if buy_score is None or not (
            FRESH_BUY_MIN_SCORE <= buy_score < FRESH_BUY_MAX_SCORE
        ):
            continue
        if reversal_score is None or reversal_score < FRESH_MIN_REVERSAL_SCORE:
            continue
        if score.get("fresh_entry_context_complete") is not True:
            continue
        if age is None or age < 0 or age > FRESH_MAX_CROSS_AGE:
            continue
        if extension is None or extension > FRESH_MAX_EXTENSION_ATR:
            continue
        if not score.get("buy_allowed"):
            continue
        if float(score.get("sell_score", 0) or 0) >= float(p["sell_threshold"]):
            continue
        if not strategy.has_new_buy_position(score, p):
            continue
        if strategy.is_blocked_entry_combo(score):
            continue
        candidates.append(score)
    return strategy.sort_candidates(candidates)


@dataclass
class FreshUnextendedEntryOrderPlanner(LocalCrossSignalOrderPlanner):
    """Fill only slots left vacant by the unchanged primary buy path."""

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = list(
            super().plan_orders(
                current_date,
                previous_date,
                broker,
                current_prices=current_prices,
            )
        )
        sold_codes = {
            str(order["code"]).split(".")[0]
            for order in orders
            if float(order.get("target_value", 0.0)) == 0.0
        }
        primary_buy_codes = {
            str(order["code"]).split(".")[0]
            for order in orders
            if order.get("reason") == "buy_signal"
        }
        held_after_sell = {
            str(code).split(".")[0]
            for code in broker.positions
            if str(code).split(".")[0] not in sold_codes
        }
        occupied = held_after_sell | primary_buy_codes
        slots = int(self.params["max_hold"]) - len(occupied)
        if slots <= 0:
            return orders

        force_stopped = {
            str(order["code"]).split(".")[0]
            for order in orders
            if order.get("reason") == "atr_stop"
        }
        candidates = [
            item
            for item in filter_fresh_unextended_buy_candidates(
                self.last_scores.values(),
                held_codes=occupied,
                params=self.params,
            )
            if item["code"] not in force_stopped
            and not self._is_in_atr_stop_cooldown(item["code"], current_date)
        ]
        total_value = self._total_value(broker, current_prices or {})
        for score in candidates[:slots]:
            orders.append(
                {
                    "code": score["code"],
                    "target_value": self._scaled_buy_target_value(
                        total_value,
                        score,
                        current_date,
                    ),
                    "reason": "fresh_unextended_buy_signal",
                }
            )
        return orders


@dataclass(frozen=True)
class FreshUnextendedLocalComparison:
    """Preliminary local screen; never an adoption decision."""

    baseline: BaselineReport
    candidate: BaselineReport
    baseline_double_friction: BaselineReport
    candidate_double_friction: BaselineReport
    filled_fresh_buys: int
    fresh_buy_years: tuple[int, ...]


def run_training_fresh_unextended_comparison(
    loader=None,
    initial_cash: float = 20000.0,
) -> FreshUnextendedLocalComparison:
    """Run the fixed 2019-2021 local screen under identical execution models."""
    from cross_signal_strategy.local.local_data_loader import (
        CrossSignalTrainingDataLoader,
    )
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )

    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = FreshUnextendedSignalAdapter(build_training_signal_adapter(loader))
    initial_planner = LocalCrossSignalOrderPlanner(source, trade_dates=trade_dates)
    cached = PrecomputedSignalAdapter.from_source(
        source,
        trade_dates=trade_dates,
        codes=initial_planner.etf_pool,
    )
    baseline_days = _run_local_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    candidate_days = _run_local_replay(
        loader,
        FreshUnextendedEntryOrderPlanner(
            cached,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    doubled = {
        "commission_rate": 0.0006,
        "min_commission": 10.0,
        "slippage_rate": 0.002,
    }
    baseline_stress_days = _run_local_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=doubled,
    )
    candidate_stress_days = _run_local_replay(
        loader,
        FreshUnextendedEntryOrderPlanner(
            cached,
            etf_pool=initial_planner.etf_pool,
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=doubled,
    )
    fresh_dates = [
        str(day.date)
        for day in candidate_days
        for order in day.orders
        if getattr(order, "filled", False)
        and getattr(order, "reason", "") == "fresh_unextended_buy_signal"
    ]
    return FreshUnextendedLocalComparison(
        baseline=build_baseline_report(baseline_days, initial_cash),
        candidate=build_baseline_report(candidate_days, initial_cash),
        baseline_double_friction=build_baseline_report(
            baseline_stress_days, initial_cash
        ),
        candidate_double_friction=build_baseline_report(
            candidate_stress_days, initial_cash
        ),
        filled_fresh_buys=len(fresh_dates),
        fresh_buy_years=tuple(sorted({pd.Timestamp(day).year for day in fresh_dates})),
    )


def format_fresh_unextended_comparison(
    comparison: FreshUnextendedLocalComparison,
) -> str:
    """Render the local screen with an explicit non-authoritative warning."""
    base = comparison.baseline
    candidate = comparison.candidate
    stress_base = comparison.baseline_double_friction
    stress_candidate = comparison.candidate_double_friction
    return "\n".join(
        [
            "Fresh-unextended entry local screen (not performance authority)",
            "baseline return={:.2%} drawdown={:.2%} win_rate={:.2%} pl={}".format(
                base.total_return,
                base.max_drawdown,
                base.win_rate,
                _format_optional(base.profit_loss_ratio),
            ),
            "candidate return={:.2%} drawdown={:.2%} win_rate={:.2%} pl={}".format(
                candidate.total_return,
                candidate.max_drawdown,
                candidate.win_rate,
                _format_optional(candidate.profit_loss_ratio),
            ),
            "fresh_filled_buys={} years={}".format(
                comparison.filled_fresh_buys,
                ",".join(str(year) for year in comparison.fresh_buy_years) or "none",
            ),
            "double_friction baseline_return={:.2%} baseline_dd={:.2%} "
            "candidate_return={:.2%} candidate_dd={:.2%}".format(
                stress_base.total_return,
                stress_base.max_drawdown,
                stress_candidate.total_return,
                stress_candidate.max_drawdown,
            ),
            "decision=local_screen_only; JoinQuant 2019-2021 is authoritative",
        ]
    )


def _run_local_replay(
    loader,
    planner,
    trade_dates: Sequence[str],
    initial_cash: float,
    broker_kwargs: Mapping[str, object] | None = None,
):
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine

    engine = LocalBacktestEngine(
        loader=loader,
        initial_cash=initial_cash,
        broker_kwargs=broker_kwargs,
    )
    return engine.run(trade_dates, planner.plan_orders)


def _format_optional(value: float | None) -> str:
    return "none" if value is None else "{:.3f}".format(value)


def _contributing_bullish_cross_ages(
    score: Mapping[str, object],
) -> tuple[list[int], bool]:
    rsi_up = bool(
        score.get("rsi6_cross_rsi12_up")
        or score.get("rsi6_cross_rsi24_up")
    )
    rsi_down = bool(
        score.get("rsi6_cross_rsi12_down")
        or score.get("rsi6_cross_rsi24_down")
    )
    ages = []
    complete = True
    for field in _BULLISH_CROSS_FIELDS:
        if not bool(score.get(field)):
            continue
        if field.startswith("rsi6_") and not (rsi_up and not rsi_down):
            continue
        raw_age = score.get(field + "_age")
        try:
            age = int(raw_age)
        except (TypeError, ValueError):
            complete = False
            continue
        if age >= 0:
            ages.append(age)
    return ages, complete


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> None:
    print(format_fresh_unextended_comparison(
        run_training_fresh_unextended_comparison()
    ))


if __name__ == "__main__":
    main()
