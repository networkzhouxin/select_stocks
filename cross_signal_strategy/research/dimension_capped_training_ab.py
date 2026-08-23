# -*- coding: utf-8 -*-
"""Isolated training-period order planner for the dimension-capped candidate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import types
from typing import List, Mapping, Sequence


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine, LocalBroker
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.friction_diagnostics import (
    FrictionScenarioConfig,
    PrecomputedSignalAdapter,
)
from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.research.baseline_report import build_baseline_report
from cross_signal_strategy.research import dimension_capped_score_candidate as candidate_rules
from cross_signal_strategy.research.dimension_capped_score_candidate import (
    DimensionCappedScoreAdapter,
    has_raw_sell_conflict,
    is_dimension_capped_buy_candidate,
    should_dimension_capped_signal_sell,
    sort_dimension_capped_candidates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
TRAINING_YEARS = (2019, 2020, 2021)
EXECUTION_TIME = "09:35"
MIN_SIGNAL_HOLD_DAYS = 5
DOUBLE_FRICTION = FrictionScenarioConfig(
    commission_rate=0.0006,
    min_commission=10.0,
    slippage_rate=0.002,
)
REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "dimension_capped_score_v04_2019_2021.md"
)


APPROVED_CANDIDATE_RULE_MANIFEST = {
    "scoring": {
        "candidate_name": "cross-v0.4.0-dimension-capped-candidate",
        "buy": {
            "reversal": {
                "cap": 25.0,
                "minimum": 12.0,
                "contributions": {
                    "rsi_group": 12.0,
                    "kdj_group": 6.0,
                    "kdj_state_k_le_20": 10.0,
                    "kdj_state_20_lt_k_le_30": 5.0,
                    "macd_confirmation": 5.0,
                },
            },
            "location": {
                "cap": 10.0,
                "minimum": 7.0,
                "aggregation": "maximum_single_contribution",
                "contributions": {
                    "between_boll_lower_mid": 10.0,
                    "cross_boll_mid_up": 8.0,
                    "near_ma20": 7.0,
                },
            },
            "trend": {
                "cap": 20.0,
                "minimum": 6.0,
                "contributions": {
                    "ma5_gt_ma10": 6.0,
                    "ma10_gt_ma20": 6.0,
                    "ma20_slope_non_negative": 5.0,
                    "close_gt_ma60": 3.0,
                },
            },
            "total_threshold": 40.0,
            "raw_sell_conflict_required_absent": True,
        },
        "sell": {
            "weakness": {
                "cap": 20.0,
                "ordinary_minimum": 10.0,
                "severe_minimum": 6.0,
                "contributions": {
                    "rsi_group": 10.0,
                    "kdj_group": 6.0,
                    "kdj_state_k_ge_80": 8.0,
                    "kdj_state_70_le_k_lt_80": 4.0,
                    "macd_confirmation": 4.0,
                },
            },
            "damage": {
                "cap": 20.0,
                "ordinary_minimum": 8.0,
                "severe_minimum": 18.0,
                "aggregation": "maximum_single_contribution",
                "contributions": {
                    "downside_continuation": 20.0,
                    "below_falling_ma10": 18.0,
                    "below_ma20": 15.0,
                    "below_boll_mid": 12.0,
                    "fell_back_inside_boll": 8.0,
                },
            },
            "ordinary_total_threshold": 24.0,
            "adx_protection": "held_position_soft_sell_only",
        },
        "ranking": [
            "buy_total_desc",
            "location_desc",
            "reversal_desc",
            "a_share_volume_desc",
            "code_asc",
        ],
    },
    "indicators": {
        "rsi": [6, 12, 24],
        "macd": [12, 26, 9],
        "kdj": [9, 3, 3],
        "boll": [20, 2.0],
        "atr": 14,
        "adx": 14,
        "ma": [5, 10, 20, 60],
        "cross_window": 3,
    },
    "portfolio": {
        "pool": [
            "159915", "512100", "159928", "513100", "513500",
            "513880", "513050", "518880", "159985",
        ],
        "max_hold": 3,
        "base_ratio": 0.95,
        "cash_buffer": 0.05,
        "target_weighting": "equal_weight",
        "atr_stress_lookback_days": 15,
        "atr_stress_min_stops": 3,
        "atr_stress_buy_scale": 0.5,
    },
    "execution": {
        "time": "09:35",
        "min_signal_hold_days": 5,
        "signal_boundary": "completed_daily_bars_through_T_minus_1",
        "execution_price_boundary": "T_09:35_only",
    },
    "friction": {
        "nominal": {
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "slippage_rate": 0.001,
        },
        "doubled": {
            "commission_rate": 0.0006,
            "min_commission": 10.0,
            "slippage_rate": 0.002,
        },
    },
    "atr_stop": {
        "period": 14,
        "highest_anchor": "highest_completed_close_since_buy",
        "trailing_multiplier": 2.5,
        "floor": 0.05,
        "cap": 0.15,
        "decision_price": "T_09:35_execution_price",
        "bypasses_signal_hold_and_adx": True,
    },
}


@dataclass(frozen=True)
class DimensionCappedTrainingConfig:
    candidate_name: str
    training_start: str
    training_end: str
    initial_cash: float
    execution_time: str
    buy_threshold: float
    ordinary_sell_threshold: float
    min_signal_hold_days: int
    max_hold: int
    base_ratio: float
    candidate_variants: int
    training_root: Path
    warmup_root: Path


@dataclass(frozen=True)
class DimensionCappedPerformance:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    win_rate: float
    profit_loss_ratio: float | None
    buy_count: int
    sell_count: int
    closed_trade_count: int
    annual_returns: dict[int, float]


@dataclass(frozen=True)
class DimensionCappedGateInputs:
    baseline: DimensionCappedPerformance
    candidate: DimensionCappedPerformance
    baseline_double_friction: DimensionCappedPerformance
    candidate_double_friction: DimensionCappedPerformance
    changed_order_days: int
    changed_days_by_year: dict[int, int]


@dataclass(frozen=True)
class DimensionCappedGateDecision:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DimensionCappedDecisionAudit:
    decision_date: str
    signal_date: str
    max_data_date: str
    code: str
    held: bool
    buy_reversal: float
    buy_location: float
    buy_trend: float
    volume_rank: float
    buy_total: float
    sell_weakness: float
    sell_damage: float
    sell_total: float
    kdj_tier: str
    macd_confirmation: str
    raw_contributions: tuple[tuple[str, float], ...]
    adx_protected: bool
    atr_stop: bool
    min_hold_blocked: bool
    hard_block_reasons: tuple[str, ...]
    order_reason: str | None


@dataclass(frozen=True)
class DimensionCappedScoreAttemptAudit:
    decision_date: str
    code: str
    status: str
    causal_boundary: str
    t_minus_one_date: str | None
    signal_date: str | None
    max_data_date: str | None
    skip_reason: str | None


@dataclass(frozen=True)
class DimensionCappedExecutionAudit:
    decision_date: str
    plan_sequence: int
    code: str
    planned_side: str
    planned_reason: str
    target_value: float
    status: str
    filled_amount: int
    execution_price: float | None
    commission: float | None
    unfilled_reason: str | None
    atr_highest_close: float | None
    atr_input: float | None
    atr_position_cost: float | None
    atr_decision_price: float | None
    atr_stop_threshold: float | None


@dataclass(frozen=True)
class DimensionCappedComparisonReport:
    config: DimensionCappedTrainingConfig
    inputs: DimensionCappedGateInputs
    gate: DimensionCappedGateDecision
    decision_audits: tuple[DimensionCappedDecisionAudit, ...]
    score_attempt_audits: tuple[DimensionCappedScoreAttemptAudit, ...] = ()
    execution_audits: tuple[DimensionCappedExecutionAudit, ...] = ()


def approved_candidate_rule_manifest() -> dict:
    return deepcopy(APPROVED_CANDIDATE_RULE_MANIFEST)


def _executable_candidate_rule_manifest() -> dict:
    params = strategy.get_default_params()
    broker_parameters = inspect.signature(LocalBroker.__init__).parameters
    nominal = {
        "commission_rate": float(broker_parameters["commission_rate"].default),
        "min_commission": float(broker_parameters["min_commission"].default),
        "slippage_rate": float(broker_parameters["slippage_rate"].default),
    }
    return {
        "scoring": candidate_rules.executable_candidate_rule_manifest(),
        "indicators": {
            "rsi": [int(params["rsi_fast"]), int(params["rsi_mid"]), int(params["rsi_slow"])],
            "macd": [int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"])],
            "kdj": [int(params["kdj_n"]), int(params["kdj_m1"]), int(params["kdj_m2"])],
            "boll": [int(params["boll_period"]), float(params["boll_std"])],
            "atr": int(params["atr_period"]),
            "adx": int(params["adx_period"]),
            "ma": [5, 10, 20, 60],
            "cross_window": int(params["cross_window"]),
        },
        "portfolio": {
            "pool": [str(code).split(".")[0] for code in strategy.get_default_etf_pool()],
            "max_hold": int(params["max_hold"]),
            "base_ratio": float(params["base_ratio"]),
            "cash_buffer": round(1.0 - float(params["base_ratio"]), 12),
            "target_weighting": "equal_weight",
            "atr_stress_lookback_days": int(params["portfolio_atr_stress_lookback_days"]),
            "atr_stress_min_stops": int(params["portfolio_atr_stress_min_stops"]),
            "atr_stress_buy_scale": float(params["portfolio_atr_stress_buy_scale"]),
        },
        "execution": {
            "time": EXECUTION_TIME,
            "min_signal_hold_days": MIN_SIGNAL_HOLD_DAYS,
            "signal_boundary": "completed_daily_bars_through_T_minus_1",
            "execution_price_boundary": "T_09:35_only",
        },
        "friction": {
            "nominal": nominal,
            "doubled": {
                "commission_rate": float(DOUBLE_FRICTION.commission_rate),
                "min_commission": float(DOUBLE_FRICTION.min_commission),
                "slippage_rate": float(DOUBLE_FRICTION.slippage_rate),
            },
        },
        "atr_stop": {
            "period": int(params["atr_period"]),
            "highest_anchor": "highest_completed_close_since_buy",
            "trailing_multiplier": float(params["trailing_atr_mult"]),
            "floor": float(params["stop_floor"]),
            "cap": float(params["stop_cap"]),
            "decision_price": "T_09:35_execution_price",
            "bypasses_signal_hold_and_adx": True,
        },
    }


def _canonical_manifest_text(manifest: Mapping[str, object]) -> str:
    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_rule_fingerprint(manifest: Mapping[str, object] | None = None) -> str:
    payload = manifest if manifest is not None else approved_candidate_rule_manifest()
    return hashlib.sha256(_canonical_manifest_text(payload).encode("utf-8")).hexdigest()


def _assert_candidate_rule_manifest_matches_executable() -> None:
    if _executable_candidate_rule_manifest() != APPROVED_CANDIDATE_RULE_MANIFEST:
        raise ValueError("executable constants do not match approved candidate rule manifest")


def dimension_capped_training_config() -> DimensionCappedTrainingConfig:
    return DimensionCappedTrainingConfig(
        candidate_name="cross-v0.4.0-dimension-capped-candidate",
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        initial_cash=20000.0,
        execution_time=EXECUTION_TIME,
        buy_threshold=candidate_rules.BUY_THRESHOLD,
        ordinary_sell_threshold=candidate_rules.ORDINARY_SELL_THRESHOLD,
        min_signal_hold_days=MIN_SIGNAL_HOLD_DAYS,
        max_hold=3,
        base_ratio=0.95,
        candidate_variants=1,
        training_root=Path(APPROVED_TRAINING_ROOT),
        warmup_root=Path(APPROVED_WARMUP_ROOT),
    )


def evaluate_dimension_capped_gate(
    inputs: DimensionCappedGateInputs,
) -> DimensionCappedGateDecision:
    """Apply the frozen materiality, retention, and robustness gates."""

    baseline = inputs.baseline
    candidate = inputs.candidate
    reasons: list[str] = []
    for performance, label in (
        (baseline, "baseline"),
        (candidate, "candidate"),
        (inputs.baseline_double_friction, "baseline doubled-friction"),
        (inputs.candidate_double_friction, "candidate doubled-friction"),
    ):
        _append_performance_validation_reasons(reasons, performance, label)
    if reasons:
        return DimensionCappedGateDecision(False, tuple(reasons))
    _append_materiality_validation_reasons(reasons, inputs)
    if reasons:
        return DimensionCappedGateDecision(False, tuple(reasons))

    if inputs.changed_order_days < 10:
        reasons.append("fewer than 10 changed filled-order days")
    for year in TRAINING_YEARS:
        if int(inputs.changed_days_by_year[year]) < 2:
            reasons.append(f"{year} has fewer than 2 changed filled-order days")
    if candidate.closed_trade_count < baseline.closed_trade_count * 0.80:
        reasons.append("candidate retains fewer than 80% of closed trades")
    if candidate.win_rate <= baseline.win_rate:
        reasons.append("candidate win rate does not strictly improve")
    if candidate.total_return < baseline.total_return * 0.95:
        reasons.append("candidate retains less than 95% of baseline return")
    if candidate.max_drawdown > baseline.max_drawdown + 0.005:
        reasons.append("candidate maximum drawdown worsens by more than 0.5pp")
    _append_ratio_gate_reason(
        reasons,
        candidate.sharpe_ratio,
        baseline.sharpe_ratio,
        "Sharpe ratio",
    )
    _append_ratio_gate_reason(
        reasons,
        candidate.sortino_ratio,
        baseline.sortino_ratio,
        "Sortino ratio",
    )
    _append_ratio_gate_reason(
        reasons,
        candidate.profit_loss_ratio,
        baseline.profit_loss_ratio,
        "profit/loss ratio",
    )
    for year in TRAINING_YEARS:
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is not None and baseline_return > 0.0 and (
            candidate_return is None or candidate_return <= 0.0
        ):
            reasons.append("a positive baseline year turns non-positive")
            break

    baseline_x2 = inputs.baseline_double_friction
    candidate_x2 = inputs.candidate_double_friction
    if candidate_x2.total_return < baseline_x2.total_return * 0.95:
        reasons.append("doubled-friction return retains less than 95%")
    if candidate_x2.win_rate < baseline_x2.win_rate:
        reasons.append("doubled-friction win rate is below baseline")
    return DimensionCappedGateDecision(not reasons, tuple(reasons))


def _append_materiality_validation_reasons(
    reasons: list[str],
    inputs: DimensionCappedGateInputs,
) -> None:
    total_valid = _is_non_negative_integer(inputs.changed_order_days)
    if not total_valid:
        reasons.append(
            "changed_order_days must be a finite non-negative integer"
        )

    yearly_values: list[int] = []
    yearly_valid = True
    for year in TRAINING_YEARS:
        if year not in inputs.changed_days_by_year:
            reasons.append(f"{year} changed filled-order days metric is missing")
            yearly_valid = False
            continue
        value = inputs.changed_days_by_year[year]
        if not _is_non_negative_integer(value):
            reasons.append(
                f"{year} changed filled-order days must be a finite non-negative integer"
            )
            yearly_valid = False
            continue
        yearly_values.append(int(float(value)))

    if total_valid and yearly_valid and int(float(inputs.changed_order_days)) != sum(yearly_values):
        reasons.append(
            "changed_order_days does not equal the 2019-2021 yearly total"
        )


def _is_non_negative_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0 and number.is_integer()


def _append_performance_validation_reasons(
    reasons: list[str],
    performance: DimensionCappedPerformance,
    label: str,
) -> None:
    for field in (
        "total_return",
        "annualized_return",
        "max_drawdown",
        "win_rate",
        "buy_count",
        "sell_count",
        "closed_trade_count",
    ):
        if not _is_finite_number(getattr(performance, field)):
            reasons.append(f"{label} {field} metric is non-finite")
    for field, field_label in (
        ("sharpe_ratio", "Sharpe ratio"),
        ("sortino_ratio", "Sortino ratio"),
        ("profit_loss_ratio", "profit/loss ratio"),
    ):
        value = getattr(performance, field)
        if value is not None and not _is_finite_number(value):
            reasons.append(f"{label} {field_label} metric is non-finite")
    annual_returns = performance.annual_returns
    for year in TRAINING_YEARS:
        if year not in annual_returns or annual_returns[year] is None:
            reasons.append(f"{label} {year} annual return metric is missing")
        elif not _is_finite_number(annual_returns[year]):
            reasons.append(
                f"{label} {year} annual return metric is non-finite"
            )


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _append_ratio_gate_reason(
    reasons: list[str],
    candidate: float | None,
    baseline: float | None,
    label: str,
) -> None:
    if baseline is None:
        return
    if candidate is None:
        reasons.append(f"candidate {label} metric is missing")
    elif candidate < baseline * 0.95:
        reasons.append(f"candidate {label} retains less than 95%")


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(
        str(date) < TRAINING_START or str(date) > TRAINING_END
        for date in dates
    ):
        raise ValueError("Replay contains dates outside 2019-2021 training window")


def _annual_returns(
    days: Sequence[object],
    initial_cash: float,
) -> dict[int, float]:
    grouped: dict[int, list[object]] = {}
    for day in days:
        grouped.setdefault(int(str(day.date)[:4]), []).append(day)
    result: dict[int, float] = {}
    start_value = float(initial_cash)
    for year, year_days in sorted(grouped.items()):
        end_value = float(year_days[-1].total_value)
        result[year] = end_value / start_value - 1.0
        start_value = end_value
    return result


def _performance(
    days: Sequence[object],
    initial_cash: float,
) -> DimensionCappedPerformance:
    report = build_baseline_report(days, initial_cash)
    return DimensionCappedPerformance(
        total_return=float(report.total_return),
        annualized_return=float(report.annualized_return),
        max_drawdown=float(report.max_drawdown),
        sharpe_ratio=report.sharpe_ratio,
        sortino_ratio=report.sortino_ratio,
        win_rate=float(report.win_rate),
        profit_loss_ratio=report.profit_loss_ratio,
        buy_count=int(report.buy_count),
        sell_count=int(report.sell_count),
        closed_trade_count=int(report.closed_trade_count),
        annual_returns=_annual_returns(days, initial_cash),
    )


def _filled_order_signature(
    day: object,
) -> tuple[tuple[str, str, str, int], ...]:
    signature = []
    for order in getattr(day, "orders", ()):
        amount = int(getattr(order, "amount_delta", 0))
        if not getattr(order, "filled", False) or amount == 0:
            continue
        signature.append((
            str(day.date),
            str(order.code).split(".")[0],
            "buy" if amount > 0 else "sell",
            abs(amount),
        ))
    return tuple(sorted(signature))


def _filled_order_changes(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
) -> tuple[int, dict[int, int]]:
    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    if baseline_dates != candidate_dates:
        raise ValueError("A/B comparison requires identical trading dates")
    _assert_training_dates(baseline_dates)
    changed_days_by_year = {year: 0 for year in TRAINING_YEARS}
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        if _filled_order_signature(baseline_day) != _filled_order_signature(candidate_day):
            changed_days_by_year[int(str(baseline_day.date)[:4])] += 1
    return sum(changed_days_by_year.values()), changed_days_by_year


def _assert_approved_loader(loader: object) -> None:
    root = getattr(loader, "root", None)
    if root is None or (
        Path(root).expanduser().resolve()
        != Path(APPROVED_TRAINING_ROOT).expanduser().resolve()
    ):
        raise ValueError(
            f"Use approved training data root only: {APPROVED_TRAINING_ROOT}"
        )


def _assert_approved_warmup_root(root: Path | str) -> None:
    if (
        Path(root).expanduser().resolve()
        != Path(APPROVED_WARMUP_ROOT).expanduser().resolve()
    ):
        raise ValueError(
            f"Use approved warm-up data root only: {APPROVED_WARMUP_ROOT}"
        )


def _run_arm(
    loader,
    signal_adapter,
    planner_class,
    params: Mapping[str, object],
    pool: Sequence[str],
    trade_dates: Sequence[str],
    initial_cash: float,
    friction: FrictionScenarioConfig | None,
):
    planner = planner_class(
        signal_adapter,
        etf_pool=list(pool),
        params=dict(params),
        trade_dates=list(trade_dates),
    )
    broker_kwargs = None
    if friction is not None:
        broker_kwargs = {
            "commission_rate": friction.commission_rate,
            "min_commission": friction.min_commission,
            "slippage_rate": friction.slippage_rate,
        }
    engine = LocalBacktestEngine(
        loader=loader,
        initial_cash=initial_cash,
        execution_time=EXECUTION_TIME,
        broker_kwargs=broker_kwargs,
    )
    return engine.run(trade_dates, planner.plan_orders), planner


def _build_gate_inputs(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
    baseline_x2_days: Sequence[object],
    candidate_x2_days: Sequence[object],
    initial_cash: float,
) -> DimensionCappedGateInputs:
    baseline_dates = [str(day.date) for day in baseline_days]
    for days in (candidate_days, baseline_x2_days, candidate_x2_days):
        if [str(day.date) for day in days] != baseline_dates:
            raise ValueError("A/B comparison requires identical trading dates")
    changed_order_days, changed_days_by_year = _filled_order_changes(
        baseline_days,
        candidate_days,
    )
    return DimensionCappedGateInputs(
        baseline=_performance(baseline_days, initial_cash),
        candidate=_performance(candidate_days, initial_cash),
        baseline_double_friction=_performance(baseline_x2_days, initial_cash),
        candidate_double_friction=_performance(candidate_x2_days, initial_cash),
        changed_order_days=changed_order_days,
        changed_days_by_year=changed_days_by_year,
    )


def _assert_score_attempt_audits_complete(
    audits: Sequence[DimensionCappedScoreAttemptAudit],
    *,
    trade_dates: Sequence[str],
    pool: Sequence[str],
) -> None:
    expected = {
        (str(date), str(code).split(".")[0])
        for date in trade_dates
        for code in pool
    }
    actual = [(audit.decision_date, audit.code) for audit in audits]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("candidate replay lacks a complete pool scoring audit")
    for audit in audits:
        if audit.causal_boundary != "completed_daily_bars_through_T_minus_1":
            raise ValueError("score attempt has an invalid T-1 causal boundary")
        if audit.status == "skipped":
            if not audit.skip_reason:
                raise ValueError("skipped score attempt lacks an exact skip reason")
            continue
        if audit.status != "scored":
            raise ValueError("score attempt has an unknown status")
        if not audit.signal_date or not audit.max_data_date:
            raise ValueError("scored attempt lacks causal data dates")
        if not (
            audit.max_data_date <= audit.signal_date < audit.decision_date
        ):
            raise ValueError("scored attempt violates the T-1 causal boundary")


def _assert_execution_audits_reconcile(
    candidate_days: Sequence[object],
    audits: Sequence[DimensionCappedExecutionAudit],
    performance: DimensionCappedPerformance,
) -> None:
    replay_orders = [
        (str(day.date), order)
        for day in candidate_days
        for order in getattr(day, "orders", ())
    ]
    if len(replay_orders) != len(audits):
        raise ValueError("planned execution audit count does not match replay orders")

    filled_buys = 0
    filled_sells = 0
    for (day_date, order), audit in zip(replay_orders, audits):
        amount = int(getattr(order, "amount_delta", 0))
        filled = bool(getattr(order, "filled", False)) and amount != 0
        if day_date != audit.decision_date or str(order.code).split(".")[0] != audit.code:
            raise ValueError("execution audit order identity does not match replay order")
        if filled != (audit.status == "filled") or amount != audit.filled_amount:
            raise ValueError("execution audit fill status does not match replay order")
        if filled:
            filled_buys += int(amount > 0)
            filled_sells += int(amount < 0)

    if filled_buys != performance.buy_count or filled_sells != performance.sell_count:
        raise ValueError("execution audit filled counts do not match replay metrics")


def run_dimension_capped_training_ab(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
):
    _assert_candidate_rule_manifest_matches_executable()
    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    _assert_approved_warmup_root(warmup_root)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)

    official_params = dict(strategy.get_default_params())
    candidate_params = dict(official_params)
    candidate_params.update({
        "buy_threshold": candidate_rules.BUY_THRESHOLD,
        "sell_threshold": candidate_rules.ORDINARY_SELL_THRESHOLD,
        "min_signal_hold_days": MIN_SIGNAL_HOLD_DAYS,
    })
    pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    official_source = build_training_signal_adapter(
        loader,
        warmup_root=warmup_root,
    )
    cached = PrecomputedSignalAdapter.from_source(
        official_source,
        trade_dates=trade_dates,
        codes=pool,
    )

    baseline_days, _ = _run_arm(
        loader,
        cached,
        LocalCrossSignalOrderPlanner,
        official_params,
        pool,
        trade_dates,
        initial_cash,
        None,
    )
    candidate_source = DimensionCappedScoreAdapter(cached)
    candidate_days, candidate_planner = _run_arm(
        loader,
        candidate_source,
        DimensionCappedOrderPlanner,
        candidate_params,
        pool,
        trade_dates,
        initial_cash,
        None,
    )
    baseline_x2_days, _ = _run_arm(
        loader,
        cached,
        LocalCrossSignalOrderPlanner,
        official_params,
        pool,
        trade_dates,
        initial_cash,
        DOUBLE_FRICTION,
    )
    candidate_x2_days, _ = _run_arm(
        loader,
        candidate_source,
        DimensionCappedOrderPlanner,
        candidate_params,
        pool,
        trade_dates,
        initial_cash,
        DOUBLE_FRICTION,
    )
    inputs = _build_gate_inputs(
        baseline_days,
        candidate_days,
        baseline_x2_days,
        candidate_x2_days,
        initial_cash,
    )
    score_attempt_audits = tuple(candidate_planner.score_attempt_audits)
    execution_audits = tuple(candidate_planner.execution_audits)
    _assert_score_attempt_audits_complete(
        score_attempt_audits,
        trade_dates=trade_dates,
        pool=pool,
    )
    _assert_execution_audits_reconcile(
        candidate_days,
        execution_audits,
        inputs.candidate,
    )
    gate = evaluate_dimension_capped_gate(inputs)
    return DimensionCappedComparisonReport(
        config=dimension_capped_training_config(),
        inputs=inputs,
        gate=gate,
        decision_audits=tuple(candidate_planner.decision_audits),
        score_attempt_audits=score_attempt_audits,
        execution_audits=execution_audits,
    )


def format_dimension_capped_comparison(
    report: DimensionCappedComparisonReport,
) -> str:
    """Render deterministic local-screen evidence and one terminal action."""

    inputs = report.inputs
    rows = (
        ("BASELINE", inputs.baseline),
        ("CANDIDATE", inputs.candidate),
        ("BASELINE_X2_FRICTION", inputs.baseline_double_friction),
        ("CANDIDATE_X2_FRICTION", inputs.candidate_double_friction),
    )
    lines = [
        "candidate=%s" % report.config.candidate_name,
        (
            "hypothesis=capping correlated score dimensions can improve win rate "
            "without materially weakening return, risk, trade count, or friction robustness"
        ),
        "authority=local_screen_only",
        (
            "frozen_window=%s..%s execution=%s candidate_variants=%d"
            % (
                report.config.training_start,
                report.config.training_end,
                report.config.execution_time,
                report.config.candidate_variants,
            )
        ),
        "future_function_audit=T-1_only;causal_score_and_order_evidence_only",
        "candidate_rule_manifest=%s" % _canonical_manifest_text(
            approved_candidate_rule_manifest()
        ),
        "candidate_rule_fingerprint=%s" % candidate_rule_fingerprint(),
        "",
        "METRICS",
    ]
    for label, performance in rows:
        lines.append(_performance_line(label, performance))
        lines.append(
            "%s_ANNUAL_RETURNS=%s" % (
                label,
                ",".join(
                    "%d:%s" % (year, _annual_return_text(performance, year))
                    for year in TRAINING_YEARS
                ),
            )
        )

    baseline_closed = inputs.baseline.closed_trade_count
    retention = (
        "not_applicable"
        if baseline_closed <= 0
        else "%.2f%%" % (
            inputs.candidate.closed_trade_count / baseline_closed * 100.0
        )
    )
    lines.extend([
        "",
        "MATERIALITY_AND_GATE",
        "changed_order_days=%d" % inputs.changed_order_days,
        "changed_days_by_year=%s" % ",".join(
            "%d:%d" % (year, inputs.changed_days_by_year.get(year, 0))
            for year in TRAINING_YEARS
        ),
        "closed_trade_retention=%s" % retention,
        "gate_passed=%s" % _bool_text(report.gate.passed),
    ])
    if report.gate.reasons:
        lines.extend("gate_reason=%s" % reason for reason in report.gate.reasons)
    else:
        lines.append("gate_reason=none")

    lines.extend(["", "CAUSAL_DECISION_AUDIT"])
    if not report.decision_audits:
        lines.append("audit=none")
    else:
        lines.extend(_audit_line(audit) for audit in report.decision_audits)
    lines.extend(["", "SCORE_ATTEMPT_AUDIT"])
    if not report.score_attempt_audits:
        lines.append("score_attempt=none")
    else:
        lines.extend(
            _score_attempt_audit_line(audit)
            for audit in report.score_attempt_audits
        )
    filled_execution_count = sum(
        audit.status == "filled" for audit in report.execution_audits
    )
    lines.extend([
        "",
        "EXECUTION_AUDIT planned_orders=%d filled_orders=%d" % (
            len(report.execution_audits),
            filled_execution_count,
        ),
    ])
    if not report.execution_audits:
        lines.append("execution=none")
    else:
        lines.extend(
            _execution_audit_line(audit)
            for audit in report.execution_audits
        )
    lines.extend([
        "",
        "terminal_action=%s" % (
            "ELIGIBLE_FOR_JOINQUANT_PLAN" if report.gate.passed else "STOP"
        ),
    ])
    return "\n".join(lines)


def write_report_text(report_path: Path | str, text: str) -> None:
    """Write a report outside both immutable market-data roots."""

    destination = Path(report_path).expanduser().resolve()
    for immutable_root in (APPROVED_TRAINING_ROOT, APPROVED_WARMUP_ROOT):
        root = Path(immutable_root).expanduser().resolve()
        if destination == root or root in destination.parents:
            raise ValueError("report path is under an immutable data root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise FileExistsError(
            "report writer refuses to overwrite an existing report"
        ) from exc


def main(report_path: Path | str = REPORT_PATH) -> int:
    """Run the one fixed local screen, persist its report, and return gate status."""

    report = run_dimension_capped_training_ab()
    text = format_dimension_capped_comparison(report) + "\n"
    write_report_text(report_path, text)
    sys.stdout.write(text)
    return 0 if report.gate.passed else 1


def _performance_line(
    label: str,
    item: DimensionCappedPerformance,
) -> str:
    return (
        "%s total_return=%.2f%% annualized_return=%.2f%% max_drawdown=%.2f%% "
        "sharpe=%s sortino=%s win_rate=%.2f%% profit_loss_ratio=%s "
        "buy_count=%d sell_count=%d closed_trade_count=%d"
        % (
            label,
            item.total_return * 100.0,
            item.annualized_return * 100.0,
            item.max_drawdown * 100.0,
            _format_optional_ratio(item.sharpe_ratio),
            _format_optional_ratio(item.sortino_ratio),
            item.win_rate * 100.0,
            _format_optional_ratio(item.profit_loss_ratio),
            item.buy_count,
            item.sell_count,
            item.closed_trade_count,
        )
    )


def _format_optional_ratio(value: float | None) -> str:
    return "not_applicable" if value is None else "%.3f" % value


def _annual_return_text(
    performance: DimensionCappedPerformance,
    year: int,
) -> str:
    value = performance.annual_returns.get(year)
    return "missing" if value is None else "%.2f%%" % (value * 100.0)


def _audit_line(audit: DimensionCappedDecisionAudit) -> str:
    raw = ",".join(
        "%s:%.3f" % (name, value)
        for name, value in audit.raw_contributions
    ) or "none"
    blocks = ",".join(audit.hard_block_reasons) or "none"
    return (
        "decision_date=%s signal_date=%s max_data_date=%s code=%s held=%s "
        "buy_reversal=%.3f buy_location=%.3f buy_trend=%.3f volume_rank=%.3f "
        "buy_total=%.3f sell_weakness=%.3f sell_damage=%.3f sell_total=%.3f "
        "kdj_tier=%s macd_confirmation=%s raw_contributions=%s "
        "adx_protected=%s atr_stop=%s min_hold_blocked=%s "
        "hard_block_reasons=%s order_reason=%s"
        % (
            audit.decision_date,
            audit.signal_date,
            audit.max_data_date,
            audit.code,
            _bool_text(audit.held),
            audit.buy_reversal,
            audit.buy_location,
            audit.buy_trend,
            audit.volume_rank,
            audit.buy_total,
            audit.sell_weakness,
            audit.sell_damage,
            audit.sell_total,
            audit.kdj_tier,
            audit.macd_confirmation,
            raw,
            _bool_text(audit.adx_protected),
            _bool_text(audit.atr_stop),
            _bool_text(audit.min_hold_blocked),
            blocks,
            audit.order_reason or "none",
        )
    )


def _score_attempt_audit_line(audit: DimensionCappedScoreAttemptAudit) -> str:
    return (
        "decision_date=%s code=%s status=%s causal_boundary=%s "
        "t_minus_one_date=%s signal_date=%s max_data_date=%s skip_reason=%s"
        % (
            audit.decision_date,
            audit.code,
            audit.status,
            audit.causal_boundary,
            audit.t_minus_one_date or "not_available",
            audit.signal_date or "not_available",
            audit.max_data_date or "not_available",
            audit.skip_reason or "none",
        )
    )


def _optional_float_text(value: float | None) -> str:
    return "not_applicable" if value is None else "%.6f" % value


def _execution_audit_line(audit: DimensionCappedExecutionAudit) -> str:
    return (
        "decision_date=%s plan_sequence=%d code=%s planned_side=%s "
        "planned_reason=%s target_value=%.6f execution_status=%s "
        "filled_amount=%d execution_price=%s commission=%s unfilled_reason=%s "
        "atr_highest_close=%s atr_input=%s atr_position_cost=%s "
        "atr_decision_price=%s atr_stop_threshold=%s"
        % (
            audit.decision_date,
            audit.plan_sequence,
            audit.code,
            audit.planned_side,
            audit.planned_reason,
            audit.target_value,
            audit.status,
            audit.filled_amount,
            _optional_float_text(audit.execution_price),
            _optional_float_text(audit.commission),
            audit.unfilled_reason or "none",
            _optional_float_text(audit.atr_highest_close),
            _optional_float_text(audit.atr_input),
            _optional_float_text(audit.atr_position_cost),
            _optional_float_text(audit.atr_decision_price),
            _optional_float_text(audit.atr_stop_threshold),
        )
    )


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


class DimensionCappedOrderPlanner(LocalCrossSignalOrderPlanner):
    """Plan capped-score candidate trades without changing shared planner rules."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params = dict(self.params)
        self.params["min_signal_hold_days"] = MIN_SIGNAL_HOLD_DAYS
        self.decision_audits: list[DimensionCappedDecisionAudit] = []
        self.score_attempt_audits: list[DimensionCappedScoreAttemptAudit] = []
        self.execution_audits: list[DimensionCappedExecutionAudit] = []
        self._current_t_minus_one_date: str | None = None
        self._pending_execution_plans: dict[str, list[dict[str, object]]] = {}

    def _score_pool(self, current_date: str) -> List[dict]:
        scores = []
        for code in self.etf_pool:
            score, reason = self.signal_adapter.score(code, current_date, return_reason=True)
            normalized_code = str(code).split(".")[0]
            if score is None:
                if not reason:
                    raise ValueError("skipped score attempt lacks an exact skip reason")
                self.score_attempt_audits.append(DimensionCappedScoreAttemptAudit(
                    decision_date=str(current_date),
                    code=normalized_code,
                    status="skipped",
                    causal_boundary="completed_daily_bars_through_T_minus_1",
                    t_minus_one_date=self._current_t_minus_one_date,
                    signal_date=None,
                    max_data_date=None,
                    skip_reason=str(reason),
                ))
                continue
            score = dict(score)
            score["code"] = str(score.get("code", normalized_code)).split(".")[0]
            signal_date = str(score.get("signal_date", "")) or None
            max_data_date = str(score.get("max_data_date", "")) or None
            self.score_attempt_audits.append(DimensionCappedScoreAttemptAudit(
                decision_date=str(current_date),
                code=normalized_code,
                status="scored",
                causal_boundary="completed_daily_bars_through_T_minus_1",
                t_minus_one_date=self._current_t_minus_one_date or signal_date,
                signal_date=signal_date,
                max_data_date=max_data_date,
                skip_reason=None,
            ))
            scores.append(score)
        return sort_dimension_capped_candidates(scores)

    def _should_force_signal_sell(self, score: Mapping[str, object]) -> bool:
        return should_dimension_capped_signal_sell(dict(score))

    def _candidate_target_value(self, broker, current_prices, current_date: str) -> float:
        equal_weight = (
            self._total_value(broker, current_prices)
            * float(self.params["base_ratio"])
            / int(self.params["max_hold"])
        )
        return equal_weight * self._portfolio_atr_stress_buy_scale(current_date)

    def plan_orders(
        self,
        current_date: str,
        previous_date: str | None,
        broker,
        current_prices: Mapping[str, float] | None = None,
    ) -> List[Mapping[str, float]]:
        prices = current_prices or {}
        self._current_t_minus_one_date = (
            str(previous_date) if previous_date is not None else None
        )
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}
        self.last_scores = score_map

        orders: List[Mapping[str, float]] = []
        atr_contexts = self._atr_audit_contexts(broker, prices)
        sold_codes = self._atr_stop_codes(broker, prices)
        force_stopped = set(sold_codes)
        for code in sorted(sold_codes):
            orders.append({"code": code, "target_value": 0.0, "reason": "atr_stop"})

        for code in list(broker.positions.keys()):
            if code in sold_codes:
                continue
            score = score_map.get(code)
            if score is None:
                continue
            if not strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params["min_signal_hold_days"],
                trade_days=self.trade_dates,
            ):
                continue
            if self._should_force_signal_sell(score):
                orders.append({
                    "code": code,
                    "target_value": 0.0,
                    "reason": "dimension_capped_signal_sell",
                })
                sold_codes.add(code)

        held_after_sell = [
            code for code in broker.positions.keys() if code not in sold_codes
        ]
        slots = int(self.params["max_hold"]) - len(held_after_sell)
        if slots > 0:
            candidates = [
                score for score in scores
                if is_dimension_capped_buy_candidate(score, set(held_after_sell))
                and score["code"] not in force_stopped
            ]
            for score in candidates[:slots]:
                orders.append({
                    "code": score["code"],
                    "target_value": self._candidate_target_value(broker, prices, current_date),
                    "reason": "dimension_capped_buy",
                })

        order_reasons = {
            str(order["code"]).split(".")[0]: str(order["reason"])
            for order in orders
        }
        originally_held = {
            str(code).split(".")[0] for code in broker.positions.keys()
        }
        for score in scores:
            code = str(score["code"]).split(".")[0]
            held = code in originally_held
            sell_allowed = strategy.can_sell_by_signal(
                self.buy_dates.get(code),
                current_date,
                min_hold_days=self.params["min_signal_hold_days"],
                trade_days=self.trade_dates,
            ) if held else True
            self.decision_audits.append(DimensionCappedDecisionAudit(
                decision_date=str(current_date),
                signal_date=str(score.get("signal_date", "")),
                max_data_date=str(score.get("max_data_date", "")),
                code=code,
                held=held,
                buy_reversal=float(score.get("reversal_score", 0.0) or 0.0),
                buy_location=float(score.get("location_score", 0.0) or 0.0),
                buy_trend=float(score.get("trend_score", 0.0) or 0.0),
                volume_rank=float(score.get("volume_rank_score", 0.0) or 0.0),
                buy_total=float(score.get("buy_score", 0.0) or 0.0),
                sell_weakness=float(score.get("sell_weakness_score", 0.0) or 0.0),
                sell_damage=float(score.get("sell_damage_score", 0.0) or 0.0),
                sell_total=float(score.get("sell_score", 0.0) or 0.0),
                kdj_tier=_kdj_tier(score.get("k")),
                macd_confirmation=_macd_confirmation(score),
                raw_contributions=_raw_contributions(score),
                adx_protected=_is_adx_protected(
                    score,
                    held=held,
                    sell_allowed=sell_allowed,
                    atr_stop=code in force_stopped,
                ),
                atr_stop=code in force_stopped,
                min_hold_blocked=held and not sell_allowed,
                hard_block_reasons=_hard_block_reasons(
                    score,
                    held=held,
                    force_stopped=code in force_stopped,
                ),
                order_reason=order_reasons.get(code),
            ))
        self._pending_execution_plans[str(current_date)] = [
            {
                "code": str(order["code"]).split(".")[0],
                "target_value": float(order["target_value"]),
                "reason": str(order["reason"]),
                "planned_side": (
                    "sell" if float(order["target_value"]) <= 0.0 else "buy"
                ),
                "atr_context": (
                    atr_contexts.get(str(order["code"]).split(".")[0], {})
                    if str(order["reason"]) == "atr_stop"
                    else {}
                ),
            }
            for order in orders
        ]
        return orders

    def _atr_audit_contexts(
        self,
        broker,
        current_prices: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        contexts: dict[str, dict[str, float]] = {}
        for code, position in broker.positions.items():
            if code not in current_prices:
                continue
            highest = self.highest_since_buy.get(code)
            atr_input = self.entry_atr.get(code)
            decision_price = float(current_prices[code])
            if highest is None or atr_input is None or decision_price <= 0.0:
                continue
            stop_threshold = strategy.calc_stop_price(
                float(highest),
                float(atr_input),
                float(position.avg_cost),
                self.params,
            )
            contexts[str(code).split(".")[0]] = {
                "atr_highest_close": float(highest),
                "atr_input": float(atr_input),
                "atr_position_cost": float(position.avg_cost),
                "atr_decision_price": decision_price,
                "atr_stop_threshold": float(stop_threshold),
            }
        return contexts

    def on_orders_filled(self, current_date: str, orders) -> None:
        pending = self._pending_execution_plans.pop(str(current_date), None)
        if pending is None or len(pending) != len(orders):
            raise ValueError("planned execution audit count does not match broker results")
        for sequence, (plan, order) in enumerate(zip(pending, orders)):
            code = str(order.code).split(".")[0]
            if code != plan["code"]:
                raise ValueError("planned execution audit order identity mismatch")
            amount = int(getattr(order, "amount_delta", 0))
            filled = bool(getattr(order, "filled", False)) and amount != 0
            unfilled_reason = None if filled else str(getattr(order, "reason", "") or "")
            if not filled and not unfilled_reason:
                raise ValueError("unfilled planned order lacks an exact broker reason")
            atr_context = dict(plan["atr_context"])
            self.execution_audits.append(DimensionCappedExecutionAudit(
                decision_date=str(current_date),
                plan_sequence=sequence,
                code=code,
                planned_side=str(plan["planned_side"]),
                planned_reason=str(plan["reason"]),
                target_value=float(plan["target_value"]),
                status="filled" if filled else "unfilled",
                filled_amount=amount,
                execution_price=float(order.exec_price) if filled else None,
                commission=float(order.commission) if filled else None,
                unfilled_reason=unfilled_reason,
                atr_highest_close=atr_context.get("atr_highest_close"),
                atr_input=atr_context.get("atr_input"),
                atr_position_cost=atr_context.get("atr_position_cost"),
                atr_decision_price=atr_context.get("atr_decision_price"),
                atr_stop_threshold=atr_context.get("atr_stop_threshold"),
            ))
        super().on_orders_filled(current_date, orders)


def _kdj_tier(value: object) -> str:
    try:
        k = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if k <= 20.0:
        return "oversold"
    if k <= 30.0:
        return "low"
    if k >= 80.0:
        return "overbought"
    if k >= 70.0:
        return "high"
    return "neutral"


def _macd_confirmation(score: Mapping[str, object]) -> str:
    upward = bool(score.get("macd_cross_up"))
    downward = bool(score.get("macd_cross_down"))
    if upward and downward:
        return "conflict"
    if upward:
        return "up"
    if downward:
        return "down"
    return "none"


def _raw_contributions(
    score: Mapping[str, object],
) -> tuple[tuple[str, float], ...]:
    groups = (
        ("raw_buy_reversal_contributions", "buy_"),
        ("raw_location_contributions", "buy_location_"),
        ("raw_trend_contributions", "buy_trend_"),
        ("raw_sell_weakness_contributions", "sell_weakness_"),
        ("raw_sell_damage_contributions", "sell_damage_"),
    )
    rows = []
    for field, prefix in groups:
        values = score.get(field, {})
        if not isinstance(values, Mapping):
            continue
        rows.extend(
            (prefix + str(name), float(value))
            for name, value in values.items()
        )
    return tuple(rows)


def _is_adx_protected(
    score: Mapping[str, object],
    *,
    held: bool,
    sell_allowed: bool,
    atr_stop: bool,
) -> bool:
    if not held or not sell_allowed or atr_stop:
        return False
    weakness = float(score.get("sell_weakness_score", 0.0) or 0.0)
    damage = float(score.get("sell_damage_score", 0.0) or 0.0)
    total = float(score.get("sell_score", 0.0) or 0.0)
    ordinary = (
        weakness >= candidate_rules.SELL_WEAKNESS_MIN
        and damage >= candidate_rules.SELL_DAMAGE_MIN
        and total >= candidate_rules.ORDINARY_SELL_THRESHOLD
    )
    severe = (
        weakness >= candidate_rules.SEVERE_WEAKNESS_MIN
        and damage >= candidate_rules.SEVERE_DAMAGE_MIN
    )
    return ordinary and not severe and strategy.is_strong_adx_uptrend(score)


def _hard_block_reasons(
    score: Mapping[str, object],
    *,
    held: bool,
    force_stopped: bool,
) -> tuple[str, ...]:
    reasons = []
    if not bool(score.get("buy_allowed")):
        reasons.append("buy_not_allowed")
    if bool(score.get("close_far_above_ma20")):
        reasons.append("close_far_above_ma20")
    if bool(score.get("downside_continuation")):
        reasons.append("downside_continuation")
    if bool(score.get("weak_repair_blocked")):
        reasons.append("weak_repair_blocked")
    if (
        float(score.get("reversal_score", 0.0) or 0.0)
        < candidate_rules.BUY_REVERSAL_MIN
    ):
        reasons.append("buy_reversal_below_12")
    if (
        float(score.get("location_score", 0.0) or 0.0)
        < candidate_rules.BUY_LOCATION_MIN
    ):
        reasons.append("buy_location_below_7")
    if (
        float(score.get("trend_score", 0.0) or 0.0)
        < candidate_rules.BUY_TREND_MIN
    ):
        reasons.append("buy_trend_below_6")
    if (
        float(score.get("buy_score", 0.0) or 0.0)
        < candidate_rules.BUY_THRESHOLD
    ):
        reasons.append("buy_total_below_40")
    if has_raw_sell_conflict(dict(score)):
        reasons.append("sell_conflict")
    if held:
        reasons.append("already_held")
    if force_stopped:
        reasons.append("same_day_atr_stop")
    return tuple(reasons)


if __name__ == "__main__":
    raise SystemExit(main())
