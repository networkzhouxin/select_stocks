# -*- coding: utf-8 -*-
"""Isolated training-period order planner for the dimension-capped candidate."""

from __future__ import annotations

from dataclasses import dataclass
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
from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
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
from cross_signal_strategy.research.dimension_capped_score_candidate import (
    DimensionCappedScoreAdapter,
    is_dimension_capped_buy_candidate,
    should_dimension_capped_signal_sell,
    sort_dimension_capped_candidates,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
TRAINING_YEARS = (2019, 2020, 2021)
DOUBLE_FRICTION = FrictionScenarioConfig(
    commission_rate=0.0006,
    min_commission=10.0,
    slippage_rate=0.002,
)


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
class DimensionCappedComparisonReport:
    config: DimensionCappedTrainingConfig
    inputs: DimensionCappedGateInputs
    gate: DimensionCappedGateDecision
    decision_audits: tuple[DimensionCappedDecisionAudit, ...]


def dimension_capped_training_config() -> DimensionCappedTrainingConfig:
    return DimensionCappedTrainingConfig(
        candidate_name="cross-v0.4.0-dimension-capped-candidate",
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        initial_cash=20000.0,
        execution_time="09:35",
        buy_threshold=40.0,
        ordinary_sell_threshold=24.0,
        min_signal_hold_days=5,
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
    if inputs.changed_order_days < 10:
        reasons.append("fewer than 10 changed filled-order days")
    for year in TRAINING_YEARS:
        if int(inputs.changed_days_by_year.get(year, 0)) < 2:
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
        execution_time="09:35",
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


def run_dimension_capped_training_ab(
    loader=None,
    initial_cash: float = 20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
):
    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    _assert_approved_warmup_root(warmup_root)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)

    official_params = dict(strategy.get_default_params())
    candidate_params = dict(official_params)
    candidate_params.update({
        "buy_threshold": 40.0,
        "sell_threshold": 24.0,
        "min_signal_hold_days": 5,
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
    gate = evaluate_dimension_capped_gate(inputs)
    return DimensionCappedComparisonReport(
        config=dimension_capped_training_config(),
        inputs=inputs,
        gate=gate,
        decision_audits=tuple(candidate_planner.decision_audits),
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
        "",
        "METRICS",
    ]
    for label, performance in rows:
        lines.append(_performance_line(label, performance))
        lines.append(
            "%s_ANNUAL_RETURNS=%s" % (
                label,
                ",".join(
                    "%d:%.2f%%" % (
                        year,
                        performance.annual_returns.get(year, 0.0) * 100.0,
                    )
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
    lines.extend([
        "",
        "terminal_action=%s" % (
            "ELIGIBLE_FOR_JOINQUANT_PLAN" if report.gate.passed else "STOP"
        ),
    ])
    return "\n".join(lines) + "\n"


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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


class DimensionCappedOrderPlanner(LocalCrossSignalOrderPlanner):
    """Plan capped-score candidate trades without changing shared planner rules."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params = dict(self.params)
        self.params["min_signal_hold_days"] = 5
        self.decision_audits: list[DimensionCappedDecisionAudit] = []

    def _score_pool(self, current_date: str) -> List[dict]:
        scores = []
        for code in self.etf_pool:
            score, reason = self.signal_adapter.score(code, current_date, return_reason=True)
            if score is None:
                continue
            score = dict(score)
            score["code"] = str(score.get("code", code)).split(".")[0]
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
        scores = self._score_pool(current_date)
        score_map = {score["code"]: score for score in scores}
        self.last_scores = score_map

        orders: List[Mapping[str, float]] = []
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
                adx_protected=_is_adx_protected(score),
                atr_stop=code in force_stopped,
                min_hold_blocked=held and not sell_allowed,
                hard_block_reasons=_hard_block_reasons(
                    score,
                    held=held,
                    force_stopped=code in force_stopped,
                ),
                order_reason=order_reasons.get(code),
            ))
        return orders


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


def _is_adx_protected(score: Mapping[str, object]) -> bool:
    weakness = float(score.get("sell_weakness_score", 0.0) or 0.0)
    damage = float(score.get("sell_damage_score", 0.0) or 0.0)
    total = float(score.get("sell_score", 0.0) or 0.0)
    ordinary = weakness >= 10.0 and damage >= 8.0 and total >= 24.0
    severe = weakness >= 6.0 and damage >= 18.0
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
    if float(score.get("reversal_score", 0.0) or 0.0) < 12.0:
        reasons.append("buy_reversal_below_12")
    if float(score.get("location_score", 0.0) or 0.0) < 7.0:
        reasons.append("buy_location_below_7")
    if float(score.get("trend_score", 0.0) or 0.0) < 6.0:
        reasons.append("buy_trend_below_6")
    if float(score.get("buy_score", 0.0) or 0.0) < 40.0:
        reasons.append("buy_total_below_40")
    if should_dimension_capped_signal_sell(dict(score)):
        reasons.append("sell_conflict")
    if held:
        reasons.append("already_held")
    if force_stopped:
        reasons.append("same_day_atr_stop")
    return tuple(reasons)
