# -*- coding: utf-8 -*-
"""Isolated three-session, tiered KDJ state-score candidate.

Only official causal T-1 snapshots are decorated.  The most recent extreme
direction inside the current and prior two decision sessions wins; points in
that direction take the maximum tier and never accumulate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    DOUBLE_FRICTION,
    ExtremeZoneComparisonReport,
    _assert_approved_loader,
    _assert_training_dates,
    _changed_days_by_year,
    _performance,
    _performance_line,
    _run_replay,
    evaluate_extreme_zone_gate,
)


STRONG_BUY_K_MAX = 20.0
NEAR_BUY_K_MAX = 30.0
NEAR_SELL_K_MIN = 70.0
STRONG_SELL_K_MIN = 80.0
STRONG_POINTS = 10.0
NEAR_POINTS = 5.0
RETENTION_SESSIONS = 3


@dataclass(frozen=True)
class KdjTieredPersistenceScoreAdapter:
    """Add one non-cumulative KDJ state bonus to an official score."""

    source: object
    trade_dates: Sequence[str]

    def __post_init__(self) -> None:
        dates = tuple(str(value) for value in self.trade_dates)
        if len(dates) != len(set(dates)) or dates != tuple(sorted(dates)):
            raise ValueError("trade_dates must be unique and ascending")
        object.__setattr__(self, "trade_dates", dates)

    def score(self, code: str, current_date: str, return_reason: bool = False):
        current_date = str(current_date)
        base_result = self.source.score(
            code,
            current_date,
            return_reason=return_reason,
        )
        if return_reason:
            base_score, reason = base_result
            if base_score is None:
                return None, reason
            return self._adjust(code, current_date, base_score), reason
        if base_result is None:
            return None
        return self._adjust(code, current_date, base_result)

    def _adjust(
        self,
        code: str,
        current_date: str,
        base_score: Mapping[str, object],
    ) -> dict:
        result = deepcopy(dict(base_score))
        direction, points = self._retained_state(code, current_date)
        buy_bonus = points if direction == "buy" else 0.0
        if bool(result.get("downside_continuation")):
            buy_bonus = 0.0
        sell_bonus = points if direction == "sell" else 0.0
        official_buy = float(result.get("buy_score", 0.0) or 0.0)
        official_sell = float(result.get("sell_score", 0.0) or 0.0)
        result.update(
            {
                "official_buy_score": official_buy,
                "official_sell_score": official_sell,
                "buy_extreme_zone_score": buy_bonus,
                "sell_extreme_zone_score": sell_bonus,
                "extreme_zone_direction": direction,
                "buy_score": max(0.0, official_buy + buy_bonus),
                "sell_score": max(0.0, official_sell + sell_bonus),
            }
        )
        return result

    def _retained_state(self, code: str, current_date: str) -> tuple[str | None, float]:
        try:
            current_index = self.trade_dates.index(current_date)
        except ValueError:
            return None, 0.0
        start = max(0, current_index - RETENTION_SESSIONS + 1)
        window = self.trade_dates[start : current_index + 1]
        newest_direction = None
        maximum_points = 0.0
        for date in reversed(window):
            score = self.source.score(code, date)
            k_value = _finite_float(score.get("k")) if score is not None else None
            direction, points = _state_tier(k_value)
            if direction is None:
                continue
            if newest_direction is None:
                newest_direction = direction
            elif direction != newest_direction:
                break
            maximum_points = max(maximum_points, points)
        return newest_direction, maximum_points


def _state_tier(k_value: float | None) -> tuple[str | None, float]:
    if k_value is None:
        return None, 0.0
    if k_value <= STRONG_BUY_K_MAX:
        return "buy", STRONG_POINTS
    if k_value <= NEAR_BUY_K_MAX:
        return "buy", NEAR_POINTS
    if k_value >= STRONG_SELL_K_MIN:
        return "sell", STRONG_POINTS
    if k_value >= NEAR_SELL_K_MIN:
        return "sell", NEAR_POINTS
    return None, 0.0


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_kdj_tiered_persistence_training_ab(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
    adapter_factory=None,
    candidate_planner_kwargs: Mapping[str, object] | None = None,
) -> ExtremeZoneComparisonReport:
    """Run the one fixed tiered-persistence candidate on training data only."""

    from cross_signal_strategy.local.local_order_planner import (
        LocalCrossSignalOrderPlanner,
        strategy,
    )
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.baseline_report import build_baseline_report
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )

    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    warmup = Path(warmup_root).expanduser().resolve()
    if warmup != Path(APPROVED_WARMUP_ROOT).expanduser().resolve():
        raise ValueError("Use approved warm-up data root only: %s" % APPROVED_WARMUP_ROOT)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)
    params = strategy.get_default_params()
    pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    official_source = build_training_signal_adapter(loader, warmup_root=warmup)
    cached = PrecomputedSignalAdapter.from_source(
        official_source,
        trade_dates=trade_dates,
        codes=pool,
    )
    factory = adapter_factory or KdjTieredPersistenceScoreAdapter
    candidate_adapter = factory(cached, trade_dates)
    candidate_planner_options = dict(candidate_planner_kwargs or {})

    baseline_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
    )
    candidate_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            candidate_adapter,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
            **candidate_planner_options,
        ),
        trade_dates,
        initial_cash,
    )
    baseline_stress_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            cached,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=DOUBLE_FRICTION,
    )
    candidate_stress_days = _run_replay(
        loader,
        LocalCrossSignalOrderPlanner(
            candidate_adapter,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
            **candidate_planner_options,
        ),
        trade_dates,
        initial_cash,
        broker_kwargs=DOUBLE_FRICTION,
    )

    baseline_report = build_baseline_report(baseline_days, initial_cash)
    candidate_report = build_baseline_report(candidate_days, initial_cash)
    baseline_stress_report = build_baseline_report(
        baseline_stress_days, initial_cash
    )
    candidate_stress_report = build_baseline_report(
        candidate_stress_days, initial_cash
    )
    baseline = _performance(baseline_report, baseline_days, initial_cash)
    candidate = _performance(candidate_report, candidate_days, initial_cash)
    baseline_stress = _performance(
        baseline_stress_report, baseline_stress_days, initial_cash
    )
    candidate_stress = _performance(
        candidate_stress_report, candidate_stress_days, initial_cash
    )
    changed_days_by_year = _changed_days_by_year(baseline_days, candidate_days)
    gate = evaluate_extreme_zone_gate(
        baseline,
        candidate,
        changed_days_by_year,
    )
    return ExtremeZoneComparisonReport(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_double_friction_report=baseline_stress_report,
        candidate_double_friction_report=candidate_stress_report,
        baseline=baseline,
        candidate=candidate,
        baseline_double_friction=baseline_stress,
        candidate_double_friction=candidate_stress,
        changed_order_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        gate=gate,
    )


def format_kdj_tiered_persistence_comparison(
    report: ExtremeZoneComparisonReport,
) -> str:
    lines = [
        "KDJ tiered three-session state candidate (2019-2021; local screen only)",
        (
            "rule=K<=20 +/-10 strong; K in (20,30] or [70,80) +/-5 near; "
            "retain 3 sessions; same-direction max; most-recent direction wins; "
            "current downside_continuation blocks buy"
        ),
        _performance_line("BASELINE", report.baseline),
        _performance_line("CANDIDATE", report.candidate),
        _performance_line("BASELINE_X2_FRICTION", report.baseline_double_friction),
        _performance_line("CANDIDATE_X2_FRICTION", report.candidate_double_friction),
        "CHANGED days=%d by_year=%s"
        % (report.changed_order_days, report.changed_days_by_year),
        "GATE=%s" % ("PASS" if report.gate.passed else "REJECT"),
    ]
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.append("authority=local_screen_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def main() -> None:
    report = run_kdj_tiered_persistence_training_ab()
    print(format_kdj_tiered_persistence_comparison(report))


if __name__ == "__main__":
    main()
