# -*- coding: utf-8 -*-
"""Isolated current-state KDJ candidate with moderate asymmetric points.

The candidate adds 20/10 points on the buy side and 10/5 points on the sell
side. Only the current causal T-1 K value is used. The official sell decider
continues to require price confirmation and apply ADX protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cross_signal_strategy.local.local_data_loader import APPROVED_WARMUP_ROOT
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    ExtremeZoneComparisonReport,
    _performance_line,
)
from cross_signal_strategy.research.kdj_tiered_current_state_candidate import (
    KdjTieredCurrentStateScoreAdapter,
)
from cross_signal_strategy.research.kdj_tiered_persistence_candidate import (
    NEAR_BUY_K_MAX,
    NEAR_SELL_K_MIN,
    STRONG_BUY_K_MAX,
    STRONG_SELL_K_MIN,
    _finite_float,
    run_kdj_tiered_persistence_training_ab,
)


STRONG_BUY_POINTS = 20.0
NEAR_BUY_POINTS = 10.0
STRONG_SELL_POINTS = 10.0
NEAR_SELL_POINTS = 5.0


@dataclass(frozen=True)
class KdjTieredModeratePointsScoreAdapter(KdjTieredCurrentStateScoreAdapter):
    """Apply the fixed moderate tier from the current T-1 snapshot only."""

    def _retained_state(self, code: str, current_date: str) -> tuple[str | None, float]:
        score = self.source.score(code, current_date)
        k_value = _finite_float(score.get("k")) if score is not None else None
        return _moderate_state_tier(k_value)


def _moderate_state_tier(k_value: float | None) -> tuple[str | None, float]:
    if k_value is None:
        return None, 0.0
    if k_value <= STRONG_BUY_K_MAX:
        return "buy", STRONG_BUY_POINTS
    if k_value <= NEAR_BUY_K_MAX:
        return "buy", NEAR_BUY_POINTS
    if k_value >= STRONG_SELL_K_MIN:
        return "sell", STRONG_SELL_POINTS
    if k_value >= NEAR_SELL_K_MIN:
        return "sell", NEAR_SELL_POINTS
    return None, 0.0


def run_kdj_tiered_moderate_points_training_ab(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> ExtremeZoneComparisonReport:
    """Run the one frozen moderate-points candidate on training data only."""

    return run_kdj_tiered_persistence_training_ab(
        loader=loader,
        initial_cash=initial_cash,
        warmup_root=warmup_root,
        adapter_factory=KdjTieredModeratePointsScoreAdapter,
    )


def format_kdj_tiered_moderate_points_comparison(
    report: ExtremeZoneComparisonReport,
) -> str:
    lines = [
        "KDJ tiered moderate-points candidate (2019-2021; local screen only)",
        (
            "buy=K<=20 +20; 20<K<=30 +10; downside continuation blocks buy; "
            "sell=70<=K<80 +5; K>=80 +10; current T-1 only"
        ),
        (
            "sell_path=official threshold 30 plus price confirmation and ADX "
            "protection; minimum five-session hold retained"
        ),
        _performance_line("BASELINE", report.baseline),
        _performance_line("CANDIDATE", report.candidate),
        _performance_line(
            "BASELINE_X2_FRICTION", report.baseline_double_friction
        ),
        _performance_line(
            "CANDIDATE_X2_FRICTION", report.candidate_double_friction
        ),
        "CHANGED days=%d by_year=%s"
        % (report.changed_order_days, report.changed_days_by_year),
        "GATE=%s" % ("PASS" if report.gate.passed else "REJECT"),
    ]
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.append("authority=local_screen_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def main() -> None:
    report = run_kdj_tiered_moderate_points_training_ab()
    print(format_kdj_tiered_moderate_points_comparison(report))


if __name__ == "__main__":
    main()
