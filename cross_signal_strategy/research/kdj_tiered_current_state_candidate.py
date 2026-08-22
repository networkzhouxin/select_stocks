# -*- coding: utf-8 -*-
"""Isolated current-session tiered KDJ state-score candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cross_signal_strategy.local.local_data_loader import APPROVED_WARMUP_ROOT
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    ExtremeZoneComparisonReport,
    _performance_line,
)
from cross_signal_strategy.research.kdj_tiered_persistence_candidate import (
    KdjTieredPersistenceScoreAdapter,
    _finite_float,
    _state_tier,
    run_kdj_tiered_persistence_training_ab,
)


@dataclass(frozen=True)
class KdjTieredCurrentStateScoreAdapter(KdjTieredPersistenceScoreAdapter):
    """Apply the fixed tier from the current causal T-1 snapshot only."""

    def _retained_state(self, code: str, current_date: str) -> tuple[str | None, float]:
        score = self.source.score(code, current_date)
        k_value = _finite_float(score.get("k")) if score is not None else None
        return _state_tier(k_value)


def run_kdj_tiered_current_state_training_ab(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> ExtremeZoneComparisonReport:
    """Run the one fixed current-state tier candidate on training data only."""

    return run_kdj_tiered_persistence_training_ab(
        loader=loader,
        initial_cash=initial_cash,
        warmup_root=warmup_root,
        adapter_factory=KdjTieredCurrentStateScoreAdapter,
    )


def format_kdj_tiered_current_state_comparison(
    report: ExtremeZoneComparisonReport,
) -> str:
    lines = [
        "KDJ tiered current-state candidate (2019-2021; local screen only)",
        (
            "rule=K<=20 +/-10 strong; K in (20,30] or [70,80) +/-5 near; "
            "current T-1 state only; no retention; "
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
    report = run_kdj_tiered_current_state_training_ab()
    print(format_kdj_tiered_current_state_comparison(report))


if __name__ == "__main__":
    main()
