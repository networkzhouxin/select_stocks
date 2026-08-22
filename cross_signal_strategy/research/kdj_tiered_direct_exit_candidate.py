# -*- coding: utf-8 -*-
"""Isolated current-state KDJ tier candidate with direct extreme exits."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from cross_signal_strategy.local.local_data_loader import APPROVED_WARMUP_ROOT
from cross_signal_strategy.local.local_order_planner import strategy
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    ExtremeZoneComparisonReport,
    _performance_line,
)
from cross_signal_strategy.research.kdj_tiered_current_state_candidate import (
    KdjTieredCurrentStateScoreAdapter,
)
from cross_signal_strategy.research.kdj_tiered_persistence_candidate import (
    run_kdj_tiered_persistence_training_ab,
)


def should_force_kdj_extreme_sell(
    score: Mapping[str, object],
    atr_stop_triggered: bool = False,
    params: Mapping[str, object] | None = None,
) -> bool:
    """Bypass price/ADX only when a current KDJ sell tier is present."""

    p = params or strategy.get_default_params()
    if atr_stop_triggered:
        return True
    extreme_bonus = float(score.get("sell_extreme_zone_score", 0.0) or 0.0)
    sell_score = float(score.get("sell_score", 0.0) or 0.0)
    if extreme_bonus > 0.0 and sell_score >= float(p["sell_threshold"]):
        return True
    return strategy.should_force_sell(score, False, p)


def run_kdj_tiered_direct_exit_training_ab(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> ExtremeZoneComparisonReport:
    """Run current-state tiers plus the direct extreme-exit rule."""

    return run_kdj_tiered_persistence_training_ab(
        loader=loader,
        initial_cash=initial_cash,
        warmup_root=warmup_root,
        adapter_factory=KdjTieredCurrentStateScoreAdapter,
        candidate_planner_kwargs={
            "signal_sell_decider": should_force_kdj_extreme_sell,
        },
    )


def format_kdj_tiered_direct_exit_comparison(
    report: ExtremeZoneComparisonReport,
) -> str:
    lines = [
        "KDJ tiered current-state direct-exit candidate "
        "(2019-2021; local screen only)",
        (
            "buy=K<=20 +10; 20<K<=30 +5; current downside continuation "
            "blocks buy; sell=70<=K<80 +5; K>=80 +10; current T-1 only"
        ),
        (
            "direct_exit=sell bonus > 0 and final sell score >= 30; "
            "bypass price confirmation and ADX; minimum five-session hold retained"
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
    report = run_kdj_tiered_direct_exit_training_ab()
    print(format_kdj_tiered_direct_exit_comparison(report))


if __name__ == "__main__":
    main()
