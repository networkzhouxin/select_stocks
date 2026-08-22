# -*- coding: utf-8 -*-
"""Isolated KDJ candidate whose buy bonus affects ranking only.

Official buy score remains the eligibility and position-sizing score. The
current T-1 KDJ buy tier becomes a secondary ranking score among candidates
that already satisfy the official threshold. The frozen KDJ sell tier remains
part of the unified sell score and keeps every official sell protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import (
    LocalCrossSignalOrderPlanner,
    strategy,
)
from cross_signal_strategy.research.extreme_zone_score_candidate import (
    DOUBLE_FRICTION,
    ExtremeZoneGateDecision,
    ExtremeZonePerformance,
    _assert_approved_loader,
    _assert_training_dates,
    _changed_days_by_year,
    _performance,
    _performance_line,
    _retains_ratio,
    _run_replay,
)
from cross_signal_strategy.research.kdj_tiered_moderate_points_candidate import (
    KdjTieredModeratePointsScoreAdapter,
)


@dataclass(frozen=True)
class KdjRankingOnlyBuyScoreAdapter(KdjTieredModeratePointsScoreAdapter):
    """Keep official buy eligibility while exposing KDJ-adjusted rank."""

    def _adjust(self, code, current_date, base_score):
        result = super()._adjust(code, current_date, base_score)
        official_buy = float(result.get("official_buy_score", 0.0) or 0.0)
        result.update({
            "buy_rank_score": float(result.get("buy_score", 0.0) or 0.0),
            "buy_score": official_buy,
            "buy_extreme_zone_role": "ranking_only",
        })
        return result


@dataclass
class KdjRankingOnlyBuyPlanner(LocalCrossSignalOrderPlanner):
    """Rank by the decorated score after official eligibility is preserved."""

    def _score_pool(self, current_date):
        scores = []
        for code in self.etf_pool:
            score, reason = self.signal_adapter.score(
                code,
                current_date,
                return_reason=True,
            )
            if score is None:
                continue
            score = dict(score)
            score["code"] = str(score.get("code", code)).split(".")[0]
            scores.append(score)
        return sorted(scores, key=lambda item: (
            -float(item.get("buy_rank_score", item.get("buy_score", 0.0)) or 0.0),
            -float(item.get("reversal_score", 0.0) or 0.0),
            item.get("code", ""),
        ))


@dataclass(frozen=True)
class KdjRankingOnlyBuyReport:
    official: ExtremeZonePerformance
    current_kdj: ExtremeZonePerformance
    candidate: ExtremeZonePerformance
    official_double_friction: ExtremeZonePerformance
    current_kdj_double_friction: ExtremeZonePerformance
    candidate_double_friction: ExtremeZonePerformance
    changed_vs_current_days: int
    changed_vs_current_by_year: Dict[int, int]
    changed_vs_official_days: int
    changed_vs_official_by_year: Dict[int, int]
    gate: ExtremeZoneGateDecision


def evaluate_kdj_ranking_only_gate(
    official: ExtremeZonePerformance,
    current_kdj: ExtremeZonePerformance,
    candidate: ExtremeZonePerformance,
    official_double_friction: ExtremeZonePerformance,
    candidate_double_friction: ExtremeZonePerformance,
    changed_vs_current_by_year: Mapping[int, int],
    changed_days_vs_official: int,
) -> ExtremeZoneGateDecision:
    """Evaluate the pre-registered accuracy-first adoption gate."""

    reasons = []
    for year in (2019, 2020, 2021):
        if int(changed_vs_current_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed filled-order day" % year)
    if changed_days_vs_official <= 0:
        reasons.append("candidate has no material effect versus official path")
    if candidate.win_rate <= current_kdj.win_rate:
        reasons.append("candidate win rate does not improve current KDJ path")
    if candidate.win_rate < official.win_rate:
        reasons.append("candidate win rate worsens official path")
    if candidate.total_return < official.total_return * 0.95:
        reasons.append("candidate retains less than 95% of official return")
    if candidate.max_drawdown > official.max_drawdown + 0.005:
        reasons.append("candidate maximum drawdown worsens by more than 0.5pp")
    for label, candidate_value, official_value in (
        ("Sharpe", candidate.sharpe_ratio, official.sharpe_ratio),
        ("Sortino", candidate.sortino_ratio, official.sortino_ratio),
        ("profit/loss ratio", candidate.profit_loss_ratio, official.profit_loss_ratio),
    ):
        if not _retains_ratio(candidate_value, official_value, 0.95):
            reasons.append("candidate %s retains less than 95%%" % label)
    for year in (2019, 2020, 2021):
        annual = candidate.annual_returns.get(year)
        if annual is None:
            reasons.append("%d candidate annual return is missing" % year)
        elif official.annual_returns.get(year, 0.0) > 0 and annual <= 0:
            reasons.append("%d candidate annual return turns non-positive" % year)
    if (
        candidate_double_friction.total_return
        < official_double_friction.total_return * 0.95
    ):
        reasons.append("double-friction return retains less than 95% of official")
    if candidate_double_friction.win_rate < official_double_friction.win_rate:
        reasons.append("double-friction stress win rate worsens official")
    return ExtremeZoneGateDecision(not reasons, tuple(reasons))


def run_kdj_ranking_only_training_comparison(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> KdjRankingOnlyBuyReport:
    """Run official, current KDJ, and ranking-only paths on training data."""

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
    source = build_training_signal_adapter(loader, warmup_root=warmup)
    official_source = PrecomputedSignalAdapter.from_source(
        source,
        trade_dates=trade_dates,
        codes=pool,
    )
    current_source = KdjTieredModeratePointsScoreAdapter(official_source, trade_dates)
    candidate_source = KdjRankingOnlyBuyScoreAdapter(official_source, trade_dates)

    def replay(score_source, planner_type, broker_kwargs=None):
        planner = planner_type(
            score_source,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        )
        return _run_replay(
            loader,
            planner,
            trade_dates,
            initial_cash,
            broker_kwargs=broker_kwargs,
        )

    official_days = replay(official_source, LocalCrossSignalOrderPlanner)
    current_days = replay(current_source, LocalCrossSignalOrderPlanner)
    candidate_days = replay(candidate_source, KdjRankingOnlyBuyPlanner)
    official_stress_days = replay(
        official_source,
        LocalCrossSignalOrderPlanner,
        DOUBLE_FRICTION,
    )
    current_stress_days = replay(
        current_source,
        LocalCrossSignalOrderPlanner,
        DOUBLE_FRICTION,
    )
    candidate_stress_days = replay(
        candidate_source,
        KdjRankingOnlyBuyPlanner,
        DOUBLE_FRICTION,
    )

    def performance(days):
        report = build_baseline_report(days, initial_cash)
        return _performance(report, days, initial_cash)

    official = performance(official_days)
    current = performance(current_days)
    candidate = performance(candidate_days)
    official_stress = performance(official_stress_days)
    current_stress = performance(current_stress_days)
    candidate_stress = performance(candidate_stress_days)
    changed_current = _changed_days_by_year(current_days, candidate_days)
    changed_official = _changed_days_by_year(official_days, candidate_days)
    gate = evaluate_kdj_ranking_only_gate(
        official,
        current,
        candidate,
        official_stress,
        candidate_stress,
        changed_current,
        sum(changed_official.values()),
    )
    return KdjRankingOnlyBuyReport(
        official=official,
        current_kdj=current,
        candidate=candidate,
        official_double_friction=official_stress,
        current_kdj_double_friction=current_stress,
        candidate_double_friction=candidate_stress,
        changed_vs_current_days=sum(changed_current.values()),
        changed_vs_current_by_year=dict(sorted(changed_current.items())),
        changed_vs_official_days=sum(changed_official.values()),
        changed_vs_official_by_year=dict(sorted(changed_official.items())),
        gate=gate,
    )


def format_kdj_ranking_only_comparison(report: KdjRankingOnlyBuyReport) -> str:
    lines = [
        "KDJ ranking-only buy candidate (2019-2021; local screen only)",
        (
            "buy qualification/size=official score >=60; rank=official score "
            "+ current KDJ tier 20/10; sell tier remains unified 10/5"
        ),
        _performance_line("OFFICIAL", report.official),
        _performance_line("CURRENT_KDJ", report.current_kdj),
        _performance_line("CANDIDATE", report.candidate),
        _performance_line("OFFICIAL_X2_FRICTION", report.official_double_friction),
        _performance_line("CURRENT_KDJ_X2_FRICTION", report.current_kdj_double_friction),
        _performance_line("CANDIDATE_X2_FRICTION", report.candidate_double_friction),
        "CHANGED_VS_CURRENT days=%d by_year=%s"
        % (report.changed_vs_current_days, report.changed_vs_current_by_year),
        "CHANGED_VS_OFFICIAL days=%d by_year=%s"
        % (report.changed_vs_official_days, report.changed_vs_official_by_year),
        "GATE=%s" % ("PASS" if report.gate.passed else "REJECT"),
    ]
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.append("authority=local_screen_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def main() -> None:
    report = run_kdj_ranking_only_training_comparison()
    print(format_kdj_ranking_only_comparison(report))


if __name__ == "__main__":
    main()
