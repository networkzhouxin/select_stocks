# -*- coding: utf-8 -*-
"""Isolated current-state KDJ candidate with rebalanced sell evidence.

The candidate preserves every causal T-1 snapshot field and the complete buy
path. It only redistributes sell weights across capped indicator families and
one non-cumulative price-weakness bucket. Formal platform strategies are not
imported for mutation and remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

from cross_signal_strategy.local.local_data_loader import (
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.local_order_planner import strategy
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
class SellScoreRebalanceAdapter(KdjTieredModeratePointsScoreAdapter):
    """Keep KDJ tiers and replace only the base sell-score composition."""

    def _adjust(
        self,
        code: str,
        current_date: str,
        base_score: Mapping[str, object],
    ) -> dict:
        result = super()._adjust(code, current_date, base_score)
        pre_rebalance = float(result.get("sell_score", 0.0) or 0.0)

        rsi_count = 0
        if strategy.rsi_group_direction(result) == "down":
            rsi_count = sum(bool(result.get(field)) for field in (
                "rsi6_cross_rsi12_down",
                "rsi6_cross_rsi24_down",
            ))
        rsi_score = 20.0 if rsi_count >= 2 else 12.0 if rsi_count == 1 else 0.0
        macd_score = 6.0 if bool(result.get("macd_cross_down")) else 0.0
        kdj_count = sum(bool(result.get(field)) for field in (
            "kdj_k_cross_down",
            "kdj_j_cross_down",
        ))
        kdj_score = 10.0 if kdj_count >= 2 else 5.0 if kdj_count == 1 else 0.0
        reversal = rsi_score + macd_score + kdj_score

        price_buckets = [0.0]
        if bool(result.get("close_below_boll_mid")) or bool(
            result.get("fell_back_inside_boll")
        ):
            price_buckets.append(6.0)
        if bool(result.get("far_above_ma20_and_rsi6_down")):
            price_buckets.append(8.0)
        if bool(result.get("close_below_ma20")):
            price_buckets.append(10.0)
        if bool(result.get("close_below_falling_ma10")) or bool(
            result.get("downside_continuation")
        ):
            price_buckets.append(12.0)
        risk = max(price_buckets)
        extreme = float(result.get("sell_extreme_zone_score", 0.0) or 0.0)
        result.update({
            "pre_rebalance_sell_score": pre_rebalance,
            "sell_reversal_score": reversal,
            "sell_risk_score": risk,
            "sell_score": max(0.0, reversal + risk + extreme),
            "sell_score_rebalance_version": "rsi20-macd6-kdj10-price12max",
        })
        return result


@dataclass(frozen=True)
class SellScoreRebalanceReport:
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


@dataclass(frozen=True)
class TargetTradePair:
    code: str
    buy_date: str
    current: object
    candidate: object
    return_delta_pct: float


TARGET_ENTRIES = (
    ("512100", "2019-09-30"),
    ("513880", "2021-03-04"),
)


def select_target_trade_pairs(
    current_trades: Sequence[object],
    candidate_trades: Sequence[object],
    targets: Sequence[tuple[str, str]] = TARGET_ENTRIES,
) -> tuple[TargetTradePair, ...]:
    """Pair predeclared entries without assuming that exit dates stay equal."""

    def index(trades):
        result = {}
        for trade in trades:
            key = (
                str(getattr(trade, "code")).split(".")[0],
                str(getattr(trade, "buy_date")),
            )
            if key in result:
                raise ValueError("duplicate closed trade entry: %s" % (key,))
            result[key] = trade
        return result

    current_index = index(current_trades)
    candidate_index = index(candidate_trades)
    pairs = []
    for raw_code, raw_date in targets:
        key = (str(raw_code).split(".")[0], str(raw_date))
        if key not in current_index or key not in candidate_index:
            raise ValueError("missing target entry in current/candidate path: %s" % (key,))
        current = current_index[key]
        candidate = candidate_index[key]
        pairs.append(TargetTradePair(
            code=key[0],
            buy_date=key[1],
            current=current,
            candidate=candidate,
            return_delta_pct=(
                float(getattr(candidate, "return_pct"))
                - float(getattr(current, "return_pct"))
            ),
        ))
    return tuple(pairs)


def evaluate_sell_score_rebalance_gate(
    official: ExtremeZonePerformance,
    current_kdj: ExtremeZonePerformance,
    candidate: ExtremeZonePerformance,
    official_double_friction: ExtremeZonePerformance,
    candidate_double_friction: ExtremeZonePerformance,
    changed_vs_current_by_year: Mapping[int, int],
) -> ExtremeZoneGateDecision:
    """Evaluate the accuracy-first gate frozen before the training result."""

    reasons = []
    for year in (2019, 2020, 2021):
        if int(changed_vs_current_by_year.get(year, 0)) <= 0:
            reasons.append("%d has no changed filled-order day" % year)
    if candidate.win_rate <= current_kdj.win_rate:
        reasons.append("candidate win rate does not improve current KDJ path")
    if candidate.win_rate <= official.win_rate:
        reasons.append("candidate win rate does not improve official path")
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
        elif annual <= 0:
            reasons.append("%d candidate annual return is non-positive" % year)
    if (
        candidate_double_friction.total_return
        < official_double_friction.total_return * 0.95
    ):
        reasons.append("double-friction return retains less than 95% of official")
    if candidate_double_friction.win_rate < official_double_friction.win_rate:
        reasons.append("double-friction win rate worsens")
    return ExtremeZoneGateDecision(not reasons, tuple(reasons))


def run_sell_score_rebalance_training_comparison(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> SellScoreRebalanceReport:
    """Run official, current-KDJ, and rebalanced paths on training data only."""

    from cross_signal_strategy.local.local_order_planner import (
        LocalCrossSignalOrderPlanner,
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
    source = build_training_signal_adapter(loader, warmup_root=warmup)
    official_source = PrecomputedSignalAdapter.from_source(
        source,
        trade_dates=trade_dates,
        codes=pool,
    )
    current_source = KdjTieredModeratePointsScoreAdapter(official_source, trade_dates)
    candidate_source = SellScoreRebalanceAdapter(official_source, trade_dates)

    def replay(score_source, broker_kwargs=None):
        planner = LocalCrossSignalOrderPlanner(
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

    official_days = replay(official_source)
    current_days = replay(current_source)
    candidate_days = replay(candidate_source)
    official_stress_days = replay(official_source, DOUBLE_FRICTION)
    current_stress_days = replay(current_source, DOUBLE_FRICTION)
    candidate_stress_days = replay(candidate_source, DOUBLE_FRICTION)

    def performance(days):
        return _performance(build_baseline_report(days, initial_cash), days, initial_cash)

    official = performance(official_days)
    current = performance(current_days)
    candidate = performance(candidate_days)
    official_stress = performance(official_stress_days)
    current_stress = performance(current_stress_days)
    candidate_stress = performance(candidate_stress_days)
    changed_current = _changed_days_by_year(current_days, candidate_days)
    changed_official = _changed_days_by_year(official_days, candidate_days)
    gate = evaluate_sell_score_rebalance_gate(
        official,
        current,
        candidate,
        official_stress,
        candidate_stress,
        changed_current,
    )
    return SellScoreRebalanceReport(
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


def run_sell_score_rebalance_target_attribution(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> tuple[TargetTradePair, ...]:
    """Replay only current/rebalanced paths and pair two predeclared entries."""

    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )
    from cross_signal_strategy.research.trade_diagnostics import (
        DiagnosticOrderPlanner,
        build_closed_trade_diagnostics,
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
    candidate_source = SellScoreRebalanceAdapter(official_source, trade_dates)

    def closed_trades(score_source):
        planner = DiagnosticOrderPlanner(
            score_source,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        )
        days = _run_replay(loader, planner, trade_dates, initial_cash)
        return build_closed_trade_diagnostics(
            days,
            planner.entry_score_snapshots,
            planner.exit_score_snapshots,
        )

    return select_target_trade_pairs(
        closed_trades(current_source),
        closed_trades(candidate_source),
    )


def format_sell_score_rebalance_target_attribution(
    pairs: Sequence[TargetTradePair],
) -> str:
    lines = ["Sell-score rebalance target-entry attribution (training only)"]
    for pair in pairs:
        lines.append(
            "TARGET code=%s buy=%s current_sell=%s current_reason=%s "
            "current_return=%.4f candidate_sell=%s candidate_reason=%s "
            "candidate_return=%.4f delta_pp=%.4f candidate_exit_score=%.1f"
            % (
                pair.code,
                pair.buy_date,
                pair.current.sell_date,
                pair.current.sell_reason,
                pair.current.return_pct,
                pair.candidate.sell_date,
                pair.candidate.sell_reason,
                pair.candidate.return_pct,
                pair.return_delta_pct,
                float(pair.candidate.exit_score.get("sell_score", 0.0) or 0.0),
            )
        )
    lines.append("authority=local_attribution_only; no rule selection from two trades")
    return "\n".join(lines)


def format_sell_score_rebalance_comparison(report: SellScoreRebalanceReport) -> str:
    lines = [
        "Sell-score rebalance candidate (2019-2021; local screen only)",
        (
            "sell=RSI one/two 12/20; MACD 6; KDJ one/two 5/10; "
            "price weakness max 6/8/10/12; current KDJ sell tier 5/10"
        ),
        "threshold=30; price confirmation, ADX, min hold, ATR and buy path unchanged",
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
    report = run_sell_score_rebalance_training_comparison()
    print(format_sell_score_rebalance_comparison(report))


if __name__ == "__main__":
    main()
