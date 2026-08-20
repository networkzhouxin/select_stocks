# -*- coding: utf-8 -*-
"""Fixed training-only evaluation for the causal 14:45 signal candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local.local_data_loader import TRAIN_END, TRAIN_START
from cross_signal_strategy.local.local_data_loader import (
    APPROVED_TRAINING_ROOT,
    APPROVED_WARMUP_ROOT,
    CrossSignalTrainingDataLoader,
)
from cross_signal_strategy.local.dual_timepoint_backtester import (
    DualTimepointBacktestEngine,
)
from cross_signal_strategy.local.dual_timepoint_order_planner import (
    DualTimepointOrderPlanner,
)
from cross_signal_strategy.local.dual_timepoint_signal_adapter import (
    DualTimepointSignalAdapter,
)
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.baseline_report import (
    BaselineReport,
    build_baseline_report,
)
from cross_signal_strategy.research.trade_diagnostics import (
    ClosedTradeDiagnostic,
    build_closed_trade_diagnostics,
)
from cross_signal_strategy.research.trade_quality_ledger import (
    TradeQualityRow,
    build_trade_quality_ledger,
)


TRAINING_YEARS = (2019, 2020, 2021)


@dataclass(frozen=True)
class DualTimepoint1445TrainingConfig:
    candidate_name: str
    decision_times: tuple[str, str]
    signal_cutoff: str
    training_start: str
    training_end: str
    training_root: Path
    warmup_root: Path
    initial_cash: float
    candidate_variants: int


def dual_timepoint_1445_training_config() -> DualTimepoint1445TrainingConfig:
    return DualTimepoint1445TrainingConfig(
        candidate_name="cross-v0.3.3-dual-timepoint-1445-candidate",
        decision_times=("09:35", "14:45"),
        signal_cutoff="14:44",
        training_start="2019-01-01",
        training_end="2021-12-31",
        training_root=Path(APPROVED_TRAINING_ROOT),
        warmup_root=Path(APPROVED_WARMUP_ROOT),
        initial_cash=20000.0,
        candidate_variants=1,
    )


@dataclass(frozen=True)
class DualTimepointGateInputs:
    total_return: float
    baseline_total_return: float
    max_drawdown: float
    baseline_max_drawdown: float
    profit_loss_ratio: float | None
    win_rate: float
    baseline_win_rate: float
    annual_win_rates: Mapping[int, float]
    baseline_annual_win_rates: Mapping[int, float]
    round_trip_count: int
    baseline_round_trip_count: int
    round_trip_improved_codes: Sequence[str]
    max_loss_streak: int
    baseline_max_loss_streak: int
    buy_count: int
    baseline_buy_count: int
    sell_count: int
    baseline_sell_count: int
    annual_coverage: Mapping[int, int]
    annual_missing: Mapping[int, int]
    double_friction_return: float
    baseline_double_friction_return: float
    double_friction_drawdown: float
    baseline_double_friction_drawdown: float


@dataclass(frozen=True)
class DualTimepointGateDecision:
    passed: bool
    reasons: Sequence[str]


@dataclass(frozen=True)
class DualTimepoint1445Report:
    config: DualTimepoint1445TrainingConfig
    baseline_report: BaselineReport
    candidate_report: BaselineReport
    baseline_double_friction_report: BaselineReport
    candidate_double_friction_report: BaselineReport
    gate_inputs: DualTimepointGateInputs
    gate: DualTimepointGateDecision
    baseline_trades: Sequence[ClosedTradeDiagnostic]
    candidate_trades: Sequence[ClosedTradeDiagnostic]
    baseline_ledger: Sequence[TradeQualityRow]
    candidate_ledger: Sequence[TradeQualityRow]
    baseline_order_signature: Sequence[tuple]
    candidate_order_signature: Sequence[tuple]
    rendered_sections: Sequence[str]


def evaluate_dual_timepoint_1445_gate(
    item: DualTimepointGateInputs,
) -> DualTimepointGateDecision:
    reasons = []
    if item.total_return + 1e-12 < 0.80 * item.baseline_total_return:
        reasons.append("total return retains less than 80% of baseline")
    if item.max_drawdown > item.baseline_max_drawdown + 1e-12:
        reasons.append("maximum drawdown worsens")
    if item.profit_loss_ratio is None or item.profit_loss_ratio < 3.0:
        reasons.append("profit/loss ratio is below 3.0")
    if item.win_rate <= item.baseline_win_rate:
        reasons.append("closed-trade win rate does not improve")
    annual_non_worse = sum(
        item.annual_win_rates.get(year, -1.0)
        >= item.baseline_annual_win_rates.get(year, 2.0)
        for year in TRAINING_YEARS
    )
    if annual_non_worse < 2:
        reasons.append("fewer than two annual win rates are non-worse")
    if item.round_trip_count > item.baseline_round_trip_count - 3:
        reasons.append("positive-to-negative round trips fall by fewer than three")
    if len(set(item.round_trip_improved_codes)) < 2:
        reasons.append(
            "round-trip improvement is concentrated in fewer than two ETFs"
        )
    if item.max_loss_streak > item.baseline_max_loss_streak:
        reasons.append("maximum losing streak worsens")
    if item.buy_count > 1.30 * item.baseline_buy_count:
        reasons.append("buy count rises by more than 30%")
    if item.sell_count > 1.30 * item.baseline_sell_count:
        reasons.append("sell count rises by more than 30%")
    if any(item.annual_coverage.get(year, 0) <= 0 for year in TRAINING_YEARS):
        reasons.append("one or more years have no usable 14:45 coverage")
    if set(item.annual_missing) != set(TRAINING_YEARS):
        reasons.append("missing coverage counts are not disclosed for every year")
    if item.double_friction_return + 1e-12 < (
        0.80 * item.baseline_double_friction_return
    ):
        reasons.append("doubled-friction return retains less than 80% of baseline")
    if item.double_friction_drawdown > (
        item.baseline_double_friction_drawdown + 1e-12
    ):
        reasons.append("doubled-friction drawdown worsens")
    return DualTimepointGateDecision(not reasons, tuple(reasons))


def build_dual_timepoint_1445_report(
    *,
    baseline_days: Iterable[object],
    candidate_days: Iterable[object],
    baseline_entry_score_snapshots: Mapping[tuple, Mapping[str, object]],
    baseline_exit_score_snapshots: Mapping[tuple, Mapping[str, object]],
    candidate_entry_score_snapshots: Mapping[tuple, Mapping[str, object]],
    candidate_exit_score_snapshots: Mapping[tuple, Mapping[str, object]],
    candidate_score_coverage: Mapping[tuple, str],
    baseline_double_friction_days: Iterable[object],
    candidate_double_friction_days: Iterable[object],
    loader,
    initial_cash: float = 20000.0,
    config: DualTimepoint1445TrainingConfig | None = None,
) -> DualTimepoint1445Report:
    """Build the frozen A/B report without feeding diagnostics into execution."""

    baseline = list(baseline_days)
    candidate = list(candidate_days)
    baseline_stress = list(baseline_double_friction_days)
    candidate_stress = list(candidate_double_friction_days)
    _assert_aligned_training_dates(
        baseline, candidate, baseline_stress, candidate_stress
    )

    baseline_report = build_baseline_report(baseline, initial_cash)
    candidate_report = build_baseline_report(candidate, initial_cash)
    baseline_stress_report = build_baseline_report(baseline_stress, initial_cash)
    candidate_stress_report = build_baseline_report(candidate_stress, initial_cash)

    baseline_trades = build_closed_trade_diagnostics(
        baseline,
        _snapshots_for_filled_orders(
            baseline, baseline_entry_score_snapshots, side="buy"
        ),
        _snapshots_for_filled_orders(
            baseline, baseline_exit_score_snapshots, side="sell"
        ),
    )
    candidate_trades = build_closed_trade_diagnostics(
        candidate,
        _snapshots_for_filled_orders(
            candidate, candidate_entry_score_snapshots, side="buy"
        ),
        _snapshots_for_filled_orders(
            candidate, candidate_exit_score_snapshots, side="sell"
        ),
    )
    baseline_ledger = build_trade_quality_ledger(baseline_trades, loader)
    candidate_ledger = build_trade_quality_ledger(candidate_trades, loader)

    baseline_round_trips = _round_trip_counts(baseline_ledger)
    candidate_round_trips = _round_trip_counts(candidate_ledger)
    improved_codes = tuple(
        sorted(
            code
            for code, count in baseline_round_trips.items()
            if count > candidate_round_trips.get(code, 0)
        )
    )
    annual_coverage, annual_missing = _coverage_by_year(candidate_score_coverage)
    gate_inputs = DualTimepointGateInputs(
        total_return=candidate_report.total_return,
        baseline_total_return=baseline_report.total_return,
        max_drawdown=candidate_report.max_drawdown,
        baseline_max_drawdown=baseline_report.max_drawdown,
        profit_loss_ratio=candidate_report.profit_loss_ratio,
        win_rate=candidate_report.win_rate,
        baseline_win_rate=baseline_report.win_rate,
        annual_win_rates=_annual_win_rates(candidate_trades),
        baseline_annual_win_rates=_annual_win_rates(baseline_trades),
        round_trip_count=sum(candidate_round_trips.values()),
        baseline_round_trip_count=sum(baseline_round_trips.values()),
        round_trip_improved_codes=improved_codes,
        max_loss_streak=_max_loss_streak(candidate_trades),
        baseline_max_loss_streak=_max_loss_streak(baseline_trades),
        buy_count=candidate_report.buy_count,
        baseline_buy_count=baseline_report.buy_count,
        sell_count=candidate_report.sell_count,
        baseline_sell_count=baseline_report.sell_count,
        annual_coverage=annual_coverage,
        annual_missing=annual_missing,
        double_friction_return=candidate_stress_report.total_return,
        baseline_double_friction_return=baseline_stress_report.total_return,
        double_friction_drawdown=candidate_stress_report.max_drawdown,
        baseline_double_friction_drawdown=baseline_stress_report.max_drawdown,
    )
    return DualTimepoint1445Report(
        config=config or dual_timepoint_1445_training_config(),
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_double_friction_report=baseline_stress_report,
        candidate_double_friction_report=candidate_stress_report,
        gate_inputs=gate_inputs,
        gate=evaluate_dual_timepoint_1445_gate(gate_inputs),
        baseline_trades=tuple(baseline_trades),
        candidate_trades=tuple(candidate_trades),
        baseline_ledger=tuple(baseline_ledger),
        candidate_ledger=tuple(candidate_ledger),
        baseline_order_signature=_filled_order_signature(baseline),
        candidate_order_signature=_filled_order_signature(candidate),
        rendered_sections=(
            "nominal",
            "annual win rates",
            "trade quality",
            "coverage",
            "double friction",
            "gate",
        ),
    )


def render_dual_timepoint_1445_report(report: DualTimepoint1445Report) -> str:
    """Render only frozen report fields; never query data or recompute metrics."""

    item = report.gate_inputs
    years = "\n".join(
        "- %d: baseline %.6f, candidate %.6f, coverage %d, missing %d"
        % (
            year,
            item.baseline_annual_win_rates.get(year, 0.0),
            item.annual_win_rates.get(year, 0.0),
            item.annual_coverage.get(year, 0),
            item.annual_missing.get(year, 0),
        )
        for year in TRAINING_YEARS
    )
    reasons = (
        "\n".join("- %s" % reason for reason in report.gate.reasons)
        if report.gate.reasons
        else "- none"
    )
    decision = "ELIGIBLE_FOR_JOINQUANT_PLAN" if report.gate.passed else "STOP"
    return """# Cross-signal fixed 14:45 training candidate

Candidate: {candidate}
Data scope: {start} through {end}; 2018 warm-up only
Decision times: {times}; signal cutoff: {cutoff}

## Nominal

- total return: baseline {base_return:.10f}, candidate {candidate_return:.10f}
- maximum drawdown: baseline {base_dd:.10f}, candidate {candidate_dd:.10f}
- win rate: baseline {base_win:.10f}, candidate {candidate_win:.10f}
- profit/loss ratio: candidate {pl_ratio}
- orders: baseline buy/sell {base_buy}/{base_sell}, candidate {candidate_buy}/{candidate_sell}

## Annual win rates and coverage

{years}

## Trade quality

- round trip: baseline {base_round_trip}, candidate {candidate_round_trip}, improved codes {codes}
- maximum loss streak: baseline {base_streak}, candidate {candidate_streak}

## Double friction

- total return: baseline {base_stress_return:.10f}, candidate {candidate_stress_return:.10f}
- maximum drawdown: baseline {base_stress_dd:.10f}, candidate {candidate_stress_dd:.10f}

## Gate reasons

{reasons}

## Decision

{decision}
""".format(
        candidate=report.config.candidate_name,
        start=report.config.training_start,
        end=report.config.training_end,
        times=", ".join(report.config.decision_times),
        cutoff=report.config.signal_cutoff,
        base_return=item.baseline_total_return,
        candidate_return=item.total_return,
        base_dd=item.baseline_max_drawdown,
        candidate_dd=item.max_drawdown,
        base_win=item.baseline_win_rate,
        candidate_win=item.win_rate,
        pl_ratio=(
            "none"
            if item.profit_loss_ratio is None
            else "%.10f" % item.profit_loss_ratio
        ),
        base_buy=item.baseline_buy_count,
        base_sell=item.baseline_sell_count,
        candidate_buy=item.buy_count,
        candidate_sell=item.sell_count,
        years=years,
        base_round_trip=item.baseline_round_trip_count,
        candidate_round_trip=item.round_trip_count,
        codes=",".join(item.round_trip_improved_codes) or "none",
        base_streak=item.baseline_max_loss_streak,
        candidate_streak=item.max_loss_streak,
        base_stress_return=item.baseline_double_friction_return,
        candidate_stress_return=item.double_friction_return,
        base_stress_dd=item.baseline_double_friction_drawdown,
        candidate_stress_dd=item.double_friction_drawdown,
        reasons=reasons,
        decision=decision,
    )


def run_training_dual_timepoint_1445_candidate(
    config: DualTimepoint1445TrainingConfig,
) -> DualTimepoint1445Report:
    """Consume the single pre-registered local training candidate."""

    _assert_exact_training_config(config)
    loader = CrossSignalTrainingDataLoader(config.training_root)
    trade_dates = get_training_trade_dates(loader)
    if not trade_dates:
        raise ValueError("Training replay has no trade dates")
    if (
        min(trade_dates) < config.training_start
        or max(trade_dates) > config.training_end
    ):
        raise ValueError("Training replay dates exceed the frozen bounds")

    baseline_adapter = build_training_signal_adapter(
        loader, warmup_root=config.warmup_root
    )
    shared_scores = DualTimepointSignalAdapter(baseline_adapter)
    baseline_days, baseline_planner = _run_one_replay(
        loader,
        shared_scores,
        trade_dates,
        config.initial_cash,
        decision_times=("09:35",),
    )
    candidate_days, candidate_planner = _run_one_replay(
        loader,
        shared_scores,
        trade_dates,
        config.initial_cash,
        decision_times=config.decision_times,
    )
    doubled_friction = {"commission_rate": 0.0006, "slippage_rate": 0.002}
    baseline_stress_days, _ = _run_one_replay(
        loader,
        shared_scores,
        trade_dates,
        config.initial_cash,
        decision_times=("09:35",),
        broker_kwargs=doubled_friction,
    )
    candidate_stress_days, _ = _run_one_replay(
        loader,
        shared_scores,
        trade_dates,
        config.initial_cash,
        decision_times=config.decision_times,
        broker_kwargs=doubled_friction,
    )
    return build_dual_timepoint_1445_report(
        baseline_days=baseline_days,
        candidate_days=candidate_days,
        baseline_entry_score_snapshots=baseline_planner.entry_score_snapshots,
        baseline_exit_score_snapshots=baseline_planner.exit_score_snapshots,
        candidate_entry_score_snapshots=candidate_planner.entry_score_snapshots,
        candidate_exit_score_snapshots=candidate_planner.exit_score_snapshots,
        candidate_score_coverage=candidate_planner.score_coverage,
        baseline_double_friction_days=baseline_stress_days,
        candidate_double_friction_days=candidate_stress_days,
        loader=loader,
        initial_cash=config.initial_cash,
        config=config,
    )


def _assert_exact_training_config(config: DualTimepoint1445TrainingConfig) -> None:
    expected = dual_timepoint_1445_training_config()
    if type(config) is not DualTimepoint1445TrainingConfig or config != expected:
        raise ValueError("Use the exact frozen 14:45 training config")
    if Path(config.training_root).resolve() != Path(APPROVED_TRAINING_ROOT).resolve():
        raise ValueError("Use the exact frozen 14:45 training config")
    if Path(config.warmup_root).resolve() != Path(APPROVED_WARMUP_ROOT).resolve():
        raise ValueError("Use the exact frozen 14:45 training config")
    if pd.Timestamp(config.training_start) != TRAIN_START:
        raise ValueError("Use the exact frozen 14:45 training config")
    if pd.Timestamp(config.training_end) != TRAIN_END:
        raise ValueError("Use the exact frozen 14:45 training config")


def _run_one_replay(
    loader,
    signal_adapter,
    trade_dates,
    initial_cash,
    *,
    decision_times,
    broker_kwargs=None,
):
    planner = DualTimepointOrderPlanner(
        signal_adapter, trade_dates=list(trade_dates)
    )
    engine = DualTimepointBacktestEngine(
        loader,
        initial_cash,
        decision_times=decision_times,
        broker_kwargs=broker_kwargs,
    )
    return engine.run(trade_dates, planner), planner


def main() -> int:
    config = dual_timepoint_1445_training_config()
    report = run_training_dual_timepoint_1445_candidate(config)
    text = render_dual_timepoint_1445_report(report)
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "dual_timepoint_1445_2019_2021.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.gate.passed else 1


def _assert_aligned_training_dates(*batches: Sequence[object]) -> None:
    if not batches or not batches[0]:
        raise ValueError("A/B report requires non-empty training replays")
    expected = tuple(str(day.date) for day in batches[0])
    for batch in batches[1:]:
        if tuple(str(day.date) for day in batch) != expected:
            raise ValueError("All A/B and friction replays must use identical dates")
    for value in expected:
        date = pd.Timestamp(value)
        if date < TRAIN_START or date > TRAIN_END:
            raise ValueError("A/B report dates must stay inside 2019-2021 training")


def _snapshots_for_filled_orders(
    days: Sequence[object],
    snapshots: Mapping[tuple, Mapping[str, object]],
    *,
    side: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
    normalized = {}
    for day in days:
        date = str(day.date)
        for order in getattr(day, "orders", ()):
            amount = int(getattr(order, "amount_delta", 0))
            matches = amount > 0 if side == "buy" else amount < 0
            if not getattr(order, "filled", False) or not matches:
                continue
            code = str(order.code).split(".")[0]
            decision_time = str(order.side_time)[-5:]
            score = snapshots.get((date, decision_time, code))
            if score is not None:
                normalized[(date, code)] = dict(score)
    return normalized


def _annual_win_rates(
    trades: Sequence[ClosedTradeDiagnostic],
) -> dict[int, float]:
    grouped = {year: [] for year in TRAINING_YEARS}
    for trade in trades:
        year = pd.Timestamp(trade.sell_date).year
        if year in grouped:
            grouped[year].append(float(trade.pnl))
    return {
        year: (sum(value > 0 for value in values) / len(values) if values else 0.0)
        for year, values in grouped.items()
    }


def _max_loss_streak(trades: Sequence[ClosedTradeDiagnostic]) -> int:
    maximum = current = 0
    for trade in trades:
        if float(trade.pnl) < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _round_trip_counts(ledger: Sequence[TradeQualityRow]) -> dict[str, int]:
    counts = {}
    for row in ledger:
        if float(row.holding_mfe) > 0 and float(row.realized_return_pct) < 0:
            counts[row.code] = counts.get(row.code, 0) + 1
    return counts


def _coverage_by_year(coverage: Mapping[tuple, str]) -> tuple[dict, dict]:
    usable = {year: 0 for year in TRAINING_YEARS}
    missing = {year: 0 for year in TRAINING_YEARS}
    for key, status in coverage.items():
        date, decision_time, _code = key
        if str(decision_time) != "14:45":
            continue
        year = pd.Timestamp(date).year
        if year not in usable:
            raise ValueError("Coverage dates must stay inside 2019-2021 training")
        if str(status) == "ok":
            usable[year] += 1
        else:
            missing[year] += 1
    return usable, missing


def _filled_order_signature(days: Sequence[object]) -> tuple[tuple, ...]:
    signature = []
    for day in days:
        for order in getattr(day, "orders", ()):
            if not getattr(order, "filled", False):
                continue
            amount = int(order.amount_delta)
            signature.append(
                (
                    str(day.date),
                    str(order.side_time)[-5:],
                    "buy" if amount > 0 else "sell",
                    str(order.code).split(".")[0],
                    abs(amount),
                    str(getattr(order, "reason", "")),
                )
            )
    return tuple(signature)


if __name__ == "__main__":
    raise SystemExit(main())
