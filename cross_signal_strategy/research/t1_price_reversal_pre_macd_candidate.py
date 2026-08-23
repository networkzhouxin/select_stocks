# -*- coding: utf-8 -*-
"""Frozen training-only T-1 price-reversal entry before MACD confirmation.

The official score, threshold, ranking, sizing, sell path, and risk controls are
unchanged.  This isolated alternative can only fill a slot left vacant by the
official score-at-least-60 queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

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
    _performance,
    _performance_line,
    _retains_ratio,
    _run_replay,
)


ALTERNATIVE_BUY_REASON = "t1_price_reversal_pre_macd_buy"


@dataclass(frozen=True)
class T1PriceReversalSignalAdapter:
    """Add two-bar causal price context without changing official scores."""

    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        raw_score, reason = self.source.score(code, current_date, return_reason=True)
        if raw_score is None:
            return (None, reason) if return_reason else None
        frame, signal_date = self.source.load_signal_frame(code, current_date)
        if signal_date is None:
            reason = "no_previous_trade_date"
            return (None, reason) if return_reason else None
        if raw_score.get("signal_date") not in (None, str(signal_date)):
            raise ValueError("score signal_date does not match causal frame")
        score = dict(raw_score)
        score["signal_date"] = str(signal_date)
        enriched = enrich_t1_price_reversal_context(score, frame)
        return (enriched, None) if return_reason else enriched


def enrich_t1_price_reversal_context(
    raw_score: Mapping[str, object],
    signal_frame: pd.DataFrame,
) -> dict:
    """Attach the exact T-1/T-2 higher-low and prior-high-break evidence."""

    score = dict(raw_score)
    signal_date = pd.Timestamp(str(score.get("signal_date", "")))
    required = {"date", "low", "high", "close"}
    missing = sorted(required.difference(signal_frame.columns))
    if missing:
        raise ValueError("signal frame missing columns: %s" % ", ".join(missing))

    frame = signal_frame.copy()
    frame["_signal_date"] = pd.to_datetime(frame["date"], errors="raise")
    if (frame["_signal_date"] > signal_date).any():
        raise ValueError("signal frame contains a row later than signal_date")
    frame = frame.sort_values("_signal_date", kind="stable")
    if frame.empty or frame["_signal_date"].iloc[-1] != signal_date:
        raise ValueError("signal frame must end exactly on signal_date")
    if len(frame) < 2:
        raise ValueError("signal frame requires at least two completed rows")

    t2 = frame.iloc[-2]
    t1 = frame.iloc[-1]
    t1_low = float(t1["low"])
    t2_low = float(t2["low"])
    t1_close = float(t1["close"])
    t2_high = float(t2["high"])
    low_not_lower = t1_low >= t2_low
    close_above_high = t1_close > t2_high
    score.update(
        {
            "max_data_date": signal_date.strftime("%Y-%m-%d"),
            "t2_date": pd.Timestamp(t2["date"]).strftime("%Y-%m-%d"),
            "t1_date": pd.Timestamp(t1["date"]).strftime("%Y-%m-%d"),
            "t2_low": t2_low,
            "t2_high": t2_high,
            "t1_low": t1_low,
            "t1_close": t1_close,
            "t1_price_reversal_context_complete": True,
            "t1_low_not_lower_than_t2": low_not_lower,
            "t1_close_above_t2_high": close_above_high,
            "t1_price_reversal_confirmed": low_not_lower and close_above_high,
        }
    )
    return score


def filter_t1_price_reversal_pre_macd_candidates(
    scores: Iterable[Mapping[str, object]],
    held_codes: Iterable[str],
    params: Mapping[str, object] | None = None,
) -> list[dict]:
    """Return the single pre-registered alternative entry family."""

    p = dict(params or strategy.get_default_params())
    held = {str(code).split(".")[0] for code in held_codes}
    candidates = []
    for raw_score in scores:
        score = dict(raw_score)
        code = str(score.get("code", "")).split(".")[0]
        score["code"] = code
        rsi_up = strategy.rsi_group_direction(score) == "up"
        kdj_up = bool(score.get("kdj_k_cross_up") or score.get("kdj_j_cross_up"))
        if not code or code in held:
            continue
        if float(score.get("buy_score", 0.0) or 0.0) >= float(p["buy_threshold"]):
            continue
        if not score.get("buy_allowed"):
            continue
        if float(score.get("sell_score", 0.0) or 0.0) >= float(p["sell_threshold"]):
            continue
        if not strategy.has_new_buy_position(score, p):
            continue
        if strategy.is_blocked_entry_combo(score):
            continue
        if bool(score.get("downside_continuation")):
            continue
        if not rsi_up or not kdj_up or bool(score.get("macd_cross_up")):
            continue
        if score.get("t1_price_reversal_context_complete") is not True:
            continue
        if not bool(score.get("t1_low_not_lower_than_t2")):
            continue
        if not bool(score.get("t1_close_above_t2_high")):
            continue
        if score.get("t1_price_reversal_confirmed") is not True:
            continue
        candidates.append(score)
    return strategy.sort_candidates(candidates)


@dataclass
class T1PriceReversalPreMacdOrderPlanner(LocalCrossSignalOrderPlanner):
    """Run official orders first, then fill only genuinely vacant slots."""

    entry_score_snapshots: Dict[tuple[str, str], dict] = field(default_factory=dict)

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
            for item in filter_t1_price_reversal_pre_macd_candidates(
                self.last_scores.values(), occupied, self.params
            )
            if item["code"] not in force_stopped
            and not self._is_in_atr_stop_cooldown(item["code"], current_date)
        ]
        total_value = self._total_value(broker, current_prices or {})
        for score in candidates[:slots]:
            entry_snapshot = dict(score)
            entry_snapshot["entry_channel"] = "t1_price_reversal_pre_macd"
            self.entry_score_snapshots[(str(current_date), score["code"])] = (
                entry_snapshot
            )
            orders.append(
                {
                    "code": score["code"],
                    "target_value": self._scaled_buy_target_value(
                        total_value, score, current_date
                    ),
                    "reason": ALTERNATIVE_BUY_REASON,
                }
            )
        return orders


@dataclass(frozen=True)
class T1PriceReversalComparisonReport:
    official: ExtremeZonePerformance
    candidate: ExtremeZonePerformance
    official_double_friction: ExtremeZonePerformance
    candidate_double_friction: ExtremeZonePerformance
    direct_fill_count: int
    direct_fill_years: tuple[int, ...]
    direct_fills: tuple[dict, ...]
    direct_closed_trades: tuple[dict, ...]
    gate: ExtremeZoneGateDecision


def evaluate_t1_price_reversal_gate(
    official: ExtremeZonePerformance,
    candidate: ExtremeZonePerformance,
    official_double_friction: ExtremeZonePerformance,
    candidate_double_friction: ExtremeZonePerformance,
    direct_fill_count: int,
    direct_fill_years: Sequence[int],
) -> ExtremeZoneGateDecision:
    """Apply the exact accuracy-first gate frozen before the replay."""

    reasons = []
    if int(direct_fill_count) < 3:
        reasons.append("fewer than 3 direct alternative fills")
    if len(set(int(year) for year in direct_fill_years)) < 2:
        reasons.append("direct fills occur in fewer than 2 training years")
    if candidate.win_rate <= official.win_rate:
        reasons.append("candidate win rate does not improve official")
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
        official_return = official.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if official_return is None or candidate_return is None:
            reasons.append("%d annual return is missing" % year)
        elif official_return > 0 and candidate_return <= 0:
            reasons.append("%d candidate annual return turns non-positive" % year)
    if (
        candidate_double_friction.total_return
        < official_double_friction.total_return * 0.95
    ):
        reasons.append("double-friction return retains less than 95% of official")
    if candidate_double_friction.win_rate < official_double_friction.win_rate:
        reasons.append("double-friction win rate worsens official")
    return ExtremeZoneGateDecision(not reasons, tuple(reasons))


def run_t1_price_reversal_training_comparison(
    loader: object | None = None,
    initial_cash: float = 20000.0,
    warmup_root: Path | str = APPROVED_WARMUP_ROOT,
) -> T1PriceReversalComparisonReport:
    """Run nominal and doubled-friction A/B on approved training data only."""

    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )
    from cross_signal_strategy.research.baseline_report import build_baseline_report
    from cross_signal_strategy.research.friction_diagnostics import (
        PrecomputedSignalAdapter,
    )
    from cross_signal_strategy.research.trade_diagnostics import (
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
    raw_source = build_training_signal_adapter(loader, warmup_root=warmup)
    enriched_source = T1PriceReversalSignalAdapter(raw_source)
    cached = PrecomputedSignalAdapter.from_source(
        enriched_source,
        trade_dates=trade_dates,
        codes=pool,
    )

    def replay(planner_type, broker_kwargs=None):
        planner = planner_type(
            cached,
            etf_pool=pool,
            params=dict(params),
            trade_dates=trade_dates,
        )
        days = _run_replay(
            loader,
            planner,
            trade_dates,
            initial_cash,
            broker_kwargs=broker_kwargs,
        )
        return days, planner

    official_days, _ = replay(LocalCrossSignalOrderPlanner)
    candidate_days, candidate_planner = replay(T1PriceReversalPreMacdOrderPlanner)
    official_stress_days, _ = replay(LocalCrossSignalOrderPlanner, DOUBLE_FRICTION)
    candidate_stress_days, _ = replay(
        T1PriceReversalPreMacdOrderPlanner, DOUBLE_FRICTION
    )

    def performance(days):
        return _performance(build_baseline_report(days, initial_cash), days, initial_cash)

    direct_fills = []
    for day in candidate_days:
        for order in day.orders:
            if not getattr(order, "filled", False):
                continue
            if getattr(order, "reason", "") != ALTERNATIVE_BUY_REASON:
                continue
            direct_fills.append(
                {
                    "date": str(day.date),
                    "code": str(order.code).split(".")[0],
                    "exec_price": float(order.exec_price),
                    "amount": int(order.amount_delta),
                }
            )

    official = performance(official_days)
    candidate = performance(candidate_days)
    official_stress = performance(official_stress_days)
    candidate_stress = performance(candidate_stress_days)
    closed = build_closed_trade_diagnostics(
        candidate_days,
        candidate_planner.entry_score_snapshots,
    )
    direct_closed = tuple(
        {
            "code": trade.code,
            "buy_date": trade.buy_date,
            "sell_date": trade.sell_date,
            "sell_reason": trade.sell_reason,
            "return_pct": float(trade.return_pct),
            "pnl": float(trade.pnl),
            "buy_score": float(trade.entry_score.get("buy_score", 0.0) or 0.0),
            "trend_score": float(trade.entry_score.get("trend_score", 0.0) or 0.0),
        }
        for trade in closed
        if trade.entry_score.get("entry_channel") == "t1_price_reversal_pre_macd"
    )
    years = tuple(sorted({pd.Timestamp(item["date"]).year for item in direct_fills}))
    gate = evaluate_t1_price_reversal_gate(
        official,
        candidate,
        official_stress,
        candidate_stress,
        len(direct_fills),
        years,
    )
    return T1PriceReversalComparisonReport(
        official=official,
        candidate=candidate,
        official_double_friction=official_stress,
        candidate_double_friction=candidate_stress,
        direct_fill_count=len(direct_fills),
        direct_fill_years=years,
        direct_fills=tuple(direct_fills),
        direct_closed_trades=direct_closed,
        gate=gate,
    )


def format_t1_price_reversal_comparison(
    report: T1PriceReversalComparisonReport,
) -> str:
    lines = [
        "T-1 price-reversal pre-MACD entry (2019-2021; local screen only)",
        (
            "rule=RSI up + KDJ up + no MACD up + T1 low>=T2 low + "
            "T1 close>T2 high; official primary queue first"
        ),
        _performance_line("OFFICIAL", report.official),
        _performance_line("CANDIDATE", report.candidate),
        _performance_line("OFFICIAL_X2_FRICTION", report.official_double_friction),
        _performance_line("CANDIDATE_X2_FRICTION", report.candidate_double_friction),
        "DIRECT_FILLS count=%d years=%s"
        % (report.direct_fill_count, report.direct_fill_years),
    ]
    for item in report.direct_fills:
        lines.append(
            "DIRECT_FILL date={date} code={code} price={exec_price:.4f} amount={amount}".format(
                **item
            )
        )
    direct_wins = sum(1 for item in report.direct_closed_trades if item["pnl"] > 0)
    direct_losses = sum(1 for item in report.direct_closed_trades if item["pnl"] < 0)
    direct_pnl = sum(item["pnl"] for item in report.direct_closed_trades)
    lines.append(
        "DIRECT_CLOSED count=%d wins=%d losses=%d pnl=%.2f"
        % (len(report.direct_closed_trades), direct_wins, direct_losses, direct_pnl)
    )
    for item in report.direct_closed_trades:
        lines.append(
            "DIRECT_TRADE code={code} buy={buy_date} sell={sell_date} "
            "reason={sell_reason} return_pct={return_pct:.4f} pnl={pnl:.2f} "
            "buy_score={buy_score:.1f} trend_score={trend_score:.1f}".format(**item)
        )
    lines.append("GATE=%s" % ("PASS" if report.gate.passed else "REJECT"))
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    lines.append("authority=local_screen_only; JoinQuant remains authoritative")
    return "\n".join(lines)


def main() -> None:
    print(format_t1_price_reversal_comparison(
        run_t1_price_reversal_training_comparison()
    ))


if __name__ == "__main__":
    main()
