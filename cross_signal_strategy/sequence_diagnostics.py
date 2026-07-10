# -*- coding: utf-8 -*-
"""Training-only timing diagnostics for active cross-signal sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_order_planner import strategy
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.trade_diagnostics import (
    ClosedTradeDiagnostic,
    DiagnosticOrderPlanner,
    build_closed_trade_diagnostics,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"


@dataclass(frozen=True)
class CrossEvent:
    direction: str
    days_ago: int


def latest_cross_event(fast, slow, window: int = 3) -> CrossEvent | None:
    fast_values = np.asarray(getattr(fast, "values", fast), dtype=float)
    slow_values = np.asarray(getattr(slow, "values", slow), dtype=float)
    if len(fast_values) < int(window) + 1 or len(slow_values) < int(window) + 1:
        return None
    diff = fast_values - slow_values
    latest = None
    for offset in range(int(window), 0, -1):
        previous = diff[-offset - 1]
        current = diff[-offset]
        if np.isnan(previous) or np.isnan(current):
            continue
        if previous <= 0 and current > 0:
            latest = CrossEvent("above", offset - 1)
        elif previous >= 0 and current < 0:
            latest = CrossEvent("below", offset - 1)
    return latest


def classify_cross_sequence(
    macd_days_ago: int | None,
    oscillator_days_ago: Sequence[int],
) -> str:
    oscillator_days = [int(value) for value in oscillator_days_ago]
    if macd_days_ago is None:
        return "no_macd_confirmation" if oscillator_days else "no_active_up_sequence"
    if not oscillator_days:
        return "macd_only"
    macd_day = int(macd_days_ago)
    if all(value > macd_day for value in oscillator_days):
        return "oscillators_lead_macd"
    if all(value < macd_day for value in oscillator_days):
        return "macd_leads_oscillators"
    if all(value == macd_day for value in oscillator_days):
        return "same_day"
    return "mixed"


@dataclass
class CrossSequenceSignalAdapter:
    source: object
    params: dict | None = None
    _cache: Dict[tuple[str, str], tuple[dict | None, str | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = strategy.get_default_params()

    def score(self, code, current_date, return_reason=False):
        code_text = str(code).split(".")[0]
        date_text = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        key = (code_text, date_text)
        if key not in self._cache:
            base_score, reason = self.source.score(code_text, date_text, return_reason=True)
            if base_score is None:
                self._cache[key] = (None, reason)
            else:
                frame, signal_date = self.source.load_signal_frame(code_text, date_text)
                score_signal_date = str(base_score.get("signal_date", signal_date))
                if signal_date is None or str(signal_date) != score_signal_date:
                    raise ValueError("cross sequence signal_date does not match base score")
                max_data_date = str(pd.to_datetime(frame["date"]).max().date())
                if max_data_date > score_signal_date:
                    raise ValueError("cross sequence frame contains data after signal_date")
                enriched = dict(base_score)
                enriched.update(self._sequence_fields(frame))
                enriched["sequence_data_date"] = max_data_date
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result

    def _sequence_fields(self, frame: pd.DataFrame) -> dict:
        p = self.params
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        rsi6 = strategy.calc_rsi(close, p["rsi_fast"])
        rsi12 = strategy.calc_rsi(close, p["rsi_mid"])
        rsi24 = strategy.calc_rsi(close, p["rsi_slow"])
        dif, dea, _ = strategy.calc_macd(
            close,
            p["macd_fast"],
            p["macd_slow"],
            p["macd_signal"],
        )
        k, d, j = strategy.calc_kdj(
            high,
            low,
            close,
            p["kdj_n"],
            p["kdj_m1"],
            p["kdj_m2"],
        )
        window = int(p["cross_window"])
        events = {
            "rsi12": latest_cross_event(rsi6, rsi12, window),
            "rsi24": latest_cross_event(rsi6, rsi24, window),
            "macd": latest_cross_event(dif, dea, window),
            "kdj_k": latest_cross_event(k, d, window),
            "kdj_j": latest_cross_event(j, d, window),
        }
        oscillator_days = [
            event.days_ago
            for name, event in events.items()
            if name != "macd" and event is not None and event.direction == "above"
        ]
        macd_event = events["macd"]
        macd_days = (
            macd_event.days_ago
            if macd_event is not None and macd_event.direction == "above"
            else None
        )
        fields = {
            "cross_sequence": classify_cross_sequence(macd_days, oscillator_days),
            "oscillator_up_cross_count": len(oscillator_days),
            "macd_up_cross_days_ago": macd_days,
        }
        for name, event in events.items():
            fields["%s_cross_direction" % name] = event.direction if event else None
            fields["%s_cross_days_ago" % name] = event.days_ago if event else None
        return fields


@dataclass(frozen=True)
class SequenceStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class SequenceGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceAttributionReport:
    by_sequence: Dict[str, SequenceStats]
    by_trend_sequence: Dict[str, SequenceStats]
    by_year_sequence: Dict[str, SequenceStats]
    mild_by_year_sequence: Dict[str, SequenceStats]
    gate: SequenceGateDecision


def build_sequence_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
) -> SequenceAttributionReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    mild_items = [trade for trade in items if _trend_group(trade.entry_score) == "mild_up"]
    by_sequence = _group_stats(items, lambda trade: _sequence(trade.entry_score))
    by_trend_sequence = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            _trend_group(trade.entry_score),
            _sequence(trade.entry_score),
        ),
    )
    by_year_sequence = _group_stats(
        items,
        lambda trade: "%s:%s" % (str(trade.buy_date)[:4], _sequence(trade.entry_score)),
    )
    mild_by_year_sequence = _group_stats(
        mild_items,
        lambda trade: "%s:%s" % (str(trade.buy_date)[:4], _sequence(trade.entry_score)),
    )
    oscillator_lead = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _sequence(trade.entry_score) == "oscillators_lead_macd"
        ])
        for year in (2019, 2020, 2021)
    }
    macd_lead = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _sequence(trade.entry_score) == "macd_leads_oscillators"
        ])
        for year in (2019, 2020, 2021)
    }
    return SequenceAttributionReport(
        by_sequence=by_sequence,
        by_trend_sequence=by_trend_sequence,
        by_year_sequence=by_year_sequence,
        mild_by_year_sequence=mild_by_year_sequence,
        gate=evaluate_sequence_gate(oscillator_lead, macd_lead),
    )


def evaluate_sequence_gate(
    oscillator_lead_by_year: Mapping[int, SequenceStats],
    macd_lead_by_year: Mapping[int, SequenceStats],
) -> SequenceGateDecision:
    reasons = []
    oscillator_total = sum(
        oscillator_lead_by_year.get(year, SequenceStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    macd_total = sum(
        macd_lead_by_year.get(year, SequenceStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if oscillator_total < 10:
        reasons.append("mild oscillator-lead subset has fewer than 10 trades")
    if macd_total < 10:
        reasons.append("mild MACD-lead subset has fewer than 10 trades")
    for year in (2019, 2020, 2021):
        oscillator = oscillator_lead_by_year.get(year, SequenceStats())
        macd = macd_lead_by_year.get(year, SequenceStats())
        if oscillator.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild oscillator-lead trades" % year)
        if macd.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild MACD-lead trades" % year)
            continue
        if oscillator.average_return <= macd.average_return:
            reasons.append("%d oscillator lead does not improve average return" % year)
        if oscillator.win_rate <= macd.win_rate:
            reasons.append("%d oscillator lead does not improve win rate" % year)
    return SequenceGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_sequence_attribution(
    loader=None,
    initial_cash: float = 20000.0,
) -> SequenceAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = CrossSequenceSignalAdapter(source)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_sequence_attribution(trades)


def _group_stats(trades, key_fn) -> Dict[str, SequenceStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {
        key: _stats(items)
        for key, items in sorted(grouped.items())
    }


def _stats(trades) -> SequenceStats:
    items = list(trades)
    return SequenceStats(
        closed_trades=len(items),
        wins=sum(1 for trade in items if float(trade.pnl) > 0),
        losses=sum(1 for trade in items if float(trade.pnl) < 0),
        realized_pnl=sum(float(trade.pnl) for trade in items),
        gross_profit=sum(float(trade.pnl) for trade in items if float(trade.pnl) > 0),
        gross_loss=sum(abs(float(trade.pnl)) for trade in items if float(trade.pnl) < 0),
        average_return=(
            sum(float(trade.return_pct) / 100.0 for trade in items) / len(items)
            if items else 0.0
        ),
    )


def _sequence(score: Mapping[str, object]) -> str:
    return str(score.get("cross_sequence", "unknown"))


def _trend_group(score: Mapping[str, object]) -> str:
    try:
        trend = float(score.get("trend_score", 0) or 0)
    except (TypeError, ValueError):
        trend = 0.0
    if trend >= 20:
        return "strong_up"
    if trend > 0:
        return "mild_up"
    if trend < 0:
        return "down"
    return "sideways"


def _assert_training_dates(dates) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Cross-sequence attribution contains dates outside 2019-2021 training window")


def format_sequence_attribution(report: SequenceAttributionReport) -> str:
    lines = ["Cross-signal active cross sequence attribution (2019-2021)"]
    for section_name, section in (
        ("SEQUENCE", report.by_sequence),
        ("TREND", report.by_trend_sequence),
        ("YEAR", report.by_year_sequence),
        ("MILD_YEAR", report.mild_by_year_sequence),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={}".format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_sequence_attribution(run_training_sequence_attribution()))


if __name__ == "__main__":
    main()
