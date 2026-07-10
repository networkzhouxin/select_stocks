# -*- coding: utf-8 -*-
"""Training-only Kaufman Efficiency Ratio diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import pandas as pd

from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
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
EFFICIENCY_PERIOD = 10


def calc_efficiency_ratio(
    frame: pd.DataFrame,
    period: int = EFFICIENCY_PERIOD,
) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    directional_change = close.diff(int(period)).abs()
    path_length = close.diff().abs().rolling(
        int(period),
        min_periods=int(period),
    ).sum()
    ratio = directional_change / path_length.replace(0.0, float("nan"))
    return ratio.mask((path_length == 0.0) & directional_change.notna(), 0.0)


@dataclass
class EfficiencyRatioSignalAdapter:
    source: object
    period: int = EFFICIENCY_PERIOD
    _cache: Dict[tuple[str, str], tuple[dict | None, str | None]] = field(default_factory=dict)

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
                    raise ValueError("efficiency ratio signal_date does not match base score")
                max_data_date = str(pd.to_datetime(frame["date"]).max().date())
                if max_data_date > score_signal_date:
                    raise ValueError("efficiency ratio frame contains data after signal_date")

                ratio = calc_efficiency_ratio(frame, period=self.period)
                current = float(ratio.iloc[-1]) if not pd.isna(ratio.iloc[-1]) else float("nan")
                previous = float(ratio.iloc[-2]) if len(ratio) >= 2 and not pd.isna(ratio.iloc[-2]) else float("nan")
                change = current - previous
                enriched = dict(base_score)
                enriched.update({
                    "efficiency_ratio": current,
                    "efficiency_ratio_prev": previous,
                    "efficiency_ratio_change": change,
                    "efficiency_ratio_direction": _ratio_direction(change),
                    "efficiency_ratio_period": int(self.period),
                    "efficiency_ratio_data_date": max_data_date,
                })
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


@dataclass(frozen=True)
class EfficiencyRatioStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_ratio: float = 0.0
    average_change: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class EfficiencyRatioGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EfficiencyRatioAttributionReport:
    by_direction: Dict[str, EfficiencyRatioStats]
    by_trend_direction: Dict[str, EfficiencyRatioStats]
    by_year_direction: Dict[str, EfficiencyRatioStats]
    mild_by_year_direction: Dict[str, EfficiencyRatioStats]
    gate: EfficiencyRatioGateDecision


def build_efficiency_ratio_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
) -> EfficiencyRatioAttributionReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    mild_items = [trade for trade in items if _trend_group(trade.entry_score) == "mild_up"]
    by_direction = _group_stats(items, lambda trade: _direction(trade.entry_score))
    by_trend_direction = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            _trend_group(trade.entry_score),
            _direction(trade.entry_score),
        ),
    )
    by_year_direction = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _direction(trade.entry_score),
        ),
    )
    mild_by_year_direction = _group_stats(
        mild_items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _direction(trade.entry_score),
        ),
    )
    mild_rising = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _direction(trade.entry_score) == "rising"
        ])
        for year in (2019, 2020, 2021)
    }
    mild_non_rising = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _direction(trade.entry_score) in {"declining", "flat"}
        ])
        for year in (2019, 2020, 2021)
    }
    return EfficiencyRatioAttributionReport(
        by_direction=by_direction,
        by_trend_direction=by_trend_direction,
        by_year_direction=by_year_direction,
        mild_by_year_direction=mild_by_year_direction,
        gate=evaluate_efficiency_ratio_gate(mild_rising, mild_non_rising),
    )


def evaluate_efficiency_ratio_gate(
    rising_by_year: Mapping[int, EfficiencyRatioStats],
    non_rising_by_year: Mapping[int, EfficiencyRatioStats],
) -> EfficiencyRatioGateDecision:
    reasons = []
    rising_total = sum(
        rising_by_year.get(year, EfficiencyRatioStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    non_rising_total = sum(
        non_rising_by_year.get(year, EfficiencyRatioStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if rising_total < 15:
        reasons.append("mild rising-efficiency subset has fewer than 15 trades")
    if non_rising_total < 15:
        reasons.append("mild non-rising-efficiency subset has fewer than 15 trades")
    for year in (2019, 2020, 2021):
        rising = rising_by_year.get(year, EfficiencyRatioStats())
        non_rising = non_rising_by_year.get(year, EfficiencyRatioStats())
        if rising.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild rising-efficiency trades" % year)
        if non_rising.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild non-rising-efficiency trades" % year)
            continue
        if rising.average_return <= non_rising.average_return:
            reasons.append("%d rising efficiency does not improve average return" % year)
        if rising.win_rate <= non_rising.win_rate:
            reasons.append("%d rising efficiency does not improve win rate" % year)
    return EfficiencyRatioGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_efficiency_ratio_attribution(
    loader=None,
    initial_cash: float = 20000.0,
) -> EfficiencyRatioAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = EfficiencyRatioSignalAdapter(source, period=EFFICIENCY_PERIOD)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_efficiency_ratio_attribution(trades)


def _group_stats(trades, key_fn) -> Dict[str, EfficiencyRatioStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {
        key: _stats(group)
        for key, group in sorted(grouped.items())
    }


def _stats(trades) -> EfficiencyRatioStats:
    items = list(trades)
    ratios = [_numeric(trade.entry_score.get("efficiency_ratio"), float("nan")) for trade in items]
    changes = [_numeric(trade.entry_score.get("efficiency_ratio_change"), float("nan")) for trade in items]
    valid_ratios = [value for value in ratios if not pd.isna(value)]
    valid_changes = [value for value in changes if not pd.isna(value)]
    return EfficiencyRatioStats(
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
        average_ratio=(sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0),
        average_change=(sum(valid_changes) / len(valid_changes) if valid_changes else 0.0),
    )


def _ratio_direction(change: float) -> str:
    if pd.isna(change):
        return "unknown"
    if change > 0:
        return "rising"
    if change < 0:
        return "declining"
    return "flat"


def _direction(score: Mapping[str, object]) -> str:
    value = str(score.get("efficiency_ratio_direction", "unknown"))
    return value if value in {"rising", "declining", "flat"} else "unknown"


def _trend_group(score: Mapping[str, object]) -> str:
    trend = _numeric(score.get("trend_score"), 0.0)
    if trend >= 20:
        return "strong_up"
    if trend > 0:
        return "mild_up"
    if trend < 0:
        return "down"
    return "sideways"


def _numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _assert_training_dates(dates) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Efficiency-ratio attribution contains dates outside 2019-2021 training window")


def format_efficiency_ratio_attribution(report: EfficiencyRatioAttributionReport) -> str:
    lines = ["Cross-signal Kaufman ER(10) attribution (2019-2021)"]
    for section_name, section in (
        ("DIRECTION", report.by_direction),
        ("TREND", report.by_trend_direction),
        ("YEAR", report.by_year_direction),
        ("MILD_YEAR", report.mild_by_year_direction),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={} er={:.4f} change={:.4f}".format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_ratio,
                    item.average_change,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_efficiency_ratio_attribution(run_training_efficiency_ratio_attribution()))


if __name__ == "__main__":
    main()
