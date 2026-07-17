# -*- coding: utf-8 -*-
"""Training-only standard BOLL BandWidth diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import pandas as pd

from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research.trade_diagnostics import (
    ClosedTradeDiagnostic,
    DiagnosticOrderPlanner,
    build_closed_trade_diagnostics,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
BOLL_PERIOD = 20
BOLL_STD_MULT = 2.0


def calc_boll_bandwidth(
    frame: pd.DataFrame,
    period: int = BOLL_PERIOD,
    std_mult: float = BOLL_STD_MULT,
) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    mid = close.rolling(int(period), min_periods=int(period)).mean()
    std = close.rolling(int(period), min_periods=int(period)).std()
    return (2.0 * float(std_mult) * std) / mid.replace(0.0, float("nan"))


@dataclass
class BollWidthSignalAdapter:
    source: object
    period: int = BOLL_PERIOD
    std_mult: float = BOLL_STD_MULT
    _cache: Dict[tuple[str, str], tuple[dict | None, str | None]] = field(default_factory=dict)

    def score(self, code, current_date, return_reason=False):
        code_text = str(code).split(".")[0]
        date_text = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        key = (code_text, date_text)
        if key not in self._cache:
            base_score, reason = self.source.score(
                code_text,
                date_text,
                return_reason=True,
            )
            if base_score is None:
                self._cache[key] = (None, reason)
            else:
                frame, signal_date = self.source.load_signal_frame(code_text, date_text)
                score_signal_date = str(base_score.get("signal_date", signal_date))
                if signal_date is None or str(signal_date) != score_signal_date:
                    raise ValueError("BOLL width signal_date does not match base score")
                max_data_date = str(pd.to_datetime(frame["date"]).max().date())
                if max_data_date > score_signal_date:
                    raise ValueError("BOLL width frame contains data after signal_date")

                width = calc_boll_bandwidth(
                    frame,
                    period=self.period,
                    std_mult=self.std_mult,
                )
                current = float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else float("nan")
                previous = float(width.iloc[-2]) if len(width) >= 2 and not pd.isna(width.iloc[-2]) else float("nan")
                change = current - previous
                enriched = dict(base_score)
                enriched.update({
                    "boll_width": current,
                    "boll_width_prev": previous,
                    "boll_width_change": change,
                    "boll_width_direction": _width_direction(change),
                    "boll_width_period": int(self.period),
                    "boll_width_std_mult": float(self.std_mult),
                    "boll_width_data_date": max_data_date,
                })
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


@dataclass(frozen=True)
class BollWidthStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_width: float = 0.0
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
class BollWidthGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BollWidthAttributionReport:
    by_direction: Dict[str, BollWidthStats]
    by_trend_direction: Dict[str, BollWidthStats]
    by_year_direction: Dict[str, BollWidthStats]
    mild_by_year_direction: Dict[str, BollWidthStats]
    gate: BollWidthGateDecision


def build_boll_width_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
) -> BollWidthAttributionReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
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
        [trade for trade in items if _trend_group(trade.entry_score) == "mild_up"],
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _direction(trade.entry_score),
        ),
    )
    mild_rising = {
        year: _stats([
            trade for trade in items
            if str(trade.buy_date).startswith(str(year))
            and _trend_group(trade.entry_score) == "mild_up"
            and _direction(trade.entry_score) == "rising"
        ])
        for year in (2019, 2020, 2021)
    }
    mild_non_rising = {
        year: _stats([
            trade for trade in items
            if str(trade.buy_date).startswith(str(year))
            and _trend_group(trade.entry_score) == "mild_up"
            and _direction(trade.entry_score) in {"declining", "flat"}
        ])
        for year in (2019, 2020, 2021)
    }
    return BollWidthAttributionReport(
        by_direction=by_direction,
        by_trend_direction=by_trend_direction,
        by_year_direction=by_year_direction,
        mild_by_year_direction=mild_by_year_direction,
        gate=evaluate_boll_width_gate(mild_rising, mild_non_rising),
    )


def evaluate_boll_width_gate(
    rising_by_year: Mapping[int, BollWidthStats],
    non_rising_by_year: Mapping[int, BollWidthStats],
) -> BollWidthGateDecision:
    reasons = []
    rising_total = sum(
        rising_by_year.get(year, BollWidthStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    non_rising_total = sum(
        non_rising_by_year.get(year, BollWidthStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if rising_total < 15:
        reasons.append("mild rising-width subset has fewer than 15 trades")
    if non_rising_total < 15:
        reasons.append("mild non-rising-width subset has fewer than 15 trades")

    for year in (2019, 2020, 2021):
        rising = rising_by_year.get(year, BollWidthStats())
        non_rising = non_rising_by_year.get(year, BollWidthStats())
        if rising.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild rising-width trades" % year)
        if non_rising.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild non-rising-width trades" % year)
            continue
        if rising.average_return <= non_rising.average_return:
            reasons.append("%d rising width does not improve mild-trend average return" % year)
        if rising.win_rate <= non_rising.win_rate:
            reasons.append("%d rising width does not improve mild-trend win rate" % year)
    return BollWidthGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_boll_width_attribution(
    loader=None,
    initial_cash: float = 20000.0,
) -> BollWidthAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = BollWidthSignalAdapter(
        source,
        period=BOLL_PERIOD,
        std_mult=BOLL_STD_MULT,
    )
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_boll_width_attribution(trades)


def _group_stats(trades, key_fn) -> Dict[str, BollWidthStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {
        key: _stats(items)
        for key, items in sorted(grouped.items())
    }


def _stats(trades) -> BollWidthStats:
    items = list(trades)
    widths = [_numeric(trade.entry_score.get("boll_width"), float("nan")) for trade in items]
    changes = [_numeric(trade.entry_score.get("boll_width_change"), float("nan")) for trade in items]
    valid_widths = [value for value in widths if not pd.isna(value)]
    valid_changes = [value for value in changes if not pd.isna(value)]
    return BollWidthStats(
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
        average_width=(sum(valid_widths) / len(valid_widths) if valid_widths else 0.0),
        average_change=(sum(valid_changes) / len(valid_changes) if valid_changes else 0.0),
    )


def _width_direction(change: float) -> str:
    if pd.isna(change):
        return "unknown"
    if change > 0:
        return "rising"
    if change < 0:
        return "declining"
    return "flat"


def _direction(score: Mapping[str, object]) -> str:
    value = str(score.get("boll_width_direction", "unknown"))
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
        raise ValueError("BOLL width attribution contains dates outside 2019-2021 training window")


def format_boll_width_attribution(report: BollWidthAttributionReport) -> str:
    lines = ["Cross-signal BOLL(20,2) BandWidth attribution (2019-2021)"]
    for section_name, section in (
        ("DIRECTION", report.by_direction),
        ("TREND", report.by_trend_direction),
        ("YEAR", report.by_year_direction),
        ("MILD_YEAR", report.mild_by_year_direction),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={} width={:.4f} change={:.4f}".format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_width,
                    item.average_change,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_boll_width_attribution(run_training_boll_width_attribution()))


if __name__ == "__main__":
    main()
