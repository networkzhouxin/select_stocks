# -*- coding: utf-8 -*-
"""Training-only T-1-safe 09:35 execution-gap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Dict, Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class GapTradeStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_gap_atr: float = 0.0
    average_mfe: float = 0.0
    average_mae: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class GapGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapExecutionReport:
    by_bucket: Dict[str, GapTradeStats]
    by_year_bucket: Dict[str, GapTradeStats]
    by_trend_bucket: Dict[str, GapTradeStats]
    gate: GapGateDecision


@dataclass(frozen=True)
class _GapObservation:
    trade: ClosedTradeDiagnostic
    gap_atr: float
    bucket: str
    mfe: float
    mae: float


def gap_atr_bucket(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if isnan(numeric):
        return "unknown"
    if numeric <= 0:
        return "non_positive"
    if numeric <= 0.5:
        return "up_to_half"
    if numeric <= 1.0:
        return "half_to_one"
    return "above_one"


def build_gap_execution_report(
    trades: Iterable[ClosedTradeDiagnostic],
    trade_dates: Sequence[str],
    entry_price_lookup,
    close_price_lookup,
) -> GapExecutionReport:
    items = list(trades)
    dates = [str(date) for date in trade_dates]
    all_dates = list(dates)
    all_dates.extend(str(trade.buy_date) for trade in items)
    all_dates.extend(str(trade.sell_date) for trade in items)
    _assert_training_dates(all_dates)

    observations = []
    for trade in items:
        signal_date = str(trade.entry_score.get("signal_date", ""))
        if not signal_date:
            raise ValueError("gap diagnostic requires entry signal_date")
        if signal_date >= str(trade.buy_date):
            raise ValueError("signal_date must precede buy_date")

        previous_close = _numeric(trade.entry_score.get("close"))
        atr = _numeric(trade.entry_score.get("atr"))
        try:
            entry_price = float(entry_price_lookup(str(trade.code).split(".")[0], str(trade.buy_date)))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            entry_price = float("nan")
        gap_atr = (
            (entry_price - previous_close) / atr
            if atr > 0 and previous_close > 0 and not isnan(entry_price)
            else float("nan")
        )
        mfe, mae = _close_excursions(
            trade,
            entry_price,
            dates,
            close_price_lookup,
        )
        observations.append(
            _GapObservation(
                trade=trade,
                gap_atr=gap_atr,
                bucket=gap_atr_bucket(gap_atr),
                mfe=mfe,
                mae=mae,
            )
        )

    by_bucket = _group_stats(observations, lambda item: item.bucket)
    by_year_bucket = _group_stats(
        observations,
        lambda item: "%s:%s" % (str(item.trade.buy_date)[:4], item.bucket),
    )
    by_trend_bucket = _group_stats(
        observations,
        lambda item: "%s:%s" % (_trend_group(item.trade.entry_score), item.bucket),
    )
    above_by_year = {
        year: _observation_stats([
            item for item in observations
            if str(item.trade.buy_date).startswith(str(year)) and item.bucket == "above_one"
        ])
        for year in (2019, 2020, 2021)
    }
    rest_by_year = {
        year: _observation_stats([
            item for item in observations
            if str(item.trade.buy_date).startswith(str(year)) and item.bucket != "above_one"
        ])
        for year in (2019, 2020, 2021)
    }
    return GapExecutionReport(
        by_bucket=by_bucket,
        by_year_bucket=by_year_bucket,
        by_trend_bucket=by_trend_bucket,
        gate=evaluate_gap_filter_gate(above_by_year, rest_by_year),
    )


def evaluate_gap_filter_gate(
    above_one_by_year: Mapping[int, GapTradeStats],
    rest_by_year: Mapping[int, GapTradeStats],
) -> GapGateDecision:
    reasons = []
    total_above = sum(
        above_one_by_year.get(year, GapTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if total_above < 10:
        reasons.append("above-one-ATR subset has fewer than 10 closed trades")

    for year in (2019, 2020, 2021):
        above = above_one_by_year.get(year, GapTradeStats())
        rest = rest_by_year.get(year, GapTradeStats())
        if above.closed_trades < 3:
            reasons.append("%d has fewer than 3 above-one-ATR trades" % year)
        if rest.closed_trades <= 0:
            reasons.append("%d has no non-above-one comparison trades" % year)
            continue
        if above.average_return >= rest.average_return:
            reasons.append("%d above-one-ATR average return does not underperform" % year)
        if above.win_rate >= rest.win_rate:
            reasons.append("%d above-one-ATR win rate does not underperform" % year)
    return GapGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_gap_execution(
    loader=None,
    initial_cash: float = 20000.0,
) -> GapExecutionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )

    def entry_price(code: str, date: str) -> float:
        return float(loader.get_minute_bar(code, date, "09:35")["close"])

    def close_price(code: str, date: str) -> float:
        frame = loader.load_daily_frame(code, date)
        rows = frame[frame["date"].astype(str) == str(date)]
        if rows.empty:
            raise KeyError("No daily close for %s %s" % (code, date))
        return float(rows.iloc[0]["close"])

    return build_gap_execution_report(
        trades=trades,
        trade_dates=trade_dates,
        entry_price_lookup=entry_price,
        close_price_lookup=close_price,
    )


def _group_stats(observations, key_fn) -> Dict[str, GapTradeStats]:
    grouped: Dict[str, list[_GapObservation]] = {}
    for item in observations:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    return {
        key: _observation_stats(items)
        for key, items in sorted(grouped.items())
    }


def _observation_stats(observations) -> GapTradeStats:
    items = list(observations)
    valid_gaps = [item.gap_atr for item in items if not isnan(item.gap_atr)]
    return GapTradeStats(
        closed_trades=len(items),
        wins=sum(1 for item in items if float(item.trade.pnl) > 0),
        losses=sum(1 for item in items if float(item.trade.pnl) < 0),
        realized_pnl=sum(float(item.trade.pnl) for item in items),
        gross_profit=sum(float(item.trade.pnl) for item in items if float(item.trade.pnl) > 0),
        gross_loss=sum(abs(float(item.trade.pnl)) for item in items if float(item.trade.pnl) < 0),
        average_return=(
            sum(float(item.trade.return_pct) / 100.0 for item in items) / len(items)
            if items else 0.0
        ),
        average_gap_atr=(sum(valid_gaps) / len(valid_gaps) if valid_gaps else 0.0),
        average_mfe=(sum(item.mfe for item in items) / len(items) if items else 0.0),
        average_mae=(sum(item.mae for item in items) / len(items) if items else 0.0),
    )


def _close_excursions(trade, entry_price, trade_dates, close_price_lookup) -> tuple[float, float]:
    if isnan(entry_price) or entry_price <= 0:
        return 0.0, 0.0
    date_index = {date: index for index, date in enumerate(trade_dates)}
    start = date_index.get(str(trade.buy_date))
    end = date_index.get(str(trade.sell_date))
    if start is None or end is None or end < start:
        return 0.0, 0.0
    returns = []
    for date in trade_dates[start:end + 1]:
        try:
            price = float(close_price_lookup(str(trade.code).split(".")[0], date))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if price > 0:
            returns.append(price / entry_price - 1.0)
    if not returns:
        return 0.0, 0.0
    return max(returns), min(returns)


def _trend_group(score: Mapping[str, object]) -> str:
    trend = _numeric(score.get("trend_score"))
    if trend >= 20:
        return "strong_up"
    if trend > 0:
        return "mild_up"
    if trend < 0:
        return "down"
    return "sideways"


def _numeric(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Gap diagnostics contain dates outside 2019-2021 training window")


def format_gap_execution(report: GapExecutionReport) -> str:
    lines = ["Cross-signal 09:35 ATR-normalized gap diagnostics (2019-2021)"]
    for key, item in report.by_bucket.items():
        lines.append(_format_stats("BUCKET", key, item))
    for key, item in report.by_year_bucket.items():
        lines.append(_format_stats("YEAR", key, item))
    for key, item in report.by_trend_bucket.items():
        lines.append(_format_stats("TREND", key, item))
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _format_stats(section: str, key: str, item: GapTradeStats) -> str:
    ratio = item.profit_loss_ratio
    return (
        "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={} "
        "gap={:.2f}ATR mfe={:.2%} mae={:.2%}"
    ).format(
        section,
        key,
        item.closed_trades,
        item.realized_pnl,
        item.average_return,
        item.win_rate,
        "n/a" if ratio is None else "%.3f" % ratio,
        item.average_gap_atr,
        item.average_mfe,
        item.average_mae,
    )


def main() -> None:
    print(format_gap_execution(run_training_gap_execution()))


if __name__ == "__main__":
    main()
