# -*- coding: utf-8 -*-
"""Training-only portfolio-dependence diagnostics for cross-signal entries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Dict, Iterable, Mapping, Sequence

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
CORRELATION_PERIOD = 20
HIGH_CORRELATION_THRESHOLD = 0.80


@dataclass(frozen=True)
class DependenceTradeStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_max_correlation: float = 0.0
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
class DependenceGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioDependenceReport:
    by_bucket: Dict[str, DependenceTradeStats]
    by_year_bucket: Dict[str, DependenceTradeStats]
    gate: DependenceGateDecision


@dataclass(frozen=True)
class _DependenceObservation:
    trade: ClosedTradeDiagnostic
    bucket: str
    max_correlation: float
    mae: float


@dataclass
class PortfolioDependenceOrderPlanner(DiagnosticOrderPlanner):
    """Add T-1 dependence labels after the official planner chooses orders."""

    correlation_period: int = CORRELATION_PERIOD
    high_correlation_threshold: float = HIGH_CORRELATION_THRESHOLD

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        sold_codes = {
            str(order["code"]).split(".")[0]
            for order in orders
            if order.get("reason") in {"signal_sell", "atr_stop"}
        }
        held_codes = [
            str(code).split(".")[0]
            for code in broker.positions
            if str(code).split(".")[0] not in sold_codes
        ]
        annotate_planned_buy_dependence(
            orders=orders,
            entry_score_snapshots=self.entry_score_snapshots,
            held_codes=held_codes,
            source=self.signal_adapter,
            current_date=str(current_date),
            period=self.correlation_period,
            threshold=self.high_correlation_threshold,
        )
        return orders


def calc_return_correlation(
    first_frame: pd.DataFrame,
    second_frame: pd.DataFrame,
    period: int = CORRELATION_PERIOD,
) -> float:
    """Calculate correlation from aligned trailing daily returns."""
    first = _return_series(first_frame, "first")
    second = _return_series(second_frame, "second")
    aligned = pd.concat([first, second], axis=1, join="inner").dropna()
    if len(aligned) < int(period):
        return float("nan")
    trailing = aligned.tail(int(period))
    return float(trailing.iloc[:, 0].corr(trailing.iloc[:, 1]))


def annotate_planned_buy_dependence(
    orders: Sequence[Mapping[str, object]],
    entry_score_snapshots: Dict[tuple[str, str], dict],
    held_codes: Iterable[str],
    source,
    current_date: str,
    period: int = CORRELATION_PERIOD,
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> None:
    """Annotate buys in official order without changing or reordering them."""
    references = list(dict.fromkeys(str(code).split(".")[0] for code in held_codes))
    date_text = str(current_date)
    for order in orders:
        if order.get("reason") != "buy_signal":
            continue
        code = str(order["code"]).split(".")[0]
        snapshot = entry_score_snapshots.get((date_text, code))
        if snapshot is None:
            raise ValueError("dependence diagnostic requires an entry score snapshot")
        signal_date = str(snapshot.get("signal_date", ""))
        if not signal_date or signal_date >= date_text:
            raise ValueError("dependence diagnostic requires a T-1 signal_date")

        candidate_frame, candidate_signal_date = source.load_signal_frame(code, date_text)
        if str(candidate_signal_date) != signal_date:
            raise ValueError("dependence candidate signal_date does not match entry snapshot")
        _assert_frame_not_after_signal(candidate_frame, signal_date)

        correlations = []
        for reference in references:
            reference_frame, _ = source.load_signal_frame(reference, date_text)
            _assert_frame_not_after_signal(reference_frame, signal_date)
            correlation = calc_return_correlation(
                candidate_frame,
                reference_frame,
                period=period,
            )
            if not isnan(correlation):
                correlations.append(correlation)

        max_correlation = max(correlations) if correlations else float("nan")
        if not references:
            bucket = "no_reference"
        elif not correlations:
            bucket = "unknown"
        elif max_correlation >= float(threshold):
            bucket = "high"
        else:
            bucket = "low"
        snapshot.update({
            "dependence_bucket": bucket,
            "dependence_max_correlation": max_correlation,
            "dependence_reference_count": len(references),
            "dependence_valid_correlation_count": len(correlations),
            "dependence_period": int(period),
            "dependence_threshold": float(threshold),
            "dependence_data_date": signal_date,
        })
        references.append(code)


def build_portfolio_dependence_report(
    trades: Iterable[ClosedTradeDiagnostic],
    adverse_excursion_lookup,
) -> PortfolioDependenceReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    observations = [
        _DependenceObservation(
            trade=trade,
            bucket=_dependence_bucket(trade.entry_score),
            max_correlation=_numeric(
                trade.entry_score.get("dependence_max_correlation"),
                float("nan"),
            ),
            mae=float(adverse_excursion_lookup(trade)),
        )
        for trade in items
    ]
    by_bucket = _group_stats(observations, lambda item: item.bucket)
    by_year_bucket = _group_stats(
        observations,
        lambda item: "%s:%s" % (str(item.trade.buy_date)[:4], item.bucket),
    )
    high_by_year = {
        year: _stats([
            item for item in observations
            if str(item.trade.buy_date).startswith(str(year)) and item.bucket == "high"
        ])
        for year in (2019, 2020, 2021)
    }
    low_by_year = {
        year: _stats([
            item for item in observations
            if str(item.trade.buy_date).startswith(str(year)) and item.bucket == "low"
        ])
        for year in (2019, 2020, 2021)
    }
    return PortfolioDependenceReport(
        by_bucket=by_bucket,
        by_year_bucket=by_year_bucket,
        gate=evaluate_dependence_gate(high_by_year, low_by_year),
    )


def evaluate_dependence_gate(
    high_by_year: Mapping[int, DependenceTradeStats],
    low_by_year: Mapping[int, DependenceTradeStats],
) -> DependenceGateDecision:
    reasons = []
    high_total = sum(
        high_by_year.get(year, DependenceTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    low_total = sum(
        low_by_year.get(year, DependenceTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if high_total < 15:
        reasons.append("high-dependence subset has fewer than 15 closed trades")
    if low_total < 15:
        reasons.append("low-dependence subset has fewer than 15 closed trades")
    for year in (2019, 2020, 2021):
        high = high_by_year.get(year, DependenceTradeStats())
        low = low_by_year.get(year, DependenceTradeStats())
        if high.closed_trades < 3:
            reasons.append("%d has fewer than 3 high-dependence trades" % year)
        if low.closed_trades < 3:
            reasons.append("%d has fewer than 3 low-dependence trades" % year)
            continue
        if high.average_return >= low.average_return:
            reasons.append("%d high dependence does not underperform return" % year)
        if high.average_mae >= low.average_mae:
            reasons.append("%d high dependence does not worsen adverse excursion" % year)
    return DependenceGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_portfolio_dependence(
    loader=None,
    initial_cash: float = 20000.0,
) -> PortfolioDependenceReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = PortfolioDependenceOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    date_index = {date: index for index, date in enumerate(trade_dates)}

    def adverse_excursion(trade: ClosedTradeDiagnostic) -> float:
        start = date_index.get(str(trade.buy_date))
        end = date_index.get(str(trade.sell_date))
        if start is None or end is None or end < start or float(trade.buy_price) <= 0:
            return 0.0
        returns = []
        for date in trade_dates[start:end + 1]:
            try:
                frame = loader.load_daily_frame(str(trade.code).split(".")[0], date)
                rows = frame[frame["date"].astype(str) == str(date)]
                if rows.empty:
                    continue
                price = float(rows.iloc[0]["close"])
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if price > 0:
                returns.append(price / float(trade.buy_price) - 1.0)
        return min(returns) if returns else 0.0

    return build_portfolio_dependence_report(trades, adverse_excursion)


def _return_series(frame: pd.DataFrame, name: str) -> pd.Series:
    data = frame.loc[:, ["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna().drop_duplicates("date", keep="last").sort_values("date")
    return data.set_index("date")["close"].pct_change(fill_method=None).rename(name)


def _assert_frame_not_after_signal(frame: pd.DataFrame, signal_date: str) -> None:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("dependence frame contains invalid dates")
    if not dates.empty and dates.max() > pd.Timestamp(signal_date):
        raise ValueError("dependence frame contains data after signal_date")


def _dependence_bucket(score: Mapping[str, object]) -> str:
    value = str(score.get("dependence_bucket", "unknown"))
    return value if value in {"high", "low", "no_reference"} else "unknown"


def _group_stats(observations, key_fn) -> Dict[str, DependenceTradeStats]:
    grouped: Dict[str, list[_DependenceObservation]] = {}
    for item in observations:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    return {key: _stats(items) for key, items in sorted(grouped.items())}


def _stats(observations) -> DependenceTradeStats:
    items = list(observations)
    correlations = [item.max_correlation for item in items if not isnan(item.max_correlation)]
    return DependenceTradeStats(
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
        average_max_correlation=(
            sum(correlations) / len(correlations) if correlations else 0.0
        ),
        average_mae=(sum(item.mae for item in items) / len(items) if items else 0.0),
    )


def _numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Portfolio-dependence diagnostics contain dates outside 2019-2021 training window")


def format_portfolio_dependence(report: PortfolioDependenceReport) -> str:
    lines = ["Cross-signal portfolio dependence attribution (2019-2021)"]
    for section_name, section in (
        ("BUCKET", report.by_bucket),
        ("YEAR", report.by_year_bucket),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={} corr={:.3f} mae={:.2%}".format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_max_correlation,
                    item.average_mae,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_portfolio_dependence(run_training_portfolio_dependence()))


if __name__ == "__main__":
    main()
