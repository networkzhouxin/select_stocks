# -*- coding: utf-8 -*-
"""Training-only ETF-pool market-breadth diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

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
BREADTH_MA_PERIOD = 20
MAJORITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class PoolBreadthSnapshot:
    breadth: float
    eligible_count: int
    above_count: int
    bucket: str
    data_date: str


@dataclass(frozen=True)
class BreadthTradeStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_breadth: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class BreadthGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketBreadthReport:
    by_bucket: Dict[str, BreadthTradeStats]
    by_year_bucket: Dict[str, BreadthTradeStats]
    mild_by_year_bucket: Dict[str, BreadthTradeStats]
    gate: BreadthGateDecision


@dataclass
class MarketBreadthOrderPlanner(DiagnosticOrderPlanner):
    """Attach one pool-wide T-1 breadth snapshot to official buy orders."""

    breadth_ma_period: int = BREADTH_MA_PERIOD
    majority_threshold: float = MAJORITY_THRESHOLD

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        annotate_planned_buy_breadth(
            orders=orders,
            entry_score_snapshots=self.entry_score_snapshots,
            source=self.signal_adapter,
            pool_codes=self.etf_pool,
            current_date=str(current_date),
            period=self.breadth_ma_period,
            threshold=self.majority_threshold,
        )
        return orders


def calc_above_ma(
    frame: pd.DataFrame,
    period: int = BREADTH_MA_PERIOD,
) -> bool | None:
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(close) < int(period):
        return None
    trailing = close.iloc[-int(period):]
    return bool(float(trailing.iloc[-1]) > float(trailing.mean()))


def calculate_pool_breadth(
    source,
    pool_codes: Iterable[str],
    current_date: str,
    signal_date: str,
    period: int = BREADTH_MA_PERIOD,
    threshold: float = MAJORITY_THRESHOLD,
) -> PoolBreadthSnapshot:
    eligible = 0
    above = 0
    for raw_code in pool_codes:
        code = str(raw_code).split(".")[0]
        try:
            frame, _ = source.load_signal_frame(code, str(current_date))
        except (FileNotFoundError, KeyError):
            continue
        _assert_frame_not_after_signal(frame, str(signal_date))
        state = calc_above_ma(frame, period=period)
        if state is None:
            continue
        eligible += 1
        above += int(state)

    breadth = float(above) / eligible if eligible else float("nan")
    if eligible == 0:
        bucket = "no_data"
    elif breadth < float(threshold):
        bucket = "below_majority"
    else:
        bucket = "majority"
    return PoolBreadthSnapshot(
        breadth=breadth,
        eligible_count=eligible,
        above_count=above,
        bucket=bucket,
        data_date=str(signal_date),
    )


def annotate_planned_buy_breadth(
    orders: Sequence[Mapping[str, object]],
    entry_score_snapshots: Dict[tuple[str, str], dict],
    source,
    pool_codes: Iterable[str],
    current_date: str,
    period: int = BREADTH_MA_PERIOD,
    threshold: float = MAJORITY_THRESHOLD,
) -> None:
    buys = [order for order in orders if order.get("reason") == "buy_signal"]
    if not buys:
        return
    date_text = str(current_date)
    first_code = str(buys[0]["code"]).split(".")[0]
    first = entry_score_snapshots.get((date_text, first_code))
    if first is None:
        raise ValueError("breadth diagnostic requires an entry score snapshot")
    signal_date = str(first.get("signal_date", ""))
    if not signal_date or signal_date >= date_text:
        raise ValueError("breadth diagnostic requires a T-1 signal_date")
    breadth = calculate_pool_breadth(
        source=source,
        pool_codes=pool_codes,
        current_date=date_text,
        signal_date=signal_date,
        period=period,
        threshold=threshold,
    )
    for order in buys:
        code = str(order["code"]).split(".")[0]
        snapshot = entry_score_snapshots.get((date_text, code))
        if snapshot is None:
            raise ValueError("breadth diagnostic requires an entry score snapshot")
        if str(snapshot.get("signal_date", "")) != signal_date:
            raise ValueError("same-day buy snapshots must share one signal_date")
        snapshot.update({
            "market_breadth": breadth.breadth,
            "breadth_bucket": breadth.bucket,
            "breadth_eligible_count": breadth.eligible_count,
            "breadth_above_count": breadth.above_count,
            "breadth_ma_period": int(period),
            "breadth_threshold": float(threshold),
            "breadth_data_date": breadth.data_date,
        })


def build_market_breadth_report(
    trades: Iterable[ClosedTradeDiagnostic],
) -> MarketBreadthReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    by_bucket = _group_stats(items, lambda trade: _breadth_bucket(trade.entry_score))
    by_year_bucket = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _breadth_bucket(trade.entry_score),
        ),
    )
    mild_items = [trade for trade in items if _is_mild_trend(trade.entry_score)]
    mild_by_year_bucket = _group_stats(
        mild_items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _breadth_bucket(trade.entry_score),
        ),
    )
    below_by_year = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _breadth_bucket(trade.entry_score) == "below_majority"
        ])
        for year in (2019, 2020, 2021)
    }
    majority_by_year = {
        year: _stats([
            trade for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _breadth_bucket(trade.entry_score) == "majority"
        ])
        for year in (2019, 2020, 2021)
    }
    return MarketBreadthReport(
        by_bucket=by_bucket,
        by_year_bucket=by_year_bucket,
        mild_by_year_bucket=mild_by_year_bucket,
        gate=evaluate_breadth_gate(below_by_year, majority_by_year),
    )


def evaluate_breadth_gate(
    below_by_year: Mapping[int, BreadthTradeStats],
    majority_by_year: Mapping[int, BreadthTradeStats],
) -> BreadthGateDecision:
    reasons = []
    below_total = sum(
        below_by_year.get(year, BreadthTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    majority_total = sum(
        majority_by_year.get(year, BreadthTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if below_total < 15:
        reasons.append("below-majority mild subset has fewer than 15 closed trades")
    if majority_total < 15:
        reasons.append("majority mild subset has fewer than 15 closed trades")
    for year in (2019, 2020, 2021):
        below = below_by_year.get(year, BreadthTradeStats())
        majority = majority_by_year.get(year, BreadthTradeStats())
        if below.closed_trades < 3:
            reasons.append("%d has fewer than 3 below-majority mild trades" % year)
        if majority.closed_trades < 3:
            reasons.append("%d has fewer than 3 majority mild trades" % year)
            continue
        if below.average_return >= majority.average_return:
            reasons.append("%d below-majority breadth does not underperform return" % year)
        if below.win_rate >= majority.win_rate:
            reasons.append("%d below-majority breadth does not underperform win rate" % year)
    return BreadthGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_market_breadth(
    loader=None,
    initial_cash: float = 20000.0,
) -> MarketBreadthReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = MarketBreadthOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_market_breadth_report(trades)


def _assert_frame_not_after_signal(frame: pd.DataFrame, signal_date: str) -> None:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("breadth frame contains invalid dates")
    if not dates.empty and dates.max() > pd.Timestamp(signal_date):
        raise ValueError("breadth frame contains data after signal_date")


def _breadth_bucket(score: Mapping[str, object]) -> str:
    value = str(score.get("breadth_bucket", "no_data"))
    return value if value in {"below_majority", "majority"} else "no_data"


def _is_mild_trend(score: Mapping[str, object]) -> bool:
    trend = _numeric(score.get("trend_score"), 0.0)
    return 0.0 < trend < 20.0


def _group_stats(trades, key_fn) -> Dict[str, BreadthTradeStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {key: _stats(items) for key, items in sorted(grouped.items())}


def _stats(trades) -> BreadthTradeStats:
    items = list(trades)
    breadth_values = [
        _numeric(trade.entry_score.get("market_breadth"), float("nan"))
        for trade in items
    ]
    breadth_values = [value for value in breadth_values if not pd.isna(value)]
    return BreadthTradeStats(
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
        average_breadth=(
            sum(breadth_values) / len(breadth_values) if breadth_values else 0.0
        ),
    )


def _numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Market-breadth diagnostics contain dates outside 2019-2021 training window")


def format_market_breadth(report: MarketBreadthReport) -> str:
    lines = ["Cross-signal ETF-pool MA20 breadth attribution (2019-2021)"]
    for section_name, section in (
        ("BUCKET", report.by_bucket),
        ("YEAR", report.by_year_bucket),
        ("MILD_YEAR", report.mild_by_year_bucket),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} pl={} breadth={:.2%}".format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_breadth,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_market_breadth(run_training_market_breadth()))


if __name__ == "__main__":
    main()
