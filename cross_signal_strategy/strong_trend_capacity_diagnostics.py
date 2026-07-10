# -*- coding: utf-8 -*-
"""Training-only diagnostics for strong-trend entries and idle capacity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

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
STRONG_TREND_SCORE = 20.0


@dataclass(frozen=True)
class StrongTrendEntryContext:
    date: str
    code: str
    trend_score: float
    is_strong: bool
    unused_slots_after_orders: int
    cash_headroom_ratio: float
    slot_ratio: float
    capacity_eligible: bool


@dataclass(frozen=True)
class StrongTrendPathStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
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
class CapacityConcentration:
    gross_profit: float = 0.0
    largest_trade_profit_share: float = 0.0
    largest_code_profit_share: float = 0.0


@dataclass(frozen=True)
class CapacityGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrongTrendCapacityReport:
    strong_entries: int
    strong_open_entries: int
    strong_stats: StrongTrendPathStats
    strong_by_year: Dict[int, StrongTrendPathStats]
    capacity_entries: int
    capacity_open_entries: int
    capacity_stats: StrongTrendPathStats
    capacity_by_year: Dict[int, StrongTrendPathStats]
    concentration: CapacityConcentration
    gate: CapacityGateDecision


@dataclass
class StrongTrendCapacityPlanner(DiagnosticOrderPlanner):
    entry_contexts: Dict[tuple[str, str], StrongTrendEntryContext] = field(default_factory=dict)

    def plan_orders(self, current_date, previous_date, broker, current_prices=None):
        orders = super().plan_orders(
            current_date,
            previous_date,
            broker,
            current_prices=current_prices,
        )
        contexts = build_entry_contexts(
            date=str(current_date),
            broker=broker,
            current_prices=current_prices or {},
            orders=orders,
            scores=self.last_scores,
            params=self.params,
        )
        for code, context in contexts.items():
            self.entry_contexts[(str(current_date), code)] = context
        return orders


def build_entry_contexts(
    date: str,
    broker,
    current_prices: Mapping[str, float],
    orders: Sequence[Mapping[str, object]],
    scores: Mapping[str, Mapping[str, object]],
    params: Mapping[str, object],
) -> Dict[str, StrongTrendEntryContext]:
    positions = getattr(broker, "positions", {})
    buy_orders = [item for item in orders if str(item.get("reason", "")) == "buy_signal"]
    sell_codes = {
        str(item.get("code", "")).split(".")[0]
        for item in orders
        if str(item.get("reason", "")) in {"signal_sell", "atr_stop"}
        and float(item.get("target_value", 0.0) or 0.0) <= 0
    }
    held_after_sell = {
        str(code).split(".")[0]
        for code in positions
        if str(code).split(".")[0] not in sell_codes
    }
    max_hold = int(params["max_hold"])
    unused_slots = max(0, max_hold - len(held_after_sell) - len(buy_orders))

    total_value = float(getattr(broker, "cash", 0.0)) + sum(
        int(position.amount)
        * float(current_prices.get(code, getattr(position, "avg_cost", 0.0)))
        for code, position in positions.items()
    )
    sale_value = sum(
        int(positions[code].amount)
        * float(current_prices.get(code, getattr(positions[code], "avg_cost", 0.0)))
        for code in positions
        if str(code).split(".")[0] in sell_codes
    )
    planned_buy_value = sum(float(item.get("target_value", 0.0) or 0.0) for item in buy_orders)
    reserve_value = total_value * max(0.0, 1.0 - float(params["base_ratio"]))
    headroom_value = max(
        0.0,
        float(getattr(broker, "cash", 0.0)) + sale_value - planned_buy_value - reserve_value,
    )
    headroom_ratio = headroom_value / total_value if total_value > 0 else 0.0

    first_strong_code = None
    for item in buy_orders:
        code = str(item.get("code", "")).split(".")[0]
        if _numeric(scores.get(code, {}).get("trend_score")) >= STRONG_TREND_SCORE:
            first_strong_code = code
            break

    contexts: Dict[str, StrongTrendEntryContext] = {}
    for item in buy_orders:
        code = str(item.get("code", "")).split(".")[0]
        score = scores.get(code, {})
        trend_score = _numeric(score.get("trend_score"))
        is_strong = trend_score >= STRONG_TREND_SCORE
        target_value = float(item.get("target_value", 0.0) or 0.0)
        slot_ratio = target_value / total_value if total_value > 0 else 0.0
        can_fund = headroom_ratio + 1e-12 >= slot_ratio and slot_ratio > 0
        contexts[code] = StrongTrendEntryContext(
            date=str(date),
            code=code,
            trend_score=trend_score,
            is_strong=is_strong,
            unused_slots_after_orders=unused_slots,
            cash_headroom_ratio=headroom_ratio,
            slot_ratio=slot_ratio,
            capacity_eligible=(
                is_strong
                and unused_slots > 0
                and code == first_strong_code
                and can_fund
            ),
        )
    return contexts


def build_strong_trend_capacity_report(
    results: Iterable[object],
    trades: Iterable[ClosedTradeDiagnostic],
    entry_contexts: Mapping[tuple[str, str], StrongTrendEntryContext],
    trade_dates: Sequence[str],
    close_price_lookup,
) -> StrongTrendCapacityReport:
    days = list(results)
    closed = list(trades)
    dates = [str(date) for date in trade_dates]
    all_dates = [str(day.date) for day in days] + dates
    all_dates.extend(str(trade.buy_date) for trade in closed)
    all_dates.extend(str(trade.sell_date) for trade in closed)
    all_dates.extend(str(key[0]) for key in entry_contexts)
    _assert_training_dates(all_dates)

    filled_buy_keys = {
        (str(day.date), str(order.code).split(".")[0])
        for day in days
        for order in getattr(day, "orders", [])
        if getattr(order, "filled", False) and int(getattr(order, "amount_delta", 0)) > 0
    }
    strong_contexts = {
        key: context
        for key, context in entry_contexts.items()
        if key in filled_buy_keys and context.is_strong
    }
    capacity_contexts = {
        key: context
        for key, context in strong_contexts.items()
        if context.capacity_eligible
    }

    strong_trades = [
        trade for trade in closed
        if (str(trade.buy_date), str(trade.code).split(".")[0]) in strong_contexts
    ]
    capacity_trades = [
        trade for trade in closed
        if (str(trade.buy_date), str(trade.code).split(".")[0]) in capacity_contexts
    ]
    path_map = {
        _trade_key(trade): _close_excursions(trade, dates, close_price_lookup)
        for trade in strong_trades
    }
    capacity_path_map = {
        _trade_key(trade): path_map[_trade_key(trade)]
        for trade in capacity_trades
    }
    strong_stats = _path_stats(strong_trades, path_map)
    strong_by_year = _year_stats(strong_trades, path_map)
    capacity_stats = _path_stats(capacity_trades, capacity_path_map)
    capacity_by_year = _year_stats(capacity_trades, capacity_path_map)
    concentration = _concentration(capacity_trades)
    gate = evaluate_capacity_gate(capacity_stats, capacity_by_year, concentration)

    return StrongTrendCapacityReport(
        strong_entries=len(strong_contexts),
        strong_open_entries=max(0, len(strong_contexts) - len(strong_trades)),
        strong_stats=strong_stats,
        strong_by_year=strong_by_year,
        capacity_entries=len(capacity_contexts),
        capacity_open_entries=max(0, len(capacity_contexts) - len(capacity_trades)),
        capacity_stats=capacity_stats,
        capacity_by_year=capacity_by_year,
        concentration=concentration,
        gate=gate,
    )


def evaluate_capacity_gate(
    stats: StrongTrendPathStats,
    by_year: Mapping[int, StrongTrendPathStats],
    concentration: CapacityConcentration,
) -> CapacityGateDecision:
    reasons = []
    if stats.closed_trades < 10:
        reasons.append("capacity subset has fewer than 10 closed trades")
    for year in (2019, 2020, 2021):
        annual = by_year.get(year, StrongTrendPathStats())
        if annual.closed_trades < 3:
            reasons.append("%d has fewer than 3 capacity trades" % year)
        if annual.realized_pnl <= 0:
            reasons.append("%d capacity PnL is not positive" % year)
    ratio = stats.profit_loss_ratio
    if stats.gross_profit <= 0 or (ratio is not None and ratio <= 1.0):
        reasons.append("capacity subset profit/loss ratio is not above 1")
    if stats.average_mfe <= abs(stats.average_mae):
        reasons.append("capacity subset favorable excursion does not dominate adverse excursion")
    if concentration.largest_trade_profit_share > 0.50:
        reasons.append("largest trade contributes more than half of capacity gross profit")
    if concentration.largest_code_profit_share > 0.50:
        reasons.append("largest ETF contributes more than half of capacity gross profit")
    return CapacityGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_strong_trend_capacity(
    loader=None,
    initial_cash: float = 20000.0,
) -> StrongTrendCapacityReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = StrongTrendCapacityPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )

    def close_price(code: str, date: str) -> float:
        frame = loader.load_daily_frame(code, date)
        rows = frame[frame["date"].astype(str) == str(date)]
        if rows.empty:
            raise KeyError("No daily close for %s %s" % (code, date))
        return float(rows.iloc[0]["close"])

    return build_strong_trend_capacity_report(
        results=results,
        trades=trades,
        entry_contexts=planner.entry_contexts,
        trade_dates=trade_dates,
        close_price_lookup=close_price,
    )


def _close_excursions(trade, trade_dates, close_price_lookup) -> tuple[float, float]:
    date_index = {date: index for index, date in enumerate(trade_dates)}
    start = date_index.get(str(trade.buy_date))
    end = date_index.get(str(trade.sell_date))
    if start is None or end is None or end < start or float(trade.buy_price) <= 0:
        return 0.0, 0.0
    returns = []
    for date in trade_dates[start:end + 1]:
        try:
            price = float(close_price_lookup(str(trade.code).split(".")[0], date))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if price > 0:
            returns.append(price / float(trade.buy_price) - 1.0)
    if not returns:
        return 0.0, 0.0
    return max(returns), min(returns)


def _path_stats(trades, path_map) -> StrongTrendPathStats:
    items = list(trades)
    excursions = [path_map.get(_trade_key(trade), (0.0, 0.0)) for trade in items]
    return StrongTrendPathStats(
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
        average_mfe=(sum(value[0] for value in excursions) / len(excursions) if excursions else 0.0),
        average_mae=(sum(value[1] for value in excursions) / len(excursions) if excursions else 0.0),
    )


def _year_stats(trades, path_map) -> Dict[int, StrongTrendPathStats]:
    grouped: Dict[int, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(int(str(trade.buy_date)[:4]), []).append(trade)
    return {
        year: _path_stats(items, path_map)
        for year, items in sorted(grouped.items())
    }


def _concentration(trades) -> CapacityConcentration:
    profitable = [trade for trade in trades if float(trade.pnl) > 0]
    gross_profit = sum(float(trade.pnl) for trade in profitable)
    if gross_profit <= 0:
        return CapacityConcentration()
    by_code: Dict[str, float] = {}
    for trade in profitable:
        code = str(trade.code).split(".")[0]
        by_code[code] = by_code.get(code, 0.0) + float(trade.pnl)
    return CapacityConcentration(
        gross_profit=gross_profit,
        largest_trade_profit_share=max(float(trade.pnl) for trade in profitable) / gross_profit,
        largest_code_profit_share=max(by_code.values()) / gross_profit,
    )


def _trade_key(trade) -> tuple[str, str, str]:
    return str(trade.buy_date), str(trade.sell_date), str(trade.code).split(".")[0]


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Strong-trend capacity diagnostics contain dates outside 2019-2021 training window")


def _numeric(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def format_strong_trend_capacity(report: StrongTrendCapacityReport) -> str:
    lines = [
        "Cross-signal strong-trend idle-capacity diagnostics (2019-2021)",
        "STRONG entries={} closed={} open={} pnl={:.2f} win={:.2%} mfe={:.2%} mae={:.2%}".format(
            report.strong_entries,
            report.strong_stats.closed_trades,
            report.strong_open_entries,
            report.strong_stats.realized_pnl,
            report.strong_stats.win_rate,
            report.strong_stats.average_mfe,
            report.strong_stats.average_mae,
        ),
        "CAPACITY entries={} closed={} open={} pnl={:.2f} win={:.2%} pl={} mfe={:.2%} mae={:.2%}".format(
            report.capacity_entries,
            report.capacity_stats.closed_trades,
            report.capacity_open_entries,
            report.capacity_stats.realized_pnl,
            report.capacity_stats.win_rate,
            "n/a" if report.capacity_stats.profit_loss_ratio is None else "%.3f" % report.capacity_stats.profit_loss_ratio,
            report.capacity_stats.average_mfe,
            report.capacity_stats.average_mae,
        ),
    ]
    for year, stats in report.capacity_by_year.items():
        lines.append(
            "YEAR {} trades={} pnl={:.2f} win={:.2%} mfe={:.2%} mae={:.2%}".format(
                year,
                stats.closed_trades,
                stats.realized_pnl,
                stats.win_rate,
                stats.average_mfe,
                stats.average_mae,
            )
        )
    lines.append(
        "CONCENTRATION largest_trade={:.2%} largest_code={:.2%}".format(
            report.concentration.largest_trade_profit_share,
            report.concentration.largest_code_profit_share,
        )
    )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_strong_trend_capacity(run_training_strong_trend_capacity()))


if __name__ == "__main__":
    main()
