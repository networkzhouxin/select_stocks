# -*- coding: utf-8 -*-
"""Training-only T-2-safe horizontal support/resistance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
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
STRUCTURE_PERIOD = 20
NEAR_DISTANCE_ATR = 1.0


@dataclass(frozen=True)
class HorizontalStructureSnapshot:
    eligible: bool
    resistance: float
    support: float
    resistance_distance_atr: float
    support_distance_atr: float
    pressure_bucket: str
    support_bucket: str
    signal_date: str
    level_data_date: str


@dataclass
class HorizontalStructureSignalAdapter:
    """Attach fixed horizontal-price structure without changing base scores."""

    source: object
    period: int = STRUCTURE_PERIOD
    _cache: Dict[tuple[str, str], tuple[dict | None, str | None]] = field(
        default_factory=dict
    )

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
                frame, signal_date = self.source.load_signal_frame(
                    code_text,
                    date_text,
                )
                score_signal_date = str(base_score.get("signal_date", signal_date))
                if signal_date is None or str(signal_date) != score_signal_date:
                    raise ValueError(
                        "horizontal structure signal_date does not match base score"
                    )
                snapshot = calc_horizontal_structure(
                    frame=frame,
                    signal_date=score_signal_date,
                    atr=_numeric(base_score.get("atr"), float("nan")),
                    period=self.period,
                )
                enriched = dict(base_score)
                enriched.update({
                    "structure_eligible": snapshot.eligible,
                    "structure_resistance": snapshot.resistance,
                    "structure_support": snapshot.support,
                    "resistance_distance_atr": snapshot.resistance_distance_atr,
                    "support_distance_atr": snapshot.support_distance_atr,
                    "pressure_bucket": snapshot.pressure_bucket,
                    "support_bucket": snapshot.support_bucket,
                    "structure_period": int(self.period),
                    "structure_near_distance_atr": NEAR_DISTANCE_ATR,
                    "structure_signal_date": snapshot.signal_date,
                    "structure_level_data_date": snapshot.level_data_date,
                })
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


@dataclass(frozen=True)
class HorizontalStructureStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_resistance_distance_atr: float = 0.0
    average_support_distance_atr: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class HorizontalStructureGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HorizontalStructureReport:
    by_pressure: Dict[str, HorizontalStructureStats]
    by_support: Dict[str, HorizontalStructureStats]
    by_year_pressure: Dict[str, HorizontalStructureStats]
    mild_by_year_pressure: Dict[str, HorizontalStructureStats]
    gate: HorizontalStructureGateDecision


def calc_horizontal_structure(
    frame: pd.DataFrame,
    signal_date: str,
    atr: float,
    period: int = STRUCTURE_PERIOD,
) -> HorizontalStructureSnapshot:
    required = {"date", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("horizontal structure missing columns: %s" % ", ".join(missing))

    visible = frame.copy()
    visible["_date"] = pd.to_datetime(visible["date"], errors="coerce")
    if visible["_date"].isna().any():
        raise ValueError("horizontal structure frame contains invalid dates")
    signal_ts = pd.Timestamp(signal_date)
    if not visible.empty and visible["_date"].max() > signal_ts:
        raise ValueError("horizontal structure frame contains data after signal_date")

    signal_rows = visible.loc[visible["_date"] == signal_ts]
    prior = visible.loc[visible["_date"] < signal_ts].copy()
    prior["high"] = pd.to_numeric(prior["high"], errors="coerce")
    prior["low"] = pd.to_numeric(prior["low"], errors="coerce")
    prior = prior.dropna(subset=["high", "low"]).sort_values("_date").tail(int(period))
    numeric_atr = _numeric(atr, float("nan"))

    if len(prior) < int(period) or signal_rows.empty or isnan(numeric_atr) or numeric_atr <= 0:
        return _no_data_snapshot(signal_date, prior)

    close = pd.to_numeric(signal_rows["close"], errors="coerce").dropna()
    if close.empty:
        return _no_data_snapshot(signal_date, prior)

    resistance = float(prior["high"].max())
    support = float(prior["low"].min())
    signal_close = float(close.iloc[-1])
    resistance_distance = (resistance - signal_close) / numeric_atr
    support_distance = (signal_close - support) / numeric_atr
    return HorizontalStructureSnapshot(
        eligible=True,
        resistance=resistance,
        support=support,
        resistance_distance_atr=resistance_distance,
        support_distance_atr=support_distance,
        pressure_bucket=_pressure_bucket_from_distance(resistance_distance),
        support_bucket=_support_bucket_from_distance(support_distance),
        signal_date=signal_ts.strftime("%Y-%m-%d"),
        level_data_date=prior["_date"].max().strftime("%Y-%m-%d"),
    )


def build_horizontal_structure_report(
    trades: Iterable[ClosedTradeDiagnostic],
) -> HorizontalStructureReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    by_pressure = _group_stats(items, lambda trade: _pressure_bucket(trade.entry_score))
    by_support = _group_stats(items, lambda trade: _support_bucket(trade.entry_score))
    by_year_pressure = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _pressure_bucket(trade.entry_score),
        ),
    )
    mild_items = [trade for trade in items if _is_mild_trend(trade.entry_score)]
    mild_by_year_pressure = _group_stats(
        mild_items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _pressure_bucket(trade.entry_score),
        ),
    )
    near_by_year = {
        year: _stats([
            trade
            for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _pressure_bucket(trade.entry_score) == "near_resistance"
        ])
        for year in (2019, 2020, 2021)
    }
    other_by_year = {
        year: _stats([
            trade
            for trade in mild_items
            if str(trade.buy_date).startswith(str(year))
            and _pressure_bucket(trade.entry_score)
            in {"breakout", "room_to_resistance"}
        ])
        for year in (2019, 2020, 2021)
    }
    return HorizontalStructureReport(
        by_pressure=by_pressure,
        by_support=by_support,
        by_year_pressure=by_year_pressure,
        mild_by_year_pressure=mild_by_year_pressure,
        gate=evaluate_near_resistance_gate(near_by_year, other_by_year),
    )


def evaluate_near_resistance_gate(
    near_by_year: Mapping[int, HorizontalStructureStats],
    other_by_year: Mapping[int, HorizontalStructureStats],
) -> HorizontalStructureGateDecision:
    reasons = []
    near_total = sum(
        near_by_year.get(year, HorizontalStructureStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    other_total = sum(
        other_by_year.get(year, HorizontalStructureStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if near_total < 15:
        reasons.append("mild near-resistance subset has fewer than 15 closed trades")
    if other_total < 15:
        reasons.append("mild comparison subset has fewer than 15 closed trades")
    for year in (2019, 2020, 2021):
        near = near_by_year.get(year, HorizontalStructureStats())
        other = other_by_year.get(year, HorizontalStructureStats())
        if near.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild near-resistance trades" % year)
        if other.closed_trades < 3:
            reasons.append("%d has fewer than 3 mild comparison trades" % year)
            continue
        if near.average_return >= other.average_return:
            reasons.append("%d near resistance does not underperform average return" % year)
        if near.win_rate >= other.win_rate:
            reasons.append("%d near resistance does not underperform win rate" % year)
    return HorizontalStructureGateDecision(
        passed=not reasons,
        reasons=tuple(reasons),
    )


def run_training_horizontal_structure(
    loader=None,
    initial_cash: float = 20000.0,
) -> HorizontalStructureReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = HorizontalStructureSignalAdapter(source, period=STRUCTURE_PERIOD)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_horizontal_structure_report(trades)


def _no_data_snapshot(signal_date: str, prior: pd.DataFrame) -> HorizontalStructureSnapshot:
    level_data_date = ""
    if not prior.empty:
        level_data_date = prior["_date"].max().strftime("%Y-%m-%d")
    return HorizontalStructureSnapshot(
        eligible=False,
        resistance=float("nan"),
        support=float("nan"),
        resistance_distance_atr=float("nan"),
        support_distance_atr=float("nan"),
        pressure_bucket="no_data",
        support_bucket="no_data",
        signal_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        level_data_date=level_data_date,
    )


def _pressure_bucket_from_distance(distance: float) -> str:
    if distance < 0:
        return "breakout"
    if distance <= NEAR_DISTANCE_ATR:
        return "near_resistance"
    return "room_to_resistance"


def _support_bucket_from_distance(distance: float) -> str:
    if distance < 0:
        return "breakdown"
    if distance <= NEAR_DISTANCE_ATR:
        return "near_support"
    return "away_from_support"


def _pressure_bucket(score: Mapping[str, object]) -> str:
    value = str(score.get("pressure_bucket", "no_data"))
    return value if value in {
        "breakout",
        "near_resistance",
        "room_to_resistance",
    } else "no_data"


def _support_bucket(score: Mapping[str, object]) -> str:
    value = str(score.get("support_bucket", "no_data"))
    return value if value in {
        "breakdown",
        "near_support",
        "away_from_support",
    } else "no_data"


def _is_mild_trend(score: Mapping[str, object]) -> bool:
    trend = _numeric(score.get("trend_score"), 0.0)
    return 0.0 < trend < 20.0


def _group_stats(trades, key_fn) -> Dict[str, HorizontalStructureStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {key: _stats(group) for key, group in sorted(grouped.items())}


def _stats(trades) -> HorizontalStructureStats:
    items = list(trades)
    resistance_distances = [
        _numeric(trade.entry_score.get("resistance_distance_atr"), float("nan"))
        for trade in items
    ]
    support_distances = [
        _numeric(trade.entry_score.get("support_distance_atr"), float("nan"))
        for trade in items
    ]
    resistance_distances = [value for value in resistance_distances if not isnan(value)]
    support_distances = [value for value in support_distances if not isnan(value)]
    return HorizontalStructureStats(
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
        average_resistance_distance_atr=(
            sum(resistance_distances) / len(resistance_distances)
            if resistance_distances else 0.0
        ),
        average_support_distance_atr=(
            sum(support_distances) / len(support_distances)
            if support_distances else 0.0
        ),
    )


def _numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _assert_training_dates(dates) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError(
            "Horizontal-structure attribution contains dates outside 2019-2021 training window"
        )


def format_horizontal_structure(report: HorizontalStructureReport) -> str:
    lines = [
        "Cross-signal horizontal structure attribution (2019-2021)",
        "LEVEL prior_20_valid_bars_ending_T-2 near_distance=1.0ATR",
    ]
    for section_name, section in (
        ("PRESSURE", report.by_pressure),
        ("SUPPORT", report.by_support),
        ("YEAR_PRESSURE", report.by_year_pressure),
        ("MILD_YEAR_PRESSURE", report.mild_by_year_pressure),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                (
                    "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} "
                    "pl={} resistance={:.2f}ATR support={:.2f}ATR"
                ).format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_resistance_distance_atr,
                    item.average_support_distance_atr,
                )
            )
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    print(format_horizontal_structure(run_training_horizontal_structure()))


if __name__ == "__main__":
    main()
