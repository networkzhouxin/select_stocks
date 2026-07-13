# -*- coding: utf-8 -*-
"""Training-only T-2-safe controlled-breakout extension diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, Iterable, Mapping

import pandas as pd

from cross_signal_strategy.horizontal_structure_diagnostics import (
    STRUCTURE_PERIOD,
    calc_horizontal_structure,
)
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
RSI6_EXTENSION = 75.0
MA20_EXTENSION = 0.10
TRAILING_PERIODS = (5, 10, 20)
VALID_LABELS = {
    "controlled_breakout",
    "extended_breakout",
    "no_breakout",
    "no_data",
}


def classify_breakout_extension(pressure_bucket, rsi6, close, ma20) -> str:
    pressure = str(pressure_bucket)
    if pressure == "no_data":
        return "no_data"
    if pressure != "breakout":
        return "no_breakout"

    rsi_value = _optional_float(rsi6)
    close_value = _optional_float(close)
    ma20_value = _optional_float(ma20)
    rsi_known = rsi_value is not None
    ma_known = close_value is not None and ma20_value is not None and ma20_value > 0
    if rsi_known and rsi_value >= RSI6_EXTENSION:
        return "extended_breakout"
    if ma_known and close_value / ma20_value - 1.0 >= MA20_EXTENSION:
        return "extended_breakout"
    if rsi_known and ma_known:
        return "controlled_breakout"
    return "no_data"


@dataclass
class BreakoutExtensionSignalAdapter:
    """Attach fixed extension diagnostics without changing official scores."""

    source: object
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
                        "breakout extension signal_date does not match base score"
                    )
                enriched = build_breakout_extension_score(
                    frame=frame,
                    base_score=base_score,
                    signal_date=score_signal_date,
                )
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


def build_breakout_extension_score(
    frame: pd.DataFrame,
    base_score: Mapping[str, object],
    signal_date: str,
) -> dict:
    score = dict(base_score)
    structure = calc_horizontal_structure(
        frame=frame,
        signal_date=str(signal_date),
        atr=_numeric(score.get("atr"), float("nan")),
        period=STRUCTURE_PERIOD,
    )
    close = _optional_float(score.get("close"))
    ma20 = _optional_float(score.get("ma20"))
    label = classify_breakout_extension(
        structure.pressure_bucket,
        score.get("rsi6"),
        close,
        ma20,
    )
    trailing = build_trailing_diagnostics(
        frame=frame,
        signal_date=str(signal_date),
        support=structure.support,
    )
    score.update(trailing)
    score.update({
        "breakout_extension_label": label,
        "breakout_extension_blocked": False,
        "breakout_rsi6": _optional_float(score.get("rsi6")),
        "breakout_ma20_distance": (
            close / ma20 - 1.0
            if close is not None and ma20 is not None and ma20 > 0
            else None
        ),
        "breakout_pressure_bucket": structure.pressure_bucket,
        "breakout_structure_period": STRUCTURE_PERIOD,
        "breakout_rsi6_extension": RSI6_EXTENSION,
        "breakout_ma20_extension": MA20_EXTENSION,
        "breakout_signal_date": structure.signal_date,
        "breakout_level_data_date": structure.level_data_date,
    })
    return score


def build_trailing_diagnostics(
    frame: pd.DataFrame,
    signal_date: str,
    support: float,
) -> dict:
    required = {"date", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "breakout extension missing columns: %s" % ", ".join(missing)
        )
    visible = frame.copy()
    visible["_date"] = pd.to_datetime(visible["date"], errors="coerce")
    if visible["_date"].isna().any():
        raise ValueError("breakout extension frame contains invalid dates")
    signal_ts = pd.Timestamp(signal_date)
    if not visible.empty and visible["_date"].max() > signal_ts:
        raise ValueError("breakout extension frame contains data after signal_date")
    visible = visible.loc[visible["_date"] <= signal_ts].sort_values("_date")
    signal_rows = visible.loc[visible["_date"] == signal_ts]
    values = {
        "breakout_return_5": None,
        "breakout_return_10": None,
        "breakout_return_20": None,
        "breakout_rise_from_low": None,
    }
    if signal_rows.empty:
        return values

    closes = pd.to_numeric(visible["close"], errors="coerce").dropna()
    if closes.empty:
        return values
    latest = float(closes.iloc[-1])
    for period in TRAILING_PERIODS:
        prior_index = len(closes) - period - 1
        if prior_index < 0:
            continue
        prior_close = float(closes.iloc[prior_index])
        if prior_close > 0:
            values["breakout_return_%d" % period] = latest / prior_close - 1.0
    support_value = _optional_float(support)
    if support_value is not None and support_value > 0:
        values["breakout_rise_from_low"] = latest / support_value - 1.0
    return values


@dataclass(frozen=True)
class BreakoutExtensionStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_rsi6: float = 0.0
    average_ma20_distance: float = 0.0
    average_return_5: float = 0.0
    average_return_10: float = 0.0
    average_return_20: float = 0.0
    average_rise_from_low: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None


@dataclass(frozen=True)
class BreakoutObservationGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BreakoutExtensionReport:
    by_label: Dict[str, BreakoutExtensionStats]
    by_year_label: Dict[str, BreakoutExtensionStats]
    gate: BreakoutObservationGateDecision


def evaluate_observation_gate(
    controlled_by_year: Mapping[int, BreakoutExtensionStats],
    extended_by_year: Mapping[int, BreakoutExtensionStats],
) -> BreakoutObservationGateDecision:
    reasons = []
    controlled_total = sum(
        controlled_by_year.get(year, BreakoutExtensionStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    extended_total = sum(
        extended_by_year.get(year, BreakoutExtensionStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if controlled_total < 6:
        reasons.append("controlled breakout subset has fewer than 6 closed trades")
    if extended_total < 6:
        reasons.append("extended breakout subset has fewer than 6 closed trades")
    for year in (2019, 2020, 2021):
        controlled = controlled_by_year.get(year, BreakoutExtensionStats())
        extended = extended_by_year.get(year, BreakoutExtensionStats())
        if controlled.closed_trades < 2:
            reasons.append(
                "%d controlled breakout subset has fewer than 2 closed trades" % year
            )
        if extended.closed_trades < 2:
            reasons.append(
                "%d extended breakout subset has fewer than 2 closed trades" % year
            )
        if controlled.closed_trades and extended.closed_trades:
            if extended.average_return >= controlled.average_return:
                reasons.append(
                    "%d extended breakouts do not underperform average return" % year
                )
            if extended.win_rate >= controlled.win_rate:
                reasons.append(
                    "%d extended breakouts do not underperform win rate" % year
                )
    return BreakoutObservationGateDecision(
        passed=not reasons,
        reasons=tuple(reasons),
    )


def build_breakout_extension_report(
    trades: Iterable[ClosedTradeDiagnostic],
) -> BreakoutExtensionReport:
    items = list(trades)
    _assert_training_dates([
        date
        for trade in items
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])
    by_label = _group_stats(items, lambda trade: _label(trade.entry_score))
    by_year_label = _group_stats(
        items,
        lambda trade: "%s:%s" % (
            str(trade.buy_date)[:4],
            _label(trade.entry_score),
        ),
    )
    controlled = {
        year: _stats([
            trade
            for trade in items
            if str(trade.buy_date).startswith(str(year))
            and _label(trade.entry_score) == "controlled_breakout"
        ])
        for year in (2019, 2020, 2021)
    }
    extended = {
        year: _stats([
            trade
            for trade in items
            if str(trade.buy_date).startswith(str(year))
            and _label(trade.entry_score) == "extended_breakout"
        ])
        for year in (2019, 2020, 2021)
    }
    return BreakoutExtensionReport(
        by_label=by_label,
        by_year_label=by_year_label,
        gate=evaluate_observation_gate(controlled, extended),
    )


def run_training_breakout_extension_observation(
    loader=None,
    initial_cash: float = 20000.0,
) -> BreakoutExtensionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    source = build_training_signal_adapter(loader)
    adapter = BreakoutExtensionSignalAdapter(source)
    planner = DiagnosticOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_breakout_extension_report(trades)


def _label(score: Mapping[str, object]) -> str:
    value = str(score.get("breakout_extension_label", "no_data"))
    return value if value in VALID_LABELS else "no_data"


def _group_stats(trades, key_fn) -> Dict[str, BreakoutExtensionStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {key: _stats(group) for key, group in sorted(grouped.items())}


def _stats(trades) -> BreakoutExtensionStats:
    items = list(trades)
    return BreakoutExtensionStats(
        closed_trades=len(items),
        wins=sum(1 for trade in items if float(trade.pnl) > 0),
        losses=sum(1 for trade in items if float(trade.pnl) < 0),
        realized_pnl=sum(float(trade.pnl) for trade in items),
        gross_profit=sum(float(trade.pnl) for trade in items if float(trade.pnl) > 0),
        gross_loss=sum(abs(float(trade.pnl)) for trade in items if float(trade.pnl) < 0),
        average_return=_average([
            float(trade.return_pct) / 100.0 for trade in items
        ]),
        average_rsi6=_score_average(items, "breakout_rsi6"),
        average_ma20_distance=_score_average(items, "breakout_ma20_distance"),
        average_return_5=_score_average(items, "breakout_return_5"),
        average_return_10=_score_average(items, "breakout_return_10"),
        average_return_20=_score_average(items, "breakout_return_20"),
        average_rise_from_low=_score_average(items, "breakout_rise_from_low"),
    )


def _score_average(trades, field_name: str) -> float:
    return _average([
        value
        for trade in trades
        for value in [_optional_float(trade.entry_score.get(field_name))]
        if value is not None
    ])


def _average(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _numeric(value, default=0.0) -> float:
    number = _optional_float(value)
    return number if number is not None else float(default)


def _assert_training_dates(dates) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError(
            "Breakout-extension attribution contains dates outside 2019-2021 training window"
        )


def format_breakout_extension(report: BreakoutExtensionReport) -> str:
    lines = [
        "Cross-signal controlled-breakout extension observation (2019-2021)",
        "RULE prior_20_valid_bars_ending_T-2 RSI6>=75 OR close/MA20-1>=10%",
    ]
    for section_name, section in (
        ("LABEL", report.by_label),
        ("YEAR", report.by_year_label),
    ):
        for key, item in section.items():
            ratio = item.profit_loss_ratio
            lines.append(
                (
                    "{} {} trades={} pnl={:.2f} avg_ret={:.2%} win={:.2%} "
                    "pl={} rsi={:.2f} ma20={:.2%} r5={:.2%} r10={:.2%} "
                    "r20={:.2%} rise_low={:.2%}"
                ).format(
                    section_name,
                    key,
                    item.closed_trades,
                    item.realized_pnl,
                    item.average_return,
                    item.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    item.average_rsi6,
                    item.average_ma20_distance,
                    item.average_return_5,
                    item.average_return_10,
                    item.average_return_20,
                    item.average_rise_from_low,
                )
            )
    lines.append("OBSERVATION_GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def main() -> None:
    report = run_training_breakout_extension_observation()
    print(format_breakout_extension(report))


if __name__ == "__main__":
    main()
