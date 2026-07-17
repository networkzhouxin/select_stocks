# -*- coding: utf-8 -*-
"""Training-only Chaikin Money Flow diagnostics for cross-signal trades."""

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


CMF_PERIOD = 20


def calc_cmf(frame: pd.DataFrame, period: int = CMF_PERIOD) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    price_range = high - low
    multiplier = ((2.0 * close - high - low) / price_range.replace(0.0, float("nan"))).fillna(0.0)
    flow_volume = multiplier * volume
    rolling_volume = volume.rolling(int(period), min_periods=int(period)).sum()
    return (
        flow_volume.rolling(int(period), min_periods=int(period)).sum()
        / rolling_volume.replace(0.0, float("nan"))
    )


@dataclass
class CmfSignalAdapter:
    source: object
    period: int = CMF_PERIOD
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
                    raise ValueError("CMF signal_date does not match base score")
                max_data_date = str(pd.to_datetime(frame["date"]).max().date())
                if max_data_date > score_signal_date:
                    raise ValueError("CMF frame contains data after signal_date")
                cmf = calc_cmf(frame, period=self.period)
                enriched = dict(base_score)
                enriched["cmf20"] = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else float("nan")
                enriched["cmf_period"] = int(self.period)
                enriched["cmf_data_date"] = max_data_date
                self._cache[key] = (enriched, None)

        cached_score, cached_reason = self._cache[key]
        result = dict(cached_score) if cached_score is not None else None
        return (result, cached_reason) if return_reason else result


@dataclass(frozen=True)
class CmfGroupStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    atr_stops: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None

    @property
    def atr_stop_rate(self) -> float:
        return self.atr_stops / self.closed_trades if self.closed_trades else 0.0


@dataclass(frozen=True)
class CmfAttributionReport:
    by_sign: Dict[str, CmfGroupStats]
    by_trend_sign: Dict[str, CmfGroupStats]
    by_year_sign: Dict[str, CmfGroupStats]


def build_cmf_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
) -> CmfAttributionReport:
    items = list(trades)
    return CmfAttributionReport(
        by_sign=_group_stats(items, lambda trade: _cmf_sign(trade.entry_score)),
        by_trend_sign=_group_stats(
            items,
            lambda trade: "%s:%s" % (
                _trend_group(trade.entry_score),
                _cmf_sign(trade.entry_score),
            ),
        ),
        by_year_sign=_group_stats(
            items,
            lambda trade: "%s:%s" % (
                str(trade.buy_date)[:4],
                _cmf_sign(trade.entry_score),
            ),
        ),
    )


def run_training_cmf_attribution(
    loader=None,
    initial_cash: float = 20000.0,
) -> CmfAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    base_adapter = build_training_signal_adapter(loader)
    cmf_adapter = CmfSignalAdapter(base_adapter, period=CMF_PERIOD)
    planner = DiagnosticOrderPlanner(cmf_adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    trades = build_closed_trade_diagnostics(
        results,
        planner.entry_score_snapshots,
        planner.exit_score_snapshots,
    )
    return build_cmf_attribution(trades)


def _group_stats(trades, key_fn) -> Dict[str, CmfGroupStats]:
    grouped: Dict[str, list[ClosedTradeDiagnostic]] = {}
    for trade in trades:
        grouped.setdefault(str(key_fn(trade)), []).append(trade)
    return {
        key: _stats(items)
        for key, items in sorted(grouped.items())
    }


def _stats(trades: list[ClosedTradeDiagnostic]) -> CmfGroupStats:
    return CmfGroupStats(
        closed_trades=len(trades),
        wins=sum(1 for trade in trades if trade.pnl > 0),
        losses=sum(1 for trade in trades if trade.pnl < 0),
        realized_pnl=sum(float(trade.pnl) for trade in trades),
        gross_profit=sum(float(trade.pnl) for trade in trades if trade.pnl > 0),
        gross_loss=sum(abs(float(trade.pnl)) for trade in trades if trade.pnl < 0),
        atr_stops=sum(1 for trade in trades if trade.sell_reason == "atr_stop"),
    )


def _cmf_sign(score: Mapping[str, object]) -> str:
    try:
        value = float(score.get("cmf20"))
    except (TypeError, ValueError):
        return "unknown"
    if pd.isna(value):
        return "unknown"
    return "positive" if value > 0 else "non_positive"


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


def format_cmf_attribution(report: CmfAttributionReport) -> str:
    lines = ["Cross-signal CMF(20) training attribution (2019-2021)"]
    for section_name, section in (
        ("SIGN", report.by_sign),
        ("TREND", report.by_trend_sign),
        ("YEAR", report.by_year_sign),
    ):
        for key, stats in section.items():
            ratio = stats.profit_loss_ratio
            lines.append(
                "{} {} trades={} pnl={:.2f} win={:.2%} pl={} atr_stop={:.2%}".format(
                    section_name,
                    key,
                    stats.closed_trades,
                    stats.realized_pnl,
                    stats.win_rate,
                    "n/a" if ratio is None else "%.3f" % ratio,
                    stats.atr_stop_rate,
                )
            )
    return "\n".join(lines)


def main() -> None:
    print(format_cmf_attribution(run_training_cmf_attribution()))


if __name__ == "__main__":
    main()
