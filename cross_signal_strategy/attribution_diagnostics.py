# -*- coding: utf-8 -*-
"""ETF-level attribution diagnostics for cross-signal training trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_training_run import get_training_trade_dates
from cross_signal_strategy.trade_diagnostics import (
    ClosedTradeDiagnostic,
    run_training_trade_diagnostics,
)


@dataclass(frozen=True)
class EtfAttributionStats:
    code: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_holding_days: float = 0.0
    atr_stop_count: int = 0
    signal_sell_count: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None

    @property
    def atr_stop_rate(self) -> float:
        return self.atr_stop_count / self.closed_trades if self.closed_trades else 0.0

    @property
    def signal_sell_rate(self) -> float:
        return self.signal_sell_count / self.closed_trades if self.closed_trades else 0.0


@dataclass(frozen=True)
class EtfAttributionReport:
    by_code: Dict[str, EtfAttributionStats] = field(default_factory=dict)
    total_realized_pnl: float = 0.0


@dataclass(frozen=True)
class EntrySignalComboStats:
    combo_key: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None

    @property
    def average_pnl(self) -> float:
        return self.realized_pnl / self.closed_trades if self.closed_trades else 0.0


@dataclass(frozen=True)
class EntryBucketStats:
    dimension: str
    bucket: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def average_pnl(self) -> float:
        return self.realized_pnl / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else None


@dataclass
class _MutableStats:
    code: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    holding_days_sum: float = 0.0
    atr_stop_count: int = 0
    signal_sell_count: int = 0

    def add(self, trade: ClosedTradeDiagnostic, holding_days: int) -> None:
        self.closed_trades += 1
        self.realized_pnl += float(trade.pnl)
        self.holding_days_sum += float(holding_days)
        if trade.pnl > 0:
            self.wins += 1
            self.gross_profit += float(trade.pnl)
        elif trade.pnl < 0:
            self.losses += 1
            self.gross_loss += abs(float(trade.pnl))
        if str(trade.sell_reason) == "atr_stop":
            self.atr_stop_count += 1
        elif str(trade.sell_reason) == "signal_sell":
            self.signal_sell_count += 1

    def freeze(self) -> EtfAttributionStats:
        return EtfAttributionStats(
            code=self.code,
            closed_trades=self.closed_trades,
            wins=self.wins,
            losses=self.losses,
            realized_pnl=self.realized_pnl,
            gross_profit=self.gross_profit,
            gross_loss=self.gross_loss,
            average_holding_days=(
                self.holding_days_sum / self.closed_trades
                if self.closed_trades else 0.0
            ),
            atr_stop_count=self.atr_stop_count,
            signal_sell_count=self.signal_sell_count,
        )


@dataclass
class _MutableComboStats:
    combo_key: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def add(self, trade: ClosedTradeDiagnostic) -> None:
        pnl = float(trade.pnl)
        self.closed_trades += 1
        self.realized_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += abs(pnl)

    def freeze(self) -> EntrySignalComboStats:
        return EntrySignalComboStats(
            combo_key=self.combo_key,
            closed_trades=self.closed_trades,
            wins=self.wins,
            losses=self.losses,
            realized_pnl=self.realized_pnl,
            gross_profit=self.gross_profit,
            gross_loss=self.gross_loss,
        )


@dataclass
class _MutableBucketStats:
    dimension: str
    bucket: str
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def add(self, trade: ClosedTradeDiagnostic) -> None:
        pnl = float(trade.pnl)
        self.closed_trades += 1
        self.realized_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += abs(pnl)

    def freeze(self) -> EntryBucketStats:
        return EntryBucketStats(
            dimension=self.dimension,
            bucket=self.bucket,
            closed_trades=self.closed_trades,
            wins=self.wins,
            losses=self.losses,
            realized_pnl=self.realized_pnl,
            gross_profit=self.gross_profit,
            gross_loss=self.gross_loss,
        )


def entry_signal_tags(entry_score: Mapping[str, object]) -> tuple[str, ...]:
    tags = []
    if entry_score.get("rsi6_cross_rsi12_up") or entry_score.get("rsi6_cross_rsi24_up"):
        tags.append("rsi_up")
    if entry_score.get("macd_cross_up"):
        tags.append("macd_up")
    if entry_score.get("kdj_k_cross_up") or entry_score.get("kdj_j_cross_up"):
        tags.append("kdj_up")
    if _numeric(entry_score.get("location_score")) == 17:
        tags.append("low_location")
    if _numeric(entry_score.get("trend_score")) >= 20:
        tags.append("strong_trend")
    elif _numeric(entry_score.get("trend_score")) > 0:
        tags.append("trend_support")
    if _numeric(entry_score.get("volume_score")) > 0:
        tags.append("volume_confirmed")
    return tuple(sorted(tags)) if tags else ("unclassified",)


def entry_combo_key(entry_score: Mapping[str, object]) -> str:
    return "+".join(entry_signal_tags(entry_score))


def summarize_entry_signal_combos(
    trades: Iterable[ClosedTradeDiagnostic],
) -> Dict[str, EntrySignalComboStats]:
    mutable: Dict[str, _MutableComboStats] = {}
    for trade in trades:
        key = entry_combo_key(trade.entry_score)
        mutable.setdefault(key, _MutableComboStats(combo_key=key)).add(trade)
    return {
        key: item.freeze()
        for key, item in sorted(
            mutable.items(),
            key=lambda entry: (-entry[1].realized_pnl, entry[0]),
        )
    }


def entry_bucket_labels(trade: ClosedTradeDiagnostic) -> Dict[str, str]:
    score = trade.entry_score
    return {
        "etf_class": etf_class(str(trade.code)),
        "buy_score_band": _score_band(_numeric(score.get("buy_score"))),
        "rsi6_band": _rsi_band(_numeric_or_none(score.get("rsi6"))),
        "location_bucket": _location_bucket(_numeric(score.get("location_score"))),
        "trend_bucket": _trend_bucket(_numeric(score.get("trend_score"))),
        "volume_bucket": "volume_confirmed" if _numeric(score.get("volume_score")) > 0 else "no_volume",
        "sell_conflict": (
            "sell_conflict"
            if _numeric(score.get("sell_score")) >= 30
            else "no_sell_conflict"
        ),
        "ma20_distance": _ma20_distance_bucket(score),
        "boll_position": _boll_position_bucket(score),
    }


def summarize_entry_buckets(
    trades: Iterable[ClosedTradeDiagnostic],
) -> Dict[str, Dict[str, EntryBucketStats]]:
    mutable: Dict[str, Dict[str, _MutableBucketStats]] = {}
    for trade in trades:
        for dimension, bucket in entry_bucket_labels(trade).items():
            dimension_stats = mutable.setdefault(dimension, {})
            dimension_stats.setdefault(
                bucket,
                _MutableBucketStats(dimension=dimension, bucket=bucket),
            ).add(trade)

    return {
        dimension: {
            bucket: item.freeze()
            for bucket, item in sorted(
                buckets.items(),
                key=lambda entry: (entry[1].realized_pnl, entry[0]),
            )
        }
        for dimension, buckets in sorted(mutable.items())
    }


def etf_class(code: str) -> str:
    normalized = str(code).split(".")[0]
    if normalized in {"510300", "159915", "512100", "159928", "510880"}:
        return "a_share"
    if normalized in {"513100", "513500", "513880", "513050", "159920"}:
        return "cross_market"
    if normalized in {"518880", "159985", "511010"}:
        return "cross_asset"
    return "other"


def _numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _numeric_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_band(score: float) -> str:
    if score >= 70:
        return "70+"
    if score >= 60:
        return "60_69"
    return "below_60"


def _rsi_band(rsi: float | None) -> str:
    if rsi is None:
        return "unknown"
    if rsi >= 75:
        return "overheated"
    if rsi >= 55:
        return "warm"
    if rsi >= 35:
        return "neutral"
    return "oversold"


def _location_bucket(location_score: float) -> str:
    if location_score >= 15:
        return "low_or_mid"
    if location_score >= 7:
        return "middle_repair"
    return "high_or_extended"


def _trend_bucket(trend_score: float) -> str:
    if trend_score >= 20:
        return "strong_trend"
    if trend_score > 0:
        return "mild_trend"
    if trend_score < 0:
        return "weak_trend"
    return "flat_trend"


def _ma20_distance_bucket(score: Mapping[str, object]) -> str:
    close = _numeric_or_none(score.get("close"))
    ma20 = _numeric_or_none(score.get("ma20"))
    if close is None or ma20 is None or ma20 <= 0:
        return "unknown"
    distance = close / ma20 - 1.0
    if distance >= 0.10:
        return "far_above_ma20"
    if distance >= 0.03:
        return "above_ma20"
    if distance >= -0.03:
        return "near_ma20"
    return "below_ma20"


def _boll_position_bucket(score: Mapping[str, object]) -> str:
    close = _numeric_or_none(score.get("close"))
    upper = _first_numeric(score, "boll_upper", "upper")
    middle = _first_numeric(score, "boll_mid", "middle")
    lower = _first_numeric(score, "boll_lower", "lower")
    if close is None or upper is None or middle is None or lower is None:
        return "unknown"
    if close >= upper:
        return "above_upper"
    if close >= middle:
        return "upper_half"
    if close >= lower:
        return "lower_half"
    return "below_lower"


def _first_numeric(score: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _numeric_or_none(score.get(key))
        if value is not None:
            return value
    return None


def build_etf_attribution(
    trades: Iterable[ClosedTradeDiagnostic],
    trade_dates: Sequence[str],
) -> EtfAttributionReport:
    date_index = {pd.Timestamp(day).strftime("%Y-%m-%d"): idx for idx, day in enumerate(trade_dates)}
    mutable: Dict[str, _MutableStats] = {}
    for trade in trades:
        code = str(trade.code).split(".")[0]
        stats = mutable.setdefault(code, _MutableStats(code=code))
        stats.add(trade, _holding_days(trade, date_index))

    by_code = {
        code: item.freeze()
        for code, item in sorted(
            mutable.items(),
            key=lambda entry: (-entry[1].realized_pnl, entry[0]),
        )
    }
    return EtfAttributionReport(
        by_code=by_code,
        total_realized_pnl=sum(item.realized_pnl for item in by_code.values()),
    )


def _holding_days(trade: ClosedTradeDiagnostic, date_index: Dict[str, int]) -> int:
    buy = pd.Timestamp(trade.buy_date).strftime("%Y-%m-%d")
    sell = pd.Timestamp(trade.sell_date).strftime("%Y-%m-%d")
    if buy not in date_index or sell not in date_index:
        return max(0, (pd.Timestamp(sell) - pd.Timestamp(buy)).days)
    return max(0, date_index[sell] - date_index[buy])


def run_training_etf_attribution(loader=None) -> EtfAttributionReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    trades = run_training_trade_diagnostics(loader=loader)
    return build_etf_attribution(trades, trade_dates)
