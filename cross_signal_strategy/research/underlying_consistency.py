# -*- coding: utf-8 -*-
"""Observation-only QDII underlying-index direction attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from cross_signal_strategy.research.trade_quality_ledger import TradeQualityRow
from cross_signal_strategy.research.underlying_market_data import (
    UNDERLYING_SPECS,
    select_underlying_direction,
)


TRAINING_START = pd.Timestamp("2019-01-01")
TRAINING_END = pd.Timestamp("2021-12-31")


@dataclass(frozen=True)
class UnderlyingTradeObservation:
    code: str
    buy_date: str
    group: str
    underlying_source_id: str
    underlying_previous_session: str
    underlying_latest_session: str
    underlying_return: float
    realized_return_pct: float
    holding_mfe: float
    holding_mae: float
    first_atr_barrier: str


@dataclass(frozen=True)
class DirectionStats:
    count: int = 0
    win_rate: float = 0.0
    mean_return_pct: float = 0.0
    mean_holding_mfe: float = 0.0
    mean_holding_mae: float = 0.0
    up_first_rate: float = 0.0
    down_first_rate: float = 0.0


@dataclass(frozen=True)
class UnderlyingCandidateGate:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnderlyingConsistencyReport:
    targeted_trades: int
    covered_trades: int
    missing_trades: int
    observations: tuple[UnderlyingTradeObservation, ...]
    aggregate: Mapping[str, DirectionStats]
    by_year: Mapping[int, Mapping[str, DirectionStats]]
    by_code: Mapping[str, Mapping[str, DirectionStats]]
    gate: UnderlyingCandidateGate

    @property
    def coverage_rate(self) -> float:
        return self.covered_trades / self.targeted_trades if self.targeted_trades else 0.0


def _stats(items: Sequence[UnderlyingTradeObservation]) -> DirectionStats:
    count = len(items)
    if not count:
        return DirectionStats()
    return DirectionStats(
        count=count,
        win_rate=sum(item.realized_return_pct > 0 for item in items) / count,
        mean_return_pct=sum(item.realized_return_pct for item in items) / count,
        mean_holding_mfe=sum(item.holding_mfe for item in items) / count,
        mean_holding_mae=sum(item.holding_mae for item in items) / count,
        up_first_rate=sum(item.first_atr_barrier == "up_first" for item in items) / count,
        down_first_rate=sum(item.first_atr_barrier == "down_first" for item in items) / count,
    )


def _group_stats(
    observations: Sequence[UnderlyingTradeObservation],
) -> Dict[str, DirectionStats]:
    return {
        group: _stats([item for item in observations if item.group == group])
        for group in ("confirmed", "unconfirmed")
    }


def evaluate_underlying_candidate_gate(
    targeted_trades: int,
    covered_trades: int,
    aggregate: Mapping[str, DirectionStats],
    by_year: Mapping[int, Mapping[str, DirectionStats]],
    by_code: Mapping[str, Mapping[str, DirectionStats]],
) -> UnderlyingCandidateGate:
    """Apply the frozen sample, annual, and cross-ETF candidate gate."""
    reasons = []
    coverage = covered_trades / targeted_trades if targeted_trades else 0.0
    if coverage < 0.90:
        reasons.append("point-in-time coverage is below 90%")
    if covered_trades < 30:
        reasons.append("fewer than 30 covered QDII trades")

    confirmed = aggregate.get("confirmed", DirectionStats())
    unconfirmed = aggregate.get("unconfirmed", DirectionStats())
    if confirmed.count < 10 or unconfirmed.count < 10:
        reasons.append("confirmed and unconfirmed groups each require at least 10 trades")
    if confirmed.mean_return_pct <= unconfirmed.mean_return_pct:
        reasons.append("confirmed aggregate average return is not higher")
    if confirmed.win_rate <= unconfirmed.win_rate:
        reasons.append("confirmed aggregate win rate is not higher")

    for year in (2019, 2020, 2021):
        groups = by_year.get(year, {})
        year_confirmed = groups.get("confirmed", DirectionStats())
        year_unconfirmed = groups.get("unconfirmed", DirectionStats())
        if year_confirmed.count < 3 or year_unconfirmed.count < 3:
            reasons.append("%d requires at least 3 trades in each group" % year)
            continue
        if year_confirmed.mean_return_pct <= year_unconfirmed.mean_return_pct:
            reasons.append("%d confirmed average return is not higher" % year)
        if year_confirmed.win_rate <= year_unconfirmed.win_rate:
            reasons.append("%d confirmed win rate is not higher" % year)

    comparable_codes = []
    for code, groups in by_code.items():
        code_confirmed = groups.get("confirmed", DirectionStats())
        code_unconfirmed = groups.get("unconfirmed", DirectionStats())
        if code_confirmed.count < 2 or code_unconfirmed.count < 2:
            continue
        comparable_codes.append(code)
        if (
            code_confirmed.mean_return_pct <= code_unconfirmed.mean_return_pct
            and code_confirmed.win_rate <= code_unconfirmed.win_rate
        ):
            reasons.append("%s confirmed group is not better on either primary metric" % code)
    if len(comparable_codes) < 3:
        reasons.append("fewer than three ETF codes have two trades in each group")
    return UnderlyingCandidateGate(passed=not reasons, reasons=tuple(reasons))


def build_underlying_consistency_report(
    quality_rows: Iterable[TradeQualityRow],
    history_lookup,
) -> UnderlyingConsistencyReport:
    """Attach the frozen point-in-time label to already completed QDII trades."""
    targeted = []
    for row in quality_rows:
        code = str(row.code).split(".")[0]
        if code not in UNDERLYING_SPECS:
            continue
        buy_date = pd.Timestamp(row.buy_date)
        if buy_date < TRAINING_START or buy_date > TRAINING_END:
            raise ValueError("trade row is outside the 2019-2021 training window")
        targeted.append((code, row))

    observations = []
    for code, row in targeted:
        decision_at = "%sT09:35:00+08:00" % pd.Timestamp(row.buy_date).strftime("%Y-%m-%d")
        try:
            history = history_lookup(code, decision_at)
        except (FileNotFoundError, KeyError):
            continue
        if history is None or history.empty:
            continue
        direction = select_underlying_direction(history, code=code, decision_at=decision_at)
        if direction is None:
            continue
        observations.append(UnderlyingTradeObservation(
            code=code,
            buy_date=pd.Timestamp(row.buy_date).strftime("%Y-%m-%d"),
            group="confirmed" if direction.confirmed else "unconfirmed",
            underlying_source_id=direction.source_id,
            underlying_previous_session=direction.previous_session_date,
            underlying_latest_session=direction.latest_session_date,
            underlying_return=direction.one_session_return,
            realized_return_pct=float(row.realized_return_pct),
            holding_mfe=float(row.holding_mfe),
            holding_mae=float(row.holding_mae),
            first_atr_barrier=str(row.first_atr_barrier),
        ))

    observation_tuple = tuple(observations)
    aggregate = _group_stats(observation_tuple)
    by_year = {
        year: _group_stats(tuple(
            item for item in observation_tuple
            if pd.Timestamp(item.buy_date).year == year
        ))
        for year in (2019, 2020, 2021)
    }
    by_code = {
        code: _group_stats(tuple(
            item for item in observation_tuple if item.code == code
        ))
        for code in UNDERLYING_SPECS
    }
    gate = evaluate_underlying_candidate_gate(
        targeted_trades=len(targeted),
        covered_trades=len(observation_tuple),
        aggregate=aggregate,
        by_year=by_year,
        by_code=by_code,
    )
    return UnderlyingConsistencyReport(
        targeted_trades=len(targeted),
        covered_trades=len(observation_tuple),
        missing_trades=len(targeted) - len(observation_tuple),
        observations=observation_tuple,
        aggregate=aggregate,
        by_year=by_year,
        by_code=by_code,
        gate=gate,
    )
