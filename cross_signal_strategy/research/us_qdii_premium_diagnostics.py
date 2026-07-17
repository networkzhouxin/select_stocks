# -*- coding: utf-8 -*-
"""Training-only US-QDII previous-NAV premium attribution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Dict, Iterable, Mapping

from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.research.trade_diagnostics import (
    ClosedTradeDiagnostic,
    run_training_trade_diagnostics,
)


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
TARGET_CODES = ("513100", "513500")
BOUNDARY_EPSILON = 1e-12


@dataclass(frozen=True)
class PremiumTradeStats:
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_return: float = 0.0
    average_premium: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    @property
    def profit_loss_ratio(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return None


@dataclass(frozen=True)
class PremiumGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class UsQdiiPremiumReport:
    targeted_trades: int
    covered_trades: int
    missing_trades: int
    by_bucket: Dict[str, PremiumTradeStats]
    by_year_bucket: Dict[str, PremiumTradeStats]
    by_code_bucket: Dict[str, PremiumTradeStats]
    gate: PremiumGateDecision

    @property
    def coverage_rate(self) -> float:
        return self.covered_trades / self.targeted_trades if self.targeted_trades else 0.0


@dataclass(frozen=True)
class _PremiumObservation:
    trade: ClosedTradeDiagnostic
    premium: float
    bucket: str


def premium_bucket(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    if isnan(numeric):
        return "missing"
    if numeric <= 0.02 + BOUNDARY_EPSILON:
        return "at_most_2"
    if numeric <= 0.05 + BOUNDARY_EPSILON:
        return "2_to_5"
    if numeric <= 0.10 + BOUNDARY_EPSILON:
        return "5_to_10"
    return "above_10"


def build_us_qdii_premium_report(
    trades: Iterable[ClosedTradeDiagnostic],
    reference_lookup,
) -> UsQdiiPremiumReport:
    targeted = [
        trade for trade in trades
        if str(trade.code).split(".")[0] in TARGET_CODES
    ]
    _assert_training_dates([
        date
        for trade in targeted
        for date in (str(trade.buy_date), str(trade.sell_date))
    ])

    observations = []
    for trade in targeted:
        code = str(trade.code).split(".")[0]
        try:
            market_price, reference_nav = reference_lookup(code, str(trade.buy_date))
            market_price = float(market_price)
            reference_nav = float(reference_nav)
            premium = (
                market_price / reference_nav - 1.0
                if market_price > 0 and reference_nav > 0
                else float("nan")
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, ZeroDivisionError):
            premium = float("nan")
        observations.append(_PremiumObservation(
            trade=trade,
            premium=premium,
            bucket=premium_bucket(premium),
        ))

    covered = [item for item in observations if item.bucket != "missing"]
    elevated_by_year = {
        year: _stats([
            item for item in covered
            if str(item.trade.buy_date).startswith(str(year)) and item.premium > 0.05
        ])
        for year in (2019, 2020, 2021)
    }
    normal_by_year = {
        year: _stats([
            item for item in covered
            if str(item.trade.buy_date).startswith(str(year)) and item.premium <= 0.05
        ])
        for year in (2019, 2020, 2021)
    }
    elevated_by_code = {
        code: _stats([
            item for item in covered
            if str(item.trade.code).split(".")[0] == code and item.premium > 0.05
        ])
        for code in TARGET_CODES
    }
    return UsQdiiPremiumReport(
        targeted_trades=len(observations),
        covered_trades=len(covered),
        missing_trades=len(observations) - len(covered),
        by_bucket=_group_stats(observations, lambda item: item.bucket),
        by_year_bucket=_group_stats(
            observations,
            lambda item: "%s:%s" % (str(item.trade.buy_date)[:4], item.bucket),
        ),
        by_code_bucket=_group_stats(
            observations,
            lambda item: "%s:%s" % (
                str(item.trade.code).split(".")[0],
                item.bucket,
            ),
        ),
        gate=evaluate_premium_candidate_gate(
            targeted_trades=len(observations),
            covered_trades=len(covered),
            elevated_by_year=elevated_by_year,
            normal_by_year=normal_by_year,
            elevated_by_code=elevated_by_code,
        ),
    )


def evaluate_premium_candidate_gate(
    targeted_trades: int,
    covered_trades: int,
    elevated_by_year: Mapping[int, PremiumTradeStats],
    normal_by_year: Mapping[int, PremiumTradeStats],
    elevated_by_code: Mapping[str, PremiumTradeStats],
) -> PremiumGateDecision:
    reasons = []
    coverage = covered_trades / targeted_trades if targeted_trades else 0.0
    if coverage < 0.80:
        reasons.append("premium coverage is below 80%")

    elevated_total = sum(
        elevated_by_year.get(year, PremiumTradeStats()).closed_trades
        for year in (2019, 2020, 2021)
    )
    if elevated_total < 10:
        reasons.append("above-5% subset has fewer than 10 closed trades")

    qualifying_years = [
        year for year in (2019, 2020, 2021)
        if elevated_by_year.get(year, PremiumTradeStats()).closed_trades >= 3
    ]
    if len(qualifying_years) < 2:
        reasons.append("above-5% subset lacks two years with at least 3 trades")
    for year in qualifying_years:
        elevated = elevated_by_year.get(year, PremiumTradeStats())
        normal = normal_by_year.get(year, PremiumTradeStats())
        if normal.closed_trades <= 0:
            reasons.append("%d has no at-or-below-5%% comparison trades" % year)
            continue
        if elevated.average_return >= normal.average_return:
            reasons.append("%d above-5%% average return does not underperform" % year)
        if elevated.win_rate >= normal.win_rate:
            reasons.append("%d above-5%% win rate does not underperform" % year)

    for code in TARGET_CODES:
        if elevated_by_code.get(code, PremiumTradeStats()).closed_trades < 3:
            reasons.append("%s has fewer than 3 above-5%% trades" % code)
    return PremiumGateDecision(passed=not reasons, reasons=tuple(reasons))


def run_training_us_qdii_premium(
    loader=None,
    trades: Iterable[ClosedTradeDiagnostic] | None = None,
) -> UsQdiiPremiumReport:
    loader = loader or CrossSignalTrainingDataLoader()
    trade_items = list(trades) if trades is not None else run_training_trade_diagnostics(loader)

    def reference_lookup(code: str, date: str) -> tuple[float, float]:
        row = loader.get_minute_bar(code, date, "09:35")
        return float(row["close"]), float(row["iopv"])

    return build_us_qdii_premium_report(trade_items, reference_lookup)


def _group_stats(observations, key_fn) -> Dict[str, PremiumTradeStats]:
    grouped: Dict[str, list[_PremiumObservation]] = {}
    for item in observations:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    return {key: _stats(items) for key, items in sorted(grouped.items())}


def _stats(observations) -> PremiumTradeStats:
    items = list(observations)
    premiums = [item.premium for item in items if not isnan(item.premium)]
    return PremiumTradeStats(
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
        average_premium=(sum(premiums) / len(premiums) if premiums else 0.0),
    )


def _assert_training_dates(dates) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Premium diagnostics contain dates outside 2019-2021 training window")


def format_us_qdii_premium_report(report: UsQdiiPremiumReport) -> str:
    lines = [
        "US-QDII 09:35 previous-NAV premium diagnostics (2019-2021)",
        "targeted=%d covered=%d missing=%d coverage=%.2f%%" % (
            report.targeted_trades,
            report.covered_trades,
            report.missing_trades,
            report.coverage_rate * 100.0,
        ),
    ]
    for key, item in report.by_bucket.items():
        lines.append(_format_stats("BUCKET", key, item))
    for key, item in report.by_year_bucket.items():
        lines.append(_format_stats("YEAR", key, item))
    for key, item in report.by_code_bucket.items():
        lines.append(_format_stats("CODE", key, item))
    lines.append("GATE passed=%s" % report.gate.passed)
    lines.extend("GATE_REASON %s" % reason for reason in report.gate.reasons)
    return "\n".join(lines)


def _format_stats(section: str, key: str, item: PremiumTradeStats) -> str:
    ratio = item.profit_loss_ratio
    return (
        "%s %s trades=%d pnl=%.2f avg_ret=%.2f%% win=%.2f%% pl=%s premium=%.2f%%"
        % (
            section,
            key,
            item.closed_trades,
            item.realized_pnl,
            item.average_return * 100.0,
            item.win_rate * 100.0,
            "n/a" if ratio is None else "%.3f" % ratio,
            item.average_premium * 100.0,
        )
    )


def main() -> None:
    print(format_us_qdii_premium_report(run_training_us_qdii_premium()))


if __name__ == "__main__":
    main()
