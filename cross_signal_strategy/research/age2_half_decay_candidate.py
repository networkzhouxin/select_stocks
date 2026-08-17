# -*- coding: utf-8 -*-
"""Isolated buy-side candidate that halves age-2 bullish-cross weights."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


_BULLISH_CROSS_WEIGHTS = {
    "rsi6_cross_rsi12_up": 12.0,
    "rsi6_cross_rsi24_up": 12.0,
    "macd_cross_up": 10.0,
    "kdj_k_cross_up": 6.0,
    "kdj_j_cross_up": 5.0,
}
_TRAINING_YEARS = (2019, 2020, 2021)
_TRAINING_START = "2019-01-01"
_TRAINING_END = "2021-12-31"


def _rsi_group_direction(snapshot: dict[str, Any]) -> str | None:
    rsi_up = bool(
        snapshot.get("rsi6_cross_rsi12_up")
        or snapshot.get("rsi6_cross_rsi24_up")
    )
    rsi_down = bool(
        snapshot.get("rsi6_cross_rsi12_down")
        or snapshot.get("rsi6_cross_rsi24_down")
    )
    if rsi_up and not rsi_down:
        return "up"
    if rsi_down and not rsi_up:
        return "down"
    return None


def _active_cross_age(snapshot: dict[str, Any], flag: str) -> int | None:
    if not snapshot.get(flag):
        return None
    age_key = f"{flag}_age"
    if age_key not in snapshot or snapshot[age_key] is None:
        raise ValueError(f"active bullish cross requires {age_key}")
    age = snapshot[age_key]
    if isinstance(age, bool) or age not in (0, 1, 2):
        raise ValueError(f"{age_key} must be one of 0, 1, 2")
    return int(age)


def _age2_penalty(snapshot: dict[str, Any]) -> float:
    ages = {
        flag: _active_cross_age(snapshot, flag)
        for flag in _BULLISH_CROSS_WEIGHTS
    }
    rsi_contributes = _rsi_group_direction(snapshot) == "up"
    penalty = 0.0
    for flag, weight in _BULLISH_CROSS_WEIGHTS.items():
        if flag.startswith("rsi6_") and not rsi_contributes:
            continue
        if ages[flag] == 2:
            penalty += weight * 0.5
    return penalty


@dataclass(frozen=True)
class Age2HalfDecaySignalAdapter:
    """Decorate the official adapter without changing its T-1 data path."""

    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        base_result = self.source.score(
            code,
            current_date,
            return_reason=return_reason,
        )
        if return_reason:
            base_score, reason = base_result
            if base_score is None:
                return None, reason
            return self._adjust(base_score), reason
        if base_result is None:
            return None
        return self._adjust(base_result)

    def _adjust(self, base_score: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(base_score)
        official_reversal = float(result["reversal_score"])
        official_buy = float(result["buy_score"])
        penalty = _age2_penalty(result)
        candidate_reversal = official_reversal - penalty
        candidate_buy = max(
            0.0,
            candidate_reversal
            + float(result["location_score"])
            + float(result["trend_score"])
            + float(result["volume_score"]),
        )
        result["official_reversal_score"] = official_reversal
        result["official_buy_score"] = official_buy
        result["age2_half_decay_penalty"] = penalty
        result["reversal_score"] = candidate_reversal
        result["buy_score"] = candidate_buy
        return result


@dataclass(frozen=True)
class Age2HalfDecayPerformance:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    win_rate: float
    profit_loss_ratio: float | None
    buy_count: int
    sell_count: int
    annual_returns: Dict[int, float]


@dataclass(frozen=True)
class Age2HalfDecayGateDecision:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangedOrderDecision:
    date: str
    baseline_orders: tuple[tuple[str, str, str], ...]
    candidate_orders: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class FilledOrderPathComparison:
    changed_order_days: int
    changed_days_by_year: Dict[int, int]
    decisions: tuple[ChangedOrderDecision, ...]


@dataclass(frozen=True)
class Age2HalfDecayComparisonReport:
    baseline_report: object | None
    candidate_report: object | None
    baseline: Age2HalfDecayPerformance
    candidate: Age2HalfDecayPerformance
    path: FilledOrderPathComparison
    gate: Age2HalfDecayGateDecision


def evaluate_age2_half_decay_gate(
    baseline: Age2HalfDecayPerformance,
    candidate: Age2HalfDecayPerformance,
    changed_days_by_year: Mapping[int, int],
) -> Age2HalfDecayGateDecision:
    """Apply the pre-registered strict 2019-2021 adoption gate."""

    reasons = []
    for year in _TRAINING_YEARS:
        if int(changed_days_by_year.get(year, 0)) <= 0:
            reasons.append(f"{year} has no changed filled-order day")
    if candidate.total_return <= baseline.total_return:
        reasons.append("candidate total return does not improve")
    if candidate.annualized_return <= baseline.annualized_return:
        reasons.append("candidate annualized return does not improve")
    if candidate.max_drawdown > baseline.max_drawdown:
        reasons.append("candidate maximum drawdown worsens")
    if not _ratio_not_worse(candidate.sharpe_ratio, baseline.sharpe_ratio):
        reasons.append("candidate Sharpe ratio worsens")
    if not _ratio_not_worse(candidate.sortino_ratio, baseline.sortino_ratio):
        reasons.append("candidate Sortino ratio worsens")
    if candidate.win_rate < baseline.win_rate:
        reasons.append("candidate win rate worsens")
    if not _ratio_not_worse(
        candidate.profit_loss_ratio,
        baseline.profit_loss_ratio,
    ):
        reasons.append("candidate profit/loss ratio worsens")
    for year in _TRAINING_YEARS:
        baseline_return = baseline.annual_returns.get(year)
        candidate_return = candidate.annual_returns.get(year)
        if baseline_return is None or candidate_return is None:
            reasons.append(f"{year} annual return is missing")
        elif candidate_return < baseline_return:
            reasons.append(f"{year} candidate annual return worsens")
    return Age2HalfDecayGateDecision(not reasons, tuple(reasons))


def compare_filled_order_paths(
    baseline_days: Sequence[object],
    candidate_days: Sequence[object],
) -> FilledOrderPathComparison:
    """Compare only filled decisions on identical 2019-2021 replay dates."""

    baseline_dates = [str(day.date) for day in baseline_days]
    candidate_dates = [str(day.date) for day in candidate_days]
    if baseline_dates != candidate_dates:
        raise ValueError("A/B comparison requires identical trading dates")
    _assert_training_dates(baseline_dates)

    changed_days_by_year: Dict[int, int] = {}
    decisions = []
    for baseline_day, candidate_day in zip(baseline_days, candidate_days):
        baseline_orders = _filled_order_signature(baseline_day)
        candidate_orders = _filled_order_signature(candidate_day)
        if baseline_orders == candidate_orders:
            continue
        date = str(baseline_day.date)
        year = int(date[:4])
        changed_days_by_year[year] = changed_days_by_year.get(year, 0) + 1
        decisions.append(ChangedOrderDecision(
            date=date,
            baseline_orders=baseline_orders,
            candidate_orders=candidate_orders,
        ))
    return FilledOrderPathComparison(
        changed_order_days=sum(changed_days_by_year.values()),
        changed_days_by_year=dict(sorted(changed_days_by_year.items())),
        decisions=tuple(decisions),
    )


def _filled_order_signature(day: object) -> tuple[tuple[str, str, str], ...]:
    signature = []
    for order in getattr(day, "orders", []):
        if not getattr(order, "filled", False):
            continue
        amount = int(getattr(order, "amount_delta", 0))
        side = "buy" if amount > 0 else "sell" if amount < 0 else "flat"
        signature.append((
            str(order.code).split(".")[0],
            side,
            str(getattr(order, "reason", "")),
        ))
    return tuple(sorted(signature))


def _ratio_not_worse(candidate: float | None, baseline: float | None) -> bool:
    if baseline is None:
        return True
    return candidate is not None and float(candidate) >= float(baseline)


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(date < _TRAINING_START or date > _TRAINING_END for date in dates):
        raise ValueError("A/B comparison contains dates outside 2019-2021 training window")


def format_age2_half_decay_comparison(
    report: Age2HalfDecayComparisonReport,
) -> str:
    """Render the frozen experiment and its one-way adoption decision."""

    lines = [
        "# Cross-signal bullish-cross age-decay experiment",
        "",
        "- Scope: approved 2019-2021 training replay; 2018 is warm-up only.",
        "- Hypothesis: an age 2 bullish cross is less timely than age 0/1.",
        "- Frozen change: keep age 0/1 at full official weight and multiply only "
        "contributing age 2 bullish RSI12/RSI24/MACD/KDJ-K/KDJ-J weights by 0.5.",
        "- Sell rules and every other strategy rule remain unchanged.",
        "",
        "## Performance",
        "",
        "| Arm | Total return | Annualized | Max drawdown | Sharpe | Sortino | "
        "Win rate | P/L ratio | Buys | Sells |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _performance_row("Baseline", report.baseline),
        _performance_row("Candidate", report.candidate),
        "",
        "Annual returns:",
        "",
        f"- Baseline: {_annual_text(report.baseline.annual_returns)}",
        f"- Candidate: {_annual_text(report.candidate.annual_returns)}",
        "",
        "## Filled-order path",
        "",
        f"- Changed filled-order days: {report.path.changed_order_days}",
        f"- By year: {_count_text(report.path.changed_days_by_year)}",
        "",
        "## Frozen gate",
        "",
        f"- Decision: {'PASS' if report.gate.passed else 'REJECT'}",
    ]
    if report.gate.reasons:
        lines.extend(f"- Failure: {reason}" for reason in report.gate.reasons)
    lines.extend([
        "",
        "## Interpretation and next action",
        "",
    ])
    if report.gate.passed:
        lines.append(
            "The local gate passed. The only permitted next action is to generate a "
            "separate JoinQuant candidate for the user's 2019-2021 authority backtest."
        )
    else:
        lines.append(
            "The local gate failed. Reject this candidate, record the failed experiment, "
            "and do not generate a JoinQuant candidate or tune a replacement rule."
        )
    return "\n".join(lines) + "\n"


def _performance_row(label: str, item: Age2HalfDecayPerformance) -> str:
    return (
        f"| {label} | {item.total_return:.2%} | {item.annualized_return:.2%} | "
        f"{item.max_drawdown:.2%} | {_format_ratio(item.sharpe_ratio)} | "
        f"{_format_ratio(item.sortino_ratio)} | {item.win_rate:.2%} | "
        f"{_format_ratio(item.profit_loss_ratio)} | {item.buy_count} | "
        f"{item.sell_count} |"
    )


def _format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _annual_text(annual_returns: Mapping[int, float]) -> str:
    return ", ".join(
        f"{year}: {annual_returns[year]:.2%}"
        for year in sorted(annual_returns)
    )


def _count_text(counts: Mapping[int, int]) -> str:
    return ", ".join(f"{year}: {counts.get(year, 0)}" for year in _TRAINING_YEARS)
