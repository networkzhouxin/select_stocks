# -*- coding: utf-8 -*-
"""Training-only audit of strategy-selection and multiple-testing risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, Sequence

from cross_signal_strategy.baseline_report import build_baseline_report
from cross_signal_strategy.local_backtester import LocalBacktestEngine
from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
from cross_signal_strategy.local_order_planner import LocalCrossSignalOrderPlanner
from cross_signal_strategy.local_training_run import (
    build_training_signal_adapter,
    get_training_trade_dates,
)
from cross_signal_strategy.research_budget import audit_research_budget


TRAINING_START = "2019-01-01"
TRAINING_END = "2021-12-31"
SELECTED_MAINLINE_COUNT = 1
DOCS_ROOT = Path(__file__).resolve().parent / "docs"
FAILED_EXPERIMENTS_PATH = DOCS_ROOT / "failed_experiments.md"
RESEARCH_BUDGET_PATH = DOCS_ROOT / "research_budget.json"


@dataclass(frozen=True)
class MultipleTestingAudit:
    start_date: str
    end_date: str
    trading_days: int
    total_return: float
    annualized_return: float
    annual_returns: Dict[int, float]
    failed_experiment_count: int
    selected_mainline_count: int
    minimum_trial_count: int
    trial_count_is_lower_bound: bool
    observed_daily_sharpe: float
    observed_annualized_sharpe: float
    return_skewness: float
    return_kurtosis: float
    single_trial_psr_vs_zero: float
    single_trial_p_value: float
    bonferroni_p_value_at_min_trials: float
    selection_adjusted_confidence_upper_bound: float
    passes_five_percent_at_min_trials: bool
    maximum_trials_passing_five_percent: int | None
    hac_lag: int
    hac_mean_t_stat: float
    hac_single_trial_p_value: float
    hac_bonferroni_p_value_at_min_trials: float
    hac_selection_adjusted_confidence_upper_bound: float
    hac_passes_five_percent_at_min_trials: bool
    canonical_dsr: float | None
    canonical_dsr_status: str
    pbo: float | None
    pbo_status: str


def build_multiple_testing_audit(
    results: Iterable[object],
    initial_cash: float,
    failed_experiment_count: int,
    periods_per_year: int = 244,
) -> MultipleTestingAudit:
    days = list(results)
    if not days:
        raise ValueError("Multiple-testing audit requires training results")
    _assert_training_dates([str(day.date) for day in days])
    failed_count = int(failed_experiment_count)
    if failed_count < 0:
        raise ValueError("failed_experiment_count must be non-negative")

    values = [float(day.total_value) for day in days]
    daily_returns = _daily_returns(values, float(initial_cash))
    daily_sharpe = _sharpe(daily_returns)
    skewness, kurtosis = _return_moments(daily_returns)
    psr = _probabilistic_sharpe_ratio(
        daily_sharpe,
        benchmark_sharpe=0.0,
        observations=len(daily_returns),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    single_p = 1.0 - psr
    minimum_trials = failed_count + SELECTED_MAINLINE_COUNT
    adjusted_p = min(1.0, single_p * minimum_trials)
    hac_lag, hac_t_stat, hac_single_p = _newey_west_mean_test(daily_returns)
    hac_adjusted_p = min(1.0, hac_single_p * minimum_trials)
    baseline = build_baseline_report(
        days,
        initial_cash=initial_cash,
        periods_per_year=periods_per_year,
    )

    return MultipleTestingAudit(
        start_date=str(days[0].date),
        end_date=str(days[-1].date),
        trading_days=len(days),
        total_return=float(baseline.total_return),
        annualized_return=float(baseline.annualized_return),
        annual_returns=_annual_returns(days, initial_cash),
        failed_experiment_count=failed_count,
        selected_mainline_count=SELECTED_MAINLINE_COUNT,
        minimum_trial_count=minimum_trials,
        trial_count_is_lower_bound=True,
        observed_daily_sharpe=daily_sharpe,
        observed_annualized_sharpe=daily_sharpe * math.sqrt(periods_per_year),
        return_skewness=skewness,
        return_kurtosis=kurtosis,
        single_trial_psr_vs_zero=psr,
        single_trial_p_value=single_p,
        bonferroni_p_value_at_min_trials=adjusted_p,
        selection_adjusted_confidence_upper_bound=1.0 - adjusted_p,
        passes_five_percent_at_min_trials=adjusted_p <= 0.05,
        maximum_trials_passing_five_percent=(
            int(0.05 / single_p) if single_p > 0 else None
        ),
        hac_lag=hac_lag,
        hac_mean_t_stat=hac_t_stat,
        hac_single_trial_p_value=hac_single_p,
        hac_bonferroni_p_value_at_min_trials=hac_adjusted_p,
        hac_selection_adjusted_confidence_upper_bound=1.0 - hac_adjusted_p,
        hac_passes_five_percent_at_min_trials=hac_adjusted_p <= 0.05,
        canonical_dsr=None,
        canonical_dsr_status=(
            "unavailable: the candidate Sharpe distribution for all tried variants "
            "was not retained"
        ),
        pbo=None,
        pbo_status=(
            "unavailable: aligned candidate daily return curves required by PBO "
            "were not retained"
        ),
    )


def run_training_multiple_testing_audit(
    loader=None,
    initial_cash: float = 20000.0,
) -> MultipleTestingAudit:
    ledger = audit_research_budget(
        FAILED_EXPERIMENTS_PATH,
        RESEARCH_BUDGET_PATH,
    )
    if ledger.errors:
        raise ValueError("Research-budget audit failed: %s" % "; ".join(ledger.errors))
    loader = loader or CrossSignalTrainingDataLoader()
    trade_dates = get_training_trade_dates(loader)
    adapter = build_training_signal_adapter(loader)
    planner = LocalCrossSignalOrderPlanner(adapter, trade_dates=trade_dates)
    engine = LocalBacktestEngine(loader=loader, initial_cash=initial_cash)
    results = engine.run(trade_dates, planner.plan_orders)
    return build_multiple_testing_audit(
        results,
        initial_cash=initial_cash,
        failed_experiment_count=ledger.failed_experiment_count,
    )


def _daily_returns(values: Sequence[float], initial_cash: float) -> list[float]:
    returns = []
    previous = float(initial_cash)
    for value in values:
        if previous <= 0:
            raise ValueError("Portfolio value must remain positive")
        returns.append(float(value) / previous - 1.0)
        previous = float(value)
    return returns


def _sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        raise ValueError("At least two daily returns are required")
    mean_return = sum(returns) / len(returns)
    variance = sum((item - mean_return) ** 2 for item in returns) / len(returns)
    if variance <= 0:
        raise ValueError("Daily return variance must be positive")
    return mean_return / math.sqrt(variance)


def _return_moments(returns: Sequence[float]) -> tuple[float, float]:
    mean_return = sum(returns) / len(returns)
    centered = [item - mean_return for item in returns]
    second = sum(item ** 2 for item in centered) / len(centered)
    if second <= 0:
        raise ValueError("Daily return variance must be positive")
    third = sum(item ** 3 for item in centered) / len(centered)
    fourth = sum(item ** 4 for item in centered) / len(centered)
    return third / (second ** 1.5), fourth / (second ** 2)


def _probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    if observations < 2:
        raise ValueError("At least two observations are required for PSR")
    variance_term = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2
    )
    if variance_term <= 0:
        raise ValueError("PSR variance term must be positive")
    statistic = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(variance_term)
    )
    return NormalDist().cdf(statistic)


def _newey_west_mean_test(returns: Sequence[float]) -> tuple[int, float, float]:
    observations = len(returns)
    if observations < 3:
        raise ValueError("At least three observations are required for HAC")
    lag = int(math.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0)))
    lag = max(1, min(observations - 1, lag))
    mean_return = sum(returns) / observations
    centered = [item - mean_return for item in returns]
    long_run_variance = sum(item ** 2 for item in centered) / observations
    for offset in range(1, lag + 1):
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, observations)
        ) / observations
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        raise ValueError("HAC long-run variance must be positive")
    standard_error = math.sqrt(long_run_variance / observations)
    statistic = mean_return / standard_error
    p_value = 1.0 - NormalDist().cdf(statistic)
    return lag, statistic, p_value


def _annual_returns(days: Sequence[object], initial_cash: float) -> Dict[int, float]:
    grouped: Dict[int, list[object]] = {}
    for day in days:
        grouped.setdefault(int(str(day.date)[:4]), []).append(day)
    annual = {}
    start_value = float(initial_cash)
    for year, year_days in sorted(grouped.items()):
        end_value = float(year_days[-1].total_value)
        annual[year] = end_value / start_value - 1.0
        start_value = end_value
    return annual


def _assert_training_dates(dates: Sequence[str]) -> None:
    if any(str(date) < TRAINING_START or str(date) > TRAINING_END for date in dates):
        raise ValueError("Multiple-testing audit contains dates outside 2019-2021 training window")


def format_multiple_testing_audit(report: MultipleTestingAudit) -> str:
    return "\n".join([
        "Cross-signal multiple-testing audit (2019-2021; not an out-of-sample validation)",
        "return={:.2%} annualized={:.2%} annualized_sharpe={:.3f}".format(
            report.total_return,
            report.annualized_return,
            report.observed_annualized_sharpe,
        ),
        "minimum trial count={} ({} failed/non-adopted + {} selected; lower bound)".format(
            report.minimum_trial_count,
            report.failed_experiment_count,
            report.selected_mainline_count,
        ),
        "single-trial PSR={:.6f} p={:.6g}".format(
            report.single_trial_psr_vs_zero,
            report.single_trial_p_value,
        ),
        "Bonferroni at minimum trials p={:.6g} confidence_upper_bound={:.6f} pass_5pct={}".format(
            report.bonferroni_p_value_at_min_trials,
            report.selection_adjusted_confidence_upper_bound,
            report.passes_five_percent_at_min_trials,
        ),
        "maximum trials passing 5%={} (PSR/Bonferroni approximation)".format(
            report.maximum_trials_passing_five_percent
        ),
        "Newey-West/HAC lag={} t={:.3f} single_p={:.6g} Bonferroni_p={:.6g} "
        "confidence_upper_bound={:.6f} pass_5pct={}".format(
            report.hac_lag,
            report.hac_mean_t_stat,
            report.hac_single_trial_p_value,
            report.hac_bonferroni_p_value_at_min_trials,
            report.hac_selection_adjusted_confidence_upper_bound,
            report.hac_passes_five_percent_at_min_trials,
        ),
        "annual_returns={}".format({
            year: round(value, 6) for year, value in report.annual_returns.items()
        }),
        "canonical DSR=unavailable | %s" % report.canonical_dsr_status,
        "PBO=unavailable | %s" % report.pbo_status,
    ])


def main() -> None:
    print(format_multiple_testing_audit(run_training_multiple_testing_audit()))


if __name__ == "__main__":
    main()
