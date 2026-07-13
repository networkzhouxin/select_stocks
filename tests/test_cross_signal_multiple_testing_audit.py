# -*- coding: utf-8 -*-
"""Tests for training-only multiple-testing risk diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIT_DOC = ROOT / "cross_signal_strategy" / "docs" / "multiple_testing_audit.md"
DECISIONS_DOC = ROOT / "cross_signal_strategy" / "docs" / "decisions.md"


def days_from_returns(start_value, dated_returns):
    from cross_signal_strategy.local_backtester import DayResult

    value = float(start_value)
    days = []
    for date, daily_return in dated_returns:
        value *= 1.0 + float(daily_return)
        days.append(DayResult(
            date=date,
            previous_date=None,
            orders=[],
            cash=value,
            positions={},
            marks={},
            total_value=value,
        ))
    return days


def positive_training_path():
    returns = [
        ("2019-01-02", 0.010),
        ("2019-01-03", -0.002),
        ("2019-01-04", 0.008),
        ("2020-01-02", 0.004),
        ("2020-01-03", -0.001),
        ("2020-01-06", 0.006),
        ("2021-01-04", 0.003),
        ("2021-01-05", -0.002),
        ("2021-01-06", 0.005),
    ]
    return days_from_returns(10000.0, returns)


def test_audit_treats_failed_ledger_plus_selected_mainline_as_trial_lower_bound():
    from cross_signal_strategy.multiple_testing_audit import build_multiple_testing_audit

    report = build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=47,
    )

    assert report.failed_experiment_count == 47
    assert report.selected_mainline_count == 1
    assert report.minimum_trial_count == 48
    assert report.trial_count_is_lower_bound is True


def test_audit_applies_bonferroni_to_psr_without_calling_it_canonical_dsr():
    from cross_signal_strategy.multiple_testing_audit import build_multiple_testing_audit

    report = build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=47,
    )

    assert report.observed_daily_sharpe > 0
    assert report.observed_annualized_sharpe > 0
    assert 0.0 < report.single_trial_p_value < 1.0
    assert report.bonferroni_p_value_at_min_trials == pytest.approx(
        min(1.0, report.single_trial_p_value * 48)
    )
    assert report.selection_adjusted_confidence_upper_bound == pytest.approx(
        1.0 - report.bonferroni_p_value_at_min_trials
    )
    assert report.canonical_dsr is None
    assert "candidate Sharpe" in report.canonical_dsr_status
    assert report.pbo is None
    assert "aligned candidate daily return" in report.pbo_status
    assert report.hac_lag >= 1
    assert 0.0 <= report.hac_single_trial_p_value <= 1.0
    assert report.hac_bonferroni_p_value_at_min_trials == pytest.approx(
        min(1.0, report.hac_single_trial_p_value * 48)
    )
    assert report.maximum_trials_passing_five_percent == int(
        0.05 // report.single_trial_p_value
    )


def test_more_unrecorded_trials_can_only_weaken_the_reported_selection_confidence():
    from cross_signal_strategy.multiple_testing_audit import build_multiple_testing_audit

    lower = build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=47,
    )
    higher = build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=99,
    )

    assert higher.minimum_trial_count == 100
    assert higher.bonferroni_p_value_at_min_trials >= lower.bonferroni_p_value_at_min_trials
    assert (
        higher.selection_adjusted_confidence_upper_bound
        <= lower.selection_adjusted_confidence_upper_bound
    )


def test_audit_reports_each_training_year_without_using_other_dates():
    from cross_signal_strategy.multiple_testing_audit import build_multiple_testing_audit

    report = build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=47,
    )

    assert set(report.annual_returns) == {2019, 2020, 2021}
    assert all(value > 0 for value in report.annual_returns.values())

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_multiple_testing_audit(
            days_from_returns(10000.0, [("2022-01-04", 0.01)]),
            initial_cash=10000.0,
            failed_experiment_count=47,
        )


def test_audit_format_labels_evidence_limits_and_not_out_of_sample_validation():
    from cross_signal_strategy.multiple_testing_audit import (
        build_multiple_testing_audit,
        format_multiple_testing_audit,
    )

    text = format_multiple_testing_audit(build_multiple_testing_audit(
        positive_training_path(),
        initial_cash=10000.0,
        failed_experiment_count=47,
    ))

    assert "minimum trial count=48" in text
    assert "canonical DSR=unavailable" in text
    assert "PBO=unavailable" in text
    assert "Newey-West/HAC" in text
    assert "maximum trials passing 5%=" in text
    assert "not an out-of-sample validation" in text


def test_repository_records_audit_result_and_evidence_limits():
    audit_text = AUDIT_DOC.read_text(encoding="utf-8")
    decisions_text = DECISIONS_DOC.read_text(encoding="utf-8")

    assert "minimum trial count: 48" in audit_text
    assert "0.00595144" in audit_text
    assert "0.00298564" in audit_text
    assert "maximum trials passing the 5% PSR/Bonferroni approximation: 403" in audit_text
    assert "Canonical DSR: unavailable" in audit_text
    assert "PBO: unavailable" in audit_text
    assert "not out-of-sample validation" in audit_text
    assert "Record The Multiple-Testing Audit Without Changing Strategy" in decisions_text
