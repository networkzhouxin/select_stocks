# -*- coding: utf-8 -*-
"""Tests for the training-only failed-year fragility atlas."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "cross_signal_strategy" / "docs"
LEDGER_PATH = DOCS_DIR / "failed_experiments.md"
BUDGET_PATH = DOCS_DIR / "research_budget.json"
ANNOTATIONS_PATH = DOCS_DIR / "failure_year_fragility_annotations.json"
REPORT_PATH = DOCS_DIR / "failure_year_fragility_atlas.md"
README_PATH = DOCS_DIR / "README.md"
DECISIONS_PATH = DOCS_DIR / "decisions.md"


def test_parser_excludes_template_and_preserves_all_real_ledger_entries():
    from cross_signal_strategy.research.failure_year_atlas import parse_failed_experiments
    from cross_signal_strategy.research.research_budget import load_research_budget

    records = parse_failed_experiments(LEDGER_PATH.read_text(encoding="utf-8"))
    budget = load_research_budget(BUDGET_PATH)

    assert len(records) == budget.expected_failed_experiment_count
    assert len({record.record_id for record in records}) == len(records)
    assert all(record.date.startswith("2026-") for record in records)
    assert all(record.experiment for record in records)


def test_annotations_are_auditable_and_never_infer_unreported_years():
    from cross_signal_strategy.research.failure_year_atlas import (
        load_annotations,
        parse_failed_experiments,
        validate_annotations,
    )

    records = parse_failed_experiments(LEDGER_PATH.read_text(encoding="utf-8"))
    annotations = load_annotations(ANNOTATIONS_PATH)
    validation = validate_annotations(records, annotations)

    assert validation.unknown_record_ids == ()
    assert validation.duplicate_record_ids == ()
    assert validation.invalid_years == ()
    assert validation.missing_evidence == ()
    assert all(annotation.failed_years for annotation in annotations)
    assert all(annotation.evidence for annotation in annotations)
    assert all(set(annotation.failed_years) <= {2019, 2020, 2021} for annotation in annotations)


def test_atlas_counts_only_explicit_annual_contradictions():
    from cross_signal_strategy.research.failure_year_atlas import (
        build_failure_year_atlas,
        load_annotations,
        parse_failed_experiments,
    )

    records = parse_failed_experiments(LEDGER_PATH.read_text(encoding="utf-8"))
    atlas = build_failure_year_atlas(records, load_annotations(ANNOTATIONS_PATH))

    assert atlas.total_experiments == len(records)
    assert atlas.annotated_experiments < atlas.total_experiments
    assert atlas.unreported_annual_experiments == (
        atlas.total_experiments - atlas.annotated_experiments
    )
    assert atlas.failed_year_counts[2021] > atlas.failed_year_counts[2019]
    assert atlas.failed_year_counts[2021] > atlas.failed_year_counts[2020]
    assert atlas.mainline_annual_returns == {
        2019: pytest.approx(0.3584),
        2020: pytest.approx(0.4974),
        2021: pytest.approx(0.0846),
    }
    reexpansion = next(
        item for item in atlas.annotated_records
        if "same-side-reexpansion-observation" in item.record.version
    )
    assert reexpansion.annotation.failed_years == (2021,)
    assert reexpansion.annotation.mechanism == "regime_reversal"
    breakeven = next(
        item for item in atlas.annotated_records
        if "entry-atr-breakeven-candidate" in item.record.version
    )
    assert breakeven.annotation.failed_years == (2020, 2021)
    assert breakeven.annotation.mechanism == "premature_exit"
    macd_free_kdj = next(
        item for item in atlas.annotated_records
        if "macd-free-kdj-exit-candidate" in item.record.version
    )
    assert macd_free_kdj.annotation.failed_years == (2019, 2020, 2021)
    assert macd_free_kdj.annotation.mechanism == "premature_exit"
    macd_fast_exit = next(
        item for item in atlas.annotated_records
        if "macd-fast-exit-candidate" in item.record.version
    )
    assert macd_fast_exit.annotation.failed_years == (2019, 2020, 2021)
    assert macd_fast_exit.annotation.mechanism == "premature_exit"
    kdj_only_exit = next(
        item for item in atlas.annotated_records
        if "kdj-only-exit-candidate" in item.record.version
    )
    assert kdj_only_exit.annotation.failed_years == (2019, 2020, 2021)
    assert kdj_only_exit.annotation.mechanism == "premature_exit"


def test_2020_minute_overlay_is_tail_execution_not_mainline_weakness():
    from cross_signal_strategy.research.failure_year_atlas import (
        build_failure_year_atlas,
        load_annotations,
        parse_failed_experiments,
    )

    records = parse_failed_experiments(LEDGER_PATH.read_text(encoding="utf-8"))
    atlas = build_failure_year_atlas(records, load_annotations(ANNOTATIONS_PATH))
    overlay = next(
        item for item in atlas.annotated_records
        if "intraday-execution-overlay-v1" in item.record.version
    )

    assert overlay.annotation.failed_years == (2020,)
    assert overlay.annotation.mechanism == "tail_execution"
    assert "119.45bp" in overlay.annotation.evidence
    assert atlas.mainline_annual_returns[2020] == max(atlas.mainline_annual_returns.values())


def test_report_states_governance_limits_and_next_independent_direction():
    from cross_signal_strategy.research.failure_year_atlas import (
        build_failure_year_atlas,
        format_failure_year_atlas,
        load_annotations,
        parse_failed_experiments,
    )

    records = parse_failed_experiments(LEDGER_PATH.read_text(encoding="utf-8"))
    text = format_failure_year_atlas(
        build_failure_year_atlas(records, load_annotations(ANNOTATIONS_PATH))
    )

    assert "只统计台账中明确记录的逐年反例" in text
    assert "不得据此修改策略" in text
    assert "2020 年不是正式主线弱年" in text
    assert "2021" in text
    assert "QDII 底层指数方向" in text
    assert "验证期" in text


def test_repository_contains_generated_failure_year_report():
    text = REPORT_PATH.read_text(encoding="utf-8")

    assert "# 上穿下穿策略失败年份脆弱性地图" in text
    assert "正式主线保持 `cross-v0.3.2`" in text
    assert "未读取验证期行情" in text


def test_repository_indexes_atlas_as_governance_not_strategy_permission():
    readme = README_PATH.read_text(encoding="utf-8")
    decisions = DECISIONS_PATH.read_text(encoding="utf-8")

    assert "failure_year_fragility_atlas.md" in readme
    assert "Record The Failure-Year Fragility Atlas Without Reopening Research" in decisions
    assert "does not authorize a strategy change" in decisions
