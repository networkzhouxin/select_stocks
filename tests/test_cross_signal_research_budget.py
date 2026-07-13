# -*- coding: utf-8 -*-
"""Tests for cross-signal research-budget governance."""

import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED_EXPERIMENTS = (
    ROOT / "cross_signal_strategy" / "docs" / "failed_experiments.md"
)
BUDGET = ROOT / "cross_signal_strategy" / "docs" / "research_budget.json"
GUIDE = ROOT / "cross_signal_strategy" / "docs" / "research_budget.md"


def test_parse_failed_experiments_requires_complete_core_fields():
    from cross_signal_strategy.research_budget import parse_failed_experiments

    text = """\
Date: 2026-07-11
Version: cross-v0.3.2
Experiment: Test one independent idea.
Why it failed: The annual gate failed.

Date: 2026-07-12
Version: cross-v0.3.2
Experiment: Test a second independent idea.
Why it failed: The sample was too small.
"""

    records = parse_failed_experiments(text)

    assert [record.experiment for record in records] == [
        "Test one independent idea.",
        "Test a second independent idea.",
    ]
    assert records[0].date == "2026-07-11"
    assert records[1].why_failed == "The sample was too small."

    with pytest.raises(ValueError, match="Why it failed"):
        parse_failed_experiments(
            "Date: 2026-07-11\nVersion: cross-v0.3.2\nExperiment: Incomplete.\n"
        )


def test_repository_budget_accounts_for_every_recorded_experiment():
    from cross_signal_strategy.research_budget import audit_research_budget

    report = audit_research_budget(FAILED_EXPERIMENTS, BUDGET)

    assert report.failed_experiment_count == 47
    assert report.expected_failed_experiment_count == 47
    assert report.duplicate_experiments == ()
    assert report.errors == ()


def test_budget_freezes_exhausted_search_and_limits_open_families():
    from cross_signal_strategy.research_budget import load_research_budget

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}

    assert families["indicator_enumeration"].status == "exhausted"
    assert families["threshold_and_period_search"].status == "exhausted"
    assert families["training_period_pool_selection"].status == "exhausted"
    assert families["portfolio_dependence"].status == "exhausted"
    assert families["market_breadth"].status == "exhausted"
    assert families["etf_microstructure"].status == "exhausted"
    assert families["etf_microstructure"].max_new_experiments == 0
    assert families["etf_microstructure"].planned_experiment is None

    open_families = [family for family in budget.families if family.status == "open"]
    assert open_families == []
    assert all(family.max_new_experiments == 1 for family in open_families)
    assert all(family.planned_experiment for family in open_families)
    assert all(family.max_new_experiments == 0 for family in budget.families if family.status != "open")


def test_experiment_gate_rejects_closed_unknown_and_multi_variant_searches(tmp_path):
    from cross_signal_strategy.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)

    exhausted_portfolio = evaluate_experiment_request(
        budget,
        family_key="portfolio_dependence",
        planned_variants=1,
    )
    exhausted_breadth = evaluate_experiment_request(
        budget,
        family_key="market_breadth",
        planned_variants=1,
    )
    closed = evaluate_experiment_request(
        budget,
        family_key="indicator_enumeration",
        planned_variants=1,
    )
    unknown = evaluate_experiment_request(
        budget,
        family_key="mystery_factor",
        planned_variants=1,
    )

    assert exhausted_portfolio.allowed is False
    assert "exhausted" in exhausted_portfolio.reason
    assert exhausted_breadth.allowed is False
    assert "exhausted" in exhausted_breadth.reason
    assert closed.allowed is False
    assert "exhausted" in closed.reason
    assert unknown.allowed is False
    assert "unknown research family" in unknown.reason

    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    payload["families"].append(dict(payload["families"][0]))
    invalid = tmp_path / "duplicate.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate research family"):
        load_research_budget(invalid)

    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    payload["max_total_open_experiments"] = 1
    microstructure = next(
        item for item in payload["families"] if item["key"] == "etf_microstructure"
    )
    microstructure.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    microstructure.pop("planned_experiment", None)
    macd = next(
        item for item in payload["families"]
        if item["key"] == "macd_half_cycle_user_authorized"
    )
    macd.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    macd.pop("planned_experiment", None)
    market = next(item for item in payload["families"] if item["key"] == "market_breadth")
    market.update({
        "status": "open",
        "max_new_experiments": 1,
        "planned_experiment": "One fixed representative.",
    })
    synthetic_open = tmp_path / "open.json"
    synthetic_open.write_text(json.dumps(payload), encoding="utf-8")
    mined = evaluate_experiment_request(
        load_research_budget(synthetic_open),
        family_key="market_breadth",
        planned_variants=2,
    )
    assert mined.allowed is False
    assert "one pre-registered variant" in mined.reason


def test_budget_is_training_only_and_forbids_validation_tuning():
    from cross_signal_strategy.research_budget import load_research_budget

    budget = load_research_budget(BUDGET)

    assert budget.strategy_scope == "cross_signal_strategy"
    assert budget.training_start == "2019-01-01"
    assert budget.training_end == "2021-12-31"
    assert budget.validation_tuning_forbidden is True
    assert budget.max_total_open_experiments == 0


def test_user_authorized_macd_budget_is_consumed_after_one_fixed_variant():
    from cross_signal_strategy.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["macd_half_cycle_user_authorized"]

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=2,
    ).allowed is False


def test_etf_microstructure_budget_is_closed_after_registered_observation():
    from cross_signal_strategy.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)

    allowed = evaluate_experiment_request(
        budget,
        family_key="etf_microstructure",
        planned_variants=1,
    )
    rejected = evaluate_experiment_request(
        budget,
        family_key="etf_microstructure",
        planned_variants=2,
    )

    assert allowed.allowed is False
    assert "exhausted" in allowed.reason
    assert rejected.allowed is False


def test_readable_research_map_matches_the_structured_budget():
    text = GUIDE.read_text(encoding="utf-8")

    assert "research_budget.json" in text
    assert "47" in text
    assert "cross-v0.3.2" in text
    assert "portfolio_dependence" in text
    assert "market_breadth" in text
    assert "indicator_enumeration" in text
    assert "macd_half_cycle_user_authorized" in text
    assert "MACD(6,13,5)" in text
    assert "不得" in text and "验证期" in text
