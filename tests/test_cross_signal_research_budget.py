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
    from cross_signal_strategy.research.research_budget import parse_failed_experiments

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
    from cross_signal_strategy.research.research_budget import audit_research_budget

    report = audit_research_budget(FAILED_EXPERIMENTS, BUDGET)

    assert report.failed_experiment_count == 56
    assert report.expected_failed_experiment_count == 56
    assert report.duplicate_experiments == ()
    assert report.errors == ()


def test_budget_freezes_exhausted_search_and_limits_open_families():
    from cross_signal_strategy.research.research_budget import load_research_budget

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
    assert all(family.max_new_experiments == 0 for family in budget.families if family.status != "open")


def test_experiment_gate_rejects_closed_unknown_and_multi_variant_searches(tmp_path):
    from cross_signal_strategy.research.research_budget import (
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
    horizontal = next(
        item for item in payload["families"]
        if item["key"] == "horizontal_price_structure"
    )
    horizontal.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    horizontal.pop("planned_experiment", None)
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
    controlled_breakout = next(
        item for item in payload["families"]
        if item["key"] == "controlled_breakout_anti_chase"
    )
    controlled_breakout.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    controlled_breakout.pop("planned_experiment", None)
    share_flow = next(
        item for item in payload["families"]
        if item["key"] == "etf_share_flow_shadow"
    )
    share_flow.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    share_flow.pop("planned_experiment", None)
    intraday = next(
        item for item in payload["families"]
        if item["key"] == "intraday_execution_overlay_v1"
    )
    intraday.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    intraday.pop("planned_experiment", None)
    reexpansion = next(
        item for item in payload["families"]
        if item["key"] == "same_side_reexpansion_user_authorized"
    )
    reexpansion.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    reexpansion.pop("planned_experiment", None)
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
    from cross_signal_strategy.research.research_budget import load_research_budget

    budget = load_research_budget(BUDGET)

    assert budget.strategy_scope == "cross_signal_strategy"
    assert budget.training_start == "2019-01-01"
    assert budget.training_end == "2021-12-31"
    assert budget.validation_tuning_forbidden is True
    assert budget.max_total_open_experiments == 0


def test_user_authorized_reexpansion_observation_is_consumed_and_rejected():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["same_side_reexpansion_user_authorized"]
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["primary_horizon"] == 5
    assert raw["descriptive_horizons"] == [1, 3, 10]
    assert raw["minimum_total_observations"] == 30
    assert raw["minimum_annual_observations"] == 5
    assert raw["candidate_action"] == "observation_only"
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["total_event_count"] == 2000
    assert raw["bullish_novel_5d_observations"] == 318
    assert raw["bullish_novel_5d_average_return"] == pytest.approx(0.0047)
    assert raw["bullish_novel_5d_win_rate"] == pytest.approx(0.5723)
    assert raw["bullish_cross_5d_average_return"] == pytest.approx(0.0062)
    assert raw["bullish_cross_5d_win_rate"] == pytest.approx(0.5932)
    assert raw["bearish_novel_5d_average_return"] == pytest.approx(0.0073)
    assert raw["bearish_novel_5d_directional_win_rate"] == pytest.approx(0.4068)
    assert raw["gate_passed"] is False
    assert raw["candidate_created"] is False
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False


def test_user_authorized_intraday_overlay_is_consumed_after_fixed_training_gate():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["intraday_execution_overlay_v1"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "intraday_execution_overlay_v1"
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["arrival_time"] == "09:35"
    assert raw["decision_interval_minutes"] == 5
    assert raw["decision_cycles"] == 6
    assert raw["fallback_time"] == "10:05"
    assert raw["eligible_side"] == "ordinary_buy_only"
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
    assert raw["eligible_orders"] == 92
    assert raw["matched_orders"] == 92
    assert raw["passive_limit_fills"] == 75
    assert raw["market_fallback_fills"] == 17
    assert raw["average_signed_improvement"] == pytest.approx(0.000263)
    assert raw["annual_average_signed_improvement"] == {
        "2019": pytest.approx(0.000102),
        "2020": pytest.approx(-0.000078),
        "2021": pytest.approx(0.000673),
    }
    assert raw["group_average_signed_improvement"] == {
        "non_qdii": pytest.approx(0.000412),
        "qdii": pytest.approx(0.000040),
    }
    assert raw["gate_passed"] is False
    assert raw["gate_reason"] == "2020 average execution price does not improve"
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


def test_user_authorized_cross_window_budget_is_consumed_after_fixed_matrix():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["cross_window_user_authorized"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "cross_window_user_authorized"
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["windows"] == [1, 2, 3, 4]
    assert raw["baseline_window"] == 3
    assert raw["selected_window"] == 3
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
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


def test_user_authorized_execution_time_budget_is_consumed_after_fixed_comparison():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["execution_time_user_authorized"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "execution_time_user_authorized"
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["execution_times"] == ["09:35", "10:00"]
    assert raw["baseline_time"] == "09:35"
    assert raw["selected_time"] == "09:35"
    assert raw["gate_passed"] is False
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False


def test_etf_share_flow_shadow_budget_is_closed_after_one_fixed_observation():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["etf_share_flow_shadow"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "etf_share_flow_shadow"
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["lookback_observations"] == 5
    assert raw["grouping"] == "positive_vs_non_positive"
    assert raw["candidate_action"] == "observation_only"
    assert raw["approved_root"] == (
        r"G:\financial\history_data\cross_signal_flow_train_2018_2021"
    )
    assert raw["eligible_codes"] == [
        "159915", "512100", "159928", "518880", "159985",
    ]
    assert raw["blocked_qdii_codes"] == [
        "513100", "513500", "513880", "513050",
    ]
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
    assert raw["comparable_closed_trades"] == 52
    assert raw["positive_closed_trades"] == 24
    assert raw["non_positive_closed_trades"] == 28
    assert raw["eligible_coverage"] == pytest.approx(1.0)
    assert raw["observation_gate_passed"] is False
    assert raw["annual_direction_consistent"] is False
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


def test_underlying_direction_budget_tracks_raw_staging_without_opening_experiment():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "underlying_market_direction"
    )
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "underlying_market_direction"
    )

    assert family.status == "blocked"
    assert raw["raw_values_ready"] is True
    assert raw["raw_source_staging_root"] == (
        r"G:\financial\history_data\cross_signal_underlying_staging_2018_2021"
    )
    assert raw["raw_source_codes"] == ["513100", "513500", "513050", "513880"]
    assert raw["blocked_availability_codes"] == ["513050", "513500"]
    assert raw["formal_root_created"] is False
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False


def test_user_authorized_controlled_breakout_budget_is_consumed_after_one_observation():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["controlled_breakout_anti_chase"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "controlled_breakout_anti_chase"
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["structure_period"] == 20
    assert raw["rsi6_extension"] == 75
    assert raw["ma20_extension"] == pytest.approx(0.10)
    assert raw["candidate_action"] == "reject_extended_breakout_only"
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
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


def test_user_authorized_horizontal_structure_budget_is_closed_after_one_observation():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["horizontal_price_structure"]

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


def test_user_authorized_macd_budget_is_consumed_after_one_fixed_variant():
    from cross_signal_strategy.research.research_budget import (
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


def test_user_authorized_entry_atr_breakeven_budget_is_consumed_and_rejected():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["entry_atr_breakeven_user_authorized"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(item for item in payload["families"] if item["key"] == family.key)

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["activation_atr"] == pytest.approx(1.0)
    assert raw["floor_return"] == pytest.approx(0.0)
    assert raw["candidate_gate_passed"] is False
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False


def test_etf_microstructure_budget_is_closed_after_registered_observation():
    from cross_signal_strategy.research.research_budget import (
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


def test_underlying_market_direction_is_preregistered_but_data_blocked():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["underlying_market_direction"]
    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(
        item for item in payload["families"]
        if item["key"] == "underlying_market_direction"
    )

    assert family.status == "blocked"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["approved_root"] == (
        r"G:\financial\history_data\cross_signal_underlying_train_2018_2021"
    )
    assert raw["eligible_codes"] == ["513100", "513500", "513050", "513880"]
    assert raw["source_ids"] == ["NDX", "SPX", "H30533", "N225"]
    assert raw["decision_time"] == "09:35 Asia/Shanghai"
    assert raw["grouping"] == "positive_vs_non_positive"
    assert raw["minimum_coverage"] == pytest.approx(0.90)
    assert raw["minimum_covered_trades"] == 30
    assert raw["minimum_group_trades"] == 10
    assert raw["minimum_annual_group_trades"] == 3
    assert raw["minimum_cross_etf_comparisons"] == 3
    assert raw["candidate_action"] == "observation_only"
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert evaluate_experiment_request(
        budget,
        family_key=family.key,
        planned_variants=1,
    ).allowed is False


def test_readable_research_map_matches_the_structured_budget():
    text = GUIDE.read_text(encoding="utf-8")

    assert "research_budget.json" in text
    assert "52" in text
    assert "cross-v0.3.2" in text
    assert "portfolio_dependence" in text
    assert "market_breadth" in text
    assert "indicator_enumeration" in text
    assert "macd_half_cycle_user_authorized" in text
    assert "horizontal_price_structure" in text
    assert "controlled_breakout_anti_chase" in text
    assert "etf_share_flow_shadow" in text
    assert "cross_window_user_authorized" in text
    assert "execution_time_user_authorized" in text
    assert "09:35" in text and "10:00" in text
    assert "positive_vs_non_positive" in text
    assert "52" in text
    assert "2019" in text and "2020" in text and "2021" in text
    assert "RSI6 >= 75" in text
    assert "MA20" in text and "10%" in text
    assert "MACD(6,13,5)" in text
    assert "不得" in text and "验证期" in text
