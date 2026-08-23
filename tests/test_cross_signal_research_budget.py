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
DUAL_TIMEPOINT_REPORT = (
    ROOT
    / "cross_signal_strategy"
    / "reports"
    / "dual_timepoint_1445_2019_2021.md"
)


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

    assert report.failed_experiment_count == 77
    assert report.expected_failed_experiment_count == 77
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
    assert families["sell_score_rebalance_user_authorized"].status == "exhausted"
    assert families["kdj_ranking_only_buy_user_authorized"].status == "exhausted"

    open_families = [family for family in budget.families if family.status == "open"]
    assert [family.key for family in open_families] == [
        "dimension_capped_score_v04_user_authorized"
    ]
    assert budget.max_total_open_experiments == 1
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
    high_anchor = next(
        item for item in payload["families"]
        if item["key"] == "intraday_high_anchor_user_authorized"
    )
    high_anchor.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    high_anchor.pop("planned_experiment", None)
    profit_gated = next(
        item for item in payload["families"]
        if item["key"] == "profit_gated_direct_sell_user_authorized"
    )
    profit_gated.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    profit_gated.pop("planned_experiment", None)
    signal_clock = next(
        item for item in payload["families"]
        if item["key"] == "intraday_signal_clock_1445_user_authorized"
    )
    signal_clock.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    signal_clock.pop("planned_experiment", None)
    late_filter = next(
        item for item in payload["families"]
        if item["key"] == "late_macd_boll_upper_filter_user_authorized"
    )
    late_filter.update({
        "status": "blocked",
        "max_new_experiments": 0,
    })
    late_filter.pop("planned_experiment", None)
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
    assert budget.max_total_open_experiments == 1


def test_user_authorized_1445_signal_clock_cannot_be_reopened_after_consumption():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["intraday_signal_clock_1445_user_authorized"]

    assert budget.max_total_open_experiments == 1
    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False
    assert [item.key for item in budget.families if item.status == "open"] == [
        "dimension_capped_score_v04_user_authorized"
    ]

    raw = json.loads(BUDGET.read_text(encoding="utf-8"))
    payload = next(item for item in raw["families"] if item["key"] == family.key)
    assert payload["decision_times"] == ["09:35", "14:45"]
    assert payload["signal_cutoff"] == "14:44"
    assert payload["candidate_variants"] == 1
    assert payload["validation_influence"] == "none"
    assert payload["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert payload["prohibit_alternatives"] is True


def test_user_authorized_1445_signal_clock_is_consumed_and_rejected():
    from cross_signal_strategy.research.research_budget import load_research_budget

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["intraday_signal_clock_1445_user_authorized"]
    raw_family = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )
    report_text = DUAL_TIMEPOINT_REPORT.read_text(encoding="utf-8")
    report_gate_passed = "ELIGIBLE_FOR_JOINQUANT_PLAN" in report_text

    assert budget.max_total_open_experiments == 1
    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw_family["candidate_gate_passed"] is report_gate_passed
    assert raw_family["candidate_created"] is False
    assert raw_family["validation_influence"] == "none"
    assert raw_family["prohibit_alternatives"] is True
    assert raw_family["baseline_total_return"] == pytest.approx(1.250025)
    assert raw_family["candidate_total_return"] == pytest.approx(0.84997)
    assert raw_family["baseline_max_drawdown"] == pytest.approx(0.0603157868)
    assert raw_family["candidate_max_drawdown"] == pytest.approx(0.0749189964)
    assert raw_family["baseline_win_rate"] == pytest.approx(0.5617977528)
    assert raw_family["candidate_win_rate"] == pytest.approx(0.4766355140)
    assert raw_family["candidate_profit_loss_ratio"] == pytest.approx(2.8131383699)
    assert raw_family["baseline_round_trip_count"] == 31
    assert raw_family["candidate_round_trip_count"] == 40
    assert raw_family["baseline_max_loss_streak"] == 5
    assert raw_family["candidate_max_loss_streak"] == 5
    assert raw_family["baseline_buy_count"] == 92
    assert raw_family["candidate_buy_count"] == 109
    assert raw_family["baseline_sell_count"] == 89
    assert raw_family["candidate_sell_count"] == 107
    assert raw_family["annual_coverage"] == {
        "2019": 1765,
        "2020": 2134,
        "2021": 2132,
    }
    assert raw_family["annual_missing"] == {
        "2019": 431,
        "2020": 53,
        "2021": 55,
    }
    assert "STOP" in report_text


def test_fresh_unextended_entry_candidate_is_exhausted_after_joinquant_rejection():
    from cross_signal_strategy.research.research_budget import load_research_budget

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["fresh_unextended_entry_user_authorized"]
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert raw["candidate_variants"] == 1
    assert raw["buy_score_band"] == [50, 59]
    assert raw["minimum_reversal_score"] == 35
    assert raw["maximum_cross_age"] == 1
    assert raw["maximum_extension_atr"] == pytest.approx(1.0)
    assert raw["primary_path_unchanged"] is True
    assert raw["sell_path_unchanged"] is True
    assert raw["local_filled_fresh_buys"] == 19
    assert raw["local_fresh_buy_years"] == [2019, 2020, 2021]
    assert raw["candidate_created"] is True
    assert raw["joinquant_status"] == "rejected"
    assert raw["joinquant_total_return"] == pytest.approx(1.1114)
    assert raw["joinquant_max_drawdown"] == pytest.approx(0.0629)
    assert raw["joinquant_win_rate"] == pytest.approx(0.490)
    assert raw["joinquant_profit_loss_ratio"] == pytest.approx(3.904)
    assert raw["joinquant_positive_to_negative_round_trips"] == 39
    assert raw["joinquant_fresh_closed_wins"] == 4
    assert raw["joinquant_fresh_closed_losses"] == 15
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True


def test_late_macd_boll_upper_filter_is_exhausted_after_joinquant_rejection():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["late_macd_boll_upper_filter_user_authorized"]
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert budget.max_total_open_experiments == 1
    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False
    assert raw["candidate_variants"] == 1
    assert raw["eligible_side"] == "new_buy_only"
    assert raw["event_source"] == "official_joinquant_filled_buys"
    assert raw["macd_cross_up_age"] == 0
    assert raw["prior_rsi_cross_age"] == [1, 2]
    assert raw["prior_kdj_cross_age"] == [1, 2]
    assert raw["price_location"] == "close_at_or_above_boll_upper"
    assert raw["minimum_total_events"] == 3
    assert raw["minimum_distinct_years"] == 2
    assert raw["official_filled_buys"] == 98
    assert raw["matched_events"] == 2
    assert raw["matched_years"] == [2019]
    assert raw["observation_gate_passed"] is False
    assert raw["candidate_created"] is True
    assert raw["user_override_after_sparse_gate"] is True
    assert raw["joinquant_status"] == "rejected"
    assert raw["joinquant_total_return"] == pytest.approx(1.2409)
    assert raw["joinquant_annual_return"] == pytest.approx(0.3183)
    assert raw["joinquant_max_drawdown"] == pytest.approx(0.0628)
    assert raw["joinquant_win_rate"] == pytest.approx(0.558)
    assert raw["joinquant_profit_loss_ratio"] == pytest.approx(5.208)
    assert raw["joinquant_sharpe"] == pytest.approx(2.185)
    assert raw["joinquant_baseline_total_return"] == pytest.approx(1.2925)
    assert raw["joinquant_baseline_win_rate"] == pytest.approx(0.558)
    assert raw["candidate_gate_passed"] is False
    assert raw["candidate_version"] == (
        "cross-v0.3.3-late-macd-boll-filter-candidate"
    )
    assert raw["candidate_build"] == "20260822.2-candidate"
    assert raw["candidate_fingerprint"] == "a46fff884685"
    assert raw["primary_path_unchanged"] is True
    assert raw["sell_path_unchanged"] is True
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True


def test_stacked_late_veto_early_pre_macd_candidate_is_rejected_after_joinquant():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["late_veto_early_pre_macd_user_authorized"]
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert family.planned_experiment is None
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False
    assert raw["candidate_variants"] == 1
    assert raw["base_candidate"] == (
        "cross-v0.3.3-late-macd-boll-filter-candidate"
    )
    assert raw["primary_minimum_buy_score"] == 60
    assert raw["early_buy_score_band"] == [50, 59]
    assert raw["maximum_rsi_cross_age"] == 1
    assert raw["maximum_kdj_cross_age"] == 1
    assert raw["macd_state"] == "not_crossed_negative_and_narrowing"
    assert raw["price_location"] == "close_below_boll_upper"
    assert raw["maximum_rsi6"] == 85
    assert raw["queue_order"] == "primary_then_early"
    assert raw["early_fills_leftover_slots_only"] is True
    assert raw["late_veto_unchanged"] is True
    assert raw["sell_path_unchanged"] is True
    assert raw["candidate_version"] == (
        "cross-v0.3.3-late-veto-early-pre-macd-candidate"
    )
    assert raw["candidate_build"] == "20260822.3-candidate"
    assert raw["candidate_fingerprint"] == "f6b08195dd3d"
    assert raw["joinquant_status"] == "completed_rejected"
    assert raw["joinquant_total_return"] == pytest.approx(0.9765)
    assert raw["joinquant_annualized_return"] == pytest.approx(0.2628)
    assert raw["joinquant_max_drawdown"] == pytest.approx(0.0675)
    assert raw["joinquant_win_rate"] == pytest.approx(0.515)
    assert raw["joinquant_profit_loss_ratio"] == pytest.approx(3.7)
    assert raw["joinquant_early_fill_count"] == 20
    assert raw["candidate_gate_passed"] is False
    assert raw["official_gate"][
        "win_rate_must_improve_vs_formal_and_late_veto"
    ] is True
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True


def test_t1_price_reversal_pre_macd_candidate_is_exhausted_after_failed_gate():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "t1_price_reversal_pre_macd_user_authorized"
    )
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False
    assert raw["candidate_variants"] == 1
    assert raw["eligible_side"] == "new_buy_leftover_slots_only"
    assert raw["primary_minimum_buy_score"] == 60
    assert raw["alternative_requires_official_score_below"] == 60
    assert raw["rsi_bullish_cross_required"] is True
    assert raw["kdj_bullish_cross_required"] is True
    assert raw["official_cross_window_sessions"] == 3
    assert raw["macd_bullish_cross_required_absent"] is True
    assert raw["t1_low_must_be_at_least_t2_low"] is True
    assert raw["t1_close_must_exceed_t2_high"] is True
    assert raw["queue_order"] == "official_primary_then_alternative"
    assert raw["alternative_fills_leftover_slots_only"] is True
    assert raw["score_inflation"] == "none"
    assert raw["sell_path_unchanged"] is True
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["frozen_gate"]["minimum_direct_alternative_fills"] == 3
    assert raw["frozen_gate"]["minimum_direct_fill_years"] == 2
    assert raw["frozen_gate"]["minimum_return_retention"] == pytest.approx(0.95)
    assert raw["frozen_gate"]["maximum_drawdown_addition"] == pytest.approx(0.005)
    assert raw["frozen_gate"]["doubled_friction_return_retention"] == pytest.approx(0.95)
    assert raw["direct_alternative_fills"] == 26
    assert raw["direct_fill_years"] == [2019, 2020, 2021]
    assert raw["direct_closed_trades"] == 25
    assert raw["direct_closed_wins"] == 8
    assert raw["direct_closed_losses"] == 17
    assert raw["official_total_return"] == pytest.approx(1.250025)
    assert raw["candidate_total_return"] == pytest.approx(0.7042, abs=0.0001)
    assert raw["official_win_rate"] == pytest.approx(0.561798)
    assert raw["candidate_win_rate"] == pytest.approx(0.474747)
    assert raw["official_double_friction_total_return"] == pytest.approx(1.0815, abs=0.0001)
    assert raw["candidate_double_friction_total_return"] == pytest.approx(0.5462, abs=0.0001)
    assert raw["candidate_gate_passed"] is False
    assert raw["joinquant_status"] == "not_run_local_gate_failed"
    assert raw["candidate_created"] is False
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True


def test_extreme_zone_score_candidate_is_exhausted_after_noop_local_screen():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "kdj_extreme_zone_score_user_authorized"
    )
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert raw["oversold_k_max"] == pytest.approx(20.0)
    assert raw["overbought_k_min"] == pytest.approx(80.0)
    assert raw["extreme_zone_points"] == pytest.approx(5.0)
    assert raw["cross_required"] is False
    assert raw["oversold_bonus_events"] == 93
    assert raw["overbought_bonus_events"] == 1382
    assert raw["buy_threshold_crossings"] == 0
    assert raw["sell_threshold_crossings"] == 11
    assert raw["local_changed_days"] == 0
    assert raw["candidate_gate_passed"] is False
    assert raw["joinquant_status"] == "not_run_local_gate_failed"
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False


def test_kdj_tiered_persistence_candidate_is_exhausted_after_failed_local_gate():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "kdj_tiered_persistence_user_authorized"
    )
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert raw["strong_points"] == pytest.approx(10.0)
    assert raw["near_points"] == pytest.approx(5.0)
    assert raw["retention_sessions"] == 3
    assert raw["same_direction_rule"] == "maximum_not_sum"
    assert raw["opposite_direction_rule"] == "most_recent_wins"
    assert raw["local_changed_days"] == 22
    assert raw["changed_days_by_year"] == {"2019": 20, "2020": 2}
    assert raw["candidate_gate_passed"] is False
    assert raw["joinquant_status"] == "not_run_local_gate_failed"
    assert raw["formal_files_unchanged"] is True
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False


def test_kdj_tiered_current_state_candidate_is_exhausted_after_noop_local_gate():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "kdj_tiered_current_state_user_authorized"
    )
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert raw["strong_points"] == pytest.approx(10.0)
    assert raw["near_points"] == pytest.approx(5.0)
    assert raw["state_scope"] == "current_t_minus_one_only"
    assert raw["retention_sessions"] == 1
    assert raw["local_changed_days"] == 0
    assert raw["candidate_gate_passed"] is False
    assert raw["joinquant_status"] == "not_run_local_gate_failed"
    assert raw["formal_files_unchanged"] is True
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False


def test_kdj_tiered_direct_exit_candidate_is_exhausted_after_failed_local_gate():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    family = next(
        item for item in budget.families
        if item.key == "kdj_tiered_direct_exit_user_authorized"
    )
    raw = next(
        item
        for item in json.loads(BUDGET.read_text(encoding="utf-8"))["families"]
        if item["key"] == family.key
    )

    assert family.status == "exhausted"
    assert family.max_new_experiments == 0
    assert raw["state_scope"] == "current_t_minus_one_only"
    assert raw["direct_sell_requires_extreme_bonus"] is True
    assert raw["direct_sell_final_threshold"] == pytest.approx(30.0)
    assert raw["bypass_price_confirmation"] is True
    assert raw["bypass_adx_protection"] is True
    assert raw["minimum_hold_sessions"] == 5
    assert raw["local_changed_days"] == 155
    assert raw["candidate_gate_passed"] is False
    assert raw["formal_files_unchanged"] is True
    assert raw["joinquant_status"] == "not_run_local_gate_failed"
    assert raw["validation_influence"] == "none"
    assert raw["prohibit_alternatives"] is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is False


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


def test_dimension_capped_v04_is_the_only_open_research_family():
    from cross_signal_strategy.research.research_budget import (
        evaluate_experiment_request,
        load_research_budget,
    )

    budget = load_research_budget(BUDGET)
    families = {item.key: item for item in budget.families}
    family = families["dimension_capped_score_v04_user_authorized"]

    assert budget.max_total_open_experiments == 1
    assert family.status == "open"
    assert family.max_new_experiments == 1
    assert family.planned_experiment == (
        "one fixed v0.4 dimension-capped buy/sell score structure with "
        "40-point buy and 24-point ordinary sell gates"
    )
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=2
    ).allowed is False
    assert [item.key for item in budget.families if item.status == "open"] == [
        "dimension_capped_score_v04_user_authorized"
    ]

    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(item for item in payload["families"] if item["key"] == family.key)
    assert raw["candidate_name"] == "cross-v0.4.0-dimension-capped-candidate"
    assert raw["candidate_variants"] == 1
    assert raw["buy_threshold"] == 40
    assert raw["ordinary_sell_threshold"] == 24
    assert raw["severe_damage_threshold"] == 18
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
