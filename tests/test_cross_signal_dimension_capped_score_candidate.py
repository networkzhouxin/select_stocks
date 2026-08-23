from copy import deepcopy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticAdapter:
    def __init__(self, score):
        self.score_value = score

    def score(self, code, current_date, return_reason=False):
        value = deepcopy(self.score_value) if self.score_value is not None else None
        reason = None if value is not None else "no_data"
        return (value, reason) if return_reason else value


def _candidate_module():
    from cross_signal_strategy.research import dimension_capped_score_candidate
    return dimension_capped_score_candidate


def _snapshot(**overrides):
    values = {
        "code": "513100",
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": False,
        "macd_cross_down": False,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
        "kdj_k_cross_down": False,
        "kdj_j_cross_down": False,
        "k": 50.0,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "ma5_gt_ma10": False,
        "ma10_gt_ma20": False,
        "ma20_slope_non_negative": False,
        "close_gt_ma60": False,
        "downside_continuation": False,
        "close_below_falling_ma10": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "fell_back_inside_boll": False,
        "buy_allowed": True,
        "volume_score": 0.0,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
        "atr": 0.1,
        "buy_score": 0.0,
        "sell_score": 0.0,
        "reversal_score": 0.0,
        "location_score": 0.0,
        "trend_score": 0.0,
        "sell_reversal_score": 0.0,
        "sell_risk_score": 0.0,
    }
    values.update(overrides)
    return values


def _eligible_snapshot(**overrides):
    values = _snapshot(
        rsi6_cross_rsi12_up=True,
        kdj_k_cross_up=True,
        close_between_boll_lower_mid=True,
        ma5_gt_ma10=True,
        ma10_gt_ma20=True,
    )
    values.update(overrides)
    return values


def test_rsi_and_kdj_groups_neutralize_mixed_directions():
    module = _candidate_module()
    mixed = _snapshot(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi24_down=True,
        kdj_k_cross_up=True,
        kdj_j_cross_down=True,
    )
    score = module.DimensionCappedScoreAdapter(StaticAdapter(mixed)).score(
        "513100", "2019-01-08"
    )
    assert score["rsi_direction"] is None
    assert score["kdj_direction"] is None
    assert score["buy_rsi_group_score"] == 0
    assert score["sell_rsi_group_score"] == 0
    assert score["buy_kdj_group_score"] == 0
    assert score["sell_kdj_group_score"] == 0


def test_same_direction_multiple_crosses_count_once():
    module = _candidate_module()
    bullish = _snapshot(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi24_up=True,
        kdj_k_cross_up=True,
        kdj_j_cross_up=True,
        k=19.0,
        macd_cross_up=True,
    )
    score = module.DimensionCappedScoreAdapter(StaticAdapter(bullish)).score(
        "513100", "2019-01-08"
    )
    assert score["buy_rsi_group_score"] == 12
    assert score["buy_kdj_group_score"] == 6
    assert score["buy_kdj_state_score"] == 10
    assert score["buy_macd_confirmation_score"] == 5
    assert score["reversal_score"] == 25


def test_frozen_buy_and_sell_contributions_and_caps_match_the_approved_rule():
    module = _candidate_module()

    bullish = module.score_snapshot(_snapshot(
        rsi6_cross_rsi12_up=True,
        kdj_k_cross_up=True,
        k=20.0,
        macd_cross_up=True,
        close_between_boll_lower_mid=True,
        close_cross_boll_mid_up=True,
        close_near_ma20=True,
        ma5_gt_ma10=True,
        ma10_gt_ma20=True,
        ma20_slope_non_negative=True,
        close_gt_ma60=True,
    ))
    assert bullish["raw_buy_reversal_contributions"] == {
        "rsi_group": 12.0,
        "kdj_group": 6.0,
        "kdj_state": 10.0,
        "macd_confirmation": 5.0,
    }
    assert bullish["reversal_score"] == 25.0
    assert bullish["raw_location_contributions"] == {
        "between_boll_lower_mid": 10.0,
        "cross_boll_mid_up": 8.0,
        "near_ma20": 7.0,
    }
    assert bullish["location_score"] == 10.0
    assert bullish["raw_trend_contributions"] == {
        "ma5_gt_ma10": 6.0,
        "ma10_gt_ma20": 6.0,
        "ma20_slope_non_negative": 5.0,
        "close_gt_ma60": 3.0,
    }
    assert bullish["trend_score"] == 20.0

    bearish = module.score_snapshot(_snapshot(
        rsi6_cross_rsi12_down=True,
        kdj_k_cross_down=True,
        k=80.0,
        macd_cross_down=True,
        downside_continuation=True,
        close_below_falling_ma10=True,
        close_below_ma20=True,
        close_below_boll_mid=True,
        fell_back_inside_boll=True,
    ))
    assert bearish["raw_sell_weakness_contributions"] == {
        "rsi_group": 10.0,
        "kdj_group": 6.0,
        "kdj_state": 8.0,
        "macd_confirmation": 4.0,
    }
    assert bearish["sell_weakness_score"] == 20.0
    assert bearish["raw_sell_damage_contributions"] == {
        "downside_continuation": 20.0,
        "below_falling_ma10": 18.0,
        "below_ma20": 15.0,
        "below_boll_mid": 12.0,
        "fell_back_inside_boll": 8.0,
    }
    assert bearish["sell_damage_score"] == 20.0


def test_location_and_damage_take_the_strongest_item_instead_of_accumulating():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(
        close_between_boll_lower_mid=True,
        close_cross_boll_mid_up=True,
        close_near_ma20=True,
        downside_continuation=True,
        close_below_falling_ma10=True,
        close_below_ma20=True,
        close_below_boll_mid=True,
        fell_back_inside_boll=True,
    ))
    assert score["location_score"] == 10
    assert score["trend_score"] == 0
    assert score["sell_damage_score"] == 20


def test_trend_is_additive_only_inside_its_twenty_point_cap():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(
        ma5_gt_ma10=True,
        ma10_gt_ma20=True,
        ma20_slope_non_negative=True,
        close_gt_ma60=True,
    ))
    assert score["trend_score"] == 20


def test_buy_requires_all_three_dimension_floors_and_total_forty():
    module = _candidate_module()
    eligible = module.score_snapshot(_eligible_snapshot())
    assert eligible["buy_score"] == 40
    assert eligible["buy_macd_confirmation_score"] == 0
    assert module.is_dimension_capped_buy_candidate(eligible, held_codes=set())

    for field, value in (
        ("reversal_score", 11),
        ("location_score", 6),
        ("trend_score", 5),
        ("buy_score", 39),
    ):
        rejected = dict(eligible, **{field: value})
        assert not module.is_dimension_capped_buy_candidate(rejected, held_codes=set())


def test_buy_hard_blocks_chasing_downside_weak_repair_and_sell_conflict():
    module = _candidate_module()
    base = module.score_snapshot(_eligible_snapshot())
    cases = [
        dict(base, close_far_above_ma20=True),
        dict(base, downside_continuation=True),
        dict(base, weak_repair_blocked=True),
        dict(base, buy_allowed=False),
        dict(base, sell_weakness_score=10, sell_damage_score=14, sell_score=24),
        dict(base, sell_weakness_score=6, sell_damage_score=18, sell_score=24),
        dict(base, code="513100"),
    ]
    for index, item in enumerate(cases):
        held = {"513100"} if index == len(cases) - 1 else set()
        assert not module.is_dimension_capped_buy_candidate(item, held)


def test_raw_sell_conflict_blocks_new_buy_even_when_adx_protects_held_soft_sell():
    module = _candidate_module()
    ordinary_conflict = dict(
        module.score_snapshot(_eligible_snapshot()),
        sell_weakness_score=10.0,
        sell_damage_score=14.0,
        sell_score=24.0,
        adx=30.0,
        plus_di=35.0,
        minus_di=10.0,
        ma20_slope_non_negative=True,
    )

    assert module.has_raw_sell_conflict(ordinary_conflict)
    assert not module.should_dimension_capped_signal_sell(ordinary_conflict)
    assert not module.is_dimension_capped_buy_candidate(
        ordinary_conflict,
        held_codes=set(),
    )


def test_frozen_rule_manifest_and_fingerprint_are_deterministic():
    module = _candidate_module()
    manifest = module.executable_candidate_rule_manifest()

    assert manifest["candidate_name"] == "cross-v0.4.0-dimension-capped-candidate"
    assert manifest["buy"]["reversal"] == {
        "cap": 25.0,
        "minimum": 12.0,
        "contributions": {
            "rsi_group": 12.0,
            "kdj_group": 6.0,
            "kdj_state_k_le_20": 10.0,
            "kdj_state_20_lt_k_le_30": 5.0,
            "macd_confirmation": 5.0,
        },
    }
    assert manifest["sell"]["weakness"] == {
        "cap": 20.0,
        "ordinary_minimum": 10.0,
        "severe_minimum": 6.0,
        "contributions": {
            "rsi_group": 10.0,
            "kdj_group": 6.0,
            "kdj_state_k_ge_80": 8.0,
            "kdj_state_70_le_k_lt_80": 4.0,
            "macd_confirmation": 4.0,
        },
    }
    first = module.candidate_rule_fingerprint(manifest)
    second = module.candidate_rule_fingerprint(
        module.executable_candidate_rule_manifest()
    )
    assert first == second
    assert len(first) == 64


def test_ranking_uses_only_the_frozen_keys():
    module = _candidate_module()
    items = [
        dict(code="513100", buy_score=40, location_score=8, reversal_score=13, volume_rank_score=0),
        dict(code="159915", buy_score=40, location_score=8, reversal_score=13, volume_rank_score=6),
        dict(code="510300", buy_score=40, location_score=10, reversal_score=12, volume_rank_score=0),
        dict(code="513050", buy_score=41, location_score=7, reversal_score=12, volume_rank_score=0),
    ]
    assert [item["code"] for item in module.sort_dimension_capped_candidates(items)] == [
        "513050", "510300", "159915", "513100"
    ]


def test_soft_sell_can_be_protected_but_severe_damage_cannot():
    module = _candidate_module()
    strong_adx = dict(adx=30.0, plus_di=35.0, minus_di=10.0, ma20_slope_non_negative=True)
    soft = dict(strong_adx, sell_weakness_score=12, sell_damage_score=12, sell_score=24)
    severe = dict(strong_adx, sell_weakness_score=6, sell_damage_score=18, sell_score=24)
    assert not module.should_dimension_capped_signal_sell(soft)
    assert module.should_dimension_capped_signal_sell(severe)


def test_high_k_without_price_damage_does_not_sell():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(k=85.0))
    assert score["sell_weakness_score"] == 8
    assert score["sell_damage_score"] == 0
    assert not module.should_dimension_capped_signal_sell(score)


def test_adapter_preserves_t_minus_one_metadata_and_source_snapshot():
    source = _snapshot(nested={"values": [1]})
    original = deepcopy(source)
    adapter = _candidate_module().DimensionCappedScoreAdapter(StaticAdapter(source))
    first, reason = adapter.score("513100", "2019-01-08", return_reason=True)
    first["nested"]["values"].append(2)
    second = adapter.score("513100", "2019-01-08")
    assert reason is None
    assert source == original
    assert second["nested"] == {"values": [1]}
    assert second["signal_date"] == "2019-01-07"
    assert second["max_data_date"] == "2019-01-07"


def test_missing_or_nonfinite_k_contributes_zero_and_is_audited():
    module = _candidate_module()
    for value in (None, float("nan"), float("inf")):
        score = module.score_snapshot(_snapshot(k=value))
        assert score["buy_kdj_state_score"] == 0
        assert score["sell_kdj_state_score"] == 0
        assert "invalid_k" in score["candidate_input_warnings"]


def test_kdj_state_uses_current_t_minus_one_only_without_retention():
    previous = _candidate_module().score_snapshot(_snapshot(k=19.0))
    current = _candidate_module().score_snapshot(_snapshot(k=50.0))
    assert previous["buy_kdj_state_score"] == 10
    assert current["buy_kdj_state_score"] == 0
    assert current["sell_kdj_state_score"] == 0
