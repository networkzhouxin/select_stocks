"""Pure, research-only capped-dimension score adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import sys
import types


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy


CANDIDATE_NAME = "cross-v0.4.0-dimension-capped-candidate"
BUY_THRESHOLD = 40.0
BUY_REVERSAL_MIN = 12.0
BUY_LOCATION_MIN = 7.0
BUY_TREND_MIN = 6.0
BUY_REVERSAL_CAP = 25.0
BUY_RSI_GROUP_POINTS = 12.0
BUY_KDJ_GROUP_POINTS = 6.0
BUY_KDJ_STATE_EXTREME_POINTS = 10.0
BUY_KDJ_STATE_MODERATE_POINTS = 5.0
BUY_MACD_CONFIRMATION_POINTS = 5.0
BUY_LOCATION_CAP = 10.0
BUY_BOLL_LOWER_MID_POINTS = 10.0
BUY_BOLL_MID_CROSS_POINTS = 8.0
BUY_NEAR_MA20_POINTS = 7.0
BUY_TREND_CAP = 20.0
BUY_MA5_GT_MA10_POINTS = 6.0
BUY_MA10_GT_MA20_POINTS = 6.0
BUY_MA20_SLOPE_POINTS = 5.0
BUY_CLOSE_GT_MA60_POINTS = 3.0
ORDINARY_SELL_THRESHOLD = 24.0
SELL_WEAKNESS_MIN = 10.0
SELL_DAMAGE_MIN = 8.0
SEVERE_DAMAGE_MIN = 18.0
SEVERE_WEAKNESS_MIN = 6.0
SELL_WEAKNESS_CAP = 20.0
SELL_RSI_GROUP_POINTS = 10.0
SELL_KDJ_GROUP_POINTS = 6.0
SELL_KDJ_STATE_EXTREME_POINTS = 8.0
SELL_KDJ_STATE_MODERATE_POINTS = 4.0
SELL_MACD_CONFIRMATION_POINTS = 4.0
SELL_DAMAGE_CAP = 20.0
SELL_DOWNSIDE_CONTINUATION_POINTS = 20.0
SELL_BELOW_FALLING_MA10_POINTS = 18.0
SELL_BELOW_MA20_POINTS = 15.0
SELL_BELOW_BOLL_MID_POINTS = 12.0
SELL_FELL_BACK_INSIDE_BOLL_POINTS = 8.0


def resolve_rsi_direction(snapshot: dict) -> str | None:
    """Return the exclusive RSI-cross direction, or neutral on conflict."""

    up = bool(snapshot.get("rsi6_cross_rsi12_up")) or bool(
        snapshot.get("rsi6_cross_rsi24_up")
    )
    down = bool(snapshot.get("rsi6_cross_rsi12_down")) or bool(
        snapshot.get("rsi6_cross_rsi24_down")
    )
    if up == down:
        return None
    return "up" if up else "down"


def resolve_kdj_direction(snapshot: dict) -> str | None:
    """Return the exclusive KDJ-cross direction, or neutral on conflict."""

    up = bool(snapshot.get("kdj_k_cross_up")) or bool(snapshot.get("kdj_j_cross_up"))
    down = bool(snapshot.get("kdj_k_cross_down")) or bool(
        snapshot.get("kdj_j_cross_down")
    )
    if up == down:
        return None
    return "up" if up else "down"


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _kdj_state_scores(k_value: object, warnings: list[str]) -> tuple[float, float]:
    k = _finite_float(k_value)
    if k is None:
        warnings.append("invalid_k")
        return 0.0, 0.0
    if k <= 20:
        return BUY_KDJ_STATE_EXTREME_POINTS, 0.0
    if k <= 30:
        return BUY_KDJ_STATE_MODERATE_POINTS, 0.0
    if k >= 80:
        return 0.0, SELL_KDJ_STATE_EXTREME_POINTS
    if k >= 70:
        return 0.0, SELL_KDJ_STATE_MODERATE_POINTS
    return 0.0, 0.0


def _score(value: object) -> float:
    return _finite_float(value) or 0.0


def _is_a_share_code(code: object) -> bool:
    return str(code).split(".")[0] in strategy.get_a_share_etf_codes()


def has_raw_sell_conflict(score: dict) -> bool:
    """Return raw ordinary/severe conflict without hold or ADX execution state."""

    weakness = _score(score.get("sell_weakness_score"))
    damage = _score(score.get("sell_damage_score"))
    total = _score(score.get("sell_score"))
    severe = damage >= SEVERE_DAMAGE_MIN and weakness >= SEVERE_WEAKNESS_MIN
    ordinary = (
        weakness >= SELL_WEAKNESS_MIN
        and damage >= SELL_DAMAGE_MIN
        and total >= ORDINARY_SELL_THRESHOLD
    )
    return severe or ordinary


def should_dimension_capped_signal_sell(score: dict) -> bool:
    """Apply damage-first sell rules without ATR or retained KDJ state."""

    weakness = _score(score.get("sell_weakness_score"))
    damage = _score(score.get("sell_damage_score"))
    if damage >= SEVERE_DAMAGE_MIN and weakness >= SEVERE_WEAKNESS_MIN:
        return True
    if not has_raw_sell_conflict(score):
        return False
    return not strategy.is_strong_adx_uptrend(score)


def executable_candidate_rule_manifest() -> dict:
    """Return the score/decision constants used by the candidate implementation."""

    return {
        "candidate_name": CANDIDATE_NAME,
        "buy": {
            "reversal": {
                "cap": BUY_REVERSAL_CAP,
                "minimum": BUY_REVERSAL_MIN,
                "contributions": {
                    "rsi_group": BUY_RSI_GROUP_POINTS,
                    "kdj_group": BUY_KDJ_GROUP_POINTS,
                    "kdj_state_k_le_20": BUY_KDJ_STATE_EXTREME_POINTS,
                    "kdj_state_20_lt_k_le_30": BUY_KDJ_STATE_MODERATE_POINTS,
                    "macd_confirmation": BUY_MACD_CONFIRMATION_POINTS,
                },
            },
            "location": {
                "cap": BUY_LOCATION_CAP,
                "minimum": BUY_LOCATION_MIN,
                "aggregation": "maximum_single_contribution",
                "contributions": {
                    "between_boll_lower_mid": BUY_BOLL_LOWER_MID_POINTS,
                    "cross_boll_mid_up": BUY_BOLL_MID_CROSS_POINTS,
                    "near_ma20": BUY_NEAR_MA20_POINTS,
                },
            },
            "trend": {
                "cap": BUY_TREND_CAP,
                "minimum": BUY_TREND_MIN,
                "contributions": {
                    "ma5_gt_ma10": BUY_MA5_GT_MA10_POINTS,
                    "ma10_gt_ma20": BUY_MA10_GT_MA20_POINTS,
                    "ma20_slope_non_negative": BUY_MA20_SLOPE_POINTS,
                    "close_gt_ma60": BUY_CLOSE_GT_MA60_POINTS,
                },
            },
            "total_threshold": BUY_THRESHOLD,
            "raw_sell_conflict_required_absent": True,
        },
        "sell": {
            "weakness": {
                "cap": SELL_WEAKNESS_CAP,
                "ordinary_minimum": SELL_WEAKNESS_MIN,
                "severe_minimum": SEVERE_WEAKNESS_MIN,
                "contributions": {
                    "rsi_group": SELL_RSI_GROUP_POINTS,
                    "kdj_group": SELL_KDJ_GROUP_POINTS,
                    "kdj_state_k_ge_80": SELL_KDJ_STATE_EXTREME_POINTS,
                    "kdj_state_70_le_k_lt_80": SELL_KDJ_STATE_MODERATE_POINTS,
                    "macd_confirmation": SELL_MACD_CONFIRMATION_POINTS,
                },
            },
            "damage": {
                "cap": SELL_DAMAGE_CAP,
                "ordinary_minimum": SELL_DAMAGE_MIN,
                "severe_minimum": SEVERE_DAMAGE_MIN,
                "aggregation": "maximum_single_contribution",
                "contributions": {
                    "downside_continuation": SELL_DOWNSIDE_CONTINUATION_POINTS,
                    "below_falling_ma10": SELL_BELOW_FALLING_MA10_POINTS,
                    "below_ma20": SELL_BELOW_MA20_POINTS,
                    "below_boll_mid": SELL_BELOW_BOLL_MID_POINTS,
                    "fell_back_inside_boll": SELL_FELL_BACK_INSIDE_BOLL_POINTS,
                },
            },
            "ordinary_total_threshold": ORDINARY_SELL_THRESHOLD,
            "adx_protection": "held_position_soft_sell_only",
        },
        "ranking": [
            "buy_total_desc",
            "location_desc",
            "reversal_desc",
            "a_share_volume_desc",
            "code_asc",
        ],
    }


def candidate_rule_fingerprint(manifest: dict | None = None) -> str:
    payload = manifest if manifest is not None else executable_candidate_rule_manifest()
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_snapshot(snapshot: dict) -> dict:
    """Re-score one official causal snapshot with bounded dimensions."""

    official_snapshot = deepcopy(snapshot)
    result = deepcopy(snapshot)
    warnings: list[str] = []

    rsi_direction = resolve_rsi_direction(official_snapshot)
    kdj_direction = resolve_kdj_direction(official_snapshot)
    buy_rsi_group_score = BUY_RSI_GROUP_POINTS if rsi_direction == "up" else 0.0
    sell_rsi_group_score = SELL_RSI_GROUP_POINTS if rsi_direction == "down" else 0.0
    buy_kdj_group_score = BUY_KDJ_GROUP_POINTS if kdj_direction == "up" else 0.0
    sell_kdj_group_score = SELL_KDJ_GROUP_POINTS if kdj_direction == "down" else 0.0
    buy_kdj_state_score, sell_kdj_state_score = _kdj_state_scores(
        official_snapshot.get("k"), warnings
    )
    buy_macd_confirmation_score = (
        BUY_MACD_CONFIRMATION_POINTS if official_snapshot.get("macd_cross_up") else 0.0
    )
    sell_macd_confirmation_score = (
        SELL_MACD_CONFIRMATION_POINTS if official_snapshot.get("macd_cross_down") else 0.0
    )

    raw_buy_reversal = {
        "rsi_group": buy_rsi_group_score,
        "kdj_group": buy_kdj_group_score,
        "kdj_state": buy_kdj_state_score,
        "macd_confirmation": buy_macd_confirmation_score,
    }
    raw_sell_weakness = {
        "rsi_group": sell_rsi_group_score,
        "kdj_group": sell_kdj_group_score,
        "kdj_state": sell_kdj_state_score,
        "macd_confirmation": sell_macd_confirmation_score,
    }
    reversal_score = min(BUY_REVERSAL_CAP, sum(raw_buy_reversal.values()))
    sell_weakness_score = min(SELL_WEAKNESS_CAP, sum(raw_sell_weakness.values()))
    raw_location = {
        "between_boll_lower_mid": BUY_BOLL_LOWER_MID_POINTS if official_snapshot.get("close_between_boll_lower_mid") else 0.0,
        "cross_boll_mid_up": BUY_BOLL_MID_CROSS_POINTS if official_snapshot.get("close_cross_boll_mid_up") else 0.0,
        "near_ma20": BUY_NEAR_MA20_POINTS if official_snapshot.get("close_near_ma20") else 0.0,
    }
    raw_trend = {
        "ma5_gt_ma10": BUY_MA5_GT_MA10_POINTS if official_snapshot.get("ma5_gt_ma10") else 0.0,
        "ma10_gt_ma20": BUY_MA10_GT_MA20_POINTS if official_snapshot.get("ma10_gt_ma20") else 0.0,
        "ma20_slope_non_negative": BUY_MA20_SLOPE_POINTS if official_snapshot.get("ma20_slope_non_negative") else 0.0,
        "close_gt_ma60": BUY_CLOSE_GT_MA60_POINTS if official_snapshot.get("close_gt_ma60") else 0.0,
    }
    raw_sell_damage = {
        "downside_continuation": SELL_DOWNSIDE_CONTINUATION_POINTS if official_snapshot.get("downside_continuation") else 0.0,
        "below_falling_ma10": SELL_BELOW_FALLING_MA10_POINTS if official_snapshot.get("close_below_falling_ma10") else 0.0,
        "below_ma20": SELL_BELOW_MA20_POINTS if official_snapshot.get("close_below_ma20") else 0.0,
        "below_boll_mid": SELL_BELOW_BOLL_MID_POINTS if official_snapshot.get("close_below_boll_mid") else 0.0,
        "fell_back_inside_boll": SELL_FELL_BACK_INSIDE_BOLL_POINTS if official_snapshot.get("fell_back_inside_boll") else 0.0,
    }
    location_score = max(raw_location.values())
    trend_score = min(BUY_TREND_CAP, sum(raw_trend.values()))
    sell_damage_score = max(raw_sell_damage.values())
    buy_score = reversal_score + location_score + trend_score
    sell_score = sell_weakness_score + sell_damage_score
    weak_repair_blocked = strategy.is_blocked_entry_combo(official_snapshot)
    volume_rank_score = _score(official_snapshot.get("volume_score")) if _is_a_share_code(
        official_snapshot.get("code")
    ) else 0.0

    result.update(
        {
            "candidate_name": CANDIDATE_NAME,
            "official_snapshot": official_snapshot,
            "candidate_input_warnings": warnings,
            "rsi_direction": rsi_direction,
            "kdj_direction": kdj_direction,
            "buy_rsi_group_score": buy_rsi_group_score,
            "sell_rsi_group_score": sell_rsi_group_score,
            "buy_kdj_group_score": buy_kdj_group_score,
            "sell_kdj_group_score": sell_kdj_group_score,
            "buy_kdj_state_score": buy_kdj_state_score,
            "sell_kdj_state_score": sell_kdj_state_score,
            "buy_macd_confirmation_score": buy_macd_confirmation_score,
            "sell_macd_confirmation_score": sell_macd_confirmation_score,
            "raw_buy_reversal_contributions": raw_buy_reversal,
            "raw_sell_weakness_contributions": raw_sell_weakness,
            "raw_location_contributions": raw_location,
            "raw_trend_contributions": raw_trend,
            "raw_sell_damage_contributions": raw_sell_damage,
            "reversal_score": reversal_score,
            "location_score": location_score,
            "trend_score": trend_score,
            "sell_weakness_score": sell_weakness_score,
            "sell_damage_score": sell_damage_score,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "weak_repair_blocked": weak_repair_blocked,
            "volume_rank_score": volume_rank_score,
        }
    )
    return result


def is_dimension_capped_buy_candidate(score: dict, held_codes: set[str]) -> bool:
    """Return whether a scored snapshot meets the fixed entry contract."""

    return (
        bool(score.get("buy_allowed"))
        and not bool(score.get("close_far_above_ma20"))
        and not bool(score.get("downside_continuation"))
        and not bool(score.get("weak_repair_blocked"))
        and _score(score.get("reversal_score")) >= BUY_REVERSAL_MIN
        and _score(score.get("location_score")) >= BUY_LOCATION_MIN
        and _score(score.get("trend_score")) >= BUY_TREND_MIN
        and _score(score.get("buy_score")) >= BUY_THRESHOLD
        and not has_raw_sell_conflict(score)
        and str(score.get("code", "")).split(".")[0] not in {
            str(code).split(".")[0] for code in held_codes
        }
    )


def sort_dimension_capped_candidates(candidates: list[dict]) -> list[dict]:
    """Rank candidates only by the frozen capped-score tie breakers."""

    return sorted(
        candidates,
        key=lambda item: (
            -_score(item.get("buy_score")),
            -_score(item.get("location_score")),
            -_score(item.get("reversal_score")),
            -_score(item.get("volume_rank_score")),
            str(item.get("code", "")),
        ),
    )


@dataclass(frozen=True)
class DimensionCappedScoreAdapter:
    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        base = self.source.score(code, current_date, return_reason=return_reason)
        if return_reason:
            snapshot, reason = base
            return (None, reason) if snapshot is None else (score_snapshot(snapshot), reason)
        return None if base is None else score_snapshot(base)
