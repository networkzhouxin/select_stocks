"""Pure, research-only capped-dimension score adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
ORDINARY_SELL_THRESHOLD = 24.0
SELL_WEAKNESS_MIN = 10.0
SELL_DAMAGE_MIN = 8.0
SEVERE_DAMAGE_MIN = 18.0
SEVERE_WEAKNESS_MIN = 6.0


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
        return 10.0, 0.0
    if k <= 30:
        return 5.0, 0.0
    if k >= 80:
        return 0.0, 8.0
    if k >= 70:
        return 0.0, 4.0
    return 0.0, 0.0


def _score(value: object) -> float:
    return _finite_float(value) or 0.0


def _is_a_share_code(code: object) -> bool:
    return str(code).split(".")[0] in strategy.get_a_share_etf_codes()


def _has_sell_conflict(score: dict) -> bool:
    return should_dimension_capped_signal_sell(score)


def should_dimension_capped_signal_sell(score: dict) -> bool:
    """Apply damage-first sell rules without ATR or retained KDJ state."""

    weakness = _score(score.get("sell_weakness_score"))
    damage = _score(score.get("sell_damage_score"))
    total = _score(score.get("sell_score"))
    if damage >= SEVERE_DAMAGE_MIN and weakness >= SEVERE_WEAKNESS_MIN:
        return True
    if not (
        weakness >= SELL_WEAKNESS_MIN
        and damage >= SELL_DAMAGE_MIN
        and total >= ORDINARY_SELL_THRESHOLD
    ):
        return False
    return not strategy.is_strong_adx_uptrend(score)


def score_snapshot(snapshot: dict) -> dict:
    """Re-score one official causal snapshot with bounded dimensions."""

    official_snapshot = deepcopy(snapshot)
    result = deepcopy(snapshot)
    warnings: list[str] = []

    rsi_direction = resolve_rsi_direction(official_snapshot)
    kdj_direction = resolve_kdj_direction(official_snapshot)
    buy_rsi_group_score = 12.0 if rsi_direction == "up" else 0.0
    sell_rsi_group_score = 12.0 if rsi_direction == "down" else 0.0
    buy_kdj_group_score = 6.0 if kdj_direction == "up" else 0.0
    sell_kdj_group_score = 6.0 if kdj_direction == "down" else 0.0
    buy_kdj_state_score, sell_kdj_state_score = _kdj_state_scores(
        official_snapshot.get("k"), warnings
    )
    buy_macd_confirmation_score = 5.0 if official_snapshot.get("macd_cross_up") else 0.0
    sell_macd_confirmation_score = 5.0 if official_snapshot.get("macd_cross_down") else 0.0

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
    reversal_score = min(25.0, sum(raw_buy_reversal.values()))
    sell_weakness_score = min(20.0, sum(raw_sell_weakness.values()))
    raw_location = {
        "between_boll_lower_mid": 10.0 if official_snapshot.get("close_between_boll_lower_mid") else 0.0,
        "cross_boll_mid_up": 8.0 if official_snapshot.get("close_cross_boll_mid_up") else 0.0,
        "near_ma20": 7.0 if official_snapshot.get("close_near_ma20") else 0.0,
    }
    raw_trend = {
        "ma5_gt_ma10": 6.0 if official_snapshot.get("ma5_gt_ma10") else 0.0,
        "ma10_gt_ma20": 6.0 if official_snapshot.get("ma10_gt_ma20") else 0.0,
        "ma20_slope_non_negative": 5.0 if official_snapshot.get("ma20_slope_non_negative") else 0.0,
        "close_gt_ma60": 3.0 if official_snapshot.get("close_gt_ma60") else 0.0,
    }
    raw_sell_damage = {
        "downside_continuation": 20.0 if official_snapshot.get("downside_continuation") else 0.0,
        "below_falling_ma10": 18.0 if official_snapshot.get("close_below_falling_ma10") else 0.0,
        "below_ma20": 15.0 if official_snapshot.get("close_below_ma20") else 0.0,
        "below_boll_mid": 12.0 if official_snapshot.get("close_below_boll_mid") else 0.0,
        "fell_back_inside_boll": 8.0 if official_snapshot.get("fell_back_inside_boll") else 0.0,
    }
    location_score = max(raw_location.values())
    trend_score = min(20.0, sum(raw_trend.values()))
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
        and not _has_sell_conflict(score)
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
