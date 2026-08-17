# -*- coding: utf-8 -*-
"""Isolated buy-side candidate that halves age-2 bullish-cross weights."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


_BULLISH_CROSS_WEIGHTS = {
    "rsi6_cross_rsi12_up": 12.0,
    "rsi6_cross_rsi24_up": 12.0,
    "macd_cross_up": 10.0,
    "kdj_k_cross_up": 6.0,
    "kdj_j_cross_up": 5.0,
}


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
