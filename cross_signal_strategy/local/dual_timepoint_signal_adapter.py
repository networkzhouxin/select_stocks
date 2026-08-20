# -*- coding: utf-8 -*-
"""Score the official 09:35 frame and one causal 14:45 provisional frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cross_signal_strategy.local.intraday_signal_frame import (
    build_intraday_signal_frame,
)
from cross_signal_strategy.local.local_signal_adapter import LocalSignalAdapter


@dataclass(frozen=True)
class DualTimepointSignalAdapter:
    baseline: LocalSignalAdapter
    _score_cache: dict = field(default_factory=dict, init=False, repr=False)

    def score_at(
        self,
        code: str,
        current_date: str,
        decision_time: str,
        return_reason: bool = False,
    ):
        time_text = str(decision_time)[:5]
        if time_text == "09:35":
            return self.baseline.score(
                code, current_date, return_reason=return_reason
            )
        if time_text != "14:45":
            raise ValueError("Only 09:35 and 14:45 are allowed")

        code_text = str(code).split(".")[0]
        date_text = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        key = (code_text, date_text, time_text)
        if key not in self._score_cache:
            t1_frame, signal_date = self.baseline.load_signal_frame(
                code_text, date_text
            )
            if signal_date is None:
                self._score_cache[key] = (None, "no_previous_trade_date")
            else:
                minute_year = self.baseline.loader.load_minute_frame(
                    code_text, date_text
                )
                minutes = minute_year.loc[
                    minute_year["date"].astype(str) == date_text
                ].copy()
                point = build_intraday_signal_frame(
                    t1_frame, minutes, date_text, time_text
                )
                self._score_cache[key] = self.baseline.score_frame(
                    code_text,
                    date_text,
                    point.frame,
                    signal_date=date_text,
                    metadata={
                        "decision_time": time_text,
                        "data_cutoff": point.audit.data_cutoff,
                        "last_minute": point.audit.last_minute,
                        "minute_count": point.audit.minute_count,
                        "partial_volume": point.audit.partial_volume,
                    },
                )

        result, reason = self._score_cache[key]
        copied = dict(result) if result is not None else None
        return (copied, reason) if return_reason else copied
