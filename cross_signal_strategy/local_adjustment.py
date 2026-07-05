# -*- coding: utf-8 -*-
"""Local adjustment-factor handling for cross-signal training replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")


TRAINING_ADJUSTMENT_RECORDS = (
    {"code": "159928", "ex_date": "2021-06-25", "ex_factor": 4.0},
    {"code": "510300", "ex_date": "2019-01-16", "ex_factor": 1.0188548484953115},
    {"code": "510300", "ex_date": "2019-12-11", "ex_factor": 1.0159230812057776},
    {"code": "510300", "ex_date": "2021-01-18", "ex_factor": 1.0132002506617996},
    {"code": "510880", "ex_date": "2019-01-16", "ex_factor": 1.038688253285561},
    {"code": "510880", "ex_date": "2020-01-17", "ex_factor": 1.0513740030198886},
    {"code": "510880", "ex_date": "2021-01-18", "ex_factor": 1.0543561221399267},
)


@dataclass(frozen=True)
class LocalAdjustmentFactors:
    """Apply known ETF ex-dividend/split factors without using future events."""

    factors: pd.DataFrame

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "LocalAdjustmentFactors":
        frame = pd.DataFrame(list(records), columns=["code", "ex_date", "ex_factor"])
        if frame.empty:
            frame = pd.DataFrame(columns=["code", "ex_date", "ex_factor"])
        frame["code"] = frame["code"].astype(str).str.split(".").str[0]
        frame["ex_date"] = pd.to_datetime(frame["ex_date"], errors="coerce")
        frame["ex_factor"] = pd.to_numeric(frame["ex_factor"], errors="coerce")
        if frame["ex_date"].isna().any() or frame["ex_factor"].isna().any():
            raise ValueError("Invalid adjustment factor records")
        if (frame["ex_factor"] <= 0).any():
            raise ValueError("Adjustment factors must be positive")
        return cls(frame.sort_values(["code", "ex_date"]).reset_index(drop=True))

    def adjust_daily_frame(self, frame: pd.DataFrame, code: str, current_date: str) -> pd.DataFrame:
        """Adjust historical OHLC rows by events known on or before current_date."""
        if frame.empty:
            return frame.copy()

        code_text = str(code).split(".")[0]
        current = pd.Timestamp(current_date)
        events = self.factors[
            (self.factors["code"] == code_text) & (self.factors["ex_date"] <= current)
        ]
        if events.empty:
            return frame.copy()

        adjusted = frame.copy()
        dates = pd.to_datetime(adjusted["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid date values in daily frame")

        divisor = pd.Series(1.0, index=adjusted.index)
        for event in events.itertuples(index=False):
            divisor.loc[dates < event.ex_date] *= float(event.ex_factor)

        for column in PRICE_COLUMNS:
            if column in adjusted.columns:
                adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") / divisor
        return adjusted


def default_training_adjustment_factors() -> LocalAdjustmentFactors:
    """Return the 2019-2021 target-ETF factors inspected from the local factor file."""
    return LocalAdjustmentFactors.from_records(TRAINING_ADJUSTMENT_RECORDS)
