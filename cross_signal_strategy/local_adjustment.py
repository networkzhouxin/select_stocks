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


TRAINING_DAILY_CORRECTION_RECORDS = (
    {"code": "512100", "date": "2020-09-02", "close": 1.001},
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


@dataclass(frozen=True)
class LocalDailyCorrections:
    """Apply confirmed local daily-bar data corrections without mutating source CSVs."""

    corrections: pd.DataFrame

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "LocalDailyCorrections":
        frame = pd.DataFrame(list(records))
        if frame.empty:
            frame = pd.DataFrame(columns=["code", "date"])
        if "code" not in frame.columns or "date" not in frame.columns:
            raise ValueError("Daily correction records require code and date")
        frame["code"] = frame["code"].astype(str).str.split(".").str[0]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("Invalid daily correction dates")
        value_columns = [c for c in frame.columns if c not in ("code", "date")]
        for column in value_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError("Invalid daily correction values")
        return cls(frame.sort_values(["code", "date"]).reset_index(drop=True))

    def apply_daily_frame(self, frame: pd.DataFrame, code: str) -> pd.DataFrame:
        if frame.empty or self.corrections.empty:
            return frame.copy()

        code_text = str(code).split(".")[0]
        rows = self.corrections[self.corrections["code"] == code_text]
        if rows.empty:
            return frame.copy()

        corrected = frame.copy()
        dates = pd.to_datetime(corrected["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid date values in daily frame")

        for row in rows.to_dict("records"):
            mask = dates == row["date"]
            for column, value in row.items():
                if column in ("code", "date") or column not in corrected.columns:
                    continue
                corrected.loc[mask, column] = value
        return corrected


def default_training_adjustment_factors() -> LocalAdjustmentFactors:
    """Return the 2019-2021 target-ETF factors inspected from the local factor file."""
    return LocalAdjustmentFactors.from_records(TRAINING_ADJUSTMENT_RECORDS)


def default_training_daily_corrections() -> LocalDailyCorrections:
    """Return confirmed local daily-bar corrections for 2019-2021 training replay."""
    return LocalDailyCorrections.from_records(TRAINING_DAILY_CORRECTION_RECORDS)
