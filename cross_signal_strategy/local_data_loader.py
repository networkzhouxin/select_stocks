# -*- coding: utf-8 -*-
"""Read-only local training data access for cross_signal_strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


APPROVED_TRAINING_ROOT = Path(r"G:\financial\history_data\cross_signal_train_2019_2021")
TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")


PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_not_training_write_path(path: PathLike) -> None:
    """Reject write/delete targets inside the read-only training data folder."""
    resolved = _resolve(path)
    training_root = _resolve(APPROVED_TRAINING_ROOT)
    if resolved == training_root or _is_relative_to(resolved, training_root):
        raise ValueError(
            "Training data root is read-only; write/delete derived files outside "
            f"{APPROVED_TRAINING_ROOT}"
        )


def assert_dates_in_training_window(frame: pd.DataFrame, date_column: str = "date") -> None:
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Invalid date values in column: {date_column}")
    if (dates < TRAIN_START).any() or (dates > TRAIN_END).any():
        raise ValueError("Data contains dates outside training window 2019-01-01 to 2021-12-31")


@dataclass(frozen=True)
class CrossSignalTrainingDataLoader:
    """Loader for the isolated 2019-2021 cross-signal training dataset."""

    root: PathLike = APPROVED_TRAINING_ROOT

    def __post_init__(self) -> None:
        resolved = _resolve(self.root)
        approved = _resolve(APPROVED_TRAINING_ROOT)
        if resolved != approved:
            raise ValueError(f"Use approved training data root only: {APPROVED_TRAINING_ROOT}")
        object.__setattr__(self, "root", resolved)

    def _year_from_date(self, trade_date: Union[str, pd.Timestamp]) -> int:
        ts = pd.Timestamp(trade_date)
        if ts < TRAIN_START or ts > TRAIN_END:
            raise ValueError("Requested date is outside training window 2019-01-01 to 2021-12-31")
        return int(ts.year)

    def _csv_path(self, kind: str, code: str, trade_date: Union[str, pd.Timestamp]) -> Path:
        year = self._year_from_date(trade_date)
        code_text = str(code).split(".")[0]
        path = self.root / kind / str(year) / f"{code_text}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def load_minute_frame(self, code: str, trade_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        frame = pd.read_csv(self._csv_path("minute_1m", code, trade_date), dtype={"code": str})
        assert_dates_in_training_window(frame)
        return frame

    def load_daily_frame(self, code: str, trade_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        frame = pd.read_csv(self._csv_path("daily", code, trade_date), dtype={"code": str})
        assert_dates_in_training_window(frame)
        return frame

    def get_minute_bar(
        self,
        code: str,
        trade_date: Union[str, pd.Timestamp],
        trade_time: str = "09:35",
    ) -> dict:
        date_text = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        time_text = str(trade_time)[:5]
        frame = self.load_minute_frame(code, trade_date)
        times = frame["time"].astype(str).str.slice(0, 5)
        rows = frame[(frame["date"].astype(str) == date_text) & (times == time_text)]
        if rows.empty:
            raise KeyError(f"No minute bar for {code} {date_text} {time_text}")
        return rows.iloc[0].to_dict()
