# -*- coding: utf-8 -*-
"""Read-only IOPV quality diagnostics for the 2019-2021 training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from cross_signal_strategy.local.local_data_loader import (
    CrossSignalTrainingDataLoader,
    assert_dates_in_training_window,
)


TRAINING_YEARS = (2019, 2020, 2021)
DEFAULT_CODES = (
    "159915",
    "159920",
    "159928",
    "159985",
    "510300",
    "510880",
    "512100",
    "513050",
    "513100",
    "513500",
    "513880",
    "518880",
)
REQUIRED_COLUMNS = (
    "code",
    "date",
    "time",
    "close",
    "volume",
    "num_trades",
    "iopv",
)


@dataclass(frozen=True)
class IopvQualityStats:
    code: str
    year: int
    rows: int
    trading_days: int
    duplicate_minute_rows: int
    minute_rows_per_day_min: int
    minute_rows_per_day_median: float
    minute_rows_per_day_max: int
    missing_iopv_rows: int
    nonpositive_iopv_rows: int
    nonfinite_iopv_rows: int
    valid_iopv_rows: int
    valid_iopv_rate: float
    bar_0935_days: int
    valid_iopv_0935_days: int
    valid_iopv_0935_rate: float
    missing_iopv_0935_dates: tuple[str, ...]
    premium_observations: int
    premium_min: float
    premium_p01: float
    premium_median: float
    premium_p99: float
    premium_max: float
    premium_0935_observations: int
    premium_0935_min: float
    premium_0935_p01: float
    premium_0935_median: float
    premium_0935_p99: float
    premium_0935_max: float
    executable_0935_days: int
    executable_valid_iopv_0935_days: int
    executable_valid_iopv_0935_rate: float
    premium_executable_0935_observations: int
    premium_executable_0935_min: float
    premium_executable_0935_p01: float
    premium_executable_0935_median: float
    premium_executable_0935_p99: float
    premium_executable_0935_max: float
    no_trade_rows: int
    no_trade_valid_iopv_rows: int
    no_trade_iopv_change_rows: int


def profile_iopv_frame(
    frame: pd.DataFrame,
    code: str,
    year: int,
) -> IopvQualityStats:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required columns: %s" % ", ".join(missing_columns))

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    assert_dates_in_training_window(data)
    code_text = str(code).split(".")[0]
    year_value = int(year)
    data["code"] = data["code"].astype(str).str.split(".").str[0]
    if not data["code"].eq(code_text).all():
        raise ValueError("Frame code does not match requested code")

    dates = pd.to_datetime(data["date"], errors="coerce")
    if not dates.dt.year.eq(year_value).all():
        raise ValueError("Frame year does not match requested year")
    data["date"] = dates.dt.strftime("%Y-%m-%d")
    data["time"] = data["time"].astype(str).str.slice(0, 5)
    data = data.sort_values(["date", "time"], kind="stable").reset_index(drop=True)

    duplicate_rows = int(data.duplicated(["code", "date", "time"], keep=False).sum())
    daily_counts = data.groupby("date", sort=False).size()

    iopv = pd.to_numeric(data["iopv"], errors="coerce")
    iopv_array = iopv.to_numpy(dtype=float)
    finite_iopv = pd.Series(np.isfinite(iopv_array), index=data.index)
    missing_iopv = iopv.isna()
    nonfinite_iopv = iopv.notna() & ~finite_iopv
    nonpositive_iopv = finite_iopv & iopv.le(0.0)
    valid_iopv = finite_iopv & iopv.gt(0.0)

    close = pd.to_numeric(data["close"], errors="coerce")
    finite_close = pd.Series(np.isfinite(close.to_numpy(dtype=float)), index=data.index)
    valid_pair = valid_iopv & finite_close & close.gt(0.0)
    premium = (close[valid_pair] / iopv[valid_pair]) - 1.0

    volume = pd.to_numeric(data["volume"], errors="coerce")
    num_trades = pd.to_numeric(data["num_trades"], errors="coerce")
    no_trade = volume.eq(0.0) & num_trades.eq(0.0)

    rows_0935 = data["time"].eq("09:35")
    bar_0935_days = int(data.loc[rows_0935, "date"].nunique())
    valid_iopv_0935_days = int(data.loc[rows_0935 & valid_iopv, "date"].nunique())
    valid_0935_dates = set(data.loc[rows_0935 & valid_iopv, "date"])
    missing_iopv_0935_dates = tuple(
        date for date in sorted(data["date"].unique()) if date not in valid_0935_dates
    )
    valid_pair_0935 = valid_pair & rows_0935
    premium_0935 = (close[valid_pair_0935] / iopv[valid_pair_0935]) - 1.0
    executable_0935 = rows_0935 & ~no_trade
    executable_valid_iopv_0935 = executable_0935 & valid_iopv
    executable_0935_days = int(data.loc[executable_0935, "date"].nunique())
    executable_valid_iopv_0935_days = int(
        data.loc[executable_valid_iopv_0935, "date"].nunique()
    )
    executable_valid_pair_0935 = executable_0935 & valid_pair
    premium_executable_0935 = (
        close[executable_valid_pair_0935] / iopv[executable_valid_pair_0935]
    ) - 1.0

    previous_iopv = iopv.groupby(data["date"], sort=False).shift(1)
    iopv_changed = valid_iopv & previous_iopv.notna() & iopv.ne(previous_iopv)

    quantiles = premium.quantile([0.01, 0.5, 0.99]) if len(premium) else pd.Series(dtype=float)
    quantiles_0935 = (
        premium_0935.quantile([0.01, 0.5, 0.99])
        if len(premium_0935)
        else pd.Series(dtype=float)
    )
    quantiles_executable_0935 = (
        premium_executable_0935.quantile([0.01, 0.5, 0.99])
        if len(premium_executable_0935)
        else pd.Series(dtype=float)
    )
    trading_days = int(data["date"].nunique())

    return IopvQualityStats(
        code=code_text,
        year=year_value,
        rows=len(data),
        trading_days=trading_days,
        duplicate_minute_rows=duplicate_rows,
        minute_rows_per_day_min=int(daily_counts.min()) if len(daily_counts) else 0,
        minute_rows_per_day_median=float(daily_counts.median()) if len(daily_counts) else 0.0,
        minute_rows_per_day_max=int(daily_counts.max()) if len(daily_counts) else 0,
        missing_iopv_rows=int(missing_iopv.sum()),
        nonpositive_iopv_rows=int(nonpositive_iopv.sum()),
        nonfinite_iopv_rows=int(nonfinite_iopv.sum()),
        valid_iopv_rows=int(valid_iopv.sum()),
        valid_iopv_rate=float(valid_iopv.mean()) if len(data) else 0.0,
        bar_0935_days=bar_0935_days,
        valid_iopv_0935_days=valid_iopv_0935_days,
        valid_iopv_0935_rate=(
            valid_iopv_0935_days / trading_days if trading_days else 0.0
        ),
        missing_iopv_0935_dates=missing_iopv_0935_dates,
        premium_observations=len(premium),
        premium_min=float(premium.min()) if len(premium) else float("nan"),
        premium_p01=float(quantiles.loc[0.01]) if len(premium) else float("nan"),
        premium_median=float(quantiles.loc[0.5]) if len(premium) else float("nan"),
        premium_p99=float(quantiles.loc[0.99]) if len(premium) else float("nan"),
        premium_max=float(premium.max()) if len(premium) else float("nan"),
        premium_0935_observations=len(premium_0935),
        premium_0935_min=float(premium_0935.min()) if len(premium_0935) else float("nan"),
        premium_0935_p01=(
            float(quantiles_0935.loc[0.01]) if len(premium_0935) else float("nan")
        ),
        premium_0935_median=(
            float(quantiles_0935.loc[0.5]) if len(premium_0935) else float("nan")
        ),
        premium_0935_p99=(
            float(quantiles_0935.loc[0.99]) if len(premium_0935) else float("nan")
        ),
        premium_0935_max=float(premium_0935.max()) if len(premium_0935) else float("nan"),
        executable_0935_days=executable_0935_days,
        executable_valid_iopv_0935_days=executable_valid_iopv_0935_days,
        executable_valid_iopv_0935_rate=(
            executable_valid_iopv_0935_days / executable_0935_days
            if executable_0935_days
            else 0.0
        ),
        premium_executable_0935_observations=len(premium_executable_0935),
        premium_executable_0935_min=(
            float(premium_executable_0935.min())
            if len(premium_executable_0935)
            else float("nan")
        ),
        premium_executable_0935_p01=(
            float(quantiles_executable_0935.loc[0.01])
            if len(premium_executable_0935)
            else float("nan")
        ),
        premium_executable_0935_median=(
            float(quantiles_executable_0935.loc[0.5])
            if len(premium_executable_0935)
            else float("nan")
        ),
        premium_executable_0935_p99=(
            float(quantiles_executable_0935.loc[0.99])
            if len(premium_executable_0935)
            else float("nan")
        ),
        premium_executable_0935_max=(
            float(premium_executable_0935.max())
            if len(premium_executable_0935)
            else float("nan")
        ),
        no_trade_rows=int(no_trade.sum()),
        no_trade_valid_iopv_rows=int((no_trade & valid_iopv).sum()),
        no_trade_iopv_change_rows=int((no_trade & iopv_changed).sum()),
    )


def audit_training_iopv(
    codes: Sequence[str] = DEFAULT_CODES,
    years: Iterable[int] = TRAINING_YEARS,
    loader_factory: Callable[[], CrossSignalTrainingDataLoader] = CrossSignalTrainingDataLoader,
) -> tuple[IopvQualityStats, ...]:
    year_values = tuple(int(year) for year in years)
    if any(year not in TRAINING_YEARS for year in year_values):
        raise ValueError("Audit years must be within 2019-2021")

    report = []
    for year in year_values:
        for code in codes:
            code_text = str(code).split(".")[0]
            loader = loader_factory()
            frame = loader.load_minute_frame(code_text, "%d-01-02" % year)
            report.append(profile_iopv_frame(frame, code_text, year))
    return tuple(report)
