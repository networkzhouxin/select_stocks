# -*- coding: utf-8 -*-
"""趋势腿只读训练数据访问：日线（前复权）+ 分钟 09:35 价 + 复权因子。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import pandas as pd

TRAINING_ROOT = Path(
    os.environ.get("TREND_MEANREV_DATA_ROOT", r"D:\test\trend_meanrev_train_2017_2021")
)
WARMUP_START = pd.Timestamp("2016-01-01")
TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")

TREND_POOL = [
    "510300", "159915", "512100", "159928",
    "513100", "513500", "159920", "513880", "513050",
    "511010", "159980",
]
MEANREV_POOL = ["518880", "510880", "159985"]
EXTRA_CODES = ["511380"]
UNIVERSE = TREND_POOL + MEANREV_POOL + EXTRA_CODES

PRICE_COLUMNS = ("open", "high", "low", "close")

PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def assert_not_training_write_path(path: PathLike) -> None:
    resolved = _resolve(path)
    root = _resolve(TRAINING_ROOT)
    if resolved == root or root in resolved.parents:
        raise ValueError(
            f"训练数据根只读，派生文件写到外面: {TRAINING_ROOT}"
        )


@dataclass(frozen=True)
class TrendMeanrevDataLoader:
    """读取趋势腿隔离训练数据根（2016 warm-up + 2017-2021 训练）。"""

    root: PathLike = TRAINING_ROOT

    def __post_init__(self) -> None:
        resolved = _resolve(self.root)
        if resolved != _resolve(TRAINING_ROOT):
            raise ValueError(f"仅允许读批准的隔离根: {TRAINING_ROOT}")
        object.__setattr__(self, "root", resolved)
        object.__setattr__(self, "_adjust_factors", self._load_adjust_factors())

    # ---- 复权因子 ----

    def _load_adjust_factors(self) -> pd.DataFrame:
        path = self.root / "meta" / "adjust_factors_targets.csv"
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
        df["code"] = df["code"].str.strip().str.zfill(6)
        df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
        df["ex_cum_factor"] = pd.to_numeric(df["ex_cum_factor"], errors="coerce")
        df = df.dropna(subset=["ex_date", "ex_cum_factor"])
        return df.sort_values(["code", "ex_date"]).reset_index(drop=True)

    def _apply_fwd_adjust(self, frame: pd.DataFrame, code: str) -> pd.DataFrame:
        """前复权：前复权价(X) = 原始价(X) × ex_cum_factor(X) / ex_cum_factor(最新)。"""
        factors = self._adjust_factors[self._adjust_factors["code"] == code]
        if factors.empty or frame.empty:
            return frame
        latest_cum = float(factors["ex_cum_factor"].iloc[-1])
        if latest_cum <= 0:
            return frame
        dates = pd.to_datetime(frame["date"])
        merged = pd.merge_asof(
            dates.rename("ex_date").to_frame(),
            factors[["ex_date", "ex_cum_factor"]],
            on="ex_date",
            direction="backward",
        )
        ratio = merged["ex_cum_factor"].fillna(latest_cum) / latest_cum
        adjusted = frame.copy()
        for col in PRICE_COLUMNS:
            adjusted[col] = adjusted[col].astype(float) * ratio.to_numpy()
        return adjusted

    # ---- 日线 ----

    def load_daily(self, code: str) -> pd.DataFrame:
        code_text = str(code).split(".")[0]
        frames = []
        for year in range(2016, 2022):
            path = self.root / "daily" / str(year) / f"{code_text}.csv"
            if not path.exists():
                continue
            part = pd.read_csv(path, encoding="utf-8-sig")
            frames.append(part)
        if not frames:
            raise FileNotFoundError(f"无日线数据: {code}")
        frame = pd.concat(frames, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in PRICE_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date"] + list(PRICE_COLUMNS))
        frame = frame.sort_values("date").reset_index(drop=True)
        return self._apply_fwd_adjust(frame, code_text)

    # ---- 分钟 09:35 价 ----

    def load_minute_0935(self, code: str) -> pd.DataFrame:
        """返回该 code 每个交易日的 09:35 bar open 价（index=date, col=open）。"""
        code_text = str(code).split(".")[0]
        rows: Dict[str, float] = {}
        for year in range(2017, 2022):
            path = self.root / "minute_1m" / str(year) / f"{code_text}.csv"
            if not path.exists():
                continue
            part = pd.read_csv(path, encoding="utf-8-sig", usecols=["date", "time", "open"])
            part = part[part["time"].astype(str).str.strip() == "09:35"]
            for _, r in part.iterrows():
                rows[str(r["date"])] = float(r["open"])
        return pd.DataFrame(
            {"date": list(rows.keys()), "open": list(rows.values())}
        ).sort_values("date").reset_index(drop=True)
