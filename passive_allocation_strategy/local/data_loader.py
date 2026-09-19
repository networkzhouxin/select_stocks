# -*- coding: utf-8 -*-
"""被动配置策略只读数据加载：日线（前复权）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

TRAINING_ROOT = Path(
    os.environ.get("TREND_MEANREV_DATA_ROOT", r"G:\financial\history_data\trend_meanrev_train_2017_2021")
)

POOL = ["510300", "510880", "159915", "513100", "513500", "518880", "511010"]

TARGET_WEIGHTS = {
    "510880": 0.20,  # 上证红利
    "510300": 0.10,  # 沪深300
    "159915": 0.05,  # 创业板
    "513100": 0.15,  # 纳指100
    "513500": 0.10,  # 标普500
    "518880": 0.05,  # 黄金
    "511010": 0.35,  # 债 = 现金（货基/逆回购，不买国债ETF）
}

PRICE_COLUMNS = ("open", "high", "low", "close")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PassiveAllocationDataLoader:
    """读取被动配置策略日线数据（前复权），复用隔离数据根。"""

    root: PathLike = TRAINING_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "_adjust", self._load_adjust())

    def _load_adjust(self) -> pd.DataFrame:
        path = self.root / "meta" / "adjust_factors_targets.csv"
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
        df["code"] = df["code"].str.strip().str.zfill(6)
        df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
        df["ex_cum_factor"] = pd.to_numeric(df["ex_cum_factor"], errors="coerce")
        df = df.dropna(subset=["ex_date", "ex_cum_factor"])
        return df.sort_values(["code", "ex_date"]).reset_index(drop=True)

    def load_daily(self, code: str) -> pd.DataFrame:
        code_text = str(code).split(".")[0]
        frames = []
        for year in range(2016, 2022):
            path = self.root / "daily" / str(year) / f"{code_text}.csv"
            if path.exists():
                frames.append(pd.read_csv(path, encoding="utf-8-sig"))
        if not frames:
            raise FileNotFoundError(f"无日线数据: {code}")
        frame = pd.concat(frames, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in PRICE_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date", "close"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return self._fwd_adjust(frame, code_text)

    def _fwd_adjust(self, frame: pd.DataFrame, code: str) -> pd.DataFrame:
        """前复权：前复权价(X) = 原始价(X) × ex_cum_factor(X) / ex_cum_factor(最新)。"""
        factors = self._adjust[self._adjust["code"] == code]
        if factors.empty:
            return frame
        latest = float(factors["ex_cum_factor"].iloc[-1])
        if latest <= 0:
            return frame
        dates = pd.to_datetime(frame["date"])
        merged = pd.merge_asof(
            dates.rename("ex_date").to_frame(),
            factors[["ex_date", "ex_cum_factor"]],
            on="ex_date",
            direction="backward",
        )
        ratio = merged["ex_cum_factor"].fillna(latest) / latest
        adj = frame.copy()
        for col in PRICE_COLUMNS:
            adj[col] = adj[col].astype(float) * ratio.to_numpy()
        return adj
