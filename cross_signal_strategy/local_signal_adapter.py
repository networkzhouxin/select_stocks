# -*- coding: utf-8 -*-
"""Adapt local daily CSV data into cross-signal strategy scores."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy
from cross_signal_strategy.local_data_loader import APPROVED_WARMUP_ROOT, assert_warmup_dates


@dataclass(frozen=True)
class LocalSignalAdapter:
    """Build T-1 daily signal snapshots from the isolated local training data."""

    loader: object
    params: dict | None = None
    warmup_root: Path | str | None = None
    adjustment_factors: object | None = None
    _daily_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.params is None:
            object.__setattr__(self, "params", strategy.get_default_params())
        if self.warmup_root is not None:
            root = Path(self.warmup_root).expanduser().resolve()
            approved = Path(APPROVED_WARMUP_ROOT).expanduser().resolve()
            if root != approved:
                raise ValueError(f"Use approved warm-up data root only: {APPROVED_WARMUP_ROOT}")
            object.__setattr__(self, "warmup_root", root)

    def _daily_frame_for_year(self, code: str, current_date: str) -> pd.DataFrame:
        code_text = str(code).split(".")[0]
        current = pd.Timestamp(current_date)
        frames = []
        warmup = self._load_warmup_frame(code_text)
        if warmup is not None:
            frames.append(warmup)
        for year in range(2019, int(current.year) + 1):
            key = (code_text, year)
            if key not in self._daily_cache:
                self._daily_cache[key] = self.loader.load_daily_frame(code_text, f"{year}-12-31")
            frames.append(self._daily_cache[key])
        return pd.concat(frames, ignore_index=True)

    def _load_warmup_frame(self, code: str) -> pd.DataFrame | None:
        if self.warmup_root is None:
            return None
        key = (str(code).split(".")[0], "warmup")
        if key in self._daily_cache:
            return self._daily_cache[key]
        path = Path(self.warmup_root) / "daily" / "2018" / f"{key[0]}.csv"
        if not path.exists():
            self._daily_cache[key] = None
            return None
        frame = pd.read_csv(path, dtype={"code": str})
        assert_warmup_dates(frame)
        self._daily_cache[key] = frame
        return frame

    def previous_signal_date(self, code: str, current_date: str) -> str | None:
        frame = self._daily_frame_for_year(code, current_date)
        current = pd.Timestamp(current_date)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        previous_dates = dates[dates < current]
        if previous_dates.empty:
            return None
        return previous_dates.max().strftime("%Y-%m-%d")

    def load_signal_frame(self, code: str, current_date: str) -> Tuple[pd.DataFrame, str | None]:
        frame = self._daily_frame_for_year(code, current_date)
        signal_date = self.previous_signal_date(code, current_date)
        if signal_date is None:
            return frame.iloc[0:0].copy(), None
        dates = pd.to_datetime(frame["date"], errors="coerce")
        visible = frame.loc[dates <= pd.Timestamp(signal_date)].copy()
        if self.adjustment_factors is not None:
            visible = self.adjustment_factors.adjust_daily_frame(visible, code, current_date)
        return visible, signal_date

    def score(self, code: str, current_date: str, return_reason: bool = False):
        p = self.params or strategy.get_default_params()
        min_len = self._local_min_len(p)
        required = ["rsi6", "rsi12", "rsi24", "dif", "dea", "k", "d", "j", "ma20", "atr", "adx"]

        frame, signal_date = self.load_signal_frame(code, current_date)
        if signal_date is None:
            reason = "no_previous_trade_date"
            return (None, reason) if return_reason else None

        reason = strategy.score_skip_reason(frame, None, required, min_len)
        if reason is not None:
            return (None, reason) if return_reason else None

        snapshot = strategy.build_signal_snapshot(frame, p)
        self._suppress_float_artifact_flags(snapshot, frame)
        reason = strategy.score_skip_reason(frame, snapshot, required, min_len)
        if reason is not None:
            return (None, reason) if return_reason else None

        result = {}
        result.update(snapshot)
        result.update(strategy.score_buy_snapshot(snapshot, p))
        result.update(strategy.score_sell_snapshot(snapshot))
        result["code"] = str(code).split(".")[0]
        result["current_date"] = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        result["signal_date"] = signal_date
        result["max_data_date"] = str(frame["date"].max())
        return (result, None) if return_reason else result

    def _local_min_len(self, params: dict) -> int:
        return max(
            int(params["rsi_slow"]),
            int(params["macd_slow"]) + int(params["macd_signal"]),
            int(params["boll_period"]),
            int(params["atr_period"]),
            int(params["adx_period"]) * 2,
        )

    def _suppress_float_artifact_flags(self, snapshot: dict, frame: pd.DataFrame) -> None:
        if not snapshot.get("close_below_falling_ma10"):
            return
        close = pd.to_numeric(frame["close"], errors="coerce")
        ma10 = close.rolling(10).mean()
        if len(ma10) < 2:
            return
        delta = ma10.iloc[-2] - ma10.iloc[-1]
        if 0 < delta < 1e-12:
            snapshot["close_below_falling_ma10"] = False
