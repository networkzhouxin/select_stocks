# -*- coding: utf-8 -*-
"""Adapt local daily CSV data into cross-signal strategy scores."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd


sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))

from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as strategy


@dataclass(frozen=True)
class LocalSignalAdapter:
    """Build T-1 daily signal snapshots from the isolated local training data."""

    loader: object
    params: dict | None = None
    _daily_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.params is None:
            object.__setattr__(self, "params", strategy.get_default_params())

    def _daily_frame_for_year(self, code: str, current_date: str) -> pd.DataFrame:
        code_text = str(code).split(".")[0]
        current = pd.Timestamp(current_date)
        frames = []
        for year in range(2019, int(current.year) + 1):
            key = (code_text, year)
            if key not in self._daily_cache:
                self._daily_cache[key] = self.loader.load_daily_frame(code_text, f"{year}-12-31")
            frames.append(self._daily_cache[key])
        return pd.concat(frames, ignore_index=True)

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
        return visible, signal_date

    def score(self, code: str, current_date: str, return_reason: bool = False):
        p = self.params or strategy.get_default_params()
        min_len = p["lookback"] - 10
        required = ["rsi6", "rsi12", "rsi24", "dif", "dea", "k", "d", "j", "ma20", "atr", "adx"]

        frame, signal_date = self.load_signal_frame(code, current_date)
        if signal_date is None:
            reason = "no_previous_trade_date"
            return (None, reason) if return_reason else None

        reason = strategy.score_skip_reason(frame, None, required, min_len)
        if reason is not None:
            return (None, reason) if return_reason else None

        snapshot = strategy.build_signal_snapshot(frame, p)
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
