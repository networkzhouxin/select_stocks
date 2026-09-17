# -*- coding: utf-8 -*-
"""趋势腿技术指标：均线、ATR、RSI、ROC、新高。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def calc_roc(close: pd.Series, period: int) -> pd.Series:
    return close / close.shift(period) - 1


def calc_new_high(close: pd.Series, period: int) -> pd.Series:
    """严格突破：close 高于此前 period 个收盘价的最高值。"""
    return close > close.rolling(period).max().shift(1)
