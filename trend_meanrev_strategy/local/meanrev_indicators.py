# -*- coding: utf-8 -*-
"""均值回归腿技术指标：布林带、KDJ。均线/ATR/RSI 复用 indicators。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_boll(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """布林带(period, num_std)，ddof=0（与 resonance_reversal 口径一致）。"""
    mid = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int = 9, m1: int = 3, m2: int = 3):
    """标准 KDJ(n,m1,m2)：RSV + K/D 3 期平滑，J = 3K - 2D。"""
    low_n = low.rolling(n, min_periods=1).min()
    high_n = high.rolling(n, min_periods=1).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((close - low_n) / denom * 100).fillna(50.0)
    k = rsv.ewm(alpha=1.0 / m1, adjust=False).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def calc_dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14):
    """Wilder DMI/ADX（与 resonance_reversal 口径一致）。返回 plus_di, minus_di, adx。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    atr_rma = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period,
    ).mean() / atr_rma
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx
