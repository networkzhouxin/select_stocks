# -*- coding: utf-8 -*-
"""
7因子综合打分 —— 逐行复刻聚宽 calc_multi_factor_score (247-439行)。

与聚宽的唯一区别：聚宽用 get_price(end_date, count=lookback) 取数据，
本地用预先切好的 DataFrame（截至 end_date 的最后 lookback 根）传入。
打分逻辑、分档阈值、权重全部照抄。
"""
import numpy as np
import pandas as pd
from indicators import calc_rsi, calc_macd, calc_bollinger, calc_kdj, calc_atr


def calc_multi_factor_score(df, params, base_weights):
    """
    df: 截至 T-1 的日线，至少 lookback-10 根，含 open/close/high/low/volume。
        最后一行是 T-1（信号日）。
    返回 dict 或 None（数据不足/异常）。
    """
    p = params
    if df is None or len(df) < p['lookback'] - 10:
        return None

    C, H, L, V = df['close'], df['high'], df['low'], df['volume']
    sd = p['smooth_days']

    if C.iloc[-1] <= 0 or V.iloc[-5:].sum() == 0:
        return None

    # ---- 1. RSI ----
    rsi = calc_rsi(C, p['rsi_period'])
    rsi_vals = rsi.iloc[-sd:]
    if rsi_vals.isnull().any():
        return None
    rsi_val = rsi_vals.mean()

    if rsi_val < 30:
        rsi_score = 20
    elif rsi_val < 40:
        rsi_score = 35
    elif rsi_val < 50:
        rsi_score = 50
    elif rsi_val < 60:
        rsi_score = 65
    elif rsi_val < 70:
        rsi_score = 80
    elif rsi_val < 80:
        rsi_score = 75
    else:
        rsi_score = 70

    # ---- 2. MACD ----
    dif, dea, macd_hist = calc_macd(C, p['macd_fast'], p['macd_slow'], p['macd_signal'])
    dif_val = dif.iloc[-sd:].mean()
    dea_val = dea.iloc[-sd:].mean()
    hist_val = macd_hist.iloc[-sd:].mean()
    hist_prev = macd_hist.iloc[-sd - 3:-3].mean()

    macd_score = 50
    macd_score += 20 if dif_val > dea_val else -20
    if hist_val > 0 and hist_val > hist_prev:
        macd_score += 15
    elif hist_val > 0:
        macd_score += 5
    elif hist_val < 0 and hist_val > hist_prev:
        macd_score -= 5
    else:
        macd_score -= 15
    if dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] >= dea.iloc[-1]:
        macd_score += 10
    elif dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] <= dea.iloc[-1]:
        macd_score -= 10
    macd_score = max(0, min(100, macd_score))

    # ---- 3. 布林带 + squeeze ----
    bb_upper, bb_mid, bb_lower = calc_bollinger(C, p['bb_period'], p['bb_std'])
    bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    if bb_width <= 0 or pd.isna(bb_width):
        bb_score = 50
    else:
        bb_pos = (C.iloc[-1] - bb_lower.iloc[-1]) / bb_width
        if bb_pos < 0.2:
            bb_score = 20
        elif bb_pos < 0.4:
            bb_score = 40
        elif bb_pos < 0.6:
            bb_score = 60
        elif bb_pos < 0.8:
            bb_score = 80
        elif bb_pos < 0.95:
            bb_score = 75
        else:
            bb_score = 55
        bb_width_20 = (bb_upper - bb_lower).iloc[-20:]
        if len(bb_width_20.dropna()) >= 10:
            avg_w = bb_width_20.mean()
            if avg_w > 0:
                ratio = bb_width / avg_w
                if ratio < 0.6:
                    bb_score += 5
                elif ratio < 0.8:
                    bb_score += 2
        bb_score = max(0, min(100, bb_score))

    # ---- 4. 动量ROC20（3日平滑）----
    mp = p['momentum_period']
    roc = (C.iloc[-1] / C.iloc[-mp] - 1
           + C.iloc[-2] / C.iloc[-mp - 1] - 1
           + C.iloc[-3] / C.iloc[-mp - 2] - 1) / 3.0

    if roc > 0.15:
        mom_score = 95
    elif roc > 0.10:
        mom_score = 85
    elif roc > 0.05:
        mom_score = 75
    elif roc > 0.02:
        mom_score = 65
    elif roc > 0:
        mom_score = 55
    elif roc > -0.03:
        mom_score = 40
    elif roc > -0.08:
        mom_score = 25
    else:
        mom_score = 10

    # ---- 5. 成交量 ----
    vol_ma = V.iloc[-p['vol_ma_period']:].mean()
    vol_recent = V.iloc[-3:].mean()
    if vol_ma <= 0 or pd.isna(vol_ma):
        vol_score = 50
    else:
        vol_ratio = vol_recent / vol_ma
        price_up = C.iloc[-1] > C.iloc[-3]
        if price_up and vol_ratio > 1.3:
            vol_score = 85
        elif price_up and vol_ratio > 0.8:
            vol_score = 65
        elif price_up:
            vol_score = 50
        elif not price_up and vol_ratio > 1.3:
            vol_score = 25
        elif not price_up and vol_ratio > 0.8:
            vol_score = 40
        else:
            vol_score = 50

    # ---- 6. KDJ ----
    k, d, j = calc_kdj(H, L, C, p['kdj_n'], p['kdj_m1'], p['kdj_m2'])
    k_val, d_val, j_val = k.iloc[-sd:].mean(), d.iloc[-sd:].mean(), j.iloc[-sd:].mean()

    kdj_score = 50
    kdj_score += 15 if k_val > d_val else -15
    if j_val > 80:
        kdj_score += 10
    elif j_val > 50:
        kdj_score += 5
    elif j_val < 20:
        kdj_score -= 15
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] >= d.iloc[-1]:
        kdj_score += 10
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] <= d.iloc[-1]:
        kdj_score -= 10
    kdj_score = max(0, min(100, kdj_score))

    # ---- 7. 均线趋势 ----
    ma10, ma20, ma60 = C.iloc[-10:].mean(), C.iloc[-20:].mean(), C.iloc[-60:].mean()
    cur = C.iloc[-1]

    ma_score = 50
    ma_score += 10 if cur > ma10 else -10
    ma_score += 10 if cur > ma20 else -10
    ma_score += 10 if cur > ma60 else -10
    if ma10 > ma20 > ma60:
        ma_score += 15
    elif ma10 < ma20 < ma60:
        ma_score -= 15
    ma20_5d_ago = C.iloc[-25:-5].mean()
    ma_score += 5 if ma20 > ma20_5d_ago else -5
    ma_score = max(0, min(100, ma_score))

    # ---- 综合得分（固定权重）----
    scores = {
        'rsi': rsi_score, 'macd': macd_score, 'bollinger': bb_score,
        'momentum': mom_score, 'volume': vol_score, 'kdj': kdj_score,
        'ma_trend': ma_score,
    }
    final_score = 0.0
    for ks in base_weights:
        final_score += scores[ks] * base_weights[ks]

    # ---- ATR + 波动率 ----
    atr_val = calc_atr(H, L, C, p['atr_period']).iloc[-1]
    if pd.isna(atr_val):
        atr_val = cur * 0.02

    vol = C.pct_change().iloc[-20:].std() * np.sqrt(252)
    if pd.isna(vol) or vol <= 0:
        vol = 0.20

    return {
        'final_score': final_score,
        'roc': roc, 'close': cur,
        'atr': atr_val, 'volatility': vol, 'rsi': rsi_val,
    }
