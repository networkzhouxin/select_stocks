# -*- coding: utf-8 -*-
"""周期性品种日线诊断（只读网盘解压文件）：512400 有色 / 512880 证券 / 159930 能源。

复用 512800 银行诊断口径：统计买入共振触发、回归中轨率、平均收益，
判断"供需周期锚"是否强到值得纳入均值回归池。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_meanrev_strategy.local import indicators as ind
from trend_meanrev_strategy.local import meanrev_indicators as mi

NETDISK = "D:/BaiduNetdiskDownload"
CODES = {
    "512400": "有色(工业金属)",
    "512880": "证券(市场情绪)",
    "159930": "能源(煤炭石油)",
}


def load_code(code: str) -> pd.DataFrame:
    frames = []
    for y in (2017, 2018, 2019, 2020, 2021):
        df = pd.read_csv(f"{NETDISK}/{y}/{code}.csv", encoding="utf-8-sig")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def diagnose(code: str) -> None:
    df = load_code(code)
    c, h, l = df["close"], df["high"], df["low"]
    mid, upper, lower = mi.calc_boll(c, 20, 2.0)
    rsi = ind.calc_rsi(c, 14)
    k, d, j = mi.calc_kdj(h, l, c, 9, 3, 3)

    touch_lower = (l <= lower) | (c <= lower)
    boll_buy = (touch_lower | touch_lower.shift(1)) & (c > lower) & (c > c.shift(1))
    rsi_buy = (rsi.shift(1) <= 30) & (rsi > rsi.shift(1))
    kdj_low_prev = (k.shift(1) <= 20) | (d.shift(1) <= 20) | (j.shift(1) <= 0)
    kd_diff = k - d
    kdj_buy = kdj_low_prev & (j > j.shift(1)) & (kd_diff > kd_diff.shift(1))
    resonance = boll_buy & (rsi_buy | kdj_buy)

    events = []
    for i in df.index:
        if i < 2:
            continue
        if not resonance.iloc[i - 1]:
            continue
        if not pd.isna(rsi.iloc[i - 1]) and rsi.iloc[i - 1] > 75:
            continue
        events.append(i)

    rows = []
    for i in events:
        entry_close = c.iloc[i - 1]
        mid_i = mid.iloc[i - 1]
        hit = None
        for fwd in range(1, 21):
            pos = i - 1 + fwd
            if pos >= len(c):
                break
            if c.iloc[pos] >= mid_i:
                hit = fwd
                break
        end_pos = min(i - 1 + hit, len(c) - 1) if hit else min(i - 1 + 20, len(c) - 1)
        ret = c.iloc[end_pos] / entry_close - 1
        rows.append((df["date"].iloc[i], hit, ret))

    hit = [r for r in rows if r[1] is not None]
    rets = [r[2] for r in rows]
    name = CODES[code]
    print(f"\n=== {code} {name} ===")
    print(f"样本: {len(df)} 天 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    print(f"BOLL 触下轨收回: {int(boll_buy.sum())}  买入共振: {len(events)}")
    print(f"回归中轨率: {len(hit)}/{len(rows)} ({len(hit)/len(rows)*100:.1f}%)" if rows else "  无共振")
    if rows:
        print(f"平均收益: {np.mean(rets)*100:+.2f}%  中位: {np.median(rets)*100:+.2f}%")
        print(f"最差: {min(rets)*100:+.2f}%  最好: {max(rets)*100:+.2f}%")
        worst = sorted(rows, key=lambda x: x[2])[:5]
        print("最差 5 笔:")
        for date, hd, ret in worst:
            hd_s = f"{hd}天" if hd is not None else "未回归"
            print(f"  {date.date()}  回归:{hd_s:>6}  收益:{ret*100:+6.2f}%")


def main() -> None:
    for code in CODES:
        diagnose(code)


if __name__ == "__main__":
    main()
