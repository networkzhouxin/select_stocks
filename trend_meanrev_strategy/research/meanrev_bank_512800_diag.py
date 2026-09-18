# -*- coding: utf-8 -*-
"""512800 银行 ETF 日线诊断（只读网盘解压文件，不导入数据根）。

统计训练期 2017-2021 内买入共振（BOLL 超卖收回 + RSI/KDJ 至少一个低位拐头）
的触发次数、回归中轨胜率、平均收益，判断银行"破净修复"锚是否强到值得纳入。

注意：执行价用 T-1 收盘近似（无分钟数据时），有轻微乐观偏差，仅作初筛。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_meanrev_strategy.local import indicators as ind
from trend_meanrev_strategy.local import meanrev_indicators as mi

NETDISK = "D:/BaiduNetdiskDownload"


def load_512800() -> pd.DataFrame:
    frames = []
    for y in (2017, 2018, 2019, 2020, 2021):
        df = pd.read_csv(f"{NETDISK}/{y}/512800.csv", encoding="utf-8-sig")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    df = load_512800()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    mid, upper, lower = mi.calc_boll(c, 20, 2.0)
    rsi = ind.calc_rsi(c, 14)
    k, d, j = mi.calc_kdj(h, l, c, 9, 3, 3)

    touch_lower = (l <= lower) | (c <= lower)
    boll_buy = (touch_lower | touch_lower.shift(1)) & (c > lower) & (c > c.shift(1))
    rsi_buy = (rsi.shift(1) <= 30) & (rsi > rsi.shift(1))
    kdj_low_prev = (k.shift(1) <= 20) | (d.shift(1) <= 20) | (j.shift(1) <= 0)
    kdj_diff = k - d
    kdj_buy = kdj_low_prev & (j > j.shift(1)) & (kdj_diff > kdj_diff.shift(1))

    # 买入共振（require_boll=True）：BOLL 必选 + RSI/KDJ 至少一个
    resonance = boll_buy & (rsi_buy | kdj_buy)

    # 新鲜度窗口：T-1 或 T-2 事件，至少一个在 T-1（引擎口径）
    idx = df.index
    events = []
    for i in idx:
        if i < 2:
            continue
        t1 = resonance.iloc[i - 1]
        t2 = resonance.iloc[i - 2]
        if not (t1 or t2):
            continue
        if not t1:
            continue  # 引擎要求至少一个支持事件在 T-1
        # 防追高：RSI>75 否决
        if not pd.isna(rsi.iloc[i - 1]) and rsi.iloc[i - 1] > 75:
            continue
        events.append(i)

    print(f"日线样本: {len(df)} 天 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    print(f"BOLL 触下轨收回事件: {int(boll_buy.sum())}")
    print(f"买入共振触发次数: {len(events)}")
    print(f"RSI 低位回升次数: {int(rsi_buy.sum())}")
    print(f"KDJ 低位拐头次数: {int(kdj_buy.sum())}")

    # 对每次共振，统计后续是否回归中轨、持有期、收益
    rows = []
    for i in events:
        entry_close = c.iloc[i - 1]  # T-1 收盘近似执行价
        mid_i = mid.iloc[i - 1]
        # 后续最多 20 天，看是否触及中轨
        hit = None
        for fwd in range(1, 21):
            pos = i - 1 + fwd
            if pos >= len(c):
                break
            if c.iloc[pos] >= mid.iloc[i - 1]:
                hit = fwd
                break
        if hit is None:
            # 未回归中轨：用第 20 天收盘算（或持有期末）
            end_pos = min(i - 1 + 20, len(c) - 1)
            ret = c.iloc[end_pos] / entry_close - 1
            rows.append((df["date"].iloc[i], hit, ret))
        else:
            end_pos = i - 1 + hit
            ret = c.iloc[end_pos] / entry_close - 1
            rows.append((df["date"].iloc[i], hit, ret))

    hit = [r for r in rows if r[1] is not None]
    print(f"\n共振后回归中轨: {len(hit)}/{len(rows)} ({len(hit)/len(rows)*100:.1f}%)")
    print(f"平均持有至回归天数: {np.mean([r[1] for r in hit]):.1f}" if hit else "  (无回归)")
    rets = [r[2] for r in rows]
    print(f"平均收益(至回归或20天): {np.mean(rets)*100:+.2f}%")
    print(f"中位收益: {np.median(rets)*100:+.2f}%")
    print(f"最差: {min(rets)*100:+.2f}%  最好: {max(rets)*100:+.2f}%")

    # 对比：现有 3 只的回归中轨率约 62.9%（立项方案路径诊断）
    print("\n明细（前 30 笔）:")
    for date, hd, ret in rows[:30]:
        hd_s = f"{hd}天" if hd is not None else "未回归"
        print(f"  {date.date()}  回归:{hd_s:>6}  收益:{ret*100:+6.2f}%")


if __name__ == "__main__":
    main()
