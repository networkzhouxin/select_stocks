# -*- coding: utf-8 -*-
"""回归后路径诊断：触下轨收回后，回到中轨继续涨到上轨 vs 掉头跌回下轨。

用训练期数据量化"回归中轨即卖"是否卖早，决定了结点是否上移。
观察性诊断，不改策略规则。
"""

from __future__ import annotations

import pandas as pd

from trend_meanrev_strategy.local.data_loader import MEANREV_POOL, TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

TRACK_DAYS = 20


def main() -> None:
    eng = MeanrevEngine(loader=TrendMeanrevDataLoader())
    days = eng.calendar_str
    n = len(days)

    events = 0
    reached_mid = 0
    mid_exit_gains = []       # 在中轨了结的收益（相对 entry）
    hold_upper = []           # 继续持有：先涨到上轨的收益
    hold_lower = []           # 继续持有：先跌回下轨的损失
    hold_neither = []         # 20 天内既没到上轨也没回下轨

    for code in MEANREV_POOL:
        sig = eng.signals[code]
        closes = eng.closes[code]
        for i, ds in enumerate(days):
            row = sig.get(ds)
            if row is None:
                continue
            b = row.get("boll_buy")
            if b is None or pd.isna(b) or not bool(b):
                continue
            entry = closes.get(ds)
            mid = row.get("mid")
            upper = row.get("upper")
            lower = row.get("lower")
            if entry is None or entry <= 0 or pd.isna(mid) or pd.isna(upper) or pd.isna(lower):
                continue
            events += 1
            first_mid = None
            first_upper = None
            first_lower = None
            for j in range(i + 1, min(i + 1 + TRACK_DAYS, n)):
                c2 = closes.get(days[j])
                if c2 is None:
                    continue
                if first_mid is None and c2 >= float(mid):
                    first_mid = j
                    first_mid_close = c2
                    continue
                if first_mid is not None:
                    if first_upper is None and c2 >= float(upper):
                        first_upper = j
                        first_upper_close = c2
                        break
                    if first_lower is None and c2 <= float(lower):
                        first_lower = j
                        first_lower_close = c2
                        break
            if first_mid is None:
                continue
            reached_mid += 1
            mid_exit_gains.append(first_mid_close / entry - 1)
            if first_upper is not None and (first_lower is None or first_upper < first_lower):
                hold_upper.append(first_upper_close / entry - 1)
            elif first_lower is not None:
                hold_lower.append(first_lower_close / entry - 1)
            else:
                # 20 天窗口内没到上轨也没回下轨，取窗口末收盘
                last_close = closes.get(days[min(i + TRACK_DAYS, n - 1)])
                hold_neither.append((last_close / entry - 1) if last_close else 0.0)

    print("=" * 64)
    print(f"触下轨收回事件总数: {events}")
    print(f"20 天内回到中轨: {reached_mid} ({reached_mid/events*100:.1f}%)")
    print("=" * 64)
    print(f"\n[中轨即卖] 平均收益: {sum(mid_exit_gains)/len(mid_exit_gains)*100:+.2f}%  (样本 {len(mid_exit_gains)})")
    print(f"\n[继续持有] 先涨到上轨: {len(hold_upper)} 笔, 平均 {sum(hold_upper)/len(hold_upper)*100:+.2f}%" if hold_upper else "\n[继续持有] 先涨到上轨: 0 笔")
    print(f"[继续持有] 先跌回下轨: {len(hold_lower)} 笔, 平均 {sum(hold_lower)/len(hold_lower)*100:+.2f}%" if hold_lower else "[继续持有] 先跌回下轨: 0 笔")
    print(f"[继续持有] 窗口内未到两轨: {len(hold_neither)} 笔, 平均 {sum(hold_neither)/len(hold_neither)*100:+.2f}%" if hold_neither else "[继续持有] 窗口内未到两轨: 0 笔")

    up = sum(hold_upper)
    down = sum(hold_lower)
    flat = sum(hold_neither)
    total = up + down + flat
    n_hold = len(hold_upper) + len(hold_lower) + len(hold_neither)
    print(f"\n[继续持有] 总平均收益: {total/n_hold*100:+.2f}%  vs  [中轨即卖] {sum(mid_exit_gains)/len(mid_exit_gains)*100:+.2f}%")


if __name__ == "__main__":
    main()
