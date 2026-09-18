# -*- coding: utf-8 -*-
"""买入候选并发诊断（只读，不改交易规则）：统计每天合格候选数量。

量比作为质量排序是否有用，取决于"同一天多个合格候选竞争有限槽位"的频率。
本脚本重放买入筛选逻辑，统计每天 formal 候选数分布与竞争频率。
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine, _flag


def main() -> None:
    loader = TrendMeanrevDataLoader()
    eng = MeanrevEngine(loader=loader)
    eng._prepare()

    # 复用引擎状态，手动重放买入阶段统计候选
    positions: dict = {}
    cooldown: dict = {}
    per_day = []  # (ds, formal_count, slots, bought)
    for i in range(1, len(eng.calendar_str)):
        ds = eng.calendar_str[i]
        prev_ds = eng.calendar_str[i - 1]
        prev2_ds = eng.calendar_str[i - 2] if i >= 2 else None

        # 简化卖出（用于跟踪持仓，不影响候选统计口径）
        # 这里只统计候选，不真卖；持仓沿用引擎真实状态会偏移，故用引擎真实 run 前先统计信号层候选
        formal = []
        for code in eng.pool:
            if code in positions:
                continue
            if code in cooldown and i - cooldown[code] <= eng.params["cooldown_days"]:
                continue
            s1 = eng.signals[code].get(prev_ds)
            if s1 is None:
                continue
            s2 = eng.signals[code].get(prev2_ds) if prev2_ds else None
            if _flag(s1.get("knife")):
                continue
            rsi = s1.get("rsi")
            if rsi is not None and not pd.isna(rsi) and float(rsi) > eng.params["overheat_rsi"]:
                continue
            price = eng.minute_0935[code].get(ds)
            if price is None or price <= 0:
                continue
            ok, all_multi = eng._buy_resonance(s1, s2)
            if ok:
                formal.append((code, all_multi))
        per_day.append((ds, len(formal)))

    # 分布统计
    dist = Counter(n for _, n in per_day)
    total_days = len(per_day)
    print(f"交易日总数: {total_days}")
    print(f"\n每天合格候选数分布:")
    for n in sorted(dist):
        pct = dist[n] / total_days * 100
        print(f"  {n} 个候选: {dist[n]:>4} 天 ({pct:.1f}%)")

    multi = [(ds, n) for ds, n in per_day if n >= 2]
    print(f"\n有 ≥2 个合格候选的天数: {len(multi)} ({len(multi)/total_days*100:.1f}%)")
    for ds, n in multi:
        print(f"  {ds}: {n} 个候选")

    # 关键问题：max_hold=2 时，竞争槽位 = 候选 > 剩余槽位 的日子。
    # 均值回归腿平均并行持仓 0.33，几乎总是 2 个空槽，
    # 所以"竞争"= 候选 ≥3（超过 max_hold）的日子。
    crowded = [(ds, n) for ds, n in per_day if n >= 3]
    print(f"\n候选 ≥3（真正溢出 max_hold=2 槽位）的天数: {len(crowded)}")
    for ds, n in crowded:
        print(f"  {ds}: {n} 个候选")


if __name__ == "__main__":
    main()
