# -*- coding: utf-8 -*-
"""量能维度诊断（只读，不改交易规则）：训练期 2017-2021。

对现有 3 只池的全部买入订单，回溯 T-1 的量比（当日成交量 ÷ 20日均量），
统计不同量能水平下买入的胜率 / 平均盈亏，判断量能是否具备区分度。

机制假设（均值回归经典先验）：下跌末端的缩量 = 抛压衰竭，企稳反弹更可靠。
若缩量买入显著优于放量买入 → 量能可作为确认条件；无区分度 → 钉死"不做量能"。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine


def build_volume_ratio(loader: TrendMeanrevDataLoader, code: str) -> dict:
    """返回 {date_str: volume_ratio(T-1 当日量/20日均量)}，用前复权前的原始量即可（量不随复权变）。"""
    daily = loader.load_daily(code)
    vol = daily.set_index("date")["volume"].astype(float)
    ma20 = vol.rolling(20).mean()
    ratio = vol / ma20
    return {
        d.strftime("%Y-%m-%d"): (float(ratio.loc[d]) if not pd.isna(ratio.loc[d]) else None)
        for d in daily["date"]
    }


def main() -> None:
    loader = TrendMeanrevDataLoader()
    s = MeanrevEngine(loader=loader).run()

    vol_ratio: dict = {}
    for code in ["518880", "510880", "159985"]:
        vol_ratio[code] = build_volume_ratio(loader, code)

    # 订单 → 每笔买入对应的 T-1 量比
    # 引擎买入在 T 日执行，信号基于 prev_ds=T-1；量能确认应看 T-1 当日量比
    trades = []
    prev_date: dict = {}
    for code in ["518880", "510880", "159985"]:
        dates = sorted(vol_ratio[code].keys())
        for i in range(1, len(dates)):
            prev_date[(code, dates[i])] = dates[i - 1]

    # 配对 FIFO 算每笔盈亏 + 记录买入时 T-1 量比
    entry: dict = {}
    pnls_by_bucket = defaultdict(list)
    vr_by_trade = []
    for o in s.orders:
        if o.side == "BUY":
            entry[o.code] = (o.price, o.date)
        else:
            if o.code in entry:
                buy_price, buy_date = entry.pop(o.code)
                pnl = o.price / buy_price - 1
                pd_t1 = prev_date.get((o.code, buy_date))
                vr = vol_ratio[o.code].get(pd_t1) if pd_t1 else None
                vr_by_trade.append((o.code, buy_date, o.date, vr, pnl))

    print(f"总配对交易数: {len(vr_by_trade)}")
    valid = [x for x in vr_by_trade if x[3] is not None]
    print(f"有量比数据的交易: {len(valid)}")

    # 量比分桶
    buckets = [(0.0, 0.6, "<0.6 显著缩量"), (0.6, 0.8, "0.6-0.8 缩量"),
               (0.8, 1.0, "0.8-1.0 平量"), (1.0, 1.5, "1.0-1.5 放量"),
               (1.5, 9e9, ">1.5 显著放量")]
    print(f"\n{'量比区间':<18}{'笔数':>5}{'胜率%':>8}{'平均盈亏%':>10}{'中位盈亏%':>10}")
    print("-" * 54)
    for lo, hi, label in buckets:
        grp = [x[4] for x in valid if lo <= x[3] < hi]
        if not grp:
            print(f"{label:<18}{0:>5}{'-':>8}{'-':>10}{'-':>10}")
            continue
        win = sum(1 for p in grp if p > 0) / len(grp) * 100
        avg = sum(grp) / len(grp) * 100
        med = float(np.median(grp)) * 100
        print(f"{label:<18}{len(grp):>5}{win:>8.1f}{avg:>10.2f}{med:>10.2f}")

    # 缩量(<1.0) vs 放量(>=1.0) 二分
    print("\n二分对比:")
    for lo, hi, label in [(0.0, 1.0, "缩量 量比<1.0"), (1.0, 9e9, "放量 量比>=1.0")]:
        grp = [x[4] for x in valid if lo <= x[3] < hi]
        if not grp:
            continue
        win = sum(1 for p in grp if p > 0) / len(grp) * 100
        avg = sum(grp) / len(grp) * 100
        print(f"  {label}: {len(grp)}笔  胜率{win:.1f}%  平均{avg:+.2f}%")


if __name__ == "__main__":
    main()
