# -*- coding: utf-8 -*-
"""波段化 A/B：探底回升买入 + 顶背离卖出 vs 现状共振 + 回归中轨。

用户方案：RSI 不超买（防追高已内置）+ 各项指标都探底回升（相对拐点三项齐全）买入，
明显顶背离卖出。拆成 4 组对照定位每个部件的影响。
"""

from __future__ import annotations

from collections import Counter

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

SCENARIOS = [
    ("A 现状（共振买入+回归中轨卖）", {}),
    ("B 只换买入（探底回升）", {"buy_mode": "relative"}),
    ("C 只换卖出（顶背离）", {"divergence_sell_enabled": True}),
    ("D 完整方案（探底回升+顶背离）", {"buy_mode": "relative", "divergence_sell_enabled": True}),
]


def trade_pnl(orders):
    """FIFO 配对算每笔盈亏（相对 entry_cost，不含手续费）。"""
    entry = {}
    pnls = []
    for o in orders:
        if o.side == "BUY":
            entry[o.code] = o.price
        else:
            if o.code in entry:
                pnls.append(o.price / entry[o.code] - 1)
                del entry[o.code]
    return pnls


def main() -> None:
    loader = TrendMeanrevDataLoader()
    header = (
        f"{'场景':<30}{'收益%':>8}{'回撤%':>8}{'买':>4}{'卖':>4}"
        f"{'顶背离卖%':>10}{'回归结%':>9}{'胜率%':>7}{'平均盈亏%':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, params in SCENARIOS:
        s = MeanrevEngine(loader=loader, params=params).run()
        reasons = Counter(o.reason for o in s.orders if o.side == "SELL")
        div_pct = reasons.get("divergence", 0) / s.sell_count * 100 if s.sell_count else 0
        mr_pct = reasons.get("mean_revert", 0) / s.sell_count * 100 if s.sell_count else 0
        pnls = trade_pnl(s.orders)
        win = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        avg = sum(pnls) / len(pnls) * 100 if pnls else 0
        print(
            f"{label:<30}{s.total_return*100:>8.2f}{s.max_drawdown*100:>8.2f}"
            f"{s.buy_count:>4}{s.sell_count:>4}{div_pct:>10.1f}{mr_pct:>9.1f}"
            f"{win:>7.1f}{avg:>10.2f}"
        )


if __name__ == "__main__":
    main()
