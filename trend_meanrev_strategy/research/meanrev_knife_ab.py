# -*- coding: utf-8 -*-
"""接飞刀防护 knife 变体 A/B：训练期 2017-2021 对比。"""

from __future__ import annotations

from collections import Counter

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

MODES = [
    ("ma60_slope（现状）", "ma60_slope"),
    ("ma60_slope_weak（+close<MA20）", "ma60_slope_weak"),
    ("new_low_10（创10日新低）", "new_low_10"),
    ("none（对照，无防护）", "none"),
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
        f"{'模式':<30}{'收益%':>9}{'回撤%':>8}{'买':>4}{'卖':>4}{'持仓':>5}"
        f"{'回归结%':>9}{'胜率%':>7}{'平均盈亏%':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, mode in MODES:
        s = MeanrevEngine(loader=loader, params={"knife_mode": mode}).run()
        reasons = Counter(o.reason for o in s.orders if o.side == "SELL")
        mr_pct = reasons.get("mean_revert", 0) / s.sell_count * 100 if s.sell_count else 0
        pnls = trade_pnl(s.orders)
        win = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        avg = sum(pnls) / len(pnls) * 100 if pnls else 0
        print(
            f"{label:<30}{s.total_return*100:>9.2f}{s.max_drawdown*100:>8.2f}"
            f"{s.buy_count:>4}{s.sell_count:>4}{s.max_holdings:>5}"
            f"{mr_pct:>9.1f}{win:>7.1f}{avg:>10.2f}"
        )


if __name__ == "__main__":
    main()
