# -*- coding: utf-8 -*-
"""最低持有期 min_hold_days A/B：训练期 2017-2021 对比。

均值回归的本质是快进快出，min_hold=5 从趋势腿继承，可能卡住 1-3 天的快速回归。
"""

from __future__ import annotations

from collections import Counter

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

MIN_HOLDS = [5, 3, 2, 1, 0]


def trade_pnl(orders):
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
        f"{'min_hold':<10}{'收益%':>9}{'回撤%':>8}{'买':>4}{'卖':>4}{'持仓':>5}"
        f"{'回归结%':>9}{'胜率%':>7}{'平均盈亏%':>10}"
    )
    print(header)
    print("-" * len(header))
    for mh in MIN_HOLDS:
        s = MeanrevEngine(loader=loader, params={"min_hold_days": mh}).run()
        reasons = Counter(o.reason for o in s.orders if o.side == "SELL")
        mr_pct = reasons.get("mean_revert", 0) / s.sell_count * 100 if s.sell_count else 0
        pnls = trade_pnl(s.orders)
        win = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        avg = sum(pnls) / len(pnls) * 100 if pnls else 0
        print(
            f"{mh:<10}{s.total_return*100:>9.2f}{s.max_drawdown*100:>8.2f}"
            f"{s.buy_count:>4}{s.sell_count:>4}{s.max_holdings:>5}"
            f"{mr_pct:>9.1f}{win:>7.1f}{avg:>10.2f}"
        )


if __name__ == "__main__":
    main()
