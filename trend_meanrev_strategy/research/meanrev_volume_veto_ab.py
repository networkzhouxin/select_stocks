# -*- coding: utf-8 -*-
"""量能否决 A/B：训练期 2017-2021，量比≥阈值不买 vs 现状。

预登记假设：量能单调有效（缩量越显著越优），但做成否决门槛会砍掉 44% 盈利单，
预期负优化。本实验实证该判断。
"""

from __future__ import annotations

from collections import Counter

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

SCENARIOS = [
    ("现状（不否决）", {}),
    ("量比≥1.0 不买", {"volume_veto_enabled": True, "volume_ratio_threshold": 1.0}),
    ("量比≥0.8 不买", {"volume_veto_enabled": True, "volume_ratio_threshold": 0.8}),
    ("量比≥1.5 不买", {"volume_veto_enabled": True, "volume_ratio_threshold": 1.5}),
]


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
        f"{'场景':<18}{'收益%':>9}{'回撤%':>8}{'买':>4}{'卖':>4}{'持仓':>5}"
        f"{'回归结%':>9}{'胜率%':>7}{'平均盈亏%':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, params in SCENARIOS:
        s = MeanrevEngine(loader=loader, params=params).run()
        reasons = Counter(o.reason for o in s.orders if o.side == "SELL")
        mr_pct = reasons.get("mean_revert", 0) / s.sell_count * 100 if s.sell_count else 0
        pnls = trade_pnl(s.orders)
        win = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        avg = sum(pnls) / len(pnls) * 100 if pnls else 0
        print(
            f"{label:<18}{s.total_return*100:>9.2f}{s.max_drawdown*100:>8.2f}"
            f"{s.buy_count:>4}{s.sell_count:>4}{s.max_holdings:>5}"
            f"{mr_pct:>9.1f}{win:>7.1f}{avg:>10.2f}"
        )


if __name__ == "__main__":
    main()
