# -*- coding: utf-8 -*-
"""国债 511010 纳入重测：训练期 2017-2021，3 只 vs 3+国债。

背景：数据根分钟 09:35 数据已补齐（2017-2021 全覆盖），
立项方案里"国债分钟缺失、暂缓纳入"的前提已不存在，重测确认纳入或排除。
"""

from __future__ import annotations

from collections import Counter

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.meanrev_engine import MeanrevEngine

BASE_POOL = ["518880", "510880", "159985"]
SCENARIOS = [
    ("3只（现状）", BASE_POOL),
    ("3只+国债511010", BASE_POOL + ["511010"]),
    ("3只+国债+有色159980", BASE_POOL + ["511010", "159980"]),
    ("3只+国债+有色+可转债511380", BASE_POOL + ["511010", "159980", "511380"]),
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
        f"{'池':<26}{'收益%':>9}{'回撤%':>8}{'买':>4}{'卖':>4}{'持仓':>5}"
        f"{'回归结%':>9}{'胜率%':>7}{'平均盈亏%':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, pool in SCENARIOS:
        s = MeanrevEngine(loader=loader, pool=pool).run()
        reasons = Counter(o.reason for o in s.orders if o.side == "SELL")
        mr_pct = reasons.get("mean_revert", 0) / s.sell_count * 100 if s.sell_count else 0
        pnls = trade_pnl(s.orders)
        win = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        avg = sum(pnls) / len(pnls) * 100 if pnls else 0
        print(
            f"{label:<26}{s.total_return*100:>9.2f}{s.max_drawdown*100:>8.2f}"
            f"{s.buy_count:>4}{s.sell_count:>4}{s.max_holdings:>5}"
            f"{mr_pct:>9.1f}{win:>7.1f}{avg:>10.2f}"
        )

    # 单独看国债自身贡献：3+国债 池里 511010 的交易明细
    print("\n国债 511010 在 3+国债 池中的交易明细:")
    s4 = MeanrevEngine(loader=loader, pool=BASE_POOL + ["511010"]).run()
    for o in s4.orders:
        if o.code == "511010":
            print(f"  {o.date}  {o.side:4s}  @{o.price:.3f}  金额{o.value:,.0f}  {o.reason}")


if __name__ == "__main__":
    main()
