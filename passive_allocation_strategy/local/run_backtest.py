# -*- coding: utf-8 -*-
"""被动配置 + 再平衡 本地回测入口。"""

from __future__ import annotations

from passive_allocation_strategy.local.data_loader import PassiveAllocationDataLoader
from passive_allocation_strategy.local.engine import RebalanceEngine


def main() -> None:
    engine = RebalanceEngine(loader=PassiveAllocationDataLoader())
    s = engine.run()

    print("=" * 70)
    print(f"被动配置+再平衡 回测: {s.start_date} ~ {s.end_date}")
    print(f"交易日数: {s.trading_days}")
    print(f"初始资金: {s.initial_cash:,.0f}")
    print(f"期末资产: {s.final_value:,.0f}")
    print(f"总收益: {s.total_return * 100:.2f}%")
    print(f"最大回撤: {s.max_drawdown * 100:.2f}%")
    print(f"再平衡次数: {s.rebalance_count}")
    print("分年度收益:")
    for y, r in s.annual_returns.items():
        print(f"  {y}: {r * 100:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
