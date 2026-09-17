# -*- coding: utf-8 -*-
"""趋势腿训练期本地回放入口（2017-2021）。"""

from __future__ import annotations

from trend_meanrev_strategy.local.data_loader import (
    EXTRA_CODES, TREND_POOL, TrendMeanrevDataLoader,
)
from trend_meanrev_strategy.local.engine import TrendLegEngine


def main() -> None:
    loader = TrendMeanrevDataLoader()
    engine = TrendLegEngine(loader=loader, pool=TREND_POOL + EXTRA_CODES)
    summary = engine.run()

    print("=" * 70)
    print(f"训练期回放: {summary.start_date} ~ {summary.end_date}")
    print(f"交易日数: {summary.trading_days}")
    print(f"初始资金: {summary.start_value:,.0f}")
    print(f"期末资产: {summary.end_value:,.0f}")
    print(f"总收益: {summary.total_return * 100:.2f}%")
    print(f"最大回撤: {summary.max_drawdown * 100:.2f}%")
    print(f"买入: {summary.buy_count}  卖出: {summary.sell_count}  最大持仓: {summary.max_holdings}")
    print(f"期末持仓: {summary.final_holdings}")
    print("=" * 70)

    print("\n订单明细（前 40 条）:")
    for o in summary.orders[:40]:
        print(f"  {o.date}  {o.side:4s}  {o.code}  @{o.price:.3f}  金额{o.value:,.0f}  {o.reason}")


if __name__ == "__main__":
    main()
