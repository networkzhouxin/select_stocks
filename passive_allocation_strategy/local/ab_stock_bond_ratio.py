# -*- coding: utf-8 -*-
"""被动配置 股债比例 A/B 扫描（2017-2021 训练期）。

新池 7 只，黄金固定 5%，股权内部结构固定（红利20/沪深300 10/创业板5/纳指15/标普10），
唯一变量是股权总权重 E，国债 = 95% - E。
另附旧池（6 只）@60% 作为参照行，验证改池子的价值。
"""

from __future__ import annotations

import numpy as np

from passive_allocation_strategy.local.data_loader import PassiveAllocationDataLoader
from passive_allocation_strategy.local.engine import RebalanceEngine

EQUITY_SLEEVE = {
    "510880": 0.20,   # 上证红利
    "510300": 0.10,   # 沪深300
    "159915": 0.05,   # 创业板
    "513100": 0.15,   # 纳指
    "513500": 0.10,   # 标普500
}
GOLD_CODE = "518880"
GOLD_WEIGHT = 0.05
BOND_CODE = "511010"

OLD_POOL = {
    "510880": 0.20,
    "510300": 0.15,
    "159915": 0.10,
    "513100": 0.15,
    "518880": 0.05,
    "511010": 0.35,
}

SLEEVE_TOTAL = sum(EQUITY_SLEEVE.values())
TRADING_DAYS_PER_YEAR = 244


def make_weights(equity_total: float) -> dict:
    scale = equity_total / SLEEVE_TOTAL
    w = {code: wt * scale for code, wt in EQUITY_SLEEVE.items()}
    w[GOLD_CODE] = GOLD_WEIGHT
    w[BOND_CODE] = 1.0 - equity_total - GOLD_WEIGHT
    return w


def run_scenario(loader, weights) -> dict:
    engine = RebalanceEngine(loader=loader, target_weights=weights)
    s = engine.run()
    eq = np.array([v for _, v in engine.equity], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    years = len(eq) / TRADING_DAYS_PER_YEAR
    annualized = (1 + s.total_return) ** (1 / years) - 1
    if len(rets) and rets.std() > 0:
        ann_vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        ann_vol = 0.0
        sharpe = 0.0
    rr = s.total_return / s.max_drawdown if s.max_drawdown > 0 else float("inf")
    return {
        "total": s.total_return,
        "annualized": annualized,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": s.max_drawdown,
        "rr": rr,
        "rebalance": s.rebalance_count,
        "annual": s.annual_returns,
    }


def main() -> None:
    loader = PassiveAllocationDataLoader()
    rows = []
    for e in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        label = f"股{round(e * 100)}/债{round((0.95 - e) * 100)}"
        rows.append((label, run_scenario(loader, make_weights(e))))
    rows.append(("旧池@60%", run_scenario(loader, OLD_POOL)))

    sep = "-" * 92
    print("=" * 92)
    print("被动配置 股债比例 A/B（2017-2021, 10万起始, 前复权, 黄金固定5%）")
    print("=" * 92)
    print(f"{'场景':<10}{'总收益':>9}{'年化':>8}{'年化波动':>9}{'夏普':>7}{'最大回撤':>9}{'收益/回撤':>9}{'再平衡':>7}")
    print(sep)
    for label, r in rows:
        rr_s = "inf" if r["rr"] == float("inf") else f"{r['rr']:.2f}"
        print(f"{label:<10}{r['total'] * 100:>8.2f}%{r['annualized'] * 100:>7.2f}%"
              f"{r['ann_vol'] * 100:>8.2f}%{r['sharpe']:>7.3f}"
              f"{r['max_dd'] * 100:>8.2f}%{rr_s:>9}{r['rebalance']:>7}")
    print(sep)

    print("\n分年度收益:")
    years = ["2017", "2018", "2019", "2020", "2021"]
    print(f"{'场景':<10}" + "".join(f"{y:>9}" for y in years))
    for label, r in rows:
        line = f"{label:<10}"
        for y in years:
            v = r["annual"].get(y)
            line += f"{v * 100:>8.2f}%" if v is not None else f"{'-':>9}"
        print(line)


if __name__ == "__main__":
    main()
