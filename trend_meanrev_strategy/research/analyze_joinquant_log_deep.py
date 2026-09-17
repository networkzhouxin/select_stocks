# -*- coding: utf-8 -*-
"""训练期日志深度挖掘：回吐分析、买入质量、止损性质、持仓轨迹。"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

BUY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*\[买入\] (\S+) ROC20=([-\d.]+)% @([\d.]+)")
SELL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*\[卖出\] (\S+) 原因=(\S+) 现价([\d.]+) 成本([\d.]+) 盈亏([-\d.]+)%")
POS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*  (\S+) 成本([\d.]+) 现([\d.]+) 高([\d.]+) 盈亏([-\d.]+)%")


def parse(path):
    buys, sells = [], []
    positions = defaultdict(dict)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BUY_RE.match(line)
            if m:
                buys.append({"date": m.group(1), "code": m.group(2), "roc": float(m.group(3)), "price": float(m.group(4))})
                continue
            m = SELL_RE.match(line)
            if m:
                sells.append({"date": m.group(1), "code": m.group(2), "reason": m.group(3),
                              "price": float(m.group(4)), "cost": float(m.group(5)), "pnl": float(m.group(6))})
                continue
            m = POS_RE.match(line)
            if m:
                positions[m.group(2)][m.group(1)] = float(m.group(6))
    return buys, sells, positions


def pair_trades(buys, sells):
    open_buy = {}
    trades = []
    for ev in sorted(buys + sells, key=lambda x: (x["date"], 0 if "roc" in x else 1)):
        if "roc" in ev:
            open_buy[ev["code"]] = ev
        else:
            b = open_buy.pop(ev["code"], None)
            if b:
                trades.append({**ev, "buy_date": b["date"], "buy_roc": b["roc"], "buy_price": b["price"]})
    return trades


def main(path):
    buys, sells, positions = parse(path)
    trades = pair_trades(buys, sells)

    # 给每笔交易挂上持仓期峰值浮盈
    for t in trades:
        pd_ = positions.get(t["code"], {})
        keys = sorted(k for k in pd_ if t["buy_date"] <= k <= t["date"])
        pnls = [pd_[k] for k in keys]
        t["peak"] = max(pnls) if pnls else t["pnl"]
        t["trough"] = min(pnls) if pnls else t["pnl"]

    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]

    print("=" * 72)
    print("A) 回吐分析（利润有没有跑起来）")
    for label, grp in [("全部", trades), ("盈利单", winners)]:
        if not grp:
            continue
        peak = sum(t["peak"] for t in grp) / len(grp)
        final = sum(t["pnl"] for t in grp) / len(grp)
        giveback = sum(t["peak"] - t["pnl"] for t in grp) / len(grp)
        big_give = [t for t in grp if t["peak"] > 10 and t["pnl"] < t["peak"] * 0.5]
        print(f"   {label}: 平均峰值浮盈 {peak:+.1f}%, 平均实现 {final:+.1f}%, 平均回吐 {giveback:.1f}%")
        print(f"     峰值>10%但最终只实现不到一半的: {len(big_give)} 笔")

    print("\nB) 买入质量（盈利单 vs 亏损单的买入特征）")
    for label, grp in [("盈利单", winners), ("亏损单", losers)]:
        if not grp:
            continue
        roc = sum(t["buy_roc"] for t in grp) / len(grp) * 100
        print(f"   {label}: {len(grp)} 笔, 平均买入ROC20 {roc:+.1f}%")

    print("\nC) 止损性质（ATR 止损时的盈亏分布）")
    stops = [t for t in trades if t["reason"] == "atr_stop"]
    profitable_stops = [t for t in stops if t["pnl"] > 0]
    loss_stops = [t for t in stops if t["pnl"] <= 0]
    print(f"   ATR止损 {len(stops)} 笔: 盈利离场 {len(profitable_stops)} 笔(止盈), 亏损离场 {len(loss_stops)} 笔(割肉)")
    if profitable_stops:
        print(f"      止盈单平均 +{sum(t['pnl'] for t in profitable_stops)/len(profitable_stops):.1f}%")
    if loss_stops:
        print(f"      割肉单平均 {sum(t['pnl'] for t in loss_stops)/len(loss_stops):.1f}%")

    print("\nD) 亏损单的亏损过程（买入即亏 vs 先赚后亏）")
    immediate = [t for t in losers if t["peak"] <= 1.0]
    roundtrip = [t for t in losers if t["peak"] > 1.0]
    print(f"   买入后从未浮盈>1%就亏的: {len(immediate)} 笔 (买入即亏)")
    print(f"   曾经浮盈>1%然后转亏的: {len(roundtrip)} 笔 (先赚后亏)")

    print("\nE) 盈利单的持仓期峰值浮盈分布")
    buckets = {"<5%": 0, "5-10%": 0, "10-20%": 0, ">20%": 0}
    for t in winners:
        p = t["peak"]
        if p < 5:
            buckets["<5%"] += 1
        elif p < 10:
            buckets["5-10%"] += 1
        elif p < 20:
            buckets["10-20%"] += 1
        else:
            buckets[">20%"] += 1
    print("   " + "  ".join(f"{k}:{v}" for k, v in buckets.items()))
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1])
