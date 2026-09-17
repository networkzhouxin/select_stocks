# -*- coding: utf-8 -*-
"""解析聚宽回测日志，产出趋势腿的 8 项日志挖掘分析。"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

BUY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*\[买入\] (\S+) ROC20=([-\d.]+)% @([\d.]+)")
SELL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*\[卖出\] (\S+) 原因=(\S+) 现价([\d.]+) 成本([\d.]+) 盈亏([-\d.]+)%")
CLOSE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*\[收盘\] 总值(\d+) 现金(\d+) 持仓(\d)/3")
POS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}).*  (\S+) 成本([\d.]+) 现([\d.]+) 高([\d.]+) 盈亏([-\d.]+)%")


def parse(path):
    buys, sells, closes, positions = [], [], [], defaultdict(dict)
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
            m = CLOSE_RE.match(line)
            if m:
                closes.append({"date": m.group(1), "total": int(m.group(2)), "cash": int(m.group(3)), "hold": int(m.group(4))})
                continue
            m = POS_RE.match(line)
            if m:
                positions[m.group(2)][m.group(1)] = float(m.group(6))
    return buys, sells, closes, positions


def trading_days(closes):
    return [c["date"] for c in closes]


def main(path):
    buys, sells, closes, positions = parse(path)
    cal = trading_days(closes)
    idx = {d: i for i, d in enumerate(cal)}

    print("=" * 70)
    print("1) 分年度表现")
    by_year = defaultdict(list)
    for c in closes:
        by_year[c["date"][:4]].append(c)
    prev_end = None
    for y in sorted(by_year):
        end_total = by_year[y][-1]["total"]
        start_total = prev_end if prev_end else 20000
        ret = end_total / start_total - 1
        print(f"   {y}: 期末总值 {end_total:,}  年收益 {ret*100:+.1f}%")
        prev_end = end_total

    print("\n2) 卖出原因分布")
    by_reason = defaultdict(list)
    for s in sells:
        by_reason[s["reason"]].append(s["pnl"])
    for r, pnls in sorted(by_reason.items()):
        wins = [p for p in pnls if p > 0]
        avg = sum(pnls) / len(pnls)
        print(f"   {r}: {len(pnls)} 笔, 胜率 {len(wins)/len(pnls)*100:.1f}%, 平均盈亏 {avg:+.2f}%")

    print("\n3) 持仓时长分布（赢单 vs 输单，交易日）")
    open_pos = {}
    trades = []
    for ev in sorted(buys + sells, key=lambda x: (x["date"], 0 if "roc" in x else 1)):
        if "roc" in ev:
            open_pos[ev["code"]] = ev
        else:
            b = open_pos.pop(ev["code"], None)
            if b:
                held = idx.get(ev["date"], 0) - idx.get(b["date"], 0)
                trades.append({"held": held, "pnl": ev["pnl"], "code": ev["code"], "reason": ev["reason"]})
    win_held = [t["held"] for t in trades if t["pnl"] > 0]
    loss_held = [t["held"] for t in trades if t["pnl"] <= 0]
    if win_held:
        print(f"   赢单: {len(win_held)} 笔, 平均持仓 {sum(win_held)/len(win_held):.1f} 天, 中位 {sorted(win_held)[len(win_held)//2]} 天")
    if loss_held:
        print(f"   输单: {len(loss_held)} 笔, 平均持仓 {sum(loss_held)/len(loss_held):.1f} 天, 中位 {sorted(loss_held)[len(loss_held)//2]} 天")

    print("\n4) 单笔盈亏分布")
    pnls = sorted(t["pnl"] for t in trades)
    if pnls:
        print(f"   最大单笔盈利 {pnls[-1]:+.1f}%, 最大单笔亏损 {pnls[0]:+.1f}%")
        big_win = [p for p in pnls if p > 10]
        big_loss = [p for p in pnls if p < -8]
        print(f"   盈利>10% 的单: {len(big_win)} 笔 (合计 {sum(big_win):+.0f}%)")
        print(f"   亏损<-8% 的单: {len(big_loss)} 笔 (合计 {sum(big_loss):+.0f}%)")

    print("\n5) 分品种表现")
    by_code = defaultdict(list)
    for t in trades:
        by_code[t["code"]].append(t)
    for code in sorted(by_code):
        ts = by_code[code]
        wins = [t for t in ts if t["pnl"] > 0]
        stops = [t for t in ts if t["reason"] == "atr_stop"]
        avg = sum(t["pnl"] for t in ts) / len(ts)
        print(f"   {code}: {len(ts)} 笔, 胜率 {len(wins)/len(ts)*100:.0f}%, 平均 {avg:+.2f}%, ATR止损 {len(stops)} 笔")

    print("\n6) 连续亏损段")
    streak = max_streak = 0
    cur_sum = worst_sum = 0.0
    for t in trades:
        if t["pnl"] <= 0:
            streak += 1
            cur_sum += t["pnl"]
            max_streak = max(max_streak, streak)
            worst_sum = min(worst_sum, cur_sum)
        else:
            streak = 0
            cur_sum = 0.0
    print(f"   最长连亏 {max_streak} 笔, 连亏段最大累计 {worst_sum:+.1f}%")

    print("\n7) 空仓时间占比")
    empty = sum(1 for c in closes if c["hold"] == 0)
    print(f"   空仓天数 {empty}/{len(closes)} = {empty/len(closes)*100:.1f}%")
    partial = sum(1 for c in closes if 0 < c["hold"] < 3)
    print(f"   未满仓(1-2只)天数 {partial}/{len(closes)} = {partial/len(closes)*100:.1f}%")

    print("\n8) 买入后 N 天平均盈亏%")
    for n in (1, 3, 5, 10):
        vals = []
        for b in buys:
            pd_ = positions.get(b["code"], {})
            bi = idx.get(b["date"])
            if bi is None or bi + n >= len(cal):
                continue
            target = cal[bi + n]
            if target in pd_:
                vals.append(pd_[target])
        if vals:
            print(f"   +{n} 天: 平均 {sum(vals)/len(vals):+.2f}%  (样本 {len(vals)})")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1])
