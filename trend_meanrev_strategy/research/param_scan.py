# -*- coding: utf-8 -*-
"""趋势腿参数敏感性扫描：一次只变一个参数，看默认值是否稳健/最优。"""

from __future__ import annotations

from collections import defaultdict

from trend_meanrev_strategy.local.data_loader import TrendMeanrevDataLoader
from trend_meanrev_strategy.local.engine import PARAMS, TrendLegEngine


def analyze(override: dict):
    e = TrendLegEngine(loader=TrendMeanrevDataLoader(), params=override)
    s = e.run()
    comm, mc = PARAMS["commission"], PARAMS["min_commission"]
    by_code = defaultdict(list)
    for o in s.orders:
        by_code[o.code].append(o)
    trades = []
    for code, ods in by_code.items():
        i = 0
        while i < len(ods):
            if ods[i].side == "BUY" and i + 1 < len(ods) and ods[i + 1].side == "SELL":
                b, se = ods[i], ods[i + 1]
                trades.append((se.value - max(se.value * comm, mc)) - (b.value + max(b.value * comm, mc)))
                i += 2
            else:
                i += 1
    w = [t for t in trades if t > 0]
    tp = sum(w)
    tl = abs(sum(t for t in trades if t <= 0))
    return dict(ret=s.total_return, dd=s.max_drawdown,
                win=len(w) / len(trades) if trades else 0,
                pf=tp / tl if tl else 999, buys=s.buy_count)


def show(name, override):
    r = analyze(override)
    print(f"{name:22s} 收益{r['ret']*100:+7.2f}% 回撤{r['dd']*100:5.2f}% 胜率{r['win']*100:5.1f}% 盈亏比{r['pf']:5.2f} 买{r['buys']}")


def main():
    base = analyze({})
    print(f"{'默认(bear_filter开)':22s} 收益{base['ret']*100:+7.2f}% 回撤{base['dd']*100:5.2f}% 胜率{base['win']*100:5.1f}% 盈亏比{base['pf']:5.2f} 买{base['buys']}")
    print()

    print("[新高天数 high_period]")
    for v in (10, 20, 30, 55):
        show(f"  {v}天", {"high_period": v})

    print("[均线组合 ma_fast/ma_slow]")
    for f, s in ((10, 50), (20, 50), (20, 60), (10, 60)):
        show(f"  MA{f}/MA{s}", {"ma_fast": f, "ma_slow": s})

    print("[ATR止损倍数]")
    for v in (2.0, 2.5, 3.0):
        show(f"  {v}x", {"trailing_atr_mult": v})

    print("[冷却天数]")
    for v in (3, 5, 7, 10):
        show(f"  {v}天", {"cooldown_days": v})

    print("[最低持有天数]")
    for v in (3, 5, 7, 10):
        show(f"  {v}天", {"min_hold_days": v})

    print("[防追高RSI阈值]")
    for v in (70, 75, 80, 999):
        show(f"  {v if v < 999 else '关闭'}", {"overheat_rsi": v})

    print("[最大持仓数]")
    for v in (2, 3, 4):
        show(f"  {v}只", {"max_hold": v})


if __name__ == "__main__":
    main()
