# -*- coding: utf-8 -*-
import json
from urllib.request import Request, urlopen

from .models import FundFlowAnalysis, FundFlowRow


EASTMONEY_FUND_FLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
TENCENT_PANKOU_URL = "https://qt.gtimg.cn/q=s_pk%s"


def _to_float(value):
    try:
        if value in (None, "", "-"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_fund_flow_payload(text):
    payload = json.loads(text)
    rows = (payload.get("data") or {}).get("klines") or []
    parsed = []
    for raw in rows:
        parts = str(raw).split(",")
        if len(parts) < 13:
            continue
        parsed.append(FundFlowRow(
            date=parts[0],
            main_net=_to_float(parts[1]),
            super_large_net=_to_float(parts[2]),
            large_net=_to_float(parts[3]),
            medium_net=_to_float(parts[4]),
            small_net=_to_float(parts[5]),
            main_pct=_to_float(parts[6]),
            close=_to_float(parts[11]),
            pct_change=_to_float(parts[12]),
        ))
    parsed.sort(key=lambda row: row.date)
    return parsed


def _sum_main(rows, n):
    values = [row.main_net for row in rows[-n:] if row.main_net is not None]
    return sum(values) if values else None


def _positive_days(rows, n):
    return sum(1 for row in rows[-n:] if row.main_net is not None and row.main_net > 0)


def analyze_fund_flow(rows):
    rows = list(rows or [])
    if not rows:
        return FundFlowAnalysis()
    net_5 = _sum_main(rows, 5)
    net_10 = _sum_main(rows, 10)
    net_20 = _sum_main(rows, 20)
    pos_5 = _positive_days(rows, 5)
    pos_10 = _positive_days(rows, 10)

    prev_5 = _sum_main(rows[:-5], 5) if len(rows) >= 10 else None
    if net_5 is not None and net_5 > 0 and pos_5 >= 3:
        trend = "连续流入"
    elif net_5 is not None and prev_5 is not None and net_5 > prev_5:
        trend = "流入改善"
    elif net_5 is not None and net_5 < 0 and pos_5 <= 1:
        trend = "持续流出"
    else:
        trend = "分歧"

    unit = 100000000.0
    parts = []
    if net_5 is not None:
        parts.append("5日主力净额%.2f亿" % (net_5 / unit))
    if net_10 is not None:
        parts.append("10日主力净额%.2f亿" % (net_10 / unit))
    parts.append("近5日流入天数%d天" % pos_5)
    return FundFlowAnalysis(
        rows=rows,
        net_5=round(net_5, 2) if net_5 is not None else None,
        net_10=round(net_10, 2) if net_10 is not None else None,
        net_20=round(net_20, 2) if net_20 is not None else None,
        positive_days_5=pos_5,
        positive_days_10=pos_10,
        trend=trend,
        comment="；".join(parts),
        source="东方财富资金流",
    )


def parse_tencent_pankou_payload(text):
    if not text or "none_match" in text:
        return None
    start = text.find('"')
    end = text.rfind('"')
    if start < 0 or end <= start:
        return None
    parts = text[start + 1:end].split("~")
    if len(parts) < 4:
        return None
    values = [_to_float(part) for part in parts[:4]]
    if any(value is None for value in values):
        return None
    return {
        "buy_large_ratio": values[0],
        "buy_small_ratio": values[1],
        "sell_large_ratio": values[2],
        "sell_small_ratio": values[3],
    }


def analyze_tencent_pankou(ratios):
    if not ratios:
        return FundFlowAnalysis()
    buy_large = ratios.get("buy_large_ratio")
    buy_small = ratios.get("buy_small_ratio")
    sell_large = ratios.get("sell_large_ratio")
    sell_small = ratios.get("sell_small_ratio")
    buy_total = (buy_large or 0) + (buy_small or 0)
    sell_total = (sell_large or 0) + (sell_small or 0)
    diff = buy_total - sell_total
    if diff >= 0.05:
        trend = "盘口偏买"
    elif diff <= -0.05:
        trend = "盘口偏卖"
    else:
        trend = "盘口均衡"
    comment = (
        "腾讯盘口比例兜底：买盘%.1f%%，卖盘%.1f%%；"
        "该数据只反映盘口买卖力量，不等同于历史主力资金流。"
    ) % (buy_total * 100, sell_total * 100)
    return FundFlowAnalysis(
        trend=trend,
        comment=comment,
        source="腾讯盘口比例",
        fallback_source="腾讯s_pk",
        buy_large_ratio=buy_large,
        buy_small_ratio=buy_small,
        sell_large_ratio=sell_large,
        sell_small_ratio=sell_small,
        pankou_comment=comment,
    )


def _secid(stock):
    market_id = "1" if stock.market == "SH" else "0"
    return "%s.%s" % (market_id, stock.code)


def fetch_fund_flow(stock, limit=60, timeout=10):
    url = (
        "%s?lmt=%d&klt=101&secid=%s&fields1=f1,f2,f3,f7"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
    ) % (EASTMONEY_FUND_FLOW_URL, limit, _secid(stock))
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return parse_fund_flow_payload(raw)


def fetch_tencent_pankou_flow(stock, timeout=10):
    symbol = getattr(stock, "tencent_code", "")
    url = TENCENT_PANKOU_URL % symbol
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://stockapp.finance.qq.com/",
    })
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("gb18030", errors="replace")
    ratios = parse_tencent_pankou_payload(raw)
    return analyze_tencent_pankou(ratios)
