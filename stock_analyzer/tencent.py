# -*- coding: utf-8 -*-
import json
import re
from urllib.parse import quote
from urllib.request import urlopen

from .models import KLineRow, Quote, Stock


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_escaped(text):
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def decode_tencent_bytes(data):
    for encoding in ("gb18030", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_search_hint(text):
    match = re.search(r'v_hint="([^"]*)"', text)
    if not match:
        raise ValueError("腾讯搜索没有返回匹配股票")
    for item in match.group(1).split("^"):
        parts = item.split("~")
        if len(parts) < 5:
            continue
        market_raw, code, name, _, stock_type = parts[:5]
        if stock_type != "GP-A":
            continue
        market = "SH" if market_raw == "sh" else "SZ" if market_raw == "sz" else market_raw.upper()
        return Stock(
            code=code,
            name=_decode_escaped(name),
            market=market,
            tencent_code=market_raw + code,
        )
    raise ValueError("腾讯搜索结果中没有A股股票")


def parse_quote_line(text):
    match = re.search(r'v_[a-z]{2}\d{6}="([^"]*)"', text)
    if not match:
        raise ValueError("腾讯行情返回格式无法解析")
    fields = match.group(1).split("~")
    if len(fields) < 47:
        raise ValueError("腾讯行情字段不完整")

    return Quote(
        name=fields[1],
        code=fields[2],
        price=_to_float(fields[3]) or 0.0,
        prev_close=_to_float(fields[4]),
        open=_to_float(fields[5]),
        timestamp=fields[30],
        change=_to_float(fields[31]),
        pct_change=_to_float(fields[32]),
        high=_to_float(fields[33]),
        low=_to_float(fields[34]),
        turnover=_to_float(fields[38]),
        pe=_to_float(fields[39]),
        market_cap=_to_float(fields[44]),
        pb=_to_float(fields[46]),
    )


def parse_kline_payload(text, tencent_code):
    payload = json.loads(text)
    stock_data = payload.get("data", {}).get(tencent_code, {})
    rows = stock_data.get("qfqday") or stock_data.get("day") or []
    parsed = []
    for row in rows:
        if len(row) < 6:
            continue
        parsed.append(KLineRow(
            date=row[0],
            open=float(row[1]),
            close=float(row[2]),
            high=float(row[3]),
            low=float(row[4]),
            volume=float(row[5]),
        ))
    return parsed


def fetch_text(url, timeout=8):
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    return decode_tencent_bytes(data)


def search_stock_by_name(name):
    url = "https://smartbox.gtimg.cn/s3/?v=2&q=%s&t=all" % quote(name)
    return parse_search_hint(fetch_text(url))


def fetch_quote(tencent_code):
    url = "https://qt.gtimg.cn/q=%s" % tencent_code
    return parse_quote_line(fetch_text(url))


def fetch_kline(tencent_code, count=120):
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param=%s,day,,,%d,qfq" % (tencent_code, count)
    )
    return parse_kline_payload(fetch_text(url), tencent_code)
