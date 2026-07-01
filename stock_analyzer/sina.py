# -*- coding: utf-8 -*-
from urllib.request import Request, urlopen

from .models import Quote


SINA_QUOTE_URL = "https://hq.sinajs.cn/list=%s"


def _to_float(value):
    try:
        if value in (None, "", "-"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sina_symbol(stock):
    return ("sh" if stock.market == "SH" else "sz") + stock.code


def parse_sina_quote_line(text, symbol):
    marker = 'hq_str_%s="' % symbol
    if marker not in text:
        return None
    start = text.find(marker) + len(marker)
    end = text.find('";', start)
    if end < 0:
        return None
    parts = text[start:end].split(",")
    if len(parts) < 33 or not parts[0]:
        return None
    price = _to_float(parts[3])
    prev_close = _to_float(parts[2])
    pct_change = None
    if price is not None and prev_close and prev_close > 0:
        pct_change = (price / prev_close - 1) * 100
    return Quote(
        code=symbol[2:],
        name=parts[0],
        price=price or 0.0,
        prev_close=prev_close,
        open=_to_float(parts[1]),
        high=_to_float(parts[4]),
        low=_to_float(parts[5]),
        pct_change=pct_change,
        timestamp=("%s %s" % (parts[30], parts[31])).strip(),
    )


def fetch_sina_quote(stock, timeout=10):
    symbol = _sina_symbol(stock)
    req = Request(
        SINA_QUOTE_URL % symbol,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn/",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("gb18030", errors="replace")
    quote = parse_sina_quote_line(text, symbol)
    if quote is None:
        raise ValueError("新浪行情返回为空或格式异常")
    return quote
