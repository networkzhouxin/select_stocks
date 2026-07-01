# -*- coding: utf-8 -*-
import re

from .models import Stock


def market_for_code(code):
    if code.startswith(("5", "6", "9")) or code.startswith("688"):
        return "SH"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    raise ValueError("无法判断市场，请输入6位A股代码或股票名称")


def resolve_numeric_code(text):
    code = text.strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("股票代码应为6位数字")
    market = market_for_code(code)
    prefix = "sh" if market == "SH" else "sz"
    return Stock(code=code, name=code, market=market, tencent_code=prefix + code)


def is_numeric_code(text):
    return re.fullmatch(r"\d{6}", text.strip()) is not None
