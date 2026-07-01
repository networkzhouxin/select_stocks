# -*- coding: utf-8 -*-
import json
from urllib.request import Request, urlopen

from .models import FinancialQuarter, FinancialYear


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_financial_payload(text, limit=5):
    payload = json.loads(text)
    rows = payload.get("data") or []
    result = []
    for row in rows:
        if row.get("REPORT_TYPE") not in (None, "年报"):
            continue
        year = str(row.get("REPORT_YEAR") or "")[:4]
        if not year:
            continue
        result.append(FinancialYear(
            year=year,
            revenue=_to_float(row.get("TOTALOPERATEREVE")),
            parent_net_profit=_to_float(row.get("PARENTNETPROFIT")),
            deduct_net_profit=_to_float(row.get("KCFJCXSYJLR")),
            revenue_yoy=_to_float(row.get("TOTALOPERATEREVETZ")),
            net_profit_yoy=_to_float(row.get("PARENTNETPROFITTZ")),
            roe=_to_float(row.get("ROEJQ")),
            gross_margin=_to_float(row.get("XSMLL")),
            net_margin=_to_float(row.get("XSJLL")),
            debt_ratio=_to_float(row.get("ZCFZL")),
            eps=_to_float(row.get("EPSJB")),
            operating_cashflow_per_share=_to_float(row.get("MGJYXJJE")),
        ))
    result.sort(key=lambda x: x.year, reverse=True)
    return result[:limit]


def parse_quarterly_financial_payload(text, limit=8):
    payload = json.loads(text)
    rows = payload.get("data") or []
    result = []
    for row in rows:
        period = str(row.get("REPORT_DATE") or "")[:10]
        if not period:
            continue
        report_type = str(row.get("REPORT_TYPE") or "")
        if not report_type:
            continue
        result.append(FinancialQuarter(
            period=period,
            report_name=str(row.get("REPORT_DATE_NAME") or report_type),
            revenue=_to_float(row.get("TOTALOPERATEREVE")),
            parent_net_profit=_to_float(row.get("PARENTNETPROFIT")),
            deduct_net_profit=_to_float(row.get("KCFJCXSYJLR")),
            revenue_yoy=_to_float(row.get("TOTALOPERATEREVETZ")),
            net_profit_yoy=_to_float(row.get("PARENTNETPROFITTZ")),
            deduct_net_profit_yoy=_to_float(row.get("KCFJCXSYJLRTZ")),
            gross_margin=_to_float(row.get("XSMLL")),
            net_margin=_to_float(row.get("XSJLL")),
            operating_cashflow_per_share=_to_float(row.get("MGJYXJJE")),
        ))
    result.sort(key=lambda x: x.period, reverse=True)
    return result[:limit]


def eastmoney_code(stock):
    return "%s%s" % (stock.market, stock.code)


def fetch_financials(stock, timeout=10):
    url = (
        "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/"
        "ZYZBAjaxNew?type=1&code=%s" % eastmoney_code(stock)
    )
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/",
    })
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return parse_financial_payload(data.decode("utf-8", errors="replace"))


def fetch_quarterly_financials(stock, timeout=10):
    url = (
        "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/"
        "ZYZBAjaxNew?type=0&code=%s" % eastmoney_code(stock)
    )
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/",
    })
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return parse_quarterly_financial_payload(data.decode("utf-8", errors="replace"))
