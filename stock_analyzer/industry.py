# -*- coding: utf-8 -*-
import json
from urllib.request import Request, urlopen

from .models import IndustryInfo


def parse_company_survey_payload(text):
    payload = json.loads(text)
    rows = payload.get("jbzl") or []
    if not rows:
        return IndustryInfo()
    row = rows[0]
    return IndustryInfo(
        exchange_board=str(row.get("SECURITY_TYPE") or ""),
        market=str(row.get("TRADE_MARKET") or ""),
        csrc_industry=str(row.get("INDUSTRYCSRC1") or ""),
        eastmoney_industry=str(row.get("EM2016") or ""),
        profile=str(row.get("ORG_PROFILE") or ""),
    )


def eastmoney_code(stock):
    return "%s%s" % (stock.market, stock.code)


def fetch_industry_info(stock, timeout=10):
    url = (
        "https://emweb.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax"
        "?code=%s" % eastmoney_code(stock)
    )
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.eastmoney.com/",
    })
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return parse_company_survey_payload(data.decode("utf-8", errors="replace"))
