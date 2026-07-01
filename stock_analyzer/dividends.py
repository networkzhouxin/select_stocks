# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dividends.json")


@dataclass
class DividendRecord:
    year: str
    cash_dividend_per_10: float
    plan_notice_date: str = ""
    ex_dividend_date: str = ""
    progress: str = ""
    plan: str = ""


@dataclass
class AnnualDividend:
    year: str
    cash_dividend_per_10: float
    cash_dividend_per_share: float
    record_count: int


def load_dividends(path=DATA_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dividend_per_share(code, dividends=None):
    data = dividends if dividends is not None else load_dividends()
    item = data.get(code)
    if not item:
        return None
    value = item.get("last_cash_dividend_per_share")
    return float(value) if value is not None else None


def get_dividend_note(code, dividends=None):
    data = dividends if dividends is not None else load_dividends()
    item = data.get(code)
    if not item:
        return ""
    return item.get("note", "")


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _year_from_report_date(value):
    return str(value or "")[:4]


def parse_bonus_payload(text):
    payload = json.loads(text)
    rows = payload.get("result", {}).get("data") or []
    records = []
    for row in rows:
        cash_per_10 = _to_float(row.get("PRETAX_BONUS_RMB"))
        year = _year_from_report_date(row.get("REPORT_DATE"))
        if not year or cash_per_10 is None or cash_per_10 <= 0:
            continue
        records.append(DividendRecord(
            year=year,
            cash_dividend_per_10=cash_per_10,
            plan_notice_date=str(row.get("PLAN_NOTICE_DATE") or ""),
            ex_dividend_date=str(row.get("EX_DIVIDEND_DATE") or ""),
            progress=str(row.get("ASSIGN_PROGRESS") or ""),
            plan=str(row.get("IMPL_PLAN_PROFILE") or ""),
        ))
    records.sort(key=lambda r: (r.year, r.plan_notice_date), reverse=True)
    return records


def summarize_dividend_history(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record.year, []).append(record)
    summary = []
    for year, items in grouped.items():
        cash_per_10 = round(sum(item.cash_dividend_per_10 for item in items), 4)
        summary.append(AnnualDividend(
            year=year,
            cash_dividend_per_10=cash_per_10,
            cash_dividend_per_share=round(cash_per_10 / 10.0, 6),
            record_count=len(items),
        ))
    summary.sort(key=lambda r: r.year, reverse=True)
    return summary


def latest_annual_dividend_per_share(summary):
    if not summary:
        return None
    return summary[0].cash_dividend_per_share


def fetch_bonus_history(stock, timeout=10):
    filter_expr = '(SECURITY_CODE="%s")' % stock.code
    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get"
        "?sortColumns=PLAN_NOTICE_DATE&sortTypes=-1&pageSize=50&pageNumber=1"
        "&reportName=RPT_SHAREBONUS_DET&columns=ALL&source=WEB&client=WEB"
        "&filter=%s" % quote(filter_expr, safe="()=")
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return parse_bonus_payload(data.decode("utf-8", errors="replace"))
