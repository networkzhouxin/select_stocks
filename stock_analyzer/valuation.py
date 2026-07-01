# -*- coding: utf-8 -*-
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import ValuationHistoryAnalysis, ValuationRow, ValuationWindowAnalysis


EASTMONEY_VALUATION_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _to_float(value):
    try:
        if value in (None, "", "-"):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _short_date(value):
    return str(value or "")[:10]


def parse_valuation_payload(text):
    payload = json.loads(text)
    result = payload.get("result") or {}
    rows = result.get("data") or []
    parsed = []
    for row in rows:
        pe = _to_float(row.get("PE_TTM"))
        pb = _to_float(row.get("PB_MRQ"))
        if pe is None and pb is None:
            continue
        parsed.append(ValuationRow(
            date=_short_date(row.get("TRADE_DATE")),
            pe_ttm=pe,
            pb=pb,
            close_price=_to_float(row.get("CLOSE_PRICE")),
            board_code=str(row.get("BOARD_CODE") or ""),
            board_name=str(row.get("BOARD_NAME") or ""),
        ))
    return parsed


def _percentile(values, current):
    cleaned = sorted([v for v in values if v is not None and v > 0])
    if not cleaned or current is None or current <= 0:
        return None
    below = sum(1 for v in cleaned if v < current)
    equal = sum(1 for v in cleaned if v == current)
    return (below + equal * 0.5) / len(cleaned)


def _zone(value):
    if value is None:
        return "N/A"
    if value <= 0.2:
        return "低估区"
    if value <= 0.4:
        return "中低区"
    if value <= 0.6:
        return "中位区"
    if value <= 0.8:
        return "中高区"
    return "高估区"


def _range(values):
    cleaned = [v for v in values if v is not None and v > 0]
    if not cleaned:
        return None, None
    return min(cleaned), max(cleaned)


WINDOWS = (
    ("3年", 750),
    ("5年", 1250),
    ("10年", 2500),
)


def _analyze_window(label, expected_size, rows, current_pe, current_pb):
    pe_values = [row.pe_ttm for row in rows]
    pb_values = [row.pb for row in rows]
    pe_percentile = _percentile(pe_values, current_pe)
    pb_percentile = _percentile(pb_values, current_pb)
    pe_min, pe_max = _range(pe_values)
    pb_min, pb_max = _range(pb_values)

    return ValuationWindowAnalysis(
        label=label,
        sample_size=len(rows),
        expected_size=expected_size,
        is_full_window=len(rows) >= expected_size,
        pe_percentile=round(pe_percentile, 6) if pe_percentile is not None else None,
        pb_percentile=round(pb_percentile, 6) if pb_percentile is not None else None,
        pe_min=round(pe_min, 2) if pe_min is not None else None,
        pe_max=round(pe_max, 2) if pe_max is not None else None,
        pb_min=round(pb_min, 2) if pb_min is not None else None,
        pb_max=round(pb_max, 2) if pb_max is not None else None,
    )


def _primary_metric(category_code):
    if category_code == "dividend_stable":
        return "股息率"
    if category_code == "turnaround_watch":
        return "PB"
    if category_code == "growth":
        return "PE/成长匹配"
    return "PE/PB"


def _framework_comment(category_code):
    if category_code == "dividend_stable":
        return "红利稳定型以股息率、分红率和现金流覆盖为主，PE/PB用于判断是否明显偏离自身历史。"
    if category_code == "turnaround_watch":
        return "困境反转型利润偏弱时PE会失真，优先看PB、现金流和反转证据，PE只作风险提示。"
    if category_code == "growth":
        return "成长型不能只看PE高低，要结合收入利润增速、ROE和现金流兑现能力。"
    return "综合观察型同时参考PE、PB、财务质量和技术位置，不用单一指标下结论。"


def analyze_valuation_history(rows, current_pe=None, current_pb=None, category_code=""):
    rows = list(rows or [])
    windows = {}
    for label, size in WINDOWS:
        windows[label] = _analyze_window(label, size, rows[:size], current_pe, current_pb)

    primary_window = windows["3年"]
    parts = []
    if primary_window.pe_percentile is not None:
        parts.append("PE历史分位%.1f%%（%s）" % (
            primary_window.pe_percentile * 100,
            _zone(primary_window.pe_percentile),
        ))
    if primary_window.pb_percentile is not None:
        parts.append("PB历史分位%.1f%%（%s）" % (
            primary_window.pb_percentile * 100,
            _zone(primary_window.pb_percentile),
        ))

    return ValuationHistoryAnalysis(
        sample_size=len(rows),
        pe_percentile=primary_window.pe_percentile,
        pb_percentile=primary_window.pb_percentile,
        pe_min=primary_window.pe_min,
        pe_max=primary_window.pe_max,
        pb_min=primary_window.pb_min,
        pb_max=primary_window.pb_max,
        windows=windows,
        primary_metric=_primary_metric(category_code),
        framework_comment=_framework_comment(category_code),
        comment="；".join(parts) if parts else "暂缺PE/PB历史估值分位数据。",
    )


def fetch_valuation_history(stock, page_size=1000, max_pages=3, timeout=10):
    all_rows = []
    total_pages = None
    for page in range(1, max_pages + 1):
        rows, pages = _fetch_valuation_history_page(stock, page, page_size, timeout)
        if total_pages is None:
            total_pages = pages
        if not rows:
            break
        all_rows.extend(rows)
        if total_pages is not None and page >= total_pages:
            break
    return all_rows


def fetch_industry_valuation_rows(board_code, trade_date, page_size=200, timeout=10):
    if not board_code or not trade_date:
        return []
    params = {
        "reportName": "RPT_VALUEANALYSIS_DET",
        "columns": "ALL",
        "quoteColumns": "",
        "pageNumber": 1,
        "pageSize": page_size,
        "sortColumns": "PE_TTM",
        "sortTypes": 1,
        "source": "WEB",
        "client": "WEB",
        "filter": "(TRADE_DATE='%s')(BOARD_CODE=\"%s\")" % (trade_date, board_code),
    }
    url = "%s?%s" % (EASTMONEY_VALUATION_URL, urlencode(params))
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/gzfx/",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return parse_valuation_payload(raw)


def _fetch_valuation_history_page(stock, page_number, page_size, timeout):
    params = {
        "reportName": "RPT_VALUEANALYSIS_DET",
        "columns": "ALL",
        "quoteColumns": "",
        "pageNumber": page_number,
        "pageSize": page_size,
        "sortColumns": "TRADE_DATE",
        "sortTypes": -1,
        "source": "WEB",
        "client": "WEB",
        "filter": '(SECURITY_CODE="%s")' % stock.code,
    }
    url = "%s?%s" % (EASTMONEY_VALUATION_URL, urlencode(params))
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/gzfx/detail/%s.html" % stock.code,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    pages = payload.get("result", {}).get("pages")
    return parse_valuation_payload(raw), pages
