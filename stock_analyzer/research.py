# -*- coding: utf-8 -*-
import json
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import ResearchAnalysis, ResearchReport


EASTMONEY_RESEARCH_URL = "https://reportapi.eastmoney.com/report/list"


def _short_date(value):
    return str(value or "")[:10]


def _report_url(info_code):
    if not info_code:
        return ""
    return "https://data.eastmoney.com/report/zw_stock.jshtml?encodeUrl=%s" % info_code


def parse_research_payload(text):
    payload = json.loads(text)
    rows = payload.get("data") or payload.get("result", {}).get("data") or []
    reports = []
    for row in rows:
        title = str(row.get("title") or row.get("TITLE") or "").strip()
        if not title:
            continue
        info_code = str(row.get("infoCode") or row.get("INFO_CODE") or "")
        reports.append(ResearchReport(
            date=_short_date(row.get("publishDate") or row.get("PUBLISH_DATE")),
            title=title,
            org=str(row.get("orgSName") or row.get("orgName") or row.get("ORG_S_NAME") or ""),
            rating=str(row.get("emRatingName") or row.get("rating") or row.get("RATING_NAME") or ""),
            analyst=str(row.get("researcher") or row.get("author") or row.get("ANALYST") or ""),
            url=_report_url(info_code),
            summary=str(row.get("summary") or row.get("SUMMARY") or ""),
        ))
    return reports


def analyze_research_reports(reports):
    reports = list(reports or [])
    if not reports:
        return ResearchAnalysis()
    rating_counts = {}
    for report in reports:
        if report.rating:
            rating_counts[report.rating] = rating_counts.get(report.rating, 0) + 1
    rating_summary = "；".join(
        "%s%d篇" % (rating, count)
        for rating, count in sorted(rating_counts.items(), key=lambda item: item[1], reverse=True)
    )
    if rating_summary:
        comment = "近期券商研报共%d篇，评级分布：%s。研报观点只能作为预期差线索。" % (
            len(reports), rating_summary)
    else:
        comment = "近期有%d篇研报，但缺少可解析评级，重点看标题和盈利预测是否变化。" % len(reports)
    return ResearchAnalysis(
        reports=reports,
        report_count=len(reports),
        latest_rating=reports[0].rating,
        rating_summary=rating_summary,
        comment=comment,
    )


def fetch_research_reports(stock, page_size=10, timeout=10, days=365):
    end = date.today()
    begin = end - timedelta(days=days)
    params = urlencode({
        "pageNo": "1",
        "pageSize": str(page_size),
        "code": stock.code,
        "industryCode": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": begin.isoformat(),
        "endTime": end.isoformat(),
        "fields": "",
        "qType": "0",
    })
    req = Request(
        EASTMONEY_RESEARCH_URL + "?" + params,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return parse_research_payload(raw)
