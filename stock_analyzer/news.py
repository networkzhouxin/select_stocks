# -*- coding: utf-8 -*-
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import NewsAnalysis, NewsItem


EASTMONEY_NEWS_URL = "https://np-listapi.eastmoney.com/comm/web/getListInfo"

POSITIVE_KEYWORDS = (
    "回购", "增持", "分红", "派息", "中标", "订单", "合同", "预增", "扭亏",
    "增长", "创新高", "突破", "并购", "重组", "补助", "股东回报",
)
NEGATIVE_KEYWORDS = (
    "减持", "处罚", "立案", "诉讼", "仲裁", "问询", "亏损", "下滑", "暴雷",
    "退市", "ST", "质押", "担保", "逾期", "风险提示", "监管函",
)


def _short_date(value):
    return str(value or "")[:10]


def classify_news_title(title):
    text = title or ""
    tags = []
    sentiment = "neutral"
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in text:
            tags.append(keyword)
            sentiment = "negative"
    for keyword in POSITIVE_KEYWORDS:
        if keyword in text:
            tags.append(keyword)
            if sentiment != "negative":
                sentiment = "positive"
    return sentiment, tags


def parse_news_payload(text):
    payload = json.loads(text)
    data = payload.get("data") or {}
    if isinstance(data, dict):
        rows = data.get("list") or data.get("data") or []
    else:
        rows = data if isinstance(data, list) else []
    items = []
    for row in rows:
        title = str(row.get("Art_Title") or row.get("title") or row.get("Title") or "").strip()
        if not title:
            continue
        sentiment, tags = classify_news_title(title)
        items.append(NewsItem(
            date=_short_date(row.get("Art_ShowTime") or row.get("showTime") or row.get("date")),
            title=title,
            url=str(row.get("Art_Url") or row.get("Art_OriginUrl") or row.get("url") or ""),
            source=str(row.get("Art_Source") or row.get("source") or ""),
            sentiment=sentiment,
            tags=tags,
        ))
    return items


def analyze_news(items):
    items = list(items or [])
    positive = sum(1 for item in items if item.sentiment == "positive")
    negative = sum(1 for item in items if item.sentiment == "negative")
    neutral = len(items) - positive - negative
    if not items:
        comment = "暂缺近期资讯数据。"
    elif negative:
        comment = "近期资讯中有%d条偏风险标题，需核对是否来自公告或基本面变化。" % negative
    elif positive:
        comment = "近期资讯偏正面或中性，但标题只能作为线索，仍需结合公告和财报验证。"
    else:
        comment = "近期资讯以中性行业或公司动态为主。"
    return NewsAnalysis(
        items=items,
        positive_count=positive,
        negative_count=negative,
        neutral_count=neutral,
        comment=comment,
    )


def fetch_news(stock, page_size=20, timeout=10):
    market_type = "1" if stock.market == "SH" else "0"
    params = urlencode({
        "cfh": "1",
        "client": "web",
        "mTypeAndCode": "%s.%s" % (market_type, stock.code),
        "type": "1",
        "pageSize": str(page_size),
        "traceId": "1234567890",
    })
    req = Request(
        EASTMONEY_NEWS_URL + "?" + params,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return parse_news_payload(raw)
