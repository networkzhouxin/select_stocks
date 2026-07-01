# -*- coding: utf-8 -*-
import html
import io
import json
import os
import re
import site
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Announcement, AnnouncementFact


KEYWORD_TAGS = [
    ("业绩预告", "important"),
    ("业绩快报", "important"),
    ("年度报告", "important"),
    ("半年度报告", "important"),
    ("季度报告", "important"),
    ("利润分配", "important"),
    ("分红", "important"),
    ("回购", "important"),
    ("增持", "important"),
    ("减持", "risk"),
    ("质押", "risk"),
    ("担保", "risk"),
    ("诉讼", "risk"),
    ("仲裁", "risk"),
    ("处罚", "risk"),
    ("问询函", "risk"),
    ("立案", "risk"),
    ("重大合同", "important"),
    ("资产重组", "important"),
    ("停牌", "risk"),
]

DEEP_READ_TAGS = set([
    "业绩预告",
    "业绩快报",
    "年度报告",
    "半年度报告",
    "季度报告",
    "利润分配",
    "分红",
    "回购",
    "增持",
    "减持",
    "质押",
    "担保",
    "诉讼",
    "仲裁",
    "处罚",
    "问询函",
    "立案",
    "重大合同",
    "资产重组",
])

CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/fulltextSearch/full"
CNINFO_STATIC_PREFIX = "https://static.cninfo.com.cn/"
CNINFO_DETAIL_URL = "https://www.cninfo.com.cn/new/disclosure/detail"


def _short_date(value):
    return str(value or "")[:10]


def _category(row):
    columns = row.get("columns") or []
    if not columns:
        return ""
    names = [str(item.get("column_name") or "") for item in columns]
    return "、".join([name for name in names if name])


def _detail_url(stock_code, art_code):
    return "https://data.eastmoney.com/notices/detail/%s/%s.html" % (stock_code, art_code)


def classify_announcement(title, category):
    haystack = "%s %s" % (title or "", category or "")
    tags = []
    importance = "normal"
    for keyword, level in KEYWORD_TAGS:
        if keyword in haystack:
            tags.append(keyword)
            if level == "risk":
                importance = "risk"
            elif importance != "risk":
                importance = "important"
    return tags, importance


def parse_announcement_payload(text, stock_code):
    payload = json.loads(text)
    rows = payload.get("data", {}).get("list") or []
    items = []
    for row in rows:
        art_code = str(row.get("art_code") or "")
        title = str(row.get("title") or row.get("title_ch") or "")
        category = _category(row)
        tags, importance = classify_announcement(title, category)
        items.append(Announcement(
            date=_short_date(row.get("notice_date") or row.get("display_time")),
            title=title,
            category=category,
            url=_detail_url(stock_code, art_code) if art_code else "",
            tags=tags,
            importance=importance,
        ))
    return items


def _strip_html_marks(text):
    text = re.sub(r"(?is)<[^>]+>", "", text or "")
    return html.unescape(text).strip()


def parse_cninfo_search_payload(text):
    payload = json.loads(text)
    rows = payload.get("announcements") or []
    items = []
    for row in rows:
        adjunct_url = str(row.get("adjunctUrl") or "")
        pdf_url = ""
        if adjunct_url:
            pdf_url = adjunct_url
            if not pdf_url.startswith(("http://", "https://")):
                pdf_url = CNINFO_STATIC_PREFIX + pdf_url.lstrip("/")
        sec_code = str(row.get("secCode") or "")
        org_id = str(row.get("orgId") or "")
        announcement_id = str(row.get("announcementId") or "")
        view_url = ""
        if sec_code and announcement_id:
            query = urlencode({
                "stockCode": sec_code,
                "announcementId": announcement_id,
                "orgId": org_id,
            })
            view_url = "%s?%s" % (CNINFO_DETAIL_URL, query)
        items.append({
            "sec_code": sec_code,
            "sec_name": str(row.get("secName") or ""),
            "title": _strip_html_marks(row.get("announcementTitle") or row.get("shortTitle") or ""),
            "time": row.get("announcementTime"),
            "pdf_url": pdf_url,
            "view_url": view_url,
            "type": str(row.get("adjunctType") or ""),
        })
    return items


def search_cninfo_announcements(stock, keyword, page_size=5, timeout=10):
    body = urlencode({
        "searchkey": keyword,
        "pageNum": 1,
        "pageSize": page_size,
        "sortName": "pubdate",
        "sortType": "desc",
    }).encode("utf-8")
    req = Request(
        CNINFO_SEARCH_URL,
        data=body,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Referer": "http://www.cninfo.com.cn/",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    rows = parse_cninfo_search_payload(raw)
    return [row for row in rows if row["sec_code"] == stock.code]


def fetch_announcements(stock, page_size=12, timeout=10):
    url = (
        "https://np-anotice-stock.eastmoney.com/api/security/ann"
        "?sr=-1&page_size=%d&page_index=1&ann_type=A&client_source=web"
        "&stock_list=%s&f_node=0&s_node=0" % (page_size, stock.code)
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return parse_announcement_payload(data.decode("utf-8", errors="replace"), stock.code)


def parse_announcement_detail_html(text):
    text = re.sub(r"(?is)<script.*?</script>", " ", text or "")
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)</div\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*", "\n", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join([line for line in lines if line])
    return cleaned.strip()


def _fetch_detail_text(url, timeout=10):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl.create_default_context()
    if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
        context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
    try:
        with urlopen(req, timeout=timeout, context=context) as resp:
            data = resp.read()
    except Exception:
        if not url.startswith("https://"):
            raise
        fallback = "http://" + url[len("https://"):]
        req = Request(fallback, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    raw = data.decode("utf-8", errors="replace")
    return parse_announcement_detail_html(raw)


def _announcement_keyword(item):
    if item.tags:
        return item.tags[0]
    title = re.sub(r"[:：].*$", "", item.title or "")
    return title or item.title


def _title_score(source, target):
    source = re.sub(r"\s+", "", source or "")
    target = re.sub(r"\s+", "", target or "")
    if not source or not target:
        return 0
    if source in target or target in source:
        return max(len(source), len(target))
    common = set(source) & set(target)
    return len(common)


def _best_cninfo_match(stock, item, timeout=10):
    keyword = "%s %s" % (stock.name, _announcement_keyword(item))
    rows = search_cninfo_announcements(stock, keyword, page_size=8, timeout=timeout)
    if not rows:
        rows = search_cninfo_announcements(stock, item.title, page_size=8, timeout=timeout)
    if not rows:
        return None
    return sorted(rows, key=lambda row: _title_score(item.title, row["title"]), reverse=True)[0]


def _download_url(url, timeout=10):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "http://www.cninfo.com.cn/"})
    context = ssl.create_default_context()
    if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
        context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
    try:
        with urlopen(req, timeout=timeout, context=context) as resp:
            return resp.read()
    except Exception:
        if not url.startswith("https://"):
            raise
        fallback = "http://" + url[len("https://"):]
        req = Request(fallback, headers={"User-Agent": "Mozilla/5.0", "Referer": "http://www.cninfo.com.cn/"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()


def _extract_pdf_text(data, max_pages=4):
    runtime_packages = os.path.expanduser(
        r"~\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
    )
    if os.path.isdir(runtime_packages):
        site.addsitedir(runtime_packages)
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("本地未安装pypdf，无法自动解析巨潮PDF正文") from exc
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def fetch_cninfo_detail_text(stock, item, timeout=10):
    match = _best_cninfo_match(stock, item, timeout=timeout)
    if not match or not match.get("pdf_url"):
        raise RuntimeError("巨潮资讯未匹配到对应公告PDF")
    pdf_url = match["pdf_url"]
    view_url = match.get("view_url") or pdf_url
    try:
        pdf_data = _download_url(pdf_url, timeout=timeout)
        if not pdf_data.startswith(b"%PDF"):
            raise RuntimeError("巨潮PDF访问返回非PDF内容，可能被网络策略拦截")
        text = _extract_pdf_text(pdf_data)
        if not text:
            raise RuntimeError("巨潮PDF未解析出有效文本")
        return text, "巨潮资讯", view_url
    except RuntimeError as exc:
        return "巨潮PDF链接：%s（%s）" % (pdf_url, exc), "巨潮资讯", view_url


def _should_deep_read(item):
    if item.importance in ("risk", "important"):
        return True
    return bool(set(item.tags) & DEEP_READ_TAGS)


def _summarize_detail(text, max_chars=600):
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("\ufeff", "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _first(pattern, text):
    match = re.search(pattern, text or "")
    return match.group(1) if match else ""


def extract_announcement_facts(tags, text):
    tags = set(tags or [])
    text = re.sub(r"\s+", "", text or "")
    facts = []
    if "减持" in tags or "增持" in tags:
        action = "减持" if "减持" in tags else "增持"
        max_shares = _first(r"(?:不超过|合计不超过|拟减持|拟增持)([0-9,，.]+股)", text)
        max_ratio = _first(r"(?:比例|占.*?比例|占公司总股本|占本公司总股本比例)([0-9.]+%)", text)
        holder = (
            _first(r"([一-龥]{2,8}(?:先生|女士))持有公司股份", text)
            or _first(r"收到([一-龥]{2,8}(?:先生|女士))的", text)
            or _first(r"([一-龥]{2,8}(?:先生|女士))计划", text)
            or _first(r"([一-龥]{2,12}(?:公司))计划", text)
        )
        for prefix in ("实际控制人", "控股股东", "董事长及总经理", "董事长", "总经理", "股东"):
            if holder.startswith(prefix):
                holder = holder[len(prefix):]
                break
        for marker in ("董事长及总经理", "长及总经理", "总经理", "董事长", "实际控制人", "控股股东"):
            if marker in holder:
                holder = holder.split(marker)[-1]
        period = _first(r"(20\d{2}年\d{1,2}月\d{1,2}日至20\d{2}年\d{1,2}月\d{1,2}日)", text)
        fields = {}
        if holder:
            fields["holder"] = holder
        if max_shares:
            fields["max_shares"] = max_shares.replace("，", ",")
        if max_ratio:
            fields["max_ratio"] = max_ratio
        if period:
            fields["period"] = period
        if fields:
            parts = []
            if holder:
                parts.append("主体%s" % holder)
            if max_shares or max_ratio:
                parts.append("上限%s/%s" % (fields.get("max_shares", "N/A"), fields.get("max_ratio", "N/A")))
            if period:
                parts.append("期间%s" % period)
            facts.append(AnnouncementFact(action, fields, "；".join(parts)))

    if "回购" in tags:
        fields = {}
        cancel_shares = _first(r"(?:注销|减少注册资本).*?([0-9,，.]+股)", text)
        amount_floor = _first(r"回购资金总额不低于人民币([0-9,，.]+[万亿]?元)", text)
        amount_cap = _first(r"回购资金总额.*?不超过人民币([0-9,，.]+[万亿]?元)", text)
        price_cap = _first(r"回购价格不超过人民币([0-9.]+元/股)", text)
        purpose = "注销并减少注册资本" if "注销并减少注册资本" in text else ""
        if cancel_shares:
            fields["cancel_shares"] = cancel_shares.replace("，", ",")
        if amount_floor:
            fields["amount_floor"] = amount_floor.replace("，", ",")
        if amount_cap:
            fields["amount_cap"] = amount_cap.replace("，", ",")
        if price_cap:
            fields["price_cap"] = price_cap
        if purpose:
            fields["purpose"] = purpose
        if fields:
            summary = "；".join(["%s：%s" % (k, v) for k, v in fields.items()])
            fact_type = "回购注销" if purpose or cancel_shares else "回购"
            facts.append(AnnouncementFact(fact_type, fields, summary))

    if "分红" in tags or "利润分配" in tags:
        cash_per_10 = _first(r"每10股(?:派发)?现金(?:红利)?([0-9.]+元)", text)
        fields = {}
        if cash_per_10:
            fields["cash_per_10"] = cash_per_10
            facts.append(AnnouncementFact("分红", fields, "每10股派息%s" % cash_per_10))

    if "业绩预告" in tags or "业绩快报" in tags:
        profit_range = _first(r"归属于上市公司股东的净利润(?:为|预计为)?([0-9,，.]+万元至[0-9,，.]+万元)", text)
        yoy = _first(r"同比(?:增长|下降)([0-9.]+%至[0-9.]+%|[0-9.]+%)", text)
        fields = {}
        if profit_range:
            fields["profit_range"] = profit_range.replace("，", ",")
        if yoy:
            fields["yoy"] = yoy
        if fields:
            facts.append(AnnouncementFact("业绩", fields, "；".join(["%s：%s" % (k, v) for k, v in fields.items()])))
    return facts


def _looks_like_block_page(text):
    haystack = text or ""
    return (
        "URL过滤" in haystack
        or "访问被拒绝" in haystack
        or "网络管理员" in haystack
        or "Access Denied" in haystack
    )


def _looks_like_unusable_detail(text):
    haystack = text or ""
    return (
        "公告正文 _ 数据中心" in haystack
        and "点击查看PDF原文" in haystack
        and "郑重声明" in haystack
    )


def enrich_important_announcements(items, max_items=4, timeout=10, fetcher=None,
                                   fallback_fetcher=None, stock=None):
    fetcher = fetcher or _fetch_detail_text
    if fallback_fetcher is None and stock is not None:
        fallback_fetcher = lambda item, timeout=10: fetch_cninfo_detail_text(stock, item, timeout=timeout)
    count = 0
    for item in items:
        if count >= max_items:
            break
        if not item.url or not _should_deep_read(item):
            continue
        try:
            detail_text = fetcher(item.url, timeout=timeout)
            if _looks_like_block_page(detail_text):
                raise RuntimeError("公告详情页访问被网络策略拦截")
            if _looks_like_unusable_detail(detail_text):
                raise RuntimeError("公告详情页未返回可解析正文")
            item.detail_summary = _summarize_detail(detail_text)
            item.detail_error = ""
            item.detail_source = "东方财富"
            item.detail_url = item.url
        except Exception as exc:
            primary_error = str(exc)
            if fallback_fetcher is not None:
                try:
                    detail_text, source, detail_url = fallback_fetcher(item, timeout=timeout)
                    item.detail_summary = _summarize_detail(detail_text)
                    item.detail_error = ""
                    item.detail_source = source
                    item.detail_url = detail_url
                    item.facts = extract_announcement_facts(item.tags, detail_text)
                except Exception as fallback_exc:
                    item.detail_summary = ""
                    item.detail_error = "%s；备用源失败：%s" % (primary_error, fallback_exc)
            else:
                item.detail_summary = ""
                item.detail_error = primary_error
        if item.detail_summary and not item.facts:
            item.facts = extract_announcement_facts(item.tags, item.detail_summary)
        count += 1
    return items
