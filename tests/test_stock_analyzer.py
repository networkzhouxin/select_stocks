# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stock_analyzer.analysis import analyze_stock
from stock_analyzer.announcements import (
    enrich_important_announcements,
    extract_announcement_facts,
    parse_cninfo_search_payload,
    parse_announcement_detail_html,
    parse_announcement_payload,
)
from stock_analyzer.ai import OpenAICompatibleClient, build_ai_context, build_ai_prompt, request_ai_analysis
from stock_analyzer.dividends import (
    AnnualDividend,
    latest_annual_dividend_per_share,
    parse_bonus_payload,
    summarize_dividend_history,
)
from stock_analyzer.financials import parse_financial_payload
from stock_analyzer.financials import parse_quarterly_financial_payload
from stock_analyzer.fundflow import (
    analyze_fund_flow,
    analyze_tencent_pankou,
    parse_fund_flow_payload,
    parse_tencent_pankou_payload,
)
from stock_analyzer.industry import parse_company_survey_payload
from stock_analyzer.report import render_markdown_report
from stock_analyzer.resolver import resolve_numeric_code
from stock_analyzer.models import Quote, ValuationRow
from stock_analyzer.news import analyze_news, parse_news_payload
from stock_analyzer.research import analyze_research_reports, parse_research_payload
from stock_analyzer.tencent import decode_tencent_bytes, parse_kline_payload, parse_quote_line, parse_search_hint
from stock_analyzer.sina import parse_sina_quote_line
from stock_analyzer.valuation import analyze_valuation_history, parse_valuation_payload


def test_resolve_numeric_code_adds_market_prefix():
    assert resolve_numeric_code("600900").tencent_code == "sh600900"
    assert resolve_numeric_code("002572").tencent_code == "sz002572"
    assert resolve_numeric_code("300750").tencent_code == "sz300750"
    assert resolve_numeric_code("688981").tencent_code == "sh688981"


def test_parse_search_hint_decodes_stock_name():
    hint = 'v_hint="sh~600900~\\u957f\\u6c5f\\u7535\\u529b~cjdl~GP-A"'

    stock = parse_search_hint(hint)

    assert stock.code == "600900"
    assert stock.name == "长江电力"
    assert stock.market == "SH"
    assert stock.tencent_code == "sh600900"


def test_decode_tencent_bytes_accepts_gb18030_payload():
    raw = 'v_sz002572="51~索菲亚~002572~7.50";'.encode("gb18030")

    assert "索菲亚" in decode_tencent_bytes(raw)


def test_parse_tencent_quote_line_extracts_core_fields():
    line = (
        'v_sh600900="1~长江电力~600900~26.46~26.47~26.44~215718~'
        '103886~111803~26.46~71~26.45~419~26.44~295~26.43~426~'
        '26.42~1147~26.47~241~26.48~212~26.49~336~26.50~1722~'
        '26.51~687~~20260701095632~-0.01~-0.04~26.65~26.40~'
        '26.46/215718/571977287~215718~57198~0.09~17.94~~26.65~'
        '26.40~0.94~6474.29~6474.29~2.84~29.12~23.82";'
    )

    quote = parse_quote_line(line)

    assert quote.name == "长江电力"
    assert quote.code == "600900"
    assert quote.price == 26.46
    assert quote.pe == 17.94
    assert quote.pb == 2.84
    assert quote.pct_change == -0.04


def test_parse_sina_quote_line_extracts_cross_check_fields():
    line = (
        'var hq_str_sh600900="长江电力,26.440,26.470,26.640,'
        '26.720,26.360,26.630,26.640,90939756,2417436712.000,'
        '436100,26.630,42700,26.620,75100,26.610,120200,26.600,'
        '64800,26.590,307233,26.640,365697,26.650,360600,26.660,'
        '215600,26.670,327300,26.680,2026-07-01,15:00:02,00,";'
    )

    quote = parse_sina_quote_line(line, "sh600900")

    assert quote.name == "长江电力"
    assert quote.code == "600900"
    assert quote.price == 26.64
    assert quote.prev_close == 26.47
    assert quote.timestamp == "2026-07-01 15:00:02"


def test_parse_kline_payload_returns_ma_inputs():
    payload = (
        '{"code":0,"data":{"sh600900":{"qfqday":['
        '["2026-01-01","10","11","12","9","100"],'
        '["2026-01-02","11","12","13","10","100"],'
        '["2026-01-03","12","10","12","9","100"]'
        ']}}}'
    )

    rows = parse_kline_payload(payload, "sh600900")

    assert [r.close for r in rows] == [11.0, 12.0, 10.0]
    assert rows[-1].high == 12.0


def test_parse_eastmoney_financial_payload_extracts_annual_metrics():
    payload = (
        '{"data":[{'
        '"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":86241940222.2,'
        '"PARENTNETPROFIT":34502809176.39,'
        '"KCFJCXSYJLR":33445575299.94,'
        '"TOTALOPERATEREVETZ":2.071287620863,'
        '"PARENTNETPROFITTZ":6.174992912414,'
        '"ROEJQ":15.9,"XSMLL":61.6684875716,'
        '"XSJLL":40.5246806803,"ZCFZL":58.2694597691,'
        '"EPSJB":1.4101,"MGJYXJJE":2.4751670217'
        '}]}'
    )

    rows = parse_financial_payload(payload)

    assert len(rows) == 1
    assert rows[0].year == "2025"
    assert rows[0].revenue == 86241940222.2
    assert rows[0].roe == 15.9
    assert rows[0].operating_cashflow_per_share == 2.4751670217


def test_parse_bonus_payload_sums_cash_dividend_by_report_year():
    payload = (
        '{"result":{"data":['
        '{"SECURITY_CODE":"600900","REPORT_DATE":"2025-12-31 00:00:00",'
        '"PRETAX_BONUS_RMB":7.9,"PLAN_NOTICE_DATE":"2026-04-30 00:00:00",'
        '"ASSIGN_PROGRESS":"passed","IMPL_PLAN_PROFILE":"10 cash 7.90"},'
        '{"SECURITY_CODE":"600900","REPORT_DATE":"2025-09-30 00:00:00",'
        '"PRETAX_BONUS_RMB":2.1,"PLAN_NOTICE_DATE":"2025-12-31 00:00:00",'
        '"ASSIGN_PROGRESS":"done","IMPL_PLAN_PROFILE":"10 cash 2.10"},'
        '{"SECURITY_CODE":"600900","REPORT_DATE":"2024-12-31 00:00:00",'
        '"PRETAX_BONUS_RMB":7.33,"PLAN_NOTICE_DATE":"2025-04-30 00:00:00",'
        '"ASSIGN_PROGRESS":"done","IMPL_PLAN_PROFILE":"10 cash 7.33"}'
        ']}}'
    )

    records = parse_bonus_payload(payload)
    summary = summarize_dividend_history(records)

    assert latest_annual_dividend_per_share(summary) == 1.0
    assert summary[0].year == "2025"
    assert summary[0].cash_dividend_per_share == 1.0
    assert summary[0].cash_dividend_per_10 == 10.0
    assert summary[0].record_count == 2
    assert summary[1].cash_dividend_per_share == 0.733


def test_parse_company_survey_payload_extracts_industry_tags():
    payload = (
        '{"jbzl":[{'
        '"SECURITY_CODE":"002115",'
        '"SECURITY_TYPE":"深交所主板A股",'
        '"TRADE_MARKET":"深圳证券交易所",'
        '"EM2016":"文化传媒-营销服务-营销服务",'
        '"INDUSTRYCSRC1":"租赁和商务服务业-商务服务业",'
        '"ORG_PROFILE":"sample profile"'
        '}]}'
    )

    info = parse_company_survey_payload(payload)

    assert info.exchange_board == "深交所主板A股"
    assert info.market == "深圳证券交易所"
    assert info.eastmoney_industry == "文化传媒-营销服务-营销服务"
    assert info.csrc_industry == "租赁和商务服务业-商务服务业"


def test_parse_announcement_payload_extracts_links_and_flags_keywords():
    payload = (
        '{"data":{"list":[{'
        '"art_code":"AN202604301234567890",'
        '"notice_date":"2026-04-30 00:00:00",'
        '"title":"测试股份:2025年年度报告",'
        '"columns":[{"column_name":"年度报告"}],'
        '"codes":[{"stock_code":"600900"}]'
        '},{'
        '"art_code":"AN202605011234567891",'
        '"notice_date":"2026-05-01 00:00:00",'
        '"title":"测试股份:关于控股股东减持计划的公告",'
        '"columns":[{"column_name":"股东增减持公告"}],'
        '"codes":[{"stock_code":"600900"}]'
        '}]}}'
    )

    items = parse_announcement_payload(payload, "600900")

    assert len(items) == 2
    assert items[0].date == "2026-04-30"
    assert items[0].category == "年度报告"
    assert items[0].url.endswith("/600900/AN202604301234567890.html")
    assert items[1].importance == "risk"
    assert "减持" in items[1].tags


def test_parse_announcement_detail_html_extracts_main_text():
    html = """
    <html><head><script>ignore()</script><style>.x{}</style></head>
    <body>
      <h1>关于控股股东减持计划的公告</h1>
      <div class="content">
        公司控股股东计划在未来三个月内减持不超过公司总股本的1.00%。
        减持原因为自身资金需求。本次减持不会导致控制权发生变化。
      </div>
    </body></html>
    """

    text = parse_announcement_detail_html(html)

    assert "减持不超过公司总股本的1.00%" in text
    assert "控制权发生变化" in text
    assert "ignore" not in text


def test_enrich_important_announcements_marks_block_page_as_error():
    items = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1","notice_date":"2026-05-01 00:00:00",'
        '"title":"测试股份:关于控股股东减持计划的公告",'
        '"columns":[{"column_name":"股东增减持公告"}],'
        '"codes":[{"stock_code":"600900"}]}]}}',
        "600900",
    )

    enrich_important_announcements(
        items,
        fetcher=lambda url, timeout=10: "URL过滤 访问被拒绝 请联系网络管理员",
    )

    assert items[0].detail_summary == ""
    assert "网络策略拦截" in items[0].detail_error


def test_parse_cninfo_search_payload_extracts_pdf_metadata():
    payload = (
        '{"announcements":[{'
        '"secCode":"002115","secName":"三维通信","orgId":"9900001234",'
        '"announcementId":"1225283091",'
        '"announcementTitle":"<em>三维通信</em>：关于公司实际控制人<em>减持</em>股份的预披露公告",'
        '"announcementTime":1778198400000,'
        '"adjunctUrl":"finalpage/2026-05-08/1225283091.PDF",'
        '"adjunctType":"PDF"'
        '}]}'
    )

    rows = parse_cninfo_search_payload(payload)

    assert rows[0]["sec_code"] == "002115"
    assert rows[0]["title"] == "三维通信：关于公司实际控制人减持股份的预披露公告"
    assert rows[0]["pdf_url"] == "https://static.cninfo.com.cn/finalpage/2026-05-08/1225283091.PDF"
    assert rows[0]["view_url"].startswith("https://www.cninfo.com.cn/new/disclosure/detail")
    assert "announcementId=1225283091" in rows[0]["view_url"]


def test_enrich_important_announcements_uses_cninfo_fallback_after_block():
    items = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1","notice_date":"2026-05-01 00:00:00",'
        '"title":"测试股份:关于控股股东减持计划的公告",'
        '"columns":[{"column_name":"股东增减持公告"}],'
        '"codes":[{"stock_code":"600900"}]}]}}',
        "600900",
    )

    enrich_important_announcements(
        items,
        fetcher=lambda url, timeout=10: "URL过滤 访问被拒绝 请联系网络管理员",
        fallback_fetcher=lambda item, timeout=10: (
            "巨潮PDF正文摘要：控股股东拟减持不超过1.00%。",
            "巨潮资讯",
            "http://static.cninfo.com.cn/test.PDF",
        ),
    )

    assert "减持不超过1.00%" in items[0].detail_summary
    assert items[0].detail_source == "巨潮资讯"
    assert items[0].detail_url.endswith("test.PDF")
    assert items[0].detail_error == ""


def test_enrich_important_announcements_only_reads_key_items():
    payload = (
        '{"data":{"list":[{'
        '"art_code":"AN1","notice_date":"2026-05-01 00:00:00",'
        '"title":"测试股份:关于控股股东减持计划的公告",'
        '"columns":[{"column_name":"股东增减持公告"}],'
        '"codes":[{"stock_code":"600900"}]'
        '},{'
        '"art_code":"AN2","notice_date":"2026-05-02 00:00:00",'
        '"title":"测试股份:普通会议资料",'
        '"columns":[{"column_name":"股东大会资料"}],'
        '"codes":[{"stock_code":"600900"}]'
        '}]}}'
    )
    items = parse_announcement_payload(payload, "600900")
    fetched = []

    def fake_fetch(url, timeout=10):
        fetched.append(url)
        return "控股股东拟减持不超过1.00%，减持原因为自身资金需求。"

    enrich_important_announcements(items, fetcher=fake_fetch)

    assert len(fetched) == 1
    assert items[0].detail_summary.startswith("控股股东拟减持")
    assert items[1].detail_summary == ""


def test_extract_announcement_facts_for_reduction_and_buyback_cancel():
    reduce_text = (
        "实际控制人李越伦先生持有公司股份数量为76,912,700股，占公司总股本的9.5539%。"
        "计划减持股份合计不超过8,000,000股，占本公司总股本比例0.9937%，"
        "减持期间为2026年6月1日至2026年8月31日。"
    )
    buyback_text = (
        "回购股份用途由实施股权激励或员工持股计划变更为注销并减少注册资本，"
        "将5,954,000股回购股份进行注销，公司总股本将由810,991,332股减少至805,037,332股。"
    )

    reduce_facts = extract_announcement_facts(["减持"], reduce_text)
    buyback_facts = extract_announcement_facts(["回购"], buyback_text)

    assert reduce_facts[0].fact_type == "减持"
    assert reduce_facts[0].fields["holder"] == "李越伦先生"
    assert reduce_facts[0].fields["max_shares"] == "8,000,000股"
    assert reduce_facts[0].fields["max_ratio"] == "0.9937%"
    assert "2026年6月1日" in reduce_facts[0].fields["period"]
    assert buyback_facts[0].fact_type == "回购注销"
    assert buyback_facts[0].fields["cancel_shares"] == "5,954,000股"


def test_analyze_stock_calculates_dividend_buy_prices():
    stock = resolve_numeric_code("600900")
    quote_line = (
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    quote = parse_quote_line(quote_line)

    result = analyze_stock(
        stock=stock,
        quote=quote,
        kline=[],
        dividend_per_share=1.0,
        financials=parse_financial_payload(
            '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
            '"TOTALOPERATEREVE":86241940222.2,'
            '"PARENTNETPROFIT":34502809176.39,'
            '"TOTALOPERATEREVETZ":2.071287620863,'
            '"PARENTNETPROFITTZ":6.174992912414,'
            '"ROEJQ":15.9,"XSMLL":61.6684875716,'
            '"XSJLL":40.5246806803,"ZCFZL":58.2694597691,'
            '"EPSJB":1.4101,"MGJYXJJE":2.4751670217}]}'
        ),
        bond_yield=0.0175,
    )

    assert result.dividend.current_yield == 0.04
    assert result.dividend.target_prices[0.04] == 25.0
    assert round(result.dividend.bond_risk_prices[0.0175 + 0.025], 2) == 23.53
    assert round(result.financial.latest_dividend_payout_ratio, 4) == 0.7092
    assert round(result.financial.latest_cash_dividend_coverage, 4) == 2.4752
    assert result.decision.action in ("可分批关注", "估值合理，可关注")
    assert "25.00元" in result.decision.buy_zone
    assert "跌破MA60" in result.decision.risk_control


def test_analyze_stock_calculates_price_percentile_from_kline():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    kline = parse_kline_payload(
        '{"code":0,"data":{"sh600900":{"qfqday":['
        + ",".join(
            '["2025-01-%02d","%.2f","%.2f","%.2f","%.2f","100"]' % (
                (i % 28) + 1,
                10 + i,
                10 + i,
                10 + i,
                10 + i,
            )
            for i in range(120)
        )
        + ']}}}',
        "sh600900",
    )

    result = analyze_stock(stock, quote, kline, 1.0, [], 0.0175)

    assert result.technical.price_percentile_120 is not None
    assert result.technical.price_percentile_120 < 0.2
    assert "近120日价格分位" in result.technical.price_position_comment


def test_technical_indicators_and_trade_gates_are_reported():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=28.0, pe=18.0, pb=2.8)
    rows = []
    for i in range(80):
        close = 20 + i * 0.1
        rows.append(type("Row", (), {
            "date": "2026-01-%02d" % ((i % 28) + 1),
            "open": close - 0.05,
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "volume": 1000 + i * 10,
        })())

    result = analyze_stock(stock, quote, rows, 1.0, [], 0.0175)
    report = render_markdown_report(result)

    assert result.technical.rsi14 is not None
    assert result.technical.macd_signal in ("多头", "空头", "震荡")
    assert result.technical.boll_position is not None
    assert result.technical.volume_ratio_5_20 is not None
    assert result.technical.left_gate_comment
    assert "技术闸门" in result.trade_price_zones.left_buy_zone
    assert "RSI14" in report
    assert "MACD" in report
    assert "布林带" in report
    assert "技术闸门" in report


def test_parse_valuation_payload_extracts_pe_pb_history():
    payload = (
        '{"result":{"data":['
        '{"TRADE_DATE":"2026-06-30 00:00:00","PE_TTM":18.5,"PB_MRQ":2.8},'
        '{"TRADE_DATE":"2026-06-29 00:00:00","PE_TTM":17.5,"PB_MRQ":2.7},'
        '{"TRADE_DATE":"2026-06-28 00:00:00","PE_TTM":-1,"PB_MRQ":0}'
        ']}}'
    )

    rows = parse_valuation_payload(payload)

    assert len(rows) == 2
    assert rows[0].date == "2026-06-30"
    assert rows[0].pe_ttm == 18.5
    assert rows[0].pb == 2.8


def test_analyze_valuation_history_calculates_percentiles():
    rows = parse_valuation_payload(
        '{"result":{"data":['
        '{"TRADE_DATE":"2026-06-30 00:00:00","PE_TTM":30,"PB_MRQ":3.0},'
        '{"TRADE_DATE":"2026-06-29 00:00:00","PE_TTM":20,"PB_MRQ":2.0},'
        '{"TRADE_DATE":"2026-06-28 00:00:00","PE_TTM":10,"PB_MRQ":1.0}'
        ']}}'
    )

    analysis = analyze_valuation_history(rows, current_pe=20, current_pb=2.0)

    assert analysis.sample_size == 3
    assert round(analysis.pe_percentile, 4) == 0.5
    assert round(analysis.pb_percentile, 4) == 0.5
    assert analysis.pe_min == 10
    assert analysis.pe_max == 30
    assert analysis.pb_min == 1.0
    assert analysis.pb_max == 3.0
    assert "PE历史分位50.0%" in analysis.comment


def test_analyze_valuation_history_builds_multi_window_view_and_primary_metric():
    rows = [
        ValuationRow("2026-06-%02d" % ((i % 28) + 1), pe_ttm=float(i + 1), pb=float(i + 1) / 10)
        for i in range(1300)
    ]

    analysis = analyze_valuation_history(rows, current_pe=20, current_pb=2.0, category_code="turnaround_watch")

    assert analysis.primary_metric == "PB"
    assert "PE会失真" in analysis.framework_comment
    assert analysis.windows["3年"].sample_size == 750
    assert analysis.windows["5年"].sample_size == 1250
    assert analysis.windows["10年"].sample_size == 1300
    assert not analysis.windows["10年"].is_full_window
    assert analysis.pe_percentile == analysis.windows["3年"].pe_percentile
    assert analysis.pb_percentile == analysis.windows["3年"].pb_percentile


def test_ai_context_uses_structured_decision_data():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175)

    context = build_ai_context(result)

    assert "综合判断" in context
    assert "买入区间" in context
    assert "长江电力" in context


def test_ai_context_includes_deep_read_announcements():
    announcements = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1",'
        '"notice_date":"2026-05-01 00:00:00",'
        '"title":"长江电力:关于利润分配方案的公告",'
        '"columns":[{"column_name":"分配方案"}],'
        '"codes":[{"stock_code":"600900"}]}]}}',
        "600900",
    )
    announcements[0].detail_summary = "拟每10股派发现金红利10元，现金分红合计占归母净利润比例约70%。"
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, [], "", None, announcements)

    context = build_ai_context(result)

    assert "重点公告深读" in context
    assert "每10股派发现金红利10元" in context


def test_request_ai_analysis_accepts_injected_client():
    class FakeClient:
        def complete(self, prompt):
            assert "只基于以下结构化数据" in prompt
            return "AI分析：等待更好的安全边际。"

    text = request_ai_analysis("测试上下文", client=FakeClient())

    assert text == "AI分析：等待更好的安全边际。"


def test_openai_client_timeout_uses_env_default():
    old_value = os.environ.get("STOCK_ANALYZER_AI_TIMEOUT")
    os.environ["STOCK_ANALYZER_AI_TIMEOUT"] = "120"
    try:
        client = OpenAICompatibleClient(api_key="test-key")
    finally:
        if old_value is None:
            os.environ.pop("STOCK_ANALYZER_AI_TIMEOUT", None)
        else:
            os.environ["STOCK_ANALYZER_AI_TIMEOUT"] = old_value

    assert client.timeout == 120


def test_ai_prompt_requires_industry_aware_risk_interpretation():
    prompt = build_ai_prompt("测试上下文")

    assert "结合行业属性" in prompt
    assert "不要把单一指标机械化判定为风险" in prompt
    assert "不要改变规则引擎给出的买入区间" in prompt


def test_report_mentions_limitations_when_financials_missing():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    result = analyze_stock(stock, quote, [], 1.0, 0.0175)

    report = render_markdown_report(result)

    assert "长江电力" in report
    assert "买入价格区间" in report
    assert "综合判断" in report
    assert "最终结论" in report
    assert "当前是否适合买入" in report
    assert "财报深度数据" in report


def test_report_includes_historical_valuation_percentiles():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=20.0, pe=20.0, pb=2.0)
    result = analyze_stock(
        stock,
        quote,
        [],
        1.0,
        [],
        0.0175,
        valuation_rows=parse_valuation_payload(
            '{"result":{"data":['
            '{"TRADE_DATE":"2026-06-30 00:00:00","PE_TTM":30,"PB_MRQ":3.0},'
            '{"TRADE_DATE":"2026-06-29 00:00:00","PE_TTM":20,"PB_MRQ":2.0},'
            '{"TRADE_DATE":"2026-06-28 00:00:00","PE_TTM":10,"PB_MRQ":1.0}'
            ']}}'
        ),
    )

    report = render_markdown_report(result)

    assert "## 历史估值分位" in report
    assert "主估值指标" in report
    assert "近3年" in report
    assert "近5年" in report
    assert "近10年" in report
    assert "PE(TTM)历史分位" in report
    assert "PB历史分位" in report
    assert "50.00%" in report


def test_analysis_adds_score_breakdown_profit_quality_and_buy_sell_points():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=25.0, pe=15.0, pb=2.5)
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":10000000000,'
        '"PARENTNETPROFIT":2000000000,'
        '"KCFJCXSYJLR":1900000000,'
        '"TOTALOPERATEREVETZ":8,'
        '"PARENTNETPROFITTZ":10,'
        '"ROEJQ":15,"XSMLL":50,'
        '"XSJLL":20,"ZCFZL":45,'
        '"EPSJB":1.5,"MGJYXJJE":2.2},'
        '{"REPORT_YEAR":"2024","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":9000000000,'
        '"PARENTNETPROFIT":1800000000,'
        '"KCFJCXSYJLR":1700000000,'
        '"ROEJQ":14,"XSMLL":48,'
        '"XSJLL":19,"ZCFZL":46,'
        '"EPSJB":1.4,"MGJYXJJE":2.0}]}'
    )

    result = analyze_stock(stock, quote, [], 1.0, financials, 0.0175)

    assert result.financial.quality.cash_profit_ratio == 1.47
    assert result.financial.quality.deduct_profit_ratio == 0.95
    assert result.score_breakdown.total == result.decision.score
    assert result.decision.left_buy_point
    assert result.decision.right_buy_point
    assert result.decision.reduce_point

    report = render_markdown_report(result)
    assert "## 评分拆解" in report
    assert "## 利润质量" in report
    assert "左侧买点" in report


def test_analysis_adds_dividend_yield_history_from_prices_and_dividends():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=20.0, pe=15.0, pb=2.5)
    dividends = [
        AnnualDividend("2025", 10.0, 1.0, 1),
        AnnualDividend("2024", 8.0, 0.8, 1),
    ]
    rows = [
        ValuationRow("2026-06-30", pe_ttm=15, pb=2.5, close_price=20.0),
        ValuationRow("2025-06-30", pe_ttm=16, pb=2.6, close_price=16.0),
        ValuationRow("2024-06-30", pe_ttm=17, pb=2.7, close_price=20.0),
    ]

    result = analyze_stock(
        stock,
        quote,
        [],
        1.0,
        [],
        0.0175,
        dividend_history=dividends,
        valuation_rows=rows,
    )

    assert result.dividend.yield_history.sample_size == 3
    assert result.dividend.yield_history.current_yield_percentile is not None
    assert "股息率历史分位" in result.dividend.yield_history.comment
    assert "历史股息率分位" in render_markdown_report(result)


def test_analysis_adds_announcement_risk_levels_and_industry_relative_valuation():
    stock = resolve_numeric_code("002115")
    quote = Quote(code="002115", name="三维通信", price=10.0, pe=80.0, pb=4.0)
    announcements = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1","notice_date":"2026-05-01 00:00:00",'
        '"title":"三维通信:关于实际控制人减持股份的预披露公告",'
        '"columns":[{"column_name":"股东/实际控制人股份减持"}],'
        '"codes":[{"stock_code":"002115"}]}]}}',
        "002115",
    )
    announcements[0].facts = [
        extract_announcement_facts(
            ["减持"],
            "李越伦先生计划减持股份合计不超过8,000,000股，占本公司总股本比例0.9937%，减持期间为2026年6月1日至2026年8月31日。",
        )[0]
    ]
    peers = [
        ValuationRow("2026-06-30", pe_ttm=20, pb=2.0),
        ValuationRow("2026-06-30", pe_ttm=40, pb=3.0),
        ValuationRow("2026-06-30", pe_ttm=80, pb=4.0),
    ]

    result = analyze_stock(
        stock,
        quote,
        [],
        0.05,
        [],
        0.0175,
        announcements=announcements,
        industry_valuation_rows=peers,
    )

    assert result.announcement_risk.level in ("中", "高")
    assert "减持" in result.announcement_risk.comment
    assert result.industry_valuation.peer_count == 3
    assert result.industry_valuation.pb_percentile > 0.8
    report = render_markdown_report(result)
    assert "## 公告风险扫描" in report
    assert "## 行业相对估值" in report


def test_report_includes_financial_quality_when_available():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":86241940222.2,'
        '"PARENTNETPROFIT":34502809176.39,'
        '"TOTALOPERATEREVETZ":2.071287620863,'
        '"PARENTNETPROFITTZ":6.174992912414,'
        '"ROEJQ":15.9,"XSMLL":61.6684875716,'
        '"XSJLL":40.5246806803,"ZCFZL":58.2694597691,'
        '"EPSJB":1.4101,"MGJYXJJE":2.4751670217}]}'
    )
    result = analyze_stock(stock, quote, [], 1.0, financials, 0.0175)

    report = render_markdown_report(result)

    assert "财务质量" in report
    assert "分红可持续性" in report
    assert "2025" in report
    assert "15.90%" in report


def test_report_includes_dividend_history_when_available():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~A~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    history = [
        AnnualDividend("2025", 10.0, 1.0, 2),
        AnnualDividend("2024", 9.43, 0.943, 2),
    ]
    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, history)

    report = render_markdown_report(result)

    assert "近年现金分红" in report
    assert "2025" in report
    assert "1.00" in report
    assert "2次" in report


def test_classifies_weak_loss_maker_as_turnaround_watch():
    stock = resolve_numeric_code("002115")
    quote = parse_quote_line(
        'v_sz002115="1~三维通信~002115~10.46~10.30~10.20~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.16~1.55~10.56~10.23~0/0/0~0~0~0~1193.69~~'
        '10.56~10.23~0.00~78.68~78.68~4.11";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":11951000000,'
        '"PARENTNETPROFIT":-13000000,'
        '"TOTALOPERATEREVETZ":9.16,'
        '"PARENTNETPROFITTZ":95.14,'
        '"ROEJQ":-0.64,"XSMLL":4.31,'
        '"XSJLL":-0.01,"ZCFZL":53.29,'
        '"EPSJB":-0.02,"MGJYXJJE":-0.17},'
        '{"REPORT_YEAR":"2024","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":10949000000,'
        '"PARENTNETPROFIT":-271000000,'
        '"ROEJQ":-12.14,"XSMLL":4.81,'
        '"XSJLL":-2.28,"ZCFZL":42.16,'
        '"EPSJB":-0.33,"MGJYXJJE":0.17}]}'
    )

    result = analyze_stock(stock, quote, [], 0.05, financials, 0.0175, [])

    assert result.category.code == "turnaround_watch"
    assert not result.category.dividend_valuation_applicable
    assert "红利估值不适用" in result.category.framework


def test_classifies_high_dividend_cashflow_stock_as_dividend_stable():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":86241940222.2,'
        '"PARENTNETPROFIT":34502809176.39,'
        '"TOTALOPERATEREVETZ":2.071287620863,'
        '"PARENTNETPROFITTZ":6.174992912414,'
        '"ROEJQ":15.9,"XSMLL":61.6684875716,'
        '"XSJLL":40.5246806803,"ZCFZL":58.2694597691,'
        '"EPSJB":1.4101,"MGJYXJJE":2.4751670217}]}'
    )

    result = analyze_stock(stock, quote, [], 1.0, financials, 0.0175, [])

    assert result.category.code == "dividend_stable"
    assert result.category.dividend_valuation_applicable


def test_report_includes_stock_category_section():
    stock = resolve_numeric_code("002115")
    quote = parse_quote_line(
        'v_sz002115="1~三维通信~002115~10.46~10.30~10.20~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.16~1.55~10.56~10.23~0/0/0~0~0~0~1193.69~~'
        '10.56~10.23~0.00~78.68~78.68~4.11";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":11951000000,'
        '"PARENTNETPROFIT":-13000000,'
        '"TOTALOPERATEREVETZ":9.16,'
        '"PARENTNETPROFITTZ":95.14,'
        '"ROEJQ":-0.64,"XSMLL":4.31,'
        '"XSJLL":-0.01,"ZCFZL":53.29,'
        '"EPSJB":-0.02,"MGJYXJJE":-0.17}]}'
    )
    result = analyze_stock(stock, quote, [], 0.05, financials, 0.0175, [])

    report = render_markdown_report(result)

    assert "股票类型" in report
    assert "困境/反转观察型" in report
    assert "红利估值不适用" in report
    assert "当前不适合用股息率买入价做主判断" in report


def test_turnaround_report_includes_dedicated_checklist():
    stock = resolve_numeric_code("002115")
    quote = parse_quote_line(
        'v_sz002115="1~三维通信~002115~10.46~10.30~10.20~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.16~1.55~10.56~10.23~0/0/0~0~0~0~1193.69~~'
        '10.56~10.23~0.00~78.68~78.68~4.11";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":11951000000,'
        '"PARENTNETPROFIT":-13000000,'
        '"TOTALOPERATEREVETZ":9.16,'
        '"PARENTNETPROFITTZ":95.14,'
        '"ROEJQ":-0.64,"XSMLL":4.31,'
        '"XSJLL":-0.01,"ZCFZL":53.29,'
        '"EPSJB":-0.02,"MGJYXJJE":-0.17}]}'
    )
    result = analyze_stock(stock, quote, [], 0.05, financials, 0.0175, [])

    report = render_markdown_report(result)

    assert "## 困境反转检查表" in report
    assert "扣非/归母利润转正" in report
    assert "经营现金流转正" in report
    assert "毛利率改善" in report
    assert "不通过" in report


def test_dividend_report_includes_dedicated_checklist():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":86241940222.2,'
        '"PARENTNETPROFIT":34502809176.39,'
        '"TOTALOPERATEREVETZ":2.071287620863,'
        '"PARENTNETPROFITTZ":6.174992912414,'
        '"ROEJQ":15.9,"XSMLL":61.6684875716,'
        '"XSJLL":40.5246806803,"ZCFZL":58.2694597691,'
        '"EPSJB":1.4101,"MGJYXJJE":2.4751670217}]}'
    )
    result = analyze_stock(stock, quote, [], 1.0, financials, 0.0175, [])

    report = render_markdown_report(result)

    assert "## 红利股检查表" in report
    assert "股息率达标" in report
    assert "现金流覆盖分红" in report
    assert "分红率不过高" in report
    assert "通过" in report


def test_report_includes_industry_tags_when_available():
    stock = resolve_numeric_code("002115")
    quote = parse_quote_line(
        'v_sz002115="1~三维通信~002115~10.46~10.30~10.20~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.16~1.55~10.56~10.23~0/0/0~0~0~0~1193.69~~'
        '10.56~10.23~0.00~78.68~78.68~4.11";'
    )
    industry = parse_company_survey_payload(
        '{"jbzl":[{"SECURITY_TYPE":"深交所主板A股",'
        '"TRADE_MARKET":"深圳证券交易所",'
        '"EM2016":"文化传媒-营销服务-营销服务",'
        '"INDUSTRYCSRC1":"租赁和商务服务业-商务服务业"}]}'
    )
    result = analyze_stock(stock, quote, [], 0.05, [], 0.0175, [], "", industry)

    report = render_markdown_report(result)

    assert "行业分类" in report
    assert "文化传媒-营销服务-营销服务" in report
    assert "租赁和商务服务业-商务服务业" in report


def test_report_includes_recent_announcements_section():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    announcements = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1",'
        '"notice_date":"2026-04-30 00:00:00",'
        '"title":"长江电力:2025年年度报告",'
        '"columns":[{"column_name":"年度报告"}],'
        '"codes":[{"stock_code":"600900"}]}]}}',
        "600900",
    )
    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, [], "", None, announcements)

    report = render_markdown_report(result)

    assert "近期重要公告" in report
    assert "2025年年度报告" in report
    assert "年度报告" in report


def test_parse_news_payload_extracts_items_and_sentiment():
    payload = (
        '{"code":1,"data":{"list":['
        '{"Art_ShowTime":"2026-06-26 15:23:13",'
        '"Art_Title":"公司完成回购并提升股东回报",'
        '"Art_Url":"http://finance.eastmoney.com/a/1.html","Art_Source":"证券时报"},'
        '{"Art_ShowTime":"2026-06-20 08:00:00",'
        '"Art_Title":"控股股东拟减持公司股份",'
        '"Art_Url":"http://finance.eastmoney.com/a/2.html"},'
        '{"Art_ShowTime":"2026-06-18 08:00:00",'
        '"Art_Title":"行业景气度持续改善"}'
        ']}}'
    )

    items = parse_news_payload(payload)
    summary = analyze_news(items)

    assert len(items) == 3
    assert items[0].sentiment == "positive"
    assert items[1].sentiment == "negative"
    assert "回购" in items[0].tags
    assert summary.positive_count == 1
    assert summary.negative_count == 1


def test_parse_research_payload_extracts_rating_summary():
    payload = (
        '{"hits":2,"data":['
        '{"publishDate":"2026-06-01 00:00:00.000",'
        '"title":"业绩稳健，配置价值突出",'
        '"orgSName":"华源证券","emRatingName":"买入","researcher":"张三",'
        '"infoCode":"AP1"},'
        '{"publishDate":"2026-05-01 00:00:00.000",'
        '"title":"现金流稳定","orgName":"中信证券","emRatingName":"增持"}'
        ']}'
    )

    reports = parse_research_payload(payload)
    summary = analyze_research_reports(reports)

    assert len(reports) == 2
    assert reports[0].rating == "买入"
    assert reports[0].org == "华源证券"
    assert summary.report_count == 2
    assert summary.latest_rating == "买入"
    assert "买入" in summary.rating_summary


def test_report_and_ai_context_include_news_and_research():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    news = parse_news_payload(
        '{"data":{"list":[{"Art_ShowTime":"2026-06-20 08:00:00",'
        '"Art_Title":"控股股东拟减持公司股份"}]}}'
    )
    reports = parse_research_payload(
        '{"data":[{"publishDate":"2026-06-01 00:00:00.000",'
        '"title":"业绩稳健","orgSName":"华源证券","emRatingName":"买入"}]}'
    )

    result = analyze_stock(
        stock, quote, [], 1.0, [], 0.0175, [], "", None, [],
        news_items=news,
        research_reports=reports,
    )
    report = render_markdown_report(result)
    context = build_ai_context(result)

    assert "近期资讯事件" in report
    assert "机构研报观点" in report
    assert "控股股东拟减持" in report
    assert "华源证券" in report
    assert "近期资讯" in context
    assert "机构研报" in context


def test_parse_quarterly_financial_payload_extracts_recent_reports():
    payload = (
        '{"data":['
        '{"REPORT_DATE":"2026-03-31 00:00:00","REPORT_TYPE":"一季报",'
        '"REPORT_DATE_NAME":"2026一季报","TOTALOPERATEREVE":18111540767.5,'
        '"PARENTNETPROFIT":6761006898.48,"KCFJCXSYJLR":6237332251.35,'
        '"TOTALOPERATEREVETZ":6.44,"PARENTNETPROFITTZ":30.50,'
        '"KCFJCXSYJLRTZ":19.20,"XSMLL":59.2,"XSJLL":37.3,'
        '"MGJYXJJE":0.48},'
        '{"REPORT_DATE":"2025-12-31 00:00:00","REPORT_TYPE":"年报",'
        '"REPORT_DATE_NAME":"2025年报","TOTALOPERATEREVE":86241940222.2,'
        '"PARENTNETPROFIT":34502809176.39,"TOTALOPERATEREVETZ":2.07,'
        '"PARENTNETPROFITTZ":6.17,"XSMLL":61.67,"XSJLL":40.52,'
        '"MGJYXJJE":2.48}'
        ']}}'.replace("]}}", "]}")
    )

    rows = parse_quarterly_financial_payload(payload)

    assert len(rows) == 2
    assert rows[0].period == "2026-03-31"
    assert rows[0].report_name == "2026一季报"
    assert rows[0].revenue_yoy == 6.44
    assert rows[0].operating_cashflow_per_share == 0.48


def test_analysis_adds_quarterly_trend_risk_radar_and_typed_valuation():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":1000000000,"PARENTNETPROFIT":100000000,'
        '"KCFJCXSYJLR":90000000,"TOTALOPERATEREVETZ":5,'
        '"PARENTNETPROFITTZ":8,"ROEJQ":12,"XSMLL":40,"XSJLL":10,'
        '"ZCFZL":65,"EPSJB":1.0,"MGJYXJJE":0.2}]}'
    )
    quarters = parse_quarterly_financial_payload(
        '{"data":['
        '{"REPORT_DATE":"2026-03-31 00:00:00","REPORT_TYPE":"一季报",'
        '"REPORT_DATE_NAME":"2026一季报","TOTALOPERATEREVE":300000000,'
        '"PARENTNETPROFIT":1000000,"TOTALOPERATEREVETZ":10,'
        '"PARENTNETPROFITTZ":-60,"XSMLL":22,"XSJLL":0.3,"MGJYXJJE":-0.1},'
        '{"REPORT_DATE":"2025-12-31 00:00:00","REPORT_TYPE":"年报",'
        '"REPORT_DATE_NAME":"2025年报","TOTALOPERATEREVE":1000000000,'
        '"PARENTNETPROFIT":100000000,"TOTALOPERATEREVETZ":5,'
        '"PARENTNETPROFITTZ":8,"XSMLL":40,"XSJLL":10,"MGJYXJJE":0.2}'
        ']}'
    )
    announcements = parse_announcement_payload(
        '{"data":{"list":[{"art_code":"AN1",'
        '"notice_date":"2026-06-20 00:00:00",'
        '"title":"控股股东拟减持公司股份",'
        '"columns":[{"column_name":"股东减持"}]}]}}',
        "600900",
    )
    valuation_rows = [
        ValuationRow("2026-01-01", pe_ttm=18, pb=2.5, close_price=25),
        ValuationRow("2025-01-01", pe_ttm=30, pb=4.0, close_price=20),
    ]

    result = analyze_stock(
        stock, quote, [], 1.0, financials, 0.0175, [], "",
        None, announcements, valuation_rows=valuation_rows,
        quarterly_financials=quarters,
    )
    report = render_markdown_report(result)

    assert result.quarterly_trend.periods[0].period == "2026-03-31"
    assert result.quarterly_trend.alerts
    assert result.risk_radar.level in ("中", "高")
    assert any("减持" in item for item in result.risk_radar.items)
    assert result.typed_valuation.conclusion
    assert "季度财务趋势" in report
    assert "风险雷达" in report
    assert "类型化估值结论" in report


def test_parse_and_analyze_fund_flow_payload_summarizes_windows():
    payload = (
        '{"data":{"klines":['
        '"2026-06-01,-100,10,-110,50,50,-1,0.1,-1.1,0.5,0.5,10,-1",'
        '"2026-06-02,-80,20,-100,40,40,-0.8,0.2,-1,0.4,0.4,9.9,-1",'
        '"2026-06-03,-60,30,-90,30,30,-0.6,0.3,-0.9,0.3,0.3,9.8,-1",'
        '"2026-06-04,120,80,40,-40,-80,1.2,0.8,0.4,-0.4,-0.8,10.1,3",'
        '"2026-06-05,150,100,50,-50,-100,1.5,1,0.5,-0.5,-1,10.2,1"'
        ']}}'
    )

    rows = parse_fund_flow_payload(payload)
    summary = analyze_fund_flow(rows)

    assert len(rows) == 5
    assert rows[-1].main_net == 150
    assert summary.net_5 == 30
    assert summary.positive_days_5 == 2
    assert summary.trend in ("流入改善", "分歧")


def test_parse_tencent_pankou_payload_builds_fallback_flow_analysis():
    ratios = parse_tencent_pankou_payload('v_s_pksh600900="0.298~0.196~0.260~0.246";')
    summary = analyze_tencent_pankou(ratios)

    assert ratios["buy_large_ratio"] == 0.298
    assert summary.source == "腾讯盘口比例"
    assert summary.fallback_source == "腾讯s_pk"
    assert summary.trend == "盘口均衡"
    assert "不等同于历史主力资金流" in summary.comment


def test_analysis_adds_fund_flow_market_environment_and_support_observation():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~10.20~10.00~10.00~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.20~2.00~10.30~10.00~0/0/0~0~0~0~15.00~~'
        '10.30~10.00~2.00~6000~6000~2.50";'
    )
    kline = parse_kline_payload(
        '{"code":0,"data":{"sh600900":{"qfqday":['
        '["2026-06-01","10","10","10.5","9.8","100"],'
        '["2026-06-02","10","9.9","10.1","9.7","90"],'
        '["2026-06-03","9.9","9.8","10.0","9.7","80"],'
        '["2026-06-04","9.8","10.1","10.2","9.8","120"],'
        '["2026-06-05","10.1","10.2","10.3","10.0","130"]'
        ']}}}',
        "sh600900",
    )
    fund_rows = parse_fund_flow_payload(
        '{"data":{"klines":['
        '"2026-06-01,-100,10,-110,50,50,-1,0.1,-1.1,0.5,0.5,10,-1",'
        '"2026-06-02,-80,20,-100,40,40,-0.8,0.2,-1,0.4,0.4,9.9,-1",'
        '"2026-06-03,-20,30,-50,20,0,-0.2,0.3,-0.5,0.2,0,9.8,-1",'
        '"2026-06-04,120,80,40,-40,-80,1.2,0.8,0.4,-0.4,-0.8,10.1,3",'
        '"2026-06-05,150,100,50,-50,-100,1.5,1,0.5,-0.5,-1,10.2,1"'
        ']}}'
    )
    market_klines = {
        "上证指数": kline,
        "创业板指": kline,
    }

    result = analyze_stock(
        stock, quote, kline, 0.3, [], 0.0175,
        fund_flow_rows=fund_rows,
        market_klines=market_klines,
    )
    report = render_markdown_report(result)

    assert result.fund_flow.net_5 == 70
    assert result.support_observation.level in ("中", "强")
    assert result.market_environment.indices
    assert "资金流动与筹码温度" in report
    assert "大盘环境判断" in report
    assert "隐性承接观察" in report


def test_analysis_uses_tencent_pankou_when_fund_flow_rows_are_missing():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~10.20~10.00~10.00~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.20~2.00~10.30~10.00~0/0/0~0~0~0~15.00~~'
        '10.30~10.00~2.00~6000~6000~2.50";'
    )
    fallback = analyze_tencent_pankou(parse_tencent_pankou_payload(
        'v_s_pksh600900="0.300~0.250~0.220~0.230";'
    ))

    result = analyze_stock(
        stock, quote, [], 0.3, [], 0.0175,
        fund_flow_fallback=fallback,
    )
    report = render_markdown_report(result)

    assert result.fund_flow.source == "腾讯盘口比例"
    assert result.fund_flow.fallback_source == "腾讯s_pk"
    assert result.fund_flow.trend == "盘口偏买"
    assert "腾讯盘口比例" in report
    assert "Level-2逐笔数据" in report


def test_growth_stock_report_includes_actionable_trade_price_zones():
    stock = resolve_numeric_code("603179")
    quote = parse_quote_line(
        'v_sh603179="1~新泉股份~603179~50.00~49.50~49.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.50~1.01~51.00~49.00~0/0/0~0~0~0~28.00~~'
        '51.00~49.00~2.00~300~300~3.50";'
    )
    kline_rows = []
    for i in range(120):
        close = 40 + i * 0.1
        kline_rows.append('["2026-01-%02d","%.2f","%.2f","%.2f","%.2f","100"]' % (
            (i % 28) + 1,
            close,
            close,
            close + 1,
            close - 1,
        ))
    kline = parse_kline_payload(
        '{"code":0,"data":{"sh603179":{"qfqday":[%s]}}}' % ",".join(kline_rows),
        "sh603179",
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":1000000000,"PARENTNETPROFIT":120000000,'
        '"KCFJCXSYJLR":110000000,"TOTALOPERATEREVETZ":22,'
        '"PARENTNETPROFITTZ":28,"ROEJQ":15,"XSMLL":28,"XSJLL":12,'
        '"ZCFZL":45,"EPSJB":1.8,"MGJYXJJE":1.2}]}'
    )

    result = analyze_stock(stock, quote, kline, 0.2, financials, 0.0175)
    report = render_markdown_report(result)

    zones = result.trade_price_zones
    assert zones.left_buy_price is not None
    assert zones.right_buy_low is not None
    assert zones.stop_loss_price is not None
    assert "交易价格区间" in report
    assert "左侧低吸区" in report
    assert "右侧确认区" in report
    assert "元" in zones.left_buy_zone
    assert "元" in zones.stop_loss_zone


def test_reliability_assessment_marks_missing_core_data_low_confidence():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )

    result = analyze_stock(stock, quote, [], None, [], 0.0175)
    report = render_markdown_report(result)

    assert result.reliability.confidence == "低"
    assert result.reliability.data_quality_issues
    assert "结论摘要卡" in report
    assert "结论置信度" in report
    assert "核心依据" in report
    assert "弱依据" in report


def test_reliability_assessment_layers_core_timing_and_weak_evidence():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    kline = parse_kline_payload(
        '{"code":0,"data":{"sh600900":{"qfqday":[' +
        ",".join(
            '["2026-01-%02d","%.2f","%.2f","%.2f","%.2f","100"]' % (
                (i % 28) + 1,
                20 + i * 0.05,
                20 + i * 0.05,
                21 + i * 0.05,
                19 + i * 0.05,
            )
            for i in range(120)
        ) +
        ']}}}',
        "sh600900",
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2025","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":1000000000,"PARENTNETPROFIT":120000000,'
        '"KCFJCXSYJLR":110000000,"TOTALOPERATEREVETZ":8,'
        '"PARENTNETPROFITTZ":10,"ROEJQ":12,"XSMLL":35,"XSJLL":12,'
        '"ZCFZL":50,"EPSJB":1.2,"MGJYXJJE":1.5}]}'
    )
    valuation_rows = [
        ValuationRow("2026-01-01", pe_ttm=18, pb=2.5, close_price=25),
        ValuationRow("2025-01-01", pe_ttm=22, pb=3.0, close_price=20),
        ValuationRow("2024-01-01", pe_ttm=26, pb=3.5, close_price=18),
    ]
    news = parse_news_payload(
        '{"data":{"list":[{"Art_ShowTime":"2026-06-20 08:00:00",'
        '"Art_Title":"公司持续提升股东回报"}]}}'
    )

    result = analyze_stock(
        stock, quote, kline, 1.0, financials, 0.0175,
        valuation_rows=valuation_rows,
        news_items=news,
    )

    assert result.reliability.confidence in ("中", "高")
    assert result.reliability.core_evidence
    assert result.reliability.timing_evidence
    assert result.reliability.weak_evidence


def test_reliability_v2_flags_stale_data_and_downgrades_confidence():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20240101100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    financials = parse_financial_payload(
        '{"data":[{"REPORT_YEAR":"2022","REPORT_TYPE":"年报",'
        '"TOTALOPERATEREVE":1000000000,"PARENTNETPROFIT":120000000,'
        '"KCFJCXSYJLR":110000000,"TOTALOPERATEREVETZ":8,'
        '"PARENTNETPROFITTZ":10,"ROEJQ":12,"XSMLL":35,"XSJLL":12,'
        '"ZCFZL":50,"EPSJB":1.2,"MGJYXJJE":1.5}]}'
    )
    valuation_rows = [ValuationRow("2022-01-01", pe_ttm=18, pb=2.5, close_price=25)]

    result = analyze_stock(
        stock, quote, [], 1.0, financials, 0.0175, valuation_rows=valuation_rows)

    assert result.reliability.freshness_issues
    assert result.reliability.downgrade_reasons
    assert result.reliability.confidence == "低"


def test_reliability_v2_warns_when_low_confidence_has_precise_price_zones():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )

    result = analyze_stock(stock, quote, [], None, [], 0.0175)
    report = render_markdown_report(result)

    assert result.trade_price_zones.left_buy_price is not None
    assert result.reliability.confidence == "低"
    assert result.reliability.consistency_warnings
    assert "一致性警告" in report


def test_reliability_v2_abnormal_value_protection_handles_extreme_pe_and_yield():
    stock = resolve_numeric_code("002115")
    quote = Quote(
        code="002115",
        name="三维通信",
        price=10.0,
        pct_change=0.0,
        high=10.2,
        low=9.8,
        pe=180.0,
        pb=18.0,
        market_cap=80.0,
        timestamp="20260701100000",
    )

    result = analyze_stock(stock, quote, [], 1.2, [], 0.0175)

    joined = "；".join(result.reliability.abnormal_warnings)
    assert "PE" in joined
    assert "PB" in joined
    assert "股息率" in joined
    assert result.reliability.confidence == "低"


def test_quote_cross_check_reports_consistent_secondary_quote():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=26.64, pe=18.0, pb=2.8)
    check_quote = Quote(code="600900", name="长江电力", price=26.65, timestamp="2026-07-01 15:00:02")

    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, quote_check=check_quote)
    report = render_markdown_report(result)

    assert result.quote_cross_check.check_source == "新浪"
    assert result.quote_cross_check.status == "一致"
    assert result.quote_cross_check.warnings == []
    assert "行情交叉校验" in report


def test_quote_cross_check_large_mismatch_adds_reliability_warning():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=26.64, pe=18.0, pb=2.8)
    check_quote = Quote(code="600900", name="长江电力", price=27.20, timestamp="2026-07-01 15:00:02")

    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, quote_check=check_quote)

    assert result.quote_cross_check.status == "明显偏离"
    assert result.reliability.abnormal_warnings
    assert any("腾讯与新浪" in item for item in result.reliability.abnormal_warnings)
    assert result.reliability.downgrade_reasons


def test_report_shows_placeholder_when_no_risk_notes():
    stock = resolve_numeric_code("600900")
    quote = parse_quote_line(
        'v_sh600900="1~长江电力~600900~25.00~24.90~24.80~0~0~0~'
        '0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~'
        '20260701100000~0.10~0.40~25.20~24.70~0/0/0~0~0~0~15.00~~'
        '25.20~24.70~2.00~6000~6000~2.50";'
    )
    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175)
    result.risk_notes = []

    report = render_markdown_report(result)

    assert "暂未发现明显硬性风险" in report


def test_report_and_ai_context_include_data_source_health_summary():
    stock = resolve_numeric_code("600900")
    quote = Quote(code="600900", name="长江电力", price=26.64, pe=18.0, pb=2.8)
    check_quote = Quote(code="600900", name="长江电力", price=26.64, timestamp="2026-07-01 15:00:02")

    result = analyze_stock(stock, quote, [], 1.0, [], 0.0175, quote_check=check_quote)
    report = render_markdown_report(result)
    context = build_ai_context(result)

    assert "数据源健康检查" in report
    assert "腾讯实时行情" in report
    assert "新浪行情校验" in report
    assert "核心数据缺失" in report
    assert "数据源健康" in context
    assert "新浪校验:一致" in context


if __name__ == "__main__":
    test_resolve_numeric_code_adds_market_prefix()
    test_parse_search_hint_decodes_stock_name()
    test_decode_tencent_bytes_accepts_gb18030_payload()
    test_parse_tencent_quote_line_extracts_core_fields()
    test_parse_sina_quote_line_extracts_cross_check_fields()
    test_parse_kline_payload_returns_ma_inputs()
    test_parse_eastmoney_financial_payload_extracts_annual_metrics()
    test_parse_bonus_payload_sums_cash_dividend_by_report_year()
    test_parse_company_survey_payload_extracts_industry_tags()
    test_parse_announcement_payload_extracts_links_and_flags_keywords()
    test_parse_announcement_detail_html_extracts_main_text()
    test_enrich_important_announcements_marks_block_page_as_error()
    test_parse_cninfo_search_payload_extracts_pdf_metadata()
    test_enrich_important_announcements_uses_cninfo_fallback_after_block()
    test_enrich_important_announcements_only_reads_key_items()
    test_extract_announcement_facts_for_reduction_and_buyback_cancel()
    test_analyze_stock_calculates_dividend_buy_prices()
    test_analyze_stock_calculates_price_percentile_from_kline()
    test_technical_indicators_and_trade_gates_are_reported()
    test_parse_valuation_payload_extracts_pe_pb_history()
    test_analyze_valuation_history_calculates_percentiles()
    test_analyze_valuation_history_builds_multi_window_view_and_primary_metric()
    test_ai_context_uses_structured_decision_data()
    test_ai_context_includes_deep_read_announcements()
    test_request_ai_analysis_accepts_injected_client()
    test_openai_client_timeout_uses_env_default()
    test_ai_prompt_requires_industry_aware_risk_interpretation()
    test_report_mentions_limitations_when_financials_missing()
    test_report_includes_historical_valuation_percentiles()
    test_analysis_adds_score_breakdown_profit_quality_and_buy_sell_points()
    test_analysis_adds_dividend_yield_history_from_prices_and_dividends()
    test_analysis_adds_announcement_risk_levels_and_industry_relative_valuation()
    test_report_includes_financial_quality_when_available()
    test_report_includes_dividend_history_when_available()
    test_classifies_weak_loss_maker_as_turnaround_watch()
    test_classifies_high_dividend_cashflow_stock_as_dividend_stable()
    test_report_includes_stock_category_section()
    test_turnaround_report_includes_dedicated_checklist()
    test_dividend_report_includes_dedicated_checklist()
    test_report_includes_industry_tags_when_available()
    test_report_includes_recent_announcements_section()
    test_parse_news_payload_extracts_items_and_sentiment()
    test_parse_research_payload_extracts_rating_summary()
    test_report_and_ai_context_include_news_and_research()
    test_parse_quarterly_financial_payload_extracts_recent_reports()
    test_analysis_adds_quarterly_trend_risk_radar_and_typed_valuation()
    test_parse_and_analyze_fund_flow_payload_summarizes_windows()
    test_parse_tencent_pankou_payload_builds_fallback_flow_analysis()
    test_analysis_adds_fund_flow_market_environment_and_support_observation()
    test_analysis_uses_tencent_pankou_when_fund_flow_rows_are_missing()
    test_growth_stock_report_includes_actionable_trade_price_zones()
    test_reliability_assessment_marks_missing_core_data_low_confidence()
    test_reliability_assessment_layers_core_timing_and_weak_evidence()
    test_reliability_v2_flags_stale_data_and_downgrades_confidence()
    test_reliability_v2_warns_when_low_confidence_has_precise_price_zones()
    test_reliability_v2_abnormal_value_protection_handles_extreme_pe_and_yield()
    test_quote_cross_check_reports_consistent_secondary_quote()
    test_quote_cross_check_large_mismatch_adds_reliability_warning()
    test_report_shows_placeholder_when_no_risk_notes()
    test_report_and_ai_context_include_data_source_health_summary()
    print("OK")
