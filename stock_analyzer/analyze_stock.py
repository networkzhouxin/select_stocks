# -*- coding: utf-8 -*-
import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_analyzer.analysis import analyze_stock
from stock_analyzer.ai import attach_ai_analysis
from stock_analyzer.announcements import enrich_important_announcements, fetch_announcements
from stock_analyzer.dividends import (
    fetch_bonus_history,
    get_dividend_note,
    get_dividend_per_share,
    latest_annual_dividend_per_share,
    load_dividends,
    summarize_dividend_history,
)
from stock_analyzer.financials import fetch_financials, fetch_quarterly_financials
from stock_analyzer.fundflow import fetch_fund_flow, fetch_tencent_pankou_flow
from stock_analyzer.industry import fetch_industry_info
from stock_analyzer.news import fetch_news
from stock_analyzer.report import render_markdown_report
from stock_analyzer.research import fetch_research_reports
from stock_analyzer.resolver import is_numeric_code, resolve_numeric_code
from stock_analyzer.sina import fetch_sina_quote
from stock_analyzer.tencent import fetch_kline, fetch_quote, search_stock_by_name
from stock_analyzer.valuation import fetch_industry_valuation_rows, fetch_valuation_history


MARKET_INDEX_CODES = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
}


def resolve_stock(query):
    if is_numeric_code(query):
        return resolve_numeric_code(query)
    return search_stock_by_name(query)


def build_report(query, bond_yield, use_ai=False):
    dividends = load_dividends()
    stock = resolve_stock(query)
    quote = fetch_quote(stock.tencent_code)
    kline = fetch_kline(stock.tencent_code, 750)

    quote_check_note = None
    try:
        quote_check = fetch_sina_quote(stock)
    except Exception as exc:
        quote_check = None
        quote_check_note = "新浪行情交叉校验失败：%s" % exc

    industry_note = None
    try:
        industry = fetch_industry_info(stock)
    except Exception as exc:
        industry = None
        industry_note = "行业分类接口获取失败：%s" % exc

    announcement_note = None
    try:
        announcements = fetch_announcements(stock)
        enrich_important_announcements(announcements, stock=stock)
    except Exception as exc:
        announcements = []
        announcement_note = "公告接口获取失败：%s" % exc

    dividend_note = None
    dividend_source = ""
    try:
        dividend_records = fetch_bonus_history(stock)
        dividend_history = summarize_dividend_history(dividend_records)
        dividend_per_share = latest_annual_dividend_per_share(dividend_history)
        dividend_source = "东方财富历史分红"
        if dividend_per_share is None:
            dividend_note = "未抓取到有效现金分红记录，已尝试使用本地配置。"
    except Exception as exc:
        dividend_history = []
        dividend_per_share = None
        dividend_note = "历史分红接口获取失败：%s" % exc

    if dividend_per_share is None:
        dividend_per_share = get_dividend_per_share(stock.code, dividends)
        if dividend_per_share is not None:
            dividend_source = "本地分红配置"

    finance_note = None
    try:
        financials = fetch_financials(stock)
    except Exception as exc:
        financials = []
        finance_note = "财务接口获取失败：%s" % exc

    quarterly_note = None
    try:
        quarterly_financials = fetch_quarterly_financials(stock)
    except Exception as exc:
        quarterly_financials = []
        quarterly_note = "季度财务接口获取失败：%s" % exc

    valuation_note = None
    try:
        valuation_rows = fetch_valuation_history(stock)
    except Exception as exc:
        valuation_rows = []
        valuation_note = "历史估值接口获取失败：%s" % exc

    industry_valuation_note = None
    try:
        latest_valuation = valuation_rows[0] if valuation_rows else None
        industry_valuation_rows = fetch_industry_valuation_rows(
            latest_valuation.board_code if latest_valuation else "",
            latest_valuation.date if latest_valuation else "",
        )
    except Exception as exc:
        industry_valuation_rows = []
        industry_valuation_note = "行业相对估值接口获取失败：%s" % exc

    news_note = None
    try:
        news_items = fetch_news(stock)
    except Exception as exc:
        news_items = []
        news_note = "近期资讯接口获取失败：%s" % exc

    research_note = None
    try:
        research_reports = fetch_research_reports(stock)
    except Exception as exc:
        research_reports = []
        research_note = "机构研报接口获取失败：%s" % exc

    fund_flow_note = None
    fund_flow_fallback = None
    try:
        fund_flow_rows = fetch_fund_flow(stock)
    except Exception as exc:
        fund_flow_rows = []
        try:
            fund_flow_fallback = fetch_tencent_pankou_flow(stock)
            fund_flow_note = "资金流接口获取失败：%s；已使用腾讯盘口比例兜底。" % exc
        except Exception as fallback_exc:
            fund_flow_note = "资金流接口获取失败：%s；腾讯盘口比例兜底也失败：%s" % (exc, fallback_exc)

    market_note = None
    market_klines = {}
    try:
        for index_name, index_code in MARKET_INDEX_CODES.items():
            market_klines[index_name] = fetch_kline(index_code, 120)
    except Exception as exc:
        market_klines = {}
        market_note = "大盘指数K线获取失败：%s" % exc

    result = analyze_stock(
        stock=stock,
        quote=quote,
        kline=kline,
        dividend_per_share=dividend_per_share,
        financials=financials,
        bond_yield=bond_yield,
        dividend_history=dividend_history,
        dividend_source=dividend_source,
        industry=industry,
        announcements=announcements,
        valuation_rows=valuation_rows,
        industry_valuation_rows=industry_valuation_rows,
        news_items=news_items,
        research_reports=research_reports,
        quarterly_financials=quarterly_financials,
        fund_flow_rows=fund_flow_rows,
        fund_flow_fallback=fund_flow_fallback,
        market_klines=market_klines,
        quote_check=quote_check,
    )
    note = get_dividend_note(stock.code, dividends)
    if note and dividend_source == "本地分红配置":
        result.data_notes.insert(2, "分红配置说明：%s" % note)
    if dividend_note:
        result.data_notes.insert(2, dividend_note)
    if finance_note:
        result.data_notes.insert(2, finance_note)
    if quarterly_note:
        result.data_notes.insert(2, quarterly_note)
    if valuation_note:
        result.data_notes.insert(2, valuation_note)
    if industry_valuation_note:
        result.data_notes.insert(2, industry_valuation_note)
    if news_note:
        result.data_notes.insert(2, news_note)
    if research_note:
        result.data_notes.insert(2, research_note)
    if fund_flow_note:
        result.data_notes.insert(2, fund_flow_note)
    if market_note:
        result.data_notes.insert(2, market_note)
    if quote_check_note:
        result.data_notes.insert(2, quote_check_note)
    if industry_note:
        result.data_notes.insert(2, industry_note)
    if announcement_note:
        result.data_notes.insert(2, announcement_note)
    if use_ai:
        try:
            attach_ai_analysis(result)
            result.data_notes.append("AI综合分析来自配置的大模型接口，仅用于辅助归纳，不替代原始数据核对。")
        except Exception as exc:
            result.data_notes.append("AI综合分析已跳过：%s" % exc)
    return result


def default_output_path(result):
    name = "%s_%s_analysis.md" % (result.stock.code, result.stock.name)
    return os.path.join(os.getcwd(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="A股股票深度分析命令行工具")
    parser.add_argument("query", help="股票代码或名称，例如 600900 / 长江电力")
    parser.add_argument("--bond-yield", type=float, default=0.0175,
                        help="10年期国债收益率，默认0.0175表示1.75%%")
    parser.add_argument("--ai", action="store_true",
                        help="启用AI综合分析，需要设置OPENAI_API_KEY；可选STOCK_ANALYZER_AI_MODEL和OPENAI_BASE_URL")
    parser.add_argument("--save", action="store_true", help="保存Markdown报告到当前目录")
    parser.add_argument("--output", help="指定Markdown报告保存路径")
    args = parser.parse_args(argv)

    try:
        result = build_report(args.query, args.bond_yield, use_ai=args.ai)
        report = render_markdown_report(result)
    except Exception as exc:
        print("分析失败：%s" % exc, file=sys.stderr)
        return 1

    print(report)
    if args.save or args.output:
        output = args.output or default_output_path(result)
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
        print("报告已保存：%s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
