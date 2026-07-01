# -*- coding: utf-8 -*-
import json
import os
from urllib import request


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 90


def _env_timeout(default=DEFAULT_TIMEOUT):
    raw = os.environ.get("STOCK_ANALYZER_AI_TIMEOUT")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _fmt(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return ("%.2f" % value) + suffix
    return str(value) + suffix


def _pct(value):
    if value is None:
        return "N/A"
    return "%.2f%%" % (value * 100)


def _source_status_summary(result):
    cross = result.quote_cross_check
    fund = result.fund_flow
    items = [
        "腾讯行情:%s" % ("可用" if result.quote.price else "缺失"),
        "新浪校验:%s" % (cross.status if cross.check_source else "未校验"),
        "年度财务:%s" % ("可用" if result.financial.years else "缺失"),
        "历史估值:%s" % ("可用" if result.valuation_history.sample_size else "缺失"),
        "公告:%s" % ("可用" if result.announcements else "缺失"),
        "资金流:%s" % ("可用" if fund.rows else ("腾讯盘口兜底" if fund.fallback_source else "缺失")),
    ]
    return "；".join(items)


def build_ai_context(result):
    latest = result.financial.years[0] if result.financial.years else None
    announcements = result.announcements[:8]
    announcement_lines = []
    for item in announcements:
        announcement_lines.append("- %s %s/%s %s" % (
            item.date or "N/A",
            item.importance,
            ",".join(item.tags) if item.tags else "-",
            item.title,
        ))

    lines = [
        "股票：%s %s.%s" % (result.stock.name, result.stock.code, result.stock.market),
        "当前价：%s" % _fmt(result.quote.price, "元"),
        "行情交叉校验：%s；%s" % (
            result.quote_cross_check.status,
            result.quote_cross_check.comment,
        ),
        "数据源健康：%s" % _source_status_summary(result),
        "PE：%s；PB：%s；总市值：%s亿元" % (
            _fmt(result.quote.pe),
            _fmt(result.quote.pb),
            _fmt(result.quote.market_cap),
        ),
        "投资类型：%s" % result.category.name,
        "分析框架：%s" % result.category.framework,
        "综合判断：%s；估值状态：%s；当前操作：%s；综合分：%s/100" % (
            result.decision.current_view,
            result.decision.valuation_state,
            result.decision.action,
            result.decision.score,
        ),
        "结论可靠性：%s（%s/100）；%s" % (
            result.reliability.confidence,
            result.reliability.score,
            result.reliability.summary,
        ),
        "核心依据：%s" % ("；".join(result.reliability.core_evidence[:6]) or "暂缺"),
        "择时依据：%s" % ("；".join(result.reliability.timing_evidence[:5]) or "暂缺"),
        "弱依据：%s" % ("；".join(result.reliability.weak_evidence[:5]) or "暂缺"),
        "新鲜度提示：%s" % ("；".join(result.reliability.freshness_issues[:5]) or "暂无"),
        "结论降级原因：%s" % ("；".join(result.reliability.downgrade_reasons[:5]) or "暂无"),
        "一致性警告：%s" % ("；".join(result.reliability.consistency_warnings[:5]) or "暂无"),
        "买入区间：%s" % result.decision.buy_zone,
        "观察区间：%s" % result.decision.watch_zone,
        "偏贵区：%s" % result.decision.expensive_zone,
        "风控位：%s" % result.decision.risk_control,
        "股息率：%s；每股分红：%s；分红率：%s；现金流覆盖：%s倍" % (
            _pct(result.dividend.current_yield),
            _fmt(result.dividend.per_share, "元"),
            _pct(result.financial.latest_dividend_payout_ratio),
            _fmt(result.financial.latest_cash_dividend_coverage),
        ),
        "技术状态：%s；MA20：%s；MA60：%s；距120日高点回撤：%s" % (
            result.technical.trend_comment,
            _fmt(result.technical.ma20),
            _fmt(result.technical.ma60),
            _pct(result.technical.drawdown_from_120_high),
        ),
        "技术指标：RSI14=%s；MACD=%s；布林位置=%s；5/20日量比=%s；技术闸门=%s" % (
            _fmt(result.technical.rsi14),
            result.technical.macd_signal or "N/A",
            _pct(result.technical.boll_position),
            _fmt(result.technical.volume_ratio_5_20),
            result.technical.technical_comment,
        ),
        "价格分位：%s" % result.technical.price_position_comment,
        "历史估值分位：%s；样本数：%s" % (
            result.valuation_history.comment,
            result.valuation_history.sample_size,
        ),
        "行业相对估值：%s；同行样本数：%s" % (
            result.industry_valuation.comment,
            result.industry_valuation.peer_count,
        ),
        "利润质量：%s" % (
            result.financial.quality.comment if result.financial.quality else "利润质量信息不足。"
        ),
        "历史股息率：%s" % result.dividend.yield_history.comment,
        "公告风险扫描：%s；等级：%s" % (
            result.announcement_risk.comment,
            result.announcement_risk.level,
        ),
        "评分拆解：估值%s 财务%s 分红%s 技术%s 公告%s 行业%s 总分%s" % (
            result.score_breakdown.valuation,
            result.score_breakdown.financial_quality,
            result.score_breakdown.dividend_quality,
            result.score_breakdown.technical_position,
            result.score_breakdown.announcement_risk,
            result.score_breakdown.industry_relative,
            result.score_breakdown.total,
        ),
        "类型化估值：%s；%s" % (
            result.typed_valuation.framework,
            result.typed_valuation.conclusion,
        ),
        "季度财务趋势：%s；预警：%s" % (
            result.quarterly_trend.comment,
            "；".join(result.quarterly_trend.alerts) if result.quarterly_trend.alerts else "无",
        ),
        "风险雷达：等级%s；风险分%s；%s" % (
            result.risk_radar.level,
            result.risk_radar.score,
            result.risk_radar.comment,
        ),
        "交易价格区间：%s；%s；%s；%s" % (
            result.trade_price_zones.left_buy_zone,
            result.trade_price_zones.right_buy_zone,
            result.trade_price_zones.reduce_zone,
            result.trade_price_zones.stop_loss_zone,
        ),
        "资金流动：%s；来源：%s%s；%s" % (
            result.fund_flow.trend,
            result.fund_flow.source,
            "（%s兜底）" % result.fund_flow.fallback_source if result.fund_flow.fallback_source else "",
            result.fund_flow.comment,
        ),
        "大盘环境：%s；%s" % (
            result.market_environment.level,
            result.market_environment.comment,
        ),
        "隐性承接观察：%s；%s" % (
            result.support_observation.level,
            result.support_observation.comment,
        ),
    ]
    if latest is not None:
        lines.append(
            "最新财务：年度%s；营收同比%s；净利同比%s；ROE%s；净利率%s；负债率%s；EPS%s；每股经营现金流%s" % (
                latest.year,
                _fmt(latest.revenue_yoy, "%"),
                _fmt(latest.net_profit_yoy, "%"),
                _fmt(latest.roe, "%"),
                _fmt(latest.net_margin, "%"),
                _fmt(latest.debt_ratio, "%"),
                _fmt(latest.eps),
                _fmt(latest.operating_cashflow_per_share),
            )
        )
    if result.risk_notes:
        lines.append("风险清单：%s" % "；".join(result.risk_notes[:8]))
    if announcement_lines:
        lines.append("近期公告：")
        lines.extend(announcement_lines)
    if result.news_analysis.items:
        lines.append("近期资讯：%s" % result.news_analysis.comment)
        for item in result.news_analysis.items[:8]:
            lines.append("- %s %s %s" % (
                item.date or "N/A",
                item.sentiment,
                item.title,
            ))
    if result.research_analysis.reports:
        lines.append("机构研报：%s" % result.research_analysis.comment)
        for item in result.research_analysis.reports[:6]:
            lines.append("- %s %s %s %s" % (
                item.date or "N/A",
                item.org or "N/A",
                item.rating or "N/A",
                item.title,
            ))
    deep_lines = []
    for item in announcements:
        if item.detail_summary:
            fact_text = ""
            if item.facts:
                fact_text = "；结构化要点：" + "；".join(
                    "%s-%s" % (fact.fact_type, fact.summary) for fact in item.facts
                )
            deep_lines.append("- %s %s（%s）：%s" % (
                item.date or "N/A",
                item.title,
                item.detail_source or "公告详情",
                item.detail_summary + fact_text,
            ))
    if deep_lines:
        lines.append("重点公告深读：")
        lines.extend(deep_lines)
    return "\n".join(lines)


def build_ai_prompt(context):
    return (
        "你是一名谨慎的A股投研分析师。只基于以下结构化数据做分析，不要编造未给出的事实，"
        "不要引用未提供的新闻、研报、行业均值或未来预测。\n\n"
        "分析规则：\n"
        "1. 先给结论，再给依据；结论要与规则引擎的综合判断保持一致。\n"
        "2. 不要改变规则引擎给出的买入区间、观察区间、偏贵区和风控位；只能解释这些区间为什么成立。\n"
        "3. 必须结合行业属性、商业模式和股票类型理解指标。不要把单一指标机械化判定为风险，"
        "例如公用事业、水电、电力等重资产行业的负债率，需要结合现金流覆盖、盈利稳定性和融资成本解释。\n"
        "4. 风险要分为硬风险和观察风险。硬风险来自亏损、现金流恶化、公告风险、分红不可持续、趋势破位等；"
        "观察风险可以来自估值不便宜、安全边际不足、行业属性导致的杠杆或周期波动。\n"
        "5. 对红利稳定型，重点解释分红可持续性、股息率安全边际、现金流覆盖和股债收益率比较。"
        "对困境反转型，不要用股息率作为主估值框架，重点解释反转证据是否足够。\n"
        "6. 如果提供了重点公告深读，必须优先结合公告正文摘要判断影响；如果只有公告标题，"
        "只能说标题层面提示风险或事项，不能推断正文细节。\n"
        "7. 如果提供了资讯和研报，只能把它们作为市场预期和事件线索；资讯标题不能替代公告，研报评级不能替代估值和财务判断。\n"
        "8. 不要承诺收益，不要使用确定性措辞，不构成投资建议。\n\n"
        "请用中文输出以下小节：当前结论、价格区间解读、买入条件、卖出/减仓条件、需要核对的风险。\n\n"
        "%s" % context
    )


class OpenAICompatibleClient:
    def __init__(self, api_key=None, model=None, base_url=None, timeout=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("STOCK_ANALYZER_AI_MODEL") or DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else _env_timeout()

    def complete(self, prompt):
        if not self.api_key:
            raise RuntimeError("未设置OPENAI_API_KEY，已跳过AI综合分析。")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是谨慎、专业、重证据的A股投研助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def request_ai_analysis(context, client=None):
    client = client or OpenAICompatibleClient()
    return client.complete(build_ai_prompt(context)).strip()


def attach_ai_analysis(result, client=None):
    context = build_ai_context(result)
    result.ai_analysis = request_ai_analysis(context, client=client)
    return result
