# -*- coding: utf-8 -*-


def _pct(value):
    if value is None:
        return "N/A"
    return "%.2f%%" % (value * 100)


def _pct_point(value):
    if value is None:
        return "N/A"
    return "%.2f%%" % value


def _num(value, suffix=""):
    if value is None:
        return "N/A"
    return ("%.2f" % value) + suffix


def _money_yi(value):
    if value is None:
        return "N/A"
    return "%.2f亿" % (value / 100000000.0)


def _status(ok):
    return "通过" if ok else "不通过"


def _latest(financial):
    return financial.years[0] if financial.years else None


def _buy_suitability(action):
    if action in ("可分批关注", "估值合理，可关注", "可关注", "小仓位试错"):
        return "可以小仓位或分批关注，但仍需满足买入条件。"
    if action in ("等待回调", "回调后关注", "等待企稳", "综合观察"):
        return "暂不适合直接追买，更适合等待回调或趋势企稳。"
    return "当前不适合买入，先观察风险和基本面变化。"


def _sentiment_label(value):
    if value == "positive":
        return "偏正面"
    if value == "negative":
        return "偏风险"
    return "中性"


def _append_reliability_summary(lines, result):
    reliability = result.reliability
    decision = result.decision
    lines.append("## 结论摘要卡")
    lines.append("")
    lines.append("| 项目 | 结论 |")
    lines.append("|---|---|")
    lines.append("| 当前操作 | %s |" % decision.action)
    lines.append("| 结论置信度 | %s（%d/100） |" % (reliability.confidence, reliability.score))
    lines.append("| 当前是否适合买入 | %s |" % _buy_suitability(decision.action))
    lines.append("| 主要买入条件 | %s |" % (result.trade_price_zones.left_buy_zone or decision.buy_zone))
    lines.append("| 主要风控位 | %s |" % (result.trade_price_zones.stop_loss_zone or decision.risk_control))
    lines.append("")
    lines.append("- 置信度说明：%s" % reliability.summary)
    if reliability.abnormal_warnings:
        lines.append("- 异常提示：%s" % "；".join(reliability.abnormal_warnings[:4]))
    if reliability.data_quality_issues:
        lines.append("- 数据缺口：%s" % "；".join(reliability.data_quality_issues[:5]))
    if reliability.freshness_issues:
        lines.append("- 新鲜度提示：%s" % "；".join(reliability.freshness_issues[:5]))
    if reliability.downgrade_reasons:
        lines.append("- 结论降级：%s" % "；".join(reliability.downgrade_reasons[:5]))
    if reliability.consistency_warnings:
        lines.append("- 一致性警告：%s" % "；".join(reliability.consistency_warnings[:5]))
    lines.append("- 可靠性规则版本：%s" % reliability.rule_version)
    lines.append("")
    lines.append("### 依据分层")
    lines.append("")
    lines.append("- 核心依据：%s" % ("；".join(reliability.core_evidence[:6]) if reliability.core_evidence else "暂缺"))
    lines.append("- 择时依据：%s" % ("；".join(reliability.timing_evidence[:5]) if reliability.timing_evidence else "暂缺"))
    lines.append("- 弱依据：%s" % ("；".join(reliability.weak_evidence[:5]) if reliability.weak_evidence else "暂缺"))
    lines.append("")


def _source_status_rows(result):
    fund = result.fund_flow
    cross = result.quote_cross_check
    return [
        ("核心", "腾讯实时行情", "可用" if result.quote.price else "缺失",
         "当前价、PE/PB、总市值"),
        ("核心", "新浪行情校验",
         cross.status if cross.check_source else "未校验",
         cross.comment),
        ("核心", "年度财务", "可用" if result.financial.years else "缺失",
         "盈利质量、ROE、现金流、负债"),
        ("核心", "历史估值", "可用" if result.valuation_history.sample_size else "缺失",
         "PE/PB历史分位"),
        ("核心", "近期公告", "可用" if result.announcements else "缺失",
         "公告风险与重大事项"),
        ("辅助", "季度财务", "可用" if result.quarterly_trend.periods else "缺失",
         "近期经营趋势"),
        ("辅助", "行业相对估值", "可用" if result.industry_valuation.peer_count else "缺失",
         "同行PE/PB位置"),
        ("辅助", "资金流", "可用" if fund.rows else ("兜底" if fund.fallback_source else "缺失"),
         fund.source + ("/%s" % fund.fallback_source if fund.fallback_source else "")),
        ("辅助", "大盘环境", "可用" if result.market_environment.indices else "缺失",
         "只作择时背景"),
        ("弱辅助", "资讯标题", "可用" if result.news_analysis.items else "缺失",
         "事件线索，不作事实依据"),
        ("弱辅助", "机构研报", "可用" if result.research_analysis.reports else "缺失",
         "市场预期线索"),
    ]


def _append_source_health(lines, result):
    lines.append("## 数据源健康检查")
    lines.append("")
    lines.append("| 层级 | 数据源 | 状态 | 用途/说明 |")
    lines.append("|---|---|---|---|")
    for level, source, status, note in _source_status_rows(result):
        lines.append("| %s | %s | %s | %s |" % (
            level,
            source,
            status,
            (note or "").replace("|", "/"),
        ))
    lines.append("")
    lines.append("> 核心数据缺失或行情校验明显偏离时，最终结论应自动降级；辅助数据只能改善观察质量，不能单独决定买卖。")
    lines.append("")


def _append_final_conclusion(lines, result):
    decision = result.decision
    zones = result.trade_price_zones
    stock_name = result.stock.name
    lines.append("## 最终结论")
    lines.append("")
    lines.append("- 当前操作：%s" % decision.action)
    lines.append("- 当前是否适合买入：%s" % _buy_suitability(decision.action))
    lines.append("- 买入区间：%s" % (zones.left_buy_zone or decision.buy_zone))
    lines.append("- 左侧买点：%s" % ((zones.left_buy_zone or decision.left_buy_point).replace("左侧低吸区：", "").replace("左侧买点：", "")))
    lines.append("- 右侧买点：%s" % ((zones.right_buy_zone or decision.right_buy_point).replace("右侧确认区：", "").replace("右侧买点：", "")))
    lines.append("- 观察区间：%s" % (zones.watch_zone or decision.watch_zone))
    lines.append("- 卖出/减仓区间：%s" % (zones.reduce_zone or decision.expensive_zone))
    lines.append("- 触发减仓：%s" % ((zones.reduce_zone or decision.reduce_point).replace("第一减仓区：", "").replace("减仓点：", "")))
    lines.append("- 风控位：%s" % (zones.stop_loss_zone or decision.risk_control))
    lines.append("- 一句话结论：%s当前%s；买入前应优先确认价格区间、公告风险和基本面条件是否同时满足。"
                 % (stock_name, decision.action))
    lines.append("")


def _append_type_specific_section(lines, result):
    category = result.category
    dividend = result.dividend
    financial = result.financial
    tech = result.technical
    latest = _latest(financial)

    if category.code == "dividend_stable":
        lines.append("## 红利股检查表")
        lines.append("")
        checks = [
            ("股息率达标", dividend.current_yield is not None and dividend.current_yield >= 0.035,
             _pct(dividend.current_yield)),
            ("分红率不过高", financial.latest_dividend_payout_ratio is not None
             and financial.latest_dividend_payout_ratio <= 0.90,
             _pct(financial.latest_dividend_payout_ratio)),
            ("现金流覆盖分红", financial.latest_cash_dividend_coverage is not None
             and financial.latest_cash_dividend_coverage >= 1.0,
             _num(financial.latest_cash_dividend_coverage, "倍")),
            ("ROE具备稳定性", latest is not None and latest.roe is not None and latest.roe >= 8,
             _pct_point(latest.roe if latest else None)),
            ("负债率可观察", latest is not None and latest.debt_ratio is not None and latest.debt_ratio <= 70,
             _pct_point(latest.debt_ratio if latest else None)),
        ]
        lines.append("| 检查项 | 结果 | 当前值 |")
        lines.append("|---|---|---:|")
        for name, ok, value in checks:
            lines.append("| %s | %s | %s |" % (name, _status(ok), value))
        lines.append("")
        lines.append("红利股重点不是看短期股价弹性，而是确认分红是否可持续、现金流是否能覆盖、买入股息率是否有安全边际。")
        lines.append("")
        return

    if category.code == "turnaround_watch":
        lines.append("## 困境反转检查表")
        lines.append("")
        checks = [
            ("扣非/归母利润转正", latest is not None and latest.parent_net_profit is not None
             and latest.parent_net_profit > 0, _money_yi(latest.parent_net_profit if latest else None)),
            ("经营现金流转正", latest is not None and latest.operating_cashflow_per_share is not None
             and latest.operating_cashflow_per_share > 0,
             _num(latest.operating_cashflow_per_share if latest else None)),
            ("毛利率改善", latest is not None and latest.gross_margin is not None and latest.gross_margin >= 8,
             _pct_point(latest.gross_margin if latest else None)),
            ("净利率恢复", latest is not None and latest.net_margin is not None and latest.net_margin >= 2,
             _pct_point(latest.net_margin if latest else None)),
            ("技术趋势修复", tech.ma60 is not None and result.quote.price >= tech.ma60,
             "现价%s / MA60 %s" % (_num(result.quote.price), _num(tech.ma60))),
        ]
        lines.append("| 检查项 | 结果 | 当前值 |")
        lines.append("|---|---|---:|")
        for name, ok, value in checks:
            lines.append("| %s | %s | %s |" % (name, _status(ok), value))
        lines.append("")
        lines.append("困境反转型的关键不是股价跌了多少，而是反转证据是否成立。至少要看到盈利、现金流、利润率和趋势中的多项改善。")
        lines.append("")
        return

    if category.code == "growth":
        lines.append("## 成长股检查表")
        lines.append("")
        latest_yoy = latest.net_profit_yoy if latest else None
        checks = [
            ("营收保持增长", latest is not None and latest.revenue_yoy is not None and latest.revenue_yoy >= 10,
             _pct_point(latest.revenue_yoy if latest else None)),
            ("利润保持增长", latest_yoy is not None and latest_yoy >= 10, _pct_point(latest_yoy)),
            ("ROE较强", latest is not None and latest.roe is not None and latest.roe >= 12,
             _pct_point(latest.roe if latest else None)),
            ("现金流质量可接受", latest is not None and latest.operating_cashflow_per_share is not None
             and latest.operating_cashflow_per_share > 0,
             _num(latest.operating_cashflow_per_share if latest else None)),
        ]
        lines.append("| 检查项 | 结果 | 当前值 |")
        lines.append("|---|---|---:|")
        for name, ok, value in checks:
            lines.append("| %s | %s | %s |" % (name, _status(ok), value))
        lines.append("")
        lines.append("成长型股票重点看增长质量和估值成长匹配，股息率通常不是主判断依据。")
        lines.append("")
        return

    lines.append("## 综合观察检查表")
    lines.append("")
    lines.append("| 检查项 | 观察重点 |")
    lines.append("|---|---|")
    lines.append("| 财务质量 | ROE、净利率、现金流是否稳定 |")
    lines.append("| 估值位置 | PE/PB是否与行业和自身历史匹配 |")
    lines.append("| 分红质量 | 分红是否连续、分红率是否合理 |")
    lines.append("| 技术位置 | 是否站回中期均线、是否停止创新低 |")
    lines.append("")


def render_markdown_report(result):
    stock = result.stock
    quote = result.quote
    dividend = result.dividend
    financial = result.financial
    category = result.category
    industry = result.industry
    tech = result.technical
    decision = result.decision

    lines = []
    lines.append("# %s %s.%s 深度分析" % (stock.name, stock.code, stock.market))
    lines.append("")
    _append_reliability_summary(lines, result)
    _append_source_health(lines, result)

    lines.append("## 核心结论")
    lines.append("")
    lines.append("- %s" % result.valuation_comment)
    lines.append("- 股票类型：%s" % category.name)
    lines.append("- 财务质量：%s" % financial.quality_comment)
    lines.append("- 分红可持续性：%s" % financial.dividend_comment)
    lines.append("- 技术状态：%s" % tech.trend_comment)
    lines.append("")

    lines.append("## 行业分类")
    lines.append("")
    if any([industry.exchange_board, industry.market, industry.csrc_industry, industry.eastmoney_industry]):
        lines.append("| 分类体系 | 结果 |")
        lines.append("|---|---|")
        lines.append("| 交易所/板块 | %s / %s |" % (industry.market or "N/A", industry.exchange_board or "N/A"))
        lines.append("| 证监会行业 | %s |" % (industry.csrc_industry or "N/A"))
        lines.append("| 东方财富行业 | %s |" % (industry.eastmoney_industry or "N/A"))
    else:
        lines.append("暂缺行业分类数据。")
    lines.append("")

    lines.append("## 股票类型")
    lines.append("")
    lines.append("- 类型：%s" % category.name)
    lines.append("- 分析框架：%s" % category.framework)
    lines.append("- 红利估值适用性：%s" % ("适用" if category.dividend_valuation_applicable else "不适用，仅作数学参考"))
    if category.focus_points:
        lines.append("- 重点关注：%s" % "；".join(category.focus_points))
    lines.append("")

    lines.append("## 综合判断")
    lines.append("")
    lines.append("| 项目 | 结论 |")
    lines.append("|---|---|")
    lines.append("| 当前判断 | %s |" % decision.current_view)
    lines.append("| 估值状态 | %s |" % decision.valuation_state)
    lines.append("| 当前操作 | %s |" % decision.action)
    lines.append("| 综合分 | %s/100 |" % decision.score)
    lines.append("| 买入区间 | %s |" % decision.buy_zone)
    lines.append("| 观察区间 | %s |" % decision.watch_zone)
    lines.append("| 偏贵/追高区 | %s |" % decision.expensive_zone)
    lines.append("| 风控位 | %s |" % decision.risk_control)
    lines.append("")
    if decision.reasons:
        lines.append("### 判断依据")
        lines.append("")
        for reason in decision.reasons:
            lines.append("- %s" % reason)
        lines.append("")
    if decision.sell_signals:
        lines.append("### 卖出/减仓条件")
        lines.append("")
        for signal in decision.sell_signals:
            lines.append("- %s" % signal)
        lines.append("")

    zones = result.trade_price_zones
    lines.append("## 交易价格区间")
    lines.append("")
    lines.append("- %s" % (zones.comment or "价格区间基于估值、均线和近期高低点生成。"))
    lines.append("")
    lines.append("| 区间 | 价格/条件 |")
    lines.append("|---|---|")
    lines.append("| 左侧低吸区 | %s |" % (zones.left_buy_zone or "N/A"))
    lines.append("| 右侧确认区 | %s |" % (zones.right_buy_zone or "N/A"))
    lines.append("| 观察区 | %s |" % (zones.watch_zone or "N/A"))
    lines.append("| 追高风险区 | %s |" % (zones.chase_risk_zone or "N/A"))
    lines.append("| 第一减仓区 | %s |" % (zones.reduce_zone or "N/A"))
    lines.append("| 风险止损位 | %s |" % (zones.stop_loss_zone or "N/A"))
    lines.append("")

    typed = result.typed_valuation
    lines.append("## 类型化估值结论")
    lines.append("")
    lines.append("- 框架：%s" % (typed.framework or "暂缺类型化估值框架。"))
    lines.append("- 结论：%s" % (typed.conclusion or "暂缺类型化估值结论。"))
    if typed.key_metrics:
        lines.append("- 关键指标：%s" % "；".join(typed.key_metrics))
    lines.append("")

    qtrend = result.quarterly_trend
    lines.append("## 季度财务趋势")
    lines.append("")
    lines.append("- %s" % qtrend.comment)
    if qtrend.alerts:
        lines.append("- 预警：%s" % "；".join(qtrend.alerts))
    if qtrend.periods:
        lines.append("")
        lines.append("| 报告期 | 类型 | 营收同比 | 净利同比 | 扣非同比 | 毛利率 | 净利率 | 每股经营现金流 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in qtrend.periods[:6]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.period,
                row.report_name or "N/A",
                _pct_point(row.revenue_yoy),
                _pct_point(row.net_profit_yoy),
                _pct_point(row.deduct_net_profit_yoy),
                _pct_point(row.gross_margin),
                _pct_point(row.net_margin),
                _num(row.operating_cashflow_per_share),
            ))
    else:
        lines.append("暂缺可解析的季度财务数据。")
    lines.append("")

    radar = result.risk_radar
    lines.append("## 风险雷达")
    lines.append("")
    lines.append("- 风险等级：%s" % radar.level)
    lines.append("- 风险分：%d" % radar.score)
    lines.append("- 风险说明：%s" % radar.comment)
    if radar.items:
        lines.append("")
        for item in radar.items[:8]:
            lines.append("- %s" % item)
    lines.append("")

    fund = result.fund_flow
    lines.append("## 资金流动与筹码温度")
    lines.append("")
    lines.append("- 数据来源：%s%s" % (
        fund.source,
        "（%s兜底）" % fund.fallback_source if fund.fallback_source else "",
    ))
    lines.append("- 资金趋势：%s" % fund.trend)
    lines.append("- 资金说明：%s" % fund.comment)
    lines.append("")
    if fund.rows:
        lines.append("| 窗口 | 主力净额 | 流入天数 |")
        lines.append("|---|---:|---:|")
        lines.append("| 5日 | %s | %d |" % (_money_yi(fund.net_5), fund.positive_days_5))
        lines.append("| 10日 | %s | %d |" % (_money_yi(fund.net_10), fund.positive_days_10))
        lines.append("| 20日 | %s | N/A |" % _money_yi(fund.net_20))
        lines.append("")
        lines.append("| 日期 | 主力净额 | 超大单 | 大单 | 收盘价 | 涨跌幅 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in fund.rows[-5:]:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                row.date,
                _money_yi(row.main_net),
                _money_yi(row.super_large_net),
                _money_yi(row.large_net),
                _num(row.close),
                _pct_point(row.pct_change),
            ))
    elif fund.fallback_source:
        lines.append("| 腾讯盘口项 | 比例 |")
        lines.append("|---|---:|")
        lines.append("| 买盘大单 | %s |" % _pct(fund.buy_large_ratio))
        lines.append("| 买盘小单 | %s |" % _pct(fund.buy_small_ratio))
        lines.append("| 卖盘大单 | %s |" % _pct(fund.sell_large_ratio))
        lines.append("| 卖盘小单 | %s |" % _pct(fund.sell_small_ratio))
        lines.append("")
        lines.append("> 腾讯盘口比例只作资金温度兜底，不等同于东方财富历史主力净额，也不等同于Level-2逐笔数据。")
    else:
        lines.append("暂缺资金流数据。")
    lines.append("")

    market = result.market_environment
    lines.append("## 大盘环境判断")
    lines.append("")
    lines.append("- 环境：%s" % market.level)
    lines.append("- 说明：%s" % market.comment)
    if market.indices:
        lines.append("")
        lines.append("| 指数 | 收盘 | MA20 | MA60 | 趋势 |")
        lines.append("|---|---:|---:|---:|---|")
        for item in market.indices:
            lines.append("| %s | %s | %s | %s | %s |" % (
                item.name,
                _num(item.close),
                _num(item.ma20),
                _num(item.ma60),
                item.trend,
            ))
    lines.append("")

    support = result.support_observation
    lines.append("## 隐性承接观察")
    lines.append("")
    lines.append("- 承接强度：%s" % support.level)
    lines.append("- 承接分：%d" % support.score)
    lines.append("- 观察结论：%s" % support.comment)
    if support.signals:
        lines.append("")
        for signal in support.signals:
            lines.append("- %s" % signal)
    lines.append("")

    score = result.score_breakdown
    lines.append("## 评分拆解")
    lines.append("")
    lines.append("> 分项分用于解释优势和短板，综合分由规则引擎结合股票类型给出，不做机械相加。")
    lines.append("")
    lines.append("| 维度 | 分数 |")
    lines.append("|---|---:|")
    lines.append("| 估值位置 | %d |" % score.valuation)
    lines.append("| 财务质量 | %d |" % score.financial_quality)
    lines.append("| 分红质量 | %d |" % score.dividend_quality)
    lines.append("| 技术位置 | %d |" % score.technical_position)
    lines.append("| 公告风险 | %d |" % score.announcement_risk)
    lines.append("| 行业比较 | %d |" % score.industry_relative)
    lines.append("| 综合分 | %d |" % score.total)
    lines.append("")

    if result.ai_analysis:
        lines.append("## AI综合分析")
        lines.append("")
        lines.append(result.ai_analysis.strip())
        lines.append("")

    _append_type_specific_section(lines, result)

    risk = result.announcement_risk
    lines.append("## 公告风险扫描")
    lines.append("")
    lines.append("- 风险等级：%s" % risk.level)
    lines.append("- 风险说明：%s" % risk.comment)
    lines.append("")

    lines.append("## 近期重要公告")
    lines.append("")
    if result.announcements:
        lines.append("| 日期 | 标记 | 类型 | 标题 | 链接 |")
        lines.append("|---|---|---|---|---|")
        for item in result.announcements[:12]:
            mark = "风险" if item.importance == "risk" else "重要" if item.importance == "important" else "普通"
            tag_text = "、".join(item.tags) if item.tags else "-"
            title = item.title.replace("|", "/")
            link = "[查看](%s)" % item.url if item.url else ""
            lines.append("| %s | %s/%s | %s | %s | %s |" % (
                item.date or "N/A",
                mark,
                tag_text,
                item.category or "N/A",
                title,
                link,
            ))
    else:
        lines.append("暂未获取到近期公告。")
    lines.append("")

    deep_items = [item for item in result.announcements if item.detail_summary or item.detail_error]
    if deep_items:
        lines.append("### 重点公告深读")
        lines.append("")
        for item in deep_items:
            lines.append("- **%s %s**" % (item.date or "N/A", item.title))
            if item.detail_source:
                source_line = item.detail_source
                if item.detail_url:
                    source_line += "：[查看](%s)" % item.detail_url
                lines.append("  - 来源：%s" % source_line)
            if item.detail_summary:
                lines.append("  - 摘要：%s" % item.detail_summary)
            if item.facts:
                for fact in item.facts:
                    lines.append("  - 结构化要点：%s - %s" % (fact.fact_type, fact.summary))
            if item.detail_error:
                lines.append("  - 读取失败：%s" % item.detail_error)
        lines.append("")

    news = result.news_analysis
    lines.append("## 近期资讯事件")
    lines.append("")
    lines.append("- %s" % news.comment)
    if news.items:
        lines.append("")
        lines.append("| 日期 | 情绪 | 标签 | 标题 | 来源 | 链接 |")
        lines.append("|---|---|---|---|---|---|")
        for item in news.items[:12]:
            tag_text = "、".join(item.tags) if item.tags else "-"
            link = "[查看](%s)" % item.url if item.url else ""
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                item.date or "N/A",
                _sentiment_label(item.sentiment),
                tag_text,
                item.title.replace("|", "/"),
                item.source or "N/A",
                link,
            ))
    else:
        lines.append("暂未获取到近期资讯。")
    lines.append("")

    research = result.research_analysis
    lines.append("## 机构研报观点")
    lines.append("")
    lines.append("- %s" % research.comment)
    if research.reports:
        lines.append("")
        lines.append("| 日期 | 机构 | 评级 | 标题 | 分析师 | 链接 |")
        lines.append("|---|---|---|---|---|---|")
        for item in research.reports[:10]:
            link = "[查看](%s)" % item.url if item.url else ""
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                item.date or "N/A",
                item.org or "N/A",
                item.rating or "N/A",
                item.title.replace("|", "/"),
                item.analyst or "N/A",
                link,
            ))
        lines.append("")
        lines.append("> 研报代表卖方机构观点，重点用于观察市场预期和盈利预测变化，不直接决定买卖。")
    else:
        lines.append("暂未获取到近期机构研报。")
    lines.append("")

    lines.append("## 行情概览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append("| 当前价 | %s |" % _num(quote.price, "元"))
    lines.append("| 涨跌幅 | %s |" % (("%.2f%%" % quote.pct_change) if quote.pct_change is not None else "N/A"))
    lines.append("| 最高/最低 | %s / %s |" % (_num(quote.high), _num(quote.low)))
    lines.append("| PE | %s |" % _num(quote.pe))
    lines.append("| PB | %s |" % _num(quote.pb))
    lines.append("| 总市值 | %s亿元 |" % _num(quote.market_cap))
    lines.append("| 更新时间 | %s |" % (quote.timestamp or "N/A"))
    lines.append("")

    cross = result.quote_cross_check
    if cross.check_source:
        lines.append("### 行情交叉校验")
        lines.append("")
        lines.append("- 状态：%s" % cross.status)
        lines.append("- 说明：%s" % cross.comment)
        if cross.timestamp:
            lines.append("- 新浪时间：%s" % cross.timestamp)
        if cross.warnings:
            lines.append("- 警告：%s" % "；".join(cross.warnings))
        lines.append("")

    valuation = result.valuation_history
    lines.append("## 历史估值分位")
    lines.append("")
    if valuation.sample_size:
        lines.append("- 主估值指标：%s" % valuation.primary_metric)
        lines.append("- 框架说明：%s" % valuation.framework_comment)
        lines.append("- 近3年估值位置：%s" % valuation.comment)
        lines.append("")
        lines.append("| 窗口 | 样本 | PE(TTM)历史分位 | PE区间 | PB历史分位 | PB区间 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label in ("3年", "5年", "10年"):
            window = valuation.windows.get(label)
            if window is None:
                continue
            sample_note = "%d" % window.sample_size
            if not window.is_full_window:
                sample_note += "（样本不足）"
            lines.append("| 近%s | %s | %s | %s-%s | %s | %s-%s |" % (
                label,
                sample_note,
                _pct(window.pe_percentile),
                _num(window.pe_min),
                _num(window.pe_max),
                _pct(window.pb_percentile),
                _num(window.pb_min),
                _num(window.pb_max),
            ))
    else:
        lines.append("暂缺PE/PB历史估值分位数据。")
    lines.append("")

    industry_val = result.industry_valuation
    lines.append("## 行业相对估值")
    lines.append("")
    if industry_val.peer_count:
        lines.append("- 同行样本数：%d" % industry_val.peer_count)
        lines.append("- 相对位置：%s" % industry_val.comment)
        lines.append("")
        lines.append("| 指标 | 当前值 | 行业中位数 | 行业内分位 |")
        lines.append("|---|---:|---:|---:|")
        lines.append("| PE | %s | %s | %s |" % (
            _num(quote.pe),
            _num(industry_val.industry_pe_median),
            _pct(industry_val.pe_percentile),
        ))
        lines.append("| PB | %s | %s | %s |" % (
            _num(quote.pb),
            _num(industry_val.industry_pb_median),
            _pct(industry_val.pb_percentile),
        ))
    else:
        lines.append("暂缺行业相对估值数据。")
    lines.append("")

    quality = financial.quality
    lines.append("## 利润质量")
    lines.append("")
    if quality:
        lines.append("- %s" % quality.comment)
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---:|")
        lines.append("| 净利润现金含量 | %s |" % _num(quality.cash_profit_ratio, "倍"))
        lines.append("| 扣非净利润占比 | %s |" % _num(quality.deduct_profit_ratio))
        lines.append("| 毛利率趋势 | %s |" % (quality.gross_margin_trend or "N/A"))
        lines.append("| 净利率趋势 | %s |" % (quality.net_margin_trend or "N/A"))
        lines.append("| 经营现金流趋势 | %s |" % (quality.cashflow_trend or "N/A"))
    else:
        lines.append("利润质量信息不足。")
    lines.append("")

    lines.append("## 财务质量")
    lines.append("")
    if not financial.years:
        lines.append("暂缺最近年报财务数据，无法自动分析营收、利润、ROE、现金流和负债。")
    else:
        lines.append("| 年度 | 营收 | 归母净利润 | 营收同比 | 净利同比 | ROE | 毛利率 | 净利率 | 负债率 | EPS | 每股经营现金流 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in financial.years:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    row.year,
                    _money_yi(row.revenue),
                    _money_yi(row.parent_net_profit),
                    _pct_point(row.revenue_yoy),
                    _pct_point(row.net_profit_yoy),
                    _pct_point(row.roe),
                    _pct_point(row.gross_margin),
                    _pct_point(row.net_margin),
                    _pct_point(row.debt_ratio),
                    _num(row.eps),
                    _num(row.operating_cashflow_per_share),
                )
            )
    lines.append("")

    lines.append("## 股息与买入价格区间")
    lines.append("")
    if not category.dividend_valuation_applicable:
        lines.append("> 当前股票类型不适合把股息率买入价作为主估值框架，以下价格仅作数学参考。")
        lines.append("")
    if dividend.per_share is None or dividend.current_yield is None:
        lines.append("暂缺可持续每股分红，无法计算股息率买入价。")
    else:
        lines.append("- 每股现金分红：%.2f元" % dividend.per_share)
        lines.append("- 当前股息率：%s" % _pct(dividend.current_yield))
        lines.append("")
        lines.append("| 目标股息率 | 对应买入价 |")
        lines.append("|---:|---:|")
        for target, price in dividend.target_prices.items():
            lines.append("| %s | %.2f元 |" % (_pct(target), price))
        lines.append("")
        lines.append("| 国债收益率+风险溢价 | 对应买入价 |")
        lines.append("|---:|---:|")
        for required, price in dividend.bond_risk_prices.items():
            lines.append("| %s | %.2f元 |" % (_pct(required), price))
    lines.append("")

    lines.append("## 分红可持续性")
    lines.append("")
    lines.append("- %s" % financial.dividend_comment)
    lines.append("- 分红率（每股分红/EPS）：%s" % _pct(financial.latest_dividend_payout_ratio))
    lines.append("- 经营现金流覆盖倍数（每股经营现金流/每股分红）：%s" % _num(financial.latest_cash_dividend_coverage, "倍"))
    if dividend.source:
        lines.append("- 分红数据来源：%s" % dividend.source)
    if dividend.yield_history.sample_size:
        lines.append("- 历史股息率分位：%s" % dividend.yield_history.comment)
    lines.append("")

    lines.append("### 近年现金分红")
    lines.append("")
    if dividend.history:
        lines.append("| 年度 | 每10股派息 | 每股现金分红 | 分红次数 |")
        lines.append("|---|---:|---:|---:|")
        for item in dividend.history[:8]:
            lines.append("| %s | %.2f元 | %.3f元 | %d次 |" % (
                item.year,
                item.cash_dividend_per_10,
                item.cash_dividend_per_share,
                item.record_count,
            ))
    else:
        lines.append("暂缺可解析的历史分红记录。")
    lines.append("")

    lines.append("## 技术状态")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append("| MA20 | %s |" % _num(tech.ma20))
    lines.append("| MA60 | %s |" % _num(tech.ma60))
    lines.append("| MA120 | %s |" % _num(tech.ma120))
    lines.append("| 120日高点 | %s |" % _num(tech.high_120))
    lines.append("| 120日低点 | %s |" % _num(tech.low_120))
    lines.append("| 250日高点 | %s |" % _num(tech.high_250))
    lines.append("| 250日低点 | %s |" % _num(tech.low_250))
    lines.append("| 750日高点 | %s |" % _num(tech.high_750))
    lines.append("| 750日低点 | %s |" % _num(tech.low_750))
    lines.append("| 距120日高点回撤 | %s |" % _pct(tech.drawdown_from_120_high))
    lines.append("| 近120日价格分位 | %s |" % _pct(tech.price_percentile_120))
    lines.append("| 近250日价格分位 | %s |" % _pct(tech.price_percentile_250))
    lines.append("| 近750日价格分位 | %s |" % _pct(tech.price_percentile_750))
    lines.append("| RSI14 | %s |" % _num(tech.rsi14))
    lines.append("| MACD | %s / DIF %s / DEA %s / 柱 %s |" % (
        tech.macd_signal or "N/A",
        _num(tech.macd_dif),
        _num(tech.macd_dea),
        _num(tech.macd_hist),
    ))
    lines.append("| 布林带 | 下轨 %s / 中轨 %s / 上轨 %s / 位置 %s |" % (
        _num(tech.boll_lower),
        _num(tech.boll_mid),
        _num(tech.boll_upper),
        _pct(tech.boll_position),
    ))
    lines.append("| 5日/20日量比 | %s |" % _num(tech.volume_ratio_5_20))
    lines.append("")
    lines.append("- 价格位置：%s" % tech.price_position_comment)
    lines.append("- 技术闸门：%s" % tech.technical_comment)
    lines.append("")

    lines.append("## 风险清单")
    lines.append("")
    if result.risk_notes:
        for note in result.risk_notes:
            lines.append("- %s" % note)
    else:
        lines.append("- 暂未发现明显硬性风险，但仍需结合行业、公告和估值继续核对。")
    lines.append("")

    _append_final_conclusion(lines, result)

    lines.append("## 数据可信度分级")
    lines.append("")
    for item in result.reliability.source_reliability:
        lines.append("- %s" % item)
    lines.append("")

    lines.append("## 数据说明")
    lines.append("")
    for note in result.data_notes:
        lines.append("- %s" % note)
    lines.append("")
    return "\n".join(lines)
