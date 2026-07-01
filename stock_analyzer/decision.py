# -*- coding: utf-8 -*-
from .models import DecisionAnalysis


def _price(value):
    if value is None:
        return "N/A"
    return "%.2f元" % value


def _has_risk_announcement(announcements):
    return any(item.importance == "risk" for item in announcements or [])


def _target_price(dividend, target):
    return dividend.target_prices.get(target)


def build_decision(quote, dividend, financial, category, technical, announcements=None, valuation_history=None):
    announcements = list(announcements or [])
    risk_announcement = _has_risk_announcement(announcements)
    latest = financial.years[0] if financial.years else None

    reasons = []
    sell_signals = []
    if risk_announcement:
        reasons.append("近期公告存在风险标签，买入前应先核对公告原文。")
        sell_signals.append("出现减持、处罚、诉讼、问询函等风险公告且基本面无法解释时，降低仓位。")
    if technical.ma60 is not None:
        sell_signals.append("跌破MA60且无法快速收回时，先按风险位处理。")
    if technical.ma20 is not None:
        sell_signals.append("短线跌破MA20并伴随财务或公告恶化时，减少试错。")
    if technical.price_position_comment:
        reasons.append(technical.price_position_comment)
    if valuation_history is not None and valuation_history.sample_size:
        reasons.append(valuation_history.comment)

    if category.code == "dividend_stable":
        fair_price = _target_price(dividend, 0.04)
        attractive_price = _target_price(dividend, 0.045)
        expensive_price = _target_price(dividend, 0.035)
        current_yield = dividend.current_yield

        if current_yield is None:
            valuation_state = "缺少稳定分红数据，暂无法用股息率估值。"
            action = "谨慎观察"
            score = 45
        elif risk_announcement:
            valuation_state = "红利逻辑可用，但公告风险降低当前确定性。"
            action = "暂缓买入，先核对公告"
            score = 55
        elif current_yield >= 0.045:
            valuation_state = "股息率进入较优区间，安全边际相对更厚。"
            action = "可分批关注"
            score = 78
        elif current_yield >= 0.04:
            valuation_state = "股息率处于合理区间，具备观察和分批价值。"
            action = "估值合理，可关注"
            score = 70
        elif current_yield >= 0.035:
            valuation_state = "估值不算贵，但安全边际不厚。"
            action = "等待回调"
            score = 60
        else:
            valuation_state = "当前股息率偏低，红利买入吸引力不足。"
            action = "不追高"
            score = 45

        if financial.latest_cash_dividend_coverage is not None and financial.latest_cash_dividend_coverage >= 1:
            reasons.append("经营现金流可以覆盖当前分红。")
        if financial.latest_dividend_payout_ratio is not None and financial.latest_dividend_payout_ratio > 0.9:
            reasons.append("分红率偏高，估值时需要提高安全边际。")
        if technical.drawdown_from_120_high is not None and technical.drawdown_from_120_high > -0.03:
            reasons.append("股价接近120日高位，不适合一次性重仓追入。")

        return DecisionAnalysis(
            current_view="以红利稳定性和股息率安全边际为主。",
            valuation_state=valuation_state,
            action=action,
            buy_zone="合理买入区：%s以下；较优买入区：%s以下。" % (
                _price(fair_price),
                _price(attractive_price),
            ),
            watch_zone="观察区：%s至%s，适合小仓位跟踪或等待回调。" % (
                _price(fair_price),
                _price(expensive_price),
            ),
            expensive_zone="偏贵区：高于%s时，除非分红继续上调，否则不宜追高。" % _price(expensive_price),
            risk_control="风险位：跌破MA60或分红现金流覆盖恶化时减仓观察。",
            sell_signals=sell_signals,
            reasons=reasons,
            score=score,
        )

    if category.code == "turnaround_watch":
        improvement_count = 0
        if latest is not None and latest.parent_net_profit is not None and latest.parent_net_profit > 0:
            improvement_count += 1
            reasons.append("归母净利润已转正。")
        if latest is not None and latest.operating_cashflow_per_share is not None and latest.operating_cashflow_per_share > 0:
            improvement_count += 1
            reasons.append("经营现金流已转正。")
        if latest is not None and latest.gross_margin is not None and latest.gross_margin >= 8:
            improvement_count += 1
            reasons.append("毛利率已有修复迹象。")
        if technical.ma60 is not None and quote.price >= technical.ma60:
            improvement_count += 1
            reasons.append("价格站回MA60，技术趋势初步修复。")

        if risk_announcement:
            action = "暂不买入"
            valuation_state = "公告风险叠加反转证据不足，当前赔率不清晰。"
            score = 30
        elif improvement_count >= 3:
            action = "小仓位试错"
            valuation_state = "反转证据开始出现，但仍需连续验证。"
            score = 62
        else:
            action = "谨慎观察"
            valuation_state = "反转证据不足，PE和股息率参考意义较弱。"
            score = 38

        return DecisionAnalysis(
            current_view="以反转证据为主，不用股息率作为主估值框架。",
            valuation_state=valuation_state,
            action=action,
            buy_zone="条件买点：放量站回MA20/MA60，且盈利、现金流、毛利率至少两项改善。",
            watch_zone="观察区：现价至MA60之间只适合跟踪，不适合按便宜逻辑加仓。",
            expensive_zone="偏热区：短期急涨但财务未修复时，优先兑现而不是追买。",
            risk_control="风险位：跌破120日低点、继续亏损或出现减持/处罚/诉讼公告时退出观察。",
            sell_signals=sell_signals,
            reasons=reasons,
            score=score,
        )

    if category.code == "growth":
        pe_state = "PE缺失"
        if quote.pe is not None:
            pe_state = "PE %.2f倍" % quote.pe
        if risk_announcement:
            action = "暂缓买入"
            score = 45
        elif technical.ma20 is not None and quote.price >= technical.ma20:
            action = "回调后关注"
            score = 62
        else:
            action = "等待企稳"
            score = 52

        return DecisionAnalysis(
            current_view="以成长质量和估值成长匹配为主。",
            valuation_state="当前%s，需要结合利润增速、ROE和现金流判断是否匹配。" % pe_state,
            action=action,
            buy_zone="条件买点：利润增速保持、现金流不恶化，并在MA20/MA60附近企稳。",
            watch_zone="观察区：业绩增长能解释估值时，可分批跟踪。",
            expensive_zone="偏贵区：估值明显高于增长速度，或放量冲高后业绩没有同步改善。",
            risk_control="风险位：利润增速失速、跌破MA60或公告出现经营风险时减仓。",
            sell_signals=sell_signals,
            reasons=reasons,
            score=score,
        )

    action = "综合观察"
    score = 55
    if risk_announcement:
        action = "暂缓买入"
        score = 42
    elif technical.ma60 is not None and quote.price >= technical.ma60:
        action = "可关注"
        score = 60

    return DecisionAnalysis(
        current_view="同时参考估值、财务质量、分红质量和技术位置。",
        valuation_state="当前没有单一主导框架，需用多指标交叉验证。",
        action=action,
        buy_zone="条件买点：估值不高、财务稳定，并站稳MA20/MA60。",
        watch_zone="观察区：基本面没有恶化但技术趋势仍未确认时，先跟踪。",
        expensive_zone="偏贵区：PE/PB偏高且股价接近120日高位时，不追。",
        risk_control="风险位：财务恶化、公告风险或跌破MA60时降低仓位。",
        sell_signals=sell_signals,
        reasons=reasons,
        score=score,
    )
