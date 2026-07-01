# -*- coding: utf-8 -*-
from datetime import date
from statistics import mean, pstdev

from .decision import build_decision
from .models import (
    AnnouncementRiskAnalysis,
    CategoryAnalysis,
    DividendAnalysis,
    FinancialAnalysis,
    IndustryInfo,
    IndustryRelativeValuation,
    IndexTrend,
    MarketEnvironmentAnalysis,
    ProfitQualityAnalysis,
    QuoteCrossCheckAnalysis,
    QuarterlyTrendAnalysis,
    ReliabilityAssessment,
    RiskRadarAnalysis,
    ScoreBreakdown,
    StockAnalysis,
    SupportObservation,
    TechnicalAnalysis,
    TradePriceZones,
    TypedValuationConclusion,
    ValuationHistoryAnalysis,
)
from .fundflow import analyze_fund_flow
from .news import analyze_news
from .research import analyze_research_reports
from .valuation import analyze_valuation_history


TARGET_YIELDS = (0.035, 0.04, 0.045, 0.05)
RISK_PREMIUMS = (0.02, 0.025, 0.03)
RELIABILITY_RULE_VERSION = "reliability-v2-2026-07-01"


def _round_price(value):
    return round(value, 2)


def _ma(values, n):
    if len(values) < n:
        return None
    return mean(values[-n:])


def _ema_series(values, n):
    if not values:
        return []
    alpha = 2.0 / (n + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def _rsi(values, n=14):
    if len(values) <= n:
        return None
    gains = []
    losses = []
    for prev, cur in zip(values[-n - 1:-1], values[-n:]):
        diff = cur - prev
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _macd(values):
    if len(values) < 35:
        return None, None, None, "数据不足"
    ema12 = _ema_series(values, 12)
    ema26 = _ema_series(values, 26)
    dif = [a - b for a, b in zip(ema12, ema26)]
    dea = _ema_series(dif, 9)
    hist = (dif[-1] - dea[-1]) * 2
    if dif[-1] >= dea[-1] and hist > 0:
        signal = "多头"
    elif dif[-1] < dea[-1] and hist < 0:
        signal = "空头"
    else:
        signal = "震荡"
    return dif[-1], dea[-1], hist, signal


def _bollinger(values, n=20, width=2.0):
    if len(values) < n:
        return None, None, None, None
    window = values[-n:]
    mid = mean(window)
    std = pstdev(window)
    upper = mid + width * std
    lower = mid - width * std
    position = None
    if upper > lower:
        position = (values[-1] - lower) / (upper - lower)
    return mid, upper, lower, position


def _volume_ratio(kline, short=5, long=20):
    volumes = [row.volume for row in kline if row.volume is not None]
    if len(volumes) < long:
        return None
    base = mean(volumes[-long:])
    if base <= 0:
        return None
    return mean(volumes[-short:]) / base


def _avg(values):
    cleaned = [v for v in values if v is not None]
    return mean(cleaned) if cleaned else None


def _range_position(price, lows, highs):
    if not lows or not highs:
        return None
    low = min(lows)
    high = max(highs)
    if high <= low:
        return None
    value = (price - low) / (high - low)
    return max(0.0, min(1.0, value))


def _percentile(values, current):
    cleaned = sorted([v for v in values if v is not None and v > 0])
    if not cleaned or current is None or current <= 0:
        return None
    below = sum(1 for v in cleaned if v < current)
    equal = sum(1 for v in cleaned if v == current)
    return (below + equal * 0.5) / len(cleaned)


def _trend(values, better_higher=True):
    cleaned = [v for v in values if v is not None]
    if len(cleaned) < 2:
        return "数据不足"
    improved = cleaned[0] >= cleaned[-1] if better_higher else cleaned[0] <= cleaned[-1]
    worsened = cleaned[0] < cleaned[-1] if better_higher else cleaned[0] > cleaned[-1]
    if improved:
        return "改善"
    if worsened:
        return "恶化"
    return "持平"


def analyze_profit_quality(rows):
    rows = list(rows or [])
    if not rows:
        return ProfitQualityAnalysis()
    latest = rows[0]
    cash_profit_ratio = None
    deduct_profit_ratio = None
    if (latest.operating_cashflow_per_share is not None
            and latest.eps is not None and latest.eps > 0):
        cash_profit_ratio = latest.operating_cashflow_per_share / latest.eps
    if (latest.deduct_net_profit is not None
            and latest.parent_net_profit is not None and latest.parent_net_profit > 0):
        deduct_profit_ratio = latest.deduct_net_profit / latest.parent_net_profit

    gross_trend = _trend([r.gross_margin for r in rows[:3]])
    net_trend = _trend([r.net_margin for r in rows[:3]])
    cashflow_trend = _trend([r.operating_cashflow_per_share for r in rows[:3]])
    parts = []
    if cash_profit_ratio is not None:
        parts.append("净利润现金含量%.2f倍" % cash_profit_ratio)
    if deduct_profit_ratio is not None:
        parts.append("扣非净利润占比%.2f" % deduct_profit_ratio)
    if gross_trend != "数据不足":
        parts.append("毛利率%s" % gross_trend)
    if cashflow_trend != "数据不足":
        parts.append("经营现金流%s" % cashflow_trend)
    return ProfitQualityAnalysis(
        cash_profit_ratio=round(cash_profit_ratio, 2) if cash_profit_ratio is not None else None,
        deduct_profit_ratio=round(deduct_profit_ratio, 2) if deduct_profit_ratio is not None else None,
        gross_margin_trend=gross_trend,
        net_margin_trend=net_trend,
        cashflow_trend=cashflow_trend,
        comment="；".join(parts) if parts else "利润质量信息不足。",
    )


def analyze_dividend_yield_history(current_yield, dividend_history, valuation_rows):
    history = list(dividend_history or [])
    rows = list(valuation_rows or [])
    if current_yield is None or not history or not rows:
        from .models import DividendYieldHistoryAnalysis
        return DividendYieldHistoryAnalysis()

    dividend_by_year = {}
    for item in history:
        try:
            dividend_by_year[int(item.year)] = item.cash_dividend_per_share
        except (TypeError, ValueError):
            continue
    if not dividend_by_year:
        from .models import DividendYieldHistoryAnalysis
        return DividendYieldHistoryAnalysis()

    years = sorted(dividend_by_year)
    yields = []
    for row in rows:
        if row.close_price is None or row.close_price <= 0:
            continue
        try:
            year = int(str(row.date)[:4])
        except (TypeError, ValueError):
            year = years[-1]
        eligible = [y for y in years if y <= year - 1]
        div_year = eligible[-1] if eligible else years[-1]
        dividend = dividend_by_year.get(div_year)
        if dividend is not None and dividend > 0:
            yields.append(dividend / row.close_price)

    from .models import DividendYieldHistoryAnalysis
    if not yields:
        return DividendYieldHistoryAnalysis()
    pct = _percentile(yields, current_yield)
    avg_yield = sum(yields) / len(yields)
    comment = "股息率历史分位%.1f%%，历史区间%.2f%%-%.2f%%" % (
        pct * 100 if pct is not None else 0,
        min(yields) * 100,
        max(yields) * 100,
    )
    return DividendYieldHistoryAnalysis(
        sample_size=len(yields),
        current_yield_percentile=round(pct, 6) if pct is not None else None,
        min_yield=round(min(yields), 6),
        max_yield=round(max(yields), 6),
        avg_yield=round(avg_yield, 6),
        comment=comment,
    )


def analyze_dividend(price, dividend_per_share, bond_yield, dividend_history=None, dividend_source="",
                     valuation_rows=None):
    history = list(dividend_history or [])
    if dividend_per_share is None or dividend_per_share <= 0 or price <= 0:
        return DividendAnalysis(
            per_share=dividend_per_share,
            current_yield=None,
            history=history,
            source=dividend_source,
        )

    target_prices = {
        target: _round_price(dividend_per_share / target)
        for target in TARGET_YIELDS
    }
    bond_risk_prices = {
        bond_yield + premium: _round_price(dividend_per_share / (bond_yield + premium))
        for premium in RISK_PREMIUMS
    }
    current_yield = round(dividend_per_share / price, 6)
    return DividendAnalysis(
        per_share=dividend_per_share,
        current_yield=current_yield,
        target_prices=target_prices,
        bond_risk_prices=bond_risk_prices,
        history=history,
        yield_history=analyze_dividend_yield_history(current_yield, history, valuation_rows),
        source=dividend_source,
    )


def analyze_financials(financials, dividend_per_share):
    rows = list(financials or [])
    if not rows:
        return FinancialAnalysis(
            years=[],
            quality_comment="暂缺最近年报财务数据，无法做财务质量判断。",
            dividend_comment="暂缺EPS和经营现金流数据，无法判断分红可持续性。",
        )

    latest = rows[0]
    payout = None
    cash_coverage = None
    if dividend_per_share is not None and dividend_per_share > 0:
        if latest.eps is not None and latest.eps > 0:
            payout = dividend_per_share / latest.eps
        if (latest.operating_cashflow_per_share is not None
                and latest.operating_cashflow_per_share > 0):
            cash_coverage = latest.operating_cashflow_per_share / dividend_per_share

    avg_roe = _avg([r.roe for r in rows[:5]])
    avg_net_margin = _avg([r.net_margin for r in rows[:5]])
    latest_profit_yoy = latest.net_profit_yoy
    latest_revenue_yoy = latest.revenue_yoy

    quality_parts = []
    if avg_roe is not None:
        if avg_roe >= 15:
            quality_parts.append("近年ROE优秀")
        elif avg_roe >= 10:
            quality_parts.append("近年ROE尚可")
        else:
            quality_parts.append("近年ROE偏弱")
    if avg_net_margin is not None:
        if avg_net_margin >= 20:
            quality_parts.append("净利率较高")
        elif avg_net_margin < 8:
            quality_parts.append("净利率偏薄")
    if latest_profit_yoy is not None:
        quality_parts.append("最新净利润同比%s%.1f%%" % (
            "增长" if latest_profit_yoy >= 0 else "下降", abs(latest_profit_yoy)))
    if latest_revenue_yoy is not None:
        quality_parts.append("营收同比%s%.1f%%" % (
            "增长" if latest_revenue_yoy >= 0 else "下降", abs(latest_revenue_yoy)))
    quality_comment = "；".join(quality_parts) if quality_parts else "财务指标可用但信息不足。"

    dividend_parts = []
    if payout is not None:
        if payout <= 0.7:
            dividend_parts.append("分红率较健康")
        elif payout <= 0.9:
            dividend_parts.append("分红率偏高但仍可观察")
        else:
            dividend_parts.append("分红率过高，需警惕下调")
    if cash_coverage is not None:
        if cash_coverage >= 1.5:
            dividend_parts.append("经营现金流对分红覆盖较充足")
        elif cash_coverage >= 1.0:
            dividend_parts.append("经营现金流基本覆盖分红")
        else:
            dividend_parts.append("经营现金流覆盖分红不足")
    dividend_comment = "；".join(dividend_parts) if dividend_parts else "分红可持续性信息不足。"

    return FinancialAnalysis(
        years=rows,
        latest_dividend_payout_ratio=round(payout, 6) if payout is not None else None,
        latest_cash_dividend_coverage=round(cash_coverage, 6) if cash_coverage is not None else None,
        quality=analyze_profit_quality(rows),
        quality_comment=quality_comment,
        dividend_comment=dividend_comment,
    )


def analyze_announcement_risk(announcements):
    penalty = 0
    items = []
    for item in announcements or []:
        if item.importance == "risk":
            penalty += 10
            items.append("%s：%s" % (item.date or "N/A", item.title))
        for fact in item.facts:
            if fact.fact_type == "减持":
                ratio_text = fact.fields.get("max_ratio", "").replace("%", "")
                try:
                    ratio = float(ratio_text)
                except ValueError:
                    ratio = 0
                penalty += 8 if ratio < 1 else 15
                items.append("减持上限%s" % (fact.fields.get("max_ratio") or "未明"))
            elif "质押" in fact.fact_type or "诉讼" in fact.fact_type or "处罚" in fact.fact_type:
                penalty += 15
                items.append("%s风险" % fact.fact_type)
    if penalty >= 25:
        level = "高"
    elif penalty >= 10:
        level = "中"
    else:
        level = "低"
    comment = "；".join(items[:4]) if items else "近期公告未识别到明显硬风险。"
    return AnnouncementRiskAnalysis(level=level, score_penalty=penalty, items=items, comment=comment)


def analyze_industry_relative_valuation(quote, rows):
    rows = list(rows or [])
    pe_values = [r.pe_ttm for r in rows if r.pe_ttm is not None and r.pe_ttm > 0]
    pb_values = [r.pb for r in rows if r.pb is not None and r.pb > 0]
    pe_pct = _percentile(pe_values, quote.pe)
    pb_pct = _percentile(pb_values, quote.pb)

    def median(values):
        values = sorted(values)
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    if not rows:
        return IndustryRelativeValuation()
    parts = []
    if pe_pct is not None:
        parts.append("行业PE分位%.1f%%" % (pe_pct * 100))
    if pb_pct is not None:
        parts.append("行业PB分位%.1f%%" % (pb_pct * 100))
    return IndustryRelativeValuation(
        peer_count=len(rows),
        pe_percentile=round(pe_pct, 6) if pe_pct is not None else None,
        pb_percentile=round(pb_pct, 6) if pb_pct is not None else None,
        industry_pe_median=round(median(pe_values), 2) if median(pe_values) is not None else None,
        industry_pb_median=round(median(pb_values), 2) if median(pb_values) is not None else None,
        comment="；".join(parts) if parts else "同行估值样本不足。",
    )


def analyze_quarterly_trend(rows):
    rows = list(rows or [])
    if not rows:
        return QuarterlyTrendAnalysis()
    latest = rows[0]
    alerts = []
    if latest.revenue_yoy is not None and latest.revenue_yoy < 0:
        alerts.append("最近季度营收同比下降")
    if latest.net_profit_yoy is not None and latest.net_profit_yoy < -20:
        alerts.append("最近季度净利润同比大幅下降")
    if latest.revenue_yoy is not None and latest.revenue_yoy > 0 and latest.net_profit_yoy is not None and latest.net_profit_yoy < 0:
        alerts.append("增收不增利")
    if latest.operating_cashflow_per_share is not None and latest.operating_cashflow_per_share < 0:
        alerts.append("最近季度经营现金流为负")
    if latest.net_margin is not None and latest.net_margin < 2:
        alerts.append("最近季度净利率偏薄")

    revenue_trend = _trend([r.revenue_yoy for r in rows[:4]])
    profit_trend = _trend([r.net_profit_yoy for r in rows[:4]])
    margin_trend = _trend([r.net_margin for r in rows[:4]])
    cashflow_trend = _trend([r.operating_cashflow_per_share for r in rows[:4]])
    if alerts:
        comment = "；".join(alerts[:4])
    else:
        parts = []
        if latest.revenue_yoy is not None:
            parts.append("最近季度营收同比%s%.1f%%" % ("增长" if latest.revenue_yoy >= 0 else "下降", abs(latest.revenue_yoy)))
        if latest.net_profit_yoy is not None:
            parts.append("净利润同比%s%.1f%%" % ("增长" if latest.net_profit_yoy >= 0 else "下降", abs(latest.net_profit_yoy)))
        comment = "；".join(parts) if parts else "季度财务数据可用，但趋势信号有限。"
    return QuarterlyTrendAnalysis(
        periods=rows,
        revenue_trend=revenue_trend,
        profit_trend=profit_trend,
        margin_trend=margin_trend,
        cashflow_trend=cashflow_trend,
        alerts=alerts,
        comment=comment,
    )


def analyze_technical(quote, kline):
    closes = [row.close for row in kline]
    if not closes:
        return TechnicalAnalysis(trend_comment="缺少K线数据，无法判断均线趋势。")

    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    ma120 = _ma(closes, 120)
    high_120 = max(row.high for row in kline[-120:])
    low_120 = min(row.low for row in kline[-120:])
    high_250 = max(row.high for row in kline[-250:]) if len(kline) >= 250 else None
    low_250 = min(row.low for row in kline[-250:]) if len(kline) >= 250 else None
    high_750 = max(row.high for row in kline[-750:]) if len(kline) >= 500 else None
    low_750 = min(row.low for row in kline[-750:]) if len(kline) >= 500 else None
    drawdown = (quote.price / high_120 - 1) if high_120 else None
    pct_120 = _range_position(quote.price, [r.low for r in kline[-120:]], [r.high for r in kline[-120:]])
    pct_250 = (
        _range_position(quote.price, [r.low for r in kline[-250:]], [r.high for r in kline[-250:]])
        if len(kline) >= 250 else None
    )
    pct_750 = (
        _range_position(quote.price, [r.low for r in kline[-750:]], [r.high for r in kline[-750:]])
        if len(kline) >= 500 else None
    )
    rsi14 = _rsi(closes, 14)
    macd_dif, macd_dea, macd_hist, macd_signal = _macd(closes)
    boll_mid, boll_upper, boll_lower, boll_position = _bollinger(closes, 20, 2.0)
    volume_ratio = _volume_ratio(kline, 5, 20)

    comments = []
    if ma20 is not None:
        comments.append("站上MA20" if quote.price >= ma20 else "跌破MA20")
    if ma60 is not None:
        comments.append("站上MA60" if quote.price >= ma60 else "跌破MA60")
    if drawdown is not None:
        if drawdown > -0.03:
            comments.append("接近120日高位，追高风险偏高")
        elif drawdown < -0.20:
            comments.append("距120日高点回撤较深，需要确认基本面是否恶化")
    if rsi14 is not None:
        if rsi14 >= 75:
            comments.append("RSI过热，短线不宜追买")
        elif rsi14 <= 35:
            comments.append("RSI偏弱，左侧需等止跌")
    if macd_signal and macd_signal != "数据不足":
        comments.append("MACD%s" % macd_signal)
    if boll_position is not None:
        if boll_position >= 0.95:
            comments.append("接近布林上轨，注意冲高回落")
        elif boll_position <= 0.10:
            comments.append("接近布林下轨，关注止跌修复")

    position_parts = []
    for label, value in (("近120日", pct_120), ("近250日", pct_250), ("近750日", pct_750)):
        if value is None:
            continue
        zone = "低位" if value <= 0.25 else "中低位" if value <= 0.45 else "中位" if value <= 0.65 else "中高位" if value <= 0.85 else "高位"
        position_parts.append("%s价格分位%.1f%%（%s）" % (label, value * 100, zone))

    left_checks = []
    if rsi14 is not None:
        left_checks.append("RSI未继续极弱" if rsi14 >= 35 else "RSI仍偏弱")
    if boll_position is not None:
        left_checks.append("布林下轨附近修复" if boll_position <= 0.35 else "未到布林低位")
    if pct_120 is not None:
        left_checks.append("价格处于120日中低位" if pct_120 <= 0.45 else "价格分位不低")
    left_gate = "；".join(left_checks) if left_checks else "左侧技术闸门数据不足"

    right_checks = []
    if ma20 is not None and ma60 is not None:
        right_checks.append("站回MA20/MA60" if quote.price >= max(ma20, ma60) else "尚未站回MA20/MA60")
    if macd_signal and macd_signal != "数据不足":
        right_checks.append("MACD%s" % macd_signal)
    if volume_ratio is not None:
        right_checks.append("5日量比%.2f" % volume_ratio)
    right_gate = "；".join(right_checks) if right_checks else "右侧技术闸门数据不足"

    sell_checks = []
    if rsi14 is not None and rsi14 >= 75:
        sell_checks.append("RSI过热")
    if boll_position is not None and boll_position >= 0.95:
        sell_checks.append("接近布林上轨")
    if ma20 is not None and quote.price < ma20:
        sell_checks.append("跌破MA20")
    sell_gate = "；".join(sell_checks) if sell_checks else "暂未出现明显技术减仓信号"

    technical_comment = "左侧：%s；右侧：%s；卖出：%s" % (left_gate, right_gate, sell_gate)

    return TechnicalAnalysis(
        ma20=round(ma20, 2) if ma20 is not None else None,
        ma60=round(ma60, 2) if ma60 is not None else None,
        ma120=round(ma120, 2) if ma120 is not None else None,
        high_120=round(high_120, 2),
        low_120=round(low_120, 2),
        high_250=round(high_250, 2) if high_250 is not None else None,
        low_250=round(low_250, 2) if low_250 is not None else None,
        high_750=round(high_750, 2) if high_750 is not None else None,
        low_750=round(low_750, 2) if low_750 is not None else None,
        drawdown_from_120_high=round(drawdown, 6) if drawdown is not None else None,
        price_percentile_120=round(pct_120, 6) if pct_120 is not None else None,
        price_percentile_250=round(pct_250, 6) if pct_250 is not None else None,
        price_percentile_750=round(pct_750, 6) if pct_750 is not None else None,
        rsi14=round(rsi14, 2) if rsi14 is not None else None,
        macd_dif=round(macd_dif, 4) if macd_dif is not None else None,
        macd_dea=round(macd_dea, 4) if macd_dea is not None else None,
        macd_hist=round(macd_hist, 4) if macd_hist is not None else None,
        macd_signal=macd_signal,
        boll_mid=round(boll_mid, 2) if boll_mid is not None else None,
        boll_upper=round(boll_upper, 2) if boll_upper is not None else None,
        boll_lower=round(boll_lower, 2) if boll_lower is not None else None,
        boll_position=round(boll_position, 6) if boll_position is not None else None,
        volume_ratio_5_20=round(volume_ratio, 4) if volume_ratio is not None else None,
        technical_comment=technical_comment,
        left_gate_comment=left_gate,
        right_gate_comment=right_gate,
        sell_gate_comment=sell_gate,
        trend_comment="；".join(comments) if comments else "K线数量不足，趋势信号有限。",
        price_position_comment="；".join(position_parts) if position_parts else "价格分位数据不足。",
    )


def classify_stock(quote, dividend, financial, technical):
    latest = financial.years[0] if financial.years else None
    avg_roe = _avg([r.roe for r in financial.years[:5]])
    avg_net_margin = _avg([r.net_margin for r in financial.years[:5]])
    avg_revenue_yoy = _avg([r.revenue_yoy for r in financial.years[:3]])
    avg_profit_yoy = _avg([r.net_profit_yoy for r in financial.years[:3]])

    latest_loss = bool(latest and (
        (latest.eps is not None and latest.eps <= 0)
        or (latest.parent_net_profit is not None and latest.parent_net_profit <= 0)
    ))
    weak_quality = (
        (avg_roe is not None and avg_roe < 5)
        or (avg_net_margin is not None and avg_net_margin < 5)
    )
    extreme_valuation = (
        (quote.pe is not None and quote.pe > 80)
        or (quote.pb is not None and quote.pb > 4)
    )
    deep_drawdown = (
        technical.drawdown_from_120_high is not None
        and technical.drawdown_from_120_high < -0.20
    )

    if latest_loss or (weak_quality and (deep_drawdown or extreme_valuation)):
        return CategoryAnalysis(
            code="turnaround_watch",
            name="困境/反转观察型",
            framework="红利估值不适用；重点验证亏损收窄、毛利率改善、经营现金流转正和连续季度盈利。",
            dividend_valuation_applicable=False,
            focus_points=[
                "盈利是否连续修复",
                "经营现金流是否转正",
                "毛利率和净利率是否改善",
                "估值指标是否因低利润而失真",
            ],
        )

    healthy_dividend = (
        dividend.current_yield is not None
        and dividend.current_yield >= 0.035
        and financial.latest_dividend_payout_ratio is not None
        and financial.latest_dividend_payout_ratio <= 0.90
        and financial.latest_cash_dividend_coverage is not None
        and financial.latest_cash_dividend_coverage >= 1.0
        and avg_roe is not None
        and avg_roe >= 8
    )
    if healthy_dividend:
        return CategoryAnalysis(
            code="dividend_stable",
            name="红利稳定型",
            framework="适合用股息率、分红率、现金流覆盖和股债收益率比较作为主框架。",
            dividend_valuation_applicable=True,
            focus_points=[
                "可持续每股分红",
                "经营现金流覆盖分红",
                "ROE稳定性",
                "股债收益率差",
            ],
        )

    growth_like = (
        dividend.current_yield is not None
        and dividend.current_yield < 0.03
        and avg_roe is not None
        and avg_roe >= 12
        and avg_revenue_yoy is not None
        and avg_revenue_yoy >= 10
        and avg_profit_yoy is not None
        and avg_profit_yoy >= 10
    )
    if growth_like:
        return CategoryAnalysis(
            code="growth",
            name="成长型",
            framework="红利估值不是主框架；重点看收入利润增速、ROE、利润率趋势和估值成长匹配。",
            dividend_valuation_applicable=False,
            focus_points=[
                "收入和利润增速",
                "ROE和毛利率趋势",
                "估值与成长匹配",
                "现金流质量",
            ],
        )

    return CategoryAnalysis(
        code="general_watch",
        name="综合观察型",
        framework="需要同时参考估值、财务质量、分红和技术位置；暂不宜只用单一公式判断。",
        dividend_valuation_applicable=True,
        focus_points=[
            "财务质量是否稳定",
            "估值是否合理",
            "分红是否可持续",
            "趋势是否企稳",
        ],
    )


def build_valuation_comment(dividend, category=None):
    if category is not None and not category.dividend_valuation_applicable:
        return "当前不适合用股息率买入价做主判断，应先验证基本面是否真正改善。"
    if dividend.current_yield is None:
        return "缺少可持续每股分红，暂不做股息率买入价判断。"
    if dividend.current_yield >= 0.045:
        return "当前股息率较高，但仍需确认分红可持续性和利润质量。"
    if dividend.current_yield >= 0.038:
        return "当前股息率处于可观察区间，适合结合基本面继续跟踪。"
    return "当前股息率安全边际不算厚，适合等待更高股息率或更低价格。"


def build_risk_notes(quote, dividend, technical, financial):
    notes = []
    if dividend.current_yield is None:
        notes.append("缺少分红数据，无法判断高股息吸引力。")
    elif dividend.current_yield > 0.08:
        notes.append("股息率异常高，可能来自股价大跌或一次性高分红，需核对利润和现金流。")
    elif dividend.current_yield < 0.03:
        notes.append("股息率低于3%，如果按红利策略买入，安全边际偏弱。")

    if quote.pe is not None and quote.pe > 30:
        notes.append("市盈率高于30倍，需确认增长能否支撑估值。")
    if quote.pb is not None and quote.pb > 4:
        notes.append("市净率高于4倍，资产估值不便宜。")
    if technical.drawdown_from_120_high is not None and technical.drawdown_from_120_high > -0.03:
        notes.append("股价接近120日高位，分批买入比一次性买入更稳。")

    latest = financial.years[0] if financial.years else None
    if latest is None:
        notes.append("财报深度数据暂缺，营收、利润、现金流和负债需要继续核对年报。")
    else:
        if latest.net_profit_yoy is not None and latest.net_profit_yoy < -20:
            notes.append("最新年报净利润同比下降超过20%，需判断是周期波动还是基本面恶化。")
        if latest.debt_ratio is not None and latest.debt_ratio > 70:
            notes.append("资产负债率高于70%，需要关注财务杠杆和利息压力。")
        if financial.latest_dividend_payout_ratio is not None and financial.latest_dividend_payout_ratio > 0.9:
            notes.append("分红率超过90%，若利润下滑，未来分红下调风险较高。")
        if financial.latest_cash_dividend_coverage is not None and financial.latest_cash_dividend_coverage < 1:
            notes.append("经营现金流无法覆盖每股分红，需警惕高股息不可持续。")
        if (dividend.per_share is not None and dividend.per_share > 0
                and latest.operating_cashflow_per_share is not None
                and latest.operating_cashflow_per_share <= 0):
            notes.append("最新每股经营现金流为负，当前分红更依赖存量现金或融资能力。")
    return notes


def _score_from_percentile(percentile, low_is_good=True):
    if percentile is None:
        return 10
    value = 1 - percentile if low_is_good else percentile
    return max(0, min(20, int(round(value * 20))))


def build_score_breakdown(decision, quote, dividend, financial, technical,
                          valuation_history, announcement_risk, industry_valuation):
    valuation_score = _score_from_percentile(
        valuation_history.pb_percentile if valuation_history.primary_metric == "PB" else valuation_history.pe_percentile
    )
    if valuation_history.primary_metric == "股息率" and dividend.current_yield is not None:
        valuation_score = max(valuation_score, 12 if dividend.current_yield >= 0.035 else 6)

    latest = financial.years[0] if financial.years else None
    financial_score = 10
    if latest is not None:
        financial_score = 0
        if latest.roe is not None and latest.roe >= 10:
            financial_score += 6
        if latest.net_margin is not None and latest.net_margin >= 8:
            financial_score += 5
        if financial.quality and financial.quality.cash_profit_ratio is not None and financial.quality.cash_profit_ratio >= 1:
            financial_score += 5
        if latest.debt_ratio is not None and latest.debt_ratio <= 70:
            financial_score += 4
        financial_score = min(20, financial_score)

    dividend_score = 10
    if dividend.current_yield is not None:
        dividend_score = 0
        if dividend.current_yield >= 0.035:
            dividend_score += 7
        if financial.latest_dividend_payout_ratio is not None and financial.latest_dividend_payout_ratio <= 0.9:
            dividend_score += 6
        if financial.latest_cash_dividend_coverage is not None and financial.latest_cash_dividend_coverage >= 1:
            dividend_score += 7
        dividend_score = min(20, dividend_score)

    technical_score = 10
    if technical.ma60 is not None:
        technical_score = 14 if quote.price >= technical.ma60 else 7
        if technical.price_percentile_250 is not None and technical.price_percentile_250 <= 0.35:
            technical_score += 3
        technical_score = min(20, technical_score)

    announcement_score = max(0, 15 - min(15, announcement_risk.score_penalty))
    industry_score = _score_from_percentile(industry_valuation.pb_percentile, low_is_good=True)
    if industry_valuation.peer_count == 0:
        industry_score = 5

    return ScoreBreakdown(
        valuation=valuation_score,
        financial_quality=financial_score,
        dividend_quality=dividend_score,
        technical_position=technical_score,
        announcement_risk=announcement_score,
        industry_relative=industry_score,
        total=decision.score,
    )


def attach_trade_points(decision, category, dividend, technical):
    if category.code == "dividend_stable":
        fair = dividend.target_prices.get(0.04)
        better = dividend.target_prices.get(0.045)
        decision.left_buy_point = "左侧买点：接近%s且分红现金流覆盖不恶化时分批。" % (
            "%.2f元" % fair if fair else "合理股息率区间"
        )
        decision.right_buy_point = "右侧买点：站回MA20/MA60并保持股息率安全边际。"
        decision.reduce_point = "减仓点：高于偏贵区、跌破MA60或分红覆盖恶化。"
    elif category.code == "turnaround_watch":
        decision.left_buy_point = "左侧买点：PB/价格分位低位且亏损收窄、现金流改善，无硬风险。"
        decision.right_buy_point = "右侧买点：放量站回MA20/MA60，且盈利、现金流、毛利率至少两项改善。"
        decision.reduce_point = "减仓点：反转证据落空、继续亏损、公告风险升级或跌破120日低点。"
    elif category.code == "growth":
        decision.left_buy_point = "左侧买点：估值回到历史中低位且增长质量未恶化。"
        decision.right_buy_point = "右侧买点：站回MA20/MA60，收入利润增速继续兑现。"
        decision.reduce_point = "减仓点：增速失速、现金流恶化或估值分位进入高位。"
    else:
        decision.left_buy_point = "左侧买点：估值与价格分位同时偏低，且无硬风险。"
        decision.right_buy_point = "右侧买点：站稳MA20/MA60并出现基本面确认。"
        decision.reduce_point = "减仓点：估值偏高、趋势破位或公告/财务风险恶化。"
    return decision


def _min_positive(*values):
    cleaned = [v for v in values if v is not None and v > 0]
    return min(cleaned) if cleaned else None


def _max_positive(*values):
    cleaned = [v for v in values if v is not None and v > 0]
    return max(cleaned) if cleaned else None


def _price_text(value):
    return "%.2f元" % value if value is not None else "N/A"


def build_trade_price_zones(category, quote, dividend, technical):
    price = quote.price
    ma20 = technical.ma20
    ma60 = technical.ma60
    low_120 = technical.low_120
    high_120 = technical.high_120

    if category.code == "dividend_stable":
        fair = dividend.target_prices.get(0.04)
        better = dividend.target_prices.get(0.045)
        left = better or low_120 or fair
        right_low = _max_positive(ma20, ma60)
        right_high = fair or price
        chase = dividend.target_prices.get(0.035) or high_120
        stop = _min_positive(ma60, low_120)
        comment = "红利稳定型优先看股息率安全边际，趋势位只用于分批和风控。"
    elif category.code == "growth":
        left = _min_positive(low_120, ma60, price * 0.92)
        right_low = _max_positive(ma20, ma60)
        right_high = high_120
        chase = high_120 or price * 1.12
        stop = _min_positive(ma60, low_120, price * 0.90)
        comment = "成长型价格区间以估值回落和趋势收复为主，需配合利润增速兑现。"
    elif category.code == "turnaround_watch":
        left = _min_positive(low_120, price * 0.90)
        right_low = _max_positive(ma20, ma60)
        right_high = high_120
        chase = high_120 or price * 1.10
        stop = low_120 or _min_positive(ma60, price * 0.88)
        comment = "反转型必须等财务改善或趋势确认，跌破平台低点要降低试错。"
    else:
        left = _min_positive(low_120, ma60, price * 0.93)
        right_low = _max_positive(ma20, ma60)
        right_high = high_120
        chase = high_120 or price * 1.10
        stop = _min_positive(ma60, low_120, price * 0.90)
        comment = "综合观察型用估值、价格分位和趋势共同约束买卖区间。"

    if right_low is not None and right_high is not None and right_high < right_low:
        right_high = right_low * 1.03
    watch_low = _min_positive(left, price)
    watch_high = right_high or high_120 or price
    reduce = _max_positive(chase, high_120)

    left_zone = "左侧低吸区：%s以下，前提是基本面和现金流未恶化；技术闸门：%s。" % (
        _price_text(left), technical.left_gate_comment or "数据不足")
    if right_low is not None and right_high is not None:
        right_zone = "右侧确认区：%s至%s，要求站回关键均线并放量不冲高回落；技术闸门：%s。" % (
            _price_text(right_low), _price_text(right_high), technical.right_gate_comment or "数据不足")
    elif right_low is not None:
        right_zone = "右侧确认区：站回%s附近后再观察；技术闸门：%s。" % (
            _price_text(right_low), technical.right_gate_comment or "数据不足")
    else:
        right_zone = "右侧确认区：关键均线数据不足，暂以趋势企稳为准。"
    watch_zone = "观察区：%s至%s，适合跟踪，不宜一次性重仓。" % (
        _price_text(watch_low), _price_text(watch_high))
    chase_zone = "追高风险区：%s以上，除非业绩继续超预期，否则不宜追买。" % _price_text(chase)
    reduce_zone = "第一减仓区：%s以上或估值分位进入高位时分批兑现；技术减仓信号：%s。" % (
        _price_text(reduce), technical.sell_gate_comment or "暂无")
    stop_zone = "风险止损位：跌破%s且无法快速收回时降低仓位。" % _price_text(stop)

    return TradePriceZones(
        left_buy_price=round(left, 2) if left is not None else None,
        right_buy_low=round(right_low, 2) if right_low is not None else None,
        right_buy_high=round(right_high, 2) if right_high is not None else None,
        watch_low=round(watch_low, 2) if watch_low is not None else None,
        watch_high=round(watch_high, 2) if watch_high is not None else None,
        chase_risk_price=round(chase, 2) if chase is not None else None,
        reduce_price=round(reduce, 2) if reduce is not None else None,
        stop_loss_price=round(stop, 2) if stop is not None else None,
        left_buy_zone=left_zone,
        right_buy_zone=right_zone,
        watch_zone=watch_zone,
        chase_risk_zone=chase_zone,
        reduce_zone=reduce_zone,
        stop_loss_zone=stop_zone,
        comment=comment,
    )


def analyze_quote_cross_check(primary_quote, check_quote):
    if check_quote is None or not check_quote.price:
        return QuoteCrossCheckAnalysis(
            primary_price=primary_quote.price,
            status="未校验",
            comment="未获取到可用的第二行情源。",
        )
    diff_pct = None
    warnings = []
    status = "一致"
    if primary_quote.price and primary_quote.price > 0:
        diff_pct = abs(check_quote.price - primary_quote.price) / primary_quote.price
        if diff_pct > 0.01:
            status = "明显偏离"
            warnings.append("腾讯与新浪当前价偏差超过1%，需核对行情源或停牌/复权/延迟问题")
        elif diff_pct > 0.003:
            status = "轻微偏离"
            warnings.append("腾讯与新浪当前价存在轻微偏差，短线价格需复核")
    comment = "腾讯当前价%s，新浪当前价%s" % (_price_text(primary_quote.price), _price_text(check_quote.price))
    if diff_pct is not None:
        comment += "，偏差%.2f%%" % (diff_pct * 100)
    return QuoteCrossCheckAnalysis(
        check_source="新浪",
        primary_price=primary_quote.price,
        check_price=check_quote.price,
        price_diff_pct=diff_pct,
        timestamp=check_quote.timestamp,
        status=status,
        comment=comment,
        warnings=warnings,
    )


def assess_reliability(quote, financial, dividend, technical, valuation_history,
                       announcements, announcement_risk, news_analysis, research_analysis,
                       fund_flow, support_observation, market_environment, quarterly_trend,
                       industry_valuation):
    score = 50
    issues = []
    warnings = []
    core = []
    timing = []
    weak = []

    if quote.price and quote.price > 0:
        score += 8
        core.append("实时行情可用")
    else:
        score -= 25
        issues.append("实时行情缺失或价格异常")

    if financial.years:
        score += 15
        core.append("年度财务数据可用")
    else:
        score -= 25
        issues.append("年度财务数据缺失")

    if quarterly_trend.periods:
        score += 6
        core.append("季度财务趋势可用")
    else:
        issues.append("季度财务趋势缺失")

    if valuation_history.sample_size >= 250:
        score += 10
        core.append("历史估值样本较充分")
    elif valuation_history.sample_size > 0:
        score -= 4
        issues.append("历史估值样本不足")
    else:
        score -= 12
        issues.append("历史估值数据缺失")

    if technical.ma20 is not None and technical.ma60 is not None:
        score += 8
        timing.append("MA20/MA60趋势位置可用")
    else:
        score -= 8
        issues.append("K线不足，均线和价格分位可信度下降")

    if industry_valuation.peer_count:
        core.append("行业相对估值可用")
    else:
        issues.append("行业相对估值缺失")

    if announcements:
        core.append("近期公告列表可用")
        if any(item.detail_summary for item in announcements):
            score += 4
            core.append("重点公告已读取正文摘要")
    else:
        issues.append("近期公告数据缺失")

    if dividend.per_share is not None and dividend.per_share > 0:
        core.append("分红数据可用")
    else:
        issues.append("分红数据缺失或不可持续分红无法确认")

    if announcement_risk.score_penalty:
        warnings.append("公告风险会降低结论确定性：%s" % announcement_risk.comment)
        score -= min(15, announcement_risk.score_penalty)

    if quote.pe is not None and (quote.pe <= 0 or quote.pe > 120):
        warnings.append("PE为负或极高，不能机械使用PE估值")
        score -= 6
    if dividend.current_yield is not None and dividend.current_yield > 0.08:
        warnings.append("股息率异常高，需核对是否来自股价大跌或一次性分红")
        score -= 6
    if quarterly_trend.alerts:
        warnings.extend(quarterly_trend.alerts[:3])
        score -= min(12, len(quarterly_trend.alerts) * 4)

    if fund_flow.rows or fund_flow.fallback_source:
        timing.append("资金流只作择时辅助：%s" % fund_flow.comment)
    else:
        issues.append("资金流数据缺失")

    if market_environment.indices:
        timing.append("大盘环境只作择时辅助：%s" % market_environment.comment)
    else:
        issues.append("大盘环境数据缺失")

    if support_observation.signals:
        weak.append("隐性承接是规则推断：%s" % support_observation.comment)
    else:
        weak.append("隐性承接暂无强信号")
    if news_analysis.items:
        weak.append("资讯标题只作事件线索")
    if research_analysis.reports:
        weak.append("研报评级只作市场预期参考")
    weak.append("AI只做归纳，不作为事实来源")

    score = max(0, min(100, score))
    if score >= 75 and not warnings:
        confidence = "高"
    elif score >= 50:
        confidence = "中"
    else:
        confidence = "低"
    if confidence == "高":
        summary = "核心数据较完整，结论可用于形成观察和交易计划。"
    elif confidence == "中":
        summary = "核心数据基本可用，但仍有缺口或风险点，适合谨慎参考。"
    else:
        summary = "关键数据缺失或异常较多，结论只适合初步筛查。"

    source_reliability = [
        "高：公告正文、正式财务字段、实时行情",
        "中：分红历史、历史估值、行业相对估值、季度趋势",
        "中低：免费资金流、大盘环境、技术位置",
        "低：资讯标题、研报评级、隐性承接、AI总结",
    ]
    return ReliabilityAssessment(
        confidence=confidence,
        score=score,
        summary=summary,
        core_evidence=core,
        timing_evidence=timing,
        weak_evidence=weak,
        data_quality_issues=issues,
        abnormal_warnings=warnings,
        source_reliability=source_reliability,
    )


def _safe_year(text):
    if not text:
        return None
    digits = "".join(ch for ch in str(text)[:10] if ch.isdigit())
    if len(digits) < 4:
        return None
    try:
        year = int(digits[:4])
    except ValueError:
        return None
    if year < 1990 or year > 2100:
        return None
    return year


def _cap_reliability_confidence(reliability, max_confidence):
    order = {"低": 0, "中": 1, "高": 2}
    if order.get(reliability.confidence, 0) > order.get(max_confidence, 0):
        reliability.confidence = max_confidence


def _refresh_reliability_summary(reliability):
    reliability.score = max(0, min(100, reliability.score))
    if reliability.score >= 75 and not reliability.abnormal_warnings and not reliability.downgrade_reasons:
        reliability.confidence = "高"
    elif reliability.score >= 50:
        reliability.confidence = "中"
    else:
        reliability.confidence = "低"

    if reliability.downgrade_reasons:
        _cap_reliability_confidence(reliability, "中")
    if len(reliability.downgrade_reasons) >= 2 or len(reliability.freshness_issues) >= 3:
        _cap_reliability_confidence(reliability, "低")
    if any("风险雷达高" in item or "年度财务数据缺失" in item for item in reliability.downgrade_reasons):
        _cap_reliability_confidence(reliability, "低")

    if reliability.confidence == "高":
        reliability.summary = "核心数据较完整，结论可用于形成观察和交易计划。"
    elif reliability.confidence == "中":
        reliability.summary = "核心数据基本可用，但存在降级因素，适合谨慎参考。"
    else:
        reliability.summary = "关键数据缺失、过期或异常较多，结论只适合初步筛查。"
    reliability.rule_version = RELIABILITY_RULE_VERSION


def enhance_reliability_v2(reliability, quote, financial, dividend, valuation_rows,
                           fund_flow, quarterly_trend, risk_radar, trade_price_zones,
                           decision, support_observation, quote_cross_check):
    current_year = date.today().year
    freshness = []
    downgrades = []
    consistency = []
    abnormal = []

    quote_year = _safe_year(getattr(quote, "timestamp", ""))
    if quote_year is not None and quote_year < current_year:
        freshness.append("行情时间戳不是当年数据，需要重新抓取实时行情")

    latest_year = _safe_year(financial.years[0].year) if financial.years else None
    if latest_year is None:
        downgrades.append("年度财务数据缺失，估值和质量判断必须降级")
    elif latest_year < current_year - 1:
        freshness.append("年度财务数据不是最近一个完整年度")
        downgrades.append("年度财务数据偏旧，当前判断只能作粗筛")

    latest_quarter_year = _safe_year(quarterly_trend.periods[0].period) if quarterly_trend.periods else None
    if latest_quarter_year is not None and latest_quarter_year < current_year:
        freshness.append("季度财务数据未覆盖当年，近期经营判断需核对公告")

    valuation_year = _safe_year(valuation_rows[0].date) if valuation_rows else None
    if valuation_year is None:
        downgrades.append("历史估值样本缺失，价格区间只能参考技术和股息框架")
    elif valuation_year < current_year - 1:
        freshness.append("历史估值样本不是近期数据，分位数参考价值下降")

    flow_year = _safe_year(fund_flow.rows[0].date) if fund_flow.rows else None
    if fund_flow.fallback_source:
        pass
    elif flow_year is not None and flow_year < current_year:
        freshness.append("资金流数据不是当年数据，短线判断需重新抓取")

    if quote.pe is not None and (quote.pe <= 0 or quote.pe > 120):
        abnormal.append("PE为负或超过120倍，不能机械使用PE估值")
    if quote.pb is not None and quote.pb > 15:
        abnormal.append("PB超过15倍，需确认是否为轻资产高估值或数据异常")
    if dividend.current_yield is not None and dividend.current_yield > 0.08:
        abnormal.append("股息率超过8%，需核对是否来自股价大跌、一次性分红或分红不可持续")

    latest = financial.years[0] if financial.years else None
    if latest and latest.net_profit_yoy is not None and (latest.net_profit_yoy > 200 or latest.net_profit_yoy < -80):
        abnormal.append("年度净利润同比波动极端，需区分低基数、资产处置或主营变化")
    latest_quarter = quarterly_trend.periods[0] if quarterly_trend.periods else None
    if latest_quarter and latest_quarter.net_profit_yoy is not None and (
            latest_quarter.net_profit_yoy > 300 or latest_quarter.net_profit_yoy < -80):
        abnormal.append("季度净利润同比波动极端，不能直接年化外推")
    if fund_flow.rows and fund_flow.rows[0].main_pct is not None and abs(fund_flow.rows[0].main_pct) > 15:
        abnormal.append("单日主力净流占比异常，需防止免费资金流口径失真")

    if quote_cross_check.warnings:
        abnormal.extend(quote_cross_check.warnings)
        if quote_cross_check.status == "明显偏离":
            downgrades.append("腾讯与新浪行情明显偏离，当前价相关结论必须复核")

    if risk_radar.level == "高":
        downgrades.append("风险雷达高风险，最终结论必须降级")
    elif risk_radar.level == "中":
        downgrades.append("风险雷达中等风险，买点需附带更严格条件")

    if trade_price_zones.chase_risk_price is not None and quote.price >= trade_price_zones.chase_risk_price:
        downgrades.append("当前价进入追高风险区，不应给出积极买入结论")

    if reliability.confidence == "低" and (
            trade_price_zones.left_buy_price is not None or trade_price_zones.right_buy_low is not None):
        consistency.append("低置信度下的买卖价格只能作为条件区间，不能当作确定买点")
    if risk_radar.level in ("中", "高") and decision.action in ("可关注", "可分批关注", "小仓位试错"):
        consistency.append("风险雷达与当前操作偏积极，需先核对风险项再执行")
    if support_observation.level == "强" and risk_radar.level in ("中", "高"):
        consistency.append("隐性承接与风险雷达冲突，应优先服从硬风险")
    if trade_price_zones.chase_risk_price is not None and trade_price_zones.right_buy_high is not None:
        if trade_price_zones.right_buy_high >= trade_price_zones.chase_risk_price:
            consistency.append("右侧确认区接近追高风险区，突破买入必须等待回踩或放弃追买")

    reliability.freshness_issues.extend(item for item in freshness if item not in reliability.freshness_issues)
    reliability.downgrade_reasons.extend(item for item in downgrades if item not in reliability.downgrade_reasons)
    reliability.consistency_warnings.extend(item for item in consistency if item not in reliability.consistency_warnings)
    reliability.abnormal_warnings.extend(item for item in abnormal if item not in reliability.abnormal_warnings)

    reliability.score -= min(20, len(freshness) * 5)
    reliability.score -= min(30, len(downgrades) * 8)
    reliability.score -= min(12, len(consistency) * 3)
    reliability.score -= min(15, len(abnormal) * 4)
    _refresh_reliability_summary(reliability)
    return reliability


def analyze_risk_radar(quote, financial, technical, announcements, announcement_risk,
                       quarterly_trend, valuation_history):
    score = 0
    items = []
    latest = financial.years[0] if financial.years else None
    if latest is None:
        score += 5
        items.append("缺少最近年度财务数据")
    else:
        if latest.net_profit_yoy is not None and latest.net_profit_yoy < -20:
            score += 12
            items.append("年度净利润同比下降超过20%")
        if latest.debt_ratio is not None and latest.debt_ratio > 70:
            score += 10
            items.append("资产负债率高于70%")
        if financial.quality and financial.quality.cash_profit_ratio is not None and financial.quality.cash_profit_ratio < 0.6:
            score += 12
            items.append("经营现金流明显弱于净利润")
        if latest.operating_cashflow_per_share is not None and latest.operating_cashflow_per_share < 0:
            score += 12
            items.append("年度每股经营现金流为负")
    for alert in quarterly_trend.alerts:
        score += 8
        items.append(alert)
    if announcement_risk.score_penalty:
        score += min(25, announcement_risk.score_penalty)
        items.extend(announcement_risk.items[:3])
    if technical.drawdown_from_120_high is not None and technical.drawdown_from_120_high < -0.30:
        score += 8
        items.append("股价较120日高点回撤超过30%")
    if quote.pe is not None and quote.pe > 60 and (latest is None or latest.net_profit_yoy is None or latest.net_profit_yoy < 10):
        score += 10
        items.append("高PE但增长证据不足")
    if valuation_history.pe_percentile is not None and valuation_history.pe_percentile > 0.85:
        score += 8
        items.append("PE处于自身历史高位")

    if score >= 35:
        level = "高"
    elif score >= 15:
        level = "中"
    else:
        level = "低"
    comment = "；".join(items[:6]) if items else "暂未识别到明显硬风险。"
    return RiskRadarAnalysis(level=level, score=score, items=items, comment=comment)


def build_typed_valuation_conclusion(category, quote, dividend, financial, valuation_history,
                                     industry_valuation, quarterly_trend):
    metrics = []
    if dividend.current_yield is not None:
        metrics.append("股息率%.2f%%" % (dividend.current_yield * 100))
    if quote.pe is not None:
        metrics.append("PE %.2f" % quote.pe)
    if quote.pb is not None:
        metrics.append("PB %.2f" % quote.pb)
    if valuation_history.sample_size:
        metrics.append(valuation_history.comment)
    if industry_valuation.peer_count:
        metrics.append(industry_valuation.comment)
    if quarterly_trend.comment:
        metrics.append("季度趋势：%s" % quarterly_trend.comment)

    if category.code == "dividend_stable":
        framework = "红利稳定型：股息率、分红率、现金流覆盖、股债差优先。"
        if dividend.current_yield is not None and dividend.current_yield >= 0.04:
            conclusion = "股息率进入较有吸引力区间，但仍需确认分红和现金流覆盖稳定。"
        else:
            conclusion = "红利逻辑成立，但当前更适合等更高股息率或更清晰的趋势修复。"
    elif category.code == "turnaround_watch":
        framework = "困境/反转型：PB分位、亏损收窄、现金流转正和季度改善优先。"
        if quarterly_trend.alerts:
            conclusion = "反转证据仍不充分，需先看到季度利润、现金流或毛利率连续改善。"
        else:
            conclusion = "可观察反转线索，但买入应等待财务改善和价格企稳同时出现。"
    elif category.code == "growth":
        framework = "成长型：PE分位、利润增速、ROE和现金流质量优先。"
        conclusion = "应重点比较估值分位与利润增速是否匹配，避免高估值下增长失速。"
    else:
        framework = "综合观察型：PE/PB历史分位、行业分位、财务质量和技术位置交叉验证。"
        conclusion = "当前没有单一估值锚，需同时满足估值不过高、财务未恶化、趋势不破位。"
    return TypedValuationConclusion(
        framework=framework,
        conclusion=conclusion,
        key_metrics=metrics[:8],
    )


def analyze_market_environment(market_klines):
    items = []
    strong = 0
    weak = 0
    for name, rows in (market_klines or {}).items():
        closes = [row.close for row in rows or []]
        if not closes:
            items.append(IndexTrend(name=name))
            continue
        ma20 = _ma(closes, 20) or _ma(closes, min(5, len(closes)))
        ma60 = _ma(closes, 60) or _ma(closes, min(5, len(closes)))
        close = closes[-1]
        if ma20 is not None and ma60 is not None and close >= ma20 >= ma60:
            trend = "偏强"
            strong += 1
        elif ma20 is not None and ma60 is not None and close < ma20 < ma60:
            trend = "偏弱"
            weak += 1
        else:
            trend = "震荡"
        items.append(IndexTrend(
            name=name,
            close=round(close, 2),
            ma20=round(ma20, 2) if ma20 is not None else None,
            ma60=round(ma60, 2) if ma60 is not None else None,
            trend=trend,
        ))
    if not items:
        return MarketEnvironmentAnalysis()
    if weak >= max(1, len(items) // 2 + 1):
        level = "逆风"
        comment = "主要指数偏弱，短线买点需要更严格。"
    elif strong >= max(1, len(items) // 2 + 1):
        level = "顺风"
        comment = "主要指数偏强，右侧信号更容易延续。"
    else:
        level = "中性"
        comment = "大盘环境分化或震荡，适合降低追高冲动。"
    return MarketEnvironmentAnalysis(indices=items, level=level, comment=comment)


def analyze_support_observation(kline, fund_flow):
    rows = list(kline or [])
    signals = []
    score = 0
    if len(rows) >= 5:
        recent = rows[-5:]
        lows = [row.low for row in recent]
        if lows[-1] >= min(lows[:-1]):
            score += 2
            signals.append("近5日未继续创出明显新低")
        up_vol = sum(row.volume for row in recent if row.close >= row.open)
        down_vol = sum(row.volume for row in recent if row.close < row.open)
        if up_vol >= down_vol and up_vol > 0:
            score += 2
            signals.append("上涨日成交量不弱于下跌日")
        closes = [row.close for row in rows]
        ma5 = _ma(closes, 5)
        ma10 = _ma(closes, 10)
        if ma5 is not None and closes[-1] >= ma5:
            score += 1
            signals.append("价格站回MA5")
        if ma10 is not None and closes[-1] >= ma10:
            score += 1
            signals.append("价格站回MA10")
    if fund_flow.rows:
        if fund_flow.net_5 is not None and fund_flow.net_5 > 0:
            score += 2
            signals.append("近5日主力资金净流入")
        elif fund_flow.trend == "流入改善":
            score += 1
            signals.append("主力流出收窄或流入改善")
        if fund_flow.positive_days_5 >= 3:
            score += 1
            signals.append("近5日资金流入天数较多")
    if score >= 6:
        level = "强"
    elif score >= 3:
        level = "中"
    else:
        level = "弱"
    comment = "；".join(signals[:6]) if signals else "暂未观察到明显隐性承接。"
    return SupportObservation(level=level, score=score, signals=signals, comment=comment)


def analyze_stock(stock, quote, kline, dividend_per_share=None, financials=None,
                  bond_yield=0.0175, dividend_history=None, dividend_source="",
                  industry=None, announcements=None, valuation_rows=None,
                  industry_valuation_rows=None, news_items=None,
                  research_reports=None, quarterly_financials=None,
                  fund_flow_rows=None, market_klines=None, fund_flow_fallback=None,
                  quote_check=None):
    # Backward compatibility: old positional call was analyze_stock(..., dividend, bond_yield).
    if isinstance(financials, (int, float)) and bond_yield == 0.0175:
        bond_yield = financials
        financials = None

    if stock.name == stock.code and quote.name:
        stock.name = quote.name

    industry = industry or IndustryInfo()
    dividend = analyze_dividend(
        quote.price,
        dividend_per_share,
        bond_yield,
        dividend_history,
        dividend_source,
        valuation_rows,
    )
    financial = analyze_financials(financials, dividend_per_share)
    technical = analyze_technical(quote, kline)
    category = classify_stock(quote, dividend, financial, technical)
    valuation_history = analyze_valuation_history(valuation_rows, quote.pe, quote.pb, category.code)
    valuation_comment = build_valuation_comment(dividend, category)
    announcement_risk = analyze_announcement_risk(announcements)
    industry_valuation = analyze_industry_relative_valuation(quote, industry_valuation_rows)
    news_analysis = analyze_news(news_items)
    research_analysis = analyze_research_reports(research_reports)
    quarterly_trend = analyze_quarterly_trend(quarterly_financials)
    fund_flow = analyze_fund_flow(fund_flow_rows)
    if not fund_flow.rows and fund_flow_fallback is not None:
        fund_flow = fund_flow_fallback
    quote_cross_check = analyze_quote_cross_check(quote, quote_check)
    market_environment = analyze_market_environment(market_klines)
    support_observation = analyze_support_observation(kline, fund_flow)
    trade_price_zones = build_trade_price_zones(category, quote, dividend, technical)
    risk_radar = analyze_risk_radar(
        quote, financial, technical, announcements, announcement_risk, quarterly_trend, valuation_history)
    typed_valuation = build_typed_valuation_conclusion(
        category, quote, dividend, financial, valuation_history, industry_valuation, quarterly_trend)
    reliability = assess_reliability(
        quote, financial, dividend, technical, valuation_history, announcements,
        announcement_risk, news_analysis, research_analysis, fund_flow,
        support_observation, market_environment, quarterly_trend, industry_valuation)
    decision = build_decision(quote, dividend, financial, category, technical, announcements, valuation_history)
    decision = attach_trade_points(decision, category, dividend, technical)
    if trade_price_zones.left_buy_zone:
        decision.left_buy_point = "左侧买点：" + trade_price_zones.left_buy_zone.replace("左侧低吸区：", "")
    if trade_price_zones.right_buy_zone:
        decision.right_buy_point = "右侧买点：" + trade_price_zones.right_buy_zone.replace("右侧确认区：", "")
    if trade_price_zones.reduce_zone:
        decision.reduce_point = "减仓点：" + trade_price_zones.reduce_zone.replace("第一减仓区：", "")
    reliability = enhance_reliability_v2(
        reliability, quote, financial, dividend, valuation_rows, fund_flow,
        quarterly_trend, risk_radar, trade_price_zones, decision, support_observation,
        quote_cross_check)
    score_breakdown = build_score_breakdown(
        decision, quote, dividend, financial, technical, valuation_history, announcement_risk, industry_valuation)
    risk_notes = build_risk_notes(quote, dividend, technical, financial)
    if not category.dividend_valuation_applicable:
        risk_notes.insert(0, "%s：%s" % (category.name, category.framework))
    data_notes = [
        "实时行情来自腾讯公开行情接口。",
        "财务数据来自东方财富公开F10接口，字段可能随接口变化而调整。",
        "PE/PB历史估值分位来自东方财富估值分析接口，按近3年、5年、10年窗口计算；当前最多抓取约3000个交易日。",
        "近期资讯来自东方财富公开个股资讯接口，按标题做简单事件标签，需结合原文和公告核对。",
        "机构研报来自东方财富公开研报接口，属于第三方观点，不直接作为买卖依据。",
        "季度财务趋势来自东方财富公开F10财务接口，用于观察近期变化，需与正式财报核对。",
        "风险雷达为规则扫描结果，只提示需要核对的风险点，不等同于最终投资结论。",
        "资金流优先来自东方财富公开资金流接口；若失败，可用腾讯盘口比例兜底。两者都不等同于Level-2逐笔拆单或真实账户资金。",
        "隐性承接观察基于资金流、价格和成交量的规则推断，不能证明存在暗盘资金。",
        "分红数据优先来自东方财富历史分红接口；接口失败时才使用本地配置兜底。",
        "本工具输出用于研究和复盘，不构成投资建议。",
    ]

    return StockAnalysis(
        stock=stock,
        quote=quote,
        dividend=dividend,
        financial=financial,
        category=category,
        industry=industry,
        announcements=list(announcements or []),
        technical=technical,
        valuation_history=valuation_history,
        valuation_comment=valuation_comment,
        decision=decision,
        risk_notes=risk_notes,
        data_notes=data_notes,
        score_breakdown=score_breakdown,
        announcement_risk=announcement_risk,
        industry_valuation=industry_valuation,
        news_analysis=news_analysis,
        research_analysis=research_analysis,
        quarterly_trend=quarterly_trend,
        risk_radar=risk_radar,
        typed_valuation=typed_valuation,
        fund_flow=fund_flow,
        market_environment=market_environment,
        support_observation=support_observation,
        trade_price_zones=trade_price_zones,
        reliability=reliability,
        quote_cross_check=quote_cross_check,
    )
