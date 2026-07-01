# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stock:
    code: str
    name: str
    market: str
    tencent_code: str


@dataclass
class Quote:
    code: str
    name: str
    price: float
    prev_close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    change: Optional[float] = None
    pct_change: Optional[float] = None
    turnover: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: str = ""


@dataclass
class KLineRow:
    date: str
    open: float
    close: float
    high: float
    low: float
    volume: float


@dataclass
class ValuationRow:
    date: str
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    close_price: Optional[float] = None
    board_code: str = ""
    board_name: str = ""


@dataclass
class ValuationWindowAnalysis:
    label: str
    sample_size: int = 0
    expected_size: int = 0
    is_full_window: bool = False
    pe_percentile: Optional[float] = None
    pb_percentile: Optional[float] = None
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    pb_min: Optional[float] = None
    pb_max: Optional[float] = None


@dataclass
class ValuationHistoryAnalysis:
    sample_size: int = 0
    pe_percentile: Optional[float] = None
    pb_percentile: Optional[float] = None
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    pb_min: Optional[float] = None
    pb_max: Optional[float] = None
    windows: Dict[str, ValuationWindowAnalysis] = field(default_factory=dict)
    primary_metric: str = "PE/PB"
    framework_comment: str = "PE和PB都只作辅助，需要结合股票类型判断。"
    comment: str = "暂缺PE/PB历史估值分位数据。"


@dataclass
class FinancialYear:
    year: str
    revenue: Optional[float] = None
    parent_net_profit: Optional[float] = None
    deduct_net_profit: Optional[float] = None
    revenue_yoy: Optional[float] = None
    net_profit_yoy: Optional[float] = None
    roe: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    debt_ratio: Optional[float] = None
    eps: Optional[float] = None
    operating_cashflow_per_share: Optional[float] = None


@dataclass
class FinancialQuarter:
    period: str
    report_name: str = ""
    revenue: Optional[float] = None
    parent_net_profit: Optional[float] = None
    deduct_net_profit: Optional[float] = None
    revenue_yoy: Optional[float] = None
    net_profit_yoy: Optional[float] = None
    deduct_net_profit_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    operating_cashflow_per_share: Optional[float] = None


@dataclass
class FinancialAnalysis:
    years: List[FinancialYear] = field(default_factory=list)
    latest_dividend_payout_ratio: Optional[float] = None
    latest_cash_dividend_coverage: Optional[float] = None
    quality: object = None
    quality_comment: str = ""
    dividend_comment: str = ""


@dataclass
class QuarterlyTrendAnalysis:
    periods: List[FinancialQuarter] = field(default_factory=list)
    revenue_trend: str = ""
    profit_trend: str = ""
    margin_trend: str = ""
    cashflow_trend: str = ""
    alerts: List[str] = field(default_factory=list)
    comment: str = "暂缺季度财务趋势数据。"


@dataclass
class RiskRadarAnalysis:
    level: str = "低"
    score: int = 0
    items: List[str] = field(default_factory=list)
    comment: str = "暂未识别到明显硬风险。"


@dataclass
class TypedValuationConclusion:
    framework: str = ""
    conclusion: str = ""
    key_metrics: List[str] = field(default_factory=list)


@dataclass
class CategoryAnalysis:
    code: str
    name: str
    framework: str
    dividend_valuation_applicable: bool
    focus_points: List[str] = field(default_factory=list)


@dataclass
class IndustryInfo:
    exchange_board: str = ""
    market: str = ""
    csrc_industry: str = ""
    eastmoney_industry: str = ""
    profile: str = ""


@dataclass
class AnnouncementFact:
    fact_type: str
    fields: Dict[str, str] = field(default_factory=dict)
    summary: str = ""


@dataclass
class Announcement:
    date: str
    title: str
    category: str = ""
    url: str = ""
    tags: List[str] = field(default_factory=list)
    importance: str = "normal"
    detail_summary: str = ""
    detail_error: str = ""
    detail_source: str = ""
    detail_url: str = ""
    facts: List[AnnouncementFact] = field(default_factory=list)


@dataclass
class DecisionAnalysis:
    current_view: str
    valuation_state: str
    action: str
    buy_zone: str
    watch_zone: str
    expensive_zone: str
    risk_control: str
    sell_signals: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    score: int = 0
    left_buy_point: str = ""
    right_buy_point: str = ""
    reduce_point: str = ""


@dataclass
class TradePriceZones:
    left_buy_price: Optional[float] = None
    right_buy_low: Optional[float] = None
    right_buy_high: Optional[float] = None
    watch_low: Optional[float] = None
    watch_high: Optional[float] = None
    chase_risk_price: Optional[float] = None
    reduce_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    left_buy_zone: str = ""
    right_buy_zone: str = ""
    watch_zone: str = ""
    chase_risk_zone: str = ""
    reduce_zone: str = ""
    stop_loss_zone: str = ""
    comment: str = ""


@dataclass
class ReliabilityAssessment:
    confidence: str = "低"
    score: int = 0
    summary: str = "结论置信度较低，需先核对关键数据。"
    core_evidence: List[str] = field(default_factory=list)
    timing_evidence: List[str] = field(default_factory=list)
    weak_evidence: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)
    abnormal_warnings: List[str] = field(default_factory=list)
    freshness_issues: List[str] = field(default_factory=list)
    downgrade_reasons: List[str] = field(default_factory=list)
    consistency_warnings: List[str] = field(default_factory=list)
    source_reliability: List[str] = field(default_factory=list)
    rule_version: str = "reliability-v2-2026-07-01"


@dataclass
class ProfitQualityAnalysis:
    cash_profit_ratio: Optional[float] = None
    deduct_profit_ratio: Optional[float] = None
    gross_margin_trend: str = ""
    net_margin_trend: str = ""
    cashflow_trend: str = ""
    comment: str = "利润质量信息不足。"


@dataclass
class DividendYieldHistoryAnalysis:
    sample_size: int = 0
    current_yield_percentile: Optional[float] = None
    min_yield: Optional[float] = None
    max_yield: Optional[float] = None
    avg_yield: Optional[float] = None
    comment: str = "暂缺历史股息率分位数据。"


@dataclass
class DividendAnalysis:
    per_share: Optional[float]
    current_yield: Optional[float]
    target_prices: Dict[float, float] = field(default_factory=dict)
    bond_risk_prices: Dict[float, float] = field(default_factory=dict)
    history: List[object] = field(default_factory=list)
    yield_history: DividendYieldHistoryAnalysis = field(default_factory=DividendYieldHistoryAnalysis)
    source: str = ""


@dataclass
class ScoreBreakdown:
    valuation: int = 0
    financial_quality: int = 0
    dividend_quality: int = 0
    technical_position: int = 0
    announcement_risk: int = 0
    industry_relative: int = 0
    total: int = 0


@dataclass
class AnnouncementRiskAnalysis:
    level: str = "低"
    score_penalty: int = 0
    items: List[str] = field(default_factory=list)
    comment: str = "近期公告未识别到明显硬风险。"


@dataclass
class IndustryRelativeValuation:
    peer_count: int = 0
    pe_percentile: Optional[float] = None
    pb_percentile: Optional[float] = None
    industry_pe_median: Optional[float] = None
    industry_pb_median: Optional[float] = None
    comment: str = "暂缺行业相对估值数据。"


@dataclass
class NewsItem:
    date: str
    title: str
    url: str = ""
    source: str = ""
    sentiment: str = "neutral"
    tags: List[str] = field(default_factory=list)


@dataclass
class NewsAnalysis:
    items: List[NewsItem] = field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    comment: str = "暂缺近期资讯数据。"


@dataclass
class ResearchReport:
    date: str
    title: str
    org: str = ""
    rating: str = ""
    analyst: str = ""
    url: str = ""
    summary: str = ""


@dataclass
class ResearchAnalysis:
    reports: List[ResearchReport] = field(default_factory=list)
    report_count: int = 0
    latest_rating: str = ""
    rating_summary: str = ""
    comment: str = "暂缺机构研报数据。"


@dataclass
class FundFlowRow:
    date: str
    main_net: Optional[float] = None
    super_large_net: Optional[float] = None
    large_net: Optional[float] = None
    medium_net: Optional[float] = None
    small_net: Optional[float] = None
    main_pct: Optional[float] = None
    close: Optional[float] = None
    pct_change: Optional[float] = None


@dataclass
class FundFlowAnalysis:
    rows: List[FundFlowRow] = field(default_factory=list)
    net_5: Optional[float] = None
    net_10: Optional[float] = None
    net_20: Optional[float] = None
    positive_days_5: int = 0
    positive_days_10: int = 0
    trend: str = "数据不足"
    comment: str = "暂缺资金流数据。"
    source: str = "东方财富资金流"
    fallback_source: str = ""
    buy_large_ratio: Optional[float] = None
    buy_small_ratio: Optional[float] = None
    sell_large_ratio: Optional[float] = None
    sell_small_ratio: Optional[float] = None
    pankou_comment: str = ""


@dataclass
class IndexTrend:
    name: str
    close: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    trend: str = "数据不足"


@dataclass
class MarketEnvironmentAnalysis:
    indices: List[IndexTrend] = field(default_factory=list)
    level: str = "中性"
    comment: str = "暂缺大盘环境数据。"


@dataclass
class SupportObservation:
    level: str = "弱"
    score: int = 0
    signals: List[str] = field(default_factory=list)
    comment: str = "暂未观察到明显隐性承接。"


@dataclass
class QuoteCrossCheckAnalysis:
    primary_source: str = "腾讯"
    check_source: str = ""
    primary_price: Optional[float] = None
    check_price: Optional[float] = None
    price_diff_pct: Optional[float] = None
    timestamp: str = ""
    status: str = "未校验"
    comment: str = "暂未进行行情交叉校验。"
    warnings: List[str] = field(default_factory=list)


@dataclass
class TechnicalAnalysis:
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    ma120: Optional[float] = None
    high_120: Optional[float] = None
    low_120: Optional[float] = None
    high_250: Optional[float] = None
    low_250: Optional[float] = None
    high_750: Optional[float] = None
    low_750: Optional[float] = None
    drawdown_from_120_high: Optional[float] = None
    price_percentile_120: Optional[float] = None
    price_percentile_250: Optional[float] = None
    price_percentile_750: Optional[float] = None
    rsi14: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_signal: str = ""
    boll_mid: Optional[float] = None
    boll_upper: Optional[float] = None
    boll_lower: Optional[float] = None
    boll_position: Optional[float] = None
    volume_ratio_5_20: Optional[float] = None
    technical_comment: str = ""
    left_gate_comment: str = ""
    right_gate_comment: str = ""
    sell_gate_comment: str = ""
    trend_comment: str = ""
    price_position_comment: str = ""


@dataclass
class StockAnalysis:
    stock: Stock
    quote: Quote
    dividend: DividendAnalysis
    financial: FinancialAnalysis
    category: CategoryAnalysis
    industry: IndustryInfo
    technical: TechnicalAnalysis
    valuation_history: ValuationHistoryAnalysis
    valuation_comment: str
    decision: DecisionAnalysis
    risk_notes: List[str]
    data_notes: List[str]
    announcements: List[Announcement] = field(default_factory=list)
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    announcement_risk: AnnouncementRiskAnalysis = field(default_factory=AnnouncementRiskAnalysis)
    industry_valuation: IndustryRelativeValuation = field(default_factory=IndustryRelativeValuation)
    news_analysis: NewsAnalysis = field(default_factory=NewsAnalysis)
    research_analysis: ResearchAnalysis = field(default_factory=ResearchAnalysis)
    quarterly_trend: QuarterlyTrendAnalysis = field(default_factory=QuarterlyTrendAnalysis)
    risk_radar: RiskRadarAnalysis = field(default_factory=RiskRadarAnalysis)
    typed_valuation: TypedValuationConclusion = field(default_factory=TypedValuationConclusion)
    fund_flow: FundFlowAnalysis = field(default_factory=FundFlowAnalysis)
    market_environment: MarketEnvironmentAnalysis = field(default_factory=MarketEnvironmentAnalysis)
    support_observation: SupportObservation = field(default_factory=SupportObservation)
    trade_price_zones: TradePriceZones = field(default_factory=TradePriceZones)
    reliability: ReliabilityAssessment = field(default_factory=ReliabilityAssessment)
    quote_cross_check: QuoteCrossCheckAnalysis = field(default_factory=QuoteCrossCheckAnalysis)
    ai_analysis: str = ""
