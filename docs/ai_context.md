# AI Context Handoff

Last updated: 2026-07-01

## Project Goal

This repository contains ETF strategy files plus a newer A-share stock analysis CLI tool.

The current active work is the stock analysis tool under `stock_analyzer/`. Its purpose is to help research A-share stocks by combining:

- real-time quote and K-line data
- financial quality
- dividend sustainability
- historical valuation percentiles
- industry classification and relative valuation
- announcements and selected announcement deep-read
- news and research reports
- quarterly financial trend
- risk radar
- free fund-flow trend
- market environment
- hidden support observation
- technical gates: RSI14, MACD(12,26,9), Bollinger(20,2), and 5/20-day volume ratio
- AI-assisted summary
- reliability assessment, evidence layering, freshness checks, downgrade rules, and consistency warnings

The output is a Markdown report for research and review. It is not investment advice.

## Current CLI Usage

Run by stock code or name:

```powershell
python .\stock_analyzer\analyze_stock.py 600900
python .\stock_analyzer\analyze_stock.py 603179
```

Enable AI summary:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:STOCK_ANALYZER_AI_MODEL="deepseek-chat"
$env:STOCK_ANALYZER_AI_TIMEOUT="120"
python .\stock_analyzer\analyze_stock.py 600900 --ai
```

`STOCK_ANALYZER_AI_TIMEOUT` is optional. Default timeout is 90 seconds.

## Current Data Sources

- Tencent public quote API: real-time quote, PE/PB, market cap, daily K-line.
- Sina quote API: secondary real-time quote cross-check for Tencent quote sanity checks.
- Eastmoney F10: annual financials, quarterly financials, industry classification.
- Eastmoney historical dividend API: cash dividend history.
- Eastmoney valuation API: PE/PB historical valuation percentiles and industry relative valuation.
- Eastmoney announcement API: recent announcements.
- CNInfo fallback: announcement PDF search and text extraction when Eastmoney detail is unavailable.
- Eastmoney stock news API: recent news titles and links.
- Eastmoney research report API: sell-side report titles, organizations, ratings, analysts.
- Eastmoney free fund-flow API: daily main/super-large/large/medium/small order net flow.
- Tencent `s_pk` quote API: fallback order-book ratio when Eastmoney fund-flow fails.

## Current Report Sections

The report currently includes:

- Core conclusion
- Industry classification
- Stock type
- Comprehensive judgment
- Trade price zones
- Type-specific valuation conclusion
- Quarterly financial trend
- Risk radar
- Fund flow and chip temperature
- Market environment
- Hidden support observation
- Score breakdown
- Type-specific checklist
- Announcement risk scan
- Recent important announcements
- Recent news events
- Research report opinions
- Quote overview
- Historical valuation percentile
- Industry relative valuation
- Profit quality
- Financial quality
- Dividend and buy-price zones
- Dividend sustainability
- Technical status
- Risk list
- Final conclusion
- Data reliability grading
- Data source health check
- Data notes

## Technical Gate Rules

Technical indicators are confirmation gates, not standalone buy/sell signals.

- Existing price zones are still based on stock type, MA20/MA60, 120-day high/low, valuation/dividend framework, and price percentiles.
- RSI14 is used to flag short-term overheating or continued weakness.
- MACD(12,26,9) is used as a trend confirmation label: bullish, bearish, or choppy.
- Bollinger(20,2) is used to identify lower-band repair and upper-band overheating.
- 5/20-day volume ratio is used as a simple volume confirmation for right-side entries.
- Reports should show these as "technical gates" inside buy/sell zone explanations, not as exact predictive signals.

## Reliability V2 Rules

The analyzer should avoid giving a confident buy/sell conclusion when the evidence base is weak.

Current guardrails:

- Data freshness checks: quote timestamp, annual financial year, quarterly period, valuation sample date, and fund-flow date.
- Conclusion downgrade rules: stale or missing core data, missing valuation sample, medium/high risk radar, and current price entering chase-risk zones.
- Consistency warnings: low-confidence conclusions with precise price zones, positive action with medium/high risk radar, hidden support conflicting with hard risk, and breakout zones too close to chase-risk zones.
- Abnormal value protection: extreme PE/PB, unusually high dividend yield, extreme annual/quarterly profit growth, and abnormal fund-flow percentages.
- Report and AI context must expose downgrade reasons instead of hiding them inside internal scoring.
- Tencent quote is the primary quote source. Sina quote is only a cross-check source. If Tencent and Sina current prices differ by more than 0.3%, warn; if they differ by more than 1%, downgrade reliability and require manual quote verification.
- Reports include a data source health table near the top. It separates core data from auxiliary and weak auxiliary data, so users can judge whether the conclusion rests on solid evidence or missing/fallback sources.

Reliability rule version is currently `reliability-v2-2026-07-01`.

## Important Design Decisions

### Level-2 and hidden money

The tool does not have real Level-2 data. It cannot reliably identify true hidden money, account-level fund ownership, or real split orders.

The report uses the term "hidden support observation" instead of "dark money" or "main-force split order". This is intentional.

Hidden support is only a rule-based inference from:

- price no longer making obvious new lows
- up-day volume versus down-day volume
- MA5/MA10 recovery
- 5-day main fund flow
- improving fund-flow trend

It cannot prove real accumulation.

### Fund flow usage

Free fund-flow data is used only as a timing aid, not as the core valuation framework.

Do not let same-day fund flow dominate the investment conclusion. Prefer 5/10/20-day summaries.

If Eastmoney fund-flow fails, the CLI now attempts Tencent `s_pk` as a fallback. This fallback provides only four order-book ratios: buy large, buy small, sell large, sell small. It is a "pankou temperature" proxy, not historical main-net-flow data and not Level-2 tick/order-splitting data. Reports must label it as `腾讯盘口比例`.

### Stock type matters

Different stock types use different primary frameworks:

- Dividend stable: dividend yield, payout ratio, cash-flow coverage, bond-yield comparison.
- Growth: PE percentile, profit growth, ROE, cash-flow quality, valuation-growth match.
- Turnaround: PB percentile, loss narrowing, cash-flow turning positive, margin recovery.
- General watch: PE/PB history, industry relative valuation, financial quality, technical position.

### Trade price zones

The tool now produces structured actionable zones:

- left-side low-buy zone
- right-side confirmation zone
- watch zone
- chase-risk zone
- first reduce zone
- stop-loss/risk-control zone

These are generated from stock type, MA20/MA60, 120-day high/low, current price, and dividend-derived prices where applicable.

They are decision aids, not guaranteed buy/sell signals.

### AI usage

AI is optional and only summarizes structured data already collected by the tool.

Prompt rules:

- Do not invent missing facts.
- Do not override rule-engine buy zones.
- Treat news and research reports as auxiliary market-expectation signals.
- Treat announcements and financial data as stronger evidence.
- Do not promise returns.

### Reliability V1

The report now starts with a conclusion summary card:

- current action
- conclusion confidence
- whether it is suitable to buy now
- main buy condition
- main risk-control level
- confidence explanation
- abnormal warnings
- missing data issues

The tool separates evidence into:

- core evidence: quote, financials, valuation, announcements, dividend, industry valuation
- timing evidence: MA20/MA60, fund flow, market environment
- weak evidence: news titles, research ratings, hidden support observation, AI summary

The report also includes a data reliability grading section. This is intended to prevent weak signals such as fund flow, news, research ratings, or AI text from dominating the final conclusion.

## Important User Preferences

- User wants practical buy/sell zones, not only abstract analysis.
- User cares about avoiding buying high and then falling.
- User cares about preserving profit after a position rises.
- User prefers medium/short-term opportunity but also wants fundamental quality checks.
- User wants professional analysis but is aware of overfitting and false precision.
- User prefers conservative wording: "observation", "condition", "risk", "zone" rather than absolute buy/sell instructions.
- ETF strategy code should not be changed unless explicitly requested.

## Current Key Files

- `stock_analyzer/analyze_stock.py`: CLI entrypoint and data orchestration.
- `stock_analyzer/analysis.py`: main analysis engine.
- `stock_analyzer/models.py`: dataclasses.
- `stock_analyzer/report.py`: Markdown report rendering.
- `stock_analyzer/ai.py`: OpenAI-compatible AI integration.
- `stock_analyzer/tencent.py`: Tencent quote and K-line.
- `stock_analyzer/financials.py`: annual and quarterly financial parsing/fetching.
- `stock_analyzer/dividends.py`: dividend history.
- `stock_analyzer/valuation.py`: historical and industry valuation.
- `stock_analyzer/announcements.py`: Eastmoney announcements and CNInfo fallback.
- `stock_analyzer/news.py`: recent news.
- `stock_analyzer/research.py`: research reports.
- `stock_analyzer/fundflow.py`: free fund-flow data.
- `tests/test_stock_analyzer.py`: lightweight test suite.

## Verification Commands

Run these after changes:

```powershell
python .\tests\test_stock_analyzer.py
$files = Get-ChildItem .\stock_analyzer -Filter *.py | ForEach-Object { $_.FullName }; python -m py_compile @files .\tests\test_stock_analyzer.py
git diff --check -- .\stock_analyzer .\tests\test_stock_analyzer.py .\docs\ai_context.md
```

For a live smoke test:

```powershell
python .\stock_analyzer\analyze_stock.py 600900
python .\stock_analyzer\analyze_stock.py 603179
```

## Suggested Next Improvements

Potential future enhancements:

- Add a portfolio/position mode: input cost and position size, output hold/add/reduce plan.
- Add multi-stock comparison reports.
- Add local snapshot history so future reports can compare current view versus past view.
- Add industry-specific indicators where reliable public data exists.
- Improve AI context compression if timeout still happens.
- Add optional report saving by default into a `reports/` folder.
- Calibrate reliability confidence thresholds after observing more stocks.

## Git Handoff Workflow

To continue across computers:

1. After a working session, update this file if major context changed.
2. Run tests.
3. Commit code and docs.
4. On another computer, pull the repo and ask the assistant to read `docs/ai_context.md`.

Recommended commit command:

```powershell
git add stock_analyzer tests docs/ai_context.md
git commit -m "Enhance A-share stock analyzer"
```
