# ETF Cross-Signal Strategy Research

## 目录结构

根目录只保留三个正式 Python 入口：

- `smart_trade_joinquant_cross_signal_etf.py`：聚宽正式策略，策略逻辑权威版本。
- `smart_trade_ptrade_cross_signal_etf.py`：国金证券 PTrade 正式实盘适配版。
- `local_training_run.py`：2019-2021 只读训练期本地回放入口。

其他文件按用途归档：

- `local/`：本地数据加载、复权修正、信号适配、撮合与委托规划等正式支撑模块。
- `research/`：训练期诊断、归因、稳定性审计和研究预算工具，不是部署策略。
- `archive/candidates/`：已测试但未进入正式主线的候选策略与候选规则。
- `archive/probes/`：临时平台能力或行情数据探针；探针不下单，也不属于正式策略。
- `tools/`：交易复盘图表等报告工具。
- `docs/`：策略规范、实验记录、验证协议和部署说明。
- `reports/`：已生成的本地可视化报告。
- `../tests/`：所有自动化测试，统一保留在仓库测试目录。

归档只改变文件位置，不改变任何正式策略参数、ETF 池、信号、风控或交易行为。

This folder is an isolated research area for a new ETF cross-signal timing strategy. It must not change the current production multi-factor strategy unless a later validation process explicitly adopts it.

## Purpose

Test whether RSI/MACD/KDJ-style cross signals can identify earlier swing turning points than the current momentum-led multi-factor strategy.

The intended behavior is:

- Buy closer to early strength reversal rather than after a long momentum run.
- Sell when short-term strength fades before a large profit giveback.
- Improve sideways-market entries and exits without overfitting to one historical period.

## Validation Protocol

Use this sequence and do not skip ahead during development:

1. Development/training: 2019-01-01 to 2021-12-31.
2. Weak/sideways validation: 2022-01-01 to 2023-12-31.
3. Recent-market validation: 2024-01-01 to latest available date.
4. Stress validation: 2015-01-01 to 2018-12-31.
5. Early out-of-sample supplement: 2010-01-01 to 2014-12-31.
6. Final summary only after rule freeze: 2015-01-01 to latest available date.

Before running any reserved validation window, follow the frozen protocol in
`docs/validation_protocol.md`. Validation results may be recorded and used for
pass/fail/adoption judgment, but must not be used to tune thresholds, add
indicators, remove ETFs, or search for a new validation-fitting variant.

## Research Rules

- Use JoinQuant as the authority for performance validation.
- If data meaning, backtest mechanics, strategy intent, or evidence is unclear, state the uncertainty and ask the user before proceeding. Do not invent missing facts or treat assumptions as evidence.
- Strictly prevent look-ahead bias: daily signals may use only T-1 and earlier data; T-day 09:35 minute data is for execution price or explicitly documented intraday execution filters only, never for T-1 signal calculation.
- Historical warm-up daily bars before 2019 may be used only as an indicator lookback buffer for 2019-2021 training signals. Warm-up bars must not be counted in performance, parameter tuning, rule selection, or execution pricing.
- Strictly avoid overfitting: do not use validation periods, full-period summaries, or final-period results to tune parameters, choose indicators, change thresholds, or select rules. Validation is allowed only after rules are frozen.
- For local training, only read `G:\financial\history_data\cross_signal_train_2019_2021`.
- For local warm-up, only read `G:\financial\history_data\cross_signal_warmup_2018`.
- For the exhausted share-flow shadow diagnostic, only read `G:\financial\history_data\cross_signal_flow_train_2018_2021`; treat it as immutable and never extend its result with validation-period shares.
- For the blocked QDII underlying-index observation, only read `G:\financial\history_data\cross_signal_underlying_train_2018_2021`; require timezone-aware point-in-time availability and never substitute ETF prices or validation-period index data.
- Do not read `G:\financial\history_data\按年份合并` or other non-training-period market data while designing, tuning, or debugging training-period behavior unless the user explicitly authorizes a validation/final-summary step.
- Treat `G:\financial\history_data\cross_signal_train_2019_2021` as read-only: do not modify, overwrite, clean in place, delete, or generate derived files inside it.
- Never run delete/remove commands against `G:\financial\history_data\cross_signal_train_2019_2021` or any file below it.
- Never run delete/remove commands against `G:\financial\history_data\cross_signal_warmup_2018` or any file below it.
- Any local data loader must assert the approved training root and reject dates outside `2019-01-01` to `2021-12-31`; derived/cache data must be written outside the training data folder.
- After a clear milestone passes its tests/checks, summarize scope, verification, risks, and next step, then create a commit as a rollback point. If analysis shows the milestone is safe to commit, the agent may commit without separate user confirmation, but must report the commit summary, verification, and remaining risks.
- Start with market-standard indicator parameters.
- Do not tune from the full 2015-latest result.
- Record failed experiments, not only successful ones.
- Compare against the current production multi-factor strategy, not only against CSI 300.
- Prefer simple rules unless added complexity improves a clearly defined weakness.

## Initial Indicator Defaults

- RSI: 6, 12, 24.
- MACD: 12, 26, 9.
- KDJ: 9, 3, 3.
- BOLL: 20, 2.
- ATR: 14.
- MA: 5, 10, 20, 60.

These are starting defaults, not optimized parameters.

## Files

- `research/rsi_low_turn_shadow.py`: frozen RSI(6) low-turn event detector; it is an order-free prospective-observation primitive.
- `research/rsi_low_turn_source.py`: fail-closed, point-in-time source contract and exact 09:35 price loader for the blocked prospective observer.
- `research/rsi_low_turn_store.py`: append-only observer state for source hashes, daily evaluations, events, and matured labels.
- `research/rsi_low_turn_outcomes.py`: fixed-horizon, friction-aware observer labels and frozen evidence gate; MFE/MAE remain unavailable without an approved intraday-extrema source.
- `tools/run_rsi_low_turn_shadow.py`: order-free `collect` and state-only `summarize` CLI; it requires an explicitly approved source root and prints `orders_disabled=True`.

- `research/trade_quality_ledger.py`: observation-only unified 2019-2021 trade-quality ledger using actual fills, causal holding-path boundaries, fixed MFE/MAE labels, ATR first-barrier labels, and post-sell returns.
- `docs/trade_quality_ledger.md`: frozen ledger definitions plus the source-index/time-zone evidence required before any QDII underlying-market consistency observation may run.
- `research/underlying_market_data.py`: exact-root, read-only contract for four official underlying-index histories and China-09:35 point-in-time selection.
- `research/underlying_source_acquisition.py`: locked 2018-2021 FRED/CSI acquisition, raw-data normalization, publication-policy gate, and SHA-256 staging manifest; it cannot invent `available_at` or publish a partial formal bundle.
- `tools/fetch_underlying_sources.py`: repeatable raw-source download command. It writes only the separate staging root and leaves the formal point-in-time root absent while any availability policy is unproved.
- `research/underlying_consistency.py`: frozen observation-only positive-versus-non-positive attribution and sample/annual/cross-ETF candidate gate.
- `docs/underlying_market_direction.md`: required directory layout, CSV provenance fields, future-function boundary, frozen gate, and current missing-data blocker.
- `docs/underlying_source_acquisition.md`: source registry, real staging audit, hashes, publication-time evidence, and remaining blockers.

- `docs/strategy_spec.md`: frozen design before coding.
- `docs/backtest_notes.md`: structured backtest result log.
- `docs/decisions.md`: why each rule was added, rejected, or frozen.
- `docs/failed_experiments.md`: failed ideas and why they should not be repeated casually.
- `docs/kdj-rsi-boll-atr-strategy-spec.md`: closed `krba-v0.1-candidate` specification and frozen local-gate result; the independent KDJ/RSI/BOLL/ATR mean-reversion candidate had only 7 closed trades and did not qualify for JoinQuant or PTrade.
- `docs/research_budget.json`: machine-readable experiment-family status and remaining training-only budget.
- `docs/research_budget.md`: human-readable research map and frozen experiment families; the current open research budget is zero.
- `docs/validation_protocol.md`: frozen out-of-sample validation protocol.
- `smart_trade_joinquant_cross_signal_etf.py`: frozen formal JoinQuant strategy and business-logic source of truth.
- `smart_trade_ptrade_cross_signal_etf.py`: Guojin PTrade live adapter for `cross-v0.3.3`; QDII buys keep the failure-open 5% IOPV shadow, while blocked signal sells have an explicit PTrade-only 8% live sell override using fresh executable bid-one/IOPV premium.
- `tools/audit_ptrade_runtime_log.py`: 只读 PTrade 运行日志审计工具；可按交易日检查初始化、状态恢复、09:35 主流程、条件性 10:35 复牌补偿、收盘汇总、委托回报、错误和 QDII IOPV 日志顺序。示例：`python cross_signal_strategy/tools/audit_ptrade_runtime_log.py <日志文件> --date YYYY-MM-DD`。
- `tools/verify_release.py`: 只读正式发布检查工具；校验三个正式入口、语法、版本、构建编号、业务配置指纹、聚宽/PTrade 核心纯函数一致性、PTrade 禁用模块和状态结构。完整发布检查命令：`python cross_signal_strategy/tools/verify_release.py --run-tests`。
- `tools/archive_ptrade_forward_logs.py`: 只读归档未来导出的 PTrade 实盘日志；先校验正式发布身份，再按原始字节 SHA256 只增不改保存，不读取行情或计算收益。
- `docs/prospective_live_log_protocol.md`: 前瞻日志协议开始日、冻结构建/指纹、操作方法，以及“先登记假设、后积累独立确认样本”的防事后挑选规则。
- `docs/ptrade_deployment.md`: PTrade schedule, data boundary, order lifecycle, restart recovery, and deployment checklist.
- `research/training_stability.py`: training-only annual, concentration, exit, holding-period, regime, and doubled-friction diagnostics.
- `research/kdj_rsi_boll_atr_candidate.py`: isolated primitives and order planning for the rejected `krba-v0.1-candidate`; it is research-only and is not imported by either formal strategy.
- `research/krba_training_replay.py`: one-shot frozen 2019-2021 baseline/candidate replay and gate for the closed KRBA family.
- `local/krba_backtester.py`: causal local KRBA replay using the 09:35 minute close under the existing local convention and the 14:50 minute open for the ATR-only check.
- `research/friction_diagnostics.py`: cached-signal training replay that isolates commission-rate, minimum-commission, and slippage sensitivity.
- `research/capital_utilization_diagnostics.py`: training-only occupied-slot, vacant-slot reason, and de-duplicated shadow-candidate diagnostics.
- `archive/candidates/backup_fill_candidate.py`: rejected local candidate that tested 50-59 score backup fills without changing the main buy threshold.
- `research/cmf_diagnostics.py`: observation-only CMF(20) attribution using adjusted T-1 signal frames; it does not alter strategy scores or orders.
- `research/strong_trend_capacity_diagnostics.py`: training-only strong-trend entry, idle-slot, cash-headroom, and close-path excursion diagnostics; it never changes order targets.
- `research/gap_execution_diagnostics.py`: training-only attribution of T-day 09:35 execution gaps normalized by frozen T-1 ATR; it does not filter orders.
- `research/intraday_execution_observation.py`: rejected one-shot ordinary-buy execution counterfactual. It freezes formal 09:35 intent and quantity, tests the single pre-registered passive-limit/fallback path, and stops before portfolio replay because 2020 execution worsened.
- `research/iopv_quality_diagnostics.py`: read-only 2019-2021 minute-data audit for IOPV completeness, 09:35 point coverage, executable premium distributions, and zero-trade IOPV movement; it does not alter signals or orders.
- `docs/iopv_data_quality.md`: evidence and usage boundary for local historical IOPV; the data is diagnostic-only until exact point-in-time availability is proved.
- `archive/probes/smart_trade_joinquant_cross_signal_iopv_probe.py`: temporary no-order JoinQuant capability probe for current-data IOPV, same-day NAV leakage, and 09:34/09:35/09:36 minute boundaries. Run only as documented in its header; it is not a strategy candidate.
- `archive/probes/smart_trade_joinquant_underlying_availability_probe.py`: temporary no-order 2019-2021 JoinQuant probe that discovers the tracked index from official fund metadata, checks T-1 availability at 09:35, and runs a same-day future-data negative control. Its second stage separates API-call success from finite-value usability and cross-checks index registration, an explicit OHLC range, and `attribute_history`. It establishes only platform capability and cannot establish the publisher's original release timestamp.
- `archive/probes/smart_trade_ptrade_cross_signal_iopv_probe.py`: isolated no-order PTrade live/simulation probe for real-time QDII IOPV, quote timestamps, and ETF publication metadata; it never enters the official strategy.
- `docs/ptrade_iopv_probe.md`: PTrade probe procedure, evidence gate, and strict boundary against threshold tuning or validation inference.
- `research/us_qdii_premium_diagnostics.py`: consumed observation-only attribution for actual `513100/513500` closed buys using 09:35 price and the point-in-time reference proxy. The candidate gate failed; it does not alter signals or orders.
- `research/boll_width_diagnostics.py`: observation-only standard BOLL(20,2) BandWidth direction attribution on frozen T-1 signal frames.
- `research/sequence_diagnostics.py`: observation-only timing attribution for active RSI/KDJ/MACD crosses inside the existing three-day window.
- `archive/candidates/ranking_candidate.py`: rejected local A/B experiment that compared official total-score-first ranking with reversal-score-first ranking on identical cached T-1 signals.
- `archive/candidates/macd_parameter_candidate.py`: rejected one-shot local A/B experiment that changed only MACD from `(12,26,9)` to `(6,13,5)`; it reduced training return and most quality metrics, so no nearby period search is allowed.
- `research/multiple_testing_audit.py`: training-only PSR/Bonferroni and Newey-West/HAC audit using the retained experiment-count lower bound; it explicitly refuses to invent canonical DSR or PBO inputs.
- `docs/multiple_testing_audit.md`: frozen multiple-testing evidence, limitations, and the decision to keep `cross-v0.3.2` unchanged.
- `research/efficiency_ratio_diagnostics.py`: observation-only standard Kaufman ER(10) direction attribution on frozen T-1 signal frames.
- `research/research_budget.py`: read-only ledger parser and experiment gate that prevents repeated or multi-variant search in exhausted families.
- `research/portfolio_dependence_diagnostics.py`: rejected observation-only 20-day/0.80 portfolio-correlation attribution; it labels official buys without changing their order or size.
- `research/market_breadth_diagnostics.py`: rejected observation-only pool MA20/50% breadth attribution for mild-trend entries; it does not filter or resize orders.
- `research/horizontal_structure_diagnostics.py`: rejected observation-only prior-20-day horizontal support/resistance attribution; levels end on T-2, distances use T-1 ATR, and the failed gate never changes orders.
- `research/breakout_extension_diagnostics.py`: rejected observation-only controlled-versus-extended breakout attribution using prior-20-day resistance ending T-2; only 2 extended trades existed, so the failed gate never created a candidate or changed orders.
- `research/share_flow_diagnostics.py`: rejected observation-only five-observation ETF shares-outstanding sign attribution. All 52 eligible domestic buys were covered, but the 2019/2020 relationship reversed in 2021; QDII remains blocked and no order-changing candidate was created.
