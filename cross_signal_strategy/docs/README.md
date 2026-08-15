# Cross-Signal Research Docs

This folder stores research notes, validation records, and design decisions for the cross-signal ETF strategy.

The three formal entry files live one level up in `cross_signal_strategy/`: the
JoinQuant strategy, the PTrade adapter, and the local training replay entry.
Archived candidates and probes live under `archive/`; support and research
modules live under `local/`, `research/`, and `tools/`.

Key files:

- `上穿下穿ETF策略详细说明.md`: 当前 `cross-v0.3.3` 的中文详细说明，覆盖完整策略逻辑、参数、交易时序、风险管理和正式 ETF 池。
- `strategy_spec.md`: current strategy design and rule specification.
- `backtest_notes.md`: training, candidate, and validation result records.
- `validation_summary.md`: frozen cross-period validation summary and adoption recommendation.
- `decisions.md`: adopted decisions and rationale.
- `failed_experiments.md`: rejected experiments and why not to repeat them casually.
- `failure_year_fragility_atlas.md`: 训练期失败年份脆弱性地图；只汇总台账中明确保留的逐年反例，不授权重新调参。
- `prospective_live_log_protocol.md`: PTrade 前瞻实盘日志的冻结身份、只增不改归档方式和独立确认样本规则；不开放新研究预算。
- `validation_protocol.md`: frozen out-of-sample validation protocol before inspecting reserved-period results.
- `platform_architecture.md`: 中文平台架构、PTrade 实盘适配职责、替代平台比较与当前选型结论。
