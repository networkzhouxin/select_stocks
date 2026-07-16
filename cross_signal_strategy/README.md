# ETF Cross-Signal Strategy Research

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

- `docs/strategy_spec.md`: frozen design before coding.
- `docs/backtest_notes.md`: structured backtest result log.
- `docs/decisions.md`: why each rule was added, rejected, or frozen.
- `docs/failed_experiments.md`: failed ideas and why they should not be repeated casually.
- `docs/research_budget.json`: machine-readable experiment-family status and remaining training-only budget.
- `docs/research_budget.md`: human-readable research map and frozen experiment families; the current open research budget is zero.
- `docs/validation_protocol.md`: frozen out-of-sample validation protocol.
- `smart_trade_joinquant_cross_signal_etf.py`: JoinQuant strategy file, to be created in this folder after the design is approved.
- `smart_trade_ptrade_cross_signal_etf.py`: Guojin PTrade live adapter frozen to the official JoinQuant `cross-v0.3.2` business rules, with failure-open observation-only IOPV logging immediately before actual QDII buy submissions.
- `docs/ptrade_deployment.md`: PTrade schedule, data boundary, order lifecycle, restart recovery, and deployment checklist.
- `training_stability.py`: training-only annual, concentration, exit, holding-period, regime, and doubled-friction diagnostics.
- `friction_diagnostics.py`: cached-signal training replay that isolates commission-rate, minimum-commission, and slippage sensitivity.
- `capital_utilization_diagnostics.py`: training-only occupied-slot, vacant-slot reason, and de-duplicated shadow-candidate diagnostics.
- `backup_fill_candidate.py`: rejected local candidate that tested 50-59 score backup fills without changing the main buy threshold.
- `cmf_diagnostics.py`: observation-only CMF(20) attribution using adjusted T-1 signal frames; it does not alter strategy scores or orders.
- `strong_trend_capacity_diagnostics.py`: training-only strong-trend entry, idle-slot, cash-headroom, and close-path excursion diagnostics; it never changes order targets.
- `gap_execution_diagnostics.py`: training-only attribution of T-day 09:35 execution gaps normalized by frozen T-1 ATR; it does not filter orders.
- `iopv_quality_diagnostics.py`: read-only 2019-2021 minute-data audit for IOPV completeness, 09:35 point coverage, executable premium distributions, and zero-trade IOPV movement; it does not alter signals or orders.
- `docs/iopv_data_quality.md`: evidence and usage boundary for local historical IOPV; the data is diagnostic-only until exact point-in-time availability is proved.
- `smart_trade_joinquant_cross_signal_iopv_probe.py`: temporary no-order JoinQuant capability probe for current-data IOPV, same-day NAV leakage, and 09:34/09:35/09:36 minute boundaries. Run only as documented in its header; it is not a strategy candidate.
- `smart_trade_ptrade_cross_signal_iopv_probe.py`: isolated no-order PTrade live/simulation probe for real-time QDII IOPV, quote timestamps, and ETF publication metadata; it never enters the official strategy.
- `docs/ptrade_iopv_probe.md`: PTrade probe procedure, evidence gate, and strict boundary against threshold tuning or validation inference.
- `us_qdii_premium_diagnostics.py`: consumed observation-only attribution for actual `513100/513500` closed buys using 09:35 price and the point-in-time reference proxy. The candidate gate failed; it does not alter signals or orders.
- `boll_width_diagnostics.py`: observation-only standard BOLL(20,2) BandWidth direction attribution on frozen T-1 signal frames.
- `sequence_diagnostics.py`: observation-only timing attribution for active RSI/KDJ/MACD crosses inside the existing three-day window.
- `ranking_candidate.py`: rejected local A/B experiment that compared official total-score-first ranking with reversal-score-first ranking on identical cached T-1 signals.
- `macd_parameter_candidate.py`: rejected one-shot local A/B experiment that changed only MACD from `(12,26,9)` to `(6,13,5)`; it reduced training return and most quality metrics, so no nearby period search is allowed.
- `multiple_testing_audit.py`: training-only PSR/Bonferroni and Newey-West/HAC audit using the retained experiment-count lower bound; it explicitly refuses to invent canonical DSR or PBO inputs.
- `docs/multiple_testing_audit.md`: frozen multiple-testing evidence, limitations, and the decision to keep `cross-v0.3.2` unchanged.
- `efficiency_ratio_diagnostics.py`: observation-only standard Kaufman ER(10) direction attribution on frozen T-1 signal frames.
- `research_budget.py`: read-only ledger parser and experiment gate that prevents repeated or multi-variant search in exhausted families.
- `portfolio_dependence_diagnostics.py`: rejected observation-only 20-day/0.80 portfolio-correlation attribution; it labels official buys without changing their order or size.
- `market_breadth_diagnostics.py`: rejected observation-only pool MA20/50% breadth attribution for mild-trend entries; it does not filter or resize orders.
- `horizontal_structure_diagnostics.py`: rejected observation-only prior-20-day horizontal support/resistance attribution; levels end on T-2, distances use T-1 ATR, and the failed gate never changes orders.
- `breakout_extension_diagnostics.py`: rejected observation-only controlled-versus-extended breakout attribution using prior-20-day resistance ending T-2; only 2 extended trades existed, so the failed gate never created a candidate or changed orders.
- `share_flow_diagnostics.py`: rejected observation-only five-observation ETF shares-outstanding sign attribution. All 52 eligible domestic buys were covered, but the 2019/2020 relationship reversed in 2021; QDII remains blocked and no order-changing candidate was created.
