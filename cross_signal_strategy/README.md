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
- `docs/validation_protocol.md`: frozen out-of-sample validation protocol.
- `smart_trade_joinquant_cross_signal_etf.py`: JoinQuant strategy file, to be created in this folder after the design is approved.
- `training_stability.py`: training-only annual, concentration, exit, holding-period, regime, and doubled-friction diagnostics.
- `friction_diagnostics.py`: cached-signal training replay that isolates commission-rate, minimum-commission, and slippage sensitivity.
- `capital_utilization_diagnostics.py`: training-only occupied-slot, vacant-slot reason, and de-duplicated shadow-candidate diagnostics.
- `backup_fill_candidate.py`: rejected local candidate that tested 50-59 score backup fills without changing the main buy threshold.
- `cmf_diagnostics.py`: observation-only CMF(20) attribution using adjusted T-1 signal frames; it does not alter strategy scores or orders.
- `strong_trend_capacity_diagnostics.py`: training-only strong-trend entry, idle-slot, cash-headroom, and close-path excursion diagnostics; it never changes order targets.
- `gap_execution_diagnostics.py`: training-only attribution of T-day 09:35 execution gaps normalized by frozen T-1 ATR; it does not filter orders.
- `boll_width_diagnostics.py`: observation-only standard BOLL(20,2) BandWidth direction attribution on frozen T-1 signal frames.
- `sequence_diagnostics.py`: observation-only timing attribution for active RSI/KDJ/MACD crosses inside the existing three-day window.
- `ranking_candidate.py`: rejected local A/B experiment that compared official total-score-first ranking with reversal-score-first ranking on identical cached T-1 signals.
- `efficiency_ratio_diagnostics.py`: observation-only standard Kaufman ER(10) direction attribution on frozen T-1 signal frames.
