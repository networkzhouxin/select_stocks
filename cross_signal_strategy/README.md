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

## Research Rules

- Use JoinQuant as the authority for performance validation.
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
- `smart_trade_joinquant_cross_signal_etf.py`: JoinQuant strategy file, to be created in this folder after the design is approved.
