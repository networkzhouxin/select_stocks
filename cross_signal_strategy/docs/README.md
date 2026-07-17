# Cross-Signal Research Docs

This folder stores research notes, validation records, and design decisions for the cross-signal ETF strategy.

The three formal entry files live one level up in `cross_signal_strategy/`: the
JoinQuant strategy, the PTrade adapter, and the local training replay entry.
Archived candidates and probes live under `archive/`; support and research
modules live under `local/`, `research/`, and `tools/`.

Key files:

- `strategy_spec.md`: current strategy design and rule specification.
- `backtest_notes.md`: training, candidate, and validation result records.
- `validation_summary.md`: frozen cross-period validation summary and adoption recommendation.
- `decisions.md`: adopted decisions and rationale.
- `failed_experiments.md`: rejected experiments and why not to repeat them casually.
- `validation_protocol.md`: frozen out-of-sample validation protocol before inspecting reserved-period results.
