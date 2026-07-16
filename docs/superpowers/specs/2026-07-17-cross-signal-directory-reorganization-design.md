# Cross-Signal Directory Reorganization Design

## Goal

Make `cross_signal_strategy/` present the three supported entry surfaces clearly:

1. JoinQuant deployment strategy.
2. Guojin PTrade deployment strategy.
3. Local training-window replay entry.

The reorganization must preserve every historical experiment, diagnostic, probe,
report, test, and Git history. It must not change strategy rules, ETF pools,
parameters, data boundaries, or backtest results.

## Important Distinction

The local replay is not a third independent strategy implementation. It reuses
the frozen JoinQuant business logic through a local data adapter, order planner,
and broker simulation. `local_training_run.py` remains visible at the top level
as the public local replay entry, while its supporting modules move into a
dedicated package.

## Target Layout

```text
cross_signal_strategy/
|-- README.md
|-- smart_trade_joinquant_cross_signal_etf.py
|-- smart_trade_ptrade_cross_signal_etf.py
|-- local_training_run.py
|-- local/
|   |-- __init__.py
|   |-- local_adjustment.py
|   |-- local_backtester.py
|   |-- local_data_loader.py
|   |-- local_data_quality.py
|   |-- local_order_planner.py
|   `-- local_signal_adapter.py
|-- research/
|   |-- __init__.py
|   `-- diagnostics and reporting tools
|-- archive/
|   |-- README.md
|   |-- __init__.py
|   |-- candidates/
|   |   |-- __init__.py
|   |   `-- rejected or superseded strategy candidates
|   `-- probes/
|       |-- __init__.py
|       `-- temporary no-order platform probes
|-- docs/
`-- reports/
```

The top-level Python-file whitelist is therefore exactly:

- `smart_trade_joinquant_cross_signal_etf.py`
- `smart_trade_ptrade_cross_signal_etf.py`
- `local_training_run.py`

## File Classification

### Local replay support

Move the six `local_*` support modules into `local/` while keeping
`local_training_run.py` at the top level. Update its imports and all dependent
research/test imports to the new package path.

### Research and diagnostics

Move non-strategy analysis tools into `research/`. This includes baseline and
trade reports, attribution and stability tools, friction/capital diagnostics,
technical-indicator diagnostics, data-quality diagnostics, research-budget
audits, and chart generation. These files remain executable and tested.

The exact research-module inventory is:

- `attribution_diagnostics.py`
- `baseline_report.py`
- `boll_width_diagnostics.py`
- `breakout_extension_diagnostics.py`
- `capital_utilization_diagnostics.py`
- `cmf_diagnostics.py`
- `efficiency_ratio_diagnostics.py`
- `friction_diagnostics.py`
- `gap_execution_diagnostics.py`
- `horizontal_structure_diagnostics.py`
- `iopv_quality_diagnostics.py`
- `market_breadth_diagnostics.py`
- `multiple_testing_audit.py`
- `order_path_diagnostics.py`
- `portfolio_dependence_diagnostics.py`
- `research_budget.py`
- `sell_diagnostics.py`
- `sequence_diagnostics.py`
- `share_flow_diagnostics.py`
- `strong_trend_capacity_diagnostics.py`
- `trade_chart.py`
- `trade_diagnostics.py`
- `training_stability.py`
- `us_qdii_premium_diagnostics.py`

### Archived candidates

Move all rejected or superseded candidate implementations into
`archive/candidates/`, including both local experiment candidates and temporary
JoinQuant candidate strategy files. Archiving does not mean deletion: their
tests remain active so failed experiments stay reproducible.

The exact candidate inventory is:

- `backup_fill_candidate.py`
- `macd_parameter_candidate.py`
- `ranking_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_atr2_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_pool_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_sell35_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py`

### Archived probes

Move the 513880 suspension probe and JoinQuant/PTrade IOPV capability probes
into `archive/probes/`. They remain no-order diagnostic artifacts and are not
part of either formal deployment strategy.

The exact probe inventory is:

- `smart_trade_joinquant_cross_signal_etf_probe_513880.py`
- `smart_trade_joinquant_cross_signal_iopv_probe.py`
- `smart_trade_ptrade_cross_signal_iopv_probe.py`

### Tests, docs, and reports

- Keep all 46 cross-signal tests in the repository-level `tests/` directory, as
  approved by the user.
- Update imports and direct file paths in tests to follow moved modules.
- Keep `docs/` and `reports/` as dedicated directories.
- Update current documentation references to the new paths. Preserve the
  substance and outcome of historical experiment records.
- Add an archive manifest mapping categories and explaining that archived code
  must not be promoted without a new training-only protocol.

## Migration Rules

1. Use `git mv` for tracked files so history remains traceable.
2. Add package initializers only where needed for stable imports.
3. Do not add compatibility shims at the old top-level paths; those would
   recreate the clutter this change is intended to remove.
4. Do not alter function bodies except for import/path adjustments required by
   the move.
5. Do not read, modify, or delete any market-data directory.
6. Remove only the generated `cross_signal_strategy/__pycache__/` directory
   after verifying its resolved path is inside the workspace.

## Test-First Migration

Before moving production or research files, add a failing structure-contract
test that asserts the three-file top-level whitelist and the required archive,
research, and local package locations. Then perform the moves and update imports
until that test and all existing tests pass.

Verification sequence:

1. Structure-contract test fails against the old layout.
2. Migrate files and update imports/paths.
3. Run all cross-signal tests.
4. Run the full repository test suite.
5. Run `py_compile` over the three entries and all moved Python modules.
6. Run `git diff --check` and confirm only the intended directory is affected.
7. Remove generated bytecode caches and confirm a clean directory tree.

## Acceptance Criteria

- Only the three approved Python entry files remain directly under
  `cross_signal_strategy/`.
- Formal JoinQuant and PTrade files are byte-for-byte unchanged by the move.
- Local replay, research tools, candidates, and probes remain importable.
- All existing tests plus the new structure test pass.
- No production multi-factor file changes.
- No market-data files are read for tuning or modified in any way.
- The final commit provides a single rollback point for the reorganization.
