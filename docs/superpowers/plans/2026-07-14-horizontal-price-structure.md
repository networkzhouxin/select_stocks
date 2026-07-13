# Horizontal Price Structure Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one pre-registered, observation-only 20-day horizontal support/resistance attribution on the official 2019-2021 cross-signal training path.

**Architecture:** Wrap the frozen training signal adapter with a defensive diagnostic adapter that derives T-2-safe 20-day levels and ATR-normalized distances. Reuse the official local order path and closed-trade diagnostics, then apply one fixed annual consistency gate without changing orders.

**Tech Stack:** Python 3, pandas, pytest, existing cross-signal local replay and research-budget modules.

## Global Constraints

- Read only the approved 2018 warm-up and 2019-2021 training data.
- Never inspect validation-period results while designing or interpreting the rule.
- Levels use exactly 20 valid bars ending T-2; T-1 is comparison data only.
- The only candidate hypothesis is mild-uptrend near-resistance underperformance using a fixed one-ATR boundary.
- Write and verify failing tests before implementation code.
- Keep official JoinQuant, PTrade, and production multi-factor files unchanged.

---

### Task 1: Pre-Register The Single Research Item

**Files:**
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`

**Interfaces:**
- Consumes: `load_research_budget(path)` and `evaluate_experiment_request(...)`.
- Produces: one open family named `horizontal_price_structure` with exactly one variant.

- [ ] **Step 1: Write a failing repository-budget test**

Assert that `max_total_open_experiments == 1`, the new family is open, its
budget is one, and exactly one variant is accepted while two are rejected.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_cross_signal_research_budget.py -q`

Expected: failure because `horizontal_price_structure` does not exist and the
repository budget is zero.

- [ ] **Step 3: Add the locked family to the structured and readable budgets**

Use the design's exact 20-day/T-2/one-ATR hypothesis. Do not open any other
family.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `pytest tests/test_cross_signal_research_budget.py -q`

Expected: all research-budget tests pass.

### Task 2: Build The Observation-Only Diagnostic With TDD

**Files:**
- Create: `tests/test_cross_signal_horizontal_structure_diagnostics.py`
- Create: `cross_signal_strategy/horizontal_structure_diagnostics.py`

**Interfaces:**
- Produces: `calc_horizontal_structure(frame, signal_date, atr, period=20)`.
- Produces: `HorizontalStructureSignalAdapter` implementing `score(...)`.
- Produces: `build_horizontal_structure_report(trades)` and `run_training_horizontal_structure()`.

- [ ] **Step 1: Write failing tests for level calculation and buckets**

Cover exact T-2 exclusion, a T-1 resistance breakout, near-resistance and
near-support one-ATR boundaries, insufficient history, and invalid ATR.

- [ ] **Step 2: Run the new test file and verify RED**

Run: `pytest tests/test_cross_signal_horizontal_structure_diagnostics.py -q`

Expected: import failure because the diagnostic module does not exist.

- [ ] **Step 3: Implement the minimal structure calculation and adapter**

The adapter must reject rows after the signal date, copy the base score, and
never modify the wrapped score or trading decisions.

- [ ] **Step 4: Write and verify failing attribution-gate tests**

Cover sufficient stable annual underperformance, one-year reversal, and a
2022 trade rejection.

- [ ] **Step 5: Implement report grouping, gate, formatting, and training runner**

Reuse `DiagnosticOrderPlanner`, `LocalBacktestEngine`, and
`build_closed_trade_diagnostics`.

- [ ] **Step 6: Run focused and related tests**

Run: `pytest tests/test_cross_signal_horizontal_structure_diagnostics.py tests/test_cross_signal_research_budget.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_backtester.py -q`

Expected: all selected tests pass.

### Task 3: Run, Interpret, Close, And Record

**Files:**
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/backtest_notes.md`
- Modify: `cross_signal_strategy/docs/decisions.md`
- Modify if gate fails: `cross_signal_strategy/docs/failed_experiments.md`
- Modify: `cross_signal_strategy/README.md`
- Modify: `tests/test_cross_signal_research_budget.py`

**Interfaces:**
- Consumes: `python -m cross_signal_strategy.horizontal_structure_diagnostics`.
- Produces: an immutable training result and a closed research family.

- [ ] **Step 1: Run the diagnostic only on approved training sources**

Run: `python -m cross_signal_strategy.horizontal_structure_diagnostics`

Expected: overall, annual, mild-annual, support, and pressure statistics plus a
deterministic gate decision.

- [ ] **Step 2: Record the exact output and close the family**

If the gate fails, append one complete failed-experiment record and increment
the expected failed count. If it passes, record that only the locked candidate
is admissible for a separately reviewed JoinQuant candidate; do not tune it.

- [ ] **Step 3: Run full verification**

Run: `pytest -q`

Expected: zero failures.

- [ ] **Step 4: Audit scope and commit the milestone**

Confirm no source market data, validation data, formal strategy, or production
multi-factor file changed. Commit the diagnostic and recorded conclusion.

