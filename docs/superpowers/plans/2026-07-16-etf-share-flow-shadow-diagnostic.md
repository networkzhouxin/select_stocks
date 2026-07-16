# ETF Share-Flow Shadow Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, without changing any order, whether frozen `cross-v0.3.2` upward-cross entries have different training outcomes after positive versus non-positive five-observation ETF share flow.

**Architecture:** Add an exact-root, read-only loader for the isolated share dataset and a pure T-1/T-6 feature calculator with QDII and corporate-action neutralization. Wrap defensive copies of official local scores with shadow metadata, reuse the existing diagnostic replay, and build fixed overall/annual group statistics plus a pre-registered stability gate. Close the research attempt after one 2019-2021 run; no platform candidate is produced.

**Tech Stack:** Python 3, pandas, existing cross-signal local replay, pytest, JSON research records.

## Global Constraints

- Read share data only from `G:\financial\history_data\cross_signal_flow_train_2018_2021`.
- Read price data only through the approved 2018 warm-up and 2019-2021 training loaders.
- Use `log(shares[T-1] / shares[T-6])` over exactly six valid observations.
- QDII codes are blocked; a window crossing a registered split is neutral.
- Do not inspect validation dates, search periods or thresholds, or modify source data.
- Observation metadata must not alter scores, ranking, sizing, orders, sells, or ATR behavior.
- Do not touch formal JoinQuant/PTrade mainlines or production multi-factor files.
- Every implementation behavior starts with a focused failing test.

---

### Task 1: Pre-Register The Single Observation

**Files:**
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`

**Interfaces:**
- Consumes: all currently exhausted research families.
- Produces: one open `etf_share_flow_shadow` family with exactly one allowed observation.

- [ ] **Step 1: Write the failing research-budget test**

Require one open family with fixed horizon 5, sign-only grouping, exact approved root, five domestic codes, four blocked QDII codes, observation-only action, and no validation influence.

- [ ] **Step 2: Run the budget test and verify RED**

Run: `python -m pytest tests/test_cross_signal_research_budget.py -q`

Expected: FAIL because `etf_share_flow_shadow` is absent and the permitted open count is zero.

- [ ] **Step 3: Add the minimal structured budget entry**

```json
{
  "key": "etf_share_flow_shadow",
  "status": "open",
  "max_new_experiments": 1,
  "lookback_observations": 5,
  "grouping": "positive_vs_non_positive",
  "candidate_action": "observation_only",
  "validation_influence": "none"
}
```

Set `max_total_open_experiments` to `1` and mirror the locked constraints in the human-readable budget.

- [ ] **Step 4: Run the budget test and verify GREEN**

Run: `python -m pytest tests/test_cross_signal_research_budget.py -q`

Expected: PASS.

### Task 2: Load And Classify Share Flow Safely

**Files:**
- Create: `tests/test_cross_signal_share_flow_diagnostics.py`
- Create: `cross_signal_strategy/share_flow_diagnostics.py`

**Interfaces:**
- Produces: `ShareFlowDataLoader.load_history(code, signal_date) -> pandas.DataFrame`.
- Produces: `calculate_share_flow(frame, code, decision_date, signal_date, corporate_actions) -> ShareFlowObservation`.
- Produces immutable `ShareFlowObservation` fields `code`, `decision_date`, `signal_date`, `baseline_date`, `value`, `raw_state`, and `comparison_group`.

- [ ] **Step 1: Write failing loader tests**

Cover exact-root rejection, 2018/2019 cross-year loading, schema/date/share validation, duplicate rejection, QDII blocking, cache behavior, and defensive copies. Monkeypatch the approved root for temporary fixtures rather than weakening the production root assertion.

- [ ] **Step 2: Write failing feature tests**

Use six dated share rows to prove positive, negative, and flat classification; require endpoint exactly T-1, reject future rows and T-or-later signal dates, neutralize a split inside `(baseline, signal]`, and resume when the split is the baseline.

```python
observation = calculate_share_flow(
    frame=frame,
    code="159915",
    decision_date="2020-01-10",
    signal_date="2020-01-09",
    corporate_actions=(),
)
assert observation.value == pytest.approx(math.log(110.0 / 100.0))
assert observation.raw_state == "net_creation"
assert observation.comparison_group == "positive"
```

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py -q`

Expected: FAIL with `ModuleNotFoundError` for `share_flow_diagnostics`.

- [ ] **Step 4: Implement the minimal loader and pure calculation**

Define locked constants for the exact root, eligible codes, blocked QDII codes, training dates, warm-up dates, and `FLOW_LOOKBACK = 5`. Load at most the required prior year plus signal year, normalize code/date/share columns, validate before caching, and return copies.

The calculator must sort rows ending exactly at `signal_date`, select six rows, reject any row after the signal date, and calculate only:

```python
value = math.log(endpoint_share / baseline_share)
```

Return `value=None` for blocked QDII, insufficient history, or a corporate-action crossing.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py -q`

Expected: loader and pure-feature tests PASS.

### Task 3: Attach Shadow Metadata Without Changing Orders

**Files:**
- Modify: `tests/test_cross_signal_share_flow_diagnostics.py`
- Modify: `cross_signal_strategy/share_flow_diagnostics.py`

**Interfaces:**
- Produces: `ShareFlowSignalAdapter.score(code, current_date, return_reason=False)`.
- Adds copied score fields prefixed `share_flow_`: `value_5`, `raw_state`, `comparison_group`, `signal_date`, `baseline_date`, and `blocked`.
- Consumes: official `build_training_signal_adapter()` and `DiagnosticOrderPlanner`.

- [ ] **Step 1: Write failing adapter tests**

Use a fake official source to require exact signal-date agreement, T-1-only share access, metadata on a copied score, cached deterministic results, preserved `buy_allowed` and numeric scores, QDII neutralization, and unchanged source dictionaries.

- [ ] **Step 2: Write the order-path parity test**

Run a synthetic planner input through the base adapter and shadow adapter and assert identical order dictionaries. The only permitted difference is metadata inside diagnostic score snapshots.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py -q`

Expected: FAIL because `ShareFlowSignalAdapter` is missing.

- [ ] **Step 4: Implement the minimal adapter**

Call the source with `return_reason=True`, copy the score, require its `signal_date`, load share history only through that date, calculate the observation using `current_date` as the decision date, and add metadata. Never assign to an existing strategy field.

- [ ] **Step 5: Run focused and adjacent tests**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py tests/test_cross_signal_trade_diagnostics.py tests/test_cross_signal_local_order_planner.py -q`

Expected: PASS.

### Task 4: Build Fixed Statistics And Stability Gate

**Files:**
- Modify: `tests/test_cross_signal_share_flow_diagnostics.py`
- Modify: `cross_signal_strategy/share_flow_diagnostics.py`

**Interfaces:**
- Produces: `ShareFlowStats`, `ShareFlowGateDecision`, and `ShareFlowReport`.
- Produces: `build_share_flow_report(trades) -> ShareFlowReport`.
- Produces: `run_training_share_flow_observation(loader=None, flow_loader=None, initial_cash=20000.0) -> ShareFlowReport`.

- [ ] **Step 1: Write failing report tests**

Require raw-state counts, all-buy coverage, eligible-domestic coverage, and overall/annual positive-versus-non-positive statistics. Pass a 2022 trade and require rejection as outside the training window.

- [ ] **Step 2: Write failing gate tests**

Prove the gate requires six trades per group overall, two per group in every year, and the same group to have strictly higher average return and win rate in every year. Prove a single reversed year, tie, or sparse year fails.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py -q`

Expected: FAIL because report and gate interfaces are missing.

- [ ] **Step 4: Implement report, formatting, and training runner**

Reuse `DiagnosticOrderPlanner`, `LocalBacktestEngine`, and
`build_closed_trade_diagnostics`. Keep all raw states in coverage, but include
only `positive` and `non_positive` in comparative statistics. The CLI output
must name the locked horizon, eligible/blocked codes, coverage, group metrics,
annual metrics, gate status, and every failure reason.

- [ ] **Step 5: Run focused and adjacent tests**

Run: `python -m pytest tests/test_cross_signal_share_flow_diagnostics.py tests/test_cross_signal_trade_diagnostics.py tests/test_cross_signal_local_training_run.py -q`

Expected: PASS.

### Task 5: Execute Once, Close The Budget, And Record The Result

**Files:**
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `tests/test_cross_signal_multiple_testing_audit.py`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/backtest_notes.md`
- Modify: `cross_signal_strategy/docs/decisions.md`
- Modify: `cross_signal_strategy/docs/failed_experiments.md` when the gate fails
- Modify: `cross_signal_strategy/README.md`
- Modify: `cross_signal_strategy/docs/multiple_testing_audit.md`

**Interfaces:**
- Consumes: `python -m cross_signal_strategy.share_flow_diagnostics` output.
- Produces: one exhausted research record, a failed/non-adopted count increased from 49 to 50, and a multiple-testing lower bound increased from 50 to 51 after including the selected mainline.

- [ ] **Step 1: Run the training-only observation**

Run: `python -m cross_signal_strategy.share_flow_diagnostics`

Expected: one report restricted to 2019-2021 with no order-changing candidate.

- [ ] **Step 2: Apply the pre-registered interpretation**

If the gate fails, record the family as exhausted and prohibit neighboring
lookbacks/thresholds. If it passes, record only that a future independently
designed candidate is permitted; still close this observation and create no
strategy file.

- [ ] **Step 3: Write failing closure/accounting assertions**

Require the family to be exhausted with zero remaining experiments, the exact
training evidence in its rationale, `max_total_open_experiments` restored to
zero, the expected failed-experiment count set to 50, and the minimum-trial
count set to 51 because it includes the selected mainline.

- [ ] **Step 4: Update records and verify closure tests**

Run: `python -m pytest tests/test_cross_signal_research_budget.py tests/test_cross_signal_multiple_testing_audit.py -q`

Expected: PASS.

- [ ] **Step 5: Run the complete test suite and scope checks**

Run: `python -m pytest tests -q`

Run: `git diff --check`

Run: `git status --short`

Expected: all tests PASS; formal cross-signal platform files, source datasets,
and production multi-factor files are unchanged.

- [ ] **Step 6: Commit the milestone**

Commit the design, tests, implementation, and closed research evidence with a
message that states the observed outcome without implying strategy adoption.
