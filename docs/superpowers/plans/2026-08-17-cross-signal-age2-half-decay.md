# Cross-Signal Age-2 Half-Decay Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether halving only age-2 bullish cross contributions improves the frozen cross-signal strategy on the approved 2019-2021 local training replay, without changing the official strategy unless the strict local gate passes.

**Architecture:** Add an isolated signal-adapter decorator that consumes the official T-1 score snapshot and subtracts half of the official weight only from bullish crosses whose recorded age equals 2. Run baseline and candidate through identical local loaders, planners, brokers, dates, costs, and execution assumptions. Evaluate a pre-registered strict gate; create a separate JoinQuant candidate only after a pass, otherwise record the failed experiment and leave the official file untouched.

**Tech Stack:** Python 3, pytest, existing `cross_signal_strategy.local` replay engine, existing official signal adapter, Markdown/JSON research ledger.

## Global Constraints

- Work only in `cross_signal_strategy`, its focused tests, and its research documentation.
- Never edit `smart_trade_joinquant_multifactor_etf.py`, `smart_trade_ptrade_multifactor_etf.py`, or any multi-factor file.
- Read market bars only from `G:\financial\history_data\cross_signal_train_2019_2021`; use `G:\financial\history_data\cross_signal_warmup_2018` only as the approved read-only indicator warm-up.
- Keep source-data roots immutable; write reports only inside this worktree.
- All signal inputs remain T-1 or earlier. The T-day 09:35 price remains execution-only.
- Freeze `cross_window=3`, thresholds, sell rules, ranking, sizing, ETF pool, costs, ATR/min-hold rules, and every other official parameter.
- Candidate weights at age 2 are exactly RSI12=6, RSI24=6, MACD=5, KDJ K=3, KDJ J=2.5; ages 0 and 1 retain full official weights.
- Do not search nearby coefficients or compensate for a failed result with another rule.
- Write and observe a failing focused test before each implementation change.
- Do not create or edit a JoinQuant candidate unless every local adoption gate passes.

---

## Task 1: Implement the isolated score decorator

**Files:**

- Create: `tests/test_cross_signal_age2_half_decay_candidate.py`
- Create: `cross_signal_strategy/research/age2_half_decay_candidate.py`

- [x] Add a test-only static base adapter and a complete official-like score snapshot fixture.
- [x] Add a failing test proving an age-2 contributing RSI12/RSI24/MACD/KDJ-K/KDJ-J cross loses exactly half its official contribution, while age-0/age-1 contributions remain unchanged.
- [x] Add failing tests proving mixed/negative RSI group direction is not accidentally turned into a bullish contribution, the caller/base snapshots are not mutated, and missing age metadata for an active bullish cross fails closed.
- [x] Add a failing test proving sell scores, sell flags, dates, extrema, location/trend/volume components, and unrelated fields are unchanged.
- [x] Run `pytest -q tests/test_cross_signal_age2_half_decay_candidate.py` and capture the expected import/behavior failure.
- [x] Implement `Age2HalfDecaySignalAdapter.score(code, current_date, return_reason=False)` as a defensive-copy decorator over the official adapter.
- [x] Recompute only `reversal_score` and `buy_score`; expose observation-only `official_reversal_score`, `official_buy_score`, and `age2_half_decay_penalty` diagnostics.
- [x] Run the focused test file and confirm green.
- [ ] Commit the isolated decorator milestone.

## Task 2: Implement the frozen A/B result and adoption gate

**Files:**

- Modify: `tests/test_cross_signal_age2_half_decay_candidate.py`
- Modify: `cross_signal_strategy/research/age2_half_decay_candidate.py`

- [ ] Add failing unit tests for an all-pass result and for each independent failure class: no changed filled-order day in one calendar year, non-strict total/annualized improvement, worse drawdown, worse Sharpe/Sortino/win-rate/profit-loss ratio, and a worse annual return.
- [ ] Add a failing test that compares filled-order signatures by trading day and requires at least one changed day in each of 2019, 2020, and 2021.
- [ ] Run the focused tests and capture the expected failures.
- [ ] Implement immutable performance/result/gate dataclasses and a deterministic evaluator with no tolerance that could weaken the frozen inequalities.
- [ ] Implement report rendering that records the hypothesis, exact change, baseline/candidate metrics, annual returns, changed-order counts, gate checks, interpretation, and next permitted action.
- [ ] Run the focused tests and confirm green.
- [ ] Commit the A/B gate milestone.

## Task 3: Wire the approved local training replay

**Files:**

- Modify: `tests/test_cross_signal_age2_half_decay_candidate.py`
- Modify: `cross_signal_strategy/research/age2_half_decay_candidate.py`

- [ ] Add failing integration-contract tests proving the runner accepts only the approved loader roots/window, uses the same official adapter and replay configuration for both arms, and never requests post-T-1 signal data.
- [ ] Add a failing CLI/report-path test that forbids output under either immutable market-data root.
- [ ] Run the focused tests and capture the failures.
- [ ] Implement `run_age2_half_decay_training_ab` using `build_training_signal_adapter`, two identical local planners/engines, `initial_cash=20000`, and all approved 2019-2021 trading dates.
- [ ] Compute baseline/candidate performance and annual returns from replay outputs, compare daily filled-order signatures, evaluate the frozen gate, and return a structured result.
- [ ] Implement a CLI that writes the generated Markdown report under `cross_signal_strategy/reports/` only.
- [ ] Run focused tests and then execute the real approved local A/B once.
- [ ] Do not rerun with altered weights, indicators, ETFs, thresholds, or subperiod selections.

## Task 4A: Record a failed local experiment (only if the gate fails)

**Files:**

- Create: `cross_signal_strategy/reports/age2_half_decay_2019_2021.md`
- Modify: `tests/test_cross_signal_age2_half_decay_candidate.py`
- Modify: `cross_signal_strategy/docs/failed_experiments.md`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/research_budget.json`

- [ ] First add failing ledger tests for the exact experiment id, frozen hypothesis/change, fresh training metrics, failed gate reasons, and the prohibition on a JoinQuant candidate.
- [ ] Run the ledger tests and capture the failure.
- [ ] Write the generated report and append the failed-experiment/research-budget entries without altering prior records.
- [ ] Confirm no `age2_half_decay` JoinQuant candidate file exists.
- [ ] Run focused and full cross-signal tests.
- [ ] Commit the rejected-experiment milestone.

## Task 4B: Generate a separate JoinQuant candidate (only if every gate passes)

**Files:**

- Create: `cross_signal_strategy/reports/age2_half_decay_2019_2021.md`
- Create: `cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_age2_half_decay_candidate.py`
- Create: `tests/test_cross_signal_age2_half_decay_joinquant_candidate.py`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/research_budget.json`

- [ ] First add a failing source-level and behavior test proving the generated file differs from the official JoinQuant strategy only in the frozen age-2 buy-side weighting and version/log labels.
- [ ] Run the candidate tests and capture the failure.
- [ ] Create the separate JoinQuant candidate; do not overwrite or import it from the official strategy at runtime.
- [ ] Record the local pass and mark the candidate as awaiting the user's 2019-2021 JoinQuant authority run; do not read or run validation-period data.
- [ ] Run focused and full cross-signal tests.
- [ ] Commit the candidate-generation milestone.

## Task 5: Final verification and handoff

**Files:**

- Verify all files changed in the branch.

- [ ] Run the complete repository test suite with an isolated pytest temp/cache directory.
- [ ] Run `git diff --check`, inspect `git status --short`, and verify no multi-factor or official strategy file changed.
- [ ] Verify the report exactly matches the structured A/B result and that immutable data roots have no generated files.
- [ ] Summarize the actual training result, gate decision, changed-order evidence, files, commits, and residual limitations.
- [ ] If passed, hand the separate candidate to the user for the JoinQuant 2019-2021 authority backtest; if failed, explicitly stop without producing one.
