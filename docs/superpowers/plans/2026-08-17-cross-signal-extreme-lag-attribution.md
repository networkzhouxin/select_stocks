# Cross-Signal Extreme-Lag Attribution Implementation Plan

> **Execution note:** Agentic workers require `superpowers:subagent-driven-development` or `superpowers:executing-plans`. This task is executed inline with `superpowers:executing-plans` because the user did not authorize subagents.

**Goal:** Implement the approved observation-only Step 0 attribution for extreme entry/exit lag on the frozen `cross-v0.3.3` 2019-2021 training path, while making official filled-order alignment, T-1 signal timing, and forward-label isolation hard gates.

**Architecture:** Add one research module built around pure, typed attribution functions plus a narrow training runner. The runner reuses the frozen local signal adapter and closed-trade path, accepts an explicit JoinQuant filled-event sequence, rejects any date/side/code/quantity mismatch, then computes entry and exit observations without feeding forward outcomes back into scoring or execution. Report rendering consumes immutable observation rows and produces Markdown/JSON only under `cross_signal_strategy/reports/`.

**Tech stack:** Python 3, pandas, dataclasses, pytest, existing `LocalBacktestEngine`, `DiagnosticOrderPlanner`, `ClosedTradeDiagnostic`, and `order_path_diagnostics` utilities.

## Global Constraints

- Read market data only from `G:\financial\history_data\cross_signal_train_2019_2021` and the approved read-only 2018 warm-up root.
- Never read validation/live market data, and never write below either source-data root.
- Decisions and scores use T-1 or earlier data; T 09:35 prices are execution/evaluation marks only.
- Forward MAE/MFE and post-exit returns are output labels only and cannot enter a planner, score adapter, or order path.
- Do not edit either formal JoinQuant/PTrade strategy or any multi-factor file.
- No candidate rule, threshold search, age-decay formula, or formal-version change is authorized.
- Every implementation slice starts with a focused failing test and ends with the smallest passing implementation.

## Task 1: Safety And Official Fill-Path Gate

**Files:**

- Create: `tests/test_cross_signal_extreme_lag_attribution.py`
- Create: `cross_signal_strategy/research/extreme_lag_attribution.py`

**Interfaces:**

- `OfficialPathEvidence`: immutable alignment result with expected/actual counts and status.
- `assert_official_fill_path(expected_events, actual_events) -> OfficialPathEvidence`.
- `assert_report_path(path) -> Path`.
- `assert_training_episode_dates(buy_date, sell_date) -> None`.

**Steps:**

1. Add literal fixtures proving an exact filled path must match date, side, code, and amount; run the test and observe import/behavior failure.
2. Add tests rejecting a missing expected path, a quantity mismatch, dates outside 2019-2021, and a report target under either immutable data root; run and observe failure.
3. Implement only the guard dataclass/functions, reusing `assert_not_training_write_path`; run the focused tests green.
4. Self-review: mutate side, amount, and boundary dates mentally; each mutation must be caught by a named test. Record any uncovered mutation before continuing.

## Task 2: Entry Cross-Age Attribution

**Files:**

- Modify: `tests/test_cross_signal_extreme_lag_attribution.py`
- Modify: `cross_signal_strategy/research/extreme_lag_attribution.py`

**Interfaces:**

- `EntryLagObservation`: code/fill identity, T-1 signal date, contributing bullish crosses, reversal contribution by age, age-two share, earliest-cross delay, ATR-normalized extension/gap, and evaluation-only 5-session MAE/MFE.
- `build_entry_lag_observation(trade, signal_frame, execution_close, forward_closes) -> EntryLagObservation`.

**Steps:**

1. Add a hand-checked fixture with age 0/1/2 bullish flags and literal score weights; assert exact contributions and age-two share; run red.
2. Implement cross-name/weight mapping and require each active flag to have an age in `[0, 2]`; run green.
3. Add red tests proving the signal frame ends on `signal_date < buy_date`, earliest cross uses trading-session offsets, and ATR normalization retains `None` when ATR/evidence is unavailable.
4. Implement the T-1/date/ATR calculations; run green.
5. Add red tests proving forward 5-session closes affect only `evaluation_mae_5`/`evaluation_mfe_5`, never the signal-derived fields; implement the isolated label helper and run green.
6. Self-review dataclass types and missing-data semantics; retain missing evidence explicitly rather than imputing zero.

## Task 3: Exit-Lag Attribution

**Files:**

- Modify: `tests/test_cross_signal_extreme_lag_attribution.py`
- Modify: `cross_signal_strategy/research/extreme_lag_attribution.py`

**Interfaces:**

- `ExitSignalDay`: current execution date, T-1 signal date, sell score, confirmation/protection state, and 09:35 mark.
- `ExitLagObservation`: first eligible high-score state, delay, profits/givebacks, exit type, and evaluation-only post-exit returns.
- `build_exit_lag_observation(trade, trade_dates, signal_days, peak_close, post_exit_closes) -> ExitLagObservation`.

**Steps:**

1. Add a red test where the first `sell_score >= 30` lacks confirmation, a later day is protected, and the actual filled exit occurs later; assert first-date selection and trading-session delay.
2. Implement eligibility after the official minimum hold and classify the first high-score day as `confirmation_absent`, `confirmation_present`, or `protected`; run green.
3. Add red tests for signal versus ATR exits, missing first-high-score evidence, and exact peak/first-score/exit giveback arithmetic; implement and run green.
4. Add red tests proving post-exit 3/5-session returns are evaluation-only and remain `None` at the boundary; implement and run green.
5. Self-review event ordering, profit units, and missing-state retention.

## Task 4: Distribution, Annual, ETF, And Exit-Type Summaries

**Files:**

- Modify: `tests/test_cross_signal_extreme_lag_attribution.py`
- Modify: `cross_signal_strategy/research/extreme_lag_attribution.py`

**Interfaces:**

- `DistributionSummary`: total count, usable count, missing count, median, Q1, Q3, min, max.
- `ExtremeLagReport`: path evidence, observations, grouped distributions, concentration audit, and Step 0 decision.
- `summarize_extreme_lag(entry_rows, exit_rows, path_evidence) -> ExtremeLagReport`.
- `format_extreme_lag_report(report) -> str`.

**Steps:**

1. Add red tests for literal median/quartile/missing results and required groups: full period, 2019/2020/2021, ETF code, signal/ATR exit type.
2. Implement generic continuous distribution summaries without inventing an extreme threshold; run green.
3. Add red tests ensuring a pattern is stopped when annual direction is inconsistent, ETF concentration dominates, sample evidence is missing, or path evidence is not aligned.
4. Implement the conservative stop/eligible decision and explanatory reasons; run green.
5. Add a Markdown rendering test based on semantic sections/status, not exact prose; implement and run green.
6. Self-review that tail rows are examples only and no multiple candidate rules are ranked.

## Task 5: Training Runner And Artifact Generation

**Files:**

- Modify: `tests/test_cross_signal_extreme_lag_attribution.py`
- Modify: `cross_signal_strategy/research/extreme_lag_attribution.py`
- Generate when evidence is available: `cross_signal_strategy/reports/extreme_lag_attribution_2019_2021.md`
- Generate when evidence is available: `cross_signal_strategy/reports/extreme_lag_attribution_2019_2021.json`

**Interfaces:**

- `run_training_extreme_lag_attribution(joinquant_events, loader=None, initial_cash=20000.0) -> ExtremeLagReport`.
- CLI `python -m cross_signal_strategy.research.extreme_lag_attribution --joinquant-transactions <path> --report-dir <path>`.

**Steps:**

1. Add a red integration test with a tiny fake loader/adapter path proving the runner validates official fills before computing observations and refuses an absent official event sequence.
2. Implement the runner by reusing the existing training adapter/planner/engine and `extract_local_order_events`; no strategy logic is duplicated or changed.
3. Add a red CLI/report-path test ensuring outputs stay outside immutable roots; implement atomic report writes under the approved repository report directory.
4. Run the focused test module, then the complete cross-signal test selection.
5. Run the 2019-2021 report only if a machine-readable official JoinQuant filled-order artifact exists. If it does not, generate no pseudo-official result; report `BLOCKED_MISSING_OFFICIAL_FILL_PATH` and request that artifact.
6. Inspect `git diff --name-only` and prove neither formal strategy nor any multi-factor file changed.

## Task 6: Verification, Research Record, And Commit

**Files:**

- Modify only if a valid completed report exists: `cross_signal_strategy/docs/backtest_notes.md`
- Modify only if Step 0 reaches a supported decision: `cross_signal_strategy/docs/decisions.md`

**Steps:**

1. Run import/compile checks and all relevant pytest tests from a clean command.
2. Re-run the report command and verify deterministic Markdown/JSON hashes if official events are available.
3. Review the final diff against the approved design, date/root guards, and non-goals.
4. Document hypothesis, implementation, training result or exact blocker, interpretation, and next step. Do not record a candidate when the evidence gate is blocked.
5. Commit the isolated milestone with a scoped `cross-signal` message.
