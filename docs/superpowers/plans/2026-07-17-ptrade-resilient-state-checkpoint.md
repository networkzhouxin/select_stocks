# PTrade Resilient State Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the formal cross-signal PTrade adapter survive interrupted checkpoint writes and compatible code redeployments while preventing new exposure when held-position risk state cannot be proved.

**Architecture:** Replace the single primary pickle with two checksummed generation slots and retain the old file only as a migration source. Keep broker position/cost as live facts, add a conservative new-buy recovery gate, and produce startup recovery-source summaries without changing business rules.

**Tech Stack:** Python 3, pickle protocol 4, hashlib SHA256, pytest, PTrade documented file and broker APIs.

## Global Constraints

- Modify only the cross-signal PTrade adapter, its tests, and PTrade documentation.
- Do not change JoinQuant code, local-backtest business logic, ETF pool, indicators, thresholds, sizing, or signal dates.
- Do not import or call `os`; PTrade forbids it.
- Write every behavioral test before its production implementation and observe the expected failure.
- Never synthesize missing entry facts; unresolved held positions remain unverified.
- Preserve broker `cost_basis` as the authoritative cost source.

---

### Task 1: Dual-slot envelope and restore selection

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

**Interfaces:**
- Produces: `_live_state_slot_paths(path=None) -> dict[str, str]`
- Produces: `_encode_live_state_envelope(state, generation) -> dict`
- Produces: `_decode_live_state_envelope(envelope) -> tuple[int, dict]`
- Changes: `_persist_live_state(path=None) -> bool`
- Changes: `_restore_live_state(path=None) -> bool`

- [ ] Write tests that require alternating `.a`/`.b` generations, protocol-4 payloads, SHA256 validation, newest-valid selection, and fallback when the newest slot is truncated.
- [ ] Run the new tests and confirm they fail because dual-slot helpers and files do not exist.
- [ ] Add schema/envelope helpers and the smallest two-slot persistence/restore implementation using only `open`, `pickle`, and `hashlib`.
- [ ] Run the focused tests and confirm they pass, then run the existing state tests.

### Task 2: Compatible versions and legacy migration

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

**Interfaces:**
- Produces: `LIVE_STATE_SCHEMA_VERSION = 1`
- Consumes: the legacy path returned by `_cached_live_state_path(path)`
- Produces: private runtime checkpoint source/generation metadata on `g`

- [ ] Write tests that accept a different producer strategy version under schema 1, reject an unknown schema without mutating state, and restore the existing legacy single-file format when both slots are absent.
- [ ] Run the new tests and confirm the expected failures.
- [ ] Separate state schema from producer strategy version, add legacy fallback, and record the chosen restore source/generation in private runtime fields.
- [ ] Run focused and complete PTrade state tests.

### Task 3: Unverified-position new-buy gate

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

**Interfaces:**
- Changes: `execute_buy_candidates(context, all_scores, today) -> int`
- Preserves: verified holdings' ATR and signal exits

- [ ] Write a test with one broker-held unverified ETF and a qualified different candidate; assert no buy order is submitted.
- [ ] Write a control test proving a verified holding may still trigger its normal exit while the recovery buy gate is active.
- [ ] Run both tests and confirm only the new-buy-gate test fails.
- [ ] Add the minimal held/unverified intersection guard before candidate submission.
- [ ] Run focused order and recovery tests.

### Task 4: Startup recovery audit

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

**Interfaces:**
- Produces: `_log_live_recovery_summary(context) -> None`
- Changes: broker rebuild records per-code source as `get-trades` or `get-deliver`

- [ ] Write tests requiring checkpoint source/generation logging and one status line per held ETF containing amount, cost, buy date, ATR, high, status, and source.
- [ ] Run the tests and confirm the summary helper is absent or output is incomplete.
- [ ] Add private per-position source metadata and call the summary after startup recovery.
- [ ] Run focused lifecycle and recovery tests.

### Task 5: Documentation, regression verification, and milestone commit

**Files:**
- Modify: `cross_signal_strategy/docs/ptrade_deployment.md`
- Modify: `cross_signal_strategy/docs/decisions.md`

**Interfaces:**
- Documents: dual-slot format, legacy migration, recovery gate, source logs, and simulation drill.

- [ ] Replace the obsolete single-file atomic-write claim with the dual-slot failure model and deployment checks.
- [ ] Run `python -m pytest -q tests/test_cross_signal_ptrade_strategy.py` and require zero failures.
- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Scan the formal PTrade adapter for forbidden `os` usage and review `git diff --check` plus the complete diff.
- [ ] Commit the tested milestone with a focused message and report verification evidence and remaining platform-only risk.
