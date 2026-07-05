# Cross Signal v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated JoinQuant ETF cross-signal timing strategy under `cross_signal_strategy/`.

**Architecture:** Keep the runnable strategy in one JoinQuant-compatible file and expose pure helper functions for tests. Use T-1 daily bars for all scores, Tuesday/Thursday rotation, equal-weight sizing, ATR trailing stop, and no production strategy changes.

**Tech Stack:** Python, pandas, numpy, JoinQuant APIs, local regression tests with importlib and a mocked `jqdata` module.

---

### Task 1: Pure Signal Tests

**Files:**
- Create: `tests/test_cross_signal_strategy.py`
- Create: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`

- [ ] Write failing tests for recent cross detection, buy score composition, sell score composition, deterministic candidate ordering, and same-day sell blocking.
- [ ] Run `python tests/test_cross_signal_strategy.py` and confirm it fails because the strategy file is missing.
- [ ] Implement the minimal pure helpers and strategy defaults.
- [ ] Run `python tests/test_cross_signal_strategy.py` and confirm it passes.

### Task 2: JoinQuant Strategy Shell

**Files:**
- Modify: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`

- [ ] Add `initialize`, scheduled functions, ETF pool, scoring over T-1 data, 09:35 trading, and 15:30 logs.
- [ ] Compile with `python -m py_compile cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`.
- [ ] Run existing multifactor tests to ensure the production strategy was not affected.

### Task 3: Documentation Update

**Files:**
- Modify: `cross_signal_strategy/docs/decisions.md`

- [ ] Record that v0.1 implementation was created from the frozen spec without consulting validation results.
- [ ] Run `git diff --check`.
