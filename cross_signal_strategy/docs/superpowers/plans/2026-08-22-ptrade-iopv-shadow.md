# PTrade IOPV Shadow Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each implementation task and superpowers:verification-before-completion before claiming success.

**Goal:** Add observation-only PTrade IOPV shadow evidence at 09:35, 10:35, and blocked signal sells without changing formal trading behavior.

**Architecture:** Reuse the fresh snapshot already acquired by the PTrade execution adapter. Add pure observation/classification helpers, ephemeral same-day shadow state, and a recheck at the start of the existing 10:35 callback. Keep all shadow outputs outside the business fingerprint and persistent state.

**Tech Stack:** Python 3, pytest, PTrade adapter APIs, AST-based release verifier.

**Spec:** `cross_signal_strategy/docs/superpowers/specs/2026-08-22-ptrade-iopv-shadow-design.md`

## Global Constraints

- No market-data or validation-period reads.
- No new scheduled callback.
- No changes to scores, thresholds, positions, order quantities, or order decisions.
- Preserve the user's untracked handoff document.
- Use `apply_patch` for file changes and prove every behavior with a failing test first.

---

## Task 1: Lock the executable-price contract

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

1. Add a test where snapshot `last_px` differs from the supplied executable price.
2. Run the test and confirm it fails because latest price currently wins.
3. Change `build_iopv_observation` so a valid supplied executable price wins, with latest price only as fallback.
4. Run the focused test.

## Task 2: Add 09:35 and 10:35 buy shadows

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

1. Add tests for high, normal, and unavailable 09:35 premium classifications while asserting the real order remains identical.
2. Add tests that 10:35 logs `拟恢复买入` or `拟继续放弃`, clears the runtime record, and never calls an order API.
3. Run the new tests and confirm RED.
4. Add the fixed shadow threshold, runtime state initialization/reset, buy-side logger/recorder, and recheck helper.
5. Invoke the recheck before `halt_recover` can return early.
6. Run the focused tests.

## Task 3: Add blocked-sell shadow evidence

**Files:**
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

1. Add tests for a QDII at sell score 30+ that is blocked by price confirmation or ADX.
2. Assert high bid-side premium logs `拟加速卖出`, identifies blockers, and never calls `execute_sell`.
3. Assert non-QDII and below-threshold cases do not emit the shadow log.
4. Run the tests and confirm RED.
5. Add a strict bid-one quote reader and non-binding sell observation hook after the existing `should_force_sell` check.
6. Run the focused tests.

## Task 4: Release identity and documentation

**Files:**
- Modify: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
- Modify: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`
- Modify: `tests/test_cross_signal_ptrade_strategy.py`
- Modify: `tests/test_cross_signal_release_verifier.py`
- Modify: `cross_signal_strategy/docs/ptrade_deployment.md`
- Modify: `cross_signal_strategy/docs/validation_summary.md`
- Modify: `cross_signal_strategy/docs/decisions.md`

1. Add failing expectations for the next shared deployment build.
2. Advance both formal build markers without changing JoinQuant business logic.
3. Document the observation-only boundary and exact log semantics.
4. Confirm business fingerprint and state schema remain unchanged.

## Task 5: Verification and safe point

1. Run focused PTrade and release tests.
2. Run all cross-signal tests with pytest cache disabled.
3. Compile the three formal Python entrypoints.
4. Run `cross_signal_strategy/tools/verify_release.py`.
5. Run `git diff --check`, review `git diff`, and confirm only scoped files changed.
6. Commit the milestone with a concise rollback-safe message.
