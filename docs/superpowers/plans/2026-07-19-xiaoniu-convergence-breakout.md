# Xiaoniu Convergence Breakout V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Build an isolated JoinQuant V4 stock strategy for the PDF-derived pattern “moving-average convergence → volume breakout → low-volume pullback confirmation,” with auditable T-1 signals and fixed, non-optimized rules.

**Architecture:** Keep V1–V3 unchanged. Add one self-contained JoinQuant strategy file whose core calculations are pure Python/numpy functions, then wrap those functions with JoinQuant scheduling, point-in-time universe selection, execution guards, and position state. Add focused pytest coverage for every signal gate, risk sizing, deterministic ranking, and future-data boundary.

**Tech Stack:** Python 3, numpy, JoinQuant APIs, pytest.

## Global Constraints

- Signals at T 09:35 may use only daily bars ending at T-1. T current price is execution-only.
- Freeze all initial rules before seeing validation results. Training window is 2019-01-01 through 2021-12-31.
- Reserve 2022-01-01 through 2023-12-31 for validation, 2024-01-01 onward for recent out-of-sample review, and 2015-01-01 through 2018-12-31 for stress review.
- JoinQuant is the authority for returns. Local tests validate logic and data boundaries, not performance.
- Use no grid search or threshold optimization. Any later rule change needs a new hypothesis and experiment record.
- Preserve V1–V3 and keep V4 independent so failed research can be discarded safely.

## Frozen V4 Rules

- Universe: point-in-time HS300 plus CSI 500 constituents on T-1; main-board A shares only; exclude ST, suspended, delisting-labelled, and invalid-quote securities.
- Market gate: T-1 CSI 300 close is at or above MA60 and MA60 is no lower than five sessions earlier.
- Setup: MA10, MA20, and MA60 maximum-to-minimum spread is at most 3% on at least three of the five sessions ending immediately before breakout.
- Breakout: close exceeds the highest close of the prior 20 sessions, volume is at least 1.5 times prior-20-session mean volume, close is above MA20, and MA20 is not below MA60.
- Confirmation: one to five completed sessions after breakout, close stays at or above the breakout level, low stays no more than 1% below that level, volume is below breakout-day volume, and the confirmation candle closes no lower than it opens.
- Entry: buy at T 09:35 only when T-1 is the confirmation day. Skip paused, invalid-price, or limit-up quotes. Do not chase an execution price more than 3% above the confirmation close.
- Ranking: confirmation close relative to breakout level descending, then breakout volume ratio descending, then code ascending. This is deterministic and contains no unrelated factors.
- Portfolio: maximum three holdings, 1% account-equity risk budget per new trade, 30% single-position value cap, 100-share lots, and no order when one lot is unaffordable.
- Initial stop: 1% below the lower of breakout level and confirmation-day low. This is the only protective stop model in V4.
- Exit: sell when T-1 close is below the stored stop, below MA20, or holding age reaches 20 trading sessions. Current-day price determines only order execution.
- No same-day sell/rebuy of the same security.

---

### Task 1: Lock the research protocol and create the first RED test

**Files:**
- Create: `xiaoniustock/convergence_breakout_v4_research.md`
- Create: `tests/test_xiaoniu_convergence_breakout_strategy.py`
- Create later after RED: `xiaoniustock/xiaoniustock_joinquant_v4.py`

**Step 1: Write the research protocol**

Document the hypothesis, frozen rules above, T-1 decision/execution timeline, train/validation partitions, required JoinQuant metrics, and promotion gates.

**Step 2: Write a failing test for a valid pattern**

Create synthetic OHLCV arrays with:
- a five-day convergence setup,
- a close above the previous 20-session high on at least 1.5x volume,
- and a later low-volume confirmation that holds the breakout level.

Assert `detect_convergence_breakout(bars)` returns a candidate containing `breakout_index`, `confirmation_index`, `breakout_level`, `volume_ratio`, and `stop_price`.

**Step 3: Run the focused test and verify RED**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`

Expected: FAIL because `xiaoniustock_joinquant_v4.py` does not yet exist or the detector is absent.

**Step 4: Implement the minimal detector skeleton**

Add fixed constants and `detect_convergence_breakout(bars)` without JoinQuant calls. Validate input lengths and finite positive prices/volumes.

**Step 5: Run the focused test and verify GREEN**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`

Expected: PASS for the valid-pattern test.

### Task 2: Drive every signal rejection gate with tests

**Files:**
- Modify: `tests/test_xiaoniu_convergence_breakout_strategy.py`
- Modify: `xiaoniustock/xiaoniustock_joinquant_v4.py`

**Step 1: Add one failing test per rejection reason**

Cover:
- moving averages do not converge,
- breakout does not exceed the prior 20-session high,
- breakout volume is below 1.5x,
- confirmation occurs after five sessions,
- confirmation closes below support,
- confirmation low violates the 1% tolerance,
- confirmation volume is not lower than breakout volume,
- confirmation candle closes below its open,
- NaN/zero-volume input.

**Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py -k "reject or invalid"`

Expected: new tests fail for missing gates.

**Step 3: Add the smallest implementations for the gates**

Keep breakout lookback and confirmation window causal. Search only backward from the final completed bar, so a returned candidate always uses the final bar as confirmation.

**Step 4: Run and verify GREEN**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`

Expected: all detector tests pass.

### Task 3: Test and implement ranking, risk sizing, market gate, and exits

**Files:**
- Modify: `tests/test_xiaoniu_convergence_breakout_strategy.py`
- Modify: `xiaoniustock/xiaoniustock_joinquant_v4.py`

**Step 1: Add failing pure-function tests**

Test that:
- `rank_candidates` uses support strength, volume ratio, then code as a stable tie-break;
- `calculate_order_shares` respects 1% risk, 30% cap, cash, price, and 100-share lots;
- invalid or non-positive stop distances return zero shares;
- `market_gate_is_open` requires both close >= MA60 and non-declining five-session MA60;
- `should_exit_position` triggers only for stored-stop breach, MA20 breach, or 20-session maximum age.

**Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py -k "rank or shares or market or exit"`

Expected: FAIL because helpers are missing.

**Step 3: Implement minimal pure helpers**

Avoid global state and JoinQuant objects in these helpers. Return explicit results that the platform wrapper can log.

**Step 4: Run and verify GREEN**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`

Expected: all pure-function tests pass.

### Task 4: Integrate the JoinQuant daily lifecycle without future data

**Files:**
- Modify: `tests/test_xiaoniu_convergence_breakout_strategy.py`
- Modify: `xiaoniustock/xiaoniustock_joinquant_v4.py`

**Step 1: Add failing integration/source-boundary tests**

Use jqdata stubs and source inspection to assert:
- `initialize` schedules state refresh/trading at 09:30/09:35 and close bookkeeping at 15:00/15:30;
- market and stock daily-history calls use an explicit previous-trading-day end date;
- current quotes are read only after candidates are already calculated;
- universe constituents are requested with T-1 `date`;
- limit-up quotes are skipped;
- sell decisions use T-1 close/MA20 plus stored state, while order price remains current-day execution data;
- sold codes cannot re-enter on the same day.

**Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py -k "initialize or previous or execution or limit or reenter"`

Expected: FAIL because the platform lifecycle is incomplete.

**Step 3: Implement the platform wrapper**

Add:
- `initialize`, daily callbacks, and persistent dictionaries;
- explicit previous-trade-date lookup;
- point-in-time universe construction and filters;
- T-1 market gate and batch signal scan;
- sell-before-buy ordering;
- deterministic selection, position sizing, order calls, state updates, and concise audit logs.

Do not copy subjective PDF narratives or add discretionary factors.

**Step 4: Run and verify GREEN**

Run: `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`

Expected: all V4 tests pass.

### Task 5: Verify, document handoff, and create the milestone commit

**Files:**
- Modify: `xiaoniustock/convergence_breakout_v4_research.md`
- Verify: `xiaoniustock/xiaoniustock_joinquant_v4.py`
- Verify: `tests/test_xiaoniu_convergence_breakout_strategy.py`

**Step 1: Record implementation status without claiming performance**

Add test commands and state clearly that no JoinQuant return, drawdown, or Sharpe result exists until an authorized training backtest is run.

**Step 2: Run fresh verification**

Run:
- `python -m py_compile xiaoniustock/xiaoniustock_joinquant_v4.py`
- `python -m pytest -q tests/test_xiaoniu_convergence_breakout_strategy.py`
- `python -m pytest -q tests/test_multifactor_stable_order.py`
- `git diff --check`

If the full suite cannot complete in a reasonable window, report the timeout separately and do not convert it into a passing result.

**Step 3: Review future-function and overfitting risks**

Confirm every signal path ends at T-1, today's quote is execution-only, the parameters match the frozen protocol, and no validation-period result influenced the implementation.

**Step 4: Commit the milestone**

Run:
- `git status --short`
- `git add docs/superpowers/plans/2026-07-19-xiaoniu-convergence-breakout.md xiaoniustock/convergence_breakout_v4_research.md xiaoniustock/xiaoniustock_joinquant_v4.py tests/test_xiaoniu_convergence_breakout_strategy.py`
- `git commit -m "feat: add convergence breakout v4 strategy"`

**Step 5: Report the remaining gate**

The next step is a JoinQuant training-window backtest on 2019-2021. Validation windows remain sealed until rules and the training interpretation are explicitly frozen.
