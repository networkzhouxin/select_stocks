# Controlled Breakout Anti-Chase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, using only 2019-2021 training data, whether rejecting already-eligible cross-signal buys that break 20-day resistance while overextended improves the frozen strategy without turning it into a breakout strategy.

**Architecture:** Add an isolated signal adapter that enriches defensive copies of official scores with T-2-safe resistance and fixed T-1 extension diagnostics. First run an observation-only annual/sample gate; only if it passes may the same module run one candidate that flips `buy_allowed` to false for `extended_breakout`. Official JoinQuant/PTrade files remain untouched unless every local gate passes, in which case only an isolated JoinQuant candidate is created.

**Tech Stack:** Python 3, pandas, existing local replay engine, pytest, structured research-budget JSON.

## Global Constraints

- Read only `G:\financial\history_data\cross_signal_train_2019_2021` and approved read-only 2018 warm-up through existing loaders.
- Signals and extension labels use T-1 or earlier; resistance uses exactly 20 valid bars ending T-2.
- `extended_breakout` is fixed as breakout and (`RSI6 >= 75` or close/MA20-1 `>= 10%`).
- Breakout never creates a buy, adds score, changes sizing, or affects sells.
- Do not inspect validation periods, search neighboring thresholds, or modify source market data.
- Do not touch production multi-factor files or formal cross-signal platform strategies.
- Every production-code behavior starts with a failing test.

---

### Task 1: Register The One-Shot Research Budget

**Files:**
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `tests/test_cross_signal_research_budget.py`

**Interfaces:**
- Consumes: existing `horizontal_price_structure` closed record.
- Produces: one open `controlled_breakout_anti_chase` item with the exact period, thresholds, gates, and prohibited neighboring searches from the design spec.

- [ ] **Step 1: Write the failing budget test**

Assert that the new family is open, contains exactly one variant, fixes RSI at 75 and MA20 distance at 0.10, restricts data to training/warm-up, and prohibits validation influence and parameter alternatives.

- [ ] **Step 2: Run the budget test and verify RED**

Run: `python -m pytest tests/test_cross_signal_research_budget.py -q`

Expected: FAIL because `controlled_breakout_anti_chase` is absent.

- [ ] **Step 3: Add the minimal structured and human-readable budget entry**

Use this locked content rather than introducing a parameter list:

```json
{
  "family": "controlled_breakout_anti_chase",
  "status": "open",
  "variants_allowed": 1,
  "structure_period": 20,
  "rsi6_extension": 75,
  "ma20_extension": 0.10,
  "candidate_action": "reject_extended_breakout_only",
  "validation_influence": "none"
}
```

- [ ] **Step 4: Run the budget test and verify GREEN**

Run: `python -m pytest tests/test_cross_signal_research_budget.py -q`

Expected: PASS.

### Task 2: Build T-2-Safe Extension Classification

**Files:**
- Create: `tests/test_cross_signal_breakout_extension_diagnostics.py`
- Create: `cross_signal_strategy/breakout_extension_diagnostics.py`

**Interfaces:**
- Produces: `BreakoutExtensionSignalAdapter.score(code, current_date, return_reason=False)`.
- Produces score fields: `breakout_extension_label`, `breakout_extension_blocked`, `breakout_rsi6`, `breakout_ma20_distance`, `breakout_return_5`, `breakout_return_10`, `breakout_return_20`, `breakout_rise_from_low`, `breakout_signal_date`, and `breakout_level_data_date`.
- Reuses: `calc_horizontal_structure()` and `STRUCTURE_PERIOD` from `horizontal_structure_diagnostics.py`.

- [ ] **Step 1: Write failing tests for fixed classification and boundaries**

Cover `no_breakout`, `controlled_breakout`, RSI exactly 75, MA20 distance exactly 10%, invalid MA20, and defensive-copy behavior.

```python
assert classify_breakout("breakout", rsi6=74.9, close=109.9, ma20=100) == "controlled_breakout"
assert classify_breakout("breakout", rsi6=75.0, close=100, ma20=100) == "extended_breakout"
assert classify_breakout("breakout", rsi6=40.0, close=110, ma20=100) == "extended_breakout"
assert classify_breakout("near_resistance", rsi6=90, close=120, ma20=100) == "no_breakout"
```

- [ ] **Step 2: Write failing tests for timing and trailing diagnostics**

Require T-1 ending returns, T-2 resistance inputs, rejection of any row after signal date, and exactly 20 prior valid bars. Verify the source score is unchanged.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_cross_signal_breakout_extension_diagnostics.py -q`

Expected: FAIL with `ModuleNotFoundError` for the new module.

- [ ] **Step 4: Implement the minimal pure functions and adapter**

Define constants and signatures exactly as follows:

```python
RSI6_EXTENSION = 75.0
MA20_EXTENSION = 0.10
TRAILING_PERIODS = (5, 10, 20)

def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_breakout(pressure_bucket, rsi6, close, ma20) -> str:
    if str(pressure_bucket) != "breakout":
        return "no_breakout"
    rsi_value = _optional_float(rsi6)
    close_value = _optional_float(close)
    ma20_value = _optional_float(ma20)
    rsi_extended = rsi_value is not None and rsi_value >= RSI6_EXTENSION
    ma_extended = (
        close_value is not None
        and ma20_value is not None
        and ma20_value > 0
        and close_value / ma20_value - 1.0 >= MA20_EXTENSION
    )
    return (
        "extended_breakout"
        if rsi_extended or ma_extended
        else "controlled_breakout"
    )


def build_trailing_diagnostics(frame, signal_date, support):
    visible = frame.copy()
    visible["_date"] = pd.to_datetime(visible["date"], errors="raise")
    signal_ts = pd.Timestamp(signal_date)
    if not visible.empty and visible["_date"].max() > signal_ts:
        raise ValueError("breakout extension frame contains data after signal_date")
    visible = visible.loc[visible["_date"] <= signal_ts].sort_values("_date")
    closes = pd.to_numeric(visible["close"], errors="coerce").dropna()
    latest = float(closes.iloc[-1])
    values = {}
    for period in TRAILING_PERIODS:
        values["breakout_return_%d" % period] = (
            latest / float(closes.iloc[-period - 1]) - 1.0
            if len(closes) > period and float(closes.iloc[-period - 1]) > 0
            else None
        )
    values["breakout_rise_from_low"] = (
        latest / float(support) - 1.0
        if float(support) > 0
        else None
    )
    return values


def build_breakout_extension_score(frame, base_score, signal_date):
    score = dict(base_score)
    structure = calc_horizontal_structure(
        frame=frame,
        signal_date=str(signal_date),
        atr=float(score["atr"]),
        period=STRUCTURE_PERIOD,
    )
    close = _optional_float(score.get("close"))
    ma20 = _optional_float(score.get("ma20"))
    label = classify_breakout(
        structure.pressure_bucket,
        score.get("rsi6"),
        close,
        ma20,
    )
    score.update(build_trailing_diagnostics(frame, signal_date, structure.support))
    score.update({
        "breakout_extension_label": label,
        "breakout_extension_blocked": False,
        "breakout_rsi6": _optional_float(score.get("rsi6")),
        "breakout_ma20_distance": (
            close / ma20 - 1.0
            if close is not None and ma20 is not None and ma20 > 0
            else None
        ),
        "breakout_signal_date": structure.signal_date,
        "breakout_level_data_date": structure.level_data_date,
    })
    return score

@dataclass
class BreakoutExtensionSignalAdapter:
    source: object
    reject_extended: bool = False

    def score(self, code, current_date, return_reason=False):
        base_score, reason = self.source.score(
            code, current_date, return_reason=True
        )
        if base_score is None:
            return (None, reason) if return_reason else None
        frame, signal_date = self.source.load_signal_frame(code, current_date)
        enriched = build_breakout_extension_score(
            frame=frame,
            base_score=dict(base_score),
            signal_date=signal_date,
        )
        if (
            self.reject_extended
            and enriched["breakout_extension_label"] == "extended_breakout"
        ):
            enriched["buy_allowed"] = False
            enriched["breakout_extension_blocked"] = True
        result = dict(enriched)
        return (result, None) if return_reason else result
```

When `reject_extended` is true, set `buy_allowed=False` only on a copied score labeled `extended_breakout`; preserve every other field and decision.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_cross_signal_breakout_extension_diagnostics.py -q`

Expected: classification, timing, and defensive-copy tests PASS.

### Task 3: Add Observation And Candidate Gates

**Files:**
- Modify: `tests/test_cross_signal_breakout_extension_diagnostics.py`
- Modify: `cross_signal_strategy/breakout_extension_diagnostics.py`

**Interfaces:**
- Produces: `BreakoutExtensionStats`, `BreakoutObservationGate`, `BreakoutCandidateGate`, and `BreakoutExtensionReport`.
- Produces: `build_breakout_extension_report(trades)` and `run_training_breakout_extension(loader=None, initial_cash=20000.0)`.
- Consumes: existing `DiagnosticOrderPlanner`, `PrecomputedSignalAdapter`, `LocalCrossSignalOrderPlanner`, `LocalBacktestEngine`, and `build_baseline_report`.

- [ ] **Step 1: Write failing observation-gate tests**

Prove the gate requires six trades per group overall, two per group in every year, and lower extended-group average return and win rate in every year. Prove one reversed annual comparison fails.

- [ ] **Step 2: Write failing candidate-gate tests**

Prove the gate requires an active order-path change, strictly higher total return, non-worse drawdown/Sharpe/Sortino, and non-worse return in all three years.

- [ ] **Step 3: Write the validation-date rejection test**

Pass a synthetic 2022 trade/result and require `ValueError` containing `outside 2019-2021 training window`.

- [ ] **Step 4: Run focused tests and verify RED**

Run: `python -m pytest tests/test_cross_signal_breakout_extension_diagnostics.py -q`

Expected: FAIL because report and gate functions are missing.

- [ ] **Step 5: Implement statistics, annual gates, and conditional candidate run**

Run the enriched cached scores through an observation-only baseline. If the observation gate fails, return `candidate=None` without instantiating or running the blocking adapter. If it passes, run exactly one candidate with `reject_extended=True`, compare the complete path, and evaluate the locked candidate gate.

- [ ] **Step 6: Run focused and adjacent tests**

Run: `python -m pytest tests/test_cross_signal_breakout_extension_diagnostics.py tests/test_cross_signal_horizontal_structure_diagnostics.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_order_planner.py -q`

Expected: PASS.

### Task 4: Execute The Training Experiment And Close The Branch

**Files:**
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/backtest_notes.md`
- Modify: `cross_signal_strategy/docs/decisions.md`
- Modify: `cross_signal_strategy/docs/failed_experiments.md` if rejected
- Modify: `cross_signal_strategy/README.md`
- Modify: `cross_signal_strategy/docs/multiple_testing_audit.md`
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `tests/test_cross_signal_multiple_testing_audit.py`
- Conditional create: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_breakout_anti_chase_candidate.py`
- Conditional create: candidate parity test only if every local gate passes

**Interfaces:**
- Consumes: formatted output from `python -m cross_signal_strategy.breakout_extension_diagnostics`.
- Produces: a closed research record and updated minimum trial count.

- [ ] **Step 1: Run the training-only diagnostic**

Run: `python -m cross_signal_strategy.breakout_extension_diagnostics`

Expected: a report for 2019-2021 only, with group/year metrics, observation gate, and candidate metrics only when the observation gate passes.

- [ ] **Step 2: Apply the pre-registered branch without reinterpretation**

If the observation gate fails, record rejection and do not create a candidate. If it passes but the candidate gate fails, record candidate rejection. Only if both pass, create the isolated JoinQuant candidate without touching formal JoinQuant/PTrade files.

- [ ] **Step 3: Close the research budget and increment multiple-testing accounting**

Change the family to `exhausted`, record exact evidence, append one failed/non-adopted ledger entry when appropriate, and increase the minimum trial count by one. Do not reopen alternative thresholds.

- [ ] **Step 4: Run the multiple-testing audit**

Run: `python -m cross_signal_strategy.multiple_testing_audit`

Expected: report uses the updated minimum trial count and remains labeled in-sample.

- [ ] **Step 5: Run the complete cross-signal suite**

Run: `python -m pytest tests -q`

Expected: all tests PASS with no validation-period reads.

- [ ] **Step 6: Verify scope and commit the milestone**

Run: `git diff --check`

Run: `git status --short`

Confirm formal JoinQuant/PTrade mainlines and production multi-factor files are unchanged unless the strictly conditional isolated candidate file was created. Commit the result with a message describing the observation outcome rather than implying adoption.
