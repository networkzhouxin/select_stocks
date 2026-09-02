# Cross-Signal Win-Rate Nondegradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce and execute the frozen late-veto + early-pre-MACD JoinQuant experiment so it can be adopted only when return, drawdown, and closed-trade win-rate gates all pass without validation-period tuning.

**Architecture:** Keep the frozen candidate source and fingerprint unchanged. Add a pure-Python paired-result gate that validates run identity/configuration and compares exact-decimal metrics, encode the stricter protocol in the research budget, strengthen characterization tests, then use a runbook and structured JSON evidence to execute the authoritative JoinQuant sequence.

**Tech Stack:** Python 3, standard-library `dataclasses`/`decimal`/`enum`/`json`, pytest, Markdown/JSON research governance, JoinQuant daily backtests.

**Spec:** `cross_signal_strategy/docs/superpowers/specs/2026-09-02-cross-signal-win-rate-nondegradation-design.md`

## Global Constraints

- Candidate identity remains exactly `cross-v0.3.3-late-veto-early-pre-macd-candidate`, `20260822.3-candidate`, `f6b08195dd3d`.
- Formal identity remains exactly `cross-v0.3.3`, `20260822.2`, `77e44d93d255`.
- Signals use T-1 and earlier daily data; T-day 09:35 is execution only.
- Do not change formal JoinQuant, PTrade, IOPV, sell, ATR, sizing, ETF-pool, hold-period, or ranking behavior.
- Training and final-full-period win rate must improve by at least `0.03`; other paired windows must be win-rate non-worse.
- Every paired window requires candidate return `>=` baseline and candidate maximum drawdown `<=` baseline.
- Unclosed positions are excluded from the supplied closed-trade win rate.
- Compare raw precision; do not use rounded display percentages when exports contain more precision.
- Validation results must never alter candidate rules, thresholds, queueing, or gates.
- Approved `G:` training/warm-up roots remain read-only; no substitute data root is permitted.
- Missing-`G:` test failures and stale historical experiment-count assertions are reported, not repaired here.

---

### Task 1: Add an exact paired-result gate

**Files:**
- Create: `cross_signal_strategy/research/late_veto_early_pre_macd_gate.py`
- Create: `tests/test_cross_signal_late_veto_early_pre_macd_gate.py`

**Interfaces:**
- Consumes: one JSON file containing `kind`, `baseline`, and `candidate` run objects.
- Produces: `GateKind`, `RunConfig`, `RunMetrics`, `RunResult`, `PairedRun`, `GateDecision`, `load_paired_run(path)`, `evaluate_pair(pair)`, and CLI exit codes `0=pass`, `1=gate failure`, `2=invalid evidence/configuration`.

Use an enum rather than a boolean. Its complete control-flow contract is:

| `GateKind` | Core comparison | Extra requirements |
|---|---|---|
| `TRAINING_NOMINAL` | return/DD non-worse, win-rate delta `>=0.03` | early sample, late-veto comparison, P/L floor, positive annual returns, positive-to-negative count |
| `TRAINING_DOUBLE_FRICTION` | return/DD/win rate non-worse | doubled-friction profile |
| four `VALIDATION_*` values | return/DD/win rate non-worse | exact approved window |
| `FULL_PERIOD` | return/DD non-worse, win-rate delta `>=0.03` | shared full-period dates; no automatic adoption |

- [ ] **Step 1: Write identity/configuration tests**

Create valid JSON helpers and these tests:

```python
def test_load_pair_preserves_decimal_precision_and_frozen_identities(tmp_path):
    pair = load_paired_run(write_pair(tmp_path, valid_training_payload()))
    assert pair.baseline.metrics.total_return == Decimal("1.2925001")
    assert pair.baseline.config.fingerprint == "77e44d93d255"
    assert pair.candidate.config.fingerprint == "f6b08195dd3d"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", "wrong"),
        ("start_date", "2019-01-02"),
        ("initial_cash", 10000),
        ("execution_time", "15:00"),
    ],
)
def test_invalid_or_unpaired_identity_is_rejected(tmp_path, field, value):
    payload = valid_training_payload()
    payload["candidate"]["config"][field] = value
    with pytest.raises(ValueError):
        load_paired_run(write_pair(tmp_path, payload))
```

The fixture contains nonempty 64-character SHA-256 values for both the complete log and transaction export. Add tests rejecting missing/malformed hashes and missing closed-trade counts.

- [ ] **Step 2: Verify the test is red**

```powershell
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_gate.py -q
```

Expected: collection fails because the gate module does not exist.

- [ ] **Step 3: Implement immutable models and exact JSON loading**

Use these public types:

```python
class GateKind(str, Enum):
    TRAINING_NOMINAL = "training_nominal"
    TRAINING_DOUBLE_FRICTION = "training_double_friction"
    VALIDATION_2022_2023 = "validation_2022_2023"
    VALIDATION_2024_LATEST = "validation_2024_latest"
    VALIDATION_2015_2018 = "validation_2015_2018"
    VALIDATION_2010_2014 = "validation_2010_2014"
    FULL_PERIOD = "full_period"


@dataclass(frozen=True)
class RunConfig:
    strategy_version: str
    build_id: str
    fingerprint: str
    start_date: str
    end_date: str
    initial_cash: Decimal
    frequency: str
    execution_time: str
    commission_rate: Decimal
    minimum_commission: Decimal
    slippage_rate: Decimal
    friction_profile: str
    log_sha256: str
    trade_export_sha256: str


@dataclass(frozen=True)
class RunMetrics:
    total_return: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    closed_trade_count: int
    profit_loss_ratio: Decimal | None
    annual_returns: Mapping[int, Decimal]
    positive_to_negative_round_trips: int | None
    early_fills: int
    early_fill_years: tuple[int, ...]


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    metrics: RunMetrics


@dataclass(frozen=True)
class PairedRun:
    kind: GateKind
    baseline: RunResult
    candidate: RunResult


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]
```

Load via `json.loads(text, parse_float=Decimal, parse_int=Decimal)`. Convert counts to `int` only after proving they are integral/nonnegative. Require exact identities, equal paired date/cash/frequency/time/cost fields, valid hashes, rates in `[0,1]`, and stage-specific windows/profiles.

Require nominal costs to be commission `0.0003`, minimum commission `5`, and slippage `0.001`; require doubled costs to be `0.0006`, `10`, and `0.002`. A profile label cannot substitute for these numeric checks.

```python
EXPECTED_WINDOWS = {
    GateKind.TRAINING_NOMINAL: ("2019-01-01", "2021-12-31", "nominal"),
    GateKind.TRAINING_DOUBLE_FRICTION: ("2019-01-01", "2021-12-31", "double"),
    GateKind.VALIDATION_2022_2023: ("2022-01-01", "2023-12-31", "nominal"),
    GateKind.VALIDATION_2015_2018: ("2015-01-01", "2018-12-31", "nominal"),
    GateKind.VALIDATION_2010_2014: ("2010-01-01", "2014-12-31", "nominal"),
}
```

For `VALIDATION_2024_LATEST`, require start `2024-01-01`, identical end dates, and nominal friction. For `FULL_PERIOD`, require identical dates, start no later than `2010-01-01`, and nominal friction.

- [ ] **Step 4: Add red tests for every gate branch**

```python
@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: set_candidate(p, "total_return", "1.2924999"), "return"),
        (lambda p: set_candidate(p, "max_drawdown", "0.0628001"), "drawdown"),
        (lambda p: set_candidate(p, "win_rate", "0.5878946"), "win-rate"),
        (lambda p: set_candidate(p, "early_fills", 2), "early fills"),
        (lambda p: set_candidate(p, "early_fill_years", [2020]), "two years"),
        (lambda p: set_candidate(p, "profit_loss_ratio", "2.9999"), "profit/loss"),
        (lambda p: set_candidate(p, "positive_to_negative_round_trips", 32), "positive-to-negative"),
        (lambda p: set_annual(p, 2020, "0"), "annual return"),
    ],
)
def test_training_gate_rejects_each_failure(mutation, reason):
    decision = evaluate_pair(pair_from_payload(mutation(valid_training_payload())))
    assert decision.passed is False
    assert any(reason in item for item in decision.reasons)
```

Add one passing test for every enum value and prove validation accepts equal win rate while training/full require `>=0.03` improvement.

- [ ] **Step 5: Implement comparisons and CLI**

```python
def evaluate_pair(pair: PairedRun) -> GateDecision:
    base = pair.baseline.metrics
    candidate = pair.candidate.metrics
    reasons = []
    if candidate.total_return < base.total_return:
        reasons.append("candidate return is lower than paired baseline")
    if candidate.max_drawdown > base.max_drawdown:
        reasons.append("candidate maximum drawdown is higher than paired baseline")
    required_delta = (
        Decimal("0.03")
        if pair.kind in {GateKind.TRAINING_NOMINAL, GateKind.FULL_PERIOD}
        else Decimal("0")
    )
    if candidate.win_rate - base.win_rate < required_delta:
        reasons.append("candidate win-rate improvement is below required delta")
    if pair.kind is GateKind.TRAINING_NOMINAL:
        reasons.extend(_training_reasons(base, candidate))
    return GateDecision(not reasons, tuple(reasons))
```

`_training_reasons` requires candidate win rate strictly above `Decimal("0.558")`, at least three early fills over at least two years, no positive-to-negative increase, P/L ratio at least `3`, and positive 2019/2020/2021 returns.

CLI:

```powershell
python -m cross_signal_strategy.research.late_veto_early_pre_macd_gate path\to\pair.json
```

Print `PASS` or `FAIL` plus all reasons. Invalid evidence/configuration prints `INVALID` and exits `2`.

- [ ] **Step 6: Run focused tests and commit**

```powershell
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_gate.py -q
git add -- cross_signal_strategy/research/late_veto_early_pre_macd_gate.py tests/test_cross_signal_late_veto_early_pre_macd_gate.py
git commit -m "research(cross-signal): enforce paired candidate gates"
```

Expected: tests pass before the commit.

---

### Task 2: Encode the stricter gate in research governance

**Files:**
- Modify: `cross_signal_strategy/docs/research_budget.json:735-772`
- Modify: `cross_signal_strategy/docs/research_budget.md:82-99`
- Modify: `tests/test_cross_signal_research_budget.py:437-483`

**Interfaces:**
- Consumes: the approved spec and Task 1 semantics.
- Produces: machine-readable pre-registration metadata matching `evaluate_pair`.

- [ ] **Step 1: Tighten the budget test first**

```python
gate = raw["official_gate"]
assert gate["minimum_formal_return_retention"] == 1.0
assert gate["training_minimum_win_rate_improvement_percentage_points"] == 3.0
assert gate["full_period_minimum_win_rate_improvement_percentage_points"] == 3.0
assert gate["all_windows_return_must_not_worsen"] is True
assert gate["all_windows_drawdown_must_not_worsen"] is True
assert gate["all_windows_win_rate_must_not_worsen"] is True
assert gate["double_friction_must_not_worsen"] is True
assert gate["paired_run_configuration_required"] is True
assert gate["raw_precision_required"] is True
assert gate["unclosed_positions_excluded_from_win_rate"] is True
assert gate["validation_tuning_forbidden"] is True
```

- [ ] **Step 2: Verify red state**

```powershell
python -m pytest tests/test_cross_signal_research_budget.py::test_stacked_late_veto_early_pre_macd_candidate_is_frozen_pending_joinquant -q
```

Expected: failure because JSON still contains retention `0.95` and lacks new fields.

- [ ] **Step 3: Update only this candidate's `official_gate`**

Keep `blocked`, `max_new_experiments: 0`, `candidate_variants: 1`, all rule fields, and `prohibit_alternatives: true`. Set retention to `1.0` and add Step 1 fields. Do not change other families or the failed-experiment count before a real outcome.

- [ ] **Step 4: Update Markdown governance**

Document +3 percentage points for training/full, per-window return/DD/win-rate nondegradation, nominal-before-double-before-validation ordering, raw paired evidence, and stop-on-first-failure with no alternatives.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest tests/test_cross_signal_research_budget.py -q
git add -- cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md tests/test_cross_signal_research_budget.py
git commit -m "docs(cross-signal): freeze strict win-rate protocol"
```

Expected: candidate-gate assertions pass. Report unrelated stale experiment-count failures without modifying them.

---

### Task 3: Strengthen frozen-candidate characterization

**Files:**
- Modify: `tests/test_cross_signal_late_veto_early_pre_macd_candidate.py:18-145`
- Do not modify: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate.py`

**Interfaces:**
- Consumes: existing frozen helpers.
- Produces: regression evidence for identity, near misses, capacity ordering, sizing, and T-1 use.

- [ ] **Step 1: Add identity and late-veto near-miss tests**

```python
def test_candidate_identity_remains_frozen():
    candidate = candidate_module()
    assert candidate.STRATEGY_VERSION == "cross-v0.3.3-late-veto-early-pre-macd-candidate"
    assert candidate.DEPLOYMENT_BUILD_ID == "20260822.3-candidate"
    assert candidate.business_config_fingerprint() == "f6b08195dd3d"


@pytest.mark.parametrize(
    "overrides",
    [
        {"macd_cross_up": False, "macd_cross_up_age": None},
        {"macd_cross_up_age": 1},
        {"rsi6_cross_rsi12_up_age": 0},
        {"rsi6_cross_rsi12_up_age": 3},
        {"kdj_k_cross_up_age": 0},
        {"kdj_k_cross_up_age": 3},
        {"close": 2.5999},
    ],
)
def test_late_veto_requires_every_condition(overrides):
    late = eligible_score(
        buy_score=84, close=2.60, macd_cross_up=True, macd_cross_up_age=0,
        rsi6_cross_rsi12_up_age=2, kdj_k_cross_up_age=1,
    )
    late.update(overrides)
    assert candidate_module().is_late_macd_boll_upper_entry(late) is False
```

- [ ] **Step 2: Add capacity, sizing, and causal-boundary tests**

Create three primary entries plus one early entry and assert `queue[:3]` is entirely primary. Prove `entry_channel` does not alter sizing:

```python
def test_entry_channel_does_not_change_position_sizing():
    candidate = candidate_module()
    score = eligible_score(buy_score=55, volume_score=6)
    primary = dict(score, entry_channel="primary")
    early = dict(score, entry_channel="early_pre_macd")
    assert candidate.calc_stress_adjusted_buy_target_value(
        20000, primary, current_date="2021-01-04", atr_stop_history=[], trade_days=[]
    ) == candidate.calc_stress_adjusted_buy_target_value(
        20000, early, current_date="2021-01-04", atr_stop_history=[], trade_days=[]
    )
```

Use `inspect.getsource(candidate.manage_holdings)` to require `calc_cross_signal_score(code, prev_date` and reject `calc_cross_signal_score(code, today`.

- [ ] **Step 3: Run characterization and verify zero candidate diff**

```powershell
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_candidate.py -q
git diff --exit-code -- cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate.py
```

Expected: tests pass and the candidate file has no diff. A failure stops implementation for investigation; do not repair the frozen source silently.

- [ ] **Step 4: Commit tests**

```powershell
git add -- tests/test_cross_signal_late_veto_early_pre_macd_candidate.py
git commit -m "test(cross-signal): lock frozen stacked candidate"
```

---

### Task 4: Add the JoinQuant runbook and evidence template

**Files:**
- Create: `cross_signal_strategy/docs/late_veto_early_pre_macd_joinquant_runbook.md`
- Create: `cross_signal_strategy/reports/templates/late_veto_early_pre_macd_pair.json`
- Create: `tests/test_cross_signal_late_veto_early_pre_macd_runbook.py`

**Interfaces:**
- Consumes: Task 1 schema/CLI.
- Produces: a copyable evidence template and ordered execution checklist.

- [ ] **Step 1: Write a red documentation contract test**

Read the runbook/template and assert all seven `GateKind` values, both fingerprints, cash `20000`, time `09:35`, nominal-first/doubled-friction sequencing, stop-on-first-failure, and evaluator command are present. Load the example template through `load_paired_run` and evaluate it as structurally valid.

```powershell
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_runbook.py -q
```

Expected: failure because the artifacts do not exist.

- [ ] **Step 2: Write the exact run sequence**

1. `training_nominal`;
2. after pass, `training_double_friction`;
3. `validation_2022_2023`;
4. `validation_2024_latest` with one shared cutoff;
5. `validation_2015_2018`;
6. `validation_2010_2014`;
7. `full_period` with the same latest cutoff.

For each pair require exact local source, startup identity log, complete logs/transactions, raw metrics, event/fill/year counts, and hashes:

```powershell
$evidenceRoot = 'D:\test\select_stocks\cross_signal_strategy\reports\evidence\late_veto_early_pre_macd\training_nominal'
Get-FileHash -Algorithm SHA256 -LiteralPath "$evidenceRoot\formal.log"
Get-FileHash -Algorithm SHA256 -LiteralPath "$evidenceRoot\formal-trades.csv"
Get-FileHash -Algorithm SHA256 -LiteralPath "$evidenceRoot\candidate.log"
Get-FileHash -Algorithm SHA256 -LiteralPath "$evidenceRoot\candidate-trades.csv"
```

For later stages, use the same directory layout with the exact `GateKind` value as the final directory. Mismatched configuration, missing evidence, or `INVALID` requires a paired rerun.

- [ ] **Step 3: Create one complete example JSON pair**

Use `example_only: true` and a structurally valid synthetic `training_nominal` pair containing every config/metric field. The runbook requires copying it to a result file and removing `example_only` before authoritative evaluation.

- [ ] **Step 4: Run tests and commit**

```powershell
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_gate.py tests/test_cross_signal_late_veto_early_pre_macd_runbook.py -q
git add -- cross_signal_strategy/docs/late_veto_early_pre_macd_joinquant_runbook.md cross_signal_strategy/reports/templates/late_veto_early_pre_macd_pair.json tests/test_cross_signal_late_veto_early_pre_macd_runbook.py
git commit -m "docs(cross-signal): add paired JoinQuant runbook"
```

Expected: all tests pass.

---

### Task 5: Run scoped local verification

**Files:**
- Verification only; no source modification expected.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: separate static, focused, full-suite, local-data, and JoinQuant status.

- [ ] **Step 1: Re-read scope/control-flow skills and inspect the diff**

Map every hunk to gate enforcement, characterization, or runbook evidence. Confirm no source hunk changes candidate/formal/PTrade/sell/sizing/T-1/ETF-pool behavior.

- [ ] **Step 2: Run static and focused checks**

```powershell
python -m py_compile cross_signal_strategy/research/late_veto_early_pre_macd_gate.py tests/test_cross_signal_late_veto_early_pre_macd_gate.py tests/test_cross_signal_late_veto_early_pre_macd_candidate.py tests/test_cross_signal_late_veto_early_pre_macd_runbook.py
git diff --check
python -m pytest tests/test_cross_signal_late_veto_early_pre_macd_gate.py tests/test_cross_signal_late_veto_early_pre_macd_candidate.py tests/test_cross_signal_research_budget.py tests/test_cross_signal_late_veto_early_pre_macd_runbook.py -q
```

Expected: new/candidate-specific tests pass. Record any known stale count failure separately.

- [ ] **Step 3: Run release verification and classify failures**

```powershell
python cross_signal_strategy/tools/verify_release.py --run-tests
```

Expected locally: identity/static checks pass; approved `G:`-dependent tests remain blocked when roots are absent, and old hardcoded count `65` assertions may fail. Record exact failures; do not call the full suite passing.

- [ ] **Step 4: Confirm scope**

```powershell
git status --short
git diff --name-only HEAD~3..HEAD
```

Expected implementation changes are limited to evaluator/tests, the candidate characterization test, budget JSON/Markdown/test, and runbook/template/test.

---

### Task 6: Execute the authoritative training pair

**Files:**
- Create after real runs: `cross_signal_strategy/reports/late_veto_early_pre_macd_training_nominal_pair.json`
- Create after real runs: `cross_signal_strategy/reports/late_veto_early_pre_macd_2019_2021_joinquant.md`
- Failure-only modify: `cross_signal_strategy/docs/failed_experiments.md`, `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`

**Interfaces:**
- Consumes: authenticated JoinQuant, formal/candidate source, runbook, evaluator.
- Produces: one authoritative training decision. No later window runs unless it passes.

- [ ] **Step 1: Run the nominal formal/candidate pair**

Use 2019-01-01 through 2021-12-31, CNY 20,000, daily frequency, identical benchmark/cost/slippage/platform settings, and verify startup identities. If an authenticated JoinQuant session is unavailable, stop with that exact blocker; do not substitute local or PTrade returns.

- [ ] **Step 2: Export, hash, populate, and evaluate evidence**

Save complete logs/transactions outside immutable market roots, hash them, count late/early events, actual early fills/years, and positive-to-negative round trips. Run:

```powershell
python -m cross_signal_strategy.research.late_veto_early_pre_macd_gate cross_signal_strategy/reports/late_veto_early_pre_macd_training_nominal_pair.json
```

`PASS` continues; `FAIL` closes the candidate; `INVALID` reruns the invalid pair without changing rules.

- [ ] **Step 3A: On training failure, record one rejection and stop**

Append one failed-experiment record with identity, paired metrics, failed gates, early sample, and evidence-backed interpretation. Set only this family to `exhausted`, `joinquant_status: rejected`, add evidence paths, and increment the expected failed count once. Run candidate/budget tests and commit:

```powershell
git add -- cross_signal_strategy/reports/late_veto_early_pre_macd_training_nominal_pair.json cross_signal_strategy/reports/late_veto_early_pre_macd_2019_2021_joinquant.md cross_signal_strategy/docs/failed_experiments.md cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md tests/test_cross_signal_research_budget.py
git commit -m "research(cross-signal): reject stacked entry candidate"
```

Formal/PTrade remain unchanged.

- [ ] **Step 3B: On training pass, run doubled friction**

Create/evaluate `cross_signal_strategy/reports/late_veto_early_pre_macd_training_double_friction_pair.json`. A failure follows Step 3A. A pass is committed before validation:

```powershell
git add -- cross_signal_strategy/reports/late_veto_early_pre_macd_training_nominal_pair.json cross_signal_strategy/reports/late_veto_early_pre_macd_training_double_friction_pair.json cross_signal_strategy/reports/late_veto_early_pre_macd_2019_2021_joinquant.md
git commit -m "research(cross-signal): pass stacked candidate training gates"
```

---

### Task 7: Execute validation and final-full-period gates after training passes

**Files:**
- Create: paired JSON/evidence section for each remaining `GateKind`.
- Modify after decision: this candidate family and adopted/failed decision record.
- Do not modify: formal JoinQuant or PTrade strategy files.

**Interfaces:**
- Consumes: passed nominal/doubled training evidence.
- Produces: first frozen-window rejection or a complete passing evidence set plus a separate adoption proposal.

- [ ] **Step 1: Run validations strictly in order**

Run/evaluate `validation_2022_2023`, `validation_2024_latest`, `validation_2015_2018`, and `validation_2010_2014`. Stop on first `FAIL`; rerun an `INVALID` pair with corrected evidence/configuration.

- [ ] **Step 2: Record a validation rejection without tuning**

Append one candidate-level failure naming the first failed window and prior passes. Set this family to `exhausted/rejected`, increment expected failed count once for the whole candidate, commit evidence/governance, and stop without inspecting later windows.

- [ ] **Step 3: Run the final full-period pair**

Only after all four validations pass, use shared dates ending at the `validation_2024_latest` cutoff. Require return non-worse, drawdown non-worse, and win-rate delta `>=0.03`.

- [ ] **Step 4: Produce and commit the final decision**

On failure follow Step 2. On pass set evidence status `all_gates_passed_pending_adoption`, preserve `validation_influence: none`, and report every raw paired metric/hash. Commit:

```powershell
git commit -m "research(cross-signal): complete stacked candidate validation"
```

Do not merge into formal/PTrade. A passing candidate requires a separate adoption design and user confirmation.
