# Binance Spot M1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the isolated deterministic foundation for the Binance Spot dual-baseline project: frozen configuration, numeric and canonical serialization protocols, immutable domain contracts, semantic dependency evidence, and synthetic candidate/run fingerprints.

**Architecture:** Add a new top-level `binance_spot_strategy` package with no runtime dependency on `cross_signal_strategy`. Protocol modules are pure and deterministic, domain objects are immutable and fail closed, and fingerprints accept only strict versioned manifests. M1 uses synthetic checked-in fixtures and current-runtime evidence; it does not load market data or implement either trading baseline.

**Tech Stack:** CPython 3.13.6, standard-library `unittest`, Decimal 1.70/libmpdec 4.0.0, NumPy 2.2.6, pandas 3.0.1, SHA-256, UTF-8 canonical JSON.

## File Structure

```text
binance_spot_strategy/
├── __init__.py
├── README.md
├── requirements.lock.txt
├── config/
│   ├── __init__.py
│   ├── frozen.py
│   └── semantic_dependencies.lock.json
├── protocols/
│   ├── __init__.py
│   ├── canonical_json_v1.py
│   ├── float64_v1.py
│   └── numeric_v1.py
├── domain/
│   ├── __init__.py
│   └── models.py
└── identity/
    ├── __init__.py
    ├── dependency_lock_v1.py
    ├── digests.py
    └── manifests_v1.py

tests/binance_spot/
├── __init__.py
├── fixtures/
│   ├── canonical_records_v1.json
│   ├── identity_manifests_v1.json
│   └── numeric_protocol_v1.json
├── test_canonical_json.py
├── test_config_and_isolation.py
├── test_dependency_lock.py
├── test_domain_contracts.py
├── test_fingerprints.py
└── test_numeric_protocol.py
```

Do not create empty data, indicator, strategy, risk, backtest, paper, state, reporting, or entrypoint packages during M1.

## Global Constraints

- Work only in `D:/test/select_stocks/.worktrees/binance-spot-m1` on branch `codex/binance-spot-m1`.
- Do not read any G-drive market-data root, validation/holdout observations, Binance endpoint, or external network resource.
- Do not modify or runtime-import `cross_signal_strategy`; its pinned source remains evidence only.
- M1 implements no indicator, signal rule, sizing, fee/slippage execution, loader, SQLite state, matching, backtest, paper broker, activation token, formal audit workflow, or real order path.
- The approved design identity is SHA-256 of its Git LF-content bytes: `18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0`.
- Every digest is lowercase 64-character SHA-256 hexadecimal. Tracked text uses UTF-8 Git/LF bytes; runtime artifacts use raw bytes; logical paths are repository-relative POSIX paths.
- Freeze CPython 3.13.6, Decimal 1.70/libmpdec 4.0.0, Unicode database 15.1.0, NumPy 2.2.6, pandas 3.0.1, python-dateutil 2.9.0.post0, six 1.17.0, and tzdata 2025.3.
- `numeric_protocol_v1` uses a new local Decimal context with precision 50, `ROUND_HALF_EVEN`, `Emin=-999999`, `Emax=999999`, `capitals=1`, `clamp=0`, and traps for `DivisionByZero`, `FloatOperation`, `InvalidOperation`, and `Overflow` only.
- Exchange decimal grammar is `-?\d+(\.\d{1,18})?`; reject whitespace, leading `+`, exponent notation, locale separators, and every scale above 18 even when excess digits are zero.
- Q18 has exactly 18 fractional digits, normalizes negative zero to positive zero, and positive grid reductions alone use exact integer step ticks with floor semantics.
- Canonical JSON uses UTF-8 without BOM, NFC-normalized strings and keys, lexicographically sorted object keys, schema-preserved array order, and no insignificant whitespace. Reject `None`/JSON null, raw `float`, raw `Decimal`, bytes, sets, NaN, Infinity, scientific decimal notation, and NFC key collisions.
- Canonical timestamps are UTC RFC3339 with `Z`; omit fractions at whole seconds and otherwise require exactly millisecond precision. Scope IDs are strongly typed so training, reserved validation/holdout, and paper IDs cannot cross modes.
- Historical stages are half-open: training `[2018-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`, validation `[2022-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`, holdout `[2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z)`, with exactly 540 prior closed four-hour warm-up bars and formal starting balance 500.00 USDT.
- Write each behavioral test before production code, run it, and record the expected RED failure. Write only the minimum GREEN implementation, then run all prior M1 tests before committing.
- Use exactly `D:/Programs/Python/Python313/python.exe -B -m unittest`; do not call bare `python` and do not install pytest. Existing pytest-style repository tests remain explicitly unrun because pytest is absent.
- Commit each reviewed task separately with its listed focused message. Do not mix later Binance milestones or unrelated user files into these commits.

---

### Task 1: Isolated Package And Frozen Common Configuration

**Files:**
- Create: `binance_spot_strategy/__init__.py`
- Create: `binance_spot_strategy/README.md`
- Create: `binance_spot_strategy/requirements.lock.txt`
- Create: `binance_spot_strategy/config/__init__.py`
- Create: `binance_spot_strategy/config/frozen.py`
- Create: `tests/binance_spot/__init__.py`
- Create: `tests/binance_spot/test_config_and_isolation.py`

**Interfaces:**
- Produces: `StageName(StrEnum)` with `TRAINING`, `VALIDATION`, and `HOLDOUT`.
- Produces: immutable `StageWindow(name, start, end, warmup_bars=540)` and `admits(open_time, close_time) -> bool`.
- Produces: `SYMBOLS`, `BAR_INTERVAL`, `FORMAL_STARTING_BALANCE`, `STAGE_WINDOWS`, `SLIPPAGE_RATE`, `COMMISSION_RATE`, `RISK_BUDGET`, `ALLOCATION_CAP`, `ATR_MULTIPLIER`, `STOP_FLOOR`, `STOP_CAP`, and `DESIGN_REVISION_SHA256`.
- Produces: exact dependency pins in `requirements.lock.txt`; runtime verification is Task 5.

- [ ] **Step 1: Write failing isolation and configuration tests**

Create `tests/binance_spot/test_config_and_isolation.py` with standard-library `unittest`. It must import `binance_spot_strategy`, capture newly introduced `sys.modules`, and assert no introduced module equals or starts with `cross_signal_strategy`. Assert these literal values:

```python
self.assertEqual(frozen.SYMBOLS, ("BTCUSDT", "ETHUSDT"))
self.assertEqual(frozen.BAR_INTERVAL, "4h")
self.assertEqual(frozen.FORMAL_STARTING_BALANCE, Decimal("500.00"))
self.assertEqual(frozen.SLIPPAGE_RATE, Decimal("0.0005"))
self.assertEqual(frozen.COMMISSION_RATE, Decimal("0.001"))
self.assertEqual(frozen.RISK_BUDGET, Decimal("0.01"))
self.assertEqual(frozen.ALLOCATION_CAP, Decimal("0.30"))
self.assertEqual(frozen.ATR_MULTIPLIER, Decimal("2.5"))
self.assertEqual(frozen.STOP_FLOOR, Decimal("0.05"))
self.assertEqual(frozen.STOP_CAP, Decimal("0.15"))
self.assertEqual(
    frozen.DESIGN_REVISION_SHA256,
    "18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0",
)
```

Assert the three ordered stage windows and 540-bar warm-up. Assert a final 2021 training bar ending `2021-12-31T23:59:59.999Z` is admitted while a bar opening exactly at `2022-01-01T00:00:00Z` is excluded. Assert naive UTC, reversed boundaries, and a warm-up count other than 540 raise `ValueError`. Assert mutation raises `FrozenInstanceError`.

- [ ] **Step 2: Run Task 1 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_config_and_isolation.py" -v
```

Expected: import error for missing `binance_spot_strategy`, not a discovery or syntax error.

- [ ] **Step 3: Implement minimal config**

Implement `StageWindow` with this exact boundary behavior:

```python
@dataclass(frozen=True, slots=True)
class StageWindow:
    name: StageName
    start: datetime
    end: datetime
    warmup_bars: int = 540

    def __post_init__(self) -> None:
        _require_utc(self.start, "start")
        _require_utc(self.end, "end")
        if self.end <= self.start:
            raise ValueError("stage end must be after start")
        if self.warmup_bars != 540:
            raise ValueError("warmup_bars must equal 540")

    def admits(self, open_time: datetime, close_time: datetime) -> bool:
        _require_utc(open_time, "open_time")
        _require_utc(close_time, "close_time")
        if close_time < open_time:
            raise ValueError("close_time must not precede open_time")
        return self.start <= open_time < self.end and close_time < self.end
```

`_require_utc` requires `tzinfo is timezone.utc`. Define the exact constants and windows from Global Constraints. Re-export public config names only.

Create `requirements.lock.txt` with UTF-8 LF bytes exactly:

```text
numpy==2.2.6
pandas==3.0.1
python-dateutil==2.9.0.post0
six==1.17.0
tzdata==2025.3
```

`README.md` states that M1 is synthetic foundation code only, formal candidate fingerprints do not exist yet, and data/strategy/execution/paper/state modules are deliberately absent.

- [ ] **Step 4: Run focused GREEN and all M1 tests**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_config_and_isolation.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
```

- [ ] **Step 5: Commit Task 1**

```powershell
git add binance_spot_strategy tests/binance_spot/__init__.py tests/binance_spot/test_config_and_isolation.py
git commit -m "feat(binance-spot): freeze isolated M1 configuration"
```

### Task 2: Numeric Protocol V1 And Float64 Bridge

**Files:**
- Create: `binance_spot_strategy/protocols/__init__.py`
- Create: `binance_spot_strategy/protocols/numeric_v1.py`
- Create: `binance_spot_strategy/protocols/float64_v1.py`
- Create: `tests/binance_spot/fixtures/numeric_protocol_v1.json`
- Create: `tests/binance_spot/test_numeric_protocol.py`

**Interfaces:**
- Produces: `NumericProtocolError(ValueError)`.
- Produces: immutable `Q18(value: Decimal)` with exponent `-18`.
- Produces: `numeric_context()`, `parse_finite_decimal()`, `quantize_q18()`, `parse_canonical_q18()`, `format_q18()`, and `floor_positive_to_step()`.
- Produces: `IndicatorNonfiniteTag`, `MetricNonfiniteTag`, and `canonicalize_float64()`.

- [ ] **Step 1: Add literal vectors and failing tests**

Create `numeric_protocol_v1.json`:

```json
{
  "schema_version": "numeric_protocol_vectors_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "q18_cases": [
    {"input": "1.0000000000000000005", "expected": "1.000000000000000000"},
    {"input": "1.0000000000000000015", "expected": "1.000000000000000002"},
    {"input": "-1.0000000000000000005", "expected": "-1.000000000000000000"},
    {"input": "-0.0000000000000000004", "expected": "0.000000000000000000"}
  ],
  "floor_cases": [
    {"value": "1.239", "step": "0.01", "expected": "1.230000000000000000"},
    {"value": "1.23", "step": "0.01", "expected": "1.230000000000000000"},
    {"value": "1.234567899999999999", "step": "0.000001", "expected": "1.234567000000000000"}
  ],
  "finite_float_cases": [
    {"input": 0.1, "expected": "0.100000000000000000"},
    {"input": 0.30000000000000004, "expected": "0.300000000000000040"},
    {"input": -0.0, "expected": "0.000000000000000000"}
  ]
}
```

Create `test_numeric_protocol.py`. Load the fixture relative to `__file__`; expected values remain fixture literals. Cover:

```python
self.assertEqual(
    parse_finite_decimal("1.230000000000000001"),
    Decimal("1.230000000000000001"),
)
for invalid in (
    "1.2300000000000000000", " 1.2", "+1.2", "1e-8", "1,2", "1."
):
    with self.subTest(invalid=invalid):
        with self.assertRaises(NumericProtocolError):
            parse_finite_decimal(invalid)

self.assertEqual(
    format_q18(parse_canonical_q18("1.230000000000000000")),
    "1.230000000000000000",
)
for invalid in ("1.23", "01.230000000000000000", "-0.000000000000000000"):
    with self.subTest(invalid=invalid):
        with self.assertRaises(NumericProtocolError):
            parse_canonical_q18(invalid)
```

Loop over all Q18, floor, and finite-float fixture cases. Assert nonpositive and scale-above-18 floor operands fail. Assert `quantize_q18(0.1)` and `floor_positive_to_step(1.0, Decimal("0.01"))` raise `TypeError`.

Run eight calls in a four-worker `ThreadPoolExecutor`; each worker enters caller precision 3 and `ROUND_DOWN`, yet the positive half-even tie must always return `1.000000000000000002`.

Assert exact tag values:

```python
("FLOAT64:NAN", "FLOAT64:POSITIVE_INFINITY", "FLOAT64:NEGATIVE_INFINITY")
("NONFINITE:NO_OBSERVATIONS", "NONFINITE:ZERO_DENOMINATOR", "NONFINITE:INVALID_INPUT")
```

- [ ] **Step 2: Run Task 2 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_numeric_protocol.py" -v
```

Expected: missing `binance_spot_strategy.protocols`; fixture loading must succeed.

- [ ] **Step 3: Implement numeric protocol**

Use:

```python
Q18_QUANTUM = Decimal("0.000000000000000001")
EXCHANGE_DECIMAL_RE = re.compile(r"-?\d+(?:\.\d{1,18})?\Z")
CANONICAL_Q18_RE = re.compile(r"-?(?:0|[1-9]\d*)\.\d{18}\Z")
```

`numeric_context()` constructs a new `Context` with the exact Global Constraints and explicitly sets all non-approved traps false. `Q18` accepts only finite `Decimal`, exponent `-18`, and non-negative zero. `parse_finite_decimal` validates textual scale before Decimal construction and never rounds. `quantize_q18` uses `localcontext(numeric_context())`, catches `DecimalException`, and raises `NumericProtocolError`. `parse_canonical_q18` rejects negative zero. `format_q18` accepts only `Q18`.

For grid floor, reject non-Decimal, nonfinite, nonpositive, or scale-above-18 operands. Align Decimal coefficient/exponent pairs to a common exponent, use Python integer `//`, reconstruct an exact Decimal tuple, then call `quantize_q18`; do not divide Decimal values.

`float64_v1.py` is the only protocol module importing NumPy. Accept exactly Python `float` and NumPy `float64`. Finite values use `repr(float(value))`, then Decimal and Q18. Nonfinite values map to indicator tags and never enter arithmetic.

- [ ] **Step 4: Run focused GREEN and prior tests**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_numeric_protocol.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
```

- [ ] **Step 5: Commit Task 2**

```powershell
git add binance_spot_strategy/protocols tests/binance_spot/fixtures/numeric_protocol_v1.json tests/binance_spot/test_numeric_protocol.py
git commit -m "feat(binance-spot): add numeric protocol v1"
```

### Task 3: Canonical JSON V1

**Files:**
- Create: `binance_spot_strategy/protocols/canonical_json_v1.py`
- Modify: `binance_spot_strategy/protocols/__init__.py`
- Create: `tests/binance_spot/fixtures/canonical_records_v1.json`
- Create: `tests/binance_spot/test_canonical_json.py`

**Interfaces:**
- Produces: `CanonicalJsonError(ValueError)`.
- Produces: `canonical_json_bytes(payload) -> bytes`.
- Produces: `canonical_hashed_payload_bytes(payload) -> bytes`, requiring nonempty `schema_version` and `numeric_protocol_version == "numeric_protocol_v1"`.
- Consumes: `Q18`; raw `Decimal` remains forbidden.

- [ ] **Step 1: Add canonical golden records and failing tests**

Create `canonical_records_v1.json`:

```json
{
  "schema_version": "canonical_records_fixture_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "primary_expected_utf8": "{\"amount\":\"1.005000000000000000\",\"items\":[\"ETHUSDT\",\"BTCUSDT\"],\"name\":\"Café\",\"numeric_protocol_version\":\"numeric_protocol_v1\",\"schema_version\":\"test_v1\"}",
  "primary_expected_sha256": "af8caff70bb766eeb44a352274b84f9818442374281722c947acdf1b87e82745",
  "unicode_expected_utf8": "{\"a\":\"1.000000000000000000\",\"numeric_protocol_version\":\"numeric_protocol_v1\",\"schema_version\":\"golden_v1\",\"z\":[\"ETHUSDT\",\"BTCUSDT\"],\"é\":\"Café\"}",
  "unicode_expected_sha256": "679c36854682695f528e450a90e9b9921662825653ce86fc35d5bd0894804279"
}
```

Create `test_canonical_json.py`. Construct the primary payload with `quantize_q18(Decimal("1.005"))`, decomposed `"Cafe\u0301"`, and array `["ETHUSDT", "BTCUSDT"]`; compare exact bytes and independently call `hashlib.sha256(actual).hexdigest()` against the fixture. Construct the Unicode payload with decomposed key `"e\u0301"` and value, and assert its exact bytes/hash.

Add one test per rejection:

```python
for value in (None, 0.1, Decimal("1.0"), b"bytes", {"set"}):
    with self.subTest(value=value):
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({
                "schema_version": "test_v1",
                "numeric_protocol_version": "numeric_protocol_v1",
                "value": value,
            })
```

Also assert:
- keys `"é"` and `"e\u0301"` collide after NFC and fail;
- a non-string key fails;
- an unsupported object fails rather than using `default=str`;
- arrays `["ETHUSDT", "BTCUSDT"]` and reversed order produce different bytes;
- missing/empty schema version or wrong numeric protocol fails in `canonical_hashed_payload_bytes`;
- emitted bytes contain no BOM or newline.

- [ ] **Step 2: Run Task 3 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_canonical_json.py" -v
```

Expected: missing `canonical_json_v1`.

- [ ] **Step 3: Implement canonical traversal and encoding**

Normalize recursively into only `str`, `int`, `bool`, list, and ordered dict JSON primitives. Check `bool` before `int`. Convert `Q18` with `format_q18`. Accept tuple as a schema-ordered array. For mappings, require string keys, NFC-normalize before collision detection, recursively normalize values, and order normalized keys lexicographically. Reject every other type.

Encode only through:

```python
json.dumps(
    normalized,
    ensure_ascii=False,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
```

`canonical_hashed_payload_bytes` validates the two top-level versions before delegating. It does not sort arrays or remove hash fields.

- [ ] **Step 4: Run focused GREEN and prior tests**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_canonical_json.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add binance_spot_strategy/protocols tests/binance_spot/fixtures/canonical_records_v1.json tests/binance_spot/test_canonical_json.py
git commit -m "feat(binance-spot): add canonical json v1"
```

### Task 4: Immutable Domain Contracts And DecisionKey

**Files:**
- Create: `binance_spot_strategy/domain/__init__.py`
- Create: `binance_spot_strategy/domain/models.py`
- Create: `tests/binance_spot/test_domain_contracts.py`

**Interfaces:**
- Produces: `Symbol`, `BarInterval`, `Side`, `OrderState`, `ScopeType`.
- Produces: typed `TrainingStageId`, `StageActivationId`, and `PaperObservationEpochId`.
- Produces: `canonical_utc_timestamp(datetime) -> str`.
- Produces: immutable `DecisionKey`, `Bar`, `SignalIntent`, `DecisionPlan`, `ExecutionGroup`, `OrderIntent`, `Fill`, `Position`, and `EquitySnapshot`.

- [ ] **Step 1: Write failing domain tests**

Use these exact enums:

```python
Symbol.BTCUSDT.value == "BTCUSDT"
Symbol.ETHUSDT.value == "ETHUSDT"
BarInterval.FOUR_HOURS.value == "4h"
Side.BUY.value == "buy"
Side.SELL.value == "sell"
OrderState.PENDING.value == "pending"
OrderState.BLOCKED.value == "blocked"
ScopeType.TRAINING_STAGE.value == "training_stage"
ScopeType.RESERVED_STAGE.value == "reserved_stage"
ScopeType.PAPER_EPOCH.value == "paper_epoch"
```

Build a valid synthetic `Bar`; assert identity is exactly `(Symbol.BTCUSDT, BarInterval.FOUR_HOURS, open_time)`. Assert naive time, non-Decimal price, nonpositive price, negative volume, invalid OHLC envelope, or close not after open fails.

Build three valid keys:

```python
DecisionKey(run_id, ScopeType.TRAINING_STAGE, TrainingStageId("train-1"), candidate, run, close)
DecisionKey(run_id, ScopeType.RESERVED_STAGE, StageActivationId("activation-1"), candidate, run, close)
DecisionKey(run_id, ScopeType.PAPER_EPOCH, PaperObservationEpochId("epoch-1"), candidate, run, close)
```

Assert they are immutable, hashable, and pairwise unequal even when all typed ID `.value` strings are `"same"`. Assert every scope/type mismatch fails. Candidate and run fingerprints must reject uppercase, short, or nonhex text.

Assert canonical UTC formatting:

```python
canonical_utc_timestamp(datetime(2026, 8, 1, tzinfo=timezone.utc))
    == "2026-08-01T00:00:00Z"
canonical_utc_timestamp(datetime(2026, 8, 1, 3, 59, 59, 999000, tzinfo=timezone.utc))
    == "2026-08-01T03:59:59.999Z"
```

Reject non-millisecond microseconds.

For `DecisionPlan`, test exactly four valid shapes:
- hold: `sell=None`, `buy=None`;
- one sell;
- one buy with positive `risk_atr`;
- replacement: sell then a different-symbol buy with positive `risk_atr`.

Reject wrong intent side in each slot, same-symbol replacement, mismatched bar-close times, and buy without positive ATR.

For execution, assert:
- standalone pending buy or sell with no dependency is valid;
- replacement has pending sell parent then blocked buy child whose `depends_on_intent_id` equals the parent;
- two independent pending orders, blocked sell, reversed order, wrong group ID, or wrong dependency fails.

Construct and validate positive `Fill`, `Position`, and nonnegative `EquitySnapshot`; mutate any frozen object and expect `FrozenInstanceError`.

- [ ] **Step 2: Run Task 4 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_domain_contracts.py" -v
```

Expected: missing `binance_spot_strategy.domain`.

- [ ] **Step 3: Implement minimal immutable models**

All models use `@dataclass(frozen=True, slots=True)`. Use identifier grammar `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`. Require `tzinfo is timezone.utc` and millisecond-aligned microseconds. Require lowercase 64-hex fingerprints.

Use these fields:

```python
@dataclass(frozen=True, slots=True)
class DecisionKey:
    run_id: str
    scope_type: ScopeType
    scope_id: TrainingStageId | StageActivationId | PaperObservationEpochId
    candidate_strategy_fingerprint: str
    run_fingerprint: str
    bar_close_time: datetime

@dataclass(frozen=True, slots=True)
class Bar:
    symbol: Symbol
    interval: BarInterval
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

@dataclass(frozen=True, slots=True)
class SignalIntent:
    signal_intent_id: str
    symbol: Symbol
    side: Side
    bar_close_time: datetime
    reason_code: str
    risk_atr: Q18 | None = None

@dataclass(frozen=True, slots=True)
class DecisionPlan:
    decision_key: DecisionKey
    execution_group_id: str
    sell: SignalIntent | None = None
    buy: SignalIntent | None = None

@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    execution_group_id: str
    symbol: Symbol
    side: Side
    state: OrderState
    quantity: Q18
    depends_on_intent_id: str | None = None

@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    execution_group_id: str
    order_intents: tuple[OrderIntent, ...]

@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    intent_id: str
    symbol: Symbol
    side: Side
    quantity: Q18
    price: Q18
    commission: Q18
    event_time: datetime

@dataclass(frozen=True, slots=True)
class Position:
    symbol: Symbol
    quantity: Q18
    average_cost: Q18

@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    event_time: datetime
    cash: Q18
    position_value: Q18
    total_equity: Q18
```

`SignalIntent` buy requires strictly positive ATR; sell requires no ATR. Hold is represented only by an empty `DecisionPlan`. `ExecutionGroup` accepts one valid pending standalone or exactly the ordered parent/child pair. Do not implement state transitions, settlement, accounting reconciliation, or persistence.

- [ ] **Step 4: Run focused GREEN and prior tests**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_domain_contracts.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
```

- [ ] **Step 5: Commit Task 4**

```powershell
git add binance_spot_strategy/domain tests/binance_spot/test_domain_contracts.py
git commit -m "feat(binance-spot): add immutable domain contracts"
```

### Task 5: Semantic Dependency Lock And Runtime Verification

**Files:**
- Create: `binance_spot_strategy/config/semantic_dependencies.lock.json`
- Create: `binance_spot_strategy/identity/__init__.py`
- Create: `binance_spot_strategy/identity/digests.py`
- Create: `binance_spot_strategy/identity/dependency_lock_v1.py`
- Create: `tests/binance_spot/test_dependency_lock.py`

**Interfaces:**
- Produces: `sha256_bytes`, `sha256_raw_file`, `sha256_tracked_text`, `require_sha256`, and `hash_canonical_payload`.
- Produces: `DependencyLockError`, `load_semantic_dependency_lock`, `semantic_dependency_lock_hash`, and `verify_current_runtime`.
- Consumes: canonical JSON V1 and numeric context V1.

- [ ] **Step 1: Write failing digest, lock-schema, drift, and current-runtime tests**

Assert:

```python
self.assertEqual(sha256_bytes(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
self.assertEqual(
    semantic_dependency_lock_hash(load_semantic_dependency_lock()),
    "7d7c3124979b3a217b401e1e975fab8d97f837b4e7c9d732cfbeb1ea1cc74ca8",
)
verify_current_runtime(load_semantic_dependency_lock())
```

Mutate a deep copy's `python.version` to `3.13.5` and assert drift reports `python.version`. Add an unknown top-level field and assert schema validation fails. Assert `require_sha256` rejects uppercase, short, and nonhex values. Write a temporary LF text file and a CRLF equivalent and assert `sha256_tracked_text` produces the same digest; assert BOM and invalid UTF-8 fail.

- [ ] **Step 2: Run Task 5 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_dependency_lock.py" -v
```

Expected: missing identity package or dependency-lock module.

- [ ] **Step 3: Add the exact checked-in semantic lock**

Create `semantic_dependencies.lock.json` with this exact data and no null/float fields:

```json
{
  "schema_version": "semantic_dependency_lock_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "design_revision_sha256": "18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0",
  "requirements_lock_sha256": "3da0b934b0aee4c22280756eda4231b4dc374228c50d01cf837f75e084a5f1bd",
  "source_hash_convention": {
    "digest_encoding": "lowercase_hex",
    "logical_path": "repo_relative_posix",
    "runtime_artifact": "sha256_raw_bytes",
    "tracked_source": "sha256_git_blob_bytes"
  },
  "python": {
    "implementation": "CPython",
    "version": "3.13.6",
    "version_info": [3, 13, 6, "final", 0],
    "cache_tag": "cpython-313",
    "compiler": "MSC v.1944 64 bit (AMD64)",
    "build": ["tags/v3.13.6:4e66535", "Aug  6 2025 14:36:00"],
    "byteorder": "little",
    "executable_sha256": "91566dc8bb9a336c36c607ee0d5a5135e54ddce2418e2cd7728a49c8f098904a"
  },
  "float64": {
    "radix": 2,
    "mant_dig": 53,
    "dig": 15,
    "rounds": 1,
    "max_exp": 1024,
    "min_exp": -1021
  },
  "decimal": {
    "module_version": "1.70",
    "libmpdec_version": "4.0.0",
    "have_contextvar": true,
    "extension_sha256": "39b9ad652c4f2df15a8050fa7d90e12f8b335275635396f19f7671f98d2bc833",
    "context": {
      "precision": 50,
      "rounding": "ROUND_HALF_EVEN",
      "emin": -999999,
      "emax": 999999,
      "capitals": 1,
      "clamp": 0,
      "traps": ["DivisionByZero", "FloatOperation", "InvalidOperation", "Overflow"]
    }
  },
  "unicode": {"database_version": "15.1.0", "normalization": "NFC"},
  "distributions": [
    {"name": "numpy", "version": "2.2.6", "record_sha256": "f385f17365978405072ff52b6734cdd45e06cd70978e2aaccd6c8a24cb728521"},
    {"name": "pandas", "version": "3.0.1", "record_sha256": "cf4b8de9d0c4fd7e306efbdef5e6b1a59394f951be16f8ce977097c08ee23788"},
    {"name": "python-dateutil", "version": "2.9.0.post0", "record_sha256": "866badf7a5db5499c367a2fad02112433affdd3e6c43a097665922961b2c6f96"},
    {"name": "six", "version": "1.17.0", "record_sha256": "6de8d68885f918facfbdf65f4d0fec3ab6d22992593ebbff81699e54b6768081"},
    {"name": "tzdata", "version": "2025.3", "record_sha256": "d7cae9d9540cb853c62cfa39f1b7b8e9e0a37fa3135c3cf486f574f2d9e6e280"}
  ]
}
```

- [ ] **Step 4: Implement strict hashing and runtime evidence verification**

`sha256_tracked_text` reads raw bytes, rejects UTF-8 BOM/invalid UTF-8, normalizes CRLF and CR to LF, then hashes UTF-8 bytes. `hash_canonical_payload` uses `canonical_hashed_payload_bytes`. `load_semantic_dependency_lock` parses JSON and validates exact keys at every nested level, exact ordered distribution names, exact protocol versions, and every digest.

`verify_current_runtime` compares all lock fields using `platform`, `sys`, `decimal`, `unicodedata`, `_decimal.__file__`, `sys.executable`, `importlib.metadata`, and the exact `numeric_context()`. Locate each distribution's `.dist-info/RECORD` through `distribution.files`, hash its raw bytes, and compare. Compare `requirements.lock.txt` with tracked-text hashing. Collect all drift paths and raise one deterministic `DependencyLockError`; do not silently update the lock.

- [ ] **Step 5: Run focused GREEN and prior tests**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_dependency_lock.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
```

- [ ] **Step 6: Commit Task 5**

```powershell
git add binance_spot_strategy/config/semantic_dependencies.lock.json binance_spot_strategy/identity tests/binance_spot/test_dependency_lock.py
git commit -m "feat(binance-spot): lock semantic runtime evidence"
```

### Task 6: Strict Candidate And Run Fingerprints

**Files:**
- Create: `binance_spot_strategy/identity/manifests_v1.py`
- Modify: `binance_spot_strategy/identity/__init__.py`
- Create: `tests/binance_spot/fixtures/identity_manifests_v1.json`
- Create: `tests/binance_spot/test_fingerprints.py`
- Modify: `binance_spot_strategy/README.md`

**Interfaces:**
- Produces: frozen `ContractDigest`, `ModuleDigest`, `CandidateManifestV1`, `RunCommonV1`, `HistoricalRunManifestV1`, and `PaperRunManifestV1`.
- Produces: strict `candidate_manifest_from_payload` and `run_manifest_from_payload`.
- Produces: `candidate_strategy_fingerprint` and `run_fingerprint`.
- Does not produce a formal Baseline A or B fingerprint because executable strategy modules do not exist in M1.

- [ ] **Step 1: Add synthetic identity golden manifests and failing tests**

Create `identity_manifests_v1.json` with three payloads. The candidate payload is:

```json
{
  "schema_version": "candidate_manifest_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "design_revision_sha256": "18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0",
  "baseline": {"id": "baseline_a", "semantic_version": "0.1.0"},
  "conventions": {
    "universe": ["BTCUSDT", "ETHUSDT"],
    "bar_interval": "4h",
    "decision_timing": "closed_bar_only",
    "formal_starting_balance": "500.000000000000000000"
  },
  "contracts": [
    {"role": "strategy", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
    {"role": "risk", "sha256": "2222222222222222222222222222222222222222222222222222222222222222"},
    {"role": "cost", "sha256": "3333333333333333333333333333333333333333333333333333333333333333"},
    {"role": "accounting", "sha256": "4444444444444444444444444444444444444444444444444444444444444444"},
    {"role": "metric", "sha256": "5555555555555555555555555555555555555555555555555555555555555555"},
    {"role": "selection", "sha256": "6666666666666666666666666666666666666666666666666666666666666666"}
  ],
  "baseline_strategy_module": {
    "role": "baseline_strategy",
    "logical_name": "binance_spot_strategy.strategies.baseline_a",
    "sha256": "7777777777777777777777777777777777777777777777777777777777777777"
  },
  "shared_modules": [
    {"role": "indicator", "logical_name": "binance_spot_strategy.synthetic.indicator", "sha256": "8888888888888888888888888888888888888888888888888888888888888888"},
    {"role": "strategy_support", "logical_name": "binance_spot_strategy.synthetic.strategy_support", "sha256": "9999999999999999999999999999999999999999999999999999999999999999"},
    {"role": "risk", "logical_name": "binance_spot_strategy.synthetic.risk", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    {"role": "cost", "logical_name": "binance_spot_strategy.synthetic.cost", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
    {"role": "accounting", "logical_name": "binance_spot_strategy.synthetic.accounting", "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
    {"role": "metric", "logical_name": "binance_spot_strategy.synthetic.metric", "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
    {"role": "selection", "logical_name": "binance_spot_strategy.synthetic.selection", "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}
  ],
  "protocols": {
    "canonical_json_version": "canonical_json_v1",
    "canonical_protocol_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "numeric_protocol_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "golden_vectors_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "semantic_dependency_lock_sha256": "7d7c3124979b3a217b401e1e975fab8d97f837b4e7c9d732cfbeb1ea1cc74ca8"
  }
}
```

Its expected fingerprint is:

```text
b5d1981da1427653c76a6cfc3504f15a5bf88b813b34029eb56ab1ec5b2d59ad
```

The historical run payload reuses that candidate fingerprint and has:

```json
{
  "schema_version": "historical_run_manifest_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "candidate_strategy_fingerprint": "b5d1981da1427653c76a6cfc3504f15a5bf88b813b34029eb56ab1ec5b2d59ad",
  "business_artifact_manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "business_source_tree_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "source_commit": "0123456789abcdef0123456789abcdef01234567",
  "dependency_lock_sha256": "7d7c3124979b3a217b401e1e975fab8d97f837b4e7c9d732cfbeb1ea1cc74ca8",
  "runtime_manifest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "broker_adapter": {"role": "broker_adapter", "logical_name": "synthetic_historical_broker", "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
  "stage_manifest_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "execution_manifest_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "environment_contract_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "input": {"kind": "historical", "immutable_raw_data_manifest_sha256": "2222222222222222222222222222222222222222222222222222222222222222"}
}
```

Its expected fingerprint is:

```text
72bd55a8c39767c3444455aa892e41ad80183f7bc3b3d37ce564ef0f95676fdc
```

The paper run payload is:

```json
{
  "schema_version": "paper_run_manifest_v1",
  "numeric_protocol_version": "numeric_protocol_v1",
  "candidate_strategy_fingerprint": "b5d1981da1427653c76a6cfc3504f15a5bf88b813b34029eb56ab1ec5b2d59ad",
  "business_artifact_manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "business_source_tree_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "source_commit": "0123456789abcdef0123456789abcdef01234567",
  "dependency_lock_sha256": "7d7c3124979b3a217b401e1e975fab8d97f837b4e7c9d732cfbeb1ea1cc74ca8",
  "runtime_manifest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "broker_adapter": {"role": "broker_adapter", "logical_name": "synthetic_paper_broker", "sha256": "3333333333333333333333333333333333333333333333333333333333333333"},
  "stage_manifest_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "execution_manifest_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "environment_contract_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "input": {
    "kind": "paper",
    "paper_startup_semantic_hash": "4444444444444444444444444444444444444444444444444444444444444444",
    "feed_contract_sha256": "5555555555555555555555555555555555555555555555555555555555555555"
  }
}
```

Its expected fingerprint is:

```text
5fbb792d5df25200720ba766349afa7a343855c750e20eb164eb90a6f6ef3051
```

Create `test_fingerprints.py`. Load each payload through its strict parser and assert exact expected fingerprint. Assert two calls are identical. Mutation matrix:
- changing a shared semantic module, canonical protocol digest, numeric protocol digest, golden-vector digest, or semantic dependency lock changes candidate and any rebuilt run;
- changing broker adapter, stage, execution, environment, or historical raw-data digest leaves the original candidate unchanged and changes the run;
- historical and paper manifests share the candidate but have different run fingerprints;
- adding `run_id`, `database_id`, `timestamp`, `outcome`, or renderer metadata causes strict parser rejection;
- reversing contracts or shared modules causes schema-order rejection;
- changing only array order in a valid canonical payload is not silently sorted.

- [ ] **Step 2: Run Task 6 and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_fingerprints.py" -v
```

Expected: missing `manifests_v1`.

- [ ] **Step 3: Implement strict manifest types and fingerprint functions**

Contract role order is exactly:

```python
("strategy", "risk", "cost", "accounting", "metric", "selection")
```

Shared module role order is exactly:

```python
("indicator", "strategy_support", "risk", "cost", "accounting", "metric", "selection")
```

Within one role, logical names are NFC and lexicographically ordered. Require every shared role at least once. Require baseline ID `baseline_a` or `baseline_b`, semantic version grammar `\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?`, universe exactly `("BTCUSDT", "ETHUSDT")`, interval `4h`, timing `closed_bar_only`, and starting balance exactly Q18 500.

Use frozen dataclasses. Parsers reject missing and unknown keys at every level, wrong schema version, wrong discriminated run input, noncanonical logical names, wrong array order, and invalid digest/commit text. `to_payload()` returns only business fields shown above. It has no run/database/activation/audit/result/timestamp/renderer field.

`candidate_strategy_fingerprint` and `run_fingerprint` call `hash_canonical_payload(manifest.to_payload())`. Do not expose a caller-selected exclusion list. Do not create a formal fingerprint using absent strategy modules.

- [ ] **Step 4: Run focused GREEN, complete M1 suite, and static checks**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_fingerprints.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git status --short
```

Expected: all M1 tests pass, dependency-lock verification matches the current runtime, no whitespace error exists, and only planned M1 files are changed.

Update `README.md` with the exact focused test command and an explicit deferred-scope list. State that JoinQuant/PTrade/ETF code, market data, validation/holdout, Binance networking, and formal candidate generation were not used.

- [ ] **Step 5: Commit Task 6**

```powershell
git add binance_spot_strategy tests/binance_spot docs/superpowers/plans/2026-08-03-binance-spot-m1-foundation.md
git commit -m "feat(binance-spot): add reproducible identity core"
```

- [ ] **Step 6: Prepare milestone evidence**

Record exact commit range, Python/runtime versions, test count and command, design hash, semantic lock hash, three synthetic golden fingerprints, pytest limitation, and deferred scope. Do not claim the complete Binance strategy, a backtest, training result, validation result, paper qualification, or real-trading readiness.
