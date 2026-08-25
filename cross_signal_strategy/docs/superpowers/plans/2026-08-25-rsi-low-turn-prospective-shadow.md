# RSI Low-Turn Prospective Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, order-free local observer that records newly arriving RSI6 low-turn events and matures fixed-horizon outcome labels without replaying previously inspected research or validation periods.

**Architecture:** Keep signal detection, point-in-time source validation, append-only state, friction/outcome calculation, and statistical gating in separate modules. A small CLI composes them but has no PTrade, JoinQuant scheduling, portfolio, or order interface; it refuses real execution until an explicitly approved append-only source manifest passes the contract.

**Tech Stack:** Python 3, pandas, standard-library dataclasses/hashlib/json/statistics/zoneinfo, pytest, existing `LocalBroker` friction model.

**Spec:** `cross_signal_strategy/docs/superpowers/specs/2026-08-25-rsi-low-turn-prospective-shadow-design.md`

## Global Constraints

- Version is exactly `rsi-low-turn-shadow-v0.1`.
- Event collection starts no earlier than `2026-08-26`; earlier rows are warm-up only.
- Never read the 2019-2021 training root, 2018 warm-up root, any reserved validation root, or `G:\financial\history_data\按年份合并`.
- Never modify the formal JoinQuant/PTrade files, formal trading behavior, schedules, build identity, state schema, or fingerprint.
- Do not add PTrade or JoinQuant integration.
- T-day observation uses daily rows through T-1 only. Entry is the proved T-day 09:35 minute `open`, never that minute's high, low, or close.
- The real source root and field semantics require separate user approval. Until then, use pytest temporary data only and do not run the CLI on market data.
- Signal inequalities are exactly `r2 > r1`, `r0 > r1`, `r1 <= 30`, and `c0 > c1`.
- KDJ, MACD, BOLL, and ATR are background fields only.
- Use red-green TDD for every behavior and create one rollback commit per task.
- Derived files go only to `cross_signal_strategy/reports/prospective/rsi_low_turn_shadow/` or a pytest temporary directory.

## File Map

- Create `cross_signal_strategy/research/rsi_low_turn_shadow.py`: data types, RSI6, exact detector, event ID.
- Create `cross_signal_strategy/research/rsi_low_turn_source.py`: source manifest and causal CSV loading.
- Create `cross_signal_strategy/research/rsi_low_turn_store.py`: append-only JSONL state and de-duplication.
- Create `cross_signal_strategy/research/rsi_low_turn_outcomes.py`: friction, matured labels, Wilson interval, gate.
- Create `cross_signal_strategy/tools/run_rsi_low_turn_shadow.py`: order-free CLI.
- Create five focused test files; modify only README and the approved spec in the final documentation task.

---

### Task 1: Exact RSI6 low-turn signal model

**Files:**
- Create: `cross_signal_strategy/research/rsi_low_turn_shadow.py`
- Create: `tests/test_cross_signal_rsi_low_turn_shadow.py`

**Interfaces:**
- Consumes: a pandas close series and immutable `RsiTurnInput` values.
- Produces: `calculate_rsi6(close)`, `event_id_for(item)`, and `detect_rsi_low_turn(item)`.

Define this test helper in the same test file so every example below is executable:

```python
def make_input(**overrides):
    values = {
        "code": "513100",
        "arrival_dt": datetime(2026, 8, 26, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai")),
        "signal_date": date(2026, 8, 25),
        "r2": 24.0, "r1": 18.0, "r0": 21.0,
        "c1": 2.00, "c0": 2.01,
        "entry_open": 2.035, "price_proved": True,
    }
    values.update(overrides)
    return RsiTurnInput(**values)
```

- [ ] **Step 1: Write failing tests for RSI parity and exact inequalities**

```python
def test_rsi6_matches_formal_formula():
    close = pd.Series([10, 9, 8, 8.5, 8.2, 8.8, 9.1, 8.9, 9.4, 9.7])
    actual = calculate_rsi6(close)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 6, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1 / 6, min_periods=6).mean()
    expected = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    pd.testing.assert_series_equal(actual, expected)


def test_exact_low_turn_needs_no_kdj_or_macd_confirmation():
    item = make_input(r2=24, r1=18, r0=21, c1=2.00, c0=2.01)
    decision = detect_rsi_low_turn(item)
    assert decision.signal_detected is True
    assert decision.valid_event is True
    assert decision.reasons == ()


@pytest.mark.parametrize("changes", [
    {"r2": 18}, {"r0": 18}, {"r1": 30.01}, {"c0": 2.00},
])
def test_equal_or_failed_condition_is_not_a_turn(changes):
    item = replace(make_input(r2=24, r1=18, r0=21, c1=2.00, c0=2.01), **changes)
    assert detect_rsi_low_turn(item).signal_detected is False
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_shadow.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task1
```

Expected: `ModuleNotFoundError` for `rsi_low_turn_shadow`.

- [ ] **Step 3: Implement the minimal immutable model**

```python
VERSION = "rsi-low-turn-shadow-v0.1"

@dataclass(frozen=True)
class RsiTurnInput:
    code: str
    arrival_dt: datetime
    signal_date: date
    r2: float
    r1: float
    r0: float
    c1: float
    c0: float
    entry_open: float | None
    price_proved: bool
    price_reason: str | None = None
    background: Mapping[str, float] = field(default_factory=dict)
    source_hashes: tuple[str, ...] = ()

@dataclass(frozen=True)
class SignalDecision:
    item: RsiTurnInput
    event_id: str
    signal_detected: bool
    valid_event: bool
    reasons: tuple[str, ...]

def calculate_rsi6(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 6, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1.0 / 6, min_periods=6).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    result[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return result

def event_id_for(item: RsiTurnInput) -> str:
    raw = "|".join([VERSION, item.code, item.arrival_dt.date().isoformat(), item.signal_date.isoformat()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def detect_rsi_low_turn(item: RsiTurnInput) -> SignalDecision:
    reasons = []
    if not all(math.isfinite(float(v)) for v in (item.r2, item.r1, item.r0, item.c1, item.c0)):
        reasons.append("non_finite_signal_value")
    else:
        if not item.r2 > item.r1: reasons.append("rsi_not_falling_into_trough")
        if not item.r0 > item.r1: reasons.append("rsi_not_turning_up")
        if not item.r1 <= 30.0: reasons.append("rsi_trough_not_oversold")
        if not item.c0 > item.c1: reasons.append("close_not_turning_up")
    signal = not reasons
    valid = signal and item.price_proved
    if signal and not item.price_proved:
        reasons.append(item.price_reason or "price_unproved")
    return SignalDecision(item, event_id_for(item), signal, valid, tuple(reasons))
```

- [ ] **Step 4: Run Step 2 again and verify GREEN**

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- cross_signal_strategy/research/rsi_low_turn_shadow.py tests/test_cross_signal_rsi_low_turn_shadow.py
git commit -m "research: add RSI low-turn signal model"
```

---

### Task 2: Point-in-time source contract

**Files:**
- Create: `cross_signal_strategy/research/rsi_low_turn_source.py`
- Create: `tests/test_cross_signal_rsi_low_turn_source.py`

**Interfaces:**
- Consumes: explicit `data_root`, explicit `approved_root`, code, and aware 09:35 arrival datetime.
- Produces: `SourceManifest`, `SourceContractError`, `load_manifest`, `load_arrival_input`, and consumed-file hashes.

- [ ] **Step 1: Write failing root and manifest tests**

```python
VALID_MANIFEST = {
    "purpose": "rsi_low_turn_prospective_shadow",
    "version": "rsi-low-turn-shadow-v0.1",
    "collection_start": "2026-08-26",
    "timezone": "Asia/Shanghai",
    "append_only": True,
    "daily_subdir": "daily",
    "minute_subdir": "minute_0935",
}

ARRIVAL_2026_08_26 = datetime(2026, 8, 26, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai"))

def write_manifest(root: Path, payload: Mapping[str, object]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root

def build_valid_source(tmp_path: Path, minute_overrides=None) -> Path:
    root = write_manifest(tmp_path / "source", VALID_MANIFEST)
    (root / "daily").mkdir()
    (root / "minute_0935").mkdir()
    dates = pd.bdate_range(end="2026-08-25", periods=30)
    daily = pd.DataFrame({
        "code": "513100", "date": dates.date,
        "open": np.linspace(2.30, 2.01, 30),
        "high": np.linspace(2.32, 2.03, 30),
        "low": np.linspace(2.28, 1.99, 30),
        "close": np.linspace(2.30, 2.01, 30),
        "volume": 100000,
        "available_at": [f"{day}T15:01:00+08:00" for day in dates.date],
        "source": "pytest_fixture",
    })
    daily.to_csv(root / "daily" / "513100.csv", index=False)
    minute = {
        "code": "513100", "timestamp": "2026-08-26T09:35:00+08:00",
        "open": 2.035, "close": 2.035, "volume": 1000, "num_trades": 10,
        "available_at": "2026-08-26T09:35:00+08:00", "source": "pytest_fixture",
    }
    minute.update(minute_overrides or {})
    pd.DataFrame([minute]).to_csv(root / "minute_0935" / "513100.csv", index=False)
    return root

def build_source_with_future_rows(tmp_path: Path) -> Path:
    root = build_valid_source(tmp_path)
    path = root / "daily" / "513100.csv"
    frame = pd.read_csv(path)
    frame.loc[len(frame)] = [
        "513100", "2026-08-26", 9, 9, 9, 9, 1,
        "2026-08-26T15:01:00+08:00", "pytest_future_row",
    ]
    frame.to_csv(path, index=False)
    return root

def build_zero_trade_args(tmp_path: Path):
    root = build_valid_source(tmp_path, {"volume": 0, "num_trades": 0})
    return root, root, "513100", ARRIVAL_2026_08_26

def test_root_must_equal_separately_approved_root(tmp_path):
    source = write_manifest(tmp_path / "source", VALID_MANIFEST)
    with pytest.raises(SourceContractError, match="approved root"):
        load_manifest(source, tmp_path / "other")

@pytest.mark.parametrize("name", [
    "cross_signal_train_2019_2021", "cross_signal_warmup_2018", "按年份合并", "validation_2022_2023",
])
def test_forbidden_roots_are_refused(tmp_path, name):
    root = write_manifest(tmp_path / name, VALID_MANIFEST)
    with pytest.raises(SourceContractError, match="forbidden"):
        load_manifest(root, root)

def test_collection_start_cannot_precede_freeze(tmp_path):
    raw = dict(VALID_MANIFEST, collection_start="2026-08-25")
    root = write_manifest(tmp_path / "source", raw)
    with pytest.raises(SourceContractError, match="2026-08-26"):
        load_manifest(root, root)
```

- [ ] **Step 2: Write failing causal loader tests**

```python
def test_loader_uses_t_minus_one_and_exact_0935_open(tmp_path):
    root = build_valid_source(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    assert item.signal_date == date(2026, 8, 25)
    assert item.entry_open == pytest.approx(2.035)
    assert item.price_proved is True

def test_t_day_daily_and_late_available_rows_are_invisible(tmp_path):
    root = build_source_with_future_rows(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    assert item.signal_date == date(2026, 8, 25)
    assert len(item.source_hashes) == 3

def test_zero_trade_price_is_audit_only(tmp_path, monkeypatch):
    monkeypatch.setattr(source_module, "calculate_rsi6", rsi_series_ending(24, 18, 21))
    item = load_arrival_input(*build_zero_trade_args(tmp_path))
    decision = detect_rsi_low_turn(item)
    assert decision.signal_detected is True
    assert decision.valid_event is False
    assert "price_unproved" in decision.reasons

def test_background_indicators_match_formal_pure_helpers(tmp_path):
    root = build_valid_source(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    expected = calculate_formal_background_from_same_frame(root)
    assert item.background == pytest.approx(expected)
```

In the test file, `rsi_series_ending(r2, r1, r0)` returns a function that builds a NaN-aligned series with those final three values. `calculate_formal_background_from_same_frame` may stub `jqdata` inside the test process, import the formal module, and call only its pure indicator helpers on the same causal frame; production observer modules must not perform that import.

- [ ] **Step 3: Run Task 2 tests and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_shadow.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task2
```

Expected: import fails because `rsi_low_turn_source.py` is absent.

- [ ] **Step 4: Implement exact manifest and root validation**

```python
MIN_COLLECTION_START = date(2026, 8, 26)
PURPOSE = "rsi_low_turn_prospective_shadow"
SHANGHAI = ZoneInfo("Asia/Shanghai")
OBSERVED_0826 = datetime(2026, 8, 26, 9, 35, tzinfo=SHANGHAI)
OBSERVED_0827 = datetime(2026, 8, 27, 9, 35, tzinfo=SHANGHAI)
OBSERVED_0828 = datetime(2026, 8, 28, 9, 35, tzinfo=SHANGHAI)

@dataclass(frozen=True)
class SourceManifest:
    root: Path
    purpose: str
    version: str
    collection_start: date
    timezone: str
    append_only: bool
    daily_subdir: str
    minute_subdir: str

class SourceContractError(ValueError):
    pass

def validate_root(data_root: Path, approved_root: Path) -> Path:
    data = data_root.resolve(strict=True)
    approved = approved_root.resolve(strict=True)
    if data != approved:
        raise SourceContractError("data root does not equal approved root")
    forbidden = ("cross_signal_train_2019_2021", "cross_signal_warmup_2018", "按年份合并", "validation")
    if any(token.casefold() in str(data).casefold() for token in forbidden):
        raise SourceContractError("forbidden research or validation root")
    return data

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Implement `load_manifest(data_root: Path, approved_root: Path) -> SourceManifest` and `load_arrival_input(data_root: Path, approved_root: Path, code: str, arrival_dt: datetime) -> RsiTurnInput`.

`load_arrival_input` must require an aware Asia/Shanghai datetime at exactly 09:35; reject arrival before manifest start; accept daily rows only when `date <= T-1` and `available_at <= arrival_dt`; calculate RSI6 with Task 1; and use only the exact timely T-day 09:35 minute `open`. Daily CSV columns are exactly `code,date,open,high,low,close,volume,available_at,source`; minute CSV columns are exactly `code,timestamp,open,close,volume,num_trades,available_at,source`. Price proof also requires positive finite open, volume, and `num_trades`.

Compute background RSI12/24, KDJ(9,3,3), MACD(12,26,9), BOLL(20,2), and ATR14 locally with the same pandas formulas as the formal pure helpers. Runtime observer modules must not import either formal strategy or `jqdata`; the parity test may stub `jqdata` only to calculate an independent expected value. The detector must never read background values.

- [ ] **Step 5: Run Step 3 again and verify GREEN**

Expected: Task 1-2 tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- cross_signal_strategy/research/rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_source.py
git commit -m "research: enforce RSI shadow source contract"
```

---

### Task 3: Append-only state, episode de-duplication, and late imports

**Files:**
- Create: `cross_signal_strategy/research/rsi_low_turn_store.py`
- Create: `tests/test_cross_signal_rsi_low_turn_store.py`

**Interfaces:**
- Consumes: `SignalDecision`, observed-at timestamp, relative source path, and SHA-256.
- Produces: `ShadowStore.record_evaluation`, `record_source_hash`, `append_label`, `load_events`, and `load_labels`.

Define the store-test decision factory explicitly:

```python
SHANGHAI = ZoneInfo("Asia/Shanghai")

def decision_for(arrival_date: str, signal: bool = True, price: float = 2.035):
    day = date.fromisoformat(arrival_date)
    item = RsiTurnInput(
        code="513100",
        arrival_dt=datetime.combine(day, time(9, 35), SHANGHAI),
        signal_date=day - timedelta(days=1),
        r2=24.0 if signal else 18.0,
        r1=18.0,
        r0=21.0,
        c1=2.00,
        c0=price,
        entry_open=price,
        price_proved=True,
    )
    return detect_rsi_low_turn(item)
```

Use `true_decision(day)` as `decision_for(day, True)`, `false_decision(day)` as `decision_for(day, False)`, and `replace_price(decision, price)` by replacing `decision.item.c0` and re-running the detector.

- [ ] **Step 1: Write failing episode and idempotency tests**

```python
def test_first_true_emits_once_and_consecutive_true_does_not(tmp_path):
    store = ShadowStore(tmp_path)
    first = store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    second = store.record_evaluation(true_decision("2026-08-27"), OBSERVED_0827)
    assert first.event_created is True
    assert second.event_created is False
    assert second.reason == "same_active_episode"

def test_false_day_resets_episode(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0826)
    store.record_evaluation(false_decision("2026-08-27"), OBSERVED_0827)
    assert store.record_evaluation(true_decision("2026-08-28"), OBSERVED_0828).event_created is True

def test_duplicate_is_idempotent_but_conflicting_payload_is_refused(tmp_path):
    store = ShadowStore(tmp_path)
    decision = true_decision("2026-08-26")
    assert store.record_evaluation(decision, OBSERVED_0826).written is True
    assert store.record_evaluation(decision, OBSERVED_0826).written is False
    with pytest.raises(SourceRewriteError, match="conflicting evaluation"):
        store.record_evaluation(replace_price(decision, 9.99), OBSERVED_0826)
```

- [ ] **Step 2: Write failing provenance tests**

```python
def test_changed_source_hash_stops(tmp_path):
    store = ShadowStore(tmp_path)
    store.record_source_hash("daily/513100.csv", "a" * 64, OBSERVED_0826)
    with pytest.raises(SourceRewriteError, match="source hash changed"):
        store.record_source_hash("daily/513100.csv", "b" * 64, OBSERVED_0827)

def test_old_arrival_first_seen_later_is_audit_only(tmp_path):
    store = ShadowStore(tmp_path)
    result = store.record_evaluation(true_decision("2026-08-26"), OBSERVED_0827)
    assert result.event_created is False
    assert result.reason == "late_import"
```

- [ ] **Step 3: Run Task 3 tests and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_store.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task3
```

Expected: import fails because `rsi_low_turn_store.py` is absent.

- [ ] **Step 4: Implement canonical JSONL storage**

```python
EVALUATIONS_FILE = "evaluations.jsonl"
EVENTS_FILE = "events.jsonl"
HASHES_FILE = "source_hashes.jsonl"
LABELS_FILE = "labels.jsonl"

@dataclass(frozen=True)
class RecordResult:
    written: bool
    event_created: bool
    reason: str

class SourceRewriteError(RuntimeError):
    pass

class ShadowStore:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
```

Add these exact methods: `record_source_hash(relative_path: str, sha256: str, observed_at: datetime) -> bool`, `record_evaluation(decision: SignalDecision, observed_at: datetime) -> RecordResult`, `append_label(event_id: str, horizon: int, payload: Mapping[str, object]) -> RecordResult`, `load_events() -> tuple[Mapping[str, object], ...]`, and `load_labels() -> tuple[Mapping[str, object], ...]`.

Use `sort_keys=True`, `ensure_ascii=False`, ISO-8601 offsets, and one record per line. Compare canonical serialized payloads for idempotency and reject the same key with different content. Mark an evaluation `late_import` when its first `observed_at` is later than its logical 09:35 `arrival_dt`. Episode state follows every day's `signal_detected`, not `valid_event`; an unproved first day must not shift the same signal episode to a later entry.

- [ ] **Step 5: Run Tasks 1-3 and verify GREEN**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_shadow.py tests/test_cross_signal_rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_store.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task3_green
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- cross_signal_strategy/research/rsi_low_turn_store.py tests/test_cross_signal_rsi_low_turn_store.py
git commit -m "research: add append-only RSI shadow state"
```

---

### Task 4: Fixed-horizon outcomes and realistic friction

**Files:**
- Create: `cross_signal_strategy/research/rsi_low_turn_outcomes.py`
- Create: `tests/test_cross_signal_rsi_low_turn_outcomes.py`
- Modify: `cross_signal_strategy/research/rsi_low_turn_source.py`
- Modify: `tests/test_cross_signal_rsi_low_turn_source.py`

**Interfaces:**
- Consumes: stored valid events, approved-source future prices, matured daily rows, and `LocalBroker`.
- Produces: `Friction`, `RoundTripResult`, `FutureSnapshot`, `MaturedLabel`, `EventOutcomeRecord`, `calculate_round_trip`, and `mature_event_labels` for `(1, 3, 5, 10)`.

Define `valid_event()` in the outcome test as a mapping with `event_id="e1"`, `code="513100"`, `arrival_date=date(2026, 8, 26)`, and `entry_open=2.035`. Define `source_with_horizons(tmp_path, horizons)` as an in-memory `FakeFuturePriceSource` returning positive executable opens for the requested horizons; `source_missing_horizon_three` returns `status="pending_missing_executable_price"` and `exit_open=None` only for horizon 3.

- [ ] **Step 1: Write failing broker-parity tests**

```python
NOMINAL = Friction(0.0003, 5.0, 0.001)
DOUBLED = Friction(0.0006, 10.0, 0.002)
SLOT_CAPITAL = 20000.0 * 0.95 / 3.0

def test_round_trip_matches_local_broker_and_integer_lots():
    result = calculate_round_trip("513100", 2.000, 2.100, NOMINAL)
    assert result.amount % 100 == 0
    assert result.buy_commission == 5.0
    assert result.sell_commission == 5.0
    assert result.net_return == pytest.approx(result.net_pnl / SLOT_CAPITAL)

def test_doubled_friction_uses_ten_yuan_minimum():
    result = calculate_round_trip("513100", 2.000, 2.100, DOUBLED)
    assert result.buy_commission == 10.0
    assert result.sell_commission == 10.0
```

Generate the expected cash result in the test through an independently constructed `LocalBroker`.

- [ ] **Step 2: Write failing maturity tests**

```python
def test_only_arrived_horizons_mature(tmp_path):
    labels = mature_event_labels(valid_event(), source_with_horizons(tmp_path, (1, 3, 5)), ARRIVAL_PLUS_5)
    assert {label.horizon for label in labels if label.status == "matured"} == {1, 3, 5}

def test_missing_0935_price_is_not_substituted(tmp_path):
    labels = mature_event_labels(valid_event(), source_missing_horizon_three(tmp_path), ARRIVAL_PLUS_10)
    item = next(label for label in labels if label.horizon == 3)
    assert item.status == "pending_missing_executable_price"
    assert item.exit_price is None
```

- [ ] **Step 3: Run Task 4 tests and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_outcomes.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task4
```

Expected: import fails because outcome interfaces are absent.

- [ ] **Step 4: Implement friction through `LocalBroker`**

```python
HORIZONS = (1, 3, 5, 10)
SLOT_CAPITAL = 20000.0 * 0.95 / 3.0

@dataclass(frozen=True)
class Friction:
    commission_rate: float
    min_commission: float
    slippage_rate: float

@dataclass(frozen=True)
class RoundTripResult:
    amount: int
    buy_exec_price: float
    sell_exec_price: float
    buy_commission: float
    sell_commission: float
    net_pnl: float
    net_return: float

@dataclass(frozen=True)
class FutureSnapshot:
    horizon: int
    status: str
    exit_open: float | None
    mfe: float | None
    mae: float | None
    available_at: datetime | None

@dataclass(frozen=True)
class MaturedLabel:
    event_id: str
    horizon: int
    status: str
    exit_price: float | None
    nominal: RoundTripResult | None
    doubled: RoundTripResult | None
    mfe: float | None
    mae: float | None

@dataclass(frozen=True)
class EventOutcomeRecord:
    event_id: str
    code: str
    arrival_date: date
    labels: Mapping[int, MaturedLabel]

def calculate_round_trip(code: str, entry_open: float, exit_open: float, friction: Friction) -> RoundTripResult:
    broker = LocalBroker(SLOT_CAPITAL, commission_rate=friction.commission_rate,
                         min_commission=friction.min_commission,
                         slippage_rate=friction.slippage_rate)
    buy = broker.order_target_value(code, SLOT_CAPITAL, entry_open, "shadow_entry")
    if not buy.filled:
        raise ValueError(f"shadow entry not executable: {buy.reason}")
    sell = broker.order_target_value(code, 0.0, exit_open, "shadow_exit")
    if not sell.filled:
        raise ValueError(f"shadow exit not executable: {sell.reason}")
    pnl = broker.cash - SLOT_CAPITAL
    return RoundTripResult(buy.amount_delta, buy.exec_price, sell.exec_price,
                           buy.commission, sell.commission, pnl, pnl / SLOT_CAPITAL)
```

Define a `FuturePriceSource` protocol with `snapshot_for(event: Mapping[str, object], horizon: int, as_of: datetime) -> FutureSnapshot`. Test it with an in-memory fake whose constructor accepts a mapping from horizon to `FutureSnapshot`; this supplies the `source_with_horizons` and `source_missing_horizon_three` fixtures used above. Add a real source adapter that resolves the Nth future session from the approved source calendar and returns only its exact timely, executable 09:35 open. Compute MFE/MAE only after every daily bar through the horizon has matured; never feed outcome high/low back to the detector.

- [ ] **Step 5: Run Tasks 1-4 and verify GREEN**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_shadow.py tests/test_cross_signal_rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_store.py tests/test_cross_signal_rsi_low_turn_outcomes.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task4_green
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add -- cross_signal_strategy/research/rsi_low_turn_outcomes.py cross_signal_strategy/research/rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_outcomes.py tests/test_cross_signal_rsi_low_turn_source.py
git commit -m "research: mature RSI shadow outcomes"
```

---

### Task 5: Frozen statistical gate and summary

**Files:**
- Modify: `cross_signal_strategy/research/rsi_low_turn_outcomes.py`
- Modify: `tests/test_cross_signal_rsi_low_turn_outcomes.py`

**Interfaces:**
- Consumes: matured five-day and ten-day doubled-friction records.
- Produces: `wilson_interval`, `GateDecision`, `evaluate_gate`, and `build_summary`.

The test factory `make_records(count, five_day_wins)` must return `EventOutcomeRecord` objects distributed across six ETF codes and seven calendar months by default. Each record has matured five-day and ten-day doubled-friction labels; `mutate_passing_records(name)` changes only the named gate input and keeps all other passing defaults fixed.

- [ ] **Step 1: Write failing Wilson and accumulating tests**

```python
def test_wilson_interval_is_not_the_raw_rate():
    lower, upper = wilson_interval(6, 7)
    assert lower == pytest.approx(0.486872, abs=1e-6)
    assert upper == pytest.approx(0.974321, abs=1e-6)

def test_under_fifty_is_accumulating():
    gate = evaluate_gate(make_records(count=49, five_day_wins=40))
    assert gate.status == "accumulating"
    assert gate.reasons == ("fewer_than_50_matured_five_day_events",)
```

- [ ] **Step 2: Write one failing test per frozen gate**

```python
@pytest.mark.parametrize("mutation,reason", [
    ("span_under_six_months", "observation_span_under_six_months"),
    ("only_four_etfs", "fewer_than_five_etfs"),
    ("one_etf_over_40_percent", "single_etf_share_over_40_percent"),
    ("wilson_lower_not_above_half", "five_day_wilson_lower_not_above_50_percent"),
    ("five_day_mean_non_positive", "five_day_double_mean_not_positive"),
    ("five_day_median_non_positive", "five_day_double_median_not_positive"),
    ("ten_day_mean_negative", "ten_day_double_mean_negative"),
    ("ten_day_median_negative", "ten_day_double_median_negative"),
    ("top_winner_dependency", "leave_top_winner_out_mean_not_positive"),
])
def test_each_gate_fails_closed(mutation, reason):
    gate = evaluate_gate(mutate_passing_records(mutation))
    assert gate.status == "stop"
    assert reason in gate.reasons

def test_all_frozen_gates_can_pass_together():
    gate = evaluate_gate(make_records(count=60, five_day_wins=50))
    assert gate.status == "pass"
    assert gate.reasons == ()
```

- [ ] **Step 3: Run Task 5 tests and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_outcomes.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task5
```

Expected: missing Wilson and gate interfaces fail.

- [ ] **Step 4: Implement the statistical interfaces**

```python
@dataclass(frozen=True)
class GateDecision:
    status: str
    reasons: tuple[str, ...]
    metrics: Mapping[str, float | int]

def wilson_interval(successes: int, total: int,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half
```

`evaluate_gate` returns `accumulating` below 50 valid five-day events. At 50 or more, evaluate every gate without short-circuiting and return `pass` only with no reasons. `build_summary` includes version, collection start, generation timestamp, counts, date span, ETF distribution, nominal/doubled metrics for 1/3/5/10 days, Wilson bounds, leave-one-top-winner mean, status, and reasons.

- [ ] **Step 5: Run Task 5 again and verify GREEN**

Expected: all outcome tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- cross_signal_strategy/research/rsi_low_turn_outcomes.py tests/test_cross_signal_rsi_low_turn_outcomes.py
git commit -m "research: freeze RSI shadow evidence gate"
```

---

### Task 6: Order-free CLI, documentation, and complete verification

**Files:**
- Create: `cross_signal_strategy/tools/run_rsi_low_turn_shadow.py`
- Create: `tests/test_cross_signal_rsi_low_turn_cli.py`
- Modify: `cross_signal_strategy/README.md`
- Modify: `cross_signal_strategy/docs/superpowers/specs/2026-08-25-rsi-low-turn-prospective-shadow-design.md`

**Interfaces:**
- Consumes: Tasks 1-5 and explicit CLI arguments.
- Produces: `collect`, `summarize`, append-only state, `summary.json`, Chinese status output, and no order/platform side effects.

Define `run_cli(*args)` in the test with `subprocess.run([sys.executable, str(CLI_PATH), *args], text=True, capture_output=True)`. The source fixture reuses Task 2's exact manifest and schemas; `hash_tree(root)` returns sorted `(relative_path, sha256)` pairs so source mutation is detected byte-for-byte.

For the AST dependency test, define `OBSERVER_MODULE_PATHS` as the four research modules plus the CLI. `imported_module_names(tree)` collects names from `ast.Import` and `ast.ImportFrom`; `called_function_names(tree)` collects the terminal attribute or name from every `ast.Call`.

- [ ] **Step 1: Write failing CLI refusal and source-immutability tests**

```python
def test_collect_refuses_nonmatching_approved_root(tmp_path):
    result = run_cli("collect", "--data-root", str(tmp_path / "source"),
                     "--approved-root", str(tmp_path / "other"),
                     "--state-dir", str(tmp_path / "state"),
                     "--as-of", "2026-08-26T09:35:00+08:00")
    assert result.returncode == 2
    assert "approved root" in result.stdout

def test_observer_modules_have_no_platform_or_order_dependency():
    forbidden = {"jqdata", "smart_trade_joinquant_cross_signal_etf", "smart_trade_ptrade_cross_signal_etf"}
    for path in OBSERVER_MODULE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = imported_module_names(tree)
        called = called_function_names(tree)
        assert not (imported & forbidden)
        assert not any(name.startswith("order") or name == "execute_sell" for name in called)

def test_collect_and_summarize_do_not_modify_source(tmp_path):
    root = build_valid_source(tmp_path)
    before = hash_tree(root)
    state = tmp_path / "state"
    assert run_collect_and_summarize(root, state).returncode == 0
    assert hash_tree(root) == before
    assert (state / "summary.json").exists()
```

- [ ] **Step 2: Run CLI tests and verify RED**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_cli.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_task6
```

Expected: subprocess fails because the CLI file is absent.

- [ ] **Step 3: Implement exact commands and arguments**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--data-root", type=Path, required=True)
    collect.add_argument("--approved-root", type=Path, required=True)
    collect.add_argument("--state-dir", type=Path, required=True)
    collect.add_argument("--as-of", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--state-dir", type=Path, required=True)
    summarize.add_argument("--generated-at", required=True)
    return parser
```

`collect` validates root and manifest before creating state, loops through the frozen nine ETF codes, records all true/false evaluations, creates only first-day valid events, matures only available labels, and prints `orders_disabled=True`. `summarize` reads state only and atomically replaces `summary.json` through a temporary file inside the same state directory. Catch `SourceContractError` and `SourceRewriteError`, print one Chinese error line, and return 2; leave unexpected tracebacks visible.

- [ ] **Step 4: Run Step 2 again and verify GREEN**

Expected: all CLI tests pass.

- [ ] **Step 5: Update documentation without claiming market evidence**

Add README entries for the four modules and CLI. Change the spec status to exactly:

```text
状态：规格已确认；本地观察器实现完成并通过临时数据测试；真实数据源仍未批准，状态保持 blocked_on_data_source。尚未产生前瞻事件、收益结果、聚宽/PTrade版本或正式策略改动。
```

Do not increment historical failed-experiment counts and do not report performance.

- [ ] **Step 6: Run the complete scoped verification**

```powershell
python run_pytest_sandbox.py tests/test_cross_signal_rsi_low_turn_shadow.py tests/test_cross_signal_rsi_low_turn_source.py tests/test_cross_signal_rsi_low_turn_store.py tests/test_cross_signal_rsi_low_turn_outcomes.py tests/test_cross_signal_rsi_low_turn_cli.py tests/test_cross_signal_local_backtester.py -q -p no:cacheprovider --basetemp G:\financial\select_stocks\.codex_pytest_rsi_shadow_final
python -m py_compile cross_signal_strategy/research/rsi_low_turn_shadow.py cross_signal_strategy/research/rsi_low_turn_source.py cross_signal_strategy/research/rsi_low_turn_store.py cross_signal_strategy/research/rsi_low_turn_outcomes.py cross_signal_strategy/tools/run_rsi_low_turn_shadow.py
git diff --check
```

Expected: pytest and compilation exit 0 and diff check has no errors. Inspect `git status --short` and prove that neither formal strategy file appears.

- [ ] **Step 7: Remove only verified task-owned temporary directories**

Use this exact PowerShell pattern so every target is explicit and verified:

```powershell
$workspacePath = [System.IO.Path]::GetFullPath('G:\financial\select_stocks')
$taskDirs = @(
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task1',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task2',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task3',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task3_green',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task4',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task4_green',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task5',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_task6',
  'G:\financial\select_stocks\.codex_pytest_rsi_shadow_final'
)
foreach ($taskDir in $taskDirs) {
  if (Test-Path -LiteralPath $taskDir) {
    $resolved = (Resolve-Path -LiteralPath $taskDir).Path
    if (-not $resolved.StartsWith($workspacePath + [System.IO.Path]::DirectorySeparatorChar)) {
      throw "Refusing cleanup outside workspace: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
}
```

Do not touch unrelated pytest directories or any market-data root.

- [ ] **Step 8: Commit Task 6**

```powershell
git add -- cross_signal_strategy/tools/run_rsi_low_turn_shadow.py tests/test_cross_signal_rsi_low_turn_cli.py cross_signal_strategy/README.md cross_signal_strategy/docs/superpowers/specs/2026-08-25-rsi-low-turn-prospective-shadow-design.md
git commit -m "research: add blocked prospective RSI shadow observer"
```

After committing, run `git show --stat --oneline HEAD` and `git status --short --untracked-files=no`. Report that real collection remains blocked until the user approves the exact source root and point-in-time semantics.
