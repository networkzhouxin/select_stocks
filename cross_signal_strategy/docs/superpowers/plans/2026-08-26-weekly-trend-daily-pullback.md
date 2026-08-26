# Weekly Trend Daily Pullback Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one isolated JoinQuant candidate that permits a daily BOLL/KDJ/RSI pullback entry only when the same ETF's last completed weekly bar is above a rising weekly MA20, while preserving causal ATR and trend-failure exits.

**Architecture:** A platform-neutral research module owns weekly aggregation, signal predicates, position state, deterministic ranking, and order planning. A standalone JoinQuant upload file mirrors those frozen primitives using Python syntax compatible with the platform and supplies data loading, scheduling, logging, and order-lifecycle protection. Neither formal `cross-v0.3.3` file imports or is imported by the candidate.

**Tech Stack:** Python 3, pandas, NumPy, pytest, JoinQuant strategy API.

**Spec:** `cross_signal_strategy/docs/superpowers/specs/2026-08-26-weekly-trend-daily-pullback-design.md`

## Global Constraints

- Candidate family: `weekly_trend_daily_pullback_user_authorized`; exactly one pre-registered variant.
- Candidate version: `weekly-trend-pullback-v0.1-joinquant-candidate`.
- Daily signals use only exact T-1-or-earlier completed bars; stale T-1 data fails closed.
- Weekly signals exclude the decision date's entire natural week and use only the most recent completed ETF-specific week.
- Weekly gate: close above weekly MA20 and weekly MA20 strictly rising; weekly break is the exact symmetric downside condition.
- Daily entry is one AND rule: BOLL lower-half inside the band, fresh K/D gold cross, rising RSI6, RSI6 at or below 50.
- Exit priority: ATR, completed-week trend break, five-session daily pullback failure.
- Execution schedule is only 09:35 full processing and 14:50 ATR-only processing.
- Pool is the frozen nine-ETF formal pool; maximum holdings 3; base ratio 0.95; fixed slot target is total value times 0.95 divided by 3.
- ATR uses frozen entry ATR14, multiplier 2.5, 5%-15% clamp, and the inherited 3% floor for `518880.XSHG`.
- Formal JoinQuant SHA-256 must remain `9ADB96E523BBA2B1E5C42CB2BDDA06A8BB06065EA1AE25194651177987DF0F52`.
- Formal PTrade SHA-256 must remain `E4FA39CC79350A8E790074E1D3C75D0A4638ECA23DB8155F4EE1203869E7D6F8`.
- Do not read validation-period data or `G:\financial\history_data\按年份合并` during implementation.
- Do not run neighboring parameter, period, ranking, ETF, timing, or exit variants.

---

### Task 1: Open Exactly One Governed Research Family

**Files:**
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `tests/test_cross_signal_research_budget.py`

**Interfaces:**
- Consumes: `load_research_budget(path)` and `evaluate_experiment_request(budget, family_key, planned_variants)` from `cross_signal_strategy.research.research_budget`.
- Produces: one open family named `weekly_trend_daily_pullback_user_authorized` with one available variant and exact `planned_experiment` value `weekly-trend-pullback-v0.1-joinquant-candidate`.

- [ ] **Step 1: Write the failing governance test**

Append a focused test that proves the user authorization opens only this fixed family:

```python
def test_weekly_trend_daily_pullback_has_exactly_one_open_variant():
    budget = load_research_budget(BUDGET_PATH)
    families = {family.key: family for family in budget.families}
    family = families["weekly_trend_daily_pullback_user_authorized"]

    assert budget.max_total_open_experiments == 1
    assert [item.key for item in budget.families if item.status == "open"] == [
        "weekly_trend_daily_pullback_user_authorized"
    ]
    assert family.max_new_experiments == 1
    assert family.planned_experiment == "weekly-trend-pullback-v0.1-joinquant-candidate"
    assert evaluate_experiment_request(budget, family.key, 1).allowed is True
    assert evaluate_experiment_request(budget, family.key, 2).allowed is False
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_cross_signal_research_budget.py::test_weekly_trend_daily_pullback_has_exactly_one_open_variant -q
```

Expected: FAIL because the family does not yet exist and the global open budget is zero.

- [ ] **Step 3: Register the single family**

Change `max_total_open_experiments` from `0` to `1` and add this JSON object without changing failed-experiment counts:

```json
{
  "key": "weekly_trend_daily_pullback_user_authorized",
  "label": "User-authorized completed-week trend plus daily pullback candidate",
  "status": "open",
  "max_new_experiments": 1,
  "rationale": "The user authorized one structurally separate strategy family: each ETF must be above a rising completed-week MA20 before the fixed daily BOLL/KDJ/RSI pullback entry can trade. This does not reopen formal cross-v0.3.3 confirmation tuning or the exhausted KRBA family.",
  "planned_experiment": "weekly-trend-pullback-v0.1-joinquant-candidate",
  "candidate_variants": 1,
  "validation_influence": "none",
  "data_scope": "2018_warmup_plus_2019_2021_training_only",
  "prohibit_alternatives": true
}
```

Update `research_budget.md` Current Accounting and Open Families so they state that this is the sole open family and restate the frozen rule and one-variant boundary. Do not describe any result because no replay has run.

- [ ] **Step 4: Run governance tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_cross_signal_research_budget.py -q
```

Expected: PASS with the new family as the only open budget.

- [ ] **Step 5: Commit the governance milestone**

```powershell
git add cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md tests/test_cross_signal_research_budget.py
git commit -m "research: open weekly pullback candidate budget"
```

---

### Task 2: Implement Completed-Week Aggregation and Weekly Gates

**Files:**
- Create: `cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py`
- Create: `tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py`

**Interfaces:**
- Consumes: a pandas daily frame with `date`, `open`, `high`, `low`, `close`, and optional `volume`, plus a decision date.
- Produces: `aggregate_completed_weeks(frame, decision_date) -> pd.DataFrame`, `build_weekly_context(frame, decision_date) -> tuple[dict | None, str | None]`, `weekly_entry_allowed(snapshot) -> bool`, and `weekly_trend_broken(snapshot) -> bool`.

- [ ] **Step 1: Write failing completed-week boundary tests**

Create fixtures with 22 prior calendar weeks plus bars in the current decision week. Cover all of these assertions:

```python
def test_completed_weeks_exclude_the_entire_decision_week():
    weeks = candidate.aggregate_completed_weeks(frame, "2021-03-10")
    assert weeks.iloc[-1]["last_trade_date"].date().isoformat() == "2021-03-05"
    assert current_monday_close not in weeks["close"].tolist()

def test_short_holiday_week_is_complete_after_its_calendar_week_ends():
    weeks = candidate.aggregate_completed_weeks(holiday_frame, "2021-02-22")
    assert weeks.iloc[-1]["close"] == pytest.approx(last_holiday_session_close)

def test_weekly_context_needs_21_completed_weeks():
    context, reason = candidate.build_weekly_context(twenty_week_frame, "2021-03-10")
    assert context is None
    assert reason == "insufficient_weekly_history"

def test_each_etf_frame_produces_its_own_weekly_gate():
    allowed, _ = candidate.build_weekly_context(rising_etf_frame, "2021-03-10")
    blocked, _ = candidate.build_weekly_context(falling_etf_frame, "2021-03-10")
    assert candidate.weekly_entry_allowed(allowed) is True
    assert candidate.weekly_entry_allowed(blocked) is False
```

Also test malformed columns, no completed weekly row, non-finite values, and equality at either weekly threshold.

- [ ] **Step 2: Run weekly tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -k "week" -q
```

Expected: collection/import failure because the research module does not exist.

- [ ] **Step 3: Implement the minimum weekly functions**

Use natural weeks ending Sunday, exclude the current week before grouping, and return explicit reason codes:

```python
VERSION = "weekly-trend-pullback-v0.1-research-candidate"

def aggregate_completed_weeks(frame, decision_date):
    work = frame.copy()
    dates = pd.to_datetime(work["date"] if "date" in work else work.index)
    current_monday = pd.Timestamp(decision_date).normalize() - pd.Timedelta(
        days=pd.Timestamp(decision_date).weekday()
    )
    work = work.loc[dates < current_monday].copy()
    work["date"] = dates[dates < current_monday]
    work["week"] = work["date"].dt.to_period("W-SUN")
    return work.groupby("week", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        last_trade_date=("date", "last"),
    )

def build_weekly_context(frame, decision_date):
    weeks = aggregate_completed_weeks(frame, decision_date)
    if len(weeks) < 21:
        return None, "insufficient_weekly_history"
    closes = pd.to_numeric(weeks["close"], errors="coerce")
    ma20 = closes.rolling(20).mean()
    values = {
        "weekly_close": closes.iloc[-1],
        "weekly_ma20": ma20.iloc[-1],
        "weekly_ma20_prev": ma20.iloc[-2],
        "weekly_period_end": weeks.index[-1].end_time.date().isoformat(),
        "weekly_last_trade_date": weeks.iloc[-1]["last_trade_date"].date().isoformat(),
    }
    numeric_keys = ("weekly_close", "weekly_ma20", "weekly_ma20_prev")
    if not all(math.isfinite(_number(values[key])) for key in numeric_keys):
        return None, "invalid_weekly_indicator"
    return values, None
```

Implement exact strict predicates:

```python
def weekly_entry_allowed(snapshot):
    return snapshot["weekly_close"] > snapshot["weekly_ma20"] > snapshot["weekly_ma20_prev"]

def weekly_trend_broken(snapshot):
    return snapshot["weekly_close"] < snapshot["weekly_ma20"] < snapshot["weekly_ma20_prev"]
```

Defensively coerce values and return `False` rather than raising for missing/non-finite predicate input.

- [ ] **Step 4: Run weekly tests and verify GREEN**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -k "week" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the weekly-boundary milestone**

```powershell
git add cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py
git commit -m "research: add completed-week trend gates"
```

---

### Task 3: Implement Daily Entry, Ranking, Position State, and Exit Priority

**Files:**
- Modify: `cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py`
- Modify: `tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py`

**Interfaces:**
- Consumes: one combined snapshot containing weekly context plus daily `k_prev`, `d_prev`, `k`, `d`, `rsi6_prev`, `rsi6`, `close`, `boll_lower`, `boll_mid`, and `atr`.
- Produces: `is_daily_entry_eligible(snapshot)`, `is_entry_eligible(snapshot)`, `build_buy_queue(snapshots, excluded_codes, etf_pool)`, `PositionSignalState`, `calc_frozen_atr_stop(state, code="", multiplier=2.5, floor=0.05, cap=0.15)`, `update_highest_close_from_t1(state, close)`, and `choose_exit_reason(state, snapshot, current_price, hold_days, code="")`.

- [ ] **Step 1: Write failing entry and ranking tests**

Use a default eligible snapshot and independently break every condition:

```python
def test_entry_requires_weekly_gate_and_every_daily_condition():
    assert candidate.is_entry_eligible(snapshot()) is True
    for broken in (
        {"weekly_close": 9.9, "weekly_ma20": 10.0},
        {"close": 9.7, "boll_lower": 9.7},
        {"close": 10.6, "boll_mid": 10.5},
        {"k_prev": 21.0, "d_prev": 20.0},
        {"k": 20.0, "d": 20.0},
        {"rsi6": 40.0, "rsi6_prev": 40.0},
        {"rsi6": 50.01},
    ):
        assert candidate.is_entry_eligible(snapshot(**broken)) is False

def test_buy_queue_ranks_weekly_strength_then_kd_then_pool_order():
    queue = candidate.build_buy_queue(
        [weak_week, strong_small_cross, strong_large_cross, tied_pool_later],
        excluded_codes={"513500"},
        etf_pool=["159915", "513100", "513500", "518880"],
    )
    assert [item["code"] for item in queue] == ["513100", "518880", "159915"]
```

Test equality semantics and all missing/non-finite fields explicitly.

- [ ] **Step 2: Write failing exit-priority tests**

```python
def test_exit_priority_is_atr_then_weekly_break_then_daily_failure():
    state = candidate.PositionSignalState("2021-01-04", 10.0, 0.2, 11.0)
    assert candidate.choose_exit_reason(
        state, broken_week_and_daily_death_cross, 10.44, 8, "513100"
    ) == "atr_stop"
    assert candidate.choose_exit_reason(
        state, broken_week_and_daily_death_cross, 10.80, 8, "513100"
    ) == "weekly_trend_break"
    assert candidate.choose_exit_reason(
        state, allowed_week_and_daily_death_cross, 10.80, 8, "513100"
    ) == "daily_pullback_failure"

def test_daily_failure_waits_five_sessions_but_weekly_break_does_not():
    assert candidate.choose_exit_reason(
        state, allowed_week_and_daily_death_cross, 10.80, 4, "513100"
    ) is None
    assert candidate.choose_exit_reason(
        state, broken_week_and_daily_death_cross, 10.80, 1, "513100"
    ) == "weekly_trend_break"
```

Also prove that an upper-band touch is not an exit and that `518880` retains the 3% floor.

- [ ] **Step 3: Run signal tests and verify RED**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -k "entry or queue or exit or atr" -q
```

Expected: FAIL because the daily and state interfaces are absent.

- [ ] **Step 4: Implement the frozen signal primitives**

Implement strict, finite-safe predicates and a minimal state:

```python
@dataclass
class PositionSignalState:
    entry_date: str
    entry_price: float
    entry_atr: float
    highest_close: float

def is_daily_entry_eligible(snapshot):
    return bool(
        snapshot["close"] > snapshot["boll_lower"]
        and snapshot["close"] <= snapshot["boll_mid"]
        and snapshot["k_prev"] <= snapshot["d_prev"]
        and snapshot["k"] > snapshot["d"]
        and snapshot["rsi6"] > snapshot["rsi6_prev"]
        and snapshot["rsi6"] <= 50.0
    )

def is_entry_eligible(snapshot):
    return weekly_entry_allowed(snapshot) and is_daily_entry_eligible(snapshot)
```

Implement ranking as exact tuples:

```python
key=lambda item: (
    -(item["weekly_close"] / item["weekly_ma20"] - 1.0),
    -(item["k"] - item["d"]),
    pool_rank[item["code"]],
)
```

Implement `choose_exit_reason` in the frozen priority order. `daily_pullback_failure` requires `hold_days >= 5`, `close < boll_mid`, and a fresh K/D death cross. Do not add upper-band, profit, MACD, ADX, or maximum-hold exits.

- [ ] **Step 5: Run all research-module tests and verify GREEN**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the signal milestone**

```powershell
git add cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py
git commit -m "research: add weekly pullback signal rules"
```

---

### Task 4: Add the Platform-Neutral Order Planner

**Files:**
- Modify: `cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py`
- Modify: `tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py`

**Interfaces:**
- Consumes: precomputed combined snapshots, held codes, causal position states, current executable prices, trade calendar, total value, sold-today set, and decision time.
- Produces: `TrendPullbackOrderPlanner.plan_orders_at(current_date, previous_date, broker, decision_time, current_prices=None)`, `_plan_1450_atr(broker, current_prices)`, `on_orders_processed(current_date, decision_time, plans, results)`, and `on_after_close(current_date, marks)` with order-plan dictionaries shaped as `{"code", "target_value", "reason"}` for sells and `{"code", "target_value", "reason", "entry_atr"}` for buys.

- [ ] **Step 1: Write failing 09:35 planning tests**

Prove sell-before-buy behavior and no same-day rebuy:

```python
def test_0935_plans_sells_before_fixed_slot_buys():
    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker_with_one_broken_week_holding,
        decision_time="09:35",
        current_prices=prices,
    )
    assert plans[0] == {
        "code": held_code,
        "target_value": 0.0,
        "reason": "weekly_trend_break",
    }
    assert plans[1]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
    assert sold_today_code not in [item["code"] for item in plans[1:]]
```

Add tests for no forced replacement at three healthy holdings, deterministic ranking, and actual fill required before state creation.

- [ ] **Step 2: Write failing ATR-isolation tests**

```python
def test_0935_atr_survives_missing_signal_snapshot():
    code = "513100"
    broker = SimpleNamespace(
        positions={code: SimpleNamespace(amount=100, avg_cost=10.0)},
        cash=10000.0,
    )
    planner.position_states[code] = candidate.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 11.0
    )
    planner.signal_adapter = SimpleNamespace(
        score=lambda *args, **kwargs: (None, "stale_signal_date")
    )
    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker,
        decision_time="09:35",
        current_prices={code: 10.44},
    )
    assert plans == [{"code": code, "target_value": 0.0, "reason": "atr_stop"}]

def test_1450_calls_no_signal_adapter_and_only_returns_atr_sells():
    stopped_code = "513100"
    broker = SimpleNamespace(
        positions={stopped_code: SimpleNamespace(amount=100, avg_cost=10.0)},
        cash=10000.0,
    )
    planner.position_states[stopped_code] = candidate.PositionSignalState(
        "2021-03-01", 10.0, 0.2, 11.0
    )
    planner.signal_adapter = SimpleNamespace(
        score=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("14:50 loaded signals")
        )
    )
    plans = planner.plan_orders_at(
        current_date="2021-03-08",
        previous_date="2021-03-05",
        broker=broker,
        decision_time="14:50",
        current_prices={stopped_code: 10.44},
    )
    assert plans == [
        {"code": stopped_code, "target_value": 0.0, "reason": "atr_stop"}
    ]
```

- [ ] **Step 3: Run planner tests and verify RED**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -k "planner or 0935 or 1450 or orders" -q
```

Expected: FAIL because `TrendPullbackOrderPlanner` is not defined.

- [ ] **Step 4: Implement the minimum planner**

Model the causal interfaces on the existing KRBA planner, but use only the new frozen signals:

```python
@dataclass
class TrendPullbackOrderPlanner:
    signal_adapter: object
    etf_pool: Iterable[str]
    trade_dates: list[str] | None = None
    params: dict = field(default_factory=lambda: {"max_hold": 3, "base_ratio": 0.95})
    position_states: dict[str, PositionSignalState] = field(default_factory=dict)

    def plan_orders_at(self, current_date, previous_date, broker, decision_time, current_prices=None):
        if str(decision_time) == "14:50":
            return self._plan_1450_atr(broker, current_prices or {})
        if str(decision_time) != "09:35":
            raise ValueError("candidate supports only 09:35 and 14:50")
        # score once, plan all sells first, calculate real empty slots, then rank entries
```

`on_orders_processed` creates entry state only for a confirmed positive fill and removes state only for a confirmed sell fill. `on_after_close` may receive only already-completed close marks and updates `highest_close` monotonically.

- [ ] **Step 5: Run the full research-module test file and verify GREEN**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the planner milestone**

```powershell
git add cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py
git commit -m "research: add weekly pullback order planner"
```

---

### Task 5: Build the Standalone JoinQuant Snapshot and Signal Layer

**Files:**
- Create: `cross_signal_strategy/smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py`
- Create: `tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py`

**Interfaces:**
- Consumes: JoinQuant `get_price`, explicit `signal_date`, and explicit `decision_date`.
- Produces: `get_default_params()`, `get_default_etf_pool()`, indicator helpers, `_snapshot_from_daily_frame(code, signal_date, decision_date, frame)`, `load_signal_snapshot(code, signal_date, decision_date, return_reason=False)`, and platform-compatible copies of the frozen signal predicates/state.

- [ ] **Step 1: Write the failing upload-compatibility test**

```python
def test_joinquant_candidate_compiles_without_future_annotations_or_dataclasses():
    source = CANDIDATE_PATH.read_text(encoding="utf-8")
    compiled = compile(source, str(CANDIDATE_PATH), "exec", dont_inherit=True)
    assert compiled.co_flags & __future__.annotations.compiler_flag == 0
    namespace = {"__name__": "weekly_pullback_joinquant_probe"}
    exec(compiled, namespace)
    state = namespace["PositionSignalState"]("2021-03-08", 10.0, 0.2, 10.0)
    assert state.entry_atr == pytest.approx(0.2)
```

Inject a fake `jqdata` module as existing JoinQuant candidate tests do. Block imports of `dataclasses` during `exec`.

- [ ] **Step 2: Write failing data-boundary and parity tests**

Test these exact contracts:

```python
def test_loader_requests_daily_bars_ending_at_explicit_t1():
    snapshot, reason = candidate.load_signal_snapshot(
        code, signal_date=date(2021, 3, 9), decision_date=date(2021, 3, 10), return_reason=True
    )
    assert get_price_calls[0]["end_date"] == date(2021, 3, 9)
    assert snapshot["signal_date"] == "2021-03-09"
    assert snapshot["weekly_period_end"] == "2021-03-07"
    assert snapshot["weekly_last_trade_date"] == "2021-03-05"

def test_loader_fails_closed_when_last_daily_bar_is_older_than_t1():
    result, reason = candidate._snapshot_from_daily_frame(
        "513100.XSHG",
        date(2021, 3, 9),
        date(2021, 3, 10),
        daily_frame.loc[:"2021-03-08"],
    )
    assert result is None
    assert reason == "stale_signal_date"

def test_joinquant_predicates_match_research_predicates():
    assert candidate.is_entry_eligible(snapshot) == research.is_entry_eligible(snapshot)
    joinquant_state = candidate.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    research_state = research.PositionSignalState("2021-03-01", 10.0, 0.2, 11.0)
    assert candidate.choose_exit_reason(
        joinquant_state, snapshot, 10.80, 8, "513100.XSHG"
    ) == research.choose_exit_reason(
        research_state, snapshot, 10.80, 8, "513100.XSHG"
    )
```

Also test the partial decision week is excluded, 21 completed weeks are required, malformed data fails closed, and zero recent volume is rejected.

- [ ] **Step 3: Run JoinQuant snapshot tests and verify RED**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py -k "compile or loader or snapshot or parity" -q
```

Expected: FAIL because the JoinQuant candidate file does not exist.

- [ ] **Step 4: Implement the platform-compatible snapshot layer**

Create a standalone file with:

```python
STRATEGY_VERSION = "weekly-trend-pullback-v0.1-joinquant-candidate"
DEPLOYMENT_BUILD_ID = "20260826.1-candidate"
LOOKBACK = 180
```

Use `get_price(code, end_date=signal_date, count=LOOKBACK, frequency="daily", fields=["open", "high", "low", "close", "volume"], skip_paused=True, fq="pre", panel=False)`. Compute daily RSI6, KDJ(9,3,3), BOLL(20,2), and ATR14 from the exact T-1 frame. Aggregate weekly data only after excluding the decision week. Include all daily and weekly values needed by entry, exit, ranking, and logs in one snapshot.

Do not import the research module from the upload file; duplicate the small frozen primitives so the uploaded script remains self-contained, then keep parity tests as the guard against drift.

- [ ] **Step 5: Run snapshot and compatibility tests and verify GREEN**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py -k "compile or loader or snapshot or parity" -q
```

Expected: PASS.

- [ ] **Step 6: Commit the JoinQuant signal milestone**

```powershell
git add cross_signal_strategy/smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py
git commit -m "research: add weekly pullback JoinQuant signals"
```

---

### Task 6: Implement JoinQuant Scheduling, Orders, and Lifecycle Protection

**Files:**
- Modify: `cross_signal_strategy/smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py`
- Modify: `tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py`

**Interfaces:**
- Consumes: JoinQuant context, exact trade calendar T-1, current quotes, candidate snapshots, and portfolio positions.
- Produces: `initialize(context)`, `do_trading(context)`, `check_atr_1450(context)`, `plan_0935_orders(snapshots, held_codes, position_states, current_prices, today, trade_days, total_value, sold_today=None, params=None)`, `classify_sell_submission(order, before_amount, after_amount)`, `_record_sell_submission(code, order, before_amount, after_amount)`, and stable logs.

- [ ] **Step 1: Write failing schedule and 14:50 isolation tests**

```python
def test_initialize_registers_only_0935_and_1450():
    candidate.initialize(SimpleNamespace())
    assert scheduled == [("do_trading", "09:35"), ("check_atr_1450", "14:50")]
    assert ("avoid_future_data", True) in options

def test_1450_never_loads_daily_or_weekly_signals():
    monkeypatch.setattr(candidate, "load_signal_snapshot", exploding_loader)
    candidate.check_atr_1450(context)
    assert orders == [(stopped_code, 0)]
```

- [ ] **Step 2: Write failing order-sequence and lifecycle tests**

Cover all frozen behaviors:

```python
def test_0935_executes_all_planned_sells_before_any_buy():
    candidate.do_trading(context)
    assert [item[0] for item in orders] == ["sell", "buy"]

@pytest.mark.parametrize("order,before,after,expected", [
    (None, 100, 100, "rejected"),
    (SimpleNamespace(filled=0, status="held"), 100, 100, "pending"),
    (SimpleNamespace(filled=40, status="held"), 100, 60, "partial_pending"),
    (SimpleNamespace(filled=40, status="canceled"), 100, 60, "partial"),
    (SimpleNamespace(filled=100, status="filled"), 100, 0, "full"),
])
def test_sell_submission_state_machine(order, before, after, expected):
    assert candidate.classify_sell_submission(order, before, after) == expected

def test_rejected_0935_atr_sell_can_retry_at_1450(monkeypatch):
    candidate.g.pending_sells = set()
    candidate.g.sold_today = set()
    assert candidate._record_sell_submission(code, None, 100, 100) == "rejected"
    assert code not in candidate.g.pending_sells
    monkeypatch.setattr(candidate, "get_current_data", lambda: stopped_quote)
    monkeypatch.setattr(candidate, "order_target", recording_sell)
    candidate.check_atr_1450(context_with_position)
    assert sell_attempts == [(code, 0)]

def test_active_pending_sell_is_not_duplicated_at_1450(monkeypatch):
    candidate.g.pending_sells = set()
    candidate.g.sold_today = set()
    order = SimpleNamespace(filled=0, status="held")
    assert candidate._record_sell_submission(code, order, 100, 100) == "pending"
    monkeypatch.setattr(candidate, "order_target", recording_sell)
    candidate.check_atr_1450(context_with_position)
    assert sell_attempts == []

def test_buy_state_is_created_only_after_a_confirmed_fill(monkeypatch):
    monkeypatch.setattr(candidate, "order_target_value", lambda code, value: None)
    candidate.do_trading(empty_context)
    assert candidate.g.position_states == {}

    def filled_buy(code, value):
        empty_context.portfolio.positions[code] = SimpleNamespace(
            total_amount=600, avg_cost=10.01
        )
        return SimpleNamespace(filled=600, status="filled")

    monkeypatch.setattr(candidate, "order_target_value", filled_buy)
    candidate.do_trading(empty_context)
    assert candidate.g.position_states[eligible_code].entry_atr == pytest.approx(0.2)
```

The last three tests must construct full fake contexts and assert exact calls/state, not merely call helper functions.

- [ ] **Step 3: Run execution tests and verify RED**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py -k "initialize or 1450 or submission or retry or pending or executes or fill" -q
```

Expected: FAIL because orchestration and lifecycle functions are absent.

- [ ] **Step 4: Implement initialization and execution flow**

Initialize only candidate-owned state:

```python
g.params = get_default_params()
g.etf_pool = get_default_etf_pool()
g.position_states = {}
g.last_snapshots = {}
g.sold_today = set()
g.pending_sells = set()
g.sold_guard_date = None
run_daily(do_trading, time="09:35")
run_daily(check_atr_1450, time="14:50")
```

At 09:35, obtain T-1 from `get_trade_days(end_date=today, count=2)`, load each ETF once with explicit signal and decision dates, collect valid current prices, plan all exits before entries, execute sell plans before buy plans, and only freeze `entry_atr` after a position or positive fill proves execution.

At 14:50, iterate actual holdings, skip active pending sells, calculate only the frozen ATR stop, and apply `_record_sell_submission`. Use `builtins.any` explicitly in order-state classification so `from jqdata import *` cannot shadow the builtin.

- [ ] **Step 5: Add precise causal logs**

Each 09:35 snapshot log must include decision date, signal date, completed week end, weekly close/MA20/current slope, daily BOLL/KDJ/RSI/ATR values, and exact eligibility. Each order log must include reason, current execution price, target value or zero target, and classified result. The 14:50 log must contain ATR state only and must not mention recomputed indicator values.

- [ ] **Step 6: Run the complete JoinQuant candidate tests and verify GREEN**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the executable-candidate milestone**

```powershell
git add cross_signal_strategy/smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py
git commit -m "research: complete weekly pullback JoinQuant candidate"
```

---

### Task 7: Verify Isolation, Regression Safety, and Upload Readiness

**Files:**
- Modify: `tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py`
- Modify: `cross_signal_strategy/docs/decisions.md`

**Interfaces:**
- Consumes: completed candidate files, all relevant tests, formal file hashes, and the approved design spec.
- Produces: a reproducible implementation decision record and a candidate ready for one 2019-2021 JoinQuant run, not a performance or live-readiness claim.

- [ ] **Step 1: Write the failing formal-integrity test**

```python
def test_formal_cross_signal_files_keep_design_time_sha256():
    assert sha256(FORMAL_JOINQUANT.read_bytes()).hexdigest().upper() == (
        "9ADB96E523BBA2B1E5C42CB2BDDA06A8BB06065EA1AE25194651177987DF0F52"
    )
    assert sha256(FORMAL_PTRADE.read_bytes()).hexdigest().upper() == (
        "E4FA39CC79350A8E790074E1D3C75D0A4638ECA23DB8155F4EE1203869E7D6F8"
    )
```

Run it before adding imports and expect RED due to missing `sha256`/path constants in the new test file; then add only those test helpers. Do not modify formal files to make this test pass.

- [ ] **Step 2: Run targeted candidate and governance suites**

```powershell
python -m pytest tests/test_cross_signal_weekly_trend_daily_pullback_candidate.py tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py tests/test_cross_signal_research_budget.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full repository test suite**

```powershell
python -m pytest -q
```

Expected: PASS. If unrelated pre-existing failures occur, record exact test names and demonstrate the targeted suites still pass; do not alter unrelated strategy behavior.

- [ ] **Step 4: Verify source integrity and compile readiness**

```powershell
Get-FileHash -Algorithm SHA256 cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py,cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
python -m py_compile cross_signal_strategy/research/weekly_trend_daily_pullback_candidate.py cross_signal_strategy/smart_trade_joinquant_weekly_trend_daily_pullback_candidate.py
git diff --check
git status --short
```

Expected: formal hashes equal the fixed values, both candidate modules compile, and no unplanned formal/PTrade/validation artifacts appear.

- [ ] **Step 5: Record the implementation milestone**

Append a decision entry stating:

```markdown
### Prepare Independent Weekly-Trend Daily-Pullback JoinQuant Candidate

- Decision: Implement the single user-authorized `weekly-trend-pullback-v0.1-joinquant-candidate` as an isolated research and JoinQuant pair.
- Causal boundary: exact T-1 daily signals, the decision week's partial weekly bar excluded, 09:35 full processing, and 14:50 ATR only.
- Research boundary: no market replay or validation data was read during implementation; the family remains open for exactly one 2019-2021 JoinQuant run.
- Business boundary: formal JoinQuant/PTrade hashes remain unchanged; no PTrade candidate or live-readiness conclusion exists.
```

- [ ] **Step 6: Commit the verified implementation milestone**

```powershell
git add tests/test_cross_signal_weekly_trend_daily_pullback_joinquant_candidate.py cross_signal_strategy/docs/decisions.md
git commit -m "test: verify weekly pullback candidate isolation"
```

- [ ] **Step 7: Hand off the one authorized JoinQuant run**

Report the exact candidate path, version/build identity, passing test commands, formal hashes, and the frozen 2019-2021 normal/doubled-friction gate. State explicitly that code readiness is not strategy validation and that no neighboring run is authorized if the first result fails.
