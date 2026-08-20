# Cross-Signal 14:45 Training Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate one isolated 2019-2021 local candidate that preserves the official 09:35 path and adds a causal 14:45 full buy/sell decision from completed minutes through 14:44.

**Architecture:** Add a pure point-in-time frame builder, a dual-timepoint signal adapter, and an isolated two-batch local replay that shares one broker and one planner state. Compare the unchanged official local replay with the single fixed 14:45 candidate, apply the pre-registered gates, and close the research family without touching formal JoinQuant/PTrade or any multi-factor file.

**Tech Stack:** Python 3, pandas, dataclasses, pytest, existing cross-signal local loader/backtester/planner, existing JoinQuant core scoring functions.

**Spec:** `docs/superpowers/specs/2026-08-21-cross-signal-dual-timepoint-1445-design.md`

## Global Constraints

- Scope is `cross_signal_strategy` only; no multi-factor file may change.
- Training decisions and performance dates are exactly `2019-01-01` through `2021-12-31`.
- `G:\financial\history_data\cross_signal_train_2019_2021` and `G:\financial\history_data\cross_signal_warmup_2018` are read-only; never write, delete, clean, or cache inside them.
- 2018 may supply indicator warm-up only and may not enter performance or rule selection.
- Do not read reserved validation, pressure, recent-market, full-period, or 2026 price data.
- The only candidate time is `14:45`; signal minutes must have interval start `< 14:45`, so the last eligible minute label is `14:44`.
- Preserve all official indicators, parameters, scores, thresholds, ETF pool, ranking, sizing, five-day hold, price confirmation, ADX protection, ATR construction, fees, and 09:35 execution.
- Use raw cumulative partial-day volume; do not project or normalize it.
- At 14:45, current execution price is not a signal input. The local causal execution proxy is the `open` of the `14:45` minute; never use that minute's close/high/low to form the signal.
- Existing positions keep frozen entry ATR and highest completed daily-close anchor. A 14:45 new buy stores the provisional T-day ATR.
- Every behavioral change starts with a focused failing test, followed by the minimal implementation and a passing focused test.
- JoinQuant and PTrade candidates are explicitly out of scope for this plan. If the local gate passes, stop and write a separate JoinQuant plan. If it fails, do not create a JoinQuant/PTrade candidate.
- Preserve the unrelated untracked file `cross_signal_strategy/docs/2026-08-20-cross-signal-handoff.md`.

## File Map

- Create `cross_signal_strategy/local/intraday_signal_frame.py`: validate and aggregate a causal provisional T-day bar.
- Create `cross_signal_strategy/local/dual_timepoint_signal_adapter.py`: delegate 09:35 scoring to the official local adapter and score the 14:45 provisional frame with the same core functions.
- Create `cross_signal_strategy/local/dual_timepoint_backtester.py`: execute 09:35 then 14:45 against one broker and call after-close once.
- Create `cross_signal_strategy/local/dual_timepoint_order_planner.py`: share portfolio/risk state across both decisions and enforce same-day guards.
- Create `cross_signal_strategy/research/dual_timepoint_1445_candidate.py`: run A/B, compute fixed metrics, apply gates, and render the report.
- Create `tests/test_cross_signal_dual_timepoint_1445.py`: focused data, scoring, engine, planner, and gate tests.
- Modify `cross_signal_strategy/local/local_signal_adapter.py`: extract one reusable score-from-frame method without changing official 09:35 results.
- Modify `cross_signal_strategy/local/local_backtester.py`: accept optional broker friction kwargs while preserving defaults.
- Modify `cross_signal_strategy/local/local_data_loader.py`: expose a defensive one-day minute slice so the single fixed run does not repeatedly copy an entire year; preserve the existing whole-frame API.
- Modify `tests/test_cross_signal_local_data_loader.py`: prove the one-day slice is exact and defensive.
- Modify `cross_signal_strategy/research/trade_quality_ledger.py`: accept a same-day entry signal only when its audit proves the fixed 14:45 decision and 14:44 cutoff; preserve rejection of every other same-day or future signal.
- Modify `tests/test_cross_signal_trade_quality_ledger.py`: cover the narrow point-in-time ledger exception and unchanged leakage rejection.
- Modify `cross_signal_strategy/docs/research_budget.json`, `cross_signal_strategy/docs/research_budget.md`, and `tests/test_cross_signal_research_budget.py`: open exactly one family before running it, then close it after the fixed run.
- Modify either `cross_signal_strategy/docs/failed_experiments.md` or `cross_signal_strategy/docs/decisions.md`, plus `cross_signal_strategy/docs/backtest_notes.md`: record the one empirical outcome.
- Generate `cross_signal_strategy/reports/dual_timepoint_1445_2019_2021.md`: immutable human-readable result for the consumed candidate.

---

### Task 1: Pre-register the one allowed research family

**Files:**
- Modify: `tests/test_cross_signal_research_budget.py:62-120`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md:6-78`

**Interfaces:**
- Consumes: `load_research_budget(path)` and `evaluate_experiment_request(budget, family_key, planned_variants)`.
- Produces: open family key `intraday_signal_clock_1445_user_authorized` with exactly one permitted variant.

- [ ] **Step 1: Write the failing governance test**

Append this test:

```python
def test_user_authorized_1445_signal_clock_is_the_only_open_family():
    budget = load_research_budget(BUDGET)
    families = {family.key: family for family in budget.families}
    family = families["intraday_signal_clock_1445_user_authorized"]

    assert budget.max_total_open_experiments == 1
    assert family.status == "open"
    assert family.max_new_experiments == 1
    assert family.planned_experiment == (
        "keep the official 09:35 path and add one 14:45 full signal pass "
        "using completed 1-minute bars through 14:44"
    )
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is True
    assert [item.key for item in budget.families if item.status == "open"] == [
        "intraday_signal_clock_1445_user_authorized"
    ]

    raw = json.loads(BUDGET.read_text(encoding="utf-8"))
    payload = next(item for item in raw["families"] if item["key"] == family.key)
    assert payload["decision_times"] == ["09:35", "14:45"]
    assert payload["signal_cutoff"] == "14:44"
    assert payload["candidate_variants"] == 1
    assert payload["validation_influence"] == "none"
    assert payload["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert payload["prohibit_alternatives"] is True
```

- [ ] **Step 2: Run the test and verify the family is absent**

Run:

```powershell
pytest -q tests/test_cross_signal_research_budget.py::test_user_authorized_1445_signal_clock_is_the_only_open_family
```

Expected: FAIL because `intraday_signal_clock_1445_user_authorized` does not exist.

- [ ] **Step 3: Add the exact open-family record**

Set top-level `max_total_open_experiments` to `1` and append this object to the JSON `families` array:

```json
{
  "key": "intraday_signal_clock_1445_user_authorized",
  "label": "User-authorized fixed 14:45 point-in-time signal pass",
  "status": "open",
  "max_new_experiments": 1,
  "rationale": "The user explicitly authorized one new structural family: preserve the official 09:35 decision and add one causal 14:45 full signal pass from completed minutes through 14:44. This is not a search over execution times or indicator rules.",
  "planned_experiment": "keep the official 09:35 path and add one 14:45 full signal pass using completed 1-minute bars through 14:44",
  "decision_times": ["09:35", "14:45"],
  "signal_cutoff": "14:44",
  "candidate_variants": 1,
  "validation_influence": "none",
  "data_scope": "2018_warmup_plus_2019_2021_training_only",
  "prohibit_alternatives": true
}
```

Update the readable map so `Current Accounting`, `Frozen Families`, and `Open Families` all state that this is the only open family and that MACD/KDJ/ADX/direct-sell/execution-wait families remain exhausted.

- [ ] **Step 4: Run the governance tests**

Run:

```powershell
pytest -q tests/test_cross_signal_research_budget.py
```

Expected: all tests PASS and the audit still reports the existing failed-experiment count exactly.

- [ ] **Step 5: Commit the pre-registration**

```powershell
git add -- tests/test_cross_signal_research_budget.py cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md
git commit -m "docs(cross-signal): preregister 14:45 signal candidate"
```

---

### Task 2: Build the causal provisional T-day frame

**Files:**
- Create: `cross_signal_strategy/local/intraday_signal_frame.py`
- Create: `tests/test_cross_signal_dual_timepoint_1445.py`

**Interfaces:**
- Consumes: a corrected/adjusted T-1 daily frame and one raw local minute frame for T.
- Produces: `IntradaySignalFrame(frame: pd.DataFrame, audit: IntradayFrameAudit)` from `build_intraday_signal_frame`.

- [ ] **Step 1: Write failing aggregation and cutoff tests**

Create the test file with these fixtures and assertions:

```python
import pathlib

import pandas as pd
import pytest


TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def _t1_frame():
    return pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03"],
        "open": [9.8, 10.0], "high": [10.2, 10.3],
        "low": [9.7, 9.9], "close": [10.0, 10.1],
        "volume": [1000.0, 1100.0],
    })


def _minutes():
    return pd.DataFrame([
        {"date": "2020-01-06", "time": "09:30", "prev_close": 10.1,
         "open": 10.2, "high": 10.3, "low": 10.1, "close": 10.25, "volume": 100},
        {"date": "2020-01-06", "time": "14:44", "prev_close": 10.1,
         "open": 10.4, "high": 10.6, "low": 10.35, "close": 10.5, "volume": 200},
        {"date": "2020-01-06", "time": "14:45", "prev_close": 10.1,
         "open": 99.0, "high": 100.0, "low": 1.0, "close": 99.0, "volume": 999999},
    ])


def test_1445_frame_uses_only_completed_minutes_through_1444():
    from cross_signal_strategy.local.intraday_signal_frame import (
        build_intraday_signal_frame,
    )

    result = build_intraday_signal_frame(
        _t1_frame(), _minutes(), "2020-01-06", decision_time="14:45"
    )
    bar = result.frame.iloc[-1]
    assert bar["date"] == "2020-01-06"
    assert bar["open"] == pytest.approx(10.2)
    assert bar["high"] == pytest.approx(10.6)
    assert bar["low"] == pytest.approx(10.1)
    assert bar["close"] == pytest.approx(10.5)
    assert bar["volume"] == pytest.approx(300.0)
    assert result.audit.decision_time == "14:45"
    assert result.audit.data_cutoff == "14:44"
    assert result.audit.last_minute == "14:44"
    assert result.audit.minute_count == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate", "out_of_order", "cross_day", "daily_out_of_order",
        "t_day_daily", "bad_prev_close",
    ],
)
def test_1445_frame_fails_closed_on_ambiguous_or_misaligned_data(mutation):
    from cross_signal_strategy.local.intraday_signal_frame import build_intraday_signal_frame

    daily, minutes = _t1_frame(), _minutes()
    if mutation == "duplicate":
        minutes = pd.concat([minutes, minutes.iloc[[0]]], ignore_index=True)
    elif mutation == "out_of_order":
        minutes = minutes.iloc[[1, 0, 2]].reset_index(drop=True)
    elif mutation == "cross_day":
        minutes.loc[0, "date"] = "2020-01-03"
    elif mutation == "daily_out_of_order":
        daily = daily.iloc[::-1].reset_index(drop=True)
    elif mutation == "t_day_daily":
        daily = pd.concat([daily, daily.iloc[[-1]].assign(date="2020-01-06")])
    else:
        minutes.loc[:, "prev_close"] = 10.9
    with pytest.raises(ValueError):
        build_intraday_signal_frame(daily, minutes, "2020-01-06", "14:45")
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the immutable audit and builder**

Create these public interfaces:

```python
@dataclass(frozen=True)
class IntradayFrameAudit:
    decision_time: str
    data_cutoff: str
    last_minute: str
    minute_count: int
    partial_volume: bool = True


@dataclass(frozen=True)
class IntradaySignalFrame:
    frame: pd.DataFrame
    audit: IntradayFrameAudit


def build_intraday_signal_frame(
    t1_daily_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    trade_date: str,
    decision_time: str = "14:45",
) -> IntradaySignalFrame:
    if str(decision_time)[:5] != "14:45":
        raise ValueError("Only the pre-registered 14:45 decision is allowed")
    trade_day = pd.Timestamp(trade_date).normalize()
    daily = t1_daily_frame.copy()
    daily_dates = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    if daily.empty or (daily_dates >= trade_day).any():
        raise ValueError("T-1 frame must contain completed dates before T only")
    if daily_dates.duplicated().any() or not daily_dates.is_monotonic_increasing:
        raise ValueError("T-1 dates must be unique and ordered")

    minutes = minute_frame.copy()
    if set(minutes["date"].astype(str)) != {trade_day.strftime("%Y-%m-%d")}:
        raise ValueError("Minute frame must contain exactly one requested trade date")
    timestamps = pd.to_datetime(
        minutes["date"].astype(str) + " " + minutes["time"].astype(str),
        errors="raise",
    )
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and ordered")
    cutoff = trade_day + pd.Timedelta(hours=14, minutes=45)
    visible = minutes.loc[timestamps < cutoff].copy()
    if visible.empty:
        raise ValueError("No completed minute before 14:45")

    numeric = visible[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Invalid point-in-time OHLCV")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid minute high")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid minute low")
    if (numeric["volume"] < 0).any():
        raise ValueError("Invalid minute volume")
    if round(float(visible.iloc[0]["prev_close"]), 3) != round(float(daily.iloc[-1]["close"]), 3):
        raise ValueError("Daily/minute adjustment boundary mismatch")

    partial = {
        "date": trade_day.strftime("%Y-%m-%d"),
        "open": float(numeric.iloc[0]["open"]),
        "high": float(numeric["high"].max()),
        "low": float(numeric["low"].min()),
        "close": float(numeric.iloc[-1]["close"]),
        "volume": float(numeric["volume"].sum()),
    }
    combined = pd.concat([daily, pd.DataFrame([partial])], ignore_index=True)
    return IntradaySignalFrame(
        frame=combined,
        audit=IntradayFrameAudit(
            decision_time="14:45",
            data_cutoff="14:44",
            last_minute=str(visible.iloc[-1]["time"])[:5],
            minute_count=len(visible),
        ),
    )
```

Also require all columns `date,time,prev_close,open,high,low,close,volume`; reject missing columns with one explicit `ValueError` before indexing.

- [ ] **Step 4: Run the point-in-time tests**

Run:

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py
```

Expected: all current tests PASS.

- [ ] **Step 5: Commit the causal frame**

```powershell
git add -- cross_signal_strategy/local/intraday_signal_frame.py tests/test_cross_signal_dual_timepoint_1445.py
git commit -m "feat(cross-signal): build causal 14:45 signal frame"
```

---

### Task 3: Score 09:35 and 14:45 through one official scoring path

**Files:**
- Modify: `cross_signal_strategy/local/local_signal_adapter.py:94-139`
- Create: `cross_signal_strategy/local/dual_timepoint_signal_adapter.py`
- Modify: `tests/test_cross_signal_dual_timepoint_1445.py`
- Test: `tests/test_cross_signal_local_signal_adapter.py`

**Interfaces:**
- Consumes: `IntradaySignalFrame`, `LocalSignalAdapter.load_signal_frame`, and official `build_signal_snapshot/score_buy_snapshot/score_sell_snapshot`.
- Produces: `LocalSignalAdapter.score_frame` and `DualTimepointSignalAdapter.score_at`.

- [ ] **Step 1: Write failing parity and cutoff tests**

Add this concrete adapter stub beside the Task 2 fixtures. It isolates the
time-routing contract; the adjacent real-adapter regression below proves that
the official calculations are unchanged.

```python
class _MinuteLoaderStub:
    def load_minute_frame(self, code, current_date):
        assert code == "510300"
        assert current_date == "2020-01-06"
        return _minutes()


class _BaselineAdapterStub:
    def __init__(self):
        self.loader = _MinuteLoaderStub()
        self.morning_score = {
            "code": "510300",
            "signal_date": "2020-01-03",
            "decision_time": "09:35",
        }

    def score(self, code, current_date, return_reason=False):
        value = dict(self.morning_score)
        return (value, None) if return_reason else value

    def load_signal_frame(self, code, current_date):
        return _t1_frame(), "2020-01-03"

    def score_frame(self, code, current_date, frame, signal_date, metadata=None):
        value = {
            "code": code,
            "current_date": current_date,
            "signal_date": signal_date,
            "max_data_date": str(frame["date"].max()),
            "rsi6": 51.0,
            "k": 55.0,
            "dif": 0.02,
            "adx": 18.0,
            "boll_mid": 10.0,
            "ma20": 9.9,
            "atr": 0.2,
        }
        value.update(dict(metadata or {}))
        return value, None


def test_dual_adapter_0935_delegates_and_1445_scores_partial_t_bar():
    from cross_signal_strategy.local.dual_timepoint_signal_adapter import (
        DualTimepointSignalAdapter,
    )

    baseline = _BaselineAdapterStub()
    adapter = DualTimepointSignalAdapter(baseline)
    morning = adapter.score_at("510300", "2020-01-06", "09:35")
    afternoon = adapter.score_at("510300", "2020-01-06", "14:45")

    assert morning == baseline.morning_score
    assert afternoon["signal_date"] == "2020-01-06"
    assert afternoon["decision_time"] == "14:45"
    assert afternoon["data_cutoff"] == "14:44"
    assert afternoon["max_data_date"] == "2020-01-06"
    for field in ("rsi6", "k", "dif", "adx", "boll_mid", "ma20", "atr"):
        assert pd.notna(afternoon[field])
```

In `tests/test_cross_signal_local_signal_adapter.py`, extend
`test_signal_score_matches_strategy_snapshot_scoring_after_lookback` before the
refactor: call `load_signal_frame("510300", "2019-07-01")`, then call the new
`score_frame` directly and assert `(direct, direct_reason) == (score, reason)`.
This is exact dictionary equality, not approximate metric equality.

- [ ] **Step 2: Run focused tests and verify missing interfaces**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_local_signal_adapter.py
```

Expected: FAIL because `score_frame` and `DualTimepointSignalAdapter` do not exist.

- [ ] **Step 3: Extract score-from-frame without changing official behavior**

Add to `LocalSignalAdapter`:

```python
def score_frame(
    self,
    code: str,
    current_date: str,
    frame: pd.DataFrame,
    signal_date: str,
    metadata: dict | None = None,
):
    p = self.params or strategy.get_default_params()
    min_len = self._local_min_len(p)
    required = ["rsi6", "rsi12", "rsi24", "dif", "dea", "k", "d", "j", "ma20", "atr", "adx"]
    reason = strategy.score_skip_reason(frame, None, required, min_len)
    if reason is not None:
        return None, reason
    snapshot = strategy.build_signal_snapshot(frame, p)
    self._suppress_float_artifact_flags(snapshot, frame)
    reason = strategy.score_skip_reason(frame, snapshot, required, min_len)
    if reason is not None:
        return None, reason
    result = {**snapshot, **strategy.score_buy_snapshot(snapshot, p), **strategy.score_sell_snapshot(snapshot)}
    result.update({
        "code": str(code).split(".")[0],
        "current_date": pd.Timestamp(current_date).strftime("%Y-%m-%d"),
        "signal_date": str(signal_date),
        "max_data_date": str(frame["date"].max()),
    })
    result.update(dict(metadata or {}))
    return result, None
```

Make existing `score()` call this method after `load_signal_frame` and retain the existing cache key and return semantics.

- [ ] **Step 4: Implement the dual-timepoint adapter**

Create:

```python
@dataclass(frozen=True)
class DualTimepointSignalAdapter:
    baseline: LocalSignalAdapter
    _score_cache: dict = field(default_factory=dict, init=False, repr=False)

    def score_at(self, code, current_date, decision_time, return_reason=False):
        time_text = str(decision_time)[:5]
        if time_text == "09:35":
            return self.baseline.score(code, current_date, return_reason=return_reason)
        if time_text != "14:45":
            raise ValueError("Only 09:35 and 14:45 are allowed")
        key = (str(code).split(".")[0], str(current_date), time_text)
        if key not in self._score_cache:
            t1_frame, _ = self.baseline.load_signal_frame(code, current_date)
            minutes = self.baseline.loader.load_minute_frame(code, current_date)
            point = build_intraday_signal_frame(t1_frame, minutes, current_date, time_text)
            self._score_cache[key] = self.baseline.score_frame(
                code,
                current_date,
                point.frame,
                signal_date=str(current_date),
                metadata={
                    "decision_time": time_text,
                    "data_cutoff": point.audit.data_cutoff,
                    "last_minute": point.audit.last_minute,
                    "minute_count": point.audit.minute_count,
                    "partial_volume": True,
                },
            )
        result, reason = self._score_cache[key]
        copied = dict(result) if result is not None else None
        return (copied, reason) if return_reason else copied
```

The adapter must not accept a current quote argument and must not load the T-day final daily row.

- [ ] **Step 5: Run adapter and adjacent scoring tests**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_training_run.py tests/test_cross_signal_order_path_diagnostics.py
```

Expected: PASS.

- [ ] **Step 6: Commit the shared scoring path**

```powershell
git add -- cross_signal_strategy/local/local_signal_adapter.py cross_signal_strategy/local/dual_timepoint_signal_adapter.py tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_local_signal_adapter.py
git commit -m "refactor(cross-signal): share point-in-time scoring path"
```

---

### Task 4: Execute two causal batches with shared state

**Files:**
- Modify: `cross_signal_strategy/local/local_backtester.py:37-58,173-250`
- Create: `cross_signal_strategy/local/dual_timepoint_backtester.py`
- Create: `cross_signal_strategy/local/dual_timepoint_order_planner.py`
- Modify: `tests/test_cross_signal_dual_timepoint_1445.py`
- Test: `tests/test_cross_signal_local_backtester.py`
- Test: `tests/test_cross_signal_local_order_planner.py`

**Interfaces:**
- Consumes: `DualTimepointSignalAdapter.score_at`, `LocalBroker`, `DayResult`, and official order-planner rules.
- Produces: `DualTimepointBacktestEngine.run` and `DualTimepointOrderPlanner.plan_orders_at`.

- [ ] **Step 1: Write failing engine-order tests**

Add tests proving event order, one broker, and causal prices:

```python
class _DualEngineLoader:
    MORNING_CLOSE_WITH_SLIPPAGE = 10.01
    AFTERNOON_OPEN_WITH_SLIPPAGE = 20.02

    def get_minute_bar(self, code, current_date, trade_time):
        assert current_date == "2020-01-06"
        assert code in {"AAA", "BBB"}
        assert trade_time in {"09:35", "14:45"}
        return {
            "open": 20.0 if trade_time == "14:45" else 9.9,
            "close": 10.0 if trade_time == "09:35" else 20.1,
            "volume": 1000.0,
            "num_trades": 10.0,
        }

    def load_daily_frame(self, code, current_date):
        return pd.DataFrame([{"date": current_date, "close": 15.0}])


class _RecordingPlanner:
    params = {"max_hold": 3}

    def __init__(self):
        self.calls = []

    def plan_orders_at(self, current_date, previous_date, broker, decision_time, current_prices=None):
        self.calls.append(("plan", current_date, decision_time))
        code = "AAA" if decision_time == "09:35" else "BBB"
        return [{"code": code, "target_value": 5000.0, "reason": "buy_signal"}]

    def on_orders_processed(self, current_date, decision_time, plans, results):
        self.calls.append(("processed", current_date, decision_time))

    def on_after_close(self, current_date, marks):
        self.calls.append(("after_close", current_date))


def test_dual_engine_runs_0935_then_1445_and_marks_close_once():
    planner = _RecordingPlanner()
    engine = DualTimepointBacktestEngine(_DualEngineLoader(), initial_cash=20000.0)
    days = engine.run(["2020-01-06"], planner)

    assert planner.calls == [
        ("plan", "2020-01-06", "09:35"),
        ("processed", "2020-01-06", "09:35"),
        ("plan", "2020-01-06", "14:45"),
        ("processed", "2020-01-06", "14:45"),
        ("after_close", "2020-01-06"),
    ]
    assert [order.side_time[-5:] for order in days[0].orders] == ["09:35", "14:45"]
    assert days[0].orders[0].exec_price == pytest.approx(_DualEngineLoader.MORNING_CLOSE_WITH_SLIPPAGE)
    assert days[0].orders[1].exec_price == pytest.approx(_DualEngineLoader.AFTERNOON_OPEN_WITH_SLIPPAGE)
```

Add a small-fixture parity test and this full training-path regression. The
full test preserves the current frozen `cross-v0.3.3` 181-event order
alignment by proving that the new engine's morning-only mode is exactly the old
local engine, event for event:

```python
def _filled_signature(days):
    return [
        (
            day.date,
            "BUY" if order.amount_delta > 0 else "SELL",
            str(order.code).split(".")[0],
            abs(order.amount_delta),
            order.reason,
        )
        for day in days for order in day.orders
        if order.filled and order.amount_delta != 0
    ]


def _all_order_signature(days):
    return [
        (
            day.date,
            order.code,
            order.amount_delta,
            order.exec_price,
            order.commission,
            order.side_time,
            order.filled,
            order.reason,
        )
        for day in days for order in day.orders
    ]


def test_full_training_morning_only_dual_engine_matches_official_local_path():
    from cross_signal_strategy.local.local_backtester import LocalBacktestEngine
    from cross_signal_strategy.local.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
    from cross_signal_strategy.local.dual_timepoint_backtester import DualTimepointBacktestEngine
    from cross_signal_strategy.local.dual_timepoint_order_planner import DualTimepointOrderPlanner
    from cross_signal_strategy.local.dual_timepoint_signal_adapter import DualTimepointSignalAdapter
    from cross_signal_strategy.local_training_run import (
        build_training_signal_adapter,
        get_training_trade_dates,
    )

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    dates = get_training_trade_dates(loader)
    baseline_adapter = build_training_signal_adapter(loader)
    baseline_planner = LocalCrossSignalOrderPlanner(
        baseline_adapter, trade_dates=dates
    )
    baseline_days = LocalBacktestEngine(loader, 20000.0).run(
        dates, baseline_planner.plan_orders
    )

    candidate_adapter = DualTimepointSignalAdapter(
        build_training_signal_adapter(loader)
    )
    candidate_planner = DualTimepointOrderPlanner(
        candidate_adapter, trade_dates=dates
    )
    candidate_days = DualTimepointBacktestEngine(
        loader, 20000.0, decision_times=("09:35",)
    ).run(dates, candidate_planner)

    assert len(_filled_signature(baseline_days)) == 181
    assert _filled_signature(candidate_days) == _filled_signature(baseline_days)
    assert _all_order_signature(candidate_days) == _all_order_signature(baseline_days)
    assert candidate_days[-1].total_value == pytest.approx(
        baseline_days[-1].total_value
    )
```

Also prove that
`broker_kwargs={"commission_rate": 0.0006, "slippage_rate": 0.002}` actually
doubles friction while omitting `broker_kwargs` preserves the current defaults.

- [ ] **Step 2: Write failing same-day guard tests**

Define a test-only `_candidate` by copying the complete fixed score dictionary
from `tests/test_cross_signal_local_order_planner.py::candidate`; do not shorten
it, because the official filters read the price-structure and ADX fields. Add
this time-varying adapter and execution helper:

```python
class _TimeVaryingAdapter:
    def __init__(self, scores_by_time):
        self.scores_by_time = scores_by_time
        self.calls = []

    def score_at(self, code, current_date, decision_time, return_reason=False):
        self.calls.append((code, current_date, decision_time, return_reason))
        item = self.scores_by_time.get(decision_time, {}).get(code)
        if item is None:
            return (None, "no_data") if return_reason else None
        value = dict(item)
        return (value, None) if return_reason else value


def _execute_plans(broker, plans, decision_time):
    results = []
    for plan in plans:
        result = broker.order_target_value(
            plan["code"], plan["target_value"], 10.0,
            "2020-01-06 %s" % decision_time,
        )
        if result.filled:
            result.reason = plan["reason"]
        results.append(result)
    return results


def _buy_codes(plans):
    return {
        str(item["code"]).split(".")[0]
        for item in plans if float(item["target_value"]) > 0.0
    }


def _sell_codes(plans):
    return {
        str(item["code"]).split(".")[0]
        for item in plans if float(item["target_value"]) == 0.0
    }
```

Then add the exact guard tests:

```python
def test_afternoon_recomputes_candidates_but_keeps_same_day_guards():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )

    morning_aaa = _candidate("AAA", buy_score=75)
    morning_bbb = _candidate("BBB", buy_score=0)
    afternoon_aaa = _candidate("AAA", buy_score=0, sell_score=40)
    afternoon_aaa["close_below_ma20"] = True
    afternoon_bbb = _candidate("BBB", buy_score=75)
    adapter = _TimeVaryingAdapter({
        "09:35": {"AAA": morning_aaa, "BBB": morning_bbb},
        "14:45": {"AAA": afternoon_aaa, "BBB": afternoon_bbb},
    })
    planner = DualTimepointOrderPlanner(
        adapter, etf_pool=["AAA", "BBB"], trade_dates=["2020-01-06"]
    )
    broker = LocalBroker(initial_cash=20000.0)

    morning = planner.plan_orders_at("2020-01-06", None, broker, "09:35", current_prices={})
    morning_results = _execute_plans(broker, morning, "09:35")
    planner.on_orders_processed("2020-01-06", "09:35", morning, morning_results)
    afternoon = planner.plan_orders_at("2020-01-06", None, broker, "14:45", current_prices={})

    assert _buy_codes(morning) == {"AAA"}
    assert _buy_codes(afternoon) == {"BBB"}
    assert "AAA" not in _sell_codes(afternoon)
    assert {call[2] for call in adapter.calls} == {"09:35", "14:45"}


def test_morning_sell_and_failed_buy_are_not_retried_at_1445():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.local.dual_timepoint_order_planner import (
        DualTimepointOrderPlanner,
    )

    adapter = _TimeVaryingAdapter({
        "14:45": {
            "SOLD": _candidate("SOLD", buy_score=80),
            "FAILED": _candidate("FAILED", buy_score=79),
            "SAFE": _candidate("SAFE", buy_score=78),
        },
    })
    planner = DualTimepointOrderPlanner(
        adapter, etf_pool=["SOLD", "FAILED", "SAFE"]
    )
    planner.execution_date = "2020-01-06"
    planner.sold_today.add("SOLD")
    planner.failed_buy_codes.add("FAILED")
    broker = LocalBroker(initial_cash=20000.0)

    orders = planner.plan_orders_at("2020-01-06", None, broker, "14:45", current_prices={})

    assert _buy_codes(orders) == {"SAFE"}
```

- [ ] **Step 3: Add friction injection without changing the default engine**

Change `LocalBacktestEngine.__init__` to:

```python
def __init__(
    self,
    loader,
    initial_cash: float,
    execution_time: str = "09:35",
    broker_kwargs: Mapping[str, object] | None = None,
) -> None:
    self.loader = loader
    self.broker = LocalBroker(initial_cash=initial_cash, **dict(broker_kwargs or {}))
    self.execution_time = str(execution_time)[:5]
```

Keep every default exactly equal to the current engine.

- [ ] **Step 4: Implement the dual engine**

Create `DualTimepointBacktestEngine` with fixed default batches:

```python
DECISION_PRICE_FIELDS = {"09:35": "close", "14:45": "open"}


class DualTimepointBacktestEngine:
    def __init__(self, loader, initial_cash, decision_times=("09:35", "14:45"), broker_kwargs=None):
        allowed = ("09:35", "14:45")
        if tuple(decision_times) not in (("09:35",), allowed):
            raise ValueError("Only morning baseline or fixed 09:35/14:45 candidate is allowed")
        self.loader = loader
        self.decision_times = tuple(decision_times)
        self.broker = LocalBroker(initial_cash=initial_cash, **dict(broker_kwargs or {}))

    def run(self, trade_dates, planner):
        results = []
        previous_date = None
        for current_date in [str(item) for item in trade_dates]:
            day_orders = []
            for decision_time in self.decision_times:
                current_prices = self._current_prices(current_date, decision_time)
                plans = planner.plan_orders_at(
                    current_date,
                    previous_date,
                    self.broker,
                    decision_time,
                    current_prices=current_prices,
                )
                batch_orders = self._execute_plans(
                    current_date, decision_time, plans, planner
                )
                planner.on_orders_processed(
                    current_date, decision_time, plans, batch_orders
                )
                day_orders.extend(batch_orders)

            marks = self._close_marks(current_date)
            planner.on_after_close(current_date, marks)
            positions = {
                code: Position(pos.code, pos.amount, pos.avg_cost)
                for code, pos in self.broker.positions.items()
            }
            results.append(DayResult(
                date=current_date,
                previous_date=previous_date,
                orders=day_orders,
                cash=self.broker.cash,
                positions=positions,
                marks=marks,
                total_value=self.broker.total_value(marks),
            ))
            previous_date = current_date
        return results

    def _current_prices(self, current_date, decision_time):
        prices = {}
        field = DECISION_PRICE_FIELDS[decision_time]
        for code in self.broker.positions:
            try:
                bar = self.loader.get_minute_bar(code, current_date, decision_time)
            except (FileNotFoundError, KeyError):
                continue
            if decision_time == "09:35" or _bar_has_executable_trade(bar):
                prices[code] = float(bar[field])
        return prices

    def _execute_plans(self, current_date, decision_time, plans, planner):
        orders = []
        max_holdings = _planner_max_holdings(planner)
        field = DECISION_PRICE_FIELDS[decision_time]
        for plan in plans:
            code = str(plan["code"])
            target_value = float(plan["target_value"])
            reason = str(plan.get("reason", ""))
            if (
                target_value > 0.0
                and code not in self.broker.positions
                and max_holdings is not None
                and len(self.broker.positions) >= max_holdings
            ):
                orders.append(OrderResult(
                    code, 0, 0.0, 0.0,
                    "%s %s" % (current_date, decision_time), False,
                    "no available holding slot after execution",
                ))
                continue
            try:
                bar = self.loader.get_minute_bar(code, current_date, decision_time)
            except (FileNotFoundError, KeyError):
                orders.append(OrderResult(
                    code, 0, 0.0, 0.0,
                    "%s %s" % (current_date, decision_time), False,
                    "missing execution bar at %s" % decision_time,
                ))
                continue
            price = float(bar[field])
            if not _bar_has_executable_trade(bar):
                orders.append(OrderResult(
                    code, 0, price, 0.0,
                    "%s %s" % (current_date, decision_time), False,
                    "no executable trade at %s" % decision_time,
                ))
                continue
            order = self.broker.order_target_value(
                code, target_value, price,
                "%s %s" % (current_date, decision_time),
            )
            if order.filled and reason:
                order.reason = reason
            orders.append(order)
        return orders

    def _close_marks(self, current_date):
        marks = {}
        for code in self.broker.positions:
            frame = self.loader.load_daily_frame(code, current_date)
            rows = frame[frame["date"].astype(str) == current_date]
            if rows.empty:
                raise KeyError("No daily close for %s %s" % (code, current_date))
            marks[code] = float(rows.iloc[0]["close"])
        return marks
```

Import `DayResult`, `LocalBroker`, `OrderResult`, `Position`, `_bar_has_executable_trade`, and `_planner_max_holdings` from the existing local backtester. For `14:45`, the fixed price field is `open`; for `09:35`, it remains `close`.

- [ ] **Step 5: Implement the shared-state planner**

Create a subclass with these concrete public methods and state:

```python
@dataclass
class DualTimepointOrderPlanner(LocalCrossSignalOrderPlanner):
    sold_today: set[str] = field(default_factory=set)
    failed_buy_codes: set[str] = field(default_factory=set)
    execution_date: str | None = None
    decision_time: str = "09:35"
    entry_score_snapshots: dict = field(default_factory=dict)
    exit_score_snapshots: dict = field(default_factory=dict)
    score_coverage: dict = field(default_factory=dict)

    def plan_orders_at(self, current_date, previous_date, broker, decision_time, current_prices=None):
        if self.execution_date != str(current_date):
            self.execution_date = str(current_date)
            self.sold_today.clear()
            self.failed_buy_codes.clear()
        self.decision_time = str(decision_time)[:5]
        self._current_date = str(current_date)
        proposed_orders = super().plan_orders(
            current_date, previous_date, broker, current_prices=current_prices
        )
        orders = [order for order in proposed_orders if not (
            float(order["target_value"]) > 0 and
            str(order["code"]).split(".")[0] in (self.sold_today | self.failed_buy_codes)
        )]
        for order in orders:
            code = str(order["code"]).split(".")[0]
            score = self.last_scores.get(code)
            if score is None:
                continue
            key = (str(current_date), self.decision_time, code)
            if order.get("reason") == "buy_signal":
                self.entry_score_snapshots[key] = dict(score)
            elif order.get("reason") in {"signal_sell", "atr_stop"}:
                self.exit_score_snapshots[key] = dict(score)
        return orders

    def _score_pool(self, current_date):
        scores = []
        for raw_code in self.etf_pool:
            code = str(raw_code).split(".")[0]
            score, reason = self.signal_adapter.score_at(
                code,
                current_date,
                self.decision_time,
                return_reason=True,
            )
            self.score_coverage[(str(current_date), self.decision_time, code)] = (
                "ok" if score is not None else str(reason or "unknown")
            )
            if score is None:
                continue
            item = dict(score)
            item["code"] = code
            if code in self.sold_today or code in self.failed_buy_codes:
                item["buy_allowed"] = False
            scores.append(item)
        return strategy.sort_candidates(scores)

    def _atr_stop_codes(self, broker, current_prices):
        stopped = super()._atr_stop_codes(broker, current_prices)
        return {
            code for code in stopped
            if str(self.buy_dates.get(code)) != str(self._current_date)
        }

    def on_orders_processed(self, current_date, decision_time, plans, results):
        super().on_orders_filled(current_date, results)
        plan_by_code = {str(item["code"]).split(".")[0]: item for item in plans}
        result_by_code = {str(item.code).split(".")[0]: item for item in results}
        for code, plan in plan_by_code.items():
            result = result_by_code.get(code)
            if float(plan["target_value"]) == 0.0 and result is not None and result.filled:
                self.sold_today.add(code)
            if float(plan["target_value"]) > 0.0 and (result is None or not result.filled):
                self.failed_buy_codes.add(code)
```

Do not use mutable global time state outside this candidate object. Ensure `on_orders_processed` reads the current batch's `last_scores`, so 14:45 buys freeze provisional T-day ATR.

- [ ] **Step 6: Run engine, planner, and baseline regression tests**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_local_order_planner.py
```

Expected: PASS, including exact morning-only parity.

- [ ] **Step 7: Commit the isolated dual-batch executor**

```powershell
git add -- cross_signal_strategy/local/local_backtester.py cross_signal_strategy/local/dual_timepoint_backtester.py cross_signal_strategy/local/dual_timepoint_order_planner.py tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_local_order_planner.py
git commit -m "feat(cross-signal): replay shared 09:35 and 14:45 decisions"
```

---

### Task 5: Apply the pre-registered quality and friction gates

**Files:**
- Create: `cross_signal_strategy/research/dual_timepoint_1445_candidate.py`
- Modify: `cross_signal_strategy/research/trade_quality_ledger.py`
- Modify: `tests/test_cross_signal_dual_timepoint_1445.py`
- Modify: `tests/test_cross_signal_trade_quality_ledger.py`

**Interfaces:**
- Consumes: baseline `DayResult` list, candidate `DayResult` list, score snapshots, `build_baseline_report`, `build_closed_trade_diagnostics`, and `build_trade_quality_ledger`.
- Produces: `DualTimepoint1445Report`, `DualTimepointGateDecision`, `build_dual_timepoint_1445_report`, and `run_training_dual_timepoint_1445_candidate`.

- [ ] **Step 1: Write failing exact-gate tests**

Define synthetic baseline/candidate performance fixtures and add:

```python
from dataclasses import replace


def _passing_gate_inputs():
    return DualTimepointGateInputs(
        total_return=0.96,
        baseline_total_return=1.20,
        max_drawdown=0.07,
        baseline_max_drawdown=0.07,
        profit_loss_ratio=3.1,
        win_rate=0.58,
        baseline_win_rate=0.56,
        annual_win_rates={2019: 0.60, 2020: 0.55, 2021: 0.50},
        baseline_annual_win_rates={2019: 0.59, 2020: 0.55, 2021: 0.52},
        round_trip_count=6,
        baseline_round_trip_count=9,
        round_trip_improved_codes=("AAA", "BBB"),
        max_loss_streak=3,
        baseline_max_loss_streak=3,
        buy_count=115,
        baseline_buy_count=100,
        sell_count=112,
        baseline_sell_count=100,
        annual_coverage={2019: 10, 2020: 10, 2021: 10},
        annual_missing={2019: 0, 2020: 1, 2021: 0},
        double_friction_return=0.90,
        baseline_double_friction_return=1.10,
        double_friction_drawdown=0.07,
        baseline_double_friction_drawdown=0.07,
    )


def test_1445_gate_requires_every_frozen_condition():
    passing = _passing_gate_inputs()
    assert evaluate_dual_timepoint_1445_gate(passing).passed is True

    failing_overrides = {
        "total_return": 0.95,
        "max_drawdown": 0.071,
        "profit_loss_ratio": 2.99,
        "win_rate": 0.56,
        "annual_win_rates": {2019: 0.58, 2020: 0.54, 2021: 0.51},
        "round_trip_count": 7,
        "round_trip_improved_codes": ("AAA",),
        "max_loss_streak": 4,
        "buy_count": 131,
        "sell_count": 131,
        "annual_coverage": {2019: 10, 2020: 0, 2021: 10},
        "double_friction_return": 0.87,
        "double_friction_drawdown": 0.071,
    }
    for field, value in failing_overrides.items():
        broken = replace(passing, **{field: value})
        assert evaluate_dual_timepoint_1445_gate(broken).passed is False, field
```

The exact interpretations are:

- total return candidate `>= 0.80 * baseline`;
- candidate max drawdown `<= baseline`;
- candidate profit/loss ratio `>= 3.0`;
- candidate overall closed-trade win rate `> baseline`;
- candidate annual win rate `>= baseline` in at least two of 2019/2020/2021, grouping by exit year;
- positive-to-negative round trip means `holding_mfe > 0` and realized return `< 0`;
- candidate round trips are at least three fewer than baseline and the reductions cover at least two ETF codes;
- candidate maximum consecutive losing closed trades `<= baseline`, preserving close-order sequence;
- candidate buy and sell counts are each `<= 1.30 * baseline`;
- each year has at least one valid 14:45 code-date observation and all missing counts are reported;
- under doubled commission/slippage, candidate return `>= 0.80 * doubled-friction baseline return` and candidate drawdown `<= doubled-friction baseline drawdown`.

- [ ] **Step 2: Run the gate test and verify the report module is missing**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py -k gate
```

Expected: FAIL with missing report/gate interfaces.

- [ ] **Step 3: Implement immutable report types and gate**

Create dataclasses containing every field above and implement:

```python
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DualTimepointGateInputs:
    total_return: float
    baseline_total_return: float
    max_drawdown: float
    baseline_max_drawdown: float
    profit_loss_ratio: float | None
    win_rate: float
    baseline_win_rate: float
    annual_win_rates: Mapping[int, float]
    baseline_annual_win_rates: Mapping[int, float]
    round_trip_count: int
    baseline_round_trip_count: int
    round_trip_improved_codes: Sequence[str]
    max_loss_streak: int
    baseline_max_loss_streak: int
    buy_count: int
    baseline_buy_count: int
    sell_count: int
    baseline_sell_count: int
    annual_coverage: Mapping[int, int]
    annual_missing: Mapping[int, int]
    double_friction_return: float
    baseline_double_friction_return: float
    double_friction_drawdown: float
    baseline_double_friction_drawdown: float


@dataclass(frozen=True)
class DualTimepointGateDecision:
    passed: bool
    reasons: Sequence[str]


def evaluate_dual_timepoint_1445_gate(item: DualTimepointGateInputs) -> DualTimepointGateDecision:
    reasons = []
    if item.total_return + 1e-12 < 0.80 * item.baseline_total_return:
        reasons.append("total return retains less than 80% of baseline")
    if item.max_drawdown > item.baseline_max_drawdown + 1e-12:
        reasons.append("maximum drawdown worsens")
    if item.profit_loss_ratio is None or item.profit_loss_ratio < 3.0:
        reasons.append("profit/loss ratio is below 3.0")
    if item.win_rate <= item.baseline_win_rate:
        reasons.append("closed-trade win rate does not improve")
    annual_non_worse = sum(
        item.annual_win_rates.get(year, -1.0) >= item.baseline_annual_win_rates.get(year, 2.0)
        for year in (2019, 2020, 2021)
    )
    if annual_non_worse < 2:
        reasons.append("fewer than two annual win rates are non-worse")
    if item.round_trip_count > item.baseline_round_trip_count - 3:
        reasons.append("positive-to-negative round trips fall by fewer than three")
    if len(set(item.round_trip_improved_codes)) < 2:
        reasons.append("round-trip improvement is concentrated in fewer than two ETFs")
    if item.max_loss_streak > item.baseline_max_loss_streak:
        reasons.append("maximum losing streak worsens")
    if item.buy_count > 1.30 * item.baseline_buy_count:
        reasons.append("buy count rises by more than 30%")
    if item.sell_count > 1.30 * item.baseline_sell_count:
        reasons.append("sell count rises by more than 30%")
    if any(item.annual_coverage.get(year, 0) <= 0 for year in (2019, 2020, 2021)):
        reasons.append("one or more years have no usable 14:45 coverage")
    if item.double_friction_return + 1e-12 < 0.80 * item.baseline_double_friction_return:
        reasons.append("doubled-friction return retains less than 80% of baseline")
    if item.double_friction_drawdown > item.baseline_double_friction_drawdown + 1e-12:
        reasons.append("doubled-friction drawdown worsens")
    return DualTimepointGateDecision(not reasons, tuple(reasons))
```

- [ ] **Step 4: Build reports from actual replay objects**

Implement `build_dual_timepoint_1445_report` to:

1. Assert identical 2019-2021 trading dates and reject any outside date.
2. Call `build_baseline_report` for both runs.
3. Call `build_closed_trade_diagnostics` with planner snapshots normalized from `(date,time,code)` to the filled order's date/code.
4. Call `build_trade_quality_ledger` only after all orders exist; its forward paths remain evaluation-only.
5. Compute annual win rates by exit year, maximum losing streak, round-trip counts and per-code reductions.
6. Report usable/missing 14:45 score coverage by year.
7. Compare nominal and doubled-friction runs using the exact gate above.
8. Include filled-order signatures with `side_time`, so 09:35 and 14:45 changes remain attributable.

Do not feed trade-quality rows, MFE, future closes, or gate results back into either planner.

- [ ] **Step 5: Run gate/report tests**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_baseline_report.py tests/test_cross_signal_trade_quality_ledger.py
```

Expected: PASS.

- [ ] **Step 6: Commit the fixed gate**

```powershell
git add -- cross_signal_strategy/research/dual_timepoint_1445_candidate.py tests/test_cross_signal_dual_timepoint_1445.py
git commit -m "feat(cross-signal): gate fixed 14:45 training candidate"
```

---

### Task 6: Run once, record the outcome, and close the family

**Files:**
- Modify: `cross_signal_strategy/research/dual_timepoint_1445_candidate.py`
- Modify: `tests/test_cross_signal_dual_timepoint_1445.py`
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/backtest_notes.md`
- Modify if failed: `cross_signal_strategy/docs/failed_experiments.md`
- Modify if passed: `cross_signal_strategy/docs/decisions.md`
- Create: `cross_signal_strategy/reports/dual_timepoint_1445_2019_2021.md`

**Interfaces:**
- Consumes: the fixed runner and gate from Task 5.
- Produces: one immutable report, one exhausted research-family record, and an explicit stop/continue decision.

- [ ] **Step 1: Add the end-to-end runner test before running real data**

```python
def test_training_runner_configuration_is_frozen():
    config = dual_timepoint_1445_training_config()

    assert config.candidate_name == "cross-v0.3.3-dual-timepoint-1445-candidate"
    assert config.decision_times == ("09:35", "14:45")
    assert config.signal_cutoff == "14:44"
    assert config.training_start == "2019-01-01"
    assert config.training_end == "2021-12-31"
    assert config.training_root == pathlib.Path(
        r"G:\financial\history_data\cross_signal_train_2019_2021"
    )
    assert config.warmup_root == pathlib.Path(
        r"G:\financial\history_data\cross_signal_warmup_2018"
    )
    assert config.initial_cash == pytest.approx(20000.0)
    assert config.candidate_variants == 1
```

Implement `DualTimepoint1445TrainingConfig` as a frozen dataclass and require
`run_training_dual_timepoint_1445_candidate` to accept exactly that object. The
runner must assert the two roots and both date bounds again before loading data.

Add a formatter test using `_passing_gate_inputs()` and a
`DualTimepoint1445Report` whose `gate` is constructed with
`DualTimepointGateDecision(False, ("maximum drawdown worsens",))`. Assert the
rendered text contains `nominal`, `2019`, `2020`, `2021`, `round trip`,
`maximum loss streak`, `coverage`, `missing`, `double friction`, the exact gate
reason, and `STOP`. Repeat with an empty-reason passing decision and assert
`ELIGIBLE_FOR_JOINQUANT_PLAN`. Define the report dataclass with these concrete
fields: `config`, `gate_inputs`, `gate`, `baseline_order_signature`,
`candidate_order_signature`, and `rendered_sections`; do not let the formatter
query market data or recompute any metric.

- [ ] **Step 2: Run all focused tests before the empirical run**

```powershell
pytest -q tests/test_cross_signal_dual_timepoint_1445.py tests/test_cross_signal_research_budget.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_baseline_report.py tests/test_cross_signal_trade_quality_ledger.py
```

Expected: PASS.

- [ ] **Step 3: Implement CLI output with deterministic exit status**

`python -m cross_signal_strategy.research.dual_timepoint_1445_candidate` must:

- print the complete report;
- write the same text to `cross_signal_strategy/reports/dual_timepoint_1445_2019_2021.md`;
- return exit code `0` only when every gate passes;
- return exit code `1` when any gate fails;
- never write under a data root;
- never run a second variant after failure.

- [ ] **Step 4: Run the candidate exactly once and preserve stdout**

Run from repository root:

```powershell
python -m cross_signal_strategy.research.dual_timepoint_1445_candidate
```

Interpret exit code `0` as `ELIGIBLE_FOR_JOINQUANT_PLAN` and exit code `1` as `STOP`. A gate failure is a valid research result, not a code failure. Do not change any rule after seeing output.

- [ ] **Step 5: Write a failing closure test from the consumed result**

Add a repository-budget test that requires:

```python
family = families["intraday_signal_clock_1445_user_authorized"]
assert budget.max_total_open_experiments == 0
assert family.status == "exhausted"
assert family.max_new_experiments == 0
assert raw_family["candidate_gate_passed"] is report_gate_passed
assert raw_family["candidate_created"] is False
assert raw_family["validation_influence"] == "none"
assert raw_family["prohibit_alternatives"] is True
```

Run that new test and verify it fails because the family is still open.

- [ ] **Step 6: Close governance with the exact emitted evidence**

Set `max_total_open_experiments` back to `0`; set the family to `status: exhausted`, `max_new_experiments: 0`; remove `planned_experiment`; and copy the report's exact deterministic fields into the family record:

- `candidate_gate_passed`, always a boolean;
- `candidate_created: false` because this phase never creates platform code;
- baseline/candidate total return, max drawdown, win rate, profit/loss ratio;
- annual win rates;
- baseline/candidate round-trip counts and max losing streak;
- baseline/candidate buy/sell counts;
- doubled-friction return/drawdown;
- coverage and missing counts by year;
- `validation_influence: none`, the frozen data scope, and `prohibit_alternatives: true`.

If the gate failed, append one complete entry to `failed_experiments.md` and increment `expected_failed_experiment_count` by exactly one. If it passed, add the frozen training result to `decisions.md`; do not increment the failed count. In both branches, append the full fixed hypothesis, metrics, gate reasons, and next allowed action to `backtest_notes.md` and update the readable budget map.

- [ ] **Step 7: Run governance, focused, and full regression suites**

```powershell
pytest -q tests/test_cross_signal_research_budget.py tests/test_cross_signal_dual_timepoint_1445.py
pytest -q
```

Expected: all assertions PASS. Permission warnings about unwritable `.pytest_cache` are environment warnings, not assertion failures; report them separately.

- [ ] **Step 8: Verify scope and immutable-data safety**

```powershell
git status --short
git diff --check
git diff --name-only HEAD
```

Expected changed paths are limited to the files named in this plan. No formal JoinQuant/PTrade, multi-factor, validation data, or source-data path may appear. The unrelated handoff document remains untracked and untouched.

- [ ] **Step 9: Commit the consumed outcome**

If the gate failed:

```powershell
git add -- cross_signal_strategy/reports/dual_timepoint_1445_2019_2021.md cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md cross_signal_strategy/docs/backtest_notes.md cross_signal_strategy/docs/failed_experiments.md tests/test_cross_signal_research_budget.py tests/test_cross_signal_dual_timepoint_1445.py cross_signal_strategy/research/dual_timepoint_1445_candidate.py
git commit -m "research(cross-signal): reject fixed 14:45 candidate"
```

If the gate passed:

```powershell
git add -- cross_signal_strategy/reports/dual_timepoint_1445_2019_2021.md cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md cross_signal_strategy/docs/backtest_notes.md cross_signal_strategy/docs/decisions.md tests/test_cross_signal_research_budget.py tests/test_cross_signal_dual_timepoint_1445.py cross_signal_strategy/research/dual_timepoint_1445_candidate.py
git commit -m "research(cross-signal): qualify fixed 14:45 candidate"
```

- [ ] **Step 10: Stop at the research gate**

If failed, report the failed gates and do not write another plan unless the user opens a genuinely new independent mechanism. If passed, report that this authorizes only a separate JoinQuant training-candidate implementation plan; do not create or modify JoinQuant/PTrade code in this phase.

## Deferred Spec Coverage

The following approved requirements are intentionally deferred behind the local gate and are not omissions:

- JoinQuant daily scheduling at `14:45`, minute cutoff proof, and authoritative training performance: separate plan only if this local gate passes.
- PTrade fifth scheduled task, one-minute `get_history` with `include=False`, live order-state recovery, pending-order callbacks, restart persistence, and shadow logging: separate plan only after the JoinQuant gate passes.
- Formal strategy promotion, reserved validation, and live deployment: separate user decisions after their preceding gates.
