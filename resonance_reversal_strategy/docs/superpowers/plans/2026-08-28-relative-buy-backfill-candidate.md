# Relative Buy Backfill Candidate Implementation Plan

> **执行要求：** 当前会话使用 `superpowers:executing-plans` 逐项实施。只有用户另外明确要求委派时，才可改用 `superpowers:subagent-driven-development`。步骤使用 `- [ ]` 跟踪。

**Goal:** 只把 `HARD_BOLL_SOFT_OSC + BUY_TURN` 相对观察作为正式买入之后的空槽补位候选，提高资金利用率而不引入相对卖出。

**Architecture:** 保持现有 pending 重试和 ATR 风险退出先运行；随后从 T-1 快照冻结相对买入决策，并保证它在任何正式信号卖单或买单前完成。现有正式买入队列仍先执行，并把同一个 `remaining_slots` 交给相对后备队列。相对候选复用现有下单、ATR、持仓和反重复保护，但使用独立日志来源标识。

**Tech Stack:** JoinQuant Python 3、pytest、现有正式/相对事件构造和候选绩效评估器。

**Spec:** `resonance_reversal_strategy/docs/superpowers/specs/2026-08-28-resonance-quality-candidate-program-design.md`

## Global Constraints

- 从 commit `020bc36` 的 `20260827.4` 基线独立建立；候选 build 固定为 `20260828.2`，不得包含 ATR 影子或 BOLL 失效退出代码。
- 只有 `BUY_TURN + HARD_BOLL_SOFT_OSC` 可下单；`SELL_TURN` 和 `SOFT_ALL_THREE` 永远保持观察态。
- 正式候选永远优先；相对候选只能使用正式队列留下的同一槽位计数。
- 正式 `UNKNOWN`/`NOT_FILLED` 的意向槽位消费、暂停补位、同日卖出禁止回购、已持有不加仓和处理 ID 语义必须保持。
- 相对 H1/H3/H5 结果和命中率不得进入运行时。
- 不改仓位公式、ATR、正式卖出、ETF 池、参数、09:35/T-1 边界。

## 策略测试辅助契约

新增测试继续复用 `tests/test_resonance_reversal_strategy.py` 中已经存在的 `strategy`、`fake_context`、`current_record`、`resonance_snapshot` 和 `runtime_state`。在新测试前集中定义：

```python
class FutureDataError(RuntimeError):
    pass


def context():
    return fake_context(current_date="2021-01-06")


def valid_snapshot(code="510300.XSHG", entry_atr=0.1):
    snapshot = resonance_snapshot(code)
    snapshot["entry_atr"] = entry_atr
    return snapshot


def relative_observation(
        code="510300.XSHG",
        direction=strategy.TurnDirection.BUY_TURN,
        branch="HARD_BOLL_SOFT_OSC",
        supporters=("BOLL", "RSI")):
    return {
        "relative_observation_id": "RELATIVE:" + code,
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": branch,
        "code": code,
        "direction": direction,
        "signal_date": date(2021, 1, 5),
        "supporters": tuple(supporters),
        "supporter_event_dates": {
            supporter: date(2021, 1, 5) for supporter in supporters
        },
        "hard_or_relative_source_by_indicator": {
            supporter: (
                "HARD" if supporter == "BOLL" else "RELATIVE"
            )
            for supporter in supporters
        },
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }


def relative_buy_decision(code, supporters=("BOLL", "RSI")):
    observation = relative_observation(
        code=code, supporters=supporters,
    )
    return strategy.build_relative_buy_backfill_decision(
        observation, valid_snapshot(code),
    )


def formal_buy_decision(code):
    snapshot = resonance_snapshot(code)
    return strategy.build_resonance_decision(
        code,
        strategy.TurnDirection.BUY_TURN,
        snapshot["event_book"],
        snapshot["signal_date"],
    )
```

`install_formal_stage_spies(monkeypatch)` 必须用 `runtime_state()` 安装 `g`，把 `get_current_data` 和 `build_signal_snapshots` 固定为空映射，并分别记录 `run_signal_exits`、`run_signal_buys` 的调用次数及传给后者的相对 tuple。`install_risk_and_signal_stage_spies(monkeypatch, calls)` 必须按调用顺序记录 `retry_pending_exits`、`run_atr_exits`、`run_signal_exits`、`run_signal_buys`；相对准备抛出未来数据错误后，后两项不得出现。两个辅助均不得调用订单 API。

---

### Task 1: Convert the approved relative observation branch into frozen buy decisions

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Produces: `build_relative_buy_backfill_decision(observation, snapshot) -> dict | None`。
- Produces: `collect_relative_buy_backfill_decisions(snapshots) -> tuple`，元素均为冻结决策 dict。
- Decision keys: `resonance_id`, `trade_source`, `branch`, `code`, `direction`, `signal_date`, `supporters`, `support_count`, `expires_date`。

- [ ] **Step 1: Write failing eligibility and sorting tests**

```python
def test_relative_trade_candidate_accepts_only_hard_boll_buy():
    approved = relative_observation(
        direction=strategy.TurnDirection.BUY_TURN,
        branch="HARD_BOLL_SOFT_OSC",
        supporters=("BOLL", "KDJ"),
    )
    soft_all = relative_observation(
        direction=strategy.TurnDirection.BUY_TURN,
        branch="SOFT_ALL_THREE",
    )
    sell = relative_observation(
        direction=strategy.TurnDirection.SELL_TURN,
        branch="HARD_BOLL_SOFT_OSC",
    )

    assert strategy.build_relative_buy_backfill_decision(
        approved, valid_snapshot(entry_atr=0.1),
    )["trade_source"] == "RELATIVE_BUY_BACKFILL"
    assert strategy.build_relative_buy_backfill_decision(
        soft_all, valid_snapshot(entry_atr=0.1),
    ) is None
    assert strategy.build_relative_buy_backfill_decision(
        sell, valid_snapshot(entry_atr=0.1),
    ) is None


def test_relative_buy_sort_is_support_count_then_code():
    decisions = [
        relative_buy_decision("B.XSHG", supporters=("BOLL", "KDJ")),
        relative_buy_decision("C.XSHG", supporters=("BOLL", "KDJ", "RSI")),
        relative_buy_decision("A.XSHG", supporters=("BOLL", "KDJ", "RSI")),
    ]
    assert [item["code"] for item in
            strategy.sort_relative_buy_backfill_decisions(decisions)] == [
        "A.XSHG", "C.XSHG", "B.XSHG",
    ]
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_trade_candidate or relative_buy_sort" -q
```

Expected: FAIL because trading conversion helpers do not exist.

- [ ] **Step 3: Implement strict projection without outcome fields**

```python
def build_relative_buy_backfill_decision(observation, snapshot):
    if observation.get("direction") is not TurnDirection.BUY_TURN:
        return None
    if observation.get("branch") != "HARD_BOLL_SOFT_OSC":
        return None
    if not is_finite_positive(snapshot.get("entry_atr")):
        return None
    supporters = tuple(observation["supporters"])
    return {
        "resonance_id": observation["relative_observation_id"],
        "trade_source": "RELATIVE_BUY_BACKFILL",
        "branch": observation["branch"],
        "code": observation["code"],
        "direction": observation["direction"],
        "signal_date": observation["signal_date"],
        "supporters": supporters,
        "support_count": len(supporters),
        "expires_date": observation["expires_date"],
    }


def sort_relative_buy_backfill_decisions(decisions):
    return sorted(decisions, key=lambda item: (
        -item["support_count"], item["code"],
    ))
```

The decision must not contain `outcomes`, H1/H3/H5 return, hit rate, mean, median or Q1.

- [ ] **Step 4: Run existing relative isolation and new selection tests**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative or candidate_sort" -q
```

Expected: PASS; formal fingerprints remain frozen.

- [ ] **Step 5: Commit pure selection**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "test(resonance): define relative buy backfill eligibility"
```

### Task 2: Prepare the relative queue before formal signal orders and isolate failures

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Produces: `prepare_relative_buy_backfill_decisions(snapshots) -> tuple`。
- Ordinary malformed relative data returns an empty tuple and logs `RELATIVE_BUY_PREPARATION_FAILED`。
- Future-data errors propagate before `run_signal_exits` or `run_signal_buys`；已经排在快照之前的 `retry_pending_exits` 与 `run_atr_exits` 保持原顺序和行为。

- [ ] **Step 1: Write failing control-flow tests**

```python
def test_relative_buy_preparation_error_keeps_formal_orders(monkeypatch):
    monkeypatch.setattr(
        strategy, "collect_relative_buy_backfill_decisions",
        lambda snapshots: (_ for _ in ()).throw(RuntimeError("relative")),
    )
    calls = install_formal_stage_spies(monkeypatch)
    strategy.do_trading(context())
    assert calls["signal_exits"] == 1
    assert calls["signal_buys"] == 1
    assert calls["relative_queue"] == ()


def test_relative_buy_future_error_occurs_after_risk_exits_but_before_signal_orders(monkeypatch):
    calls = []
    install_risk_and_signal_stage_spies(monkeypatch, calls)
    monkeypatch.setattr(
        strategy, "collect_relative_buy_backfill_decisions",
        lambda snapshots: (_ for _ in ()).throw(FutureDataError()),
    )
    with pytest.raises(FutureDataError):
        strategy.do_trading(context())
    assert calls == ["retry_pending_exits", "run_atr_exits"]
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_buy_preparation or relative_buy_future" -q
```

- [ ] **Step 3: Implement a responsibility-specific safe preparation function**

```python
def prepare_relative_buy_backfill_decisions(snapshots):
    try:
        return collect_relative_buy_backfill_decisions(snapshots)
    except Exception as error:
        if _is_future_data_error(error):
            raise
        _safe_relative_observation_diagnostic(
            "relative_buy_preparation", {
                "reason": "RELATIVE_BUY_PREPARATION_FAILED",
                "error_type": type(error).__name__,
            },
        )
        return ()
```

Call it after `run_relative_observation_stage(snapshots)` and before `run_signal_exits`. Do not move `retry_pending_exits` or `run_atr_exits` from their existing pre-snapshot positions. Pass the frozen tuple to `run_signal_buys`; no signal-buy or signal-sell function may recalculate relative events.

- [ ] **Step 4: Run stage-order and future-boundary tests**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "do_trading or future or relative_buy" -q
```

Expected: PASS.

- [ ] **Step 5: Commit safe preparation**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): freeze relative buys before orders"
```

### Task 3: Append relative candidates to the same formal slot budget

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Changes signature: `run_signal_buys(context, current_data, snapshots, relative_backfill_decisions=())`。
- Produces: `_run_relative_buy_backfill(context, current_data, snapshots, decisions, actual_positions, remaining_slots) -> (results, remaining_slots)`。
- Empty tuple must preserve the old return list and every old branch.

- [ ] **Step 1: Write the formal-priority and slot-consumption tests**

```python
def install_buy_harness(
        monkeypatch, formal_decisions, relative_decisions,
        max_holdings=1, tradability_by_code=None, outcome_by_code=None):
    all_decisions = tuple(formal_decisions) + tuple(relative_decisions)
    codes = tuple(decision["code"] for decision in all_decisions)
    snapshots = {code: valid_snapshot(code) for code in codes}
    current = {code: current_record(10.0, paused=False) for code in codes}
    runtime = runtime_state(max_holdings=max_holdings)
    submitted = []
    tradability_by_code = tradability_by_code or {}
    outcome_by_code = outcome_by_code or {}

    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "collect_buy_decisions",
        lambda snapshots, actual_positions: list(formal_decisions),
    )
    monkeypatch.setattr(
        strategy, "get_tradability",
        lambda current_data, code: tradability_by_code.get(
            code, strategy.Tradability.TRADEABLE,
        ),
    )

    def fake_submit_buy(context, code, snapshot, decision):
        submitted.append(code)
        return outcome_by_code.get(code, strategy.OrderOutcome.FILLED)

    monkeypatch.setattr(strategy, "submit_buy", fake_submit_buy)
    return types.SimpleNamespace(
        context=fake_context(),
        current_data=current,
        snapshots=snapshots,
        submitted=submitted,
        runtime=runtime,
    )


def test_formal_queue_stays_ahead_of_relative_queue(monkeypatch):
    formal = formal_buy_decision("FORMAL.XSHG")
    relative = relative_buy_decision("RELATIVE.XSHG")
    harness = install_buy_harness(
        monkeypatch, [formal], [relative], max_holdings=1,
    )

    results = strategy.run_signal_buys(
        harness.context, harness.current_data, harness.snapshots,
        (relative,),
    )

    assert harness.submitted == ["FORMAL.XSHG"]
    assert all(code != "RELATIVE.XSHG" for code, _ in results)


@pytest.mark.parametrize("tradability, formal_outcome", [
    (strategy.Tradability.UNKNOWN, strategy.OrderOutcome.UNKNOWN),
    (strategy.Tradability.TRADEABLE, strategy.OrderOutcome.NOT_FILLED),
])
def test_formal_consumed_intent_slot_cannot_be_reopened_for_relative(
        monkeypatch, tradability, formal_outcome):
    formal = formal_buy_decision("FORMAL.XSHG")
    relative = relative_buy_decision("RELATIVE.XSHG")
    harness = install_buy_harness(
        monkeypatch, [formal], [relative], max_holdings=1,
        tradability_by_code={"FORMAL.XSHG": tradability},
        outcome_by_code={"FORMAL.XSHG": formal_outcome},
    )

    strategy.run_signal_buys(
        harness.context, harness.current_data, harness.snapshots,
        (relative,),
    )
    assert "RELATIVE.XSHG" not in harness.submitted


def test_paused_relative_candidate_backfills_next_relative(monkeypatch):
    first = relative_buy_decision("A.XSHG")
    second = relative_buy_decision("B.XSHG")
    harness = install_buy_harness(
        monkeypatch, [], [first, second], max_holdings=1,
        tradability_by_code={
            "A.XSHG": strategy.Tradability.PAUSED,
            "B.XSHG": strategy.Tradability.TRADEABLE,
        },
    )

    strategy.run_signal_buys(
        harness.context, harness.current_data, harness.snapshots,
        (first, second),
    )
    assert harness.submitted == ["B.XSHG"]
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "formal_queue_stays or intent_slot or paused_relative" -q
```

- [ ] **Step 3: Implement the backup loop inside the existing budget**

Keep the current formal loop unchanged through its `remaining_slots -= 1`. After it finishes:

```python
relative_results, remaining_slots = _run_relative_buy_backfill(
    context=context,
    current_data=current_data,
    snapshots=snapshots,
    decisions=relative_backfill_decisions,
    actual_positions=get_actual_positions(context),
    remaining_slots=remaining_slots,
)
results.extend(relative_results)
return results
```

The helper must duplicate the current explicit guards in this order:

```text
remaining_slots → held → sold_today → daily_attempted_buys → processed ID
→ PAUSED (no slot consumed) → mark processed/attempted
→ UNKNOWN (slot consumed) → submit_buy (slot consumed unless refreshed PAUSED)
```

Do not recompute `remaining_slots` from the post-order portfolio.

- [ ] **Step 4: Run all buy-path tests**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "buy or slot or backfill or paused or processed" -q
```

Expected: PASS, including every pre-existing formal test.

- [ ] **Step 5: Commit queue integration**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): backfill empty slots with relative buys"
```

### Task 4: Add auditable logging and explicit non-target regressions

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Produces: `log_relative_buy_decision(decision, rank, accepted, reason)`。
- Emits: `relative_buy_decision` with source, branch, code, signal date, supporters, rank, accepted and reason。
- Initialization emits a separate `relative_buy_candidate_fingerprint`。

- [ ] **Step 1: Write failing audit tests**

```python
def capture_json_logs(monkeypatch):
    messages = []
    monkeypatch.setattr(
        strategy,
        "log",
        types.SimpleNamespace(info=lambda message: messages.append(message)),
        raising=False,
    )
    return messages


def test_relative_buy_log_never_claims_sell_or_soft_all_three(monkeypatch):
    messages = capture_json_logs(monkeypatch)
    decision = relative_buy_decision("510300.XSHG")

    strategy.log_relative_buy_decision(
        decision, rank=1, accepted=True, reason="BUY_ATTEMPT",
    )

    payloads = [json.loads(message) for message in messages]
    assert [item["event"] for item in payloads] == [
        "relative_buy_decision"
    ]
    assert {item["direction"] for item in payloads} == {"BUY_TURN"}
    assert {item["branch"] for item in payloads} == {
        "HARD_BOLL_SOFT_OSC"
    }


def run_formal_day(monkeypatch, explicit_empty):
    formal = formal_buy_decision("510300.XSHG")
    harness = install_buy_harness(
        monkeypatch, [formal], [], max_holdings=1,
    )
    args = (
        harness.context, harness.current_data, harness.snapshots,
    )
    results = (
        strategy.run_signal_buys(*args, ())
        if explicit_empty
        else strategy.run_signal_buys(*args)
    )
    return types.SimpleNamespace(
        orders=tuple(harness.submitted),
        results=tuple(results),
        position_states=copy.deepcopy(harness.runtime.position_states),
        processed_ids=copy.deepcopy(
            harness.runtime.processed_resonance_ids
        ),
    )


def test_empty_relative_queue_is_formally_execution_equivalent(monkeypatch):
    baseline = run_formal_day(monkeypatch, explicit_empty=False)
    candidate = run_formal_day(monkeypatch, explicit_empty=True)
    assert candidate.orders == baseline.orders
    assert candidate.results == baseline.results
    assert candidate.position_states == baseline.position_states
    assert candidate.processed_ids == baseline.processed_ids
```

- [ ] **Step 2: Run audit tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_buy_log or formally_execution_equivalent" -q
```

- [ ] **Step 3: Implement logs that cannot affect eligibility**

实现 `log_relative_buy_decision(decision, rank, accepted, reason)` 时只复制已经冻结的决策字段。Logger exceptions must be isolated after eligibility is decided and cannot add/remove/reorder candidates. Keep `relative_observation_fingerprint` unchanged and define `relative_buy_candidate_fingerprint` only from:

```python
{
    "direction": "BUY_TURN",
    "branch": "HARD_BOLL_SOFT_OSC",
    "priority": "AFTER_FORMAL",
    "sort": ("SUPPORT_COUNT_DESC", "CODE_ASC"),
}
```

- [ ] **Step 4: Run full strategy tests**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit audit contract**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "test(resonance): audit relative buy candidate source"
```

### Task 5: Freeze build, documentation, and one-shot platform evaluation

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`
- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`

**Interfaces:**
- Produces build `20260828.2` and a candidate log consumable by `analyze_candidate_performance.py`。

- [ ] **Step 1: Write the build and protected-boundary test**

```python
def test_relative_buy_candidate_build_preserves_protected_rules():
    assert strategy.DEPLOYMENT_BUILD_ID == "20260828.2"
    params = strategy.get_default_params()
    assert params["atr_multiplier"] == 2.5
    assert params["stop_floor"] == 0.05
    assert params["stop_cap"] == 0.15
    assert params["max_holdings"] == 3
    assert params["target_exposure"] == 0.95
```

- [ ] **Step 2: Update docs with the exact candidate and rejection rules**

Document formal-first ordering, approved branch/direction, slot consumption, no relative sell, no position discount and no runtime outcome metrics.

- [ ] **Step 3: Run complete local verification**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py tests/test_resonance_relative_turn_analysis.py tests/test_resonance_candidate_performance.py -q
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
git diff --check
```

- [ ] **Step 4: Commit the code-complete candidate before platform replay**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py resonance_reversal_strategy/README.md resonance_reversal_strategy/docs/strategy_spec.md
git commit -m "feat(resonance): freeze relative buy backfill candidate"
```

- [ ] **Step 5: Run exactly one ordinary-friction and one double-friction JoinQuant replay**

Both runs use 2019-01-01 through 2021-12-31, 20,000 yuan and the same ETF pool. Save both logs and evaluate them against `.4` with the candidate evaluator.

- [ ] **Step 6: Apply the independent advancement gates**

Advance only if every spec section 9.1 gate passes. Record a failed candidate with its commit, metrics and interpretation; do not change branch, ranking, thresholds or sizing in response to the same training result.
