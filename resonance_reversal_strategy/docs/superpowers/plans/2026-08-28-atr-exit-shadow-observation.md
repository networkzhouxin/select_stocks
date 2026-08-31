# ATR Exit Shadow Observation Implementation Plan

> **执行要求：** 当前会话使用 `superpowers:executing-plans` 逐项实施。只有用户另外明确要求委派时，才可改用 `superpowers:subagent-driven-development`。步骤使用 `- [ ]` 跟踪。

**Goal:** 在完全保留 ATR 卖单路径的同时，记录每次完全成交 ATR 退出后的 H1/H3/H5 价格表现。

**Architecture:** 使用独立 `ATR_SHADOW:` namespace 和 `g.atr_shadow_events` 队列。注册发生在 ATR 卖单确认清仓之后，结果只在收盘后写日志；任何影子普通异常都不得改变订单、挂起退出或持仓清理。

**Tech Stack:** JoinQuant Python 3、Python 标准库、pytest、现有结构化日志和 schema V2 manifest。

**Spec:** `resonance_reversal_strategy/docs/superpowers/specs/2026-08-28-resonance-quality-candidate-program-design.md`

## Global Constraints

- 从 commit `020bc36` 的 `20260827.4` 基线独立建立候选；候选 build 固定为 `20260828.1`。
- ATR(14,2.5)、5%--15% 边界、09:35 卖出、挂起重试、正式买卖和组合净值必须完全不变。
- 影子观察只使用 ATR 完全成交时已知的日期、状态和 09:35 执行报价；后续结果只在收盘记录。
- 不写入 `g.observation_events`，不修改相对观察指纹，不读取 2022+。
- 影子结果不能自动删除或修改 ATR。

## 策略测试辅助契约

后续策略测试继续使用现有测试模块的 `strategy`、`fake_context`、`current_record`、`runtime_state` 和 `make_position_state`。在新增测试前集中定义以下辅助，文中同名调用均指这些实现：

```python
class FutureDataError(RuntimeError):
    pass


def context():
    return fake_context(current_date="2021-01-12")


def context_at(timestamp):
    result = fake_context(current_date=str(pd.Timestamp(timestamp).date()))
    result.current_dt = pd.Timestamp(timestamp)
    return result


def current_data(price=10.0):
    return {
        "510300.XSHG": current_record(price=price, paused=False),
    }


def forbid_all_order_functions(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("ATR shadow outcome recording must not place orders")

    monkeypatch.setattr(strategy, "order_target", forbidden, raising=False)
    monkeypatch.setattr(
        strategy, "order_target_value", forbidden, raising=False,
    )
    monkeypatch.setattr(strategy, "submit_buy", forbidden)
    monkeypatch.setattr(strategy, "submit_sell", forbidden)
    return calls


def arrange_submit_sell(monkeypatch, outcome, after_amount, calls=None):
    code = "510300.XSHG"
    calls = [] if calls is None else calls
    state = strategy.make_position_state(
        pd.Timestamp("2021-01-04").date(), 0.2, 4.0,
    )
    state["entry_price"] = 4.0
    runtime = runtime_state(position_states={code: state})
    amounts = iter((1000, after_amount))
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_current_data", lambda: current_data(price=3.6),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "get_actual_amount",
        lambda context, requested_code: next(amounts),
    )
    monkeypatch.setattr(
        strategy, "classify_order_outcome",
        lambda *args, **kwargs: outcome,
    )
    monkeypatch.setattr(
        strategy, "order_target",
        lambda order_code, target: calls.append("order_target")
        or types.SimpleNamespace(amount=-1000, filled=-1000),
        raising=False,
    )
    original_sync = strategy.sync_sell_state_after_order

    def tracked_sync(*args, **kwargs):
        calls.append("state_sync")
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(strategy, "sync_sell_state_after_order", tracked_sync)
    return calls
```

`install_shadow_event` 必须用 `make_atr_shadow_event` 写入全新的 `runtime_state().atr_shadow_events`；`shadow_outcome(horizon)` 只读取该记录的 `outcomes[horizon]`，不得访问正式 `observation_events`。分析器测试中的 `analyze_fixture` 则沿用现有相对分析器测试的内存 records 模式，显式生成 138 条订单、730 条 portfolio summary、一个注册及其结果，不读真实日志。

---

### Task 1: Add isolated ATR shadow state and immutable identities

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Produces: `build_atr_shadow_id(code, buy_date, exit_date) -> str`。
- Produces: `make_atr_shadow_event(code, state, exit_date, reference_price, stop_price) -> dict`。
- Produces runtime state: `g.atr_shadow_events: dict[str, dict]`。

- [ ] **Step 1: Write failing state and identity tests**

```python
def test_runtime_state_adds_isolated_atr_shadow_queue(monkeypatch):
    strategy.g = SimpleNamespace()
    strategy.ensure_runtime_state()
    assert strategy.g.atr_shadow_events == {}
    assert strategy.g.atr_shadow_events is not strategy.g.observation_events


def test_atr_shadow_id_is_deterministic_and_namespaced():
    first = strategy.build_atr_shadow_id(
        "510300.XSHG", date(2021, 1, 4), date(2021, 1, 12),
    )
    second = strategy.build_atr_shadow_id(
        "510300.XSHG", date(2021, 1, 4), date(2021, 1, 12),
    )
    assert first == second
    assert first.startswith("ATR_SHADOW:")
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_shadow_id or isolated_atr_shadow_queue" -q
```

Expected: FAIL because the state and helpers do not exist.

- [ ] **Step 3: Implement isolated state and exact record shape**

```python
def build_atr_shadow_id(code, buy_date, exit_date):
    raw = "%s|%s|%s" % (
        code, _calendar_date(buy_date), _calendar_date(exit_date),
    )
    return "ATR_SHADOW:" + hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]


def make_atr_shadow_event(code, state, exit_date, reference_price,
                          stop_price):
    return {
        "atr_shadow_id": build_atr_shadow_id(
            code, state["buy_date"], exit_date,
        ),
        "observation_kind": "ATR_EXIT_SHADOW",
        "code": code,
        "event_date": _calendar_date(exit_date),
        "reference_price": float(reference_price),
        "entry_price": state.get("entry_price"),
        "entry_atr": state["entry_atr"],
        "highest_close_anchor": state["highest_close_anchor"],
        "stop_price": float(stop_price),
        "horizons": (1, 3, 5),
        "outcomes": {},
        "build": DEPLOYMENT_BUILD_ID,
    }
```

Extend `make_position_state` with an immutable `entry_price` field equal to the existing buy execution price. Existing states without it normalize to `None`; do not infer it from later prices.

- [ ] **Step 4: Run the focused and runtime-state tests**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "runtime_state or position_state or atr_shadow" -q
```

Expected: PASS.

- [ ] **Step 5: Commit isolated state**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "test(resonance): define isolated ATR shadow state"
```

### Task 2: Register exactly once after a fully filled ATR exit

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Consumes: `submit_sell(context, code, reason, trigger_value)` 清理前的状态副本、完成后的分类结果，以及完成卖单后才读取的 09:35 quote。
- Produces: `try_register_atr_exit_shadow(code, state, exit_date, current_data, stop_price) -> bool` without changing the sell result。

- [ ] **Step 1: Write the complete registration truth-table tests**

```python
@pytest.mark.parametrize("reason,outcome,remaining,expected", [
    (strategy.ExitReason.ATR_EXIT, strategy.OrderOutcome.FILLED, 0, 1),
    (strategy.ExitReason.ATR_EXIT, strategy.OrderOutcome.PARTIAL, 100, 0),
    (strategy.ExitReason.ATR_EXIT, strategy.OrderOutcome.NOT_FILLED, 100, 0),
    (strategy.ExitReason.SIGNAL_EXIT, strategy.OrderOutcome.FILLED, 0, 0),
])
def test_atr_shadow_registers_only_for_full_atr_exit(
        monkeypatch, reason, outcome, remaining, expected):
    calls = []
    arrange_submit_sell(monkeypatch, outcome=outcome, after_amount=remaining)
    monkeypatch.setattr(
        strategy, "try_register_atr_exit_shadow",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    strategy.submit_sell(context(), "510300.XSHG", reason, 3.5)
    assert len(calls) == expected


def test_atr_shadow_registration_error_cannot_change_filled_sell(monkeypatch):
    arrange_submit_sell(monkeypatch, outcome=strategy.OrderOutcome.FILLED,
                        after_amount=0)
    monkeypatch.setattr(
        strategy, "register_atr_exit_shadow",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("shadow")),
    )
    result = strategy.submit_sell(
        context(), "510300.XSHG", strategy.ExitReason.ATR_EXIT, 3.5,
    )
    assert result is strategy.OrderOutcome.FILLED
    assert "510300.XSHG" in strategy.g.sold_today


def test_atr_shadow_quote_is_not_read_until_after_sell_state_sync(monkeypatch):
    calls = []
    arrange_submit_sell(
        monkeypatch, outcome=strategy.OrderOutcome.FILLED,
        after_amount=0, calls=calls,
    )
    monkeypatch.setattr(
        strategy, "get_execution_price",
        lambda current_data, code: calls.append("shadow_quote") or 3.6,
    )

    strategy.submit_sell(
        context(), "510300.XSHG", strategy.ExitReason.ATR_EXIT, 3.5,
    )

    assert calls.index("order_target") < calls.index("state_sync")
    assert calls.index("state_sync") < calls.index("shadow_quote")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "shadow_registers or shadow_registration_error" -q
```

Expected: FAIL until registration is wired after outcome classification.

- [ ] **Step 3: Implement post-sell registration without changing control flow**

Before `sync_sell_state_after_order`, copy only the fields needed by the shadow record；不得为了影子观察新增任何价格、日历或行情读取。完成原有 state sync 和 order-transition logging 后：

```python
if (reason is ExitReason.ATR_EXIT
        and result is OrderOutcome.FILLED
        and after_amount == 0
        and state_before is not None):
    try_register_atr_exit_shadow(
        code, state_before, context.current_dt.date(),
        current_data, trigger_value,
    )
```

`try_register_atr_exit_shadow` 内部才调用 `get_execution_price(current_data, code)`；无有效报价时返回 `False` 并记录诊断。它必须 catch ordinary exceptions and emit `atr_shadow_registration` with `error_type`; `_is_future_data_error(error)` must still rethrow unchanged。此时原卖单、持仓同步和订单日志已经完成。它 returns a boolean that no caller may use for trading.

- [ ] **Step 4: Verify retry and duplicate behavior**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "pending_exit or retry_pending or atr_shadow" -q
```

Expected: a partial first attempt registers zero records; the later full retry registers one; repeated registration of the same ID returns `False` and places no duplicate.

- [ ] **Step 5: Commit registration**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): observe completed ATR exits"
```

### Task 3: Record H1/H3/H5 outcomes after close without orders

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Produces: `record_due_atr_shadow_outcomes(context, current_data) -> None`。
- Emits: `atr_exit_shadow_registered`, `atr_exit_shadow_outcome`, `atr_shadow_diagnostic`。

- [ ] **Step 1: Write failing temporal-boundary tests**

```python
def test_atr_shadow_outcomes_use_exact_future_sessions_and_never_order(monkeypatch):
    install_shadow_event(event_date="2021-01-04", reference_price=10.0)
    monkeypatch.setattr(strategy, "get_trade_days", lambda **kwargs: [
        date(2021, 1, 4), date(2021, 1, 5),
    ])
    order_calls = forbid_all_order_functions(monkeypatch)

    strategy.record_due_atr_shadow_outcomes(
        context_at("2021-01-05 15:30:00"), current_data(price=10.5),
    )

    assert shadow_outcome(1)["return"] == pytest.approx(0.05)
    assert order_calls == []


def test_atr_shadow_future_data_error_propagates(monkeypatch):
    install_shadow_event(event_date="2021-01-04", reference_price=10.0)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: (_ for _ in ()).throw(FutureDataError()),
    )
    with pytest.raises(FutureDataError):
        strategy.record_due_atr_shadow_outcomes(
            context_at("2021-01-05 15:30:00"), current_data(price=10.5),
        )
```

- [ ] **Step 2: Run temporal tests and verify RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_shadow_outcomes or atr_shadow_future" -q
```

- [ ] **Step 3: Implement a separate close-only recorder**

Use the same elapsed-session calculation as `_record_due_observation_outcomes_for_record`, but read and mutate only `g.atr_shadow_events`, emit only ATR shadow event names, and compute:

```python
outcome = {
    "status": "RECORDED",
    "closing_date": due_date,
    "closing_price": closing_price,
    "return": closing_price / record["reference_price"] - 1.0,
    "recovered_entry": (
        None if not is_finite_positive(record.get("entry_price"))
        else closing_price >= record["entry_price"]
    ),
}
```

Call it in `after_close` after existing formal/relative outcomes and before `log_portfolio_summary`. It must not call `submit_buy`, `submit_sell`, `order_target`, or `order_target_value`.

- [ ] **Step 4: Run stage-order and observation suites**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "after_close or observation or atr_shadow" -q
```

Expected: PASS; existing observation records remain byte-contract compatible.

- [ ] **Step 5: Commit outcome recording**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): record post-ATR shadow outcomes"
```

### Task 4: Add the read-only ATR shadow analyzer

**Files:**
- Create: `resonance_reversal_strategy/research/analyze_atr_exit_shadows.py`
- Create: `tests/test_resonance_atr_exit_shadow_analysis.py`

**Interfaces:**
- Consumes: `.4` baseline log, `20260828.1` candidate log, schema V2 manifest and frozen SHA-256。
- Produces: registration/complete/right-censored counts, H1/H3/H5 summaries, yearly summaries, entry-recovery rates, exact order/portfolio comparisons。

- [ ] **Step 1: Write failing analyzer tests**

```python
def test_shadow_analyzer_requires_exact_order_and_portfolio_path():
    report = analyze_fixture(candidate_order_price=10.01,
                             baseline_order_price=10.00)
    assert report["gates"]["formal_order_path_exact"] is False
    assert report["continue_atr_investigation"] is False


def test_shadow_analyzer_right_censors_training_end_without_2022():
    report = analyze_fixture(
        registration_date="2021-12-30", outcomes={},
    )
    assert report["data_quality"]["right_censored_count"] == 1
    assert report["data_quality"]["errors"] == []
```

- [ ] **Step 2: Run analyzer tests and verify RED**

```powershell
python -m pytest tests/test_resonance_atr_exit_shadow_analysis.py -q
```

- [ ] **Step 3: Implement strict parsing and descriptive gates**

The analyzer must validate namespace, identity, exact horizons, registration-before-outcome, no duplicates, no 2022 result, exact 138-order path, exact 730-point portfolio path and exact 23,856.40 final asset. Report, but do not auto-change ATR, when:

```python
continue_atr_investigation = all((
    data_quality_complete,
    formal_order_path_exact,
    portfolio_path_exact,
    complete_count >= 20,
    horizon_5["median"] is not None and horizon_5["median"] > 0,
    horizon_5["hit_rate"] is not None and horizon_5["hit_rate"] > 0.5,
))
```

This boolean only permits a new design discussion.

- [ ] **Step 4: Run analyzer and strategy suites**

```powershell
python -m pytest tests/test_resonance_atr_exit_shadow_analysis.py tests/test_resonance_reversal_strategy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit analyzer**

```powershell
git add resonance_reversal_strategy/research/analyze_atr_exit_shadows.py tests/test_resonance_atr_exit_shadow_analysis.py
git commit -m "feat(resonance): analyze post-ATR shadow returns"
```

### Task 5: Freeze build, docs, and platform evidence protocol

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`
- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`

**Interfaces:**
- Produces build `20260828.1` with unchanged formal/parameter/pool/relative fingerprints and a separate ATR-shadow fingerprint。

- [ ] **Step 1: Write failing build and non-interference tests**

```python
def test_atr_shadow_build_and_fingerprints_are_separate(monkeypatch):
    messages = []
    scheduled = []
    _install_initialize_platform_stubs(
        monkeypatch, messages, scheduled,
    )
    monkeypatch.setattr(
        strategy, "g", types.SimpleNamespace(), raising=False,
    )

    strategy.initialize(fake_context())
    payload = next(
        json.loads(message) for message in messages
        if json.loads(message)["event"] == "strategy_initialized"
    )

    assert strategy.DEPLOYMENT_BUILD_ID == "20260828.1"
    assert payload["event_logic_fingerprint"] == "1c0b8a22f48c97c3"
    assert payload["parameter_fingerprint"] == "e1227fbd8b4a884e"
    assert payload["pool_fingerprint"] == "9123995edeb1ed84"
    assert payload["relative_observation_fingerprint"] == (
        "f47d32b87be6d926"
    )
    assert "atr_shadow_fingerprint" in payload
```

- [ ] **Step 2: Update only the candidate build and diagnostic documentation**

Document that `ATR_SHADOW` is observation-only, its reference is the 09:35 decision quote, and ordinary sidecar failure cannot change a completed sell.

- [ ] **Step 3: Run complete verification**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py tests/test_resonance_atr_exit_shadow_analysis.py tests/test_resonance_relative_turn_analysis.py -q
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py resonance_reversal_strategy/research/analyze_atr_exit_shadows.py
git diff --check
```

Expected: all tests PASS, compile succeeds, no diff errors.

- [ ] **Step 4: Run JoinQuant 2019--2021 replay and compare evidence**

Required before interpreting shadows:

```text
formal decisions: 8808 exact
filled orders: 138 exact after entrust_id normalization
portfolio summaries: 730 exact
final asset: 23856.40 exact
ATR shadow registrations: one per fully filled ATR exit
```

If any formal path differs, stop; do not analyze shadow returns.

- [ ] **Step 5: Commit the completed observation milestone**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py resonance_reversal_strategy/README.md resonance_reversal_strategy/docs/strategy_spec.md
git commit -m "feat(resonance): add non-interfering ATR shadow study"
```
