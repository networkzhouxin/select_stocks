# BOLL 买入逻辑失效退出候选实施计划

> **执行要求：** 使用 `superpowers:executing-plans` 逐项执行。每个任务均先写失败测试，再做最小实现，再运行指定验证。本文只描述候选 `20260828.3`；不得顺带实现相对买入补位、ATR 退出影子观察或任何组合版本。

**目标：** 在不移除、不放宽 ATR 止损的前提下，为由 BOLL 支持的正式买入增加一个只使用 T-1 日线的“原买入逻辑已失效”退出。该候选用于检验更早释放失败反转仓位能否提高胜率、收益质量并降低回撤。

**架构：** 买入成交后，把买入决策所引用的正式 BOLL 买入事件低点（现有 `event_book.active.BOLL.reference_extreme`）保存到持仓状态；随后每个 T 日 09:35 只用截至 T-1 的正式信号快照比较 `close` 与该参考值。仅当 `close < entry_boll_reference_extreme` 时产生 `BOLL_THESIS_EXIT`。ATR 仍先运行且优先级最高；正式信号退出仍保留。

**技术栈：** Python 3、JoinQuant 策略 API、现有 `pytest` 测试与结构化 JSON 日志。

---

## 全局边界与基线

- 基线提交：`020bc36`。
- 基线策略：`resonance-v0.1.0` / build `20260827.4`。
- 候选 build：`20260828.3`；策略主版本号保持不变，避免把候选误标为正式版本。
- 训练期：2019-01-01 至 2021-12-31；只允许 2018 只读 warm-up。
- 日线信号与 BOLL 失效判定只能使用 T-1 及以前数据；T 日 09:35 价格只用于执行。
- 禁止读取、筛选或调参于 2022 年及以后数据。
- 不修改 `cross_signal_strategy`。
- 不移除 ATR，不修改 ATR(14)、2.5 倍数、5% 下限或 15% 上限。
- 不新增止盈、时间止损、冷却、加仓、市场状态过滤或其他退出规则。
- 不让相对观察、相对买入或 `SOFT_ALL_THREE` 参与交易。
- 该候选必须从基线独立产生；不得叠加 `20260828.1` 或 `20260828.2`。

## 范围契约

| 项目 | 契约 |
|---|---|
| 目标对象 | 已由正式共振买入并且买入支持者包含 `BOLL` 的持仓 |
| 目标阶段 | 买入状态初始化；其后每日 T-1 信号快照完成后、正式信号退出前 |
| 允许变化 | 当 T-1 收盘价严格跌破买入时冻结的正式 BOLL 买入事件低点时，提交 `BOLL_THESIS_EXIT` |
| 必须不变 | ATR 检查及优先级、正式信号退出、买入候选、排序、仓位、重试、成交同步、观察事件 |
| 禁止传播 | 非 BOLL 支持买入、相对观察、相对买入、历史遗留状态、无效快照、T 日盘中价格 |

## 控制流契约与真值表

新增状态字段 `entry_boll_reference_extreme` 只表达“本持仓买入时冻结的正式 BOLL 买入事件 `reference_extreme` 低点”，不得作为买入来源、环境、重试状态或 ATR 状态的代理，也不得替换为当时或当前的 BOLL 下轨。

| 已持仓 | 当日已卖出 | 快照有效 | 参考值有效 | `T-1 close < reference` | 行为 |
|---|---:|---:|---:|---:|---|
| 否 | 任意 | 任意 | 任意 | 任意 | 不处理 |
| 是 | 是 | 任意 | 任意 | 任意 | 不重复提交 |
| 是 | 否 | 否 | 任意 | 任意 | 不触发；保留 ATR/正式退出路径 |
| 是 | 否 | 是 | 否 | 任意 | 不触发；兼容历史状态 |
| 是 | 否 | 是 | 是 | 否 | 不触发；等于参考值也不退出 |
| 是 | 否 | 是 | 是 | 是 | 提交 `BOLL_THESIS_EXIT` |

退出原因优先级固定为：

| 原因 | 优先级 | 说明 |
|---|---:|---|
| `SIGNAL_EXIT` | 1 | 现有正式信号退出 |
| `BOLL_THESIS_EXIT` | 2 | 新增 T-1 买入逻辑失效退出 |
| `ATR_EXIT` | 3 | 现有硬风险退出，最高优先级 |

同日同一证券只允许一个卖单。高优先级原因可以升级尚未成交的低优先级 pending 原因；低优先级原因不得覆盖高优先级原因。

---

### Task 1：冻结并持久化买入时 BOLL 事件低点

**文件：**

- 修改：`resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- 修改：`tests/test_resonance_reversal_strategy.py`

#### Step 1：写失败测试

在现有策略测试模块中复用已加载的 `strategy` 模块以及 `resonance_snapshot`、`fake_context`、`current_record` 和 `runtime_state` 工厂。先在测试文件内新增下列最小工厂：

```python
def make_boll_buy_snapshot(code="510300.XSHG", close=4.0,
                           reference_extreme=3.8):
    snapshot = resonance_snapshot(code)
    snapshot["close"] = close
    snapshot["event_book"]["active"]["BOLL"][
        "reference_extreme"
    ] = reference_extreme
    return snapshot


def make_buy_decision(code="510300.XSHG", supporters=("BOLL", "RSI")):
    return {
        "code": code,
        "direction": strategy.TurnDirection.BUY_TURN,
        "supporters": tuple(supporters),
    }
```

新增测试覆盖：

```python
def test_extract_entry_boll_reference_requires_boll_support_and_active_event():
    snapshot = make_boll_buy_snapshot()

    assert strategy.extract_entry_boll_reference(
        snapshot, make_buy_decision()
    ) == pytest.approx(3.8)
    assert strategy.extract_entry_boll_reference(
        snapshot, make_buy_decision(supporters=("RSI", "KDJ"))
    ) is None


def test_extract_entry_boll_reference_rejects_missing_non_finite_or_non_positive_values():
    snapshot = make_boll_buy_snapshot()

    for value in (None, float("nan"), float("inf"), 0.0, -1.0):
        snapshot["event_book"]["active"]["BOLL"][
            "reference_extreme"
        ] = value
        assert strategy.extract_entry_boll_reference(
            snapshot, make_buy_decision()
        ) is None


def test_position_state_keeps_frozen_entry_boll_reference():
    state = strategy.make_position_state(
        buy_date="2020-01-02",
        entry_atr=0.2,
        entry_price=4.0,
        entry_boll_reference_extreme=3.8,
    )

    assert state["entry_boll_reference_extreme"] == pytest.approx(3.8)
```

再补一个成交同步测试：正式买单完全成交时把 `3.8` 写入 `g.position_states[code]`；后续快照中的 BOLL 事件低点变成 `3.9` 时，该字段仍为 `3.8`。

#### Step 2：运行失败测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "entry_boll_reference or position_state_keeps_frozen" -q
```

预期：失败，原因是提取函数、参数或状态字段尚不存在。

#### Step 3：做最小实现

在策略中增加纯函数：

```python
def extract_entry_boll_reference(snapshot, decision):
    if (not snapshot
            or decision.get("direction") is not TurnDirection.BUY_TURN):
        return None
    if "BOLL" not in tuple(decision.get("supporters", ())):
        return None

    boll_event = (
        (snapshot.get("event_book") or {}).get("active", {}).get("BOLL")
    )
    if (boll_event is None
            or boll_event.get("direction") is not TurnDirection.BUY_TURN):
        return None

    value = boll_event.get("reference_extreme")
    if not is_finite_positive(value):
        return None
    return float(value)
```

把 `make_position_state` 扩展为：

```python
def make_position_state(
        buy_date,
        entry_atr,
        entry_price,
        entry_boll_reference_extreme=None):
    return {
        "buy_date": buy_date,
        "entry_atr": float(entry_atr),
        "highest_close_anchor": float(entry_price),
        "pending_exit": None,
        "entry_boll_reference_extreme": entry_boll_reference_extreme,
    }
```

在 `submit_buy` 提交正式买入前，由当次冻结快照与正式决策计算参考值；把 `entry_boll_reference_extreme` 作为 `sync_buy_state_after_order` 的最后一个可选参数传入，再传给 `make_position_state`。只在实际增持导致首次建立/重建该持仓状态时保存，后续日线不得刷新。

`is_finite_positive` 必须复用策略已有等价数值校验；若没有，则新增只做 `math.isfinite(float(value)) and float(value) > 0` 的纯函数，不扩大其他调用点。

历史或恢复状态缺少该字段时，读取必须使用 `.get("entry_boll_reference_extreme")`，不得迁移、猜测或用当天 BOLL 值回填。

#### Step 4：运行目标测试和完整策略测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "entry_boll_reference or position_state_keeps_frozen" -q
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

预期：目标测试和完整策略测试通过。

#### Step 5：提交安全点

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): freeze entry boll reference"
```

---

### Task 2：增加纯 BOLL 失效判定与退出优先级

**文件：**

- 修改：`resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- 修改：`tests/test_resonance_reversal_strategy.py`

#### Step 1：写失败测试

测试局部工厂：

```python
def make_thesis_state(reference=3.8, pending_exit=None):
    return {
        "buy_date": "2020-01-02",
        "entry_atr": 0.2,
        "highest_close_anchor": 4.0,
        "pending_exit": pending_exit,
        "entry_boll_reference_extreme": reference,
    }


def make_close_snapshot(close=3.79, valid=True):
    return {"valid": valid, "close": close}
```

新增测试：

```python
@pytest.mark.parametrize(
    "close, expected",
    [(3.79, True), (3.80, False), (3.81, False)],
)
def test_boll_thesis_invalidation_uses_strict_t_minus_one_close(close, expected):
    assert strategy.boll_thesis_is_invalidated(
        make_thesis_state(), make_close_snapshot(close)
    ) is expected


@pytest.mark.parametrize("reference", [None, float("nan"), 0.0, -1.0])
def test_boll_thesis_invalidation_ignores_invalid_or_legacy_reference(reference):
    assert strategy.boll_thesis_is_invalidated(
        make_thesis_state(reference), make_close_snapshot()
    ) is False


def test_boll_thesis_invalidation_ignores_invalid_snapshot():
    assert strategy.boll_thesis_is_invalidated(
        make_thesis_state(), make_close_snapshot(valid=False)
    ) is False


def test_exit_priority_keeps_atr_above_boll_and_boll_above_signal():
    assert strategy.EXIT_PRIORITY[strategy.ExitReason.ATR_EXIT] == 3
    assert strategy.EXIT_PRIORITY[strategy.ExitReason.BOLL_THESIS_EXIT] == 2
    assert strategy.EXIT_PRIORITY[strategy.ExitReason.SIGNAL_EXIT] == 1
```

#### Step 2：运行失败测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "boll_thesis_invalidation or exit_priority_keeps_atr" -q
```

预期：失败，原因是新退出枚举、优先级和判定函数尚不存在。

#### Step 3：做最小实现

```python
class ExitReason(Enum):
    ATR_EXIT = "ATR_EXIT"
    BOLL_THESIS_EXIT = "BOLL_THESIS_EXIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"


EXIT_PRIORITY = {
    ExitReason.SIGNAL_EXIT: 1,
    ExitReason.BOLL_THESIS_EXIT: 2,
    ExitReason.ATR_EXIT: 3,
}


def boll_thesis_is_invalidated(position_state, snapshot):
    if not snapshot or not snapshot.get("valid"):
        return False
    reference = position_state.get("entry_boll_reference_extreme")
    close = snapshot.get("close")
    if not is_finite_positive(reference) or not is_finite_positive(close):
        return False
    return float(close) < float(reference)
```

不得以 `<=` 替代 `<`，不得改用当天实时价、当天最低价、当前 BOLL 下沿或 ATR 线。

#### Step 4：运行目标测试和完整策略测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "boll_thesis_invalidation or exit_priority_keeps_atr" -q
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

预期：全部通过。

#### Step 5：提交安全点

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): define boll thesis exit"
```

---

### Task 3：把 BOLL 失效退出接入现有卖出状态机

**文件：**

- 修改：`resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- 修改：`tests/test_resonance_reversal_strategy.py`

#### Step 1：写失败测试

复用现有卖出、持仓和 current-data 测试工厂，新增以下行为测试：

```python
def test_run_boll_thesis_exits_submits_one_sell_for_invalidated_position(monkeypatch):
    context, current_data, snapshots = arrange_held_boll_position(
        monkeypatch, reference=3.8, t_minus_one_close=3.79
    )
    submitted = capture_submit_sell(monkeypatch)

    strategy.run_boll_thesis_exits(context, current_data, snapshots)

    assert submitted == [(
        "510300.XSHG", strategy.ExitReason.BOLL_THESIS_EXIT, 3.79,
    )]


def test_run_boll_thesis_exits_does_not_sell_without_frozen_boll_reference(monkeypatch):
    context, current_data, snapshots = arrange_held_boll_position(
        monkeypatch, reference=None, t_minus_one_close=3.70
    )
    submitted = capture_submit_sell(monkeypatch)

    strategy.run_boll_thesis_exits(context, current_data, snapshots)

    assert submitted == []


def test_boll_thesis_exit_does_not_duplicate_atr_sell(monkeypatch):
    context, current_data, snapshots = arrange_held_boll_position(
        monkeypatch, reference=3.8, t_minus_one_close=3.70
    )
    strategy.g.sold_today.add("510300.XSHG")
    submitted = capture_submit_sell(monkeypatch)

    strategy.run_boll_thesis_exits(context, current_data, snapshots)

    assert submitted == []


def test_boll_thesis_exit_upgrades_signal_pending_but_not_atr_pending():
    signal_state = make_thesis_state()
    strategy.set_pending_exit(
        signal_state, strategy.ExitReason.SIGNAL_EXIT,
        "2021-01-06", 3.90, 1000,
    )
    upgraded = strategy.set_pending_exit(
        signal_state, strategy.ExitReason.BOLL_THESIS_EXIT,
        "2021-01-07", 3.79, 1000,
    )
    assert upgraded["reason"] is strategy.ExitReason.BOLL_THESIS_EXIT

    atr_state = make_thesis_state()
    strategy.set_pending_exit(
        atr_state, strategy.ExitReason.ATR_EXIT,
        "2021-01-06", 3.70, 1000,
    )
    retained = strategy.set_pending_exit(
        atr_state, strategy.ExitReason.BOLL_THESIS_EXIT,
        "2021-01-07", 3.79, 1000,
    )
    assert retained["reason"] is strategy.ExitReason.ATR_EXIT
```

为使上述片段自足，在测试文件中按现有对象形状实现局部辅助函数；其职责必须仅限测试布置与捕获：

```python
def arrange_held_boll_position(monkeypatch, reference, t_minus_one_close):
    code = "510300.XSHG"
    state = make_thesis_state(reference=reference)
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(1000)})
    current_data = {code: current_record(price=3.78, paused=False)}
    snapshots = {code: make_close_snapshot(t_minus_one_close)}
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_current_data", lambda: current_data, raising=False,
    )
    return context, current_data, snapshots


def capture_submit_sell(monkeypatch):
    submitted = []

    def fake_submit_sell(context, code, reason, trigger_value):
        submitted.append((code, reason, trigger_value))
        return strategy.OrderOutcome.FILLED

    monkeypatch.setattr(strategy, "submit_sell", fake_submit_sell)
    return submitted
```

暂停证券还要有一个独立测试：不得调用 `submit_sell`，而应通过现有 `set_pending_exit` 写入 `BOLL_THESIS_EXIT`，等待下个交易日的统一 pending 重试路径。

再加流程顺序测试，精确断言：

```text
retry pending exits
ATR exits
build T-1 snapshots
log snapshots
relative observation only
BOLL thesis exits
formal signal exits
formal signal buys
```

同时验证 BOLL 失效卖出成功后证券进入 `g.sold_today`，当日后续正式信号买入阶段不能回补该证券。

#### Step 2：运行失败测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "run_boll_thesis_exits or boll_thesis_exit or stage_order" -q
```

预期：失败，原因是 runner 尚未接入。

#### Step 3：做最小实现

新增职责单一的 runner：

```python
def run_boll_thesis_exits(context, current_data, snapshots):
    attempted = set()
    decision_date = context.current_dt.date()
    retried_codes = getattr(g, "daily_retried_exits", set())
    for code in sorted(get_actual_positions(context)):
        if code in g.sold_today:
            continue

        state = g.position_states.get(code)
        if state is None:
            continue
        snapshot = snapshots.get(code)
        if not boll_thesis_is_invalidated(state, snapshot):
            continue

        trigger_value = float(snapshot["close"])
        actual_amount = get_actual_amount(context, code)
        if code in retried_codes:
            set_pending_exit(
                state, ExitReason.BOLL_THESIS_EXIT, decision_date,
                trigger_value, actual_amount,
            )
            continue

        if get_tradability(current_data, code) is Tradability.PAUSED:
            set_pending_exit(
                state, ExitReason.BOLL_THESIS_EXIT, decision_date,
                trigger_value, actual_amount,
            )
            continue

        submit_sell(
            context, code, ExitReason.BOLL_THESIS_EXIT, trigger_value,
        )
        attempted.add(code)
    return attempted
```

必须直接复用现有 `g.position_states`、`g.daily_retried_exits`、`set_pending_exit(position_state, reason, created_date, trigger_value, remaining_amount)` 和 `submit_sell(context, code, reason, trigger_value)`；不得新增第二套 pending 状态或重构共享卖出方法。`set_pending_exit` 的现有优先级逻辑保证 ATR 不被降级。

在 `do_trading` 中只插入一次：

在现有 `run_relative_observation_stage(snapshots)` 语句之后、`run_signal_exits(context, current_data, snapshots)` 之前插入且只插入：

```python
run_boll_thesis_exits(context, current_data, snapshots)
```

不得把 BOLL 失效判定放到 ATR 之前，也不得在生成 T-1 快照前读取 T 日价格判定逻辑失效。

#### Step 4：运行目标测试和完整策略测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "run_boll_thesis_exits or boll_thesis_exit or stage_order" -q
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

预期：全部通过，现有 ATR、正式信号、相对观察、买入槽位、pending 重试测试均无回归。

#### Step 5：提交安全点

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): exit invalidated boll entries"
```

---

### Task 4：补齐结构化日志、build 标识和非目标回归证明

**文件：**

- 修改：`resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- 修改：`tests/test_resonance_reversal_strategy.py`
- 修改：`resonance_reversal_strategy/README.md`（仅在已有候选运行说明区增加一行）

#### Step 1：写失败测试

新增测试断言：

- build 精确等于 `20260828.3`；
- BOLL 失效卖单的 `order_transition.exit_reason` 为 `BOLL_THESIS_EXIT`；
- 独立 `boll_thesis_exit_decision` 日志含冻结参考值、T-1 close、signal_date、decision_date 和动作；
- 日志不得把 T 日 09:35 quote 记录为信号 close；
- 非 BOLL 持仓、无参考值历史持仓与相等边界不会产生该原因；
- 原 `ATR_EXIT` 与 `SIGNAL_EXIT` 日志字段和分类不变；
- 相对观察事件数量和 ID 规则不变。

示例核心断言：

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


def json_event(messages, event_type):
    return next(
        payload for payload in (json.loads(message) for message in messages)
        if payload["event"] == event_type
    )


def test_boll_thesis_exit_log_preserves_signal_and_execution_boundaries(monkeypatch):
    messages = capture_json_logs(monkeypatch)
    state = make_thesis_state(reference=3.8)
    snapshot = {
        "valid": True,
        "close": 3.79,
        "signal_date": date(2021, 1, 5),
    }

    strategy.log_boll_thesis_exit_decision(
        "510300.XSHG", state, snapshot,
        decision_date=date(2021, 1, 6),
        action="SELL_ATTEMPT",
    )
    event = json_event(messages, "boll_thesis_exit_decision")

    assert event["exit_reason"] == "BOLL_THESIS_EXIT"
    assert event["entry_boll_reference_extreme"] == pytest.approx(3.8)
    assert event["signal_close"] == pytest.approx(3.79)
    assert event["signal_date"] < event["decision_date"]
```

再直接调用一次扩展后的 `log_order_transition`，断言卖出事件的 `exit_reason == "BOLL_THESIS_EXIT"`；买入事件的该字段为 `None`。测试辅助只解析 `_emit_structured_log` 产生的 JSON 字符串。

#### Step 2：运行失败测试

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "boll_thesis_exit_log or deployment_build" -q
```

预期：失败，原因是 build 和新增日志字段尚未更新。

#### Step 3：做最小实现

- 把 `DEPLOYMENT_BUILD_ID` 改为 `20260828.3`。
- 把 `log_order_transition` 的最后一个可选参数扩展为 `exit_reason=None`，payload 总是写入 `exit_reason`；`submit_sell` 传入真实原因，`submit_buy` 保持默认 `None`。这只扩展日志，不改变订单分类或同步。
- 新增 `log_boll_thesis_exit_decision(code, position_state, snapshot, decision_date, action)`，固定写出：
  - `exit_reason=BOLL_THESIS_EXIT`
  - `entry_boll_reference_extreme`
  - `signal_close`
  - `signal_date`
  - `decision_date`
- `action` 只能是 `SELL_ATTEMPT`、`PAUSED_PENDING` 或 `PENDING_REASON_UPDATE`；在 Task 3 的 runner 确认失效后、执行对应分支前记录一次。
- 不改变其他退出原因的必需字段、事件类型、订单状态分类或现有 ID。
- README 只写候选 build、基线和“ATR 保留、BOLL 买入逻辑失效退出”一句话，不写未经回测的收益结论。

#### Step 4：运行静态、完整测试与差异检查

```powershell
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
python -m pytest tests/test_resonance_reversal_strategy.py -q
python -m pytest tests/test_resonance_relative_turn_analysis.py -q
git diff --check
```

预期：编译、测试和差异检查全部通过。

#### Step 5：逐 hunk 做范围映射

执行并人工检查：

```powershell
git diff -- resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py resonance_reversal_strategy/README.md
```

每个变更必须只能映射到以下一项：

1. 冻结 BOLL 参考值；
2. T-1 严格跌破判定；
3. 新退出原因与既定优先级；
4. 接入既有卖出状态机；
5. build、日志或候选说明；
6. 上述行为的测试。

任何不能映射的重排、重命名、格式化或逻辑修改都应移除。

#### Step 6：提交候选实现

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py resonance_reversal_strategy/README.md
git commit -m "feat(resonance): add boll thesis exit candidate"
```

---

### Task 5：执行一次冻结训练回测并用统一评估器作裁决

**前置条件：** `2026-08-28-resonance-candidate-evaluator.md` 已完成并通过；聚宽代码必须来自候选提交的干净工作树。

#### Step 1：记录运行身份

在聚宽中使用：

- 2019-01-01 至 2021-12-31；
- 初始资金 20,000 元；
- 与 `.4`、`cross-v0.3.3` 相同佣金、滑点、频率和基准；
- manifest SHA256：`24cfbdb7cfcac61c1e8a6f58bbdf54f851031ad63a7feb2ac331590ab7ede87f`；
- build：`20260828.3`；
- 候选提交 SHA。

一次性导出完整“日志”，不得先看分年或局部结果再改阈值。

#### Step 2：先做协议与因果边界检查

必须满足：

- session_count=973，first=2018-01-02，last=2021-12-31；
- 2019-2021 正式决策日均在 manifest；
- 所有 `signal_date < decision_date`；
- `BOLL_THESIS_EXIT` 只发生于持仓存在有效冻结参考值且 `signal_close < reference`；
- ATR 日检查仍存在，且没有同一证券同日重复卖单；
- 非目标路径的订单状态转换仍可重建。

任何一项失败，先修协议/实现错误，不能解释收益。

#### Step 3：运行统一评估器

```powershell
python resonance_reversal_strategy/research/analyze_candidate_performance.py `
  --baseline-log <absolute-path-to-resonance-20260827.4-log> `
  --candidate-log <absolute-path-to-20260828.3-log> `
  --double-friction-log <absolute-path-to-20260828.3-double-friction-log> `
  --expected-baseline-build 20260827.4 `
  --expected-candidate-build 20260828.3 `
  --session-calendar <absolute-path-to-frozen-manifest.json> `
  --session-calendar-sha256 24cfbdb7cfcac61c1e8a6f58bbdf54f851031ad63a7feb2ac331590ab7ede87f `
  --output <absolute-path-outside-source-data-folder>
```

普通摩擦和双摩擦日志必须在调用前分别完成；不得复用同一物理文件，也不得仅凭聚宽收益概览手工判定。

#### Step 4：按预注册门槛裁决

最终目标门槛：

- 总收益 `> 129.25%`；
- 完成交易胜率 `> 55.8%`，目标 `>= 60%`；
- Wilson 95% 下界 `> 50%`；
- 最大回撤 `< 6.28%`；
- 完成交易数 `>= 80`；
- 交易收益中位数 `> 0`；
- 前 10% 完成交易毛利润占比 `<= 50%`；
- 双摩擦总收益 `> 64.10%`。

该独立候选只有在以下条件全部成立时，才可进入后续“是否组合”的讨论：

1. 数据、时间边界、订单状态和公司行动检查全部通过；
2. 总收益不低于 `.4` 的 19.282%；
3. 最大回撤严格低于 `.4` 的 19.0577%；
4. 胜率不低于 `.4` 的 57.3529%；
5. 中位交易收益保持 `> 0`；
6. 前 10% 毛利润占比保持 `<= 50%`；
7. 2021 年收益严格高于 `.4` 的 -12.9805%；
8. ATR 参数、触发顺序、5%--15% 边界和正式信号退出定义均不变。

通过独立候选门槛不等于授权组合。若未达到最终目标，只能报告失败或有限改善；不得围绕同一训练日志移动 BOLL 参考值、比较符、等待天数或其他相邻阈值。

#### Step 5：形成里程碑总结并等待下一次确认

总结必须列出：

- 候选提交与 build；
- 单次训练与双摩擦运行身份；
- 完整硬门槛表；
- `BOLL_THESIS_EXIT` 交易数、胜率、净损益、持有期和对原 ATR/正式信号退出的替代关系；
- 目标与非目标路径验证；
- 结论：淘汰、保留为独立候选，或建议进入组合设计。

未经用户再次确认，不得把该候选与相对买入补位或 ATR 影子观察驱动的未来改动组合，也不得运行 2022 年以后验证。
