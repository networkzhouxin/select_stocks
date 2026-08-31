# ATR Exit Observation-Only Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `20260827.4` 基线建立独立候选 `20260828.4`，保留 ATR 全部计算和观测，但禁止 ATR 触发创建退出状态或卖单，并以普通/双倍摩擦聚宽回放检验完整的收益、胜率、回撤和反截尾门槛。

**Architecture:** 不增加功能开关或替代退出规则。把原 09:35 ATR 执行阶段替换为职责明确的 `observe_atr_exit_conditions`，该函数只计算原止损状态和输出诊断；`SIGNAL_EXIT`、挂起重试、买入、T-1 信号和收盘锚更新沿用基线调用链。共享候选评估器保持只读，候选专属的期末未平仓门槛在平台验收时单独检查。

**Tech Stack:** Python 3、pytest、JoinQuant API、PowerShell、冻结的 schema V2 交易日 manifest。

**Spec:** `resonance_reversal_strategy/docs/superpowers/specs/2026-08-28-atr-exit-observation-only-candidate-design.md`

## Global Constraints

- 基线必须是 commit `020bc36c1ed8b355205c1e821e07ed0c65da8832` 的 build `20260827.4`；不得从 `.1/.2/.3` 策略候选累积代码。
- 候选 build 固定为 `20260828.4`，只测试一个结构差异：ATR 退出从正式执行改为纯观测。
- 决策信号只使用 T-1 及以前日线；T 日 09:35 当前价仅计算是否“本会触发”以及执行合法的正式信号卖单。
- 训练收益窗口固定为 2019-01-01 至 2021-12-31；2018 只读预热，禁止读取 2022+。
- 保留 ATR(14,2.5)、5%--15% 边界、入场 ATR 有效性、最高收盘锚、ETF 池、资金和全部正式共振参数。
- 不合并 `.2/.3`，不增加 BOLL/时间/固定亏损/止盈退出，不搜索相邻参数。
- JoinQuant 是绩效权威；本地测试只证明代码与因果合同。
- 实施期间不得修改 `cross_signal_strategy`、PTrade、本地行情目录或共享候选评估器。
- 未经用户明确授权不提交或推送；每个任务结束保留可审查 diff 和验证证据。

## Scope Contract

| 字段 | 内容 |
|---|---|
| 目标对象 | 候选策略实际持仓的 ATR 风险退出阶段 |
| 目标处理阶段 | 09:35 挂起退出重试之后、T-1 快照和正式信号退出之前 |
| 允许变化 | ATR 触发仅记录；不创建/升级 `ATR_EXIT`，不调用卖单，不修改 `sold_today` |
| 必须不变 | `SIGNAL_EXIT` 资格、挂起正式退出重试、买入、ATR 指标和持仓风险状态、T-1/09:35/15:30 边界 |
| 不得传播 | 其他候选、其他策略、其他平台、验证期、参数和 ETF 选择 |

## Control-Flow Contract

本计划不新增布尔或模式参数。`ATR_EXIT_POLICY = "OBSERVE_ONLY"` 是日志身份常量，不控制
运行分支。调用方矩阵固定如下：

| 调用方 | 调用目标 | 核心业务 | 状态/清理 | 非目标行为 |
|---|---|---|---|---|
| `do_trading` | `retry_pending_exits` | 原挂起卖单先重试 | 原样同步持仓和 `sold_today` | 不受本候选影响 |
| `do_trading` | `observe_atr_exit_conditions` | 计算并记录 ATR 本会触发 | 不写订单、挂起、`sold_today` | 后续正式阶段必须继续 |
| `do_trading` | `run_signal_exits` | 原正式卖出 | 原挂起/成交语义 | 不借用 ATR 触发作为资格 |
| `do_trading` | `run_signal_buys` | 原正式买入 | 原反重复和槽位语义 | 不因 ATR 本会触发而额外禁买 |
| `after_close` | `update_highest_close_anchor` | 原收盘锚更新 | 原样保留 | 不受退出停用影响 |

---

### Task 1: 建立独立候选并以失败测试冻结 ATR 纯观测合同

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:11-14`
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:793-822`
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:840-868`
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:2226-2272`
- Modify: `tests/test_resonance_reversal_strategy.py:80-120`
- Modify: `tests/test_resonance_reversal_strategy.py:1508-1563`
- Modify: `tests/test_resonance_reversal_strategy.py:1818-1879`
- Modify: `tests/test_resonance_reversal_strategy.py:3195-3229`

**Interfaces:**
- Produces: `ATR_EXIT_POLICY: str = "OBSERVE_ONLY"`，仅供日志身份使用。
- Produces: `observe_atr_exit_conditions(context, current_data) -> set[str]`，返回本会触发 ATR 的代码集合，不提交订单。
- Preserves: `calc_stop_state(highest_close_anchor, entry_atr, params) -> dict | None`。

- [ ] **Step 1: 创建隔离 worktree 并核对基线**

执行实施时先读取 `superpowers:using-git-worktrees`，然后从主仓库运行：

```powershell
git worktree add D:\test\select_stocks\.worktrees\resonance-no-atr-exit -b codex/resonance-no-atr-exit 020bc36c1ed8b355205c1e821e07ed0c65da8832
git -C D:\test\select_stocks\.worktrees\resonance-no-atr-exit rev-parse HEAD
git -C D:\test\select_stocks\.worktrees\resonance-no-atr-exit status --short
```

Expected: HEAD 精确为 `020bc36c1ed8b355205c1e821e07ed0c65da8832`，状态为空。若分支或目录已存在，先只读检查其 HEAD 和状态；不得删除或覆盖未知工作区。

- [ ] **Step 2: 写入纯观测失败测试**

在 `tests/test_resonance_reversal_strategy.py` 新增：

```python
def test_triggered_atr_condition_is_observation_only(monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(date(2021, 1, 5), 2.0, 100.0)
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    payloads = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: payloads.append((event, payload)),
    )
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda *args: pytest.fail("ATR observation must not submit a sell"),
    )

    triggered = strategy.observe_atr_exit_conditions(
        context, {code: current_record(90.0)},
    )

    assert triggered == {code}
    assert runtime.position_states[code] == state
    assert state["pending_exit"] is None
    assert runtime.sold_today == set()
    assert payloads[0][0] == "atr_check"
    assert payloads[0][1]["triggered"] is True
    assert payloads[0][1]["execution_policy"] == "OBSERVE_ONLY"
    assert payloads[0][1]["order_submitted"] is False
```

把阶段顺序测试的期望改为：

```python
assert order == [
    "reset", "pending", "atr_observe", "signals", "signal_sells", "buys",
]
```

并让初始化日志测试断言：

```python
assert initialized_payload["build"] == "20260828.4"
assert initialized_payload["atr_exit_policy"] == "OBSERVE_ONLY"
```

- [ ] **Step 3: 运行聚焦测试确认 RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "triggered_atr_condition_is_observation_only or do_trading_stage_order or initialize" -q
```

Expected: FAIL，原因包括 `observe_atr_exit_conditions` 不存在、build 仍为 `20260827.4`，或初始化日志缺少策略字段；不能接受因夹具错误造成的失败。

- [ ] **Step 4: 实现最小纯观测函数**

策略顶部只增加身份常量并更新 build：

```python
DEPLOYMENT_BUILD_ID = "20260828.4"
ATR_EXIT_POLICY = "OBSERVE_ONLY"
```

`strategy_initialized` 增加：

```python
"atr_exit_policy": ATR_EXIT_POLICY,
```

用以下职责替换原 `run_atr_exits`：

```python
def observe_atr_exit_conditions(context, current_data):
    triggered_codes = set()
    for code in get_actual_positions(context):
        if code in g.sold_today:
            continue
        state = g.position_states.get(code)
        if state is None:
            continue
        stop_state = calc_stop_state(
            state["highest_close_anchor"], state["entry_atr"], g.params,
        )
        execution_price = get_execution_price(current_data, code)
        triggered = bool(
            stop_state is not None
            and execution_price is not None
            and execution_price <= stop_state["stop_price"]
        )
        _emit_structured_log("atr_check", {
            "code": code,
            "entry_atr": state["entry_atr"],
            "highest_close_anchor": state["highest_close_anchor"],
            "stop_price": (
                stop_state["stop_price"] if stop_state is not None else None
            ),
            "stop_pct": (
                stop_state["stop_pct"] if stop_state is not None else None
            ),
            "current_price": execution_price,
            "triggered": triggered,
            "pending_exit": state.get("pending_exit"),
            "execution_policy": ATR_EXIT_POLICY,
            "order_submitted": False,
        })
        if triggered:
            triggered_codes.add(code)
    return triggered_codes
```

`do_trading` 只把原调用替换为：

```python
observe_atr_exit_conditions(context, current_data)
```

不得在观察函数中保留 `retried_codes`、`set_pending_exit`、`submit_sell`、`order_target` 或
`sold_today.add`。

- [ ] **Step 5: 运行聚焦测试确认 GREEN**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "triggered_atr_condition_is_observation_only or do_trading_stage_order or initialize or atr_check_log" -q
```

Expected: PASS。

- [ ] **Step 6: 保存里程碑审查点**

```powershell
git diff -- resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git diff --check
```

Expected: 只有 build、身份日志、ATR 阶段调用、纯观测函数和对应测试变化。未经用户明确授权不提交。

### Task 2: 证明正式退出、挂起重试和订单归因不被 ATR 观测污染

**Files:**
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:279-289`
- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:1119-1148`
- Modify: `tests/test_resonance_reversal_strategy.py:1125-1220`
- Modify: `tests/test_resonance_reversal_strategy.py:1848-2032`
- Modify: `tests/test_resonance_reversal_strategy.py:2910-2970`

**Interfaces:**
- Changes: `log_order_transition(..., pending_exit, exit_reason=None)` 增加只读原因字段。
- Preserves: `submit_sell(context, code, reason, trigger_value) -> OrderOutcome` 的调用合同。
- Preserves: `retry_pending_exits`、`run_signal_exits`、`run_signal_buys` 的业务语义。

- [ ] **Step 1: 写 ATR 触发后正式信号仍唯一卖出的失败测试**

```python
def test_atr_observation_does_not_preempt_same_day_signal_exit(monkeypatch):
    code = "510300.XSHG"
    state = strategy.make_position_state(date(2021, 1, 4), 2.0, 100.0)
    runtime = runtime_state(position_states={code: state})
    context = fake_context(positions={code: fake_position(100)})
    calls = []
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "submit_sell",
        lambda context_arg, order_code, reason, trigger: (
            calls.append((order_code, reason, trigger))
            or strategy.OrderOutcome.FILLED
        ),
    )

    observed = strategy.observe_atr_exit_conditions(
        context, {code: current_record(90.0)},
    )
    attempted = strategy.run_signal_exits(
        context,
        {code: current_record(90.0)},
        {code: resonance_snapshot(code, direction="SELL_TURN")},
    )

    assert observed == {code}
    assert attempted == {code}
    assert calls == [(code, strategy.ExitReason.SIGNAL_EXIT, 10.0)]
```

再把原“pending retry 后 ATR 升级”测试改为断言：重试未成交后即使 ATR 本会触发，
`pending_exit["reason"]` 仍为 `SIGNAL_EXIT`，当日总卖单尝试仍只有重试的一次。

- [ ] **Step 2: 写静态能力隔离测试**

```python
def test_atr_observer_has_no_order_or_exit_state_capability():
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(strategy.observe_atr_exit_conditions)
    ))
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert {"submit_sell", "set_pending_exit", "order_target"}.isdisjoint(
        called_names
    )
    assert "add" not in {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "sold_today"
    }
    assert "pending_exit" not in attributes
```

`pending_exit` 允许只读日志访问，因此最终实现该静态断言时只禁止对
`state["pending_exit"]` 的赋值和 `set_pending_exit` 调用；不要把合法的只读
`state.get("pending_exit")` 误判为状态修改。

- [ ] **Step 3: 写订单原因日志失败测试**

调整已有订单转换测试，明确要求：

```python
strategy.log_order_transition(
    code, strategy.OrderSide.SELL, strategy.OrderOutcome.FILLED,
    100, 0, 0, None, strategy.ExitReason.SIGNAL_EXIT,
)
assert payload["exit_reason"] == "SIGNAL_EXIT"
```

买单日志继续断言 `exit_reason is None`。

- [ ] **Step 4: 运行聚焦测试确认 RED**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_observation_does_not_preempt or atr_observer_has_no_order or order_transition or pending_retry" -q
```

Expected: 新的原因字段测试先失败；行为测试不得因修改正式退出逻辑而变绿。

- [ ] **Step 5: 只增加订单原因诊断，不改卖出业务**

函数签名和 payload 使用：

```python
def log_order_transition(code, side, outcome, before_amount, after_amount,
                         requested_target, pending_exit, exit_reason=None):
    _emit_structured_log("order_transition", {
        "code": code,
        "side": side,
        "outcome": outcome,
        "before_amount": before_amount,
        "after_amount": after_amount,
        "requested_target": requested_target,
        "pending_exit": dict(pending_exit) if pending_exit is not None else None,
        "exit_reason": exit_reason,
    })
```

`submit_buy` 继续使用默认 `None`；`submit_sell` 只在原日志调用末尾传入 `reason`。不得修改
`classify_order_outcome`、`sync_sell_state_after_order`、`set_pending_exit` 或退出优先级。

- [ ] **Step 6: 运行聚焦测试确认 GREEN**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_observation_does_not_preempt or atr_observer_has_no_order or order_transition or pending_retry or signal_exit" -q
```

Expected: PASS；ATR 触发不形成订单，合法正式退出仍只形成一笔 `SIGNAL_EXIT`。

- [ ] **Step 7: 核对全部直接和间接调用方**

```powershell
rg -n "observe_atr_exit_conditions|run_atr_exits|log_order_transition|submit_sell|set_pending_exit|retry_pending_exits" resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
```

Expected:

- `do_trading` 只调用 `observe_atr_exit_conditions`，正式策略调用链不再包含 `run_atr_exits`；
- 观察函数中不存在 `submit_sell` 或 `set_pending_exit`；
- `log_order_transition` 只有买入和卖出两个生产调用方，卖出传 `reason`，买入不传；
- 通用 `retry_pending_exits` 和 `set_pending_exit` 保持原实现。

### Task 3: 更新候选文档并完成本地全量验证

**Files:**
- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`
- Test: `tests/test_resonance_reversal_strategy.py`

**Interfaces:**
- Documents: build `20260828.4`、`OBSERVE_ONLY`、独立候选边界和聚宽验收协议。
- Does not change: 任何研究分析器、manifest schema 或其他策略文档。

- [ ] **Step 1: 更新 README 候选说明**

只新增 `20260828.4` 候选段落，明确：

```text
ATR(14,2.5) 及 5%--15% 边界继续计算和记录；ATR 触发不下单。
正式 SIGNAL_EXIT 是唯一新鲜回放可创建的卖出原因。
该 build 仅用于 2019--2021 独立结构消融，不是当前正式基线或实盘版本。
```

- [ ] **Step 2: 更新策略规格的候选边界**

在 `docs/strategy_spec.md` 新增“ATR 退出停用候选”小节，引用本设计文件，并逐项写明：

- 调用顺序；
- T-1/09:35/15:30 边界；
- 不新增替代退出；
- 期末不强制平仓；
- 普通与双倍摩擦各一次；
- 失败后不进行相邻调参。

不得改写 `20260827.4` 正式基线的历史合同。

- [ ] **Step 3: 运行语法和目标测试**

```powershell
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

Expected: 语法检查成功，resonance 策略测试全部 PASS。

- [ ] **Step 4: 运行 resonance 范围回归**

```powershell
python -m pytest tests -k "resonance" -q
```

Expected: 所有 resonance 测试 PASS；不得以 `cross_signal` 数据缺失掩盖任何 resonance 失败。

- [ ] **Step 5: 运行完整仓库测试并分离非目标故障**

```powershell
python -m pytest -q
```

Expected: 若完整仓库仍因缺少
`G:\financial\history_data\cross_signal_train_2019_2021` 或 cross 失败实验计数产生既有失败，
必须逐项与实施前证据对照并单独报告；任何新增 resonance 失败都必须修复后才能继续。

- [ ] **Step 6: 执行业务和控制流完成门禁**

完成前重新读取 `preserve-business-scope` 与 `preserve-control-flow-semantics`，然后运行：

```powershell
git diff --check
git diff --stat
git diff -- resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py resonance_reversal_strategy/README.md resonance_reversal_strategy/docs/strategy_spec.md
```

逐 hunk 映射：build/日志身份、ATR 纯观测、订单原因诊断、测试、文档。任何无法映射的重构、
格式化或参数变化必须删除。未经用户明确授权不提交或推送。

### Task 4: 执行两次冻结聚宽回放并作反截尾验收

**Files:**
- Read only: `D:\test\select_stocks\.worktrees\resonance-quality-candidates\resonance_reversal_strategy\research\analyze_candidate_performance.py`
- Read only: `C:\Users\C1-CWadmin\.codex\attachments\05a4c904-cb81-4b4b-a1ae-28dd075808b2\pasted-text.txt`
- Generate outside source: `D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log`
- Generate outside source: `D:\test\select_stocks\.artifacts\resonance-no-atr-exit\double-friction.log`
- Generate outside source: `D:\test\select_stocks\.artifacts\resonance-no-atr-exit\candidate-report.json`
- Create after real results: `resonance_reversal_strategy/docs/experiments/2026-08-28-atr-exit-observation-only-result.md`

**Interfaces:**
- Consumes: 完整 JoinQuant 日志、schema V2 manifest 和其冻结 SHA-256。
- Produces: 普通/双倍摩擦指标、最终硬门槛、期末未平仓门槛和 ATR 策略日志审计。

- [ ] **Step 1: 冻结只读分析输入**

```powershell
New-Item -ItemType Directory -Force D:\test\select_stocks\.artifacts\resonance-no-atr-exit
Copy-Item -LiteralPath C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-atr-fc62a10dee0c4974868e5d58b458e6ab\joinquant_sessions_2018_2021.json -Destination D:\test\select_stocks\.artifacts\resonance-no-atr-exit\joinquant_sessions_2018_2021.json
Get-FileHash -Algorithm SHA256 D:\test\select_stocks\.artifacts\resonance-no-atr-exit\joinquant_sessions_2018_2021.json
Get-FileHash -Algorithm SHA256 D:\test\select_stocks\.worktrees\resonance-quality-candidates\resonance_reversal_strategy\research\analyze_candidate_performance.py
```

Expected:

- manifest SHA-256：`24CFBDB7CFCAC61C1E8A6F58BBDF54F851031AD63A7FEB2AC331590AB7EDE87F`；
- 评估器 SHA-256：`2F0B721431A7D888BD8276117FECF8156581D63B609EF7E0784A6864ECE26ECE`。

如果临时 manifest 已不存在，停止并要求用户重新导出同一 schema V2 文件；不得从日志日期或
网络日历重建替代品。

- [ ] **Step 2: 在 JoinQuant 运行普通摩擦回放**

使用候选源码全文，设置：

```text
回测区间：2019-01-01 至 2021-12-31
初始资金：20,000 元
基准：沪深300
基金滑点：0.001
基金开/平仓佣金：0.0003，最低佣金 5 元
```

导出完整日志为：

```text
D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log
```

只运行一次。出现代码错误、日志截断或平台故障时可以修复运行证据后重跑；不得根据收益结果
修改策略。

- [ ] **Step 3: 在 JoinQuant 运行双倍摩擦回放**

策略源码、日期、资金、基准保持相同，只把滑点和比例佣金改为：

```text
基金滑点：0.002
基金开/平仓佣金：0.0006，最低佣金仍为 5 元
```

导出完整日志为：

```text
D:\test\select_stocks\.artifacts\resonance-no-atr-exit\double-friction.log
```

- [ ] **Step 4: 执行冻结评估器**

```powershell
python D:\test\select_stocks\.worktrees\resonance-quality-candidates\resonance_reversal_strategy\research\analyze_candidate_performance.py --baseline-log C:\Users\C1-CWadmin\.codex\attachments\05a4c904-cb81-4b4b-a1ae-28dd075808b2\pasted-text.txt --candidate-log D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log --double-friction-log D:\test\select_stocks\.artifacts\resonance-no-atr-exit\double-friction.log --expected-baseline-build 20260827.4 --expected-candidate-build 20260828.4 --session-calendar D:\test\select_stocks\.artifacts\resonance-no-atr-exit\joinquant_sessions_2018_2021.json --session-calendar-sha256 24cfbdb7cfcac61c1e8a6f58bbdf54f851031ad63a7feb2ac331590ab7ede87f --output D:\test\select_stocks\.artifacts\resonance-no-atr-exit\candidate-report.json
```

Expected: 分析器接受 730 个训练组合 session，报告 baseline/candidate/double-friction，且不读取
行情或 2022 数据。任何重复日期、build 不符、截断日志或 manifest 不符都必须 fail closed。

- [ ] **Step 5: 执行候选专属反截尾门槛**

```powershell
$candidateReport = Get-Content -LiteralPath D:\test\select_stocks\.artifacts\resonance-no-atr-exit\candidate-report.json -Raw | ConvertFrom-Json
if ($candidateReport.candidate.closed_trade_count -lt 80) { throw 'NO_ATR_REJECTED: closed trades below 80' }
if ($candidateReport.candidate.open_position_count -gt 2) { throw 'NO_ATR_REJECTED: ending open positions above baseline 2' }
$candidateReport.gates
```

Expected: 只有 `closed_trade_count >= 80`、`open_position_count <= 2` 且
`gates.all_passed=True` 才能申请进入验证期。命令抛错表示候选被拒绝，不表示分析器或策略代码
需要调整。

- [ ] **Step 6: 审计 ATR 覆盖和卖出原因**

```powershell
rg -n '"atr_exit_policy": "OBSERVE_ONLY"' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log
rg -n '"execution_policy": "OBSERVE_ONLY"' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log | Select-Object -First 3
rg -n '"triggered": true' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log | Select-Object -First 3
rg -n '"order_submitted": true' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log
rg -n '"exit_reason": "ATR_EXIT"' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log
rg -n '"exit_reason": "SIGNAL_EXIT"' D:\test\select_stocks\.artifacts\resonance-no-atr-exit\ordinary.log | Select-Object -First 3
```

Expected:

- build 身份、纯观测策略字段和至少一条真实 `triggered=true` 存在；
- `order_submitted=true` 与 `exit_reason="ATR_EXIT"` 两次 `rg` 均以 exit code 1 干净无匹配；
- 卖单原因只能看到 `SIGNAL_EXIT`。

- [ ] **Step 7: 记录成功或失败实验，不做结果驱动修改**

根据真实 `candidate-report.json`，用 `apply_patch` 新建
`resonance_reversal_strategy/docs/experiments/2026-08-28-atr-exit-observation-only-result.md`，写入：

- 假设和唯一代码差异；
- 基线/candidate commit、build、manifest 和评估器 SHA-256；
- 普通与双倍摩擦全部指标；
- ATR 触发数、ATR 原因卖单数和正式信号卖单数；
- 九项晋级门槛逐项布尔结果；
- 期末未平仓数量；
- 结论为“晋级申请”或“失败实验保留”；
- 下一步只能是请求验证授权，或停止该路线。

不得在同一记录中提出相邻 ATR 倍数、替代止损、BOLL 组合或阈值微调。

## Final Verification Checklist

- [ ] 候选独立起点是 `020bc36`，不是 `.1/.2/.3`。
- [ ] 每个改动 hunk 都映射到已批准的 ATR 纯观测范围。
- [ ] `observe_atr_exit_conditions` 不具备下单或退出状态写入能力。
- [ ] `SIGNAL_EXIT`、挂起重试、买入、T-1、09:35、15:30 和 ATR 计算测试通过。
- [ ] build、参数指纹和 ETF 池指纹符合预期。
- [ ] 普通/双倍摩擦各只有一次冻结规则的有效回放。
- [ ] 完整交易、期末未平仓和全部最终硬门槛均按原始日志 fail closed。
- [ ] 失败结果被保留，没有读取 2022+ 或进行结果驱动调参。
- [ ] 未修改或运行 `cross_signal` 业务代码，未修改 PTrade 或训练数据。

本计划获用户明确批准前，不得开始 Task 1 的策略或测试代码实现。
