# Resonance Reversal 最终审查集中修复报告

日期：2026-08-27

基线：`8da83407e3bdd14e6c4715191905b3ad0b71d161`

唯一 final-fix commit：本报告与代码、测试原子地进入同一个提交（`HEAD`）。Git 提交无法在自身内容中嵌入其最终 SHA；确切 SHA 在提交完成后的交付消息中记录。

## 范围契约

- 目标对象：JoinQuant 交易日历、`current_data`、日期适配器，观察事件，普通信号卖出 pending 状态，订单结果分类，决策日志投影。
- 目标处理阶段：T 日 09:35 的 T-1 信号读取与停牌卖出，T 日 15:30 的回看观察，以及下单后实际持仓同步。
- 允许变化：不再访问未来交易日；使用 lazy mapping 下标加载报价；统一官方 `datetime.date`/DatetimeIndex 日期形状；冻结停牌普通卖出；以实际持仓变化判定订单结果；压制无事件全池拒绝日志。
- 必须不变：RSI/KDJ/BOLL 公式与阈值、BOLL 必须参与的共振真值表、候选排序、仓位与现金公式、ATR 公式和优先级、PAUSED 买入回补、非 PAUSED 买入占槽、停牌卖出之外的 processed-ID 规则、15:30 零下单路径。
- 禁止传播：其他 JoinQuant 策略、PTrade、cross-signal、本地回放、外部/训练/验证数据、参数与绩效口径。

最终状态满足该契约。README、规格、计划、progress、cross-signal、PTrade、其他策略和外部数据均未修改。

## 前置材料与门禁

编辑前完整阅读：仓库 `AGENTS.md`、`strategy_spec.md`、实施计划、progress 全部 rulings、`review-final-b8a1568..8da8340.diff`、`final-fix-brief.md`，以及 `preserve-business-scope`、`preserve-control-flow-semantics`、TDD、完成前验证 Skill。最后一次代码改动后再次完整阅读两个 preserve Skill，并按其要求完成 diff/调用方/控制流复核。

高风险提醒已在实施前给出：本任务涉及未来函数边界、交易控制流和最终提交，应使用 GPT-5.5/5.6 High；主线/实盘前审查建议 Extra High。

## 六项修复与 hunk 映射

1. Critical：未来交易日访问
   - 删除 09:35 的 `get_next_trade_date`/向未来扩展日历逻辑。
   - `build_signal_snapshots` 明确接收已知 T 日 `decision_date`，T-1 最后事件的有效期直接使用 T。
   - 观察注册仅保存固定 1/3/5 session horizon，不再预取 due date。
   - 15:30 仅调用 `get_trade_days(start_date=event_date, end_date=closing_date)`，过滤并计算事件日之后、截至当天的已发生 session。
   - 跳过的 horizon 终结为 `HORIZON_MISSED`；正好到期但无有效收盘价终结为 `PRICE_UNAVAILABLE`；1/3/5 均终结后清理记录。
   - `FutureDataError` 在注册和 15:30 日历/报价路径传播。

2. Critical：lazy `current_data`
   - 新增统一 `_get_current_record`，tradability 与 execution price 只通过 `current_data[code]` 读取。
   - `None`、普通 `KeyError`/`IndexError`/`TypeError` 映射为缺失；`FutureDataError`（即使是上述异常的子类）和其他异常传播。

3. Critical：官方日期形状统一
   - `_calendar_date` 统一 `datetime.date`、Timestamp、DatetimeIndex 元素为 `datetime.date`。
   - T-1 loader、frame 筛选、事件日期/新鲜度/有效期、processed-ID pruning/marking、daily state 与日志载荷使用一致日期类型。

4. Important：停牌普通信号卖出
   - valid held sell resonance + PAUSED 时不下单、不标记 processed ID，创建/保留 `SIGNAL_EXIT` pending，并记录实际剩余持仓。
   - 下一 session 在 pending retry 阶段先重试；ATR pending 的高优先级保持不变。

5. Important：订单结果以实际持仓为准
   - 显式 PAUSED/UNKNOWN 后：达到目标为 FILLED；实际持仓改变但未到目标为 PARTIAL；实际持仓不变为 NOT_FILLED。
   - `order.filled` 不再覆盖实际持仓。

6. Minor：拒绝日志限流
   - 完整共振仍记录并注册观察。
   - active/invalidated、第三指标冲突、新鲜度冲突以及后续真实持仓/订单路径仍保留审计。
   - 空事件、无共振的 12 ETF 双方向不再产生 24 条日常拒绝日志。

除上述 hunk 与所需测试外，没有清理、重构或参数变化。

## 真实 RED 证据

所有 RED 均在对应生产修复前运行。

| Finding | RED 选择 | 基线/修复前结果 | 关键失败证据 |
|---|---|---|---|
| 1 未来日历/观察 | 5 个 09:35、注册、15:30、FutureDataError 测试 | `5 failed` | 09:35 请求到 2021-01-22；注册预取未来；`FutureDataError` 被吞；旧 observation API 无 horizons |
| 2 lazy current_data | 5 个 lazy load/missing/exception 测试 | `5 failed` | `.get` 未触发 lazy 下标加载；缺失/异常对象产生 AttributeError |
| 3 日期形状 | 3 个 date/DatetimeIndex/processed expiry 测试 | `3 failed` | DatetimeIndex 与 `datetime.date` 比较 TypeError；processed expiry 仍为字符串 |
| 4 停牌卖出 | 跨 session pending/retry 测试 | `1 failed` | PAUSED 后 `pending_exit is None` |
| 5 实际持仓权威 | 12 项订单真值表 | `2 failed, 10 passed` | `order.filled > 0` 且实际持仓不变被误判 PARTIAL；实际改变边界未由持仓独立决定 |
| 6 日志限流 | 空事件全池测试 | `1 failed` | 产生 24 条无意义 resonance rejection log |

完成前控制流审计又把 finding 2 的异常测试收紧为 `FutureDataError(TypeError)`：生产修复前为 `1 failed`，明确显示 `DID NOT RAISE`；只调整统一 accessor 的异常优先级后为 `1 passed`。

## GREEN 证据

逐项 focused GREEN：

- finding 1：`5 passed`
- finding 2：`5 passed`；额外 TypeError 子类优先级边界 `1 passed`
- finding 3：`3 passed`
- finding 4：`1 passed`
- finding 5：`12 passed`
- finding 6：`1 passed`

完整专用测试：

```text
python -m pytest tests/test_resonance_reversal_strategy.py -v
142 passed in 0.66s
```

编译检查：

```text
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
exit 0
```

静态门禁：

- AST：唯一 `get_trade_days` 调用为 `record_due_observation_outcomes`；imports 仅 `jqdata/*`、`hashlib`、`json`、`math`、`Enum`、`numpy`、`pandas`。
- future-data：`get_next_trade_date`、`get_following_trade_days` 均无匹配；专用运行测试证明完整 09:35/15:30 不请求未来且注入的 `FutureDataError` 传播。
- lazy access：`current_data.get(` 无匹配。
- cross-signal/PTrade：目标策略内无对应 import/reference；AST 隔离测试通过。
- placeholder：TODO/TBD/placeholder/NotImplementedError/空 `pass` 等无匹配。
- `git diff --check`：exit 0；仅有 Git 的 LF/CRLF 工作区提示，无 whitespace error。

## 控制流真值表复核

### Current-data accessor

| 输入 | 结果 | 核心交易语义 |
|---|---|---|
| lazy mapping 下标可加载 | 返回 record | status/price 使用同一 record |
| `None` 或普通 Key/Index/Type missing | 返回 missing | UNKNOWN / no price，不下单 |
| `FutureDataError` | 抛出 | 不降级、不吞掉未来数据门禁 |
| 其他异常 | 抛出 | 不把平台/程序错误伪装成 UNKNOWN |

### Order outcome

| PAUSED/UNKNOWN 后的实际持仓 | 结果 | 后续同步 |
|---|---|---|
| 达到买入目标或卖出后为零 | FILLED | 使用原有完成同步 |
| 改变但未到目标 | PARTIAL | 使用原有 partial/pending 同步 |
| 不变 | NOT_FILLED | 使用原有 not-filled/pending/attempt 同步 |

订单对象的 `filled` 值不参与上述分类。

### Paused signal exit

| 条件 | 订单 | processed ID | pending |
|---|---|---|---|
| valid held signal + PAUSED | 无 | 不写入 | `SIGNAL_EXIT` + actual amount |
| 下一 session 可交易且仍持有 | pending 阶段先重试 | 保持原同步规则 | 由实际成交结果更新 |
| 已存在 ATR pending | 不降级 | 不因信号改变 | 保持 `ATR_EXIT` |

### Observation

| 时点 | 日历范围 | 写入 | 交易副作用 |
|---|---|---|---|
| 事件注册/09:35 | 不查日历 | horizon 元数据 | 无 |
| 事件日 15:30 | 不需要未来 session | 未到期 | 无 |
| 后续 15:30 | event date..closing date | RECORDED/MISSED/PRICE_UNAVAILABLE | 无订单、无持仓/processed 变化 |
| 全部 horizon 终结 | 当日范围内 | 清理 observation record | 无交易副作用 |

## 调用方与非目标路径复核

- `do_trading` -> `build_signal_snapshots(T-1, params, T)` -> `build_signal_snapshot` -> `collect_latest_events`：只替换日期/日历适配，不改指标、事件检测、共振资格和执行顺序。
- `collect_complete_resonance_decisions` -> `try_register_observation_event`：完整共振观察保留；观察错误隔离仍只限普通错误，`FutureDataError` 传播。
- `after_close` -> `record_due_observation_outcomes`：只读观察投影；原最高收盘锚更新与 portfolio summary 保留，测试确认零订单。
- `get_tradability`、`get_execution_price` -> `_get_current_record`：两个调用方共享同一 lazy accessor 和异常契约。
- `run_signal_exits` 的 PAUSED 分支只写当前持仓的 pending state；非 PAUSED processed-ID/submit 路径未改。
- `classify_order_outcome` 的所有买卖调用方继续进入原 state-sync 方法，仅分类事实来源改为实际持仓。
- 新增/变化的提前返回、异常、条件和状态写入均已逐项映射；未引入布尔/模式参数，未触及事务、Session、缓存、锁或资源清理。

代表性非目标运行验证（均包含在 142 passed 中）：默认参数与 ETF 池冻结、完整共振真值表、稳定排序、仓位/现金计算、ATR clamp 与 ATR pending 优先级、T-1 loader、PAUSED 买入回补、UNKNOWN/非 PAUSED 买入占槽、processed-ID 生命周期、完整共振观察、held/paused 观察、logger failure 隔离、15:30 零订单、KDJ observation-only、无 cross-signal dependency。

## 差异与状态

提交前产品差异仅包含：

- `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- `tests/test_resonance_reversal_strategy.py`
- 本报告 `.superpowers/sdd/2026-08-27-resonance-reversal-strategy/final-fix-report.md`

README 未修改。规格、计划、progress、review package 与 final-fix brief 仅只读。

## 未运行验证与剩余边界

- 未运行 JoinQuant 托管冒烟、托管回测或收益验证；不得据此声明托管收益或平台实盘已验证。
- JoinQuant hosted smoke 仍为后续必需验证，重点确认平台真实 lazy current-data、`FutureDataError`、官方日历返回形状和两次定时回调。
- 未读取或修改训练、验证、市场数据，也未进入验证期。
