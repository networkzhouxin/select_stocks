# ATR 退出停用候选设计

## 1. 目标

建立一个从 `20260827.4` 基线独立分出的结构消融候选：ATR(14) 及其冻结入场值、
最高收盘锚、5%--15% 止损边界和 `atr_check` 观测全部保留，但 ATR 触发不再创建
挂起退出或提交卖单。候选只依赖原正式 `SIGNAL_EXIT` 完成退出。

本候选用于回答“ATR 正式退出是否拖累交易质量”。它不是删除 ATR 指标、调整 ATR
参数、加入新退出规则或组合 `.2/.3`。

## 2. 已批准依据

- 正式基线：build `20260827.4`，commit `020bc36`，2019--2021 训练期，
  初始资金 20,000 元。
- ATR 影子观察显示止损后价格经常反弹，只足以支持一次独立结构消融，不能支持调参。
- 用户已批准“只停用 ATR 退出、保留 ATR 计算”的范围。
- 训练日历继续使用 schema V2 manifest，SHA-256 为
  `24cfbdb7cfcac61c1e8a6f58bbdf54f851031ad63a7feb2ac331590ab7ede87f`。

## 3. 范围契约

| 字段 | 契约 |
|---|---|
| 目标对象 | `resonance_reversal` 候选 build 在 09:35 对实际持仓执行的 ATR 风险退出阶段 |
| 目标阶段 | 挂起退出重试之后、T-1 信号快照和正式信号退出之前 |
| 允许改变 | ATR 触发仅记录，不调用 `submit_sell`、不创建/升级 `ATR_EXIT` 挂起状态、不写入 `sold_today` |
| 必须保持不变 | ATR(14,2.5)、5%--15% 边界、入场 ATR 有效性要求、最高收盘锚、正式共振、`SIGNAL_EXIT`、买入、挂起 `SIGNAL_EXIT` 重试、ETF 池、仓位、成本、T-1 信号、09:35 执行、15:30 收盘更新 |
| 不得传播到 | `.2/.3`、`cross_signal`、PTrade、本地行情 loader、相对卖出、BOLL 退出、参数或阈值搜索、2022+ 数据、实盘 |

## 4. 实现边界

候选 build 固定为 `20260828.4`，从 `020bc36` 新建独立 worktree。不得以 `.1`、`.2`
或 `.3` 策略文件为起点。

不新增控制布尔值或运行模式参数。候选代码本身只有一种策略语义：

```text
retry_pending_exits
  -> observe_atr_exit_conditions
  -> build_signal_snapshots(T-1)
  -> run_signal_exits
  -> run_signal_buys
```

`ATR_EXIT_POLICY = "OBSERVE_ONLY"` 只作为不可变的 build 身份和日志字段，不参与分支
选择。`observe_atr_exit_conditions` 计算与原基线相同的止损价并记录是否触发，但不得持有
任何下单能力。

为保持最小改动，`ExitReason.ATR_EXIT`、`EXIT_PRIORITY` 和通用挂起退出数据结构不做清理
或重构。新鲜的候选回放不会创建 `ATR_EXIT` 挂起状态；若日志出现该状态或 ATR 原因卖单，
本次回放直接判为无效，而不是放宽审计。

## 5. 控制流真值表

| 场景 | ATR 观测 | ATR 卖单/挂起 | 正式信号退出 | 买入与同日状态 |
|---|---:|---:|---|---|
| 无实际持仓或无风险状态 | 跳过 | 无 | 原流程 | 原流程 |
| 当前价高于止损价 | `triggered=false` | 无 | 原流程 | 原流程 |
| 当前价等于或低于止损价，无正式卖出 | `triggered=true` | 无 | 无 | 持仓继续保留，可按原买入规则处理其他代码 |
| ATR 触发且同日有合法 `SIGNAL_EXIT` | `triggered=true` | 无 | 只提交一笔 `SIGNAL_EXIT` | 由原 `sold_today` 规则阻止同日回购 |
| `SIGNAL_EXIT` 挂起重试后未成交，同时 ATR 触发 | `triggered=true` | 不升级为 ATR | 当日仍由 `daily_retried_exits` 阻止第二单 | 原挂起原因保持 `SIGNAL_EXIT` |
| `SIGNAL_EXIT` 挂起重试后完全成交 | 实际持仓已清除，跳过 | 无新增 | 已由重试完成 | 原 `sold_today` 生效 |
| 快照不足或正式指标无效，同时 ATR 触发 | `triggered=true` | 无 | 无法形成正式退出 | 持仓继续保留 |

## 6. 日志合同

`strategy_initialized` 增加：

- `atr_exit_policy="OBSERVE_ONLY"`。

每条 `atr_check` 保留基线字段并增加：

- `execution_policy="OBSERVE_ONLY"`；
- `order_submitted=false`。

`order_transition` 对卖单增加 `exit_reason`，买单为 `null`。这是只读诊断字段，不改变订单
分类、状态同步或评估器的成交配对。正式候选日志必须满足：

- 至少存在一条 `triggered=true` 的 ATR 观测，证明消融真正覆盖到原 ATR 触发场景；
- 所有 ATR 观测均为 `order_submitted=false`；
- 不存在 `exit_reason="ATR_EXIT"`；
- 所有候选卖单原因只能是 `SIGNAL_EXIT`。

## 7. 训练与反截尾合同

- 只运行 2019-01-01 至 2021-12-31；2018 仅作只读指标预热和交易日证据。
- 普通摩擦和双倍摩擦各运行一次；二者策略代码、build 和信号规则完全相同。
- 不强制期末平仓，因为这会引入只在实验结束日存在的交易规则。
- 总收益和最大回撤必须来自每日组合总资产，包含未平仓持仓的市值变化。
- 普通摩擦候选完整交易数至少 80，期末未平仓数不得超过基线的 2 只；任一不满足，
  胜率不得用于宣称改善。
- 不因结果修改阈值、加入替代止损、延长训练期或组合 `.2/.3`。

## 8. 晋级门槛

候选只有同时满足以下条件才可申请后续验证：

| 指标 | 门槛 |
|---|---:|
| 普通摩擦总收益 | `> 129.25%` |
| 完整交易胜率 | `> 55.8%` |
| 95% Wilson 胜率下界 | `> 50%` |
| 最大回撤 | `< 6.28%` |
| 完整交易数 | `>= 80` |
| 交易收益中位数 | `> 0` |
| 前 10% 完整交易占毛利润 | `<= 50%` |
| 期末未平仓数 | `<= 2` |
| 双倍摩擦总收益 | `> 64.10%` |

所有门槛通过也只允许申请冻结规则后的验证，不允许直接进入实盘。任一门槛失败即保留为
失败实验，不进行相邻改动。

## 9. 文件范围

候选实施只允许修改：

- `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`；
- `tests/test_resonance_reversal_strategy.py`；
- `resonance_reversal_strategy/README.md`；
- `resonance_reversal_strategy/docs/strategy_spec.md`；
- 本设计、实施计划和最终实验记录。

共享候选评估器只读使用，不在本候选中修改。

## 10. 验收标准

- 代码静态调用链中 ATR 观测函数不能访问卖单、挂起退出或 `sold_today` 写入能力；
- ATR 参数、入场 ATR、止损价计算、T-1 和 09:35/15:30 边界保持不变；
- ATR 触发后同日合法正式信号仍能提交唯一的 `SIGNAL_EXIT`；
- 挂起正式退出重试不被 ATR 观测升级或重复提交；
- 目标测试、完整 resonance 测试、语法检查和 diff 检查通过；
- 聚宽普通/双倍摩擦日志通过固定 manifest、build、订单原因和反截尾审计；
- 没有读取或使用 2022 及以后数据。
