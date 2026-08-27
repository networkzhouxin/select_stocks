# Task 2 实施报告：独立相对事件簿与两交易日生命周期

## 需求映射

- 目标对象：独立的相对转折事件簿及 `RELATIVE` 事件对象。
- 目标阶段：截止 `signal_date` 的最后四根完整日线中，检测 T-2/T-1 相对 RSI、KDJ、BOLL 事件，并执行注册、相反方向替换、BOLL 相对极值失效与过期。
- 允许变化：新增 `make_relative_turn_event`、`invalidate_relative_boll_structure`、`collect_latest_relative_events` 及其必要的相对触发值辅助函数；增加 Task 2 生命周期和隔离测试。
- 必须保持：Task 1 三个相对检测器接口和语义；正式 `collect_latest_events`、正式 `g.event_book`、正式事件列、正式 BOLL 轨外新极值规则，以及候选、评分、交易、下单、持仓和冷却逻辑。
- 不得传播到：正式事件簿、正式事件收集/生命周期、任何交易决策路径；相对收集器只使用 `signal_date` 以前的完整指标行。

## RED 命令与失败

先加入三个 Task 2 测试和 `relative_indicator_frame` 辅助函数，未修改生产代码。运行：

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_event_book or relative_opposite_event or relative_boll_invalidates" -v
```

结果：`3 failed, 193 deselected`。失败均为预期的接口缺失：

- `collect_latest_relative_events` 尚不存在；
- `make_relative_turn_event` 尚不存在；
- `invalidate_relative_boll_structure` 尚不存在。

未出现测试拼写错误或正式事件测试失败。

## 实现摘要

- `make_relative_turn_event` 复用现有事件字段构造，并增加明确的 `event_mode="RELATIVE"` 标记。
- `invalidate_relative_boll_structure` 使用相对事件的参考低点/高点，不附加正式 BOLL 的轨道条件；买入事件在新低、卖出事件在新高时失效，并记录专用原因。
- `collect_latest_relative_events` 独立创建事件簿，只扫描截止 `signal_date` 的最后两个三日窗口（至少需要四根完整行来同时覆盖 T-2/T-1），按交易日顺序调用既有生命周期函数，并使用 `decision_date` 作为最后事件的过期日。
- 相反方向替换、事件过期和失效记录复用现有通用生命周期函数，但整个过程只作用于新建的相对事件簿。

## 测试命令与结果

GREEN 聚焦测试：

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_event_book or relative_opposite_event or relative_boll_invalidates" -v
```

结果：`3 passed, 193 deselected`。

相对生命周期及正式事件回归：

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_event_book or relative_opposite_event or relative_boll_invalidates or collect_latest_events or expired_event or opposite_event or boll_lower_band or boll_upper_band" -v
```

结果：`8 passed, 188 deselected`。

策略专用全量测试：

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -v
```

结果：`196 passed`。

静态/编译检查：

```powershell
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
git diff --check
```

结果：均退出码 `0`；`git diff --check` 仅有 Git 的 LF/CRLF 转换提示，无 whitespace error。

## 改动文件

- `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
  - 新增相对事件构造、相对 BOLL 失效、相对触发值组装和最后四根完整日线事件收集。
- `tests/test_resonance_reversal_strategy.py`
  - 新增相对指标帧辅助函数、T-2/T-1 收集测试、相对事件反向替换隔离测试、无轨道条件相对 BOLL 新极值失效测试。

## Commit

Task 2 实现里程碑：`616478aa52df627d8758d6fb8e6d69e205afb4f7`

提交信息：`feat(resonance): add relative event book`

## 剩余风险/关注点

- 本任务仅建立独立相对事件簿；尚未接入相对观察候选、日志或任何交易路径，后续任务必须重新审查隔离边界。
- 生命周期使用传入的 `decision_date` 作为最后扫描事件的过期日；调用层需继续提供经过交易日历确认的下一交易日，不得改用自然日推算。
- 本地验证覆盖策略专用测试、编译和 diff 检查，未执行聚宽运行时或实盘 API 验证；本任务没有访问或写入任何训练源数据目录。
