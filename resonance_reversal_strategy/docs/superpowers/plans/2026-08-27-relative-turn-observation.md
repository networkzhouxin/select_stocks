# 非极值相对拐点观察路径 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变任何正式买卖、ATR、排序和订单行为的前提下，为 RSI14、KDJ(9,3,3) 与 BOLL(20,2) 增加独立的非极值相对拐点观察路径，并提供只读的 2019—2021 聚宽日志分析器。

**Architecture:** 相对事件从同一份截止 T-1 的指标帧生成，但使用独立 `relative_event_book`，只能进入 `run_relative_observation_stage` 旁路；正式 `event_book` 继续独占所有交易函数。相对候选使用 `RELATIVE:` 命名空间复用现有 1/3/5 日收盘观察队列，离线分析器只读取用户导出的 `.3` 基线日志和 `.4` 候选日志，按预注册门槛生成报告，不读取行情数据、不搜索参数。

**Tech Stack:** Python 3、JoinQuant/JQData API、pandas、numpy、pytest、标准库 `enum/hashlib/json/html/re/statistics/argparse/pathlib`

**Spec:** [`../specs/2026-08-27-relative-turn-observation-design.md`](../specs/2026-08-27-relative-turn-observation-design.md)

## Global Constraints

### 范围契约

| 项目 | 本计划冻结边界 |
|---|---|
| Target object | RSI14/KDJ/BOLL 相对事件、相对观察共振、相对观察结果和只读训练日志分析 |
| Target processing stage | T 日 09:35 使用截止 T-1 的日线生成旁路观察；T 日及后续 15:30 记录已到期结果；离线读取 2019—2021 日志 |
| Allowed behavior change | 新增独立观察诊断、观察记录、观察指纹、分析器；部署标识升为 `20260827.4` |
| Must remain unchanged | `resonance-v0.1.0`、正式阈值事件、正式共振真值表、ATR、排序、持仓、订单、ETF 池、成本和自适应资金 |
| Must not propagate to | `event_book`、`processed_resonance_ids`、`sold_today`、`daily_attempted_buys`、持仓状态、交易函数、PTrade、其他策略、行情数据目录和验证期 |

每个改动块必须映射到批准规格。若实现需要调整阈值、两日窗口、ETF 池、ATR 参数、正式候选、订单或 2022 年以后数据访问，立即停止并重新取得用户批准。

### 数据与防过拟合不变量

- 相对 RSI、KDJ、BOLL 只使用完整日线 `a=i-2`、`b=i-1`、`c=i`；T 日 09:35 的 `c` 最晚为 T-1。
- `load_signal_price_frame` 继续固定 `end_date=prev_date`、日频、前复权、跳过停牌；`avoid_future_data=True` 保持开启。
- 相对事件窗口固定为事件日和下一交易日；至少一个支持事件发生在 T-1。
- 分支 A 只允许正式 BOLL 加相对 RSI/KDJ；分支 B 必须相对 BOLL、RSI、KDJ 三项齐全。
- 任一有效正式或相对反向事件都否决候选；已有正式完整共振时不得登记相对候选。
- 不新增评分、权重、布尔开关、模式参数、滚动分位、参数搜索或第二套补救定义。
- 训练分析器只接受 2019—2021 观察与订单记录；出现 2022 年及以后相关记录必须拒绝。
- 聚宽仍是收益与订单路径权威。本地测试不解释收益，不打开验证期。

### 控制流契约

相对路径不新增控制参数，只新增职责明确的函数入口：

| 调用方 | 被调用入口 | 相对观察 | 正式交易 | 异常语义 |
|---|---|---:|---:|---|
| `build_signal_snapshot` | `collect_latest_relative_events` | 构造独立事件簿 | 正式快照照常构造 | 数据错误沿现有快照路径传播 |
| `do_trading` | `run_relative_observation_stage` | 注册旁路候选 | 随后仍执行正式卖出和买入 | 普通观察异常记录后继续；`FutureDataError` 重新抛出 |
| `after_close` | `record_due_observation_outcomes` | 记录相对 1/3/5 日结果 | 不下单 | 普通日志异常隔离；未来数据异常重新抛出 |
| 离线命令行 | `analyze_records` | 生成冻结门槛报告 | 不调用策略订单函数 | 数据污染、重复或合同不一致写入失败门槛；验证期记录直接拒绝 |

禁止在 `run_atr_exits`、`collect_complete_resonance_decisions`、`collect_buy_decisions`、`sort_buy_decisions`、`run_signal_exits`、`run_signal_buys`、`submit_buy` 或 `submit_sell` 中引用相对事件、相对候选或相对结果。

### 指纹与版本回归常量

当前 `20260827.3` 基线已经实测：

| 合同 | 冻结值 |
|---|---|
| 参数指纹 | `e1227fbd8b4a884e` |
| ETF 池指纹 | `9123995edeb1ed84` |
| 正式事件逻辑指纹 | `1c0b8a22f48c97c3` |
| 业务配置指纹 | `88fdf95966ea0368` |

`DEPLOYMENT_BUILD_ID` 可以升为 `20260827.4`，但正式事件逻辑合同必须使用独立常量 `FORMAL_EVENT_LOGIC_BUILD_ID = "20260827.3"`，从而保持正式事件指纹不变。相对规则只能进入新增的 `relative_observation_fingerprint()`。

### 文件结构

| 操作 | 文件 | 单一职责 |
|---|---|---|
| Modify | `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py` | 聚宽策略与观察旁路 |
| Modify | `tests/test_resonance_reversal_strategy.py` | 策略内相对谓词、事件簿、共振、隔离和平台契约测试 |
| Create | `resonance_reversal_strategy/research/analyze_relative_turn_observations.py` | 只读解析 `.3/.4` 聚宽导出日志并计算预注册门槛 |
| Create | `tests/test_resonance_relative_turn_analysis.py` | 日志污染、去重、统计、路径对齐和 CLI 测试 |
| Modify | `resonance_reversal_strategy/README.md` | 部署、日志导出和分析命令 |
| Modify | `resonance_reversal_strategy/docs/strategy_spec.md` | 记录观察扩展但不改首版交易规则 |
| Preserve | `resonance_reversal_strategy/docs/superpowers/specs/2026-08-27-relative-turn-observation-design.md` | 已批准冻结规格 |

---

### Task 1: 实现三类非极值相对拐点纯谓词

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:1026-1087`
- Test: `tests/test_resonance_reversal_strategy.py:270-307`

**Interfaces:**

- Consumes: 三个只读 pandas 行对象 `older`、`middle`、`current`。
- Produces: `_boll_percent_b(row) -> float | None`、`detect_relative_rsi_direction(older, middle, current) -> TurnDirection`、`detect_relative_kdj_direction(older, middle, current) -> TurnDirection`、`detect_relative_boll_direction(older, middle, current) -> TurnDirection`。
- Boundary: 不修改 `detect_rsi_direction`、`detect_kdj_direction`、`detect_boll_direction` 或 `TRADE_INDICATOR_COLUMNS`。

- [ ] **Step 1: 写相对 RSI/KDJ/BOLL 的失败测试**

在现有正式检测器测试之后增加：

```python
@pytest.mark.parametrize(
    "values,expected",
    [
        ((45.0, 40.0, 41.0), strategy.TurnDirection.BUY_TURN),
        ((55.0, 60.0, 59.0), strategy.TurnDirection.SELL_TURN),
        ((40.0, 40.0, 40.0), strategy.TurnDirection.NEUTRAL),
        ((np.nan, 40.0, 41.0), strategy.TurnDirection.NEUTRAL),
    ],
)
def test_relative_rsi_uses_local_turn_without_fixed_threshold(values, expected):
    older, middle, current = ({"rsi14": value} for value in values)
    assert strategy.detect_relative_rsi_direction(
        older, middle, current,
    ) is expected


@pytest.mark.parametrize(
    "older,middle,current,expected",
    [
        (
            {"j": 45.0, "kd_diff": -1.0},
            {"j": 40.0, "kd_diff": -2.0},
            {"j": 41.0, "kd_diff": -1.5},
            strategy.TurnDirection.BUY_TURN,
        ),
        (
            {"j": 55.0, "kd_diff": 2.0},
            {"j": 60.0, "kd_diff": 3.0},
            {"j": 59.0, "kd_diff": 2.5},
            strategy.TurnDirection.SELL_TURN,
        ),
        (
            {"j": 45.0, "kd_diff": -1.0},
            {"j": 40.0, "kd_diff": -2.0},
            {"j": 41.0, "kd_diff": -2.5},
            strategy.TurnDirection.NEUTRAL,
        ),
    ],
)
def test_relative_kdj_requires_j_and_kd_diff_to_turn_together(
        older, middle, current, expected):
    assert strategy.detect_relative_kdj_direction(
        older, middle, current,
    ) is expected


def test_relative_boll_uses_percent_b_turn_without_touching_band():
    older = {
        "close": 9.6, "low": 9.4, "high": 9.8,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    middle = {
        "close": 9.0, "low": 8.8, "high": 9.3,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    current = {
        "close": 9.2, "low": 8.8, "high": 9.5,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }

    assert middle["low"] > middle["boll_lower"]
    assert strategy.detect_relative_boll_direction(
        older, middle, current,
    ) is strategy.TurnDirection.BUY_TURN


def test_relative_boll_rejects_zero_width_nonfinite_and_new_extreme():
    valid = {
        "close": 9.0, "low": 8.8, "high": 9.3,
        "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0,
    }
    zero_width = dict(valid, boll_upper=8.0, boll_lower=8.0)
    nonfinite = dict(valid, close=np.inf)
    lower_low = dict(valid, close=9.2, low=8.7)

    assert strategy._boll_percent_b(zero_width) is None
    assert strategy._boll_percent_b(nonfinite) is None
    assert strategy.detect_relative_boll_direction(
        dict(valid, close=9.6), valid, lower_low,
    ) is strategy.TurnDirection.NEUTRAL
```

- [ ] **Step 2: 运行目标测试并确认因相对函数尚不存在而失败**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_rsi or relative_kdj or relative_boll" -v
```

Expected: `FAIL`，错误为 `AttributeError`，指向新增相对函数；现有正式指标测试不得失败。

- [ ] **Step 3: 写最小纯谓词实现**

在正式三个检测器之后加入：

```python
def _finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _boll_percent_b(row):
    close = _finite_float(row.get("close"))
    upper = _finite_float(row.get("boll_upper"))
    lower = _finite_float(row.get("boll_lower"))
    if close is None or upper is None or lower is None or upper <= lower:
        return None
    return (close - lower) / (upper - lower)


def detect_relative_rsi_direction(older, middle, current):
    values = tuple(_finite_float(row.get("rsi14"))
                   for row in (older, middle, current))
    if _builtins.any(value is None for value in values):
        return TurnDirection.NEUTRAL
    old_rsi, mid_rsi, current_rsi = values
    if old_rsi >= mid_rsi and current_rsi > mid_rsi:
        return TurnDirection.BUY_TURN
    if old_rsi <= mid_rsi and current_rsi < mid_rsi:
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_relative_kdj_direction(older, middle, current):
    rows = (older, middle, current)
    j_values = tuple(_finite_float(row.get("j")) for row in rows)
    diff_values = tuple(_finite_float(row.get("kd_diff")) for row in rows)
    if _builtins.any(value is None for value in j_values + diff_values):
        return TurnDirection.NEUTRAL
    old_j, mid_j, current_j = j_values
    old_diff, mid_diff, current_diff = diff_values
    if (old_j >= mid_j and old_diff >= mid_diff
            and current_j > mid_j and current_diff > mid_diff):
        return TurnDirection.BUY_TURN
    if (old_j <= mid_j and old_diff <= mid_diff
            and current_j < mid_j and current_diff < mid_diff):
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_relative_boll_direction(older, middle, current):
    percent_b = tuple(_boll_percent_b(row)
                      for row in (older, middle, current))
    fields = tuple(
        _finite_float(row.get(name))
        for row in (middle, current)
        for name in ("close", "low", "high", "boll_mid")
    )
    if (_builtins.any(value is None for value in percent_b)
            or _builtins.any(value is None for value in fields)):
        return TurnDirection.NEUTRAL
    old_percent, mid_percent, current_percent = percent_b
    mid_close, mid_low, mid_high, mid_mid = fields[:4]
    current_close, current_low, current_high, current_mid = fields[4:]
    if (mid_close < mid_mid and old_percent >= mid_percent
            and current_percent > mid_percent
            and current_close > mid_close and current_low >= mid_low):
        return TurnDirection.BUY_TURN
    if (mid_close > mid_mid and old_percent <= mid_percent
            and current_percent < mid_percent
            and current_close < mid_close and current_high <= mid_high):
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL
```

`current_mid` 只参与完整性验证，不新增“当前价必须在中轨哪一侧”的隐藏条件；位置条件只检查局部极值日 `b` 的 `mid_close` 与 `mid_mid`。

- [ ] **Step 4: 运行相对谓词和正式指标回归**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_ or rsi_event or kdj_buy_turn or boll_touch" -v
```

Expected: `PASS`；相对例子均未进入 30/70、20/80 或上下轨，正式检测器测试仍保持原结果。

- [ ] **Step 5: 提交纯谓词里程碑**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add relative turn predicates"
```

---

### Task 2: 建立独立相对事件簿与两交易日生命周期

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:1305-1436`
- Test: `tests/test_resonance_reversal_strategy.py:309-387`

**Interfaces:**

- Consumes: Task 1 的三个相对检测器、现有 `empty_event_book/apply_event/expire_events/invalidate_event`。
- Produces: `make_relative_turn_event(indicator, direction, event_date, expires_date, trigger_values, reference_extreme=None) -> dict`、`invalidate_relative_boll_structure(book, latest_row) -> dict | None`、`collect_latest_relative_events(indicator_frame, signal_date, decision_date) -> dict`。
- Boundary: 正式 `collect_latest_events` 和正式 BOLL “轨外新极值”失效规则保持原样；相对 BOLL 使用不附加轨道条件的新低/新高失效。

- [ ] **Step 1: 写独立事件簿、窗口和失效规则的失败测试**

增加测试辅助函数和用例：

```python
def relative_indicator_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.to_datetime([
            "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
        ][:len(rows)]),
    )


def test_relative_event_book_detects_t2_and_t1_from_last_four_complete_bars():
    rows = [
        {"rsi14": 50.0, "j": 50.0, "kd_diff": 0.0,
         "close": 9.8, "low": 9.6, "high": 10.0,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 45.0, "j": 45.0, "kd_diff": -1.0,
         "close": 9.4, "low": 9.2, "high": 9.7,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 46.0, "j": 40.0, "kd_diff": -2.0,
         "close": 9.0, "low": 8.8, "high": 9.3,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
        {"rsi14": 47.0, "j": 41.0, "kd_diff": -1.5,
         "close": 9.2, "low": 8.8, "high": 9.5,
         "boll_mid": 10.0, "boll_upper": 12.0, "boll_lower": 8.0},
    ]
    frame = relative_indicator_frame(rows)

    book = strategy.collect_latest_relative_events(
        frame, date(2021, 1, 8), date(2021, 1, 11),
    )

    assert book is not strategy.empty_event_book()
    assert book["active"]["RSI"]["event_date"] == date(2021, 1, 7)
    assert book["active"]["KDJ"]["event_date"] == date(2021, 1, 8)
    assert book["active"]["BOLL"]["event_date"] == date(2021, 1, 8)
    assert all(
        event["event_mode"] == "RELATIVE"
        for event in book["active"].values()
    )
    assert book["active"]["BOLL"]["reference_extreme"] == pytest.approx(8.8)


def test_relative_opposite_event_replaces_only_relative_book():
    relative_book = strategy.empty_event_book()
    hard_book = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-07",
    )
    strategy.apply_event(relative_book, strategy.make_relative_turn_event(
        "RSI", strategy.TurnDirection.BUY_TURN,
        "2021-01-07", "2021-01-08", {"fixture": "buy"},
    ))
    strategy.apply_event(relative_book, strategy.make_relative_turn_event(
        "RSI", strategy.TurnDirection.SELL_TURN,
        "2021-01-08", "2021-01-11", {"fixture": "sell"},
    ))

    assert relative_book["active"]["RSI"]["direction"] is (
        strategy.TurnDirection.SELL_TURN
    )
    assert relative_book["invalidated"][-1]["invalid_reason"] == (
        "REPLACED_BY_OPPOSITE_EVENT"
    )
    assert hard_book["active"]["RSI"]["direction"] is (
        strategy.TurnDirection.BUY_TURN
    )


def test_relative_boll_invalidates_on_new_extreme_without_band_requirement():
    book = strategy.empty_event_book()
    strategy.apply_event(book, strategy.make_relative_turn_event(
        "BOLL", strategy.TurnDirection.BUY_TURN,
        "2021-01-07", "2021-01-08", {"fixture": True},
        reference_extreme=8.8,
    ))

    strategy.invalidate_relative_boll_structure(
        book, {"low": 8.7, "high": 9.5},
    )

    assert "BOLL" not in book["active"]
    assert book["invalidated"][-1]["invalid_reason"] == (
        "NEW_LOWER_LOW_AFTER_RELATIVE_TURN"
    )
```

- [ ] **Step 2: 运行事件簿测试并确认缺少新接口**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_event_book or relative_opposite_event or relative_boll_invalidates" -v
```

Expected: `FAIL`，只因 `make_relative_turn_event`、`collect_latest_relative_events` 和相对 BOLL 失效函数未定义。

- [ ] **Step 3: 实现相对事件对象与相对 BOLL 失效**

```python
def make_relative_turn_event(indicator, direction, event_date, expires_date,
                             trigger_values, reference_extreme=None):
    event = make_turn_event(
        indicator, direction, event_date, expires_date,
        trigger_values, reference_extreme,
    )
    event["event_mode"] = "RELATIVE"
    return event


def invalidate_relative_boll_structure(book, latest_row):
    event = book["active"].get("BOLL")
    if event is None:
        return None
    reference = _finite_float(event.get("reference_extreme"))
    low = _finite_float(latest_row.get("low"))
    high = _finite_float(latest_row.get("high"))
    if reference is None:
        return invalidate_event(book, "BOLL", "INVALID_RELATIVE_EXTREME")
    if (event["direction"] is TurnDirection.BUY_TURN
            and low is not None and low < reference):
        return invalidate_event(
            book, "BOLL", "NEW_LOWER_LOW_AFTER_RELATIVE_TURN",
        )
    if (event["direction"] is TurnDirection.SELL_TURN
            and high is not None and high > reference):
        return invalidate_event(
            book, "BOLL", "NEW_HIGHER_HIGH_AFTER_RELATIVE_TURN",
        )
    return None
```

- [ ] **Step 4: 实现最后四根日线的相对事件收集**

```python
def _relative_trigger_values(indicator, older, middle, current):
    fields_by_indicator = {
        "RSI": ("rsi14",),
        "KDJ": ("j", "kd_diff"),
        "BOLL": (
            "low", "high", "close", "boll_mid",
            "boll_lower", "boll_upper",
        ),
    }
    fields = fields_by_indicator[indicator]
    return {
        "older": {name: older.get(name) for name in fields},
        "middle": {name: middle.get(name) for name in fields},
        "current": {name: current.get(name) for name in fields},
    }


def _make_detected_relative_event(indicator, direction, older, middle,
                                  current, event_date, expires_date):
    reference_extreme = None
    if indicator == "BOLL":
        reference_extreme = (
            middle.get("low")
            if direction is TurnDirection.BUY_TURN
            else middle.get("high")
        )
    return make_relative_turn_event(
        indicator, direction, event_date, expires_date,
        _relative_trigger_values(indicator, older, middle, current),
        reference_extreme,
    )


def collect_latest_relative_events(indicator_frame, signal_date,
                                   decision_date):
    signal_date = _calendar_date(signal_date)
    decision_date = _calendar_date(decision_date)
    frame_dates = tuple(
        _calendar_date(index_value) for index_value in indicator_frame.index
    )
    complete_frame = indicator_frame.loc[[
        frame_date is not None and frame_date <= signal_date
        for frame_date in frame_dates
    ]]
    book = empty_event_book()
    first_event_position = max(2, len(complete_frame) - 2)
    for position in range(first_event_position, len(complete_frame)):
        older = complete_frame.iloc[position - 2]
        middle = complete_frame.iloc[position - 1]
        current = complete_frame.iloc[position]
        event_date = _calendar_date(complete_frame.index[position])
        expires_date = (
            _calendar_date(complete_frame.index[position + 1])
            if position + 1 < len(complete_frame) else decision_date
        )
        expire_events(book, event_date)
        directions = {
            "BOLL": detect_relative_boll_direction(older, middle, current),
            "RSI": detect_relative_rsi_direction(older, middle, current),
            "KDJ": detect_relative_kdj_direction(older, middle, current),
        }
        for indicator in INDICATORS:
            direction = directions[indicator]
            if direction is not TurnDirection.NEUTRAL:
                apply_event(book, _make_detected_relative_event(
                    indicator, direction, older, middle, current,
                    event_date, expires_date,
                ))
        invalidate_relative_boll_structure(book, current)
    expire_events(book, signal_date)
    return book
```

- [ ] **Step 5: 运行相对生命周期及正式事件回归**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_event_book or relative_opposite_event or relative_boll_invalidates or collect_latest_events or expired_event or opposite_event or boll_lower_band or boll_upper_band" -v
```

Expected: `PASS`；硬事件与相对事件的活动/失效列表互不改写。

- [ ] **Step 6: 提交相对事件簿里程碑**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add relative event book"
```

---

### Task 3: 实现两个观察共振分支、稳定 ID 与独立指纹

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:1285-1302,1444-1583`
- Test: `tests/test_resonance_reversal_strategy.py:388-539,3298-3324`

**Interfaces:**

- Consumes: 正式 `event_book`、Task 2 的 `relative_event_book`、`build_resonance_decision`。
- Produces: `build_relative_observation_id(code, direction, branch, support_events) -> str`、`build_relative_resonance_observation(code, direction, hard_book, relative_book, signal_date, event_close) -> dict | None`、`collect_relative_resonance_observations(snapshots) -> list[dict]`、`relative_observation_logic_contract() -> dict`、`relative_observation_fingerprint() -> str`。
- Boundary: 相对候选是观察对象，不返回 `support_count`、`boll_age` 或任何交易排序字段。

- [ ] **Step 1: 写分支 A、分支 B、反向否决和正式去重的失败测试**

```python
def relative_event_book_for_directions(boll, rsi, kdj, event_date):
    active = {}
    for indicator, direction in (("BOLL", boll), ("RSI", rsi), ("KDJ", kdj)):
        enum_direction = strategy.TurnDirection[direction]
        if enum_direction is strategy.TurnDirection.NEUTRAL:
            continue
        active[indicator] = strategy.make_relative_turn_event(
            indicator, enum_direction, event_date, event_date,
            {"fixture": indicator},
            reference_extreme=(8.8 if indicator == "BOLL" else None),
        )
    return {"active": active, "invalidated": []}


def test_relative_branch_a_requires_hard_boll_and_relative_oscillator():
    hard = event_book_for_directions(
        "BUY_TURN", "NEUTRAL", "NEUTRAL", "2021-01-08",
    )
    relative = relative_event_book_for_directions(
        "NEUTRAL", "BUY_TURN", "NEUTRAL", "2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["branch"] == "HARD_BOLL_SOFT_OSC"
    assert observation["supporters"] == ("BOLL", "RSI")
    assert observation["hard_or_relative_source_by_indicator"] == {
        "BOLL": "HARD", "RSI": "RELATIVE",
    }
    assert observation["relative_observation_id"].startswith("RELATIVE:")


def test_relative_branch_b_requires_all_three_relative_indicators():
    hard = strategy.empty_event_book()
    relative = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", "2021-01-08",
    )

    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    )

    assert observation["branch"] == "SOFT_ALL_THREE"
    assert observation["supporters"] == ("BOLL", "KDJ", "RSI")


@pytest.mark.parametrize(
    "hard_directions,relative_directions",
    [
        (("BUY_TURN", "SELL_TURN", "NEUTRAL"),
         ("NEUTRAL", "BUY_TURN", "NEUTRAL")),
        (("NEUTRAL", "NEUTRAL", "NEUTRAL"),
         ("BUY_TURN", "BUY_TURN", "SELL_TURN")),
        (("NEUTRAL", "NEUTRAL", "NEUTRAL"),
         ("BUY_TURN", "BUY_TURN", "NEUTRAL")),
    ],
)
def test_relative_candidate_rejects_opposite_or_incomplete_support(
        hard_directions, relative_directions):
    hard = event_book_for_directions(
        *hard_directions, event_date="2021-01-08",
    )
    relative = relative_event_book_for_directions(
        *relative_directions, event_date="2021-01-08",
    )
    assert strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    ) is None


def test_existing_complete_hard_resonance_suppresses_relative_candidate():
    hard = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", "2021-01-08",
    )
    relative = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", "2021-01-08",
    )
    assert strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, date(2021, 1, 8), 10.0,
    ) is None
```

- [ ] **Step 2: 写指纹隔离失败测试**

```python
def test_relative_fingerprint_is_deterministic_and_formal_fingerprints_are_frozen():
    params = strategy.get_default_params()
    self_check = strategy.run_event_logic_self_check(params)

    assert strategy._value_fingerprint(params) == "e1227fbd8b4a884e"
    assert strategy._value_fingerprint(
        strategy.get_default_etf_pool(),
    ) == "9123995edeb1ed84"
    assert strategy.business_config_fingerprint(
        params, strategy.get_default_etf_pool(),
    ) == "88fdf95966ea0368"
    assert strategy.event_logic_fingerprint(
        params, self_check,
    ) == "1c0b8a22f48c97c3"
    first = strategy.relative_observation_fingerprint()
    second = strategy.relative_observation_fingerprint()
    assert first == second
    assert len(first) == 16
    json.dumps(strategy.relative_observation_logic_contract(), sort_keys=True)
```

- [ ] **Step 3: 运行目标测试并确认共振和指纹接口尚未建立**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_branch or relative_candidate or suppresses_relative or relative_fingerprint" -v
```

Expected: `FAIL`，失败只来自新增构建器和指纹接口。

- [ ] **Step 4: 冻结正式逻辑合同并定义相对观察指纹**

在版本常量旁增加 `FORMAL_EVENT_LOGIC_BUILD_ID = "20260827.3"`，并把 `event_logic_fingerprint` 合同内原有的 `"build": DEPLOYMENT_BUILD_ID` 改为 `"build": FORMAL_EVENT_LOGIC_BUILD_ID`。新增：

```python
def relative_observation_logic_contract():
    return {
        "event_mode": "RELATIVE",
        "window_sessions": 2,
        "fresh_supporter_required": True,
        "opposite_veto": "ANY_ACTIVE_HARD_OR_RELATIVE_OPPOSITE",
        "relative_predicates": {
            "RSI": "LOCAL_RSI14_TURN_A_B_C",
            "KDJ": "LOCAL_J_AND_KD_DIFF_TURN_A_B_C",
            "BOLL": "LOCAL_PERCENT_B_TURN_WITH_MID_AND_PRICE_STRUCTURE",
        },
        "branches": {
            "HARD_BOLL_SOFT_OSC": [
                "HARD_BOLL", "RELATIVE_RSI_OR_KDJ",
            ],
            "SOFT_ALL_THREE": [
                "RELATIVE_BOLL", "RELATIVE_RSI", "RELATIVE_KDJ",
            ],
        },
        "deduplication": "EXCLUDE_COMPLETE_HARD_RESONANCE",
        "boll_invalidation": "NEW_EXTREME_AFTER_RELATIVE_TURN",
    }


def relative_observation_fingerprint():
    return _value_fingerprint(relative_observation_logic_contract())
```

该合同只包含 JSON 基本类型，不允许生成器、函数对象或平台对象进入 `_value_fingerprint`。

- [ ] **Step 5: 实现稳定 ID 与两个纯观察分支**

```python
def _event_mode(event):
    return event.get("event_mode", "HARD")


def _has_active_opposite(hard_book, relative_book, direction):
    return _builtins.any(
        event is not None and event["direction"] is OPPOSITE[direction]
        for book in (hard_book, relative_book)
        for event in (
            book["active"].get("BOLL"),
            book["active"].get("RSI"),
            book["active"].get("KDJ"),
        )
    )


def build_relative_observation_id(code, direction, branch, support_events):
    parts = ["RELATIVE", branch, direction.value, code]
    for event in sorted(support_events, key=lambda item: item["indicator"]):
        parts.append("%s:%s:%s" % (
            event["indicator"], _event_mode(event),
            _calendar_date(event["event_date"]),
        ))
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return "RELATIVE:" + digest


def build_relative_resonance_observation(
        code, direction, hard_book, relative_book, signal_date, event_close):
    signal_date = _calendar_date(signal_date)
    if build_resonance_decision(
            code, direction, hard_book, signal_date) is not None:
        return None
    if _has_active_opposite(hard_book, relative_book, direction):
        return None

    hard_boll = hard_book["active"].get("BOLL")
    relative_active = relative_book["active"]
    relative_boll = relative_active.get("BOLL")
    relative_rsi = relative_active.get("RSI")
    relative_kdj = relative_active.get("KDJ")
    relative_oscillators = tuple(
        event for event in (relative_rsi, relative_kdj)
        if event is not None and event["direction"] is direction
    )

    if hard_boll is not None and hard_boll["direction"] is direction:
        if not relative_oscillators:
            return None
        branch = "HARD_BOLL_SOFT_OSC"
        support_events = (hard_boll,) + relative_oscillators
    elif _builtins.all(
            event is not None and event["direction"] is direction
            for event in (relative_boll, relative_rsi, relative_kdj)):
        branch = "SOFT_ALL_THREE"
        support_events = (relative_boll, relative_rsi, relative_kdj)
    else:
        return None

    if not _builtins.any(
            _calendar_date(event["event_date"]) == signal_date
            for event in support_events):
        return None
    ordered_events = tuple(sorted(
        support_events, key=lambda item: item["indicator"],
    ))
    source_map = {
        event["indicator"]: _event_mode(event) for event in ordered_events
    }
    date_map = {
        event["indicator"]: _calendar_date(event["event_date"])
        for event in ordered_events
    }
    return {
        "relative_observation_id": build_relative_observation_id(
            code, direction, branch, ordered_events,
        ),
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": branch,
        "code": code,
        "direction": direction,
        "signal_date": signal_date,
        "supporters": tuple(sorted(source_map)),
        "supporter_event_dates": date_map,
        "hard_or_relative_source_by_indicator": source_map,
        "expires_date": min(
            _calendar_date(event["expires_date"])
            for event in ordered_events
        ),
        "event_close": float(event_close),
    }


def collect_relative_resonance_observations(snapshots):
    observations = []
    for code in sorted(snapshots):
        snapshot = snapshots[code]
        if not snapshot.get("valid"):
            continue
        relative_book = (
            snapshot.get("relative_event_book") or empty_event_book()
        )
        for direction in (
                TurnDirection.BUY_TURN, TurnDirection.SELL_TURN):
            observation = build_relative_resonance_observation(
                code, direction,
                snapshot["event_book"], relative_book,
                snapshot["signal_date"], snapshot["close"],
            )
            if observation is not None:
                observations.append(observation)
    return observations
```

- [ ] **Step 6: 运行共振真值表、ID、正式指纹和既有正式共振回归**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_branch or relative_candidate or suppresses_relative or relative_fingerprint or complete_resonance_truth_table or two_old_events or candidate_sort or event_logic_fingerprint" -v
```

Expected: `PASS`；正式事件逻辑指纹仍严格等于 `1c0b8a22f48c97c3`。

- [ ] **Step 7: 提交观察共振纯逻辑里程碑**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): build relative observations"
```

---

### Task 4: 接入运行时旁路、命名空间观察结果与诊断日志

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py:11-12,167-192,261-570,629-671`
- Test: `tests/test_resonance_reversal_strategy.py:54-137,976-1003,1974-2106,2617-2674,3435-3561`

**Interfaces:**

- Consumes: Task 3 的 `collect_relative_resonance_observations` 和 `relative_observation_fingerprint`。
- Produces: `make_relative_observation_event(observation) -> dict`、`register_relative_observation_event(observation) -> bool`、`try_register_relative_observation_event(observation) -> bool`、`run_relative_observation_stage(snapshots) -> None`。
- Side effects: 只向 `g.observation_events` 写入 `RELATIVE:` 键并输出结构化日志；不返回或传递交易资格。

将既有 `test_diagnostic_build_id_is_bumped` 的期望值从 `20260827.3` 同步改为 `20260827.4`；这是部署标识变更的唯一既有断言调整，不能借此改动任何正式规则期望。

- [ ] **Step 1: 写快照、初始化 build 和指纹日志失败测试**

```python
def test_relative_observation_build_and_formal_fingerprints_are_separated(
        monkeypatch):
    messages = []
    _install_initialize_platform_stubs(monkeypatch, messages, [])
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    payload = json.loads(messages[-1])
    assert strategy.DEPLOYMENT_BUILD_ID == "20260827.4"
    assert payload["build"] == "20260827.4"
    assert payload["parameter_fingerprint"] == "e1227fbd8b4a884e"
    assert payload["pool_fingerprint"] == "9123995edeb1ed84"
    assert payload["event_logic_fingerprint"] == "1c0b8a22f48c97c3"
    assert payload["relative_observation_fingerprint"] == (
        strategy.relative_observation_fingerprint()
    )


def test_signal_snapshot_builds_separate_relative_event_book(monkeypatch):
    frame = make_ohlcv_frame(120)
    relative_book = relative_event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "BUY_TURN", frame.index[-1].date(),
    )
    monkeypatch.setattr(
        strategy, "load_signal_price_frame", lambda *args: frame, raising=False,
    )
    monkeypatch.setattr(
        strategy, "collect_latest_relative_events",
        lambda *args: relative_book, raising=False,
    )

    snapshot = strategy.build_signal_snapshot(
        "510300.XSHG", frame.index[-1], strategy.get_default_params(),
        frame.index[-1] + pd.offsets.BDay(1),
    )

    assert snapshot["relative_event_book"] is relative_book
    assert snapshot["event_book"] is not relative_book
```

- [ ] **Step 2: 写旁路控制流和异常语义失败测试**

```python
def test_do_trading_runs_relative_stage_without_skipping_formal_pipeline(
        monkeypatch):
    calls = []
    monkeypatch.setattr(strategy, "g", runtime_state(), raising=False)
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    monkeypatch.setattr(
        strategy, "retry_pending_exits", lambda *args: calls.append("retry"),
    )
    monkeypatch.setattr(
        strategy, "run_atr_exits", lambda *args: calls.append("atr"),
    )
    monkeypatch.setattr(
        strategy, "build_signal_snapshots",
        lambda *args: calls.append("snapshots") or {},
    )
    monkeypatch.setattr(
        strategy, "run_relative_observation_stage",
        lambda snapshots: calls.append("relative"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_exits", lambda *args: calls.append("exits"),
    )
    monkeypatch.setattr(
        strategy, "run_signal_buys", lambda *args: calls.append("buys"),
    )

    strategy.do_trading(fake_context())

    assert calls == ["retry", "atr", "snapshots", "relative", "exits", "buys"]


def test_relative_stage_isolates_ordinary_error_but_propagates_future_error(
        monkeypatch):
    logs = []
    monkeypatch.setattr(
        strategy, "_emit_structured_log",
        lambda event, payload: logs.append((event, payload)),
    )
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations",
        lambda snapshots: (_ for _ in ()).throw(RuntimeError("ordinary")),
    )
    assert strategy.run_relative_observation_stage({}) is None
    assert logs[-1][0] == "relative_observation_pipeline"

    class FutureDataError(RuntimeError):
        pass

    expected = FutureDataError("future")
    monkeypatch.setattr(
        strategy, "collect_relative_resonance_observations",
        lambda snapshots: (_ for _ in ()).throw(expected),
    )
    with pytest.raises(FutureDataError) as raised:
        strategy.run_relative_observation_stage({})
    assert raised.value is expected


def test_relative_registration_isolates_ordinary_error_and_rethrows_future(
        monkeypatch):
    observation = {
        "relative_observation_id": "RELATIVE:fixture",
        "code": "510300.XSHG",
    }
    monkeypatch.setattr(
        strategy, "register_relative_observation_event",
        lambda value: (_ for _ in ()).throw(RuntimeError("ordinary")),
    )
    assert strategy.try_register_relative_observation_event(observation) is False

    class FutureDataError(RuntimeError):
        pass

    expected = FutureDataError("future registration")
    monkeypatch.setattr(
        strategy, "register_relative_observation_event",
        lambda value: (_ for _ in ()).throw(expected),
    )
    with pytest.raises(FutureDataError) as raised:
        strategy.try_register_relative_observation_event(observation)
    assert raised.value is expected
```

- [ ] **Step 3: 写相对结果方向调整与交易隔离失败测试**

```python
def test_relative_outcome_adds_direction_adjusted_return_without_orders(
        monkeypatch):
    observation = {
        "relative_observation_id": "RELATIVE:fixture",
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": "SOFT_ALL_THREE",
        "code": "510300.XSHG",
        "direction": strategy.TurnDirection.SELL_TURN,
        "signal_date": date(2021, 1, 5),
        "supporters": ("BOLL", "KDJ", "RSI"),
        "supporter_event_dates": {
            "BOLL": date(2021, 1, 5),
            "KDJ": date(2021, 1, 5),
            "RSI": date(2021, 1, 5),
        },
        "hard_or_relative_source_by_indicator": {
            "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
        },
        "expires_date": date(2021, 1, 6),
        "event_close": 10.0,
    }
    runtime = runtime_state()
    monkeypatch.setattr(strategy, "g", runtime, raising=False)
    monkeypatch.setattr(
        strategy, "get_trade_days",
        lambda **kwargs: [date(2021, 1, 5), date(2021, 1, 6)],
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target", lambda *args: pytest.fail("no sell order"),
        raising=False,
    )
    monkeypatch.setattr(
        strategy, "order_target_value",
        lambda *args: pytest.fail("no buy order"), raising=False,
    )
    strategy.register_relative_observation_event(observation)

    strategy.record_due_observation_outcomes(
        fake_context(current_date="2021-01-06"),
        {"510300.XSHG": current_record(price=9.0)},
    )

    outcome = runtime.observation_events[
        "RELATIVE:fixture"
    ]["outcomes"][1]
    assert outcome["return"] == pytest.approx(-0.1)
    assert outcome["direction_adjusted_return"] == pytest.approx(0.1)


def test_trading_functions_have_no_relative_observation_dependency():
    forbidden = {
        "relative_event_book", "relative_observation_id",
        "relative_observation", "relative_resonance",
    }
    for function in (
        strategy.run_atr_exits,
        strategy.collect_complete_resonance_decisions,
        strategy.collect_buy_decisions,
        strategy.sort_buy_decisions,
        strategy.run_signal_exits,
        strategy.run_signal_buys,
        strategy.submit_buy,
        strategy.submit_sell,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        strings = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert forbidden.isdisjoint(names | strings), function.__name__
```

同时在测试文件导入区增加 `import textwrap`。

- [ ] **Step 4: 运行新增运行时测试并确认接口或字段缺失**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_observation_build or separate_relative_event_book or runs_relative_stage or relative_stage_isolates or relative_registration or relative_outcome or no_relative_observation_dependency" -v
```

Expected: `FAIL`，指向 `.4` build、快照字段和相对注册/旁路接口。

- [ ] **Step 5: 接入快照、初始化与信号诊断字段**

执行以下最小修改：

```python
DEPLOYMENT_BUILD_ID = "20260827.4"
```

在 `build_signal_snapshot` 的正式 `event_book` 之后独立计算：

```python
relative_event_book = collect_latest_relative_events(
    indicators, signal_date, decision_date,
)
```

并只在返回快照中增加：

```python
"relative_event_book": relative_event_book,
```

`log_signal_snapshot` 增加 `relative_active_events`、`relative_invalidated_events` 和 `relative_observation_fingerprint`；`initialize` 的 `strategy_initialized` 只增加：

```python
"relative_observation_fingerprint": relative_observation_fingerprint(),
```

参数指纹、池指纹和正式事件指纹的既有字段名及计算入口保持不变。

- [ ] **Step 6: 实现命名空间观察记录和注册旁路**

```python
def make_relative_observation_event(observation, horizons=(1, 3, 5)):
    observation_id = observation["relative_observation_id"]
    record = make_observation_event(
        observation_id, observation["code"], observation["signal_date"],
        observation["event_close"], horizons,
    )
    record.update({
        "relative_observation_id": observation_id,
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": observation["branch"],
        "direction": observation["direction"],
        "supporters": tuple(observation["supporters"]),
        "supporter_event_dates": dict(observation["supporter_event_dates"]),
        "hard_or_relative_source_by_indicator": dict(
            observation["hard_or_relative_source_by_indicator"]
        ),
        "expires_date": _calendar_date(observation["expires_date"]),
        "build": DEPLOYMENT_BUILD_ID,
        "relative_observation_fingerprint": (
            relative_observation_fingerprint()
        ),
    })
    return record


def register_relative_observation_event(observation):
    if observation is None:
        return False
    observation_id = observation["relative_observation_id"]
    if not observation_id.startswith("RELATIVE:"):
        raise ValueError("relative observation id must use RELATIVE namespace")
    if observation_id in g.observation_events:
        return False
    g.observation_events[observation_id] = make_relative_observation_event(
        observation,
    )
    return True


def try_register_relative_observation_event(observation):
    try:
        return register_relative_observation_event(observation)
    except Exception as error:
        if _is_future_data_error(error):
            raise
        _emit_structured_log("relative_observation_registration", {
            "relative_observation_id": (
                observation.get("relative_observation_id")
                if observation is not None else None
            ),
            "code": observation.get("code") if observation is not None else None,
            "reason": "RELATIVE_OBSERVATION_REGISTRATION_FAILED",
            "error_type": type(error).__name__,
        })
        return False


def log_relative_resonance_observation(observation):
    _emit_structured_log("relative_resonance_observation", {
        "version": STRATEGY_VERSION,
        "build": DEPLOYMENT_BUILD_ID,
        "parameter_fingerprint": _value_fingerprint(g.params),
        "pool_fingerprint": _value_fingerprint(g.etf_pool),
        "event_logic_fingerprint": event_logic_fingerprint(g.params),
        "relative_observation_fingerprint": (
            relative_observation_fingerprint()
        ),
        "relative_observation_id": observation["relative_observation_id"],
        "observation_kind": observation["observation_kind"],
        "branch": observation["branch"],
        "code": observation["code"],
        "direction": observation["direction"],
        "signal_date": observation["signal_date"],
        "supporters": observation["supporters"],
        "supporter_event_dates": observation["supporter_event_dates"],
        "hard_or_relative_source_by_indicator": (
            observation["hard_or_relative_source_by_indicator"]
        ),
        "expires_date": observation["expires_date"],
        "event_close": observation["event_close"],
    })


def run_relative_observation_stage(snapshots):
    try:
        observations = collect_relative_resonance_observations(snapshots)
    except Exception as error:
        if _is_future_data_error(error):
            raise
        _emit_structured_log("relative_observation_pipeline", {
            "reason": "RELATIVE_OBSERVATION_COLLECTION_FAILED",
            "error_type": type(error).__name__,
        })
        return None
    for observation in observations:
        if try_register_relative_observation_event(observation):
            log_relative_resonance_observation(observation)
    return None
```

在 `do_trading` 的快照诊断循环之后、`run_signal_exits` 之前插入单独一行：

```python
run_relative_observation_stage(snapshots)
```

不得把返回值用于分支、提前返回、排序或订单。

快照诊断循环的日志条件只增加对独立相对事件簿的检查：

```python
relative_book = snapshot.get("relative_event_book") or {}
```

当 `relative_book["active"]` 或 `relative_book["invalidated"]` 非空时允许调用 `log_signal_snapshot`；该条件只决定是否输出诊断日志，不得包围或跳过 `run_relative_observation_stage`、正式卖出或正式买入。

- [ ] **Step 7: 扩展相对结果日志但保持正式结果负载不变**

在 `record_due_observation_outcomes` 创建 `RECORDED` 结果后，仅当 `record.get("observation_kind") == "RELATIVE_RESONANCE"` 时增加：

```python
direction = record["direction"]
direction_value = (
    direction.value if isinstance(direction, TurnDirection) else direction
)
raw_return = outcome["return"]
outcome["direction_adjusted_return"] = (
    raw_return if direction_value == TurnDirection.BUY_TURN.value
    else -raw_return
)
```

相对 `observation_outcome` 日志在现有字段上增加 `relative_observation_id`、`observation_kind`、`branch`、`direction`、`supporters`、`build` 和 `relative_observation_fingerprint`。正式记录没有 `observation_kind` 时不得增加这些相对字段，避免改变 `.3` 正式观察日志合同。

- [ ] **Step 8: 运行运行时、观察迁移、未来函数和订单隔离回归**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "relative_ or observation or do_trading or after_close or future or order or atr or resonance or fingerprint" -v
```

Expected: `PASS`；既有 legacy 观察记录仍能归一化、清理并记录；相对路径存在、为空或普通失败时正式调用顺序一致。

- [ ] **Step 9: 提交运行时旁路里程碑**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): record relative observations"
```

---

### Task 5: 创建只读训练日志分析器与预注册门槛

**Files:**

- Create: `resonance_reversal_strategy/research/analyze_relative_turn_observations.py`
- Create: `tests/test_resonance_relative_turn_analysis.py`

**Interfaces:**

- Consumes: 用户导出的 `20260827.3` 基线聚宽日志和 `20260827.4` 候选聚宽日志。
- Produces: `parse_joinquant_log_line(line, ordinal) -> dict | None`、`load_log_records(paths) -> list[dict]`、`analyze_records(candidate_records, baseline_records) -> dict`、`main(argv=None) -> int`。
- CLI: `--candidate-log`、`--baseline-log` 可重复；`--output` 为输出 JSON 文件。不得暴露阈值、窗口、年份或预期基线数值覆盖参数。

- [ ] **Step 1: 写日志解析、验证期拒绝和路径对齐失败测试**

新测试文件采用独立模块导入：

```python
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_relative_turn_observations.py"
)
spec = importlib.util.spec_from_file_location("relative_analyzer", ANALYZER_PATH)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def test_parser_accepts_plain_and_html_escaped_joinquant_json():
    plain = (
        '2021-01-05 09:35:00 - INFO - '
        '{"event":"relative_resonance_observation","signal_date":"2021-01-04"}'
    )
    escaped = plain.replace('"', '&quot;')
    first = analyzer.parse_joinquant_log_line(plain, 1)
    second = analyzer.parse_joinquant_log_line(escaped, 2)
    assert first["event"] == second["event"]
    assert first["_log_date"] == "2021-01-05"
    assert second["_ordinal"] == 2


def test_analyzer_rejects_validation_period_observation():
    candidate = [{
        "event": "relative_resonance_observation",
        "relative_observation_id": "RELATIVE:2022",
        "signal_date": "2022-01-04",
        "build": "20260827.4",
        "relative_observation_fingerprint": "fixture",
    }]
    with pytest.raises(ValueError, match="2022"):
        analyzer.analyze_records(candidate, [])


def test_filled_order_path_requires_exact_date_side_code_and_amounts():
    baseline = [{
        "event": "order_transition", "_log_date": "2021-01-05",
        "_ordinal": 1, "side": "BUY", "code": "510300.XSHG",
        "outcome": "FILLED", "before_amount": 0, "after_amount": 100,
    }]
    changed = [dict(baseline[0], after_amount=200)]
    assert analyzer.extract_filled_order_path(baseline) != (
        analyzer.extract_filled_order_path(changed)
    )
```

- [ ] **Step 2: 写通过全部预注册门槛的完整合成日志测试**

```python
def make_order_path(build_date="2021-01-05"):
    return [
        {
            "event": "order_transition", "_log_date": build_date,
            "_ordinal": index, "side": "BUY" if index % 2 else "SELL",
            "code": "510300.XSHG", "outcome": "FILLED",
            "before_amount": 0 if index % 2 else 100,
            "after_amount": 100 if index % 2 else 0,
        }
        for index in range(1, 139)
    ]


def make_relative_records():
    records = [{
        "event": "strategy_initialized", "build": "20260827.4",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": "relative-fixture",
    }]
    codes = ("510300.XSHG", "159915.XSHE", "518880.XSHG")
    for index in range(30):
        year = 2019 + index // 10
        observation_id = "RELATIVE:%02d" % index
        signal_date = "%04d-01-%02d" % (year, index % 10 + 2)
        branch = (
            "HARD_BOLL_SOFT_OSC" if index % 2
            else "SOFT_ALL_THREE"
        )
        records.append({
            "event": "relative_resonance_observation",
            "relative_observation_id": observation_id,
            "observation_kind": "RELATIVE_RESONANCE",
            "branch": branch,
            "code": codes[index % len(codes)],
            "direction": "BUY_TURN",
            "signal_date": signal_date,
            "expires_date": signal_date,
            "supporters": ["BOLL", "RSI"],
            "build": "20260827.4",
            "relative_observation_fingerprint": "relative-fixture",
        })
        for horizon, value in ((1, 0.005), (3, 0.01), (5, 0.02)):
            records.append({
                "event": "observation_outcome",
                "relative_observation_id": observation_id,
                "observation_kind": "RELATIVE_RESONANCE",
                "branch": branch,
                "direction": "BUY_TURN",
                "code": codes[index % len(codes)],
                "event_date": signal_date,
                "horizon": horizon,
                "build": "20260827.4",
                "relative_observation_fingerprint": "relative-fixture",
                "outcome": {
                    "status": "RECORDED",
                    "closing_date": signal_date,
                    "return": value,
                    "direction_adjusted_return": value,
                },
            })
    records.extend(make_order_path())
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40,
    })
    return records


def make_baseline_records():
    records = make_order_path()
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40,
    })
    for index in range(30):
        resonance_id = "FORMAL:%02d" % index
        records.append({
            "event": "resonance_decision", "accepted": True,
            "reason": "COMPLETE_RESONANCE", "resonance_id": resonance_id,
            "code": "510300.XSHG", "direction": "BUY_TURN",
            "signal_date": "2021-01-05",
        })
        records.append({
            "event": "observation_outcome", "resonance_id": resonance_id,
            "code": "510300.XSHG", "event_date": "2021-01-05",
            "horizon": 5,
            "outcome": {"status": "RECORDED", "return": 0.01},
        })
    return records


def test_frozen_report_passes_only_when_every_gate_and_path_match():
    report = analyzer.analyze_records(
        make_relative_records(), make_baseline_records(),
    )
    assert report["metrics"]["candidate_count"] == 30
    assert report["metrics"]["year_counts"] == {
        "2019": 10, "2020": 10, "2021": 10,
    }
    assert report["metrics"]["direction_counts"] == {
        "BUY_TURN": 30, "SELL_TURN": 0,
    }
    assert report["metrics"]["etf_counts"] == {
        "159915.XSHE": 10, "510300.XSHG": 10, "518880.XSHG": 10,
    }
    assert report["metrics"]["horizon_5"]["median"] == pytest.approx(0.02)
    assert report["metrics"]["horizon_5"]["hit_rate"] == pytest.approx(1.0)
    assert all(report["gates"].values())
    assert report["continue_candidate"] is True
```

- [ ] **Step 3: 运行分析器测试并确认文件尚不存在**

Run:

```powershell
python -m pytest tests/test_resonance_relative_turn_analysis.py -v
```

Expected: `FAIL`，因为分析器模块尚未创建。

- [ ] **Step 4: 实现只读日志解析与训练窗口拒绝**

分析器顶部固定常量和解析接口：

```python
import argparse
import html
import json
import math
import pathlib
import re
import statistics
from datetime import date


TRAIN_START = date(2019, 1, 1)
TRAIN_END = date(2021, 12, 31)
CANDIDATE_BUILD = "20260827.4"
BASELINE_FILLED_COUNT = 138
BASELINE_FINAL_ASSET = 23856.40
PARAMETER_FINGERPRINT = "e1227fbd8b4a884e"
POOL_FINGERPRINT = "9123995edeb1ed84"
FORMAL_EVENT_FINGERPRINT = "1c0b8a22f48c97c3"
LOG_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_joinquant_log_line(line, ordinal):
    text = html.unescape(line.strip())
    payload_start = text.find("{")
    if payload_start < 0:
        return None
    try:
        payload = json.loads(text[payload_start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    match = LOG_DATE_RE.match(text)
    payload = dict(payload)
    payload["_log_date"] = match.group(1) if match else None
    payload["_ordinal"] = int(ordinal)
    return payload


def load_log_records(paths):
    records = []
    ordinal = 0
    for path_value in paths:
        path = pathlib.Path(path_value)
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                ordinal += 1
                record = parse_joinquant_log_line(line, ordinal)
                if record is not None:
                    records.append(record)
    return records


def _calendar_date(value):
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value)[:10])


def reject_nontraining_observations(records):
    relevant_events = {
        "relative_resonance_observation", "observation_outcome",
    }
    for record in records:
        if record.get("event") not in relevant_events:
            continue
        if (record.get("observation_kind") != "RELATIVE_RESONANCE"
                and not record.get("relative_observation_id")):
            continue
        outcome = record.get("outcome") or {}
        observed_dates = (
            record.get("signal_date"), record.get("event_date"),
            outcome.get("closing_date") if isinstance(outcome, dict) else None,
        )
        normalized_dates = tuple(
            _calendar_date(value) for value in observed_dates
            if value not in (None, "")
        )
        if not normalized_dates:
            raise ValueError("relative observation has no training date")
        for observed_date in normalized_dates:
            if not (TRAIN_START <= observed_date <= TRAIN_END):
                raise ValueError(
                    "relative observation outside 2019-2021: %s"
                    % observed_date
                )
```

- [ ] **Step 5: 实现确定性统计、正式对照和订单路径**

```python
def lower_quartile(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * 0.25
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_returns(values):
    values = tuple(float(value) for value in values)
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "hit_rate": None, "q1": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "hit_rate": sum(value > 0 for value in values) / len(values),
        "q1": lower_quartile(values),
    }


def extract_filled_order_path(records):
    path = []
    for record in records:
        if (record.get("event") == "order_transition"
                and record.get("outcome") == "FILLED"):
            path.append((
                record.get("_log_date"), record.get("side"),
                record.get("code"), record.get("before_amount"),
                record.get("after_amount"),
            ))
    return tuple(path)


def extract_final_asset(records):
    summaries = [
        record for record in records
        if record.get("event") == "portfolio_summary"
        and record.get("total_value") is not None
    ]
    if not summaries:
        return None
    return float(summaries[-1]["total_value"])


def _formal_five_day_returns(records):
    directions = {
        record["resonance_id"]: record.get("direction")
        for record in records
        if record.get("event") == "resonance_decision"
        and record.get("accepted") is True
        and record.get("reason") == "COMPLETE_RESONANCE"
        and record.get("resonance_id")
    }
    values = []
    for record in records:
        if (record.get("event") != "observation_outcome"
                or record.get("horizon") != 5):
            continue
        resonance_id = record.get("resonance_id")
        outcome = record.get("outcome") or {}
        if (resonance_id not in directions
                or outcome.get("status") != "RECORDED"
                or outcome.get("return") is None):
            continue
        raw_return = float(outcome["return"])
        values.append(
            raw_return if directions[resonance_id] == "BUY_TURN"
            else -raw_return
        )
    return values
```

- [ ] **Step 6: 实现预注册报告，失败项只能报告不能调参**

`analyze_records(candidate_records, baseline_records)` 必须：

1. 调用 `reject_nontraining_observations`；
2. 验证所有候选 build 为 `.4` 且相对指纹唯一；
3. 验证初始化参数/池/正式事件指纹等于冻结值；
4. 按 `relative_observation_id` 拒绝重复候选和重复 horizon；
5. 对每个候选要求 1/3/5 日 `RECORDED` 结果；
6. 统计总集和两个分支；
7. 比较正式 5 日方向调整收益 Q1；
8. 比较 `.3/.4` 的 138 条 `FILLED` 路径与期末资产。

使用以下确定性骨架；所有门槛值只出现在这里，CLI 不允许覆盖：

```python
def _initialization_errors(records):
    expected = {
        "build": CANDIDATE_BUILD,
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
    }
    initialized = [
        record for record in records
        if record.get("event") == "strategy_initialized"
    ]
    if not initialized:
        return ["missing strategy_initialized record"]
    errors = []
    for record in initialized:
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                errors.append("%s mismatch: %r" % (field, record.get(field)))
    return errors


def _safe_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def analyze_records(candidate_records, baseline_records):
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    reject_nontraining_observations(candidate_records)
    errors = _initialization_errors(candidate_records)

    candidates = []
    candidate_by_id = {}
    for record in candidate_records:
        if record.get("event") != "relative_resonance_observation":
            continue
        observation_id = record.get("relative_observation_id")
        if not isinstance(observation_id, str) or not observation_id.startswith(
                "RELATIVE:"):
            errors.append("invalid relative observation id: %r" % observation_id)
            continue
        if observation_id in candidate_by_id:
            errors.append("duplicate relative candidate: %s" % observation_id)
            continue
        if record.get("build") != CANDIDATE_BUILD:
            errors.append("candidate build mismatch: %s" % observation_id)
        if not record.get("relative_observation_fingerprint"):
            errors.append("candidate fingerprint missing: %s" % observation_id)
        if record.get("branch") not in (
                "HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE"):
            errors.append("invalid branch: %s" % observation_id)
        signal_date = _calendar_date(record.get("signal_date"))
        expires_date = _calendar_date(record.get("expires_date"))
        if signal_date is None or expires_date is None:
            errors.append("candidate date missing: %s" % observation_id)
        if (signal_date is not None and expires_date is not None
                and expires_date < signal_date):
            errors.append("expired candidate: %s" % observation_id)
        candidate_by_id[observation_id] = record
        candidates.append(record)

    fingerprints = {
        record.get("relative_observation_fingerprint")
        for record in candidate_records
        if record.get("event") in {
            "strategy_initialized", "relative_resonance_observation",
            "observation_outcome",
        }
        and (record.get("event") != "observation_outcome"
             or record.get("relative_observation_id"))
        and record.get("relative_observation_fingerprint")
    }
    if len(fingerprints) != 1:
        errors.append("relative fingerprint count is %d" % len(fingerprints))
    relative_fingerprint = (
        next(iter(fingerprints)) if len(fingerprints) == 1 else None
    )

    outcomes = {}
    for record in candidate_records:
        observation_id = record.get("relative_observation_id")
        if (record.get("event") != "observation_outcome"
                or not observation_id):
            continue
        try:
            horizon = int(record.get("horizon"))
        except (TypeError, ValueError):
            errors.append("invalid relative horizon: %s" % observation_id)
            continue
        key = (observation_id, horizon)
        if key in outcomes:
            errors.append("duplicate relative outcome: %s/%s" % key)
            continue
        if record.get("build") != CANDIDATE_BUILD:
            errors.append("outcome build mismatch: %s/%s" % key)
        if not record.get("relative_observation_fingerprint"):
            errors.append("outcome fingerprint missing: %s/%s" % key)
        outcomes[key] = record

    formal_keys = {
        (
            record.get("code"), record.get("direction"),
            str(record.get("signal_date"))[:10],
        )
        for record in candidate_records
        if record.get("event") == "resonance_decision"
        and record.get("accepted") is True
        and record.get("reason") == "COMPLETE_RESONANCE"
    }
    relative_keys = {
        (
            record.get("code"), record.get("direction"),
            str(record.get("signal_date"))[:10],
        )
        for record in candidates
    }
    formal_overlap_count = len(formal_keys & relative_keys)

    year_counts = {"2019": 0, "2020": 0, "2021": 0}
    direction_counts = {"BUY_TURN": 0, "SELL_TURN": 0}
    etf_counts = {}
    returns_by_horizon = {1: [], 3: [], 5: []}
    five_day_by_branch = {
        "HARD_BOLL_SOFT_OSC": [], "SOFT_ALL_THREE": [],
    }
    five_day_2021 = []
    positive_by_etf = {}
    missing_outcome_count = 0
    for candidate in candidates:
        observation_id = candidate["relative_observation_id"]
        signal_date = _calendar_date(candidate.get("signal_date"))
        if signal_date is not None and str(signal_date.year) in year_counts:
            year_counts[str(signal_date.year)] += 1
        direction = candidate.get("direction")
        if direction in direction_counts:
            direction_counts[direction] += 1
        else:
            errors.append("invalid candidate direction: %s" % observation_id)
        code = candidate.get("code")
        if not code:
            errors.append("candidate code missing: %s" % observation_id)
        else:
            etf_counts[code] = etf_counts.get(code, 0) + 1
        for horizon in (1, 3, 5):
            record = outcomes.get((observation_id, horizon))
            outcome = record.get("outcome") if record is not None else None
            value = _safe_number(
                outcome.get("direction_adjusted_return")
                if isinstance(outcome, dict) else None
            )
            if (not isinstance(outcome, dict)
                    or outcome.get("status") != "RECORDED"
                    or value is None):
                missing_outcome_count += 1
                continue
            returns_by_horizon[horizon].append(value)
            if horizon == 5:
                five_day_by_branch[candidate["branch"]].append(value)
                if signal_date is not None and signal_date.year == 2021:
                    five_day_2021.append(value)
                if value > 0:
                    code = candidate.get("code")
                    positive_by_etf[code] = positive_by_etf.get(code, 0.0) + value

    total_positive = sum(positive_by_etf.values())
    max_positive_contribution = (
        max(positive_by_etf.values()) / total_positive
        if total_positive > 0 else None
    )
    formal_five_day = _formal_five_day_returns(baseline_records)
    horizon_5 = summarize_returns(returns_by_horizon[5])
    year_2021 = summarize_returns(five_day_2021)
    formal_horizon_5 = summarize_returns(formal_five_day)

    candidate_path = extract_filled_order_path(candidate_records)
    baseline_path = extract_filled_order_path(baseline_records)
    candidate_asset = extract_final_asset(candidate_records)
    baseline_asset = extract_final_asset(baseline_records)
    formal_order_path_exact = (
        len(candidate_path) == BASELINE_FILLED_COUNT
        and len(baseline_path) == BASELINE_FILLED_COUNT
        and candidate_path == baseline_path
    )
    final_asset_exact = (
        candidate_asset is not None and baseline_asset is not None
        and math.isclose(candidate_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
        and math.isclose(baseline_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
        and math.isclose(candidate_asset, baseline_asset, abs_tol=0.01)
    )
    data_quality_complete = (
        not errors and formal_overlap_count == 0
        and missing_outcome_count == 0
    )

    gates = {
        "candidate_count_at_least_30": len(candidates) >= 30,
        "each_training_year_at_least_5": all(
            year_counts[str(year)] >= 5 for year in (2019, 2020, 2021)
        ),
        "horizon_5_median_positive": (
            horizon_5["median"] is not None and horizon_5["median"] > 0
        ),
        "horizon_5_hit_rate_above_half": (
            horizon_5["hit_rate"] is not None
            and horizon_5["hit_rate"] > 0.5
        ),
        "year_2021_median_nonnegative": (
            year_2021["median"] is not None and year_2021["median"] >= 0
        ),
        "horizon_5_q1_not_worse_than_formal": (
            horizon_5["q1"] is not None
            and formal_horizon_5["q1"] is not None
            and horizon_5["q1"] >= formal_horizon_5["q1"]
        ),
        "single_etf_positive_contribution_at_most_half": (
            max_positive_contribution is not None
            and max_positive_contribution <= 0.5
        ),
        "formal_order_path_exact": formal_order_path_exact,
        "final_asset_exact": final_asset_exact,
        "data_quality_complete": data_quality_complete,
    }
    return {
        "data_quality": {
            "errors": errors,
            "relative_fingerprint": relative_fingerprint,
            "formal_overlap_count": formal_overlap_count,
            "missing_outcome_count": missing_outcome_count,
        },
        "metrics": {
            "candidate_count": len(candidates),
            "year_counts": year_counts,
            "direction_counts": direction_counts,
            "etf_counts": dict(sorted(etf_counts.items())),
            "by_branch": {
                branch: summarize_returns(values)
                for branch, values in five_day_by_branch.items()
            },
            "horizon_1": summarize_returns(returns_by_horizon[1]),
            "horizon_3": summarize_returns(returns_by_horizon[3]),
            "horizon_5": horizon_5,
            "year_2021_horizon_5": year_2021,
            "formal_horizon_5": formal_horizon_5,
            "max_positive_contribution_by_etf": max_positive_contribution,
            "filled_path_count": len(candidate_path),
            "final_asset": candidate_asset,
        },
        "gates": gates,
        "continue_candidate": all(gates.values()),
    }
```

分析器不得输出自动交易、阈值建议、窗口建议或 ETF 删除建议。失败门槛只保留为 `False` 与数据质量证据。

- [ ] **Step 7: 实现固定 CLI**

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-log", action="append", required=True)
    parser.add_argument("--baseline-log", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = analyze_records(
        load_log_records(args.candidate_log),
        load_log_records(args.baseline_log),
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

输出目录由用户指定，禁止默认写入任何训练、预热、底层指数或验证行情数据目录。

- [ ] **Step 8: 运行分析器测试并验证没有行情或策略下单依赖**

Run:

```powershell
python -m pytest tests/test_resonance_relative_turn_analysis.py -v
python -m py_compile resonance_reversal_strategy/research/analyze_relative_turn_observations.py
rg -n "get_price|get_current_data|order_target|G:\\financial|2022-|threshold_search|optimize" resonance_reversal_strategy/research/analyze_relative_turn_observations.py
```

Expected: pytest 与编译 `PASS`；`rg` 对禁止依赖返回退出码 1 且无匹配。退出码 1 在此处表示静态检查通过。

- [ ] **Step 9: 提交只读分析器里程碑**

```powershell
git add resonance_reversal_strategy/research/analyze_relative_turn_observations.py tests/test_resonance_relative_turn_analysis.py
git commit -m "feat(resonance): analyze relative observation logs"
```

---

### Task 6: 更新说明、执行完整回归并形成聚宽交付点

**Files:**

- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`
- Verify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Verify: `tests/test_resonance_reversal_strategy.py`
- Verify: `tests/test_resonance_relative_turn_analysis.py`

**Interfaces:**

- Consumes: Tasks 1—5 的冻结接口和测试证据。
- Produces: 可复制到聚宽的 `.4` 策略、日志分析命令、本地验证摘要和可回滚提交。
- Boundary: 本任务不运行 2022 年以后回测、不解释收益、不合并主分支、不推送。

- [ ] **Step 1: 更新 README 的观察扩展与运行命令**

在 README 明确写入：

```markdown
## 非极值相对拐点观察（build 20260827.4）

该路径只记录未形成正式完整共振的增量候选：

- `HARD_BOLL_SOFT_OSC`：正式 BOLL 加相对 RSI 或 KDJ；
- `SOFT_ALL_THREE`：相对 BOLL、RSI、KDJ 三项齐全。

相对事件使用独立事件簿和 `RELATIVE:` 标识，不进入正式共振、排序、仓位、ATR
或订单。普通观察异常不会中断交易；`FutureDataError` 仍会让回测明确失败。

训练回测完成后，将 `20260827.3` 基线日志和 `20260827.4` 候选日志导出到行情
数据目录之外，再执行：

```powershell
python resonance_reversal_strategy/research/analyze_relative_turn_observations.py `
  --baseline-log D:\logs\resonance-20260827.3.log `
  --candidate-log D:\logs\resonance-20260827.4.log `
  --output D:\logs\relative-turn-report.json
```

分析器拒绝 2022 年及以后观察记录，也不会搜索阈值、窗口或 ETF。全部预注册门槛
通过只代表可以提出下一份交易候选规格，不代表可以自动下单或进入验证期。
```

- [ ] **Step 2: 在主策略规格追加观察扩展状态，不改第 7—12 节正式规则**

追加一节，链接批准规格并说明：

```markdown
## 观察扩展：非极值相对拐点

`20260827.4` 新增的相对拐点只属于研究观察层。正式 RSI/KDJ/BOLL 事件、完整
共振、ATR、买卖顺序和订单状态仍以本文原有章节为准。相对谓词、两个观察分支、
两日生命周期、日志合同和预注册继续门槛，以
[`docs/superpowers/specs/2026-08-27-relative-turn-observation-design.md`](
superpowers/specs/2026-08-27-relative-turn-observation-design.md) 为冻结依据。
```

- [ ] **Step 3: 运行编译、两套专用测试和全仓回归**

Run:

```powershell
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py resonance_reversal_strategy/research/analyze_relative_turn_observations.py
python -m pytest tests/test_resonance_reversal_strategy.py -v
python -m pytest tests/test_resonance_relative_turn_analysis.py -v
python -m pytest -q
```

Expected: 两个专用测试文件全部 `PASS`。全仓若有既有无关失败，记录准确测试名、错误和与本改动的隔离证据，不得把未通过报告为通过。

- [ ] **Step 4: 执行静态范围、控制流和未来函数门禁**

Run:

```powershell
git diff --check
rg -n "relative_event_book|relative_observation_id|relative_resonance" resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
rg -n "end_date=_calendar_date\(prev_date\)|set_option\(\"avoid_future_data\", True\)|run_relative_observation_stage\(snapshots\)" resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
git status --short
```

逐个检查相对标识的匹配位置只属于快照构建、旁路注册、日志和结果记录；交易函数中必须无匹配。确认 `do_trading` 顺序仍为：待卖重试 → ATR → T-1 快照 → 诊断 → 相对观察旁路 → 正式卖出 → 正式买入；旁路返回值未被读取。

- [ ] **Step 5: 将每个改动块映射回批准规格**

完成前记录以下对应关系：

| 改动 | 规格条款 | 代表性非目标验证 |
|---|---|---|
| 三类相对谓词 | 4.1—4.3 | 正式检测器和参数指纹不变 |
| 独立事件簿与失效 | 4.4 | 正式 `event_book` 不被修改 |
| 分支 A/B、否决、去重 | 5 | 完整正式共振仍只走正式观察与交易 |
| 命名空间记录和结果 | 6、8 | 相对对象不进入订单或持仓状态 |
| 旁路异常控制流 | 7 | 普通失败后正式卖出/买入继续；未来错误重抛 |
| 只读分析器和门槛 | 9—10 | 不读取行情、不允许验证期、不搜索参数 |
| 测试与文档 | 11—14 | 其他策略、数据目录和 PTrade 无改动 |

任何无法映射的改动必须删除，或停止并向用户重新确认。

- [ ] **Step 6: 提交文档与总验证里程碑**

```powershell
git add resonance_reversal_strategy/README.md resonance_reversal_strategy/docs/strategy_spec.md
git commit -m "docs(resonance): document relative observation workflow"
```

- [ ] **Step 7: 形成聚宽冒烟交付说明**

最终交付只要求用户下一步：将 `.4` 策略复制到聚宽，先运行短区间冒烟，核对初始化中的四个指纹、无未来函数错误、相对候选/结果日志以及正式订单路径没有异常。短区间通过后再运行 2019—2021 冻结训练回测；只有用户提供 `.3/.4` 完整日志后，才执行 Task 5 分析器并按原样报告门槛。

不得在本地验证完成时宣称聚宽订单路径、期末资产或观察收益已经通过。
