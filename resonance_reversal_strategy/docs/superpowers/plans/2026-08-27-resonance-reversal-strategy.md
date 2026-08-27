# RSI-KDJ-BOLL Resonance Reversal ETF Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立目录中实现一份可直接粘贴到聚宽运行的 RSI14/KDJ/BOLL 两日共振反转 ETF 策略，以 ATR14 独立控制风险，并通过本地单元测试冻结未来函数边界、业务真值表和订单状态语义。

**Architecture:** 单一聚宽策略文件同时包含纯指标函数、事件/共振决策函数、组合与订单状态函数以及平台编排入口；纯函数由本地 pytest 隔离验证，平台函数通过替换聚宽 API 做契约测试。日内入口严格按“待卖重试 → ATR → T-1 信号 → 普通卖出 → 实际仓位重算 → 买入”执行，15:30 入口只更新最高收盘锚点和回顾性观察日志。

**Tech Stack:** Python 3、JoinQuant/JQData API、pandas、numpy、pytest、标准库 `enum/json/hashlib/datetime`

**Spec:** [`strategy_spec.md`](../../strategy_spec.md)

## Global Constraints

### 范围契约

| 项目 | 本计划固定边界 |
|---|---|
| Target object | 新策略 `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py` 及其专属测试和 README |
| Target processing stage | T-1 指标 → 方向事件 → 两日共振 → T 日 09:35 订单 → T 日 15:30 收盘更新 |
| Allowed behavior change | 只新增已确认的共振策略，不修改任何现有策略 |
| Must remain unchanged | `cross_signal_strategy`、V15、多因子、PTrade、历史数据目录和现有测试全部保持原样 |
| Must not propagate to | 观察指标不得进入交易；T 日价格不得进入指标；首版不得增加回放器、参数搜索、验证期调参或 PTrade 代码 |

每个改动块必须能映射到已确认规格。若实现需要改变 ETF 池、阈值、持仓数、共振窗口、ATR 参数、成本、训练窗口或订单递补规则，立即停止并重新征求用户确认。

### 控制流契约

目标入口及唯一调用关系：

| 调用方 | 被调用入口 | 核心业务 | 必须保持的非目标行为 |
|---|---|---|---|
| 聚宽 09:35 定时任务 | `do_trading(context)` | 完整执行当日交易管线 | 某一阶段无结果不得跳过更早的风控或待卖处理 |
| `do_trading` | `retry_pending_exits` | 重试未清零卖单 | 不计算或改写指标事件 |
| `do_trading` | `run_atr_exits` | 无条件优先处理 ATR 风险 | 不受普通信号、持有期或候选排序限制 |
| `do_trading` | `build_signal_snapshots` | 只用 T-1 数据冻结信号 | 不读取 T 日价格字段 |
| `do_trading` | `run_signal_exits` | 处理顶部共振卖出 | 不处理新买入，不清理未归零持仓状态 |
| `do_trading` | `run_signal_buys` | 按固定顺序填充实际空位 | 不替换持仓、不补仓、不绕过当日禁买 |
| 聚宽 15:30 定时任务 | `after_close(context)` | 更新最高收盘锚点和观察结果 | 不重算交易信号、不提交订单 |

方向、订单和状态控制必须使用职责明确的枚举，禁止一个裸布尔值同时承担多个语义：

| 枚举 | 值 | 只允许控制的职责 | 明确不得控制 |
|---|---|---|---|
| `TurnDirection` | `BUY_TURN/SELL_TURN/NEUTRAL` | 单个指标的方向事件 | 事件是否过期、是否下单、订单结果 |
| `OrderSide` | `BUY/SELL` | 订单持仓变化方向 | 信号方向和退出原因 |
| `ExitReason` | `ATR_EXIT/SIGNAL_EXIT` | 待卖优先级和日志原因 | 指标计算、交易状态 |
| `Tradability` | `TRADEABLE/PAUSED/UNKNOWN` | 是否可提交当前订单 | 指标事件和共振资格 |
| `OrderOutcome` | `FILLED/PARTIAL/NOT_FILLED/PAUSED/UNKNOWN` | 下单后的状态同步 | 候选评分、指标事件、方向判断 |

订单结果真值表：

| 交易状态/持仓变化 | 订单结果 | 买入槽位 | 风险状态 |
|---|---|---|---|
| 明确停牌，未提交订单 | `PAUSED` | 不消耗，可递补 | 原状态不变 |
| 状态未知或报价无效 | `UNKNOWN` | 消耗，不递补 | 原状态不变 |
| 实际持仓达到目标 | `FILLED` | 消耗 | 按实际持仓建立或清理状态 |
| 实际持仓改变但未达到目标 | `PARTIAL` | 消耗 | 买入建立状态；卖出保留 `pending_exit` |
| 实际持仓未改变 | `NOT_FILLED` | 消耗 | 卖出保留 `pending_exit`；买入不建仓态 |

### 固定业务不变量

- 指标日线只允许 `end_date=prev_date`、`frequency="daily"`、`fq="pre"`、`skip_paused=True`。
- `set_option("avoid_future_data", True)` 必须开启，且不得捕获并吞掉未来数据异常。
- 交易指标只有 RSI14、KDJ(9,3,3)、BOLL(20,2)；BOLL 必须参与，RSI14/KDJ 至少一项确认。
- RSI6/12/24、ADX/+DI/-DI、量能、BOLL 带宽和中轨斜率只记录，不得传入共振、排序、仓位和订单资格函数。
- 共振窗口固定为两个交易日，且至少一个支持事件发生于 T-1；第三指标出现有效反向事件即否决。
- 最大持仓 3，只为新仓计算 `min(total_value*0.95/3, max(0, cash-total_value*0.05))`，不强制再平衡已有持仓。
- ATR14 在入场时冻结；最高价锚点只用收盘价向上更新；2.5 倍且止损比例夹在 5% 到 15%。
- 普通信号卖出最早是买入后的下一交易日；ATR 卖出不受该限制。
- 聚宽是收益权威；本计划不创建本地收益回放器，也不打开验证期。

### 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| Create | `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py` | 可直接粘贴到聚宽的完整策略 |
| Create | `tests/test_resonance_reversal_strategy.py` | 指标、事件、共振、状态、平台边界测试 |
| Create | `resonance_reversal_strategy/README.md` | 部署、回测口径、验证顺序和禁区 |
| Preserve | `resonance_reversal_strategy/docs/strategy_spec.md` | 已批准规格；实施中不得自行改写 |
| Preserve | `cross_signal_strategy/**` | 明确非目标，禁止导入或修改 |

---

## Task 1: 建立可导入的聚宽策略骨架与固定配置契约

**Files:**

- Create: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Create: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 写导入、常量、ETF 池和初始化契约的失败测试**

测试文件先建立与现有策略测试一致的 `jqdata` 隔离导入：

```python
import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRATEGY_PATH = (
    ROOT
    / "resonance_reversal_strategy"
    / "smart_trade_joinquant_resonance_reversal_etf.py"
)
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))
spec = importlib.util.spec_from_file_location("resonance_strategy", STRATEGY_PATH)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


EXPECTED_POOL = [
    "510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE",
    "510880.XSHG", "513100.XSHG", "513500.XSHG", "159920.XSHE",
    "513880.XSHG", "513050.XSHG", "518880.XSHG", "159985.XSHE",
]


def test_default_contract_is_frozen():
    assert strategy.STRATEGY_VERSION == "resonance-v0.1.0"
    assert strategy.get_default_etf_pool() == EXPECTED_POOL
    params = strategy.get_default_params()
    assert params["lookback_days"] == 120
    assert params["max_holdings"] == 3
    assert params["target_exposure"] == pytest.approx(0.95)
    assert params["resonance_window"] == 2
    assert params["rsi_period"] == 14
    assert params["kdj"] == (9, 3, 3)
    assert params["boll"] == (20, 2.0)
    assert params["atr_period"] == 14
    assert params["atr_multiplier"] == pytest.approx(2.5)
    assert params["stop_floor"] == pytest.approx(0.05)
    assert params["stop_cap"] == pytest.approx(0.15)


def test_initialize_enables_future_guard_and_fixed_schedules(monkeypatch):
    calls = []
    monkeypatch.setattr(strategy, "set_option", lambda k, v: calls.append(("option", k, v)), raising=False)
    monkeypatch.setattr(strategy, "set_benchmark", lambda code: calls.append(("benchmark", code)), raising=False)
    monkeypatch.setattr(strategy, "PriceRelatedSlippage", lambda value: ("slippage", value), raising=False)
    monkeypatch.setattr(strategy, "set_slippage", lambda value, type=None: calls.append(("slippage", value, type)), raising=False)
    monkeypatch.setattr(strategy, "OrderCost", lambda **kw: kw, raising=False)
    monkeypatch.setattr(strategy, "set_order_cost", lambda value, type=None: calls.append(("cost", value, type)), raising=False)
    monkeypatch.setattr(strategy, "run_daily", lambda fn, time, reference_security=None: calls.append(("daily", fn.__name__, time)), raising=False)
    monkeypatch.setattr(strategy, "log", types.SimpleNamespace(info=lambda *args: None), raising=False)
    monkeypatch.setattr(strategy, "g", types.SimpleNamespace(), raising=False)

    strategy.initialize(types.SimpleNamespace())

    assert ("option", "use_real_price", True) in calls
    assert ("option", "avoid_future_data", True) in calls
    assert ("benchmark", "000300.XSHG") in calls
    assert ("daily", "do_trading", "09:35") in calls
    assert ("daily", "after_close", "15:30") in calls
```

- [ ] **Step 2: 运行目标测试并确认因策略文件不存在或契约缺失而失败**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py::test_default_contract_is_frozen tests/test_resonance_reversal_strategy.py::test_initialize_enables_future_guard_and_fixed_schedules -v
```

Expected: `FAIL`，失败原因只能是模块/常量/函数尚未建立，不能是测试环境误导入现有策略。

- [ ] **Step 3: 写最小可运行策略骨架**

策略文件建立以下固定入口；`do_trading` 和 `after_close` 在本任务只执行状态初始化，后续任务逐步接入明确职责：

```python
from jqdata import *

import datetime
import hashlib
import json
from enum import Enum

import numpy as np
import pandas as pd


STRATEGY_VERSION = "resonance-v0.1.0"
DEPLOYMENT_BUILD_ID = "20260827.1"
BENCHMARK = "000300.XSHG"


class TurnDirection(Enum):
    BUY_TURN = "BUY_TURN"
    SELL_TURN = "SELL_TURN"
    NEUTRAL = "NEUTRAL"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExitReason(Enum):
    ATR_EXIT = "ATR_EXIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"


class Tradability(Enum):
    TRADEABLE = "TRADEABLE"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


class OrderOutcome(Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    NOT_FILLED = "NOT_FILLED"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"


def get_default_etf_pool():
    return [
        "510300.XSHG", "159915.XSHE", "512100.XSHG", "159928.XSHE",
        "510880.XSHG", "513100.XSHG", "513500.XSHG", "159920.XSHE",
        "513880.XSHG", "513050.XSHG", "518880.XSHG", "159985.XSHE",
    ]


def get_default_params():
    return {
        "lookback_days": 120,
        "max_holdings": 3,
        "target_exposure": 0.95,
        "resonance_window": 2,
        "rsi_period": 14,
        "observation_rsi_periods": (6, 12, 24),
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "kdj": (9, 3, 3),
        "kdj_low": 20.0,
        "kdj_high": 80.0,
        "j_low": 0.0,
        "j_high": 100.0,
        "boll": (20, 2.0),
        "atr_period": 14,
        "atr_multiplier": 2.5,
        "stop_floor": 0.05,
        "stop_cap": 0.15,
    }


def business_config_fingerprint(params=None, etf_pool=None):
    payload = {
        "params": params or get_default_params(),
        "etf_pool": etf_pool or get_default_etf_pool(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ensure_runtime_state():
    if not hasattr(g, "params"):
        g.params = get_default_params()
    if not hasattr(g, "etf_pool"):
        g.etf_pool = get_default_etf_pool()
    if not hasattr(g, "position_states"):
        g.position_states = {}
    if not hasattr(g, "processed_resonance_ids"):
        g.processed_resonance_ids = {}
    if not hasattr(g, "observation_events"):
        g.observation_events = {}
    if not hasattr(g, "sold_today"):
        g.sold_today = set()
    if not hasattr(g, "daily_attempted_buys"):
        g.daily_attempted_buys = set()


def do_trading(context):
    ensure_runtime_state()


def after_close(context):
    ensure_runtime_state()


def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark(BENCHMARK)
    set_slippage(PriceRelatedSlippage(0.001), type="fund")
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5,
    ), type="fund")
    run_daily(do_trading, time="09:35", reference_security=BENCHMARK)
    run_daily(after_close, time="15:30", reference_security=BENCHMARK)
    ensure_runtime_state()
    log.info("version=%s build=%s fingerprint=%s pool=%s",
             STRATEGY_VERSION, DEPLOYMENT_BUILD_ID,
             business_config_fingerprint(), get_default_etf_pool())
```

不得增加任何现有策略导入，也不得用 `try/except` 包住 `set_option("avoid_future_data", True)`。

- [ ] **Step 4: 运行测试和编译检查并确认通过**

Run:

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py::test_default_contract_is_frozen tests/test_resonance_reversal_strategy.py::test_initialize_enables_future_guard_and_fixed_schedules -v
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
```

Expected: `2 passed`，`py_compile` 无输出且退出码为 0。

- [ ] **Step 5: 提交配置里程碑**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add frozen JoinQuant strategy contract"
```

---

## Task 2: 实现指标计算，并物理隔离交易字段与观察字段

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 写 RSI/KDJ/BOLL/ATR 边界失败测试**

增加以下测试，使用确定性序列而不是外部行情：

```python
import numpy as np
import pandas as pd


def make_ohlcv_frame(rows):
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(np.linspace(10.0, 20.0, rows), index=index)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(1, rows + 1, dtype=float) * 1000.0,
    }, index=index)


def test_rsi_wilder_edges_and_turn_values():
    rising = pd.Series(range(1, 40), dtype=float)
    falling = pd.Series(range(40, 1, -1), dtype=float)
    flat = pd.Series([10.0] * 40)
    assert strategy.calc_rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert strategy.calc_rsi(falling, 14).iloc[-1] == pytest.approx(0.0)
    assert strategy.calc_rsi(flat, 14).iloc[-1] == pytest.approx(50.0)
    assert strategy.calc_rsi(pd.Series([np.nan] * 20), 14).isna().all()


def test_boll_uses_population_std_and_atr_uses_simple_mean():
    close = pd.Series(np.arange(1.0, 31.0))
    mid, upper, lower = strategy.calc_bollinger(close, 20, 2.0)
    window = close.iloc[-20:]
    assert mid.iloc[-1] == pytest.approx(window.mean())
    assert upper.iloc[-1] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert lower.iloc[-1] == pytest.approx(window.mean() - 2 * window.std(ddof=0))

    high = pd.Series([11, 13, 12, 15, 16, 16, 18, 19, 18, 20, 21, 22, 23, 24, 25], dtype=float)
    low = high - 2
    close2 = high - 1
    atr = strategy.calc_atr(high, low, close2, 14)
    prev_close = close2.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    assert atr.iloc[-1] == pytest.approx(tr.iloc[-14:].mean())


def test_indicator_frame_separates_trade_and_observation_columns():
    frame = strategy.build_indicator_frame(make_ohlcv_frame(140), strategy.get_default_params())
    assert set(strategy.TRADE_INDICATOR_COLUMNS) == {
        "rsi14", "k", "d", "j", "kd_diff", "boll_mid",
        "boll_upper", "boll_lower", "atr14",
    }
    assert set(strategy.OBSERVATION_COLUMNS) == {
        "rsi6", "rsi12", "rsi24", "plus_di", "minus_di", "adx14",
        "volume", "volume_ma5", "volume_ma20", "volume_ratio",
        "boll_width", "boll_mid_slope",
    }
    assert set(strategy.TRADE_INDICATOR_COLUMNS).isdisjoint(strategy.OBSERVATION_COLUMNS)
```

同时添加 `make_ohlcv_frame(rows)`，固定生成有日期索引的 OHLCV 数据，禁止测试读取训练或验证行情文件。

- [ ] **Step 2: 运行指标测试并确认因函数缺失而失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "rsi_wilder or boll_uses or indicator_frame" -v
```

Expected: `FAIL`，指出 `calc_rsi/calc_bollinger/calc_atr/build_indicator_frame` 尚未定义。

- [ ] **Step 3: 实现纯指标函数**

```python
TRADE_INDICATOR_COLUMNS = (
    "rsi14", "k", "d", "j", "kd_diff", "boll_mid",
    "boll_upper", "boll_lower", "atr14",
)
OBSERVATION_COLUMNS = (
    "rsi6", "rsi12", "rsi24", "plus_di", "minus_di", "adx14",
    "volume", "volume_ma5", "volume_ma20", "volume_ratio",
    "boll_width", "boll_mid_slope",
)


def calc_rsi(close, period):
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    rolling_high = high.rolling(n, min_periods=n).max()
    rolling_low = low.rolling(n, min_periods=n).min()
    spread = rolling_high - rolling_low
    rsv = 100.0 * (close - rolling_low) / spread.replace(0, np.nan)
    rsv = rsv.mask(spread == 0, 50.0)
    k = rsv.ewm(alpha=1.0 / m1, adjust=False, min_periods=1).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False, min_periods=1).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def calc_bollinger(close, period=20, std_mult=2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def calc_atr(high, low, close, period=14):
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def calc_dmi_adx(high, low, close, period=14):
    tr = true_range(high, low, close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr_rma = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False,
                                  min_periods=period).mean() / atr_rma
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False,
                                    min_periods=period).mean() / atr_rma
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx
```

- [ ] **Step 4: 组装指标帧，保持交易与观察输出分区**

```python
def build_indicator_frame(price_frame, params):
    frame = price_frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    frame["rsi14"] = calc_rsi(frame["close"], params["rsi_period"])
    for period in params["observation_rsi_periods"]:
        frame["rsi%s" % period] = calc_rsi(frame["close"], period)
    k, d, j = calc_kdj(frame["high"], frame["low"], frame["close"], *params["kdj"])
    frame["k"], frame["d"], frame["j"] = k, d, j
    frame["kd_diff"] = k - d
    mid, upper, lower = calc_bollinger(frame["close"], *params["boll"])
    frame["boll_mid"], frame["boll_upper"], frame["boll_lower"] = mid, upper, lower
    frame["atr14"] = calc_atr(frame["high"], frame["low"], frame["close"], params["atr_period"])
    plus_di, minus_di, adx = calc_dmi_adx(frame["high"], frame["low"], frame["close"], 14)
    frame["plus_di"], frame["minus_di"], frame["adx14"] = plus_di, minus_di, adx
    frame["volume_ma5"] = frame["volume"].rolling(5, min_periods=5).mean()
    frame["volume_ma20"] = frame["volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume_ma20"].replace(0, np.nan)
    frame["boll_width"] = (upper - lower) / mid.replace(0, np.nan)
    frame["boll_mid_slope"] = mid.diff()
    return frame
```

不得把 `OBSERVATION_COLUMNS` 参数传给 Task 3/4 的事件或共振函数；它们只用于日志投影。

- [ ] **Step 5: 运行全部指标测试并提交**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "rsi_wilder or boll_uses or indicator_frame" -v
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add isolated indicator calculations"
```

Expected: 相关测试全部通过；提交只包含新策略和其专属测试。

---

## Task 3: 实现 RSI、KDJ、BOLL 预交叉方向事件及失效生命周期

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 先写方向边界与反向失效的参数化测试**

测试至少包含：RSI 未穿越 30/70 仍能拐头；KDJ 未正式交叉仍能拐头；BOLL 仅触轨不回归不能形成事件；BOLL 沿轨继续创新低/高使旧事件失效。

```python
def make_event(indicator, direction, event_date, expires_date,
               reference_extreme=None):
    return strategy.make_turn_event(
        indicator=indicator,
        direction=direction,
        event_date=event_date,
        expires_date=expires_date,
        trigger_values={"fixture": True},
        reference_extreme=reference_extreme,
    )


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        ({"rsi14": 28.0}, {"rsi14": 29.0}, strategy.TurnDirection.BUY_TURN),
        ({"rsi14": 72.0}, {"rsi14": 71.0}, strategy.TurnDirection.SELL_TURN),
        ({"rsi14": 31.0}, {"rsi14": 32.0}, strategy.TurnDirection.NEUTRAL),
    ],
)
def test_rsi_event_does_not_require_threshold_cross(previous, current, expected):
    assert strategy.detect_rsi_direction(previous, current, strategy.get_default_params()) is expected


def test_kdj_buy_turn_can_precede_formal_golden_cross():
    previous = {"k": 15.0, "d": 20.0, "j": 5.0, "kd_diff": -5.0}
    current = {"k": 17.0, "d": 20.0, "j": 11.0, "kd_diff": -3.0}
    assert current["k"] < current["d"]
    assert strategy.detect_kdj_direction(previous, current, strategy.get_default_params()) is strategy.TurnDirection.BUY_TURN


def test_boll_touch_without_return_inside_is_neutral():
    previous = {"low": 9.0, "high": 10.0, "close": 9.2, "boll_lower": 9.3, "boll_upper": 11.0}
    current = {"low": 8.8, "high": 9.5, "close": 9.0, "boll_lower": 9.1, "boll_upper": 10.8}
    assert strategy.detect_boll_direction(previous, current) is strategy.TurnDirection.NEUTRAL


def test_new_opposite_event_replaces_old_event_and_boll_break_invalidates():
    book = strategy.empty_event_book()
    buy = make_event("BOLL", strategy.TurnDirection.BUY_TURN, "2021-01-04", "2021-01-05", reference_extreme=9.0)
    strategy.apply_event(book, buy)
    invalidated = strategy.invalidate_boll_structure(
        book, {"date": "2021-01-05", "close": 8.8, "low": 8.7,
               "boll_lower": 8.9, "high": 9.1, "boll_upper": 10.5})
    assert "BOLL" not in book["active"]
    assert invalidated["invalid_reason"] == "NEW_LOWER_LOW_OUTSIDE_LOWER_BAND"
```

- [ ] **Step 2: 运行事件测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "event or golden_cross or boll_touch" -v
```

Expected: `FAIL`，原因是事件函数和事件簿尚未实现。

- [ ] **Step 3: 实现方向检测纯函数**

```python
INDICATORS = ("BOLL", "RSI", "KDJ")


def detect_rsi_direction(previous, current, params):
    prev_rsi, curr_rsi = previous["rsi14"], current["rsi14"]
    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return TurnDirection.NEUTRAL
    if prev_rsi <= params["rsi_low"] and curr_rsi > prev_rsi:
        return TurnDirection.BUY_TURN
    if prev_rsi >= params["rsi_high"] and curr_rsi < prev_rsi:
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_kdj_direction(previous, current, params):
    required = ("k", "d", "j", "kd_diff")
    if any(pd.isna(previous[name]) or pd.isna(current[name]) for name in required):
        return TurnDirection.NEUTRAL
    buy_extreme = min(previous["k"], previous["d"]) <= params["kdj_low"] or previous["j"] <= params["j_low"]
    sell_extreme = max(previous["k"], previous["d"]) >= params["kdj_high"] or previous["j"] >= params["j_high"]
    if buy_extreme and current["j"] > previous["j"] and current["kd_diff"] > previous["kd_diff"]:
        return TurnDirection.BUY_TURN
    if sell_extreme and current["j"] < previous["j"] and current["kd_diff"] < previous["kd_diff"]:
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL


def detect_boll_direction(previous, current):
    values = [previous.get(k) for k in ("low", "high", "close", "boll_lower", "boll_upper")]
    values += [current.get(k) for k in ("low", "high", "close", "boll_lower", "boll_upper")]
    if any(pd.isna(value) for value in values):
        return TurnDirection.NEUTRAL
    touched_lower = (previous["low"] <= previous["boll_lower"] or
                     previous["close"] <= previous["boll_lower"] or
                     current["low"] <= current["boll_lower"] or
                     current["close"] <= current["boll_lower"])
    touched_upper = (previous["high"] >= previous["boll_upper"] or
                     previous["close"] >= previous["boll_upper"] or
                     current["high"] >= current["boll_upper"] or
                     current["close"] >= current["boll_upper"])
    if touched_lower and current["close"] > current["boll_lower"] and current["close"] > previous["close"]:
        return TurnDirection.BUY_TURN
    if touched_upper and current["close"] < current["boll_upper"] and current["close"] < previous["close"]:
        return TurnDirection.SELL_TURN
    return TurnDirection.NEUTRAL
```

- [ ] **Step 4: 实现可审计事件对象、事件簿和两交易日失效**

```python
def make_turn_event(indicator, direction, event_date, expires_date,
                    trigger_values, reference_extreme=None):
    return {
        "indicator": indicator,
        "direction": direction,
        "event_date": event_date,
        "expires_date": expires_date,
        "trigger_values": dict(trigger_values),
        "reference_extreme": reference_extreme,
        "invalid_reason": None,
    }


def empty_event_book():
    return {"active": {}, "invalidated": []}


def invalidate_event(book, indicator, reason):
    event = book["active"].pop(indicator, None)
    if event is not None:
        event = dict(event)
        event["invalid_reason"] = reason
        book["invalidated"].append(event)
    return event


def apply_event(book, event):
    old = book["active"].get(event["indicator"])
    if old is not None and old["direction"] is not event["direction"]:
        invalidate_event(book, event["indicator"], "REPLACED_BY_OPPOSITE_EVENT")
    book["active"][event["indicator"]] = event


def expire_events(book, signal_date):
    for indicator, event in list(book["active"].items()):
        if event["expires_date"] < signal_date:
            invalidate_event(book, indicator, "EVENT_EXPIRED")


def invalidate_boll_structure(book, latest_row):
    event = book["active"].get("BOLL")
    if event is None:
        return None
    if event["direction"] is TurnDirection.BUY_TURN:
        broken = (latest_row["close"] <= latest_row["boll_lower"] and
                  latest_row["low"] < event["reference_extreme"])
        if broken:
            return invalidate_event(book, "BOLL", "NEW_LOWER_LOW_OUTSIDE_LOWER_BAND")
    if event["direction"] is TurnDirection.SELL_TURN:
        broken = (latest_row["close"] >= latest_row["boll_upper"] and
                  latest_row["high"] > event["reference_extreme"])
        if broken:
            return invalidate_event(book, "BOLL", "NEW_HIGHER_HIGH_OUTSIDE_UPPER_BAND")
    return None
```

再实现 `collect_latest_events(indicator_frame, signal_date, next_trade_date)`：只扫描 T-2/T-1 两个完整信号日，按日期顺序应用事件、反向替换和结构失效；`next_trade_date` 由调用层从交易日历提供，不得以自然日 `+1` 代替交易日。

- [ ] **Step 5: 补齐正式交叉非必要、过期和事件日期测试并运行**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "event or golden_cross or boll_touch or expired" -v
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add pre-cross turn event lifecycle"
```

Expected: 事件测试全绿；正式交叉只可出现在日志字段断言中，不可成为事件必要条件。

---

## Task 4: 实现无评分共振真值表、防重复标识和稳定候选排序

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 用参数化测试覆盖完整真值表与新鲜度**

测试直接构造事件簿，避免把指标计算误当成共振逻辑：

```python
def event_book_for_directions(boll, rsi, kdj, t1):
    active = {}
    for indicator, direction_name in (("BOLL", boll), ("RSI", rsi), ("KDJ", kdj)):
        if direction_name == "NEUTRAL":
            continue
        active[indicator] = make_event(
            indicator,
            strategy.TurnDirection[direction_name],
            t1,
            t1,
        )
    return {"active": active, "invalidated": []}


def decision(code, support_count, boll_age):
    return {
        "code": code,
        "support_count": support_count,
        "boll_age": boll_age,
    }


@pytest.mark.parametrize(
    "boll,rsi,kdj,buy_allowed,sell_allowed",
    [
        ("BUY_TURN", "BUY_TURN", "NEUTRAL", True, False),
        ("BUY_TURN", "NEUTRAL", "BUY_TURN", True, False),
        ("BUY_TURN", "BUY_TURN", "BUY_TURN", True, False),
        ("BUY_TURN", "BUY_TURN", "SELL_TURN", False, False),
        ("BUY_TURN", "SELL_TURN", "BUY_TURN", False, False),
        ("SELL_TURN", "SELL_TURN", "NEUTRAL", False, True),
        ("SELL_TURN", "NEUTRAL", "SELL_TURN", False, True),
        ("SELL_TURN", "SELL_TURN", "SELL_TURN", False, True),
        ("SELL_TURN", "SELL_TURN", "BUY_TURN", False, False),
        ("SELL_TURN", "BUY_TURN", "SELL_TURN", False, False),
        ("NEUTRAL", "BUY_TURN", "BUY_TURN", False, False),
    ],
)
def test_complete_resonance_truth_table(boll, rsi, kdj, buy_allowed, sell_allowed):
    events = event_book_for_directions(boll, rsi, kdj, t1="2021-01-05")
    buy = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN, events, "2021-01-05")
    sell = strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.SELL_TURN, events, "2021-01-05")
    assert (buy is not None) is buy_allowed
    assert (sell is not None) is sell_allowed


def test_two_old_events_cannot_resonate_and_opposite_third_indicator_vetoes():
    old_events = event_book_for_directions(
        "BUY_TURN", "BUY_TURN", "NEUTRAL", t1="2021-01-04"
    )
    assert strategy.build_resonance_decision(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        old_events, "2021-01-05") is None


def test_candidate_sort_is_three_factor_then_boll_freshness_then_code():
    decisions = [
        decision("513100.XSHG", support_count=2, boll_age=0),
        decision("159915.XSHE", support_count=3, boll_age=1),
        decision("510300.XSHG", support_count=2, boll_age=0),
        decision("512100.XSHG", support_count=2, boll_age=1),
    ]
    assert [item["code"] for item in strategy.sort_buy_decisions(decisions)] == [
        "159915.XSHE", "510300.XSHG", "513100.XSHG", "512100.XSHG",
    ]
```

增加测试：改变事件簿外的 RSI6/12/24、ADX、量能和 BOLL 带宽值，`resonance_id`、是否共振、优先级完全不变。

- [ ] **Step 2: 运行共振测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "resonance or candidate_sort or observation_fields" -v
```

Expected: `FAIL`，共振决策和排序函数尚未实现。

- [ ] **Step 3: 实现只接收事件簿的共振决策**

```python
OPPOSITE = {
    TurnDirection.BUY_TURN: TurnDirection.SELL_TURN,
    TurnDirection.SELL_TURN: TurnDirection.BUY_TURN,
}


def active_direction(event_book, indicator):
    event = event_book["active"].get(indicator)
    return event["direction"] if event is not None else TurnDirection.NEUTRAL


def build_resonance_id(code, direction, supporters):
    parts = [direction.value, code]
    parts.extend("%s:%s" % (event["indicator"], event["event_date"])
                 for event in sorted(supporters, key=lambda item: item["indicator"]))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def build_resonance_decision(code, direction, event_book, signal_date):
    boll = event_book["active"].get("BOLL")
    if boll is None or boll["direction"] is not direction:
        return None
    oscillators = [event_book["active"].get("RSI"),
                   event_book["active"].get("KDJ")]
    supporters = [boll] + [event for event in oscillators
                           if event is not None and event["direction"] is direction]
    if len(supporters) < 2:
        return None
    if any(event is not None and event["direction"] is OPPOSITE[direction]
           for event in oscillators):
        return None
    if not any(event["event_date"] == signal_date for event in supporters):
        return None
    return {
        "code": code,
        "direction": direction,
        "supporters": tuple(event["indicator"] for event in supporters),
        "support_count": len(supporters),
        "boll_age": 0 if boll["event_date"] == signal_date else 1,
        "resonance_id": build_resonance_id(code, direction, supporters),
        "expires_date": min(event["expires_date"] for event in supporters),
    }


def sort_buy_decisions(decisions):
    return sorted(decisions, key=lambda item: (
        -item["support_count"], item["boll_age"], item["code"]
    ))
```

该函数签名不得接收综合分数、ADX、成交量、其他 RSI、账户资金或 T 日现价。

- [ ] **Step 4: 实现已处理标识清理与防重复纯函数**

```python
def prune_processed_resonance_ids(processed, signal_date):
    return {
        resonance_id: expires_date
        for resonance_id, expires_date in processed.items()
        if expires_date >= signal_date
    }


def mark_resonance_processed(processed, decision):
    processed[decision["resonance_id"]] = decision["expires_date"]
```

测试必须证明：同一支持事件只处理一次；事件过期后清理；同一 ETF 后续不同日期的新事件生成不同标识，不会被永久封锁。

- [ ] **Step 5: 运行共振测试与全量当前测试并提交**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "resonance or candidate_sort or observation_fields or processed" -v
python -m pytest tests/test_resonance_reversal_strategy.py -v
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add boolean resonance decision engine"
```

---

## Task 5: 实现自适应新仓、持仓状态和冻结 ATR 风控

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 写仓位公式、ATR 上下限和持有期失败测试**

```python
@pytest.mark.parametrize(
    "total,cash,expected",
    [
        (20000.0, 20000.0, 20000.0 * 0.95 / 3),
        (30000.0, 4000.0, 2500.0),
        (30000.0, 1000.0, 0.0),
    ],
)
def test_buy_target_adapts_to_current_assets_and_preserves_cash(total, cash, expected):
    assert strategy.calc_buy_target_value(total, cash, strategy.get_default_params()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "anchor,entry_atr,expected_pct",
    [(100.0, 1.0, 0.05), (100.0, 4.0, 0.10), (100.0, 10.0, 0.15)],
)
def test_atr_stop_clamps_percentage(anchor, entry_atr, expected_pct):
    result = strategy.calc_stop_state(anchor, entry_atr, strategy.get_default_params())
    assert result["stop_pct"] == pytest.approx(expected_pct)
    assert result["stop_price"] == pytest.approx(anchor * (1 - expected_pct))


def test_highest_anchor_only_moves_up_on_close_and_entry_atr_stays_frozen():
    state = strategy.make_position_state("2021-01-05", 2.0, 100.0)
    strategy.update_highest_close_anchor(state, 105.0)
    strategy.update_highest_close_anchor(state, 102.0)
    assert state["highest_close_anchor"] == pytest.approx(105.0)
    assert state["entry_atr"] == pytest.approx(2.0)


def test_signal_sell_is_next_trade_day_only_but_atr_has_no_hold_lock():
    assert not strategy.can_signal_sell("2021-01-05", "2021-01-05")
    assert strategy.can_signal_sell("2021-01-05", "2021-01-06")
```

- [ ] **Step 2: 运行风险与资金测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "buy_target or atr_stop or highest_anchor or signal_sell" -v
```

- [ ] **Step 3: 实现资金和风险纯函数**

```python
def calc_buy_target_value(total_value, available_cash, params):
    standard_target = total_value * params["target_exposure"] / params["max_holdings"]
    cash_reserve = total_value * (1.0 - params["target_exposure"])
    return min(standard_target, max(0.0, available_cash - cash_reserve))


def calc_stop_state(highest_close_anchor, entry_atr, params):
    if highest_close_anchor <= 0 or entry_atr <= 0 or pd.isna(entry_atr):
        return None
    raw_pct = params["atr_multiplier"] * entry_atr / highest_close_anchor
    stop_pct = min(params["stop_cap"], max(params["stop_floor"], raw_pct))
    return {
        "raw_pct": raw_pct,
        "stop_pct": stop_pct,
        "stop_price": highest_close_anchor * (1.0 - stop_pct),
    }


def make_position_state(buy_date, entry_atr, entry_price):
    return {
        "buy_date": buy_date,
        "entry_atr": float(entry_atr),
        "highest_close_anchor": float(entry_price),
        "pending_exit": None,
    }


def update_highest_close_anchor(position_state, closing_price):
    if closing_price is not None and closing_price > 0:
        position_state["highest_close_anchor"] = max(
            position_state["highest_close_anchor"], float(closing_price)
        )


def can_signal_sell(buy_date, decision_date):
    return buy_date < decision_date
```

- [ ] **Step 4: 实现每日状态重置与持仓状态保护**

```python
def reset_daily_state(decision_date, signal_date):
    ensure_runtime_state()
    if getattr(g, "state_date", None) != decision_date:
        g.state_date = decision_date
        g.sold_today = set()
        g.daily_attempted_buys = set()
    g.processed_resonance_ids = prune_processed_resonance_ids(
        g.processed_resonance_ids, signal_date
    )


def clear_position_state_if_flat(code, actual_amount):
    if actual_amount == 0:
        g.position_states.pop(code, None)
        return True
    return False
```

测试必须断言：卖出接口已调用但 `actual_amount > 0` 时返回 `False`，`buy_date/entry_atr/highest_close_anchor/pending_exit` 全部保留。

- [ ] **Step 5: 运行相关测试、全量测试并提交**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "buy_target or atr_stop or highest_anchor or signal_sell or daily_state or flat" -v
python -m pytest tests/test_resonance_reversal_strategy.py -v
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add adaptive sizing and frozen ATR risk state"
```

---

## Task 6: 实现停牌、未知、未成交、部分成交和待卖重试状态机

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 写订单结果完整真值表测试**

```python
@pytest.mark.parametrize(
    "side,before,after,target,tradability,order_amount,filled,expected",
    [
        ("BUY", 0, 0, 100, "PAUSED", None, None, "PAUSED"),
        ("BUY", 0, 0, 100, "UNKNOWN", None, None, "UNKNOWN"),
        ("BUY", 0, 100, 100, "TRADEABLE", 100, 100, "FILLED"),
        ("BUY", 0, 50, 100, "TRADEABLE", 100, 50, "PARTIAL"),
        ("BUY", 0, 0, 100, "TRADEABLE", 100, 0, "NOT_FILLED"),
        ("SELL", 100, 0, 0, "TRADEABLE", -100, -100, "FILLED"),
        ("SELL", 100, 40, 0, "TRADEABLE", -100, -60, "PARTIAL"),
        ("SELL", 100, 100, 0, "TRADEABLE", -100, 0, "NOT_FILLED"),
    ],
)
def test_order_outcome_truth_table(side, before, after, target, tradability, order_amount, filled, expected):
    order = None if order_amount is None else types.SimpleNamespace(amount=order_amount, filled=filled)
    outcome = strategy.classify_order_outcome(
        strategy.OrderSide[side], before, after, target,
        strategy.Tradability[tradability], order,
    )
    assert outcome is strategy.OrderOutcome[expected]
```

再增加状态转换测试：

- 部分买入建立 `entry_atr/buy_date/highest_close_anchor`，同日不追单；
- 买入未成交不建立持仓状态；
- 部分/未成交卖出保留所有风险字段并建立 `pending_exit`；
- ATR 待卖覆盖普通待卖，普通待卖不能覆盖 ATR；
- 实际归零才清理；
- 次日 `pending_exit` 先于新信号重试。

- [ ] **Step 2: 运行状态机测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "order_outcome or partial or pending_exit" -v
```

- [ ] **Step 3: 实现订单分类，不让结果改写信号**

```python
def classify_order_outcome(side, before_amount, after_amount, target_amount,
                           tradability, order):
    if tradability is Tradability.PAUSED:
        return OrderOutcome.PAUSED
    if tradability is Tradability.UNKNOWN:
        return OrderOutcome.UNKNOWN
    if side is OrderSide.SELL and after_amount == 0:
        return OrderOutcome.FILLED
    if side is OrderSide.BUY and target_amount is not None and after_amount >= target_amount:
        return OrderOutcome.FILLED
    requested = abs(getattr(order, "amount", 0) or 0) if order is not None else 0
    filled = abs(getattr(order, "filled", 0) or 0) if order is not None else 0
    if requested > 0 and filled >= requested:
        return OrderOutcome.FILLED
    if after_amount != before_amount or filled > 0:
        return OrderOutcome.PARTIAL
    return OrderOutcome.NOT_FILLED
```

调用层必须在下单前后读取实际持仓；不得只根据订单对象存在就认定成交。

- [ ] **Step 4: 实现待卖优先级与状态同步**

```python
EXIT_PRIORITY = {
    ExitReason.SIGNAL_EXIT: 1,
    ExitReason.ATR_EXIT: 2,
}


def set_pending_exit(position_state, reason, created_date, trigger_value,
                     remaining_amount):
    existing = position_state.get("pending_exit")
    if existing is not None and EXIT_PRIORITY[existing["reason"]] > EXIT_PRIORITY[reason]:
        return existing
    position_state["pending_exit"] = {
        "created_date": created_date,
        "reason": reason,
        "trigger_value": trigger_value,
        "remaining_amount": remaining_amount,
    }
    return position_state["pending_exit"]


def sync_buy_state_after_order(code, outcome, before_amount, after_amount,
                               decision_date, entry_atr, entry_price):
    g.daily_attempted_buys.add(code)
    if after_amount > before_amount:
        g.position_states[code] = make_position_state(
            decision_date, entry_atr, entry_price
        )
    return outcome


def sync_sell_state_after_order(code, outcome, reason, decision_date,
                                trigger_value, actual_amount):
    state = g.position_states.get(code)
    if actual_amount == 0:
        g.position_states.pop(code, None)
        g.sold_today.add(code)
        return outcome
    if state is not None:
        set_pending_exit(state, reason, decision_date, trigger_value, actual_amount)
    return outcome
```

`retry_pending_exits` 只读取现有持仓状态并调用卖出包装器；不得读取或取消共振事件。恢复交易后仍未归零就更新剩余数量并保留待卖。

- [ ] **Step 5: 运行状态机及全量测试并提交**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "order_outcome or partial or pending_exit" -v
python -m pytest tests/test_resonance_reversal_strategy.py -v
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): add explicit order lifecycle state machine"
```

---

## Task 7: 接入 T-1 数据边界和完整 09:35/15:30 聚宽编排

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`

- [ ] **Step 1: 写未来函数边界和固定调用顺序失败测试**

```python
def fake_context(previous_date, current_date):
    return types.SimpleNamespace(
        previous_date=previous_date,
        current_dt=pd.Timestamp(current_date),
        portfolio=types.SimpleNamespace(
            positions={}, total_value=20000.0, available_cash=20000.0
        ),
    )


def test_signal_loader_is_strictly_t_minus_one(monkeypatch):
    calls = []
    monkeypatch.setattr(strategy, "get_price", lambda code, **kw: calls.append((code, kw)) or make_ohlcv_frame(120), raising=False)
    strategy.load_signal_price_frame("510300.XSHG", "2021-01-05", 120)
    assert calls == [("510300.XSHG", {
        "end_date": "2021-01-05",
        "count": 120,
        "frequency": "daily",
        "fields": ["open", "high", "low", "close", "volume"],
        "skip_paused": True,
        "fq": "pre",
        "panel": False,
    })]


def test_do_trading_stage_order_has_no_broad_early_return(monkeypatch):
    order = []
    monkeypatch.setattr(strategy, "reset_daily_state", lambda *a: order.append("reset"))
    monkeypatch.setattr(strategy, "retry_pending_exits", lambda *a: order.append("pending"))
    monkeypatch.setattr(strategy, "run_atr_exits", lambda *a: order.append("atr"))
    monkeypatch.setattr(strategy, "build_signal_snapshots", lambda *a: order.append("signals") or {})
    monkeypatch.setattr(strategy, "run_signal_exits", lambda *a: order.append("signal_sells"))
    monkeypatch.setattr(strategy, "run_signal_buys", lambda *a: order.append("buys"))
    monkeypatch.setattr(strategy, "get_current_data", lambda: {}, raising=False)
    context = fake_context(previous_date="2021-01-05", current_date="2021-01-06")
    strategy.do_trading(context)
    assert order == ["reset", "pending", "atr", "signals", "signal_sells", "buys"]
```

再写平台编排测试覆盖：

- 持仓即使信号数据不足，ATR 检查仍执行；
- ATR 清仓后代码进入 `sold_today`，同日买入队列不能买回；
- 普通卖出后重新读取实际持仓再算空位；
- 满仓时不替换，已有持仓买入共振只记录不加仓；
- 明确停牌候选可由下一只递补；未知/无效报价/未成交候选消耗计划槽位且不递补；
- 部分成交消耗槽位且不追单；
- 已处理 `resonance_id` 不重复提交；
- 当前价只进入 ATR/订单包装器，改变当前价不会改变冻结的事件、排序和目标公式；
- `FutureDataError` 从 `get_price` 原样冒泡，代码中没有吞异常的 `except`。

- [ ] **Step 2: 运行编排测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "strictly_t_minus_one or stage_order or atr_before or full_portfolio or backfill or future_data" -v
```

- [ ] **Step 3: 实现严格 T-1 loader 与全池快照**

```python
def load_signal_price_frame(code, prev_date, lookback_days):
    return get_price(
        code,
        end_date=prev_date,
        count=lookback_days,
        frequency="daily",
        fields=["open", "high", "low", "close", "volume"],
        skip_paused=True,
        fq="pre",
        panel=False,
    )


def build_signal_snapshot(code, prev_date, params, next_trade_date):
    price_frame = load_signal_price_frame(code, prev_date, params["lookback_days"])
    if price_frame is None or len(price_frame) < params["lookback_days"]:
        return {"code": code, "valid": False, "reason": "INSUFFICIENT_DATA"}
    indicators = build_indicator_frame(price_frame, params)
    latest = indicators.iloc[-1]
    required = list(TRADE_INDICATOR_COLUMNS)
    if latest[required].isna().any() or latest["atr14"] <= 0:
        return {"code": code, "valid": False, "reason": "INVALID_TRADE_INDICATORS"}
    event_book = collect_latest_events(indicators, prev_date, next_trade_date)
    return {
        "code": code,
        "valid": True,
        "signal_date": prev_date,
        "close": float(latest["close"]),
        "entry_atr": float(latest["atr14"]),
        "event_book": event_book,
        "trade_values": latest[list(TRADE_INDICATOR_COLUMNS)].to_dict(),
        "observation_values": latest[list(OBSERVATION_COLUMNS)].to_dict(),
    }


def build_signal_snapshots(prev_date, params):
    snapshots = {}
    for code in get_default_etf_pool():
        snapshots[code] = build_signal_snapshot(code, prev_date, params, get_next_trade_date)
    return snapshots
```

`build_resonance_decision` 只能收到 `event_book`；`observation_values` 只传日志函数。

- [ ] **Step 4: 实现订单包装器和各阶段函数**

明确按以下唯一签名拆分函数，禁止把所有分支塞回 `do_trading`：

| 唯一函数签名 | 输入职责 | 返回契约 |
|---|---|---|
| `retry_pending_exits(context, current_data)` | 只处理已有 `pending_exit` | 每只代码及 `OrderOutcome` 的列表 |
| `run_atr_exits(context, current_data)` | 只检查实际持仓的冻结 ATR 风险 | 已尝试 ATR 卖出的代码集合 |
| `run_signal_exits(context, current_data, snapshots)` | 只处理仍持有代码的顶部共振 | 已尝试普通卖出的代码集合 |
| `run_signal_buys(context, current_data, snapshots)` | 只处理卖出后实际空位和买入队列 | 候选代码及 `OrderOutcome` 的列表 |
| `get_actual_positions(context)` | 读取当前非零实际持仓 | `code -> position` 字典 |
| `get_actual_amount(context, code)` | 读取单只 ETF 实际数量 | 非负整数 |
| `get_tradability(current_data, code)` | 映射停牌、可交易或未知 | `Tradability` |
| `get_execution_price(current_data, code)` | 读取正数实时执行价 | 正数或 `None` |
| `submit_buy(context, code, snapshot, decision)` | 内部按下单时总资产/现金计算目标并核对成交 | `OrderOutcome` |
| `submit_sell(context, code, reason, trigger_value)` | 清仓并按实际剩余数量同步待卖 | `OrderOutcome` |

`run_signal_buys` 的槽位消费必须显式实现：

```python
def run_signal_buys(context, current_data, snapshots):
    actual_positions = get_actual_positions(context)
    remaining_slots = max(0, g.params["max_holdings"] - len(actual_positions))
    if remaining_slots == 0:
        return []
    decisions = collect_buy_decisions(snapshots, actual_positions)
    results = []
    for decision in sort_buy_decisions(decisions):
        if remaining_slots == 0:
            break
        code = decision["code"]
        if code in actual_positions or code in g.sold_today or code in g.daily_attempted_buys:
            continue
        if decision["resonance_id"] in g.processed_resonance_ids:
            continue
        tradability = get_tradability(current_data, code)
        if tradability is Tradability.PAUSED:
            results.append((code, OrderOutcome.PAUSED))
            continue
        mark_resonance_processed(g.processed_resonance_ids, decision)
        g.daily_attempted_buys.add(code)
        if tradability is Tradability.UNKNOWN:
            results.append((code, OrderOutcome.UNKNOWN))
            remaining_slots -= 1
            continue
        outcome = submit_buy(context, code, snapshots[code], decision)
        results.append((code, outcome))
        remaining_slots -= 1
    return results
```

`submit_buy(context, code, snapshot, decision)` 是唯一买入包装器签名，并由测试固定；不得保留旧包装器或重载。普通卖出与 ATR 卖出都必须通过同一个 `submit_sell`，但 `ExitReason` 必须原样传递，不能用 `is_atr=True/False` 替代。

- [ ] **Step 5: 用无提前返回的入口串联固定管线**

```python
def do_trading(context):
    ensure_runtime_state()
    decision_date = context.current_dt.date()
    signal_date = context.previous_date
    reset_daily_state(decision_date, signal_date)
    current_data = get_current_data()
    retry_pending_exits(context, current_data)
    run_atr_exits(context, current_data)
    snapshots = build_signal_snapshots(signal_date, g.params)
    run_signal_exits(context, current_data, snapshots)
    run_signal_buys(context, current_data, snapshots)
```

注意：空快照只会使信号买卖阶段各自无动作；它不能阻止前面的 `pending_exit` 和 ATR。`run_signal_buys` 内部必须重新读取卖出后的实际持仓。

- [ ] **Step 6: 实现 15:30 最高收盘锚点与回顾观察**

```python
def after_close(context):
    ensure_runtime_state()
    current_data = get_current_data()
    for code, state in list(g.position_states.items()):
        actual_amount = get_actual_amount(context, code)
        if actual_amount == 0:
            clear_position_state_if_flat(code, actual_amount)
            continue
        closing_price = get_execution_price(current_data, code)
        update_highest_close_anchor(state, closing_price)
    record_due_observation_outcomes(context, current_data)
    log_portfolio_summary(context)
```

`record_due_observation_outcomes` 只处理已经收盘且到达 1/3/5 交易日观察节点的历史事件；不得预取尚未发生的价格，也不得把结果写回 `event_book` 或交易快照。

- [ ] **Step 7: 运行编排测试、全量测试、编译检查并提交**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "strictly_t_minus_one or stage_order or atr_before or full_portfolio or backfill or future_data" -v
python -m pytest tests/test_resonance_reversal_strategy.py -v
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py tests/test_resonance_reversal_strategy.py
git commit -m "feat(resonance): orchestrate strict T-1 JoinQuant trading flow"
```

---

## Task 8: 完成日志、README、静态隔离门禁和首个代码里程碑验收

**Files:**

- Modify: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- Modify: `tests/test_resonance_reversal_strategy.py`
- Create: `resonance_reversal_strategy/README.md`
- Preserve: `resonance_reversal_strategy/docs/strategy_spec.md`
- Preserve: `cross_signal_strategy/**`

- [ ] **Step 1: 写日志投影和静态边界失败测试**

```python
def test_decision_payload_cannot_receive_observation_fields():
    import inspect
    assert list(inspect.signature(strategy.build_resonance_decision).parameters) == [
        "code", "direction", "event_book", "signal_date",
    ]


def test_strategy_does_not_import_cross_signal_or_swallow_future_errors():
    source = STRATEGY_PATH.read_text(encoding="utf-8")
    assert "cross_signal_strategy" not in source
    assert "smart_trade_joinquant_cross_signal_etf" not in source
    assert 'set_option("avoid_future_data", True)' in source
    assert "FutureDataError" not in source


def test_observation_outcomes_are_retrospective_and_not_trade_inputs():
    record = strategy.make_observation_event(
        resonance_id="abc", code="510300.XSHG",
        event_date="2021-01-05", event_close=10.0,
        due_dates={1: "2021-01-06", 3: "2021-01-08", 5: "2021-01-12"},
    )
    assert strategy.due_observation_horizons(record, "2021-01-05") == []
    assert strategy.due_observation_horizons(record, "2021-01-06") == [1]
```

增加日志契约测试，捕获 logger 调用并断言至少包含：版本/构建/指纹、decision_date/signal_date、三类交易指标与 RSI6/12/24 对照、事件与拒绝原因、候选排序、ATR 状态、订单前后持仓、pending_exit、收盘资产与最高收盘锚点。

- [ ] **Step 2: 运行日志与静态测试并确认失败**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -k "decision_payload or import_cross or observation_outcomes or logging_contract" -v
```

- [ ] **Step 3: 实现结构化日志和回顾观察记录**

统一通过以下入口输出，避免日志逻辑散落并反向影响交易：

| 唯一函数签名 | 职责 | 返回契约 |
|---|---|---|
| `log_signal_snapshot(snapshot)` | 输出交易指标、观察投影、事件及失效原因 | `None` |
| `log_resonance_decision(decision, accepted, reason)` | 输出共振支持者、冲突、新鲜度与排序级别 | `None` |
| `log_order_transition(code, side, outcome, before_amount, after_amount, requested_target, pending_exit)` | 输出订单前后实际持仓与待卖状态 | `None` |
| `log_portfolio_summary(context)` | 输出总资产、现金、持仓和锚点 | `None` |
| `make_observation_event(resonance_id, code, event_date, event_close, due_dates)` | 建立只读回顾观察记录 | 观察记录字典 |
| `due_observation_horizons(record, closing_date)` | 查找今天已到期且尚未记录的 horizon | 升序整数列表 |
| `record_due_observation_outcomes(context, current_data)` | 收盘后记录已经到期的 1/3/5 日表现 | `None` |

观察记录的两个纯函数按以下内容实现并测试：

```python
def make_observation_event(resonance_id, code, event_date, event_close, due_dates):
    return {
        "resonance_id": resonance_id,
        "code": code,
        "event_date": event_date,
        "event_close": float(event_close),
        "due_dates": dict(due_dates),
        "outcomes": {},
    }


def due_observation_horizons(record, closing_date):
    return sorted(
        horizon for horizon, due_date in record["due_dates"].items()
        if due_date == closing_date and horizon not in record["outcomes"]
    )
```

日志函数只接收结果副本，不返回交易资格。`make_observation_event` 保存事件日收盘和预先由交易日历确定的 1/3/5 日到期日；`record_due_observation_outcomes` 仅在到期日收盘后记录实际涨跌幅并移除已完成 horizon。

- [ ] **Step 4: 编写部署和验证 README**

README 必须包含：

1. 本目录是独立新策略，未替代任何现有策略；
2. 12 只 ETF 池和固定参数摘要；
3. T-1 信号、T 日 09:35 执行、15:30 收盘更新的数据边界；
4. 聚宽复制运行方式和建议初始资金 20,000 元；
5. 标准成本与双倍摩擦设置；
6. 验证顺序：短区间冒烟 → 2019-2021 冻结训练 → 训练门槛判断 → 用户确认后才进入验证窗口；
7. 明确禁止本地收益替代聚宽、验证期调参、删除训练期表现差的 ETF、增加观察指标为交易条件；
8. 当前里程碑不包含 PTrade、本地回放器和实盘授权。

README 不写任何尚未实际获得的收益、夏普、回撤或“可实盘”结论。

- [ ] **Step 5: 执行完整验证与占位符扫描**

```powershell
python -m pytest tests/test_resonance_reversal_strategy.py -v
python -m py_compile resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
rg -n -i "TODO|TBD|implement later|placeholder|pass$|NotImplementedError|similar to" resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py resonance_reversal_strategy/README.md tests/test_resonance_reversal_strategy.py
rg -n "cross_signal_strategy|smart_trade_joinquant_cross_signal_etf" resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py
git diff --check
git status --short
```

Expected:

- 新策略测试全部通过；
- Python 编译通过；
- 占位符扫描无结果；
- 新策略源码对现有 cross-signal 无引用；
- `git diff --check` 无输出；
- 状态只包含本里程碑明确文件。

随后逐项映射测试证据：

| 规格风险 | 必须提供的证据 |
|---|---|
| 未来函数 | loader 参数测试、初始化选项测试、异常不吞测试 |
| 观察指标越权 | 函数签名测试、观察字段变动不改变决策测试 |
| 预交叉语义 | 未正式金叉/死叉仍可形成 KDJ 事件测试 |
| 两日共振 | 全真值表、新鲜度、反向替换、过期测试 |
| 资金自适应 | 三组资产/现金边界测试，已有仓位不再平衡测试 |
| ATR 独立优先 | 调用顺序、上下限、同日禁买、入场 ATR 冻结测试 |
| 订单清理错误 | 部分成交、未成交、停牌、未知、实际归零测试 |
| 非目标保护 | 无 cross-signal 引用、目标文件清单和 `git diff` 审查 |

- [ ] **Step 6: 再次执行业务范围与控制流差异门禁**

完成前重新读取并逐条执行：

- `preserve-business-scope`：每个改动块映射到规格；验证一个正例、边界、反例和相邻非目标；删除无法映射的改动。
- `preserve-control-flow-semantics`：复查全部新增条件、提前返回、异常和状态清理；核对本文调用方矩阵与枚举真值表；分别报告静态、编译、本地测试和未运行的聚宽回测。

如果任何实现分支无法由本计划或规格直接解释，不提交，先向用户确认。

- [ ] **Step 7: 提交首个代码里程碑并总结**

```powershell
git add resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py resonance_reversal_strategy/README.md tests/test_resonance_reversal_strategy.py
git commit -m "docs(resonance): complete JoinQuant strategy milestone guide"
git log -8 --oneline
git status --short
```

总结必须分开说明：

- 已实现范围和保护的非目标；
- pytest 运行结果；
- `py_compile` 静态编译结果；
- 尚未运行聚宽，因此没有收益结论；
- 下一步只能是用户在聚宽做短区间冒烟回测，不能直接打开验证期或调参。

---

## 规格覆盖矩阵

| 规格章节 | 实施任务 |
|---|---|
| §1-5 范围、固定参数、ETF 池、未来函数 | Task 1、Task 7、Task 8 |
| §6 指标计算与观察字段隔离 | Task 2、Task 8 |
| §7 指标事件与失效 | Task 3 |
| §8-9 共振真值表、防重复与排序 | Task 4 |
| §10-11 资金与 ATR | Task 5、Task 7 |
| §12 日内控制流 | Task 7 |
| §13-14 状态、停牌、失败与清理 | Task 5、Task 6、Task 7 |
| §15 聚宽环境 | Task 1、Task 8 |
| §16 日志与观察 | Task 2、Task 7、Task 8 |
| §17 单元测试 | Task 1-8 |
| §18 验证窗口与门槛 | Task 8 README；本里程碑不执行收益验证 |
| §19 首个实施里程碑 | Task 8 |

## 计划批准门禁

本文件获用户明确批准前，不得开始 Task 1 的策略或测试代码实现。获批后按 Task 1 → Task 8 顺序执行；每个任务必须先红后绿、通过后独立提交。任何失败实验、聚宽差异或规格歧义必须如实记录，不能通过临时调阈值绕过。
