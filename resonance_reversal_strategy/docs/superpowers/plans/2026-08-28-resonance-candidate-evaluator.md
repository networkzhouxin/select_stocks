# Resonance Candidate Evaluator Implementation Plan

> **执行要求：** 当前会话使用 `superpowers:executing-plans` 逐项实施。只有用户另外明确要求委派时，才可改用 `superpowers:subagent-driven-development`。步骤使用 `- [ ]` 跟踪。

**Goal:** 建立只读、可复现且能正确处理 ETF 份额变化的 `resonance_reversal` 候选绩效评估器。

**Architecture:** 分三层实现：日志解析层只生成不可变成交和净值记录；交易闭环层按独立买卖现金流配对；报告层计算冻结指标和门槛。分析器只读显式日志与 schema V2 manifest，不读取行情，也不参与策略运行。

**Tech Stack:** Python 3 标准库（`argparse`、`dataclasses`、`datetime`、`html`、`json`、`math`、`pathlib`、`re`、`statistics`、`tempfile`）、pytest。

**Spec:** `resonance_reversal_strategy/docs/superpowers/specs/2026-08-28-resonance-quality-candidate-program-design.md`

## Global Constraints

- 训练收益窗口固定为 `2019-01-01..2021-12-31`，日历证据窗口固定为 `2018-01-01..2021-12-31`。
- 初始资金固定为 20,000 元，基准收益固定为 64.10%，对照总收益/胜率/最大回撤固定为 129.25%/55.8%/6.28%。
- 不读取行情，不读取 2022+，不搜索参数、ETF、窗口或阈值。
- 买卖数量必须分别进入现金流，不能以最小数量配对。
- 平台盈亏比与本地 Profit Factor 分开报告。
- 输入日志、manifest 和输出必须是不同物理文件；任何输入错误 fail closed。

## 测试辅助契约

`tests/test_resonance_candidate_performance.py` 顶部按现有相对分析器测试的模式加载模块，并一次性定义后续片段使用的工厂，禁止让每个任务各造一套记录模型：

```python
import importlib.util
import json
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_candidate_performance.py"
)
spec = importlib.util.spec_from_file_location(
    "candidate_performance_analyzer", ANALYZER_PATH,
)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def write_structured_log(tmp_path, payload, name="structured.log"):
    path = tmp_path / name
    path.write_text(
        "2021-01-04 15:30:00 - INFO - "
        + json.dumps(payload, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def fill(day, code, side, price, amount, commission):
    timestamp = datetime.strptime(day + " 09:35:00", "%Y-%m-%d %H:%M:%S")
    return analyzer.Fill(
        timestamp, timestamp.date(), code, side,
        float(price), int(amount), float(commission),
    )


def portfolio_points(values):
    start = datetime(2021, 1, 4)
    return tuple(
        analyzer.PortfolioPoint(
            (start + timedelta(days=index)).date(),
            float(value), 0.0, {},
        )
        for index, value in enumerate(values)
    )


def completed_trade(pnl, return_rate):
    entry = fill("2021-01-04", "510300.XSHG", "BUY", 10.0, 100, 0.0)
    exit_fill = fill("2021-01-05", "510300.XSHG", "SELL", 10.0, 100, 0.0)
    return analyzer.CompletedTrade(
        entry, exit_fill, float(pnl), float(return_rate), 1.0,
    )


def completed_trades(return_rates):
    return tuple(
        completed_trade(1000.0 * value, value)
        for value in return_rates
    )


def completed_pnl(pnl_values):
    return tuple(
        completed_trade(value, value / 1000.0)
        for value in pnl_values
    )


def passing_candidate_metrics():
    return {
        "total_return": 1.30,
        "win_rate": 0.60,
        "wilson_lower_95": 0.51,
        "max_drawdown": 0.06,
        "closed_trade_count": 80,
        "median_trade_return": 0.01,
        "top_10pct_gross_profit_share": 0.49,
    }
```

---

### Task 1: Parse immutable fills and portfolio summaries

**Files:**
- Create: `resonance_reversal_strategy/research/analyze_candidate_performance.py`
- Create: `tests/test_resonance_candidate_performance.py`

**Interfaces:**
- Produces: `Fill(timestamp, trade_date, code, side, price, amount, commission)`。
- Produces: `PortfolioPoint(closing_date, total_value, available_cash, positions)`。
- Produces: `parse_joinquant_log(paths) -> ParsedLog`。

- [ ] **Step 1: Write failing parser tests**

```python
def test_parse_fill_decodes_html_and_preserves_independent_amounts(tmp_path):
    log = tmp_path / "candidate.log"
    log.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 5.065, "
        "amount:1600, commission: 5.0\n"
        "2021-06-25 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=close) trade price: 1.232, "
        "amount:6400, commission: 5.0\n",
        encoding="utf-8",
    )

    parsed = analyzer.parse_joinquant_log([log])

    assert [fill.amount for fill in parsed.fills] == [1600, 6400]
    assert [fill.side for fill in parsed.fills] == ["BUY", "SELL"]


def test_parse_portfolio_summary_rejects_nonfinite_total_value(tmp_path):
    log = write_structured_log(tmp_path, {
        "event": "portfolio_summary",
        "closing_date": "2021-01-04",
        "total_value": float("nan"),
        "available_cash": 1000.0,
        "positions": {},
    })

    with pytest.raises(ValueError, match="finite total_value"):
        analyzer.parse_joinquant_log([log])
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run:

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -k "parse_fill or parse_portfolio" -q
```

Expected: FAIL because the module and parser do not exist.

- [ ] **Step 3: Implement exact record types and parsers**

```python
@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    trade_date: date
    code: str
    side: str
    price: float
    amount: int
    commission: float


FILL_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"security=(?P<code>\S+).*action=(?P<action>open|close).*"
    r"trade price:\s*(?P<price>[0-9.]+),\s*"
    r"amount:\s*(?P<amount>\d+),\s*"
    r"commission:\s*(?P<commission>[0-9.]+)"
)


def _parse_fill(line):
    match = FILL_RE.search(html.unescape(line))
    if match is None:
        return None
    price = float(match.group("price"))
    amount = int(match.group("amount"))
    commission = float(match.group("commission"))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("fill price must be finite and positive")
    if amount <= 0:
        raise ValueError("fill amount must be positive")
    if not math.isfinite(commission) or commission < 0:
        raise ValueError("fill commission must be finite and nonnegative")
    timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
    return Fill(
        timestamp=timestamp,
        trade_date=timestamp.date(),
        code=match.group("code"),
        side="BUY" if match.group("action") == "open" else "SELL",
        price=price,
        amount=amount,
        commission=commission,
    )
```

Structured JSON parsing must use `html.unescape` followed by `json.loads` and reject duplicate portfolio dates, decreasing dates, non-finite money and malformed positions.

- [ ] **Step 4: Run parser tests and the existing relative analyzer tests**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -q
python -m pytest tests/test_resonance_relative_turn_analysis.py -q
```

Expected: both suites PASS.

- [ ] **Step 5: Commit the parser milestone**

```powershell
git add resonance_reversal_strategy/research/analyze_candidate_performance.py tests/test_resonance_candidate_performance.py
git commit -m "test(resonance): add candidate log parser"
```

### Task 2: Pair completed trades with corporate-action-safe cash flow

**Files:**
- Modify: `resonance_reversal_strategy/research/analyze_candidate_performance.py`
- Modify: `tests/test_resonance_candidate_performance.py`

**Interfaces:**
- Consumes: ordered `Fill` records from Task 1。
- Produces: `CompletedTrade(entry, exit, pnl, return_rate, amount_ratio)`。
- Produces: `pair_completed_trades(fills) -> TradeLedger`。

- [ ] **Step 1: Write failing cash-flow and anomaly tests**

```python
def test_pair_trade_uses_sell_amount_after_etf_split():
    ledger = analyzer.pair_completed_trades([
        fill("2021-06-18", "159928.XSHE", "BUY", 5.065, 1600, 5.0),
        fill("2021-06-25", "159928.XSHE", "SELL", 1.232, 6400, 5.0),
    ])

    trade = ledger.completed[0]
    assert trade.pnl == pytest.approx(-229.2)
    assert trade.return_rate == pytest.approx(-229.2 / 8109.0)
    assert trade.amount_ratio == pytest.approx(4.0)


@pytest.mark.parametrize("fills, message", [
    ([fill("2021-01-04", "510300.XSHG", "SELL", 5, 100, 5)], "sell without open"),
    ([fill("2021-01-04", "510300.XSHG", "BUY", 5, 100, 5),
      fill("2021-01-05", "510300.XSHG", "BUY", 5, 100, 5)], "duplicate open"),
])
def test_pair_trade_fails_closed_on_ambiguous_path(fills, message):
    with pytest.raises(ValueError, match=message):
        analyzer.pair_completed_trades(fills)
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -k "pair_trade" -q
```

Expected: FAIL because `pair_completed_trades` is missing.

- [ ] **Step 3: Implement one-open-position-per-code pairing**

```python
def _close_trade(entry, exit_fill):
    entry_cost = entry.price * entry.amount + entry.commission
    exit_proceeds = (
        exit_fill.price * exit_fill.amount - exit_fill.commission
    )
    pnl = exit_proceeds - entry_cost
    return CompletedTrade(
        entry=entry,
        exit=exit_fill,
        pnl=pnl,
        return_rate=pnl / entry_cost,
        amount_ratio=exit_fill.amount / entry.amount,
    )


def pair_completed_trades(fills):
    open_by_code = {}
    completed = []
    for fill_record in fills:
        if fill_record.side == "BUY":
            if fill_record.code in open_by_code:
                raise ValueError("duplicate open: %s" % fill_record.code)
            open_by_code[fill_record.code] = fill_record
            continue
        entry = open_by_code.pop(fill_record.code, None)
        if entry is None:
            raise ValueError("sell without open: %s" % fill_record.code)
        completed.append(_close_trade(entry, fill_record))
    return TradeLedger(tuple(completed), tuple(open_by_code.values()))
```

Do not normalize quantities before `_close_trade`.

- [ ] **Step 4: Run the full new test file**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the trade-ledger milestone**

```powershell
git add resonance_reversal_strategy/research/analyze_candidate_performance.py tests/test_resonance_candidate_performance.py
git commit -m "feat(resonance): reconstruct candidate trade cash flows"
```

### Task 3: Compute frozen metrics and hard gates

**Files:**
- Modify: `resonance_reversal_strategy/research/analyze_candidate_performance.py`
- Modify: `tests/test_resonance_candidate_performance.py`

**Interfaces:**
- Consumes: `TradeLedger`, ordered `PortfolioPoint` records, initial capital, benchmark return。
- Produces: `summarize_performance(parsed_log, initial_capital=20000.0) -> dict`。
- Produces: `evaluate_final_gates(candidate, double_friction) -> dict[str, bool]`。

- [ ] **Step 1: Write failing metric tests**

```python
def test_metrics_use_daily_equity_for_drawdown_and_wilson_lower_bound():
    points = portfolio_points([20000, 22000, 19800, 23000])
    trades = completed_trades([0.10, -0.05, 0.03, 0.02])

    report = analyzer.summarize_performance_from_records(points, trades, 20000.0)

    assert report["total_return"] == pytest.approx(0.15)
    assert report["max_drawdown"] == pytest.approx(0.10)
    assert report["closed_trade_count"] == 4
    assert report["win_rate"] == pytest.approx(0.75)
    assert 0 < report["wilson_lower_95"] < report["win_rate"]


def test_top_ten_percent_uses_all_completed_trades_as_denominator():
    trades = completed_pnl([100, 90, 80, 70, 60, 50, 40, 30, 20, -10])
    summary = analyzer.summarize_trades(trades)
    assert summary["top_10pct_trade_count"] == 1
    assert summary["top_10pct_gross_profit_share"] == pytest.approx(100 / 540)


def test_final_gates_fail_closed_without_double_friction_report():
    gates = analyzer.evaluate_final_gates(passing_candidate_metrics(), None)
    assert gates["double_friction_beats_benchmark"] is False
    assert gates["all_passed"] is False
```

- [ ] **Step 2: Run metric tests and verify RED**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -k "metric or top_ten or final_gates" -q
```

Expected: FAIL because metric functions are missing.

- [ ] **Step 3: Implement formulas with fixed constants**

```python
INITIAL_CAPITAL = 20000.0
FINAL_TOTAL_RETURN_GATE = 1.2925
FINAL_WIN_RATE_GATE = 0.558
FINAL_MAX_DRAWDOWN_GATE = 0.0628
FINAL_MIN_CLOSED_TRADES = 80
FINAL_MAX_TOP_PROFIT_SHARE = 0.50
BENCHMARK_RETURN = 0.6410


def wilson_lower_bound(wins, total, z=1.96):
    if total <= 0:
        return None
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half


def max_drawdown(values):
    peak = None
    worst = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst
```

Use `math.ceil(len(completed_trades) * 0.10)` for the top-trade count and divide those positive P&Ls by gross positive P&L. If gross positive P&L is not positive, set the share to `None` and fail the gate.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the metric milestone**

```powershell
git add resonance_reversal_strategy/research/analyze_candidate_performance.py tests/test_resonance_candidate_performance.py
git commit -m "feat(resonance): add frozen candidate quality gates"
```

### Task 4: Add manifest-bound CLI, atomic report output, and documentation

**Files:**
- Modify: `resonance_reversal_strategy/research/analyze_candidate_performance.py`
- Modify: `tests/test_resonance_candidate_performance.py`
- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`

**Interfaces:**
- Consumes CLI: `--baseline-log`, `--candidate-log`, `--double-friction-log`, `--expected-baseline-build`, `--expected-candidate-build`, `--session-calendar`, `--session-calendar-sha256`, `--output`。
- Produces UTF-8 JSON report written by same-directory temporary file plus `os.replace`。

- [ ] **Step 1: Write failing CLI and path-alias tests**

```python
def test_cli_requires_distinct_double_friction_log_and_output(tmp_path):
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    calendar = tmp_path / "calendar.json"
    output = tmp_path / "report.json"
    for path in (baseline, candidate, calendar):
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be distinct"):
        analyzer.main([
            "--baseline-log", str(baseline),
            "--candidate-log", str(candidate),
            "--double-friction-log", str(candidate),
            "--expected-candidate-build", "20260828.TEST",
            "--session-calendar", str(calendar),
            "--session-calendar-sha256", "0" * 64,
            "--output", str(output),
        ])


def test_cli_report_contains_manifest_and_all_frozen_gates(
        tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    double_friction = tmp_path / "double.log"
    calendar = tmp_path / "calendar.json"
    output = tmp_path / "report.json"
    for path in (baseline, candidate, double_friction, calendar):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer,
        "read_session_calendar_manifest_bytes",
        lambda path: b"{}",
    )
    monkeypatch.setattr(
        analyzer,
        "validate_session_calendar_manifest",
        lambda raw, digest: {"schema_version": 2},
    )
    expected_report = {
        "session_calendar": {"schema_version": 2},
        "run_identity": {
            "baseline_build": "20260827.4",
            "candidate_build": "20260828.TEST",
            "double_friction_build": "20260828.TEST",
        },
        "gates": {
            "total_return_above_cross": True,
            "win_rate_above_cross": True,
            "wilson_lower_above_half": True,
            "max_drawdown_below_cross": True,
            "closed_trades_at_least_80": True,
            "median_trade_return_positive": True,
            "top_profit_share_at_most_half": True,
            "double_friction_beats_benchmark": True,
            "all_passed": True,
        },
    }
    monkeypatch.setattr(
        analyzer, "analyze_paths",
        lambda args, manifest: expected_report,
    )

    exit_code = analyzer.main([
        "--baseline-log", str(baseline),
        "--candidate-log", str(candidate),
        "--double-friction-log", str(double_friction),
        "--expected-candidate-build", "20260828.TEST",
        "--session-calendar", str(calendar),
        "--session-calendar-sha256", "0" * 64,
        "--output", str(output),
    ])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["session_calendar"]["schema_version"] == 2
    assert report["run_identity"] == {
        "baseline_build": "20260827.4",
        "candidate_build": "20260828.TEST",
        "double_friction_build": "20260828.TEST",
    }
    assert set(report["gates"]) >= {
        "total_return_above_cross",
        "win_rate_above_cross",
        "wilson_lower_above_half",
        "max_drawdown_below_cross",
        "closed_trades_at_least_80",
        "median_trade_return_positive",
        "top_profit_share_at_most_half",
        "double_friction_beats_benchmark",
        "all_passed",
    }
```

- [ ] **Step 2: Run CLI tests and verify RED**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py -k "cli" -q
```

Expected: FAIL until CLI validation and output are implemented.

- [ ] **Step 3: Implement CLI by reusing the existing manifest validator**

```python
from analyze_relative_turn_observations import (
    read_session_calendar_manifest_bytes,
    validate_session_calendar_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", action="append", required=True)
    parser.add_argument("--candidate-log", action="append", required=True)
    parser.add_argument("--double-friction-log", action="append", required=True)
    parser.add_argument(
        "--expected-baseline-build", default="20260827.4",
    )
    parser.add_argument("--expected-candidate-build", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--session-calendar-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = validate_session_calendar_manifest(
        read_session_calendar_manifest_bytes(args.session_calendar),
        args.session_calendar_sha256,
    )
    report = analyze_paths(args, manifest)
    write_json_atomically(args.output, report)
    return 0
```

`analyze_paths` 必须精确验证：基线初始化 build 等于 `expected_baseline_build`，普通候选与双摩擦日志初始化 build 都等于 `expected_candidate_build`；初始化缺失、重复或不一致均 fail closed。文档中给出一条使用四个不同物理文件的 PowerShell 命令，并明确缺少双摩擦证据时最终验收失败。

- [ ] **Step 4: Run all targeted and strategy tests**

```powershell
python -m pytest tests/test_resonance_candidate_performance.py tests/test_resonance_relative_turn_analysis.py -q
python -m pytest tests/test_resonance_reversal_strategy.py -q
```

Expected: all PASS.

- [ ] **Step 5: Run the current `.4` baseline reconstruction as a non-gating live check**

Expected evidence to report separately from unit tests:

```text
final_asset=23856.40
total_return=0.19282
closed_trades=68
wins=39
losses=29
win_rate=0.573529
median_trade_return=0.023296
profit_factor=1.337813
159928 split trade pnl=-229.20, amount_ratio=4.0
```

Do not copy the 27--49 MB user logs into tests or the repository.

- [ ] **Step 6: Commit the evaluator milestone**

```powershell
git add resonance_reversal_strategy/research/analyze_candidate_performance.py tests/test_resonance_candidate_performance.py resonance_reversal_strategy/README.md resonance_reversal_strategy/docs/strategy_spec.md
git commit -m "feat(resonance): add reproducible candidate evaluator"
```
