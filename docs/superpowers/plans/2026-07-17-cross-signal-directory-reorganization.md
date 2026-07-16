# Cross-Signal 目录重组实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `cross_signal_strategy/` 整理为三个顶层 Python 入口，并把本地支持模块、研究工具、历史候选版本和平台探针分别迁入清晰的子包，同时保持策略行为完全不变。

**Architecture:** 顶层只公开聚宽正式策略、PTrade 正式策略和本地训练期回放入口。本地回放支持代码移入 `local/`，研究与诊断工具移入 `research/`，失败或已被取代的候选版本与无下单探针移入 `archive/`；所有调用方统一改用新包路径，不保留旧路径兼容转发文件。

**Tech Stack:** Python 3、pytest、PowerShell、Git。

## Global Constraints

- 只处理 `cross_signal_strategy/`、对应 `tests/test_cross_signal_*.py` 以及引用这些路径的 cross-signal 文档。
- 不修改生产多因子策略及其测试。
- 正式聚宽与 PTrade 策略文件必须逐字节不变。
- 不读取、写入、移动或删除任何市场数据。
- 所有行为变化之前先写测试，并确认测试在旧结构下按预期失败。
- 不引入未来函数，不读取验证期数据，不进行参数或规则优化。
- 不保留旧顶层模块的兼容转发文件。
- 迁移后的全部测试必须继续有效，失败实验不得静默删除。
- 目录重组最终形成一个独立提交，作为单一回滚点。

---

### Task 1: 建立目录结构契约

**Files:**
- Create: `tests/test_cross_signal_directory_structure.py`
- Verify unchanged: `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py`
- Verify unchanged: `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py`

**Interfaces:**
- Consumes: 当前 `cross_signal_strategy/` 文件树。
- Produces: 对顶层入口白名单和四类迁移目录的可执行结构契约。

- [ ] **Step 1: 记录正式策略基线状态**

Run:

```powershell
Get-FileHash -Algorithm SHA256 cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py
Get-FileHash -Algorithm SHA256 cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
git diff --exit-code HEAD -- cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
```

Expected: 两个 SHA256 值被记录，`git diff` 退出码为 `0`。

- [ ] **Step 2: 写入会失败的结构契约测试**

Create `tests/test_cross_signal_directory_structure.py` with:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ROOT = ROOT / "cross_signal_strategy"

EXPECTED_TOP_LEVEL_PYTHON = {
    "local_training_run.py",
    "smart_trade_joinquant_cross_signal_etf.py",
    "smart_trade_ptrade_cross_signal_etf.py",
}

EXPECTED_PACKAGE_FILES = {
    "local": {
        "__init__.py",
        "local_adjustment.py",
        "local_backtester.py",
        "local_data_loader.py",
        "local_data_quality.py",
        "local_order_planner.py",
        "local_signal_adapter.py",
    },
    "research": {
        "__init__.py",
        "attribution_diagnostics.py",
        "baseline_report.py",
        "boll_width_diagnostics.py",
        "breakout_extension_diagnostics.py",
        "capital_utilization_diagnostics.py",
        "cmf_diagnostics.py",
        "efficiency_ratio_diagnostics.py",
        "friction_diagnostics.py",
        "gap_execution_diagnostics.py",
        "horizontal_structure_diagnostics.py",
        "iopv_quality_diagnostics.py",
        "market_breadth_diagnostics.py",
        "multiple_testing_audit.py",
        "order_path_diagnostics.py",
        "portfolio_dependence_diagnostics.py",
        "research_budget.py",
        "sell_diagnostics.py",
        "sequence_diagnostics.py",
        "share_flow_diagnostics.py",
        "strong_trend_capacity_diagnostics.py",
        "trade_chart.py",
        "trade_diagnostics.py",
        "training_stability.py",
        "us_qdii_premium_diagnostics.py",
    },
    "archive/candidates": {
        "__init__.py",
        "backup_fill_candidate.py",
        "macd_parameter_candidate.py",
        "ranking_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_atr2_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_combo_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_pool_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_sell35_candidate.py",
        "smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py",
    },
    "archive/probes": {
        "__init__.py",
        "smart_trade_joinquant_cross_signal_etf_probe_513880.py",
        "smart_trade_joinquant_cross_signal_iopv_probe.py",
        "smart_trade_ptrade_cross_signal_iopv_probe.py",
    },
}


def test_cross_signal_top_level_exposes_only_supported_python_entries():
    actual = {path.name for path in STRATEGY_ROOT.glob("*.py")}
    assert actual == EXPECTED_TOP_LEVEL_PYTHON


def test_cross_signal_reorganized_packages_have_exact_module_inventory():
    for relative_dir, expected_files in EXPECTED_PACKAGE_FILES.items():
        directory = STRATEGY_ROOT / relative_dir
        assert directory.is_dir(), relative_dir
        actual = {path.name for path in directory.glob("*.py")}
        assert actual == expected_files, relative_dir


def test_cross_signal_archive_root_has_manifest_and_initializer():
    archive_root = STRATEGY_ROOT / "archive"
    actual = {path.name for path in archive_root.iterdir() if path.is_file()}
    assert actual == {"README.md", "__init__.py"}
```

- [ ] **Step 3: 运行测试并确认失败原因正确**

Run:

```powershell
python -m pytest tests/test_cross_signal_directory_structure.py -q
```

Expected: FAIL；原因是旧顶层仍包含支持/研究/候选/探针文件，并且目标子目录尚不存在。

---

### Task 2: 按职责迁移模块并建立包边界

**Files:**
- Move: `cross_signal_strategy/local_*.py` support modules to `cross_signal_strategy/local/`
- Move: 24 research modules to `cross_signal_strategy/research/`
- Move: 11 candidate modules to `cross_signal_strategy/archive/candidates/`
- Move: 3 probe modules to `cross_signal_strategy/archive/probes/`
- Create: `cross_signal_strategy/local/__init__.py`
- Create: `cross_signal_strategy/research/__init__.py`
- Create: `cross_signal_strategy/archive/__init__.py`
- Create: `cross_signal_strategy/archive/candidates/__init__.py`
- Create: `cross_signal_strategy/archive/probes/__init__.py`
- Create: `cross_signal_strategy/archive/README.md`

**Interfaces:**
- Consumes: Task 1 的精确文件清单。
- Produces: 目标目录树和稳定的新模块路径。

- [ ] **Step 1: 创建目标目录**

Run:

```powershell
New-Item -ItemType Directory -Force cross_signal_strategy/local
New-Item -ItemType Directory -Force cross_signal_strategy/research
New-Item -ItemType Directory -Force cross_signal_strategy/archive/candidates
New-Item -ItemType Directory -Force cross_signal_strategy/archive/probes
```

Expected: 四个目标目录存在。

- [ ] **Step 2: 使用 `git mv` 迁移逐文件清单中的模块**

Run:

```powershell
$localModules = @(
    'local_adjustment.py',
    'local_backtester.py',
    'local_data_loader.py',
    'local_data_quality.py',
    'local_order_planner.py',
    'local_signal_adapter.py'
)
$researchModules = @(
    'attribution_diagnostics.py',
    'baseline_report.py',
    'boll_width_diagnostics.py',
    'breakout_extension_diagnostics.py',
    'capital_utilization_diagnostics.py',
    'cmf_diagnostics.py',
    'efficiency_ratio_diagnostics.py',
    'friction_diagnostics.py',
    'gap_execution_diagnostics.py',
    'horizontal_structure_diagnostics.py',
    'iopv_quality_diagnostics.py',
    'market_breadth_diagnostics.py',
    'multiple_testing_audit.py',
    'order_path_diagnostics.py',
    'portfolio_dependence_diagnostics.py',
    'research_budget.py',
    'sell_diagnostics.py',
    'sequence_diagnostics.py',
    'share_flow_diagnostics.py',
    'strong_trend_capacity_diagnostics.py',
    'trade_chart.py',
    'trade_diagnostics.py',
    'training_stability.py',
    'us_qdii_premium_diagnostics.py'
)
$candidateModules = @(
    'backup_fill_candidate.py',
    'macd_parameter_candidate.py',
    'ranking_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_atr2_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_combo_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_pool_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_sell35_candidate.py',
    'smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py'
)
$probeModules = @(
    'smart_trade_joinquant_cross_signal_etf_probe_513880.py',
    'smart_trade_joinquant_cross_signal_iopv_probe.py',
    'smart_trade_ptrade_cross_signal_iopv_probe.py'
)

foreach ($name in $localModules) {
    git mv "cross_signal_strategy/$name" "cross_signal_strategy/local/$name"
    if ($LASTEXITCODE -ne 0) { throw "git mv failed: $name" }
}
foreach ($name in $researchModules) {
    git mv "cross_signal_strategy/$name" "cross_signal_strategy/research/$name"
    if ($LASTEXITCODE -ne 0) { throw "git mv failed: $name" }
}
foreach ($name in $candidateModules) {
    git mv "cross_signal_strategy/$name" "cross_signal_strategy/archive/candidates/$name"
    if ($LASTEXITCODE -ne 0) { throw "git mv failed: $name" }
}
foreach ($name in $probeModules) {
    git mv "cross_signal_strategy/$name" "cross_signal_strategy/archive/probes/$name"
    if ($LASTEXITCODE -ne 0) { throw "git mv failed: $name" }
}
```

Expected: `git status --short` 将所有迁移显示为 rename，而不是删除历史文件后另建无关文件。

- [ ] **Step 3: 添加包初始化文件**

Use these exact contents:

```python
# cross_signal_strategy/local/__init__.py
"""Cross-signal local replay support modules."""

# cross_signal_strategy/research/__init__.py
"""Cross-signal research and diagnostic modules."""

# cross_signal_strategy/archive/__init__.py
"""Archived cross-signal experiments and platform probes."""

# cross_signal_strategy/archive/candidates/__init__.py
"""Rejected or superseded cross-signal candidates."""

# cross_signal_strategy/archive/probes/__init__.py
"""No-order cross-signal platform probes."""
```

- [ ] **Step 4: 添加归档说明**

Create `cross_signal_strategy/archive/README.md` describing:

```markdown
# Cross-Signal 归档

本目录保留已否决或已被取代的训练期候选版本，以及用于定位平台能力的无下单探针。

- `candidates/`：失败实验和未提升为主线的候选策略。
- `probes/`：停牌、IOPV 等平台能力诊断脚本。

归档文件不是正式部署入口。未经新的测试优先、仅训练期研究流程，不得将其重新提升为正式版本；不得使用验证期结果选择归档候选。
```

- [ ] **Step 5: 重跑结构测试观察剩余失败**

Run:

```powershell
python -m pytest tests/test_cross_signal_directory_structure.py -q
```

Expected: 目录与文件清单测试通过；其他测试尚未运行，因为导入路径仍待更新。

---

### Task 3: 更新 Python 导入与直接文件路径

**Files:**
- Modify: `cross_signal_strategy/local_training_run.py`
- Modify: moved modules under `cross_signal_strategy/local/`, `research/`, and `archive/candidates/`
- Modify: `tests/test_cross_signal_*.py`

**Interfaces:**
- Consumes: Task 2 的新包路径。
- Produces: 所有 Python 调用方使用唯一的新路径，不依赖兼容转发模块。

- [ ] **Step 1: 机械更新点号形式的模块导入**

Run this exact PowerShell rewrite over cross-signal Python files and tests:

```powershell
$pythonFiles = @(
    Get-ChildItem cross_signal_strategy -Recurse -File -Filter '*.py'
    Get-ChildItem tests -File -Filter 'test_cross_signal_*.py'
)
$replacements = [ordered]@{
    'cross_signal_strategy.local_adjustment' = 'cross_signal_strategy.local.local_adjustment'
    'cross_signal_strategy.local_backtester' = 'cross_signal_strategy.local.local_backtester'
    'cross_signal_strategy.local_data_loader' = 'cross_signal_strategy.local.local_data_loader'
    'cross_signal_strategy.local_data_quality' = 'cross_signal_strategy.local.local_data_quality'
    'cross_signal_strategy.local_order_planner' = 'cross_signal_strategy.local.local_order_planner'
    'cross_signal_strategy.local_signal_adapter' = 'cross_signal_strategy.local.local_signal_adapter'
    'cross_signal_strategy.attribution_diagnostics' = 'cross_signal_strategy.research.attribution_diagnostics'
    'cross_signal_strategy.baseline_report' = 'cross_signal_strategy.research.baseline_report'
    'cross_signal_strategy.boll_width_diagnostics' = 'cross_signal_strategy.research.boll_width_diagnostics'
    'cross_signal_strategy.breakout_extension_diagnostics' = 'cross_signal_strategy.research.breakout_extension_diagnostics'
    'cross_signal_strategy.capital_utilization_diagnostics' = 'cross_signal_strategy.research.capital_utilization_diagnostics'
    'cross_signal_strategy.cmf_diagnostics' = 'cross_signal_strategy.research.cmf_diagnostics'
    'cross_signal_strategy.efficiency_ratio_diagnostics' = 'cross_signal_strategy.research.efficiency_ratio_diagnostics'
    'cross_signal_strategy.friction_diagnostics' = 'cross_signal_strategy.research.friction_diagnostics'
    'cross_signal_strategy.gap_execution_diagnostics' = 'cross_signal_strategy.research.gap_execution_diagnostics'
    'cross_signal_strategy.horizontal_structure_diagnostics' = 'cross_signal_strategy.research.horizontal_structure_diagnostics'
    'cross_signal_strategy.iopv_quality_diagnostics' = 'cross_signal_strategy.research.iopv_quality_diagnostics'
    'cross_signal_strategy.market_breadth_diagnostics' = 'cross_signal_strategy.research.market_breadth_diagnostics'
    'cross_signal_strategy.multiple_testing_audit' = 'cross_signal_strategy.research.multiple_testing_audit'
    'cross_signal_strategy.order_path_diagnostics' = 'cross_signal_strategy.research.order_path_diagnostics'
    'cross_signal_strategy.portfolio_dependence_diagnostics' = 'cross_signal_strategy.research.portfolio_dependence_diagnostics'
    'cross_signal_strategy.research_budget' = 'cross_signal_strategy.research.research_budget'
    'cross_signal_strategy.sell_diagnostics' = 'cross_signal_strategy.research.sell_diagnostics'
    'cross_signal_strategy.sequence_diagnostics' = 'cross_signal_strategy.research.sequence_diagnostics'
    'cross_signal_strategy.share_flow_diagnostics' = 'cross_signal_strategy.research.share_flow_diagnostics'
    'cross_signal_strategy.strong_trend_capacity_diagnostics' = 'cross_signal_strategy.research.strong_trend_capacity_diagnostics'
    'cross_signal_strategy.trade_chart' = 'cross_signal_strategy.research.trade_chart'
    'cross_signal_strategy.trade_diagnostics' = 'cross_signal_strategy.research.trade_diagnostics'
    'cross_signal_strategy.training_stability' = 'cross_signal_strategy.research.training_stability'
    'cross_signal_strategy.us_qdii_premium_diagnostics' = 'cross_signal_strategy.research.us_qdii_premium_diagnostics'
    'cross_signal_strategy.backup_fill_candidate' = 'cross_signal_strategy.archive.candidates.backup_fill_candidate'
    'cross_signal_strategy.macd_parameter_candidate' = 'cross_signal_strategy.archive.candidates.macd_parameter_candidate'
    'cross_signal_strategy.ranking_candidate' = 'cross_signal_strategy.archive.candidates.ranking_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_atr2_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_atr2_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_atr_stress_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_atr_stress_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_combo_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_combo_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_low_bounce_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_low_bounce_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_no_512100_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_no_512100_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_pool_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_pool_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_sell35_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_sell35_candidate'
    'cross_signal_strategy.smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate' = 'cross_signal_strategy.archive.candidates.smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate'
}
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($file in $pythonFiles) {
    $text = [System.IO.File]::ReadAllText($file.FullName)
    $updated = $text
    foreach ($entry in $replacements.GetEnumerator()) {
        $updated = $updated.Replace($entry.Key, $entry.Value)
    }
    if ($updated -ne $text) {
        [System.IO.File]::WriteAllText($file.FullName, $updated, $utf8NoBom)
    }
}
```

Keep `cross_signal_strategy.local_training_run` and both formal strategy imports unchanged.

- [ ] **Step 2: 更新包级候选版本导入**

In these test files, change only the candidate import package while keeping the imported symbol unchanged:

```python
# tests/test_cross_signal_combo_candidate.py
from cross_signal_strategy.archive.candidates import (
    smart_trade_joinquant_cross_signal_etf_combo_candidate as candidate,
)

# tests/test_cross_signal_low_bounce_candidate.py
from cross_signal_strategy.archive.candidates import (
    smart_trade_joinquant_cross_signal_etf_low_bounce_candidate as candidate,
)

# tests/test_cross_signal_weak_replacement_candidate.py
from cross_signal_strategy.archive.candidates import (
    smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate as candidate,
)

# tests/test_cross_signal_sell35_candidate_strategy.py
from cross_signal_strategy import smart_trade_joinquant_cross_signal_etf as mainline
from cross_signal_strategy.archive.candidates import (
    smart_trade_joinquant_cross_signal_etf_sell35_candidate as candidate,
)
```

Apply the package change to every occurrence in those four files. Do not move the mainline import.

- [ ] **Step 3: 更新候选策略直接文件路径**

Apply these exact test-path targets:

```text
tests/test_cross_signal_atr_stress_candidate_strategy.py -> cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py
tests/test_cross_signal_atr2_candidate_strategy.py       -> cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_atr2_candidate.py
tests/test_cross_signal_no_512100_candidate_strategy.py  -> cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py
tests/test_cross_signal_pool_candidate_strategy.py       -> cross_signal_strategy/archive/candidates/smart_trade_joinquant_cross_signal_etf_pool_candidate.py
```

- [ ] **Step 4: 更新探针测试路径**

Apply these exact test-path targets:

```text
tests/test_cross_signal_probe_strategy.py        -> cross_signal_strategy/archive/probes/smart_trade_joinquant_cross_signal_etf_probe_513880.py
tests/test_cross_signal_iopv_probe_strategy.py   -> cross_signal_strategy/archive/probes/smart_trade_joinquant_cross_signal_iopv_probe.py
tests/test_cross_signal_ptrade_iopv_probe.py     -> cross_signal_strategy/archive/probes/smart_trade_ptrade_cross_signal_iopv_probe.py
```

- [ ] **Step 5: 扫描并消除旧 Python 路径**

Run:

```powershell
rg -n --glob '*.py' 'cross_signal_strategy\.(local_adjustment|local_backtester|local_data_loader|local_data_quality|local_order_planner|local_signal_adapter|attribution_diagnostics|baseline_report|boll_width_diagnostics|breakout_extension_diagnostics|capital_utilization_diagnostics|cmf_diagnostics|efficiency_ratio_diagnostics|friction_diagnostics|gap_execution_diagnostics|horizontal_structure_diagnostics|iopv_quality_diagnostics|market_breadth_diagnostics|multiple_testing_audit|order_path_diagnostics|portfolio_dependence_diagnostics|research_budget|sell_diagnostics|sequence_diagnostics|share_flow_diagnostics|strong_trend_capacity_diagnostics|trade_chart|trade_diagnostics|training_stability|us_qdii_premium_diagnostics|backup_fill_candidate|macd_parameter_candidate|ranking_candidate)' cross_signal_strategy tests
```

Expected: 无匹配。

- [ ] **Step 6: 运行 cross-signal 测试**

Run:

```powershell
python -m pytest tests -q -k cross_signal
```

Expected: 全部 cross-signal 测试通过。

---

### Task 4: 更新目录说明和历史路径引用

**Files:**
- Modify: `cross_signal_strategy/README.md`
- Modify: current path references under `cross_signal_strategy/docs/`
- Verify: `cross_signal_strategy/reports/`

**Interfaces:**
- Consumes: Task 2 和 Task 3 的最终文件路径。
- Produces: 与实际目录一致的导航说明，同时保留历史实验结论。

- [ ] **Step 1: 更新主 README 文件地图**

Insert this exact section before the detailed historical file inventory:

```markdown
## Supported Entry Points And Layout

Only three Python files are supported directly from this directory:

- `smart_trade_joinquant_cross_signal_etf.py`: formal JoinQuant strategy and authoritative business logic.
- `smart_trade_ptrade_cross_signal_etf.py`: formal Guojin PTrade live adapter aligned with the JoinQuant strategy.
- `local_training_run.py`: local 2019-2021 replay entry that reuses the JoinQuant business logic; it is not an independent third strategy.

Supporting files are organized by responsibility:

- `local/`: local data, signal, order-planning, broker, adjustment, and data-quality support modules.
- `research/`: diagnostics, attribution, reporting, charting, and research-budget tools.
- `archive/candidates/`: rejected or superseded candidates retained for reproducibility.
- `archive/probes/`: no-order JoinQuant/PTrade platform probes.
- `docs/`: frozen protocols, decisions, deployment notes, and experiment records.
- `reports/`: generated research and trade-review artifacts.

Archived files are not deployment entry points and must not be promoted using validation-period results.
```

- [ ] **Step 2: 机械更新文档中的现行路径引用**

Run this exact path-only rewrite after all files have moved:

```powershell
$groups = [ordered]@{
    'local' = 'local'
    'research' = 'research'
    'archive/candidates' = 'archive/candidates'
    'archive/probes' = 'archive/probes'
}
$markdownFiles = Get-ChildItem cross_signal_strategy -Recurse -File -Filter '*.md'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$tick = [char]96
foreach ($file in $markdownFiles) {
    $text = [System.IO.File]::ReadAllText($file.FullName)
    $updated = $text
    foreach ($entry in $groups.GetEnumerator()) {
        $directory = Join-Path 'cross_signal_strategy' $entry.Value
        foreach ($module in Get-ChildItem $directory -File -Filter '*.py') {
            if ($module.Name -eq '__init__.py') { continue }
            $oldQualified = "cross_signal_strategy/$($module.Name)"
            $newQualified = "cross_signal_strategy/$($entry.Value)/$($module.Name)"
            $oldBare = "$tick$($module.Name)$tick"
            $newBare = "$tick$($entry.Value)/$($module.Name)$tick"
            $updated = $updated.Replace($oldQualified, $newQualified)
            $updated = $updated.Replace($oldBare, $newBare)
        }
    }
    if ($updated -ne $text) {
        [System.IO.File]::WriteAllText($file.FullName, $updated, $utf8NoBom)
    }
}
```

Do not rewrite experiment outcomes, return figures, dates, conclusions, or validation labels.

- [ ] **Step 3: 验证 README 与归档清单覆盖关键入口**

Run:

```powershell
rg -n 'smart_trade_joinquant_cross_signal_etf.py|smart_trade_ptrade_cross_signal_etf.py|local_training_run.py|archive/candidates|archive/probes|research/' cross_signal_strategy/README.md cross_signal_strategy/archive/README.md
```

Expected: 三个入口和四类子目录均有明确说明。

---

### Task 5: 完整验证、缓存清理与里程碑提交

**Files:**
- Verify: all files changed by Tasks 1-4
- Remove generated only: `cross_signal_strategy/**/__pycache__/`

**Interfaces:**
- Consumes: 完整迁移结果。
- Produces: 可回滚、测试通过、策略文件未变的目录重组提交。

- [ ] **Step 1: 运行完整测试套件**

Run:

```powershell
python -m pytest -q
```

Expected: 全仓测试通过。

- [ ] **Step 2: 编译全部 cross-signal Python 文件**

Run:

```powershell
Get-ChildItem cross_signal_strategy -Recurse -Filter *.py | ForEach-Object {
    python -m py_compile $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "py_compile failed: $($_.FullName)" }
}
```

Expected: 所有文件编译成功。

- [ ] **Step 3: 验证正式策略逐字节未变**

Run:

```powershell
git diff --exit-code HEAD -- cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
Get-FileHash -Algorithm SHA256 cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py
Get-FileHash -Algorithm SHA256 cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
```

Expected: `git diff` 退出码为 `0`，SHA256 与 Task 1 一致。

- [ ] **Step 4: 安全清理生成缓存**

Resolve `cross_signal_strategy/` to an absolute path. For every discovered `__pycache__` directory, verify the resolved path starts with that root, then remove only that directory using PowerShell `Remove-Item -LiteralPath ... -Recurse -Force`.

- [ ] **Step 5: 最终静态检查**

Run:

```powershell
git diff --check
git status --short
Get-ChildItem cross_signal_strategy -File -Filter *.py | Sort-Object Name | ForEach-Object Name
```

Expected: `git diff --check` 通过；顶层仅输出三个获准的 Python 入口；变更不包含多因子策略或市场数据。

- [ ] **Step 6: 提交目录重组里程碑**

Run:

```powershell
git add -- cross_signal_strategy tests/test_cross_signal_*.py
git commit -m "Reorganize cross-signal strategy modules"
```

Expected: 创建一个只包含目录重组、路径更新、结构测试和说明文档的提交。
