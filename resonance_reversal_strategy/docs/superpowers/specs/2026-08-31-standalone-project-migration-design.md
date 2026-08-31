# Resonance Reversal 独立项目迁移设计

## 1. 背景

`resonance_reversal_strategy` 已被确认为独立策略，后续研究、测试和实现不得读取、修改、运行或验证其他策略。当前策略虽然已有独立目录、专属测试、分支和 Git worktree，但仍位于 `D:\test\select_stocks` 仓库内，操作层面仍存在误运行全仓测试或误触其他策略文件的风险。

本设计将当前已验证内容基线 `5e81c07` 迁移为独立 Git 项目，并把本迁移设计文档的最终已提交版本一并纳入新项目。迁移只改变项目边界和开发配置，不改变策略、分析器、既有测试或既有文档内容。

## 2. 目标与非目标

### 2.1 目标

- 创建独立项目目录 `D:\test\resonance_reversal_strategy`。
- 只包含 resonance 策略、专属研究分析器、专属文档和三个专属测试文件。
- 保持现有相对路径，使策略和测试无需机械改写。
- 通过逐文件 SHA-256 对比证明迁移内容与来源提交一致。
- 默认 `pytest` 只发现 `test_resonance*.py`。
- 建立独立 Git 基线，并记录完整来源信息。
- 保持原仓库和当前 worktree 不变，作为完整历史档案和迁移回退点。

### 2.2 非目标

- 不调整任何交易规则、参数、ETF 池、ATR 行为、入场或退出逻辑。
- 不修改未来函数边界；日线信号仍只允许使用 T-1 及以前数据。
- 不修改研究结论、分析门槛、候选判定或回测口径。
- 不复制或引用其他策略代码、测试、配置或制品。
- 不复制回测日志、临时报告、训练数据、附件或缓存。
- 不重写原仓库历史，不删除原目录、分支或 worktree。
- 不创建 PTrade 版本，不执行聚宽回测。

## 3. 来源与目标

| 项目 | 固定值 |
|---|---|
| 来源仓库 | `D:\test\select_stocks` |
| 来源 worktree | `D:\test\select_stocks\.worktrees\resonance-no-atr-exit` |
| 来源分支 | `codex/resonance-no-atr-exit` |
| 策略内容基线提交 | `5e81c07` |
| 迁移设计文档 | 实施开始时的最终已提交版本 |
| 目标项目 | `D:\test\resonance_reversal_strategy` |
| Git 历史方案 | 干净基线，并记录原始提交溯源 |

实施开始前必须重新确认来源 worktree 工作区干净、`HEAD` 包含内容基线 `5e81c07`，并且本迁移设计文档已提交。目标目录必须不存在；若已经存在，必须停止并报告，不得覆盖或合并。

## 4. 目标目录结构

```text
D:\test\resonance_reversal_strategy\
├─ resonance_reversal_strategy\
│  ├─ smart_trade_joinquant_resonance_reversal_etf.py
│  ├─ research\
│  │  ├─ analyze_relative_turn_observations.py
│  │  └─ analyze_resonance_trade_risk.py
│  ├─ docs\
│  └─ README.md
├─ tests\
│  ├─ test_resonance_reversal_strategy.py
│  ├─ test_resonance_relative_turn_analysis.py
│  └─ test_resonance_trade_risk_analysis.py
├─ AGENTS.md
├─ pytest.ini
├─ .gitignore
└─ MIGRATION_SOURCE.md
```

保留仓库根目录下的同名策略子目录是有意设计：现有测试以该相对路径加载策略和分析器，保持结构可以避免与迁移目标无关的路径改写。

## 5. 文件范围

### 5.1 从内容基线原样迁移并进行哈希比对

- `resonance_reversal_strategy/**`，但不包含本迁移设计文档
- `tests/test_resonance_reversal_strategy.py`
- `tests/test_resonance_relative_turn_analysis.py`
- `tests/test_resonance_trade_risk_analysis.py`

上述文件必须从内容基线提交 `5e81c07` 读取，并在目标项目中与该提交逐文件一致。目标集合和来源集合必须双向一致：缺少文件、多出文件或任一 SHA-256 不一致都视为迁移失败。

本迁移设计文档必须从实施开始时的来源 `HEAD` 原样复制，并单独对比来源工作树与目标文件的 SHA-256。`MIGRATION_SOURCE.md` 记录该设计文档实际对应的来源提交号。

### 5.2 新项目级文件

#### `AGENTS.md`

只定义本项目规则：

- 任何实施前必须先分析、提出计划并取得确认。
- 只允许处理 resonance 策略及其专属研究、文档和测试。
- 禁止引入、读取、运行或验证其他策略。
- 严格遵守 T-1 数据边界、训练期边界和禁止验证期调参规则。
- 采用最小改动、测试先行和里程碑提交。

#### `pytest.ini`

- `testpaths = tests`
- `python_files = test_resonance*.py`

因此在项目根目录执行普通 `pytest` 时，只能发现 resonance 专属测试。

#### `.gitignore`

至少排除：

- Python 字节码和 `__pycache__`；
- `.pytest_cache`；
- 本地虚拟环境；
- `artifacts/`、`reports/`、`logs/`；
- 临时文件和编辑器缓存。

#### `MIGRATION_SOURCE.md`

记录来源仓库、worktree、分支、内容基线提交、迁移设计文档提交、迁移日期、迁移方式、验证命令和验证结果。该文件只提供溯源，不参与策略运行或研究计算。

## 6. 依赖与运行边界

聚宽策略文件继续只依赖聚宽 `jqdata`、Python 标准库、NumPy 和 Pandas，不依赖任何本地策略模块。

`analyze_resonance_trade_risk.py` 继续从同目录加载 `analyze_relative_turn_observations.py` 的交易日清单验证接口。这是 resonance 项目内部的研究依赖，只影响只读分析报告，不影响聚宽交易执行。

三个测试文件只能加载新项目中的 resonance 策略或研究分析器。项目中不得存在其他策略 Python 文件，因此测试发现和本地导入都不能触达其他策略。

## 7. 迁移流程

1. 校验来源分支、干净状态、内容基线提交祖先关系和迁移设计文档已提交状态。
2. 校验目标目录不存在且父目录为预期的 `D:\test`。
3. 创建目标目录结构。
4. 从 `5e81c07` 原样导出固定内容基线文件集合，再从来源 `HEAD` 原样复制迁移设计文档。
5. 创建四个项目级文件。
6. 对原样迁移文件执行文件集合和 SHA-256 双向比对。
7. 在新项目内执行静态检查、编译检查和 resonance 专属测试。
8. 初始化独立 Git 仓库。
9. 只暂存目标项目文件并创建基线提交：
   `chore: initialize standalone resonance project`
10. 检查提交后工作区干净，并记录基线提交号。

## 8. 验证标准

### 8.1 静态范围验证

- 目标项目不存在非 resonance 策略或测试。
- 内容基线文件集合与 `5e81c07` 中的来源集合完全一致。
- 所有内容基线文件 SHA-256 与 `5e81c07` 完全一致。
- 迁移设计文档 SHA-256 与实施时来源 `HEAD` 的文件完全一致。
- 策略 AST 不包含其他策略导入。
- Git 暂存内容全部位于目标项目内。

### 8.2 编译验证

只编译：

- `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`
- `resonance_reversal_strategy/research/analyze_relative_turn_observations.py`
- `resonance_reversal_strategy/research/analyze_resonance_trade_risk.py`

### 8.3 运行验证

在新项目根目录运行普通 `pytest`。必须只收集三个 `test_resonance*.py` 文件，且结果不得少于来源提交当前验证基线 `502 passed, 3 skipped`。若测试数量或结果不同，必须先解释差异；不得直接提交。

迁移不执行聚宽回测。现有 `.5` 回测日志和只读分析报告仍然有效，因为策略文件内容必须保持哈希一致。

## 9. 失败处理与回退

- 目标目录预先存在：停止，不覆盖。
- 来源工作区不干净或提交不符：停止，不复制。
- 文件集合或哈希不一致：停止，不初始化提交。
- 编译或测试失败：保留目标目录用于诊断，但不创建基线提交。
- Git 提交失败：不修改来源仓库；报告目标目录状态。

整个迁移不删除或修改来源仓库，因此回退方式是放弃尚未提交的新目标目录。任何删除目标目录的操作都必须另行取得用户明确授权。

## 10. 完成条件

只有同时满足以下条件才算迁移完成：

- 新项目位于已确认路径；
- 原样迁移文件哈希一致；
- 只包含 resonance 范围；
- 编译检查通过；
- resonance 专属测试达到既有验证基线；
- 独立 Git 基线提交成功；
- 新旧仓库均处于可说明状态，且来源仓库未发生迁移相关修改（设计文档提交除外）。
