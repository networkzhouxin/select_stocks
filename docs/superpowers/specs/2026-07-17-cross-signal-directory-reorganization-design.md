# Cross-Signal 目录重组设计

## 目标

整理 `cross_signal_strategy/`，清晰呈现三个受支持的入口：

1. 聚宽部署策略。
2. 国金证券 PTrade 部署策略。
3. 本地训练期回放入口。

本次整理必须保留所有历史实验、诊断工具、探针、报告、测试和 Git 历史，
不得改变策略规则、ETF 池、参数、数据边界或回测结果。

## 重要区别

本地回放不是第三套独立的策略实现。它通过本地数据适配器、订单规划器和
经纪商模拟复用已冻结的聚宽业务逻辑。`local_training_run.py` 继续保留在
顶层，作为公开的本地回放入口；其支持模块移入专用包。

## 目标目录结构

```text
cross_signal_strategy/
|-- README.md
|-- smart_trade_joinquant_cross_signal_etf.py
|-- smart_trade_ptrade_cross_signal_etf.py
|-- local_training_run.py
|-- local/
|   |-- __init__.py
|   |-- local_adjustment.py
|   |-- local_backtester.py
|   |-- local_data_loader.py
|   |-- local_data_quality.py
|   |-- local_order_planner.py
|   `-- local_signal_adapter.py
|-- research/
|   |-- __init__.py
|   `-- 诊断和报告工具
|-- archive/
|   |-- README.md
|   |-- __init__.py
|   |-- candidates/
|   |   |-- __init__.py
|   |   `-- 已否决或已被取代的策略候选版本
|   `-- probes/
|       |-- __init__.py
|       `-- 临时、无下单行为的平台探针
|-- docs/
`-- reports/
```

因此，顶层 Python 文件白名单严格限定为：

- `smart_trade_joinquant_cross_signal_etf.py`
- `smart_trade_ptrade_cross_signal_etf.py`
- `local_training_run.py`

## 文件分类

### 本地回放支持模块

将以下六个 `local_*` 支持模块移入 `local/`，同时保留顶层入口
`local_training_run.py`：

- `local_adjustment.py`
- `local_backtester.py`
- `local_data_loader.py`
- `local_data_quality.py`
- `local_order_planner.py`
- `local_signal_adapter.py`

同时更新入口文件、研究工具和测试中的导入路径，使其指向新的包路径。

### 研究与诊断工具

将非策略分析工具移入 `research/`，包括基线与交易报告、归因与稳定性工具、
摩擦成本与资金利用率诊断、技术指标诊断、数据质量诊断、研究预算审计和图表
生成工具。这些文件迁移后仍需可执行并继续接受测试。

研究模块的完整清单如下：

- `attribution_diagnostics.py`
- `baseline_report.py`
- `boll_width_diagnostics.py`
- `breakout_extension_diagnostics.py`
- `capital_utilization_diagnostics.py`
- `cmf_diagnostics.py`
- `efficiency_ratio_diagnostics.py`
- `friction_diagnostics.py`
- `gap_execution_diagnostics.py`
- `horizontal_structure_diagnostics.py`
- `iopv_quality_diagnostics.py`
- `market_breadth_diagnostics.py`
- `multiple_testing_audit.py`
- `order_path_diagnostics.py`
- `portfolio_dependence_diagnostics.py`
- `research_budget.py`
- `sell_diagnostics.py`
- `sequence_diagnostics.py`
- `share_flow_diagnostics.py`
- `strong_trend_capacity_diagnostics.py`
- `trade_chart.py`
- `trade_diagnostics.py`
- `training_stability.py`
- `us_qdii_premium_diagnostics.py`

### 归档候选版本

将所有已否决或已被取代的候选实现移入 `archive/candidates/`，其中包括本地
实验候选模块和临时聚宽候选策略。归档不等于删除：相关测试继续保留，以便
失败实验仍然可以复现。

候选版本的完整清单如下：

- `backup_fill_candidate.py`
- `macd_parameter_candidate.py`
- `ranking_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_atr2_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_atr_stress_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_combo_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_low_bounce_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_no_512100_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_pool_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_sell35_candidate.py`
- `smart_trade_joinquant_cross_signal_etf_weak_replacement_candidate.py`

### 归档探针

将 513880 停牌探针以及聚宽/PTrade IOPV 能力探针移入 `archive/probes/`。
它们继续作为无下单行为的诊断产物保留，不属于任何正式部署策略。

探针的完整清单如下：

- `smart_trade_joinquant_cross_signal_etf_probe_513880.py`
- `smart_trade_joinquant_cross_signal_iopv_probe.py`
- `smart_trade_ptrade_cross_signal_iopv_probe.py`

### 测试、文档与报告

- 按照用户已经确认的边界，将全部 46 个 cross-signal 测试继续保留在仓库级
  `tests/` 目录中。
- 更新测试中的导入路径和直接文件路径，使其指向迁移后的模块。
- 保留 `docs/` 和 `reports/` 两个专用目录。
- 更新现行文档中的路径引用，但保留历史实验记录的实质内容和结论。
- 新增归档清单，说明各归档类别，并明确：未经新的训练期专用研究流程，不得
  将归档代码提升为正式版本。

## 迁移规则

1. 对已跟踪文件使用 `git mv`，使历史记录保持可追溯。
2. 仅在稳定导入所必需的位置添加包初始化文件。
3. 不在旧的顶层路径保留兼容转发模块，否则会重新造成这次整理要解决的混乱。
4. 除迁移所必需的导入和路径调整外，不修改函数体。
5. 不读取、修改或删除任何市场数据目录。
6. 只有在确认解析后的实际路径位于工作区内部后，才删除自动生成的
   `cross_signal_strategy/__pycache__/` 目录。

## 测试优先的迁移流程

移动正式策略或研究文件之前，先新增一个会在旧目录结构下失败的结构契约测试。
该测试必须断言：顶层严格遵守三个 Python 文件的白名单，并且 `archive/`、
`research/` 和 `local/` 中存在规定的包与文件。然后再执行文件迁移和导入路径
更新，直至新测试与全部现有测试通过。

验证顺序如下：

1. 确认结构契约测试在旧目录结构下失败。
2. 迁移文件并更新导入和路径引用。
3. 运行全部 cross-signal 测试。
4. 运行仓库完整测试套件。
5. 对三个入口文件及全部迁移后的 Python 模块运行 `py_compile`。
6. 运行 `git diff --check`，并确认只有预期目录受到影响。
7. 清理生成的字节码缓存，并确认最终目录树符合设计。

## 验收标准

- `cross_signal_strategy/` 顶层只保留三个获准的 Python 入口文件。
- 正式聚宽和 PTrade 策略文件在本次迁移中保持逐字节不变。
- 本地回放、研究工具、候选版本和探针迁移后仍然可以导入。
- 全部现有测试和新增结构测试通过。
- 不改动生产多因子策略文件。
- 不为调参读取任何市场数据，也不以任何方式修改市场数据文件。
- 最终提交为整个目录重组提供一个单一、清晰的回滚点。
