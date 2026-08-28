# 训练边界与交易日 Manifest V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 分离 2018--2021 日历证据窗口与 2019--2021 训练收益窗口，并对训练期末无法形成完整 H1/H3/H5 的观察执行预注册右截尾。

**Architecture:** 只读分析器先验证 schema V2 与原始 bytes hash，再按日期角色审计日志；随后仅依赖冻结 manifest 将观察划分为 `COMPLETE` 或 `RIGHT_CENSORED`，全部固定门槛只使用完整观察。聚宽策略和正式交易路径不参与本次修改。

**Tech Stack:** Python 标准库、pytest、UTF-8 JSON、SHA-256

**Spec:** [`../specs/2026-08-28-training-boundary-manifest-design.md`](../specs/2026-08-28-training-boundary-manifest-design.md)

## Global Constraints

### 范围契约

| 项目 | 冻结边界 |
|---|---|
| Target object | Manifest V2、分析器日期角色、观察完整性分类、统计样本和报告元数据 |
| Target processing stage | CLI 输入验证、日历验证、日志日期审计、样本分类和只读报告 |
| Allowed behavior change | 接受 2018 历史信号服务 2019 决策；合法期末右截尾不再作为数据损坏；完整门槛只统计完整观察 |
| Must remain unchanged | `.3/.4` 指纹、138 笔订单、23,856.40 资产、全部门槛名称与数值、CLI 名称、hash 与路径隔离规则 |
| Must not propagate to | 聚宽策略、RSI/KDJ/BOLL/ATR、订单、持仓、资金、ETF 池、PTrade、行情和 2022+ 验证数据 |

不新增布尔或模式参数。分别使用日历证据日期和训练收益日期入口，禁止一个参数同时控制两类窗口。

### 文件范围

- Modify: `resonance_reversal_strategy/research/analyze_relative_turn_observations.py`
- Modify: `tests/test_resonance_relative_turn_analysis.py`
- Modify: `resonance_reversal_strategy/README.md`
- Modify: `resonance_reversal_strategy/docs/strategy_spec.md`
- Preserve: `resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py`

---

### Task 1: Manifest V2 与日期角色

- [x] 先把测试 manifest 升为 schema V2，并增加 V1、错误覆盖、2018 信号和 2018 决策边界失败测试。
- [x] 运行目标测试，确认旧实现因 V2 和 2018 信号失败。
- [x] 实现 V2 metadata、日历证据/收益窗口两个日期入口和 evaluation session 子集。
- [x] 运行 Manifest 与日期角色测试。
- [x] 纳入本轮统一的可回滚里程碑提交。

### Task 2: 完整样本与右截尾

- [x] 先增加正式/相对右截尾、部分结果校验、完整样本缺结果和 2022 补结果失败测试。
- [x] 运行目标测试，确认旧实现把右截尾当错误。
- [x] 实现 `COMPLETE`/`RIGHT_CENSORED` 日历状态与同一完整样本统计。
- [x] 增加正式/相对注册、完整和右截尾计数，保留现有报告字段。
- [x] 运行完整分析器测试。
- [x] 纳入本轮统一的可回滚里程碑提交。

### Task 3: 文档与完整验证

- [x] 更新 README 的 V2 导出示例、hash 流程、右截尾和 CLI 说明。
- [x] 更新策略规格的训练首尾边界，不改正式交易规则。
- [x] 运行 `py_compile`、两套专用测试、全仓回归、`git diff --check` 和静态隔离检查。
- [x] 将每个改动块映射到规格，并确认聚宽策略文件无 diff。
- [x] 文档与验证纳入本轮统一里程碑；不推送、不打开验证期。

### Platform Handoff

本地完成后，用户仍需在独立聚宽研究环境导出并冻结真实 V2 manifest，重新运行 `.3/.4` 的 2019--2021 回放，核对 138 笔订单、8,808 条正式决策、730 条组合汇总和 23,856.40 期末资产，再执行只读分析器。该流程不属于本地测试已证明的结果。
