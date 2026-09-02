# Late-Veto + Early-Pre-MACD 聚宽配对回测清单

## 1. 用途与边界

本清单只验证已经冻结的 JoinQuant 候选，不搜索参数、不修改规则，也不授权正式合并。
聚宽是收益、回撤和胜率的权威来源；本地判定器只负责检查证据结构和执行已批准门禁。

冻结身份：

| 角色 | 版本 | 构建号 | 指纹 | 本地文件 |
|---|---|---|---|---|
| 正式版 | `cross-v0.3.3` | `20260822.2` | `77e44d93d255` | `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py` |
| 候选版 | `cross-v0.3.3-late-veto-early-pre-macd-candidate` | `20260822.3-candidate` | `f6b08195dd3d` | `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf_late_veto_early_pre_macd_candidate.py` |

不得修改正式版或候选版的 ETF 池、指标、买卖规则、排序、仓位、ATR、持有期、执行时间
或 T-1 数据边界。验证期结果不得反向影响规则。

## 2. 固定执行顺序

严格按下列顺序运行；前一阶段未 `PASS` 时，不得运行后一阶段：

| 顺序 | `kind` | 日期 | 摩擦 |
|---:|---|---|---|
| 1 | `training_nominal` | 2019-01-01 至 2021-12-31 | nominal |
| 2 | `training_double_friction` | 2019-01-01 至 2021-12-31 | double |
| 3 | `validation_2022_2023` | 2022-01-01 至 2023-12-31 | nominal |
| 4 | `validation_2024_latest` | 2024-01-01 至统一最新截止日 | nominal |
| 5 | `validation_2015_2018` | 2015-01-01 至 2018-12-31 | nominal |
| 6 | `validation_2010_2014` | 2010-01-01 至 2014-12-31 | nominal |
| 7 | `full_period` | 不晚于 2010-01-01 开始，至与步骤 4 相同的截止日 | nominal |

步骤 4 一旦确定截止日，步骤 7 必须使用同一天。2010—2014 ETF 池不完整必须在报告中
披露，但不降低门槛。

## 3. 每一对回测的共同配置

正式版和候选版必须完全一致：

- 平台：JoinQuant；
- 初始资金：CNY 20,000；
- 频率：日级；
- 基准：`000300.XSHG`；
- 执行：策略 `run_daily(do_trading, time="09:35")`；
- `use_real_price=True`；
- `avoid_future_data=True`；
- 起止日期、运行时平台设置及数据版本一致。

摩擦配置：

| profile | open/close commission | minimum commission | price-related slippage |
|---|---:|---:|---:|
| `nominal` | 0.0003 | 5 | 0.001 |
| `double` | 0.0006 | 10 | 0.002 |

`training_double_friction` 使用两个临时聚宽副本，只允许把
`PriceRelatedSlippage(0.001)` 改为 `0.002`，把开仓/平仓佣金 `0.0003` 改为
`0.0006`，把最低佣金 `5` 改为 `10`。不得修改任何其他代码。临时摩擦副本不提交到仓库，
也不构成新策略候选。

## 4. 单阶段配对操作

1. 在聚宽分别建立正式版和候选版回测副本。
2. 从上表所列本地文件复制完整源码；不要使用旧的聚宽草稿或历史候选。
3. 对双倍摩擦阶段只执行第 3 节列出的三项成本替换。
4. 设置完全相同的日期、资金、频率和平台选项。
5. 运行正式版，再运行候选版。
6. 从初始化日志核对版本、构建号和业务指纹；缺少或不一致则整对作废。
7. 导出完整日志和完整成交明细。不能只保存页面截图或汇总数字。
8. 保存原始精度的总收益、最大回撤、已平仓胜率、盈亏比及逐年收益。

页面、日志、成交导出之间存在数量矛盾时，先查明口径或数据原因；原因未明确前不能填入
权威结果。

## 5. 证据目录与哈希

每个 `kind` 使用独立目录，文件名固定为：

- `formal.log`
- `formal-trades.csv`
- `candidate.log`
- `candidate-trades.csv`

在当前工作树根目录执行，下面以训练名义阶段为例：

```powershell
$gateKind = 'training_nominal'
$strategyRoot = (Resolve-Path -LiteralPath 'cross_signal_strategy').Path
$evidenceRoot = Join-Path $strategyRoot "reports\evidence\late_veto_early_pre_macd\$gateKind"
New-Item -ItemType Directory -Force -Path $evidenceRoot
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $evidenceRoot 'formal.log')
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $evidenceRoot 'formal-trades.csv')
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $evidenceRoot 'candidate.log')
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $evidenceRoot 'candidate-trades.csv')
```

先将聚宽导出的四个文件保存到该目录，再计算哈希。不得把证据写入任何只读行情数据根。
JSON 中分别记录完整的 64 位 SHA-256，不接受文件名、截图或截断哈希替代。

### 5.1 仅拒绝的成交导出例外

2026-09-02 用户明确同意本次训练首关可以不提供独立成交明细。该例外不改变完整证据标准，且只允许同时满足以下条件时关闭候选：

- `kind` 必须是 `training_nominal`；
- 正式版和候选版的完整日志均存在并有完整 SHA-256；
- 正式版和候选版的聚宽收益概述截图均存在并有完整 SHA-256；
- 双方 `trade_export_sha256` 均缺失，不能单边混用证据；
- 收益下降、最大回撤上升或胜率未达到预登记增幅中的至少一项已经形成决定性失败。

此时 JSON 必须使用 `evidence_mode: rejection_only_without_trade_exports` 并填写 `evidence_limitation`。门禁只允许返回 `FAIL`；若页面核心指标可能通过、用于其他阶段或试图据此采用候选，证据必须返回 `INVALID`。这不是后续候选的默认放宽。

## 6. 指标填写口径

从模板复制一份阶段结果：

```powershell
Copy-Item -LiteralPath 'cross_signal_strategy\reports\templates\late_veto_early_pre_macd_pair.json' -Destination 'cross_signal_strategy\reports\late_veto_early_pre_macd_training_nominal_pair.json'
```

编辑副本时必须删除顶层 `"example_only": true`，并填写真实 `kind`、配置、哈希及指标。
CLI 会拒绝仍标记为示例的文件。

- `total_return` 使用倍率形式：例如 129.25% 写成 `1.2925`；优先填写导出中的更高精度值。
- `max_drawdown`、`win_rate` 使用 0—1 小数；`annual_returns` 使用小数且允许负值；均不填百分号。
- `win_rate` 沿用聚宽已平仓交易口径；未平仓持仓不得进入分母。
- `closed_trade_count` 是用于该胜率的已平仓样本数，不是卖出日志数或订单数。
- `profit_loss_ratio` 使用聚宽同一已平仓口径；无法取得时填 `null`，训练门禁会失败。
- `positive_to_negative_round_trips` 的定义固定为：持有期间 `holding_mfe > 0`，最终
  `realized_return_pct < 0` 的已平仓交易数量。正式版和候选版必须用同一成交匹配及价格口径。
- `early_fills` 只统计真实成交。`[early-pre-macd]` 或
  `[buy] channel=early_pre_macd` 日志表示候选尝试，必须再与同日、同代码、正成交量的成交明细
  匹配；未成交订单不计入。
- `early_fill_years` 从这些真实 early 成交日期提取、去重并升序填写。
- 正式版没有 early 通道，`early_fills` 为 `0`，`early_fill_years` 为 `[]`。

训练期还必须核对 2019、2020、2021 三个年度收益均严格大于 0。

## 7. 执行门禁

运行：

```powershell
python -m cross_signal_strategy.research.late_veto_early_pre_macd_gate cross_signal_strategy\reports\late_veto_early_pre_macd_training_nominal_pair.json
```

结果含义：

- `PASS`，退出码 0：允许进入清单中的下一阶段。
- `FAIL`，退出码 1：指标证据有效但门禁失败；立即停止，记录并拒绝候选，不得调参重跑。
- `INVALID`，退出码 2：身份、配置、窗口、成本、哈希或字段无效；修正证据问题并重新运行
  整个无效配对，不能改变策略规则。

训练名义阶段额外要求：

- 收益不低于正式版；
- 最大回撤不高于正式版；
- 胜率至少高 3 个百分点，且严格高于失败的 standalone late-veto 55.8%；
- early 实际成交不少于 3 笔，覆盖至少 2 年；
- 正转负交易数不增加；
- 盈亏比不低于 3；
- 三个训练年度收益均为正。

训练双倍摩擦及四个验证窗口要求收益、最大回撤、胜率逐项非劣。最终全周期要求收益和
回撤非劣、胜率至少高 3 个百分点。

## 8. 停止与归档

- 任一阶段 `FAIL`：停止，不运行后续窗口；候选作为一个整体只记录一次失败实验。
- 任一阶段 `INVALID`：不得用另一窗口或本地收益代替；只重新生成无效配对证据。
- 本地批准的 `G:` 数据根不存在：报告本地数据验证阻塞，不换用验证期或其他行情根。
- 全部阶段 `PASS`：状态仅为 `all_gates_passed_pending_adoption`。不得自动修改正式 JoinQuant
  或 PTrade 文件，必须另行提交正式采纳设计并取得用户确认。

每个实际运行阶段的 Markdown 报告应列出配对配置、原始指标、哈希、事件/成交计数、门禁
输出以及已运行和未运行阶段。不要把静态检查、本地测试或 PTrade 回测表述为聚宽收益证明。
