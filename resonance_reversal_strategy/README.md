# RSI-KDJ-BOLL 共振反转 ETF 策略

本目录是一套独立的新聚宽策略，未替代、导入或修改仓库中的
`cross_signal_strategy`、多因子策略、V15 系列或 PTrade 策略。当前版本是首个代码
里程碑，只用于本地单元验证以及后续聚宽回测验证；它不包含收益结论、实盘授权、
PTrade 版本或本地收益回放器。

## 策略摘要

策略不计算综合分数。BOLL 必须提供价格结构方向，RSI14 或 KDJ 至少一个提供同向
动能拐点；若第三个指标出现反向事件，则否决该次共振。事件有效窗口为两个交易日，
并且至少一个支持事件必须发生在 T-1。ATR14 只负责持仓风控，不参与共振资格、候选
排序或仓位计算。

固定 ETF 池如下：

```python
[
    "510300.XSHG",  # 沪深300
    "159915.XSHE",  # 创业板
    "512100.XSHG",  # 中证1000
    "159928.XSHE",  # 消费ETF
    "510880.XSHG",  # 红利ETF
    "513100.XSHG",  # 纳指ETF
    "513500.XSHG",  # 标普500ETF
    "159920.XSHE",  # 恒生ETF
    "513880.XSHG",  # 日经ETF
    "513050.XSHG",  # 中概互联ETF
    "518880.XSHG",  # 黄金ETF
    "159985.XSHE",  # 豆粕ETF
]
```

首版固定参数：日线回看 120 根、RSI14、KDJ(9,3,3)、BOLL(20,2)、ATR14、
ATR 移动止损倍数 2.5、止损百分比限制 5% 至 15%、最多持有 3 只、总目标仓位
95%。RSI6/12/24、ADX/+DI/-DI、成交量、量比、BOLL 带宽和中轨斜率仅记录观察，
不参与交易。K-D 差值由负转正或由正转负形成的正式金叉/死叉也只写入观察日志，
不会替代预交叉拐点或成为交易条件。

## 数据和执行边界

- T 日 09:35 取得 `prev_date`，所有 RSI、KDJ、BOLL、ATR 和观察字段只使用截止
  T-1 的完整日线。
- `get_price` 显式使用 `end_date=prev_date`、日频、前复权并跳过停牌；
  `avoid_future_data=True` 与真实价格模式均开启。
- T 日现价只用于可交易状态、ATR 止损、下单和订单后实际持仓核对，不能反向修改
  T-1 信号、候选顺序或仓位公式。
- 15:30 只记录已经到期的 1/3/5 个交易日回顾观察、清理实际已归零状态、以当日
  收盘价向上更新最高收盘锚点并输出组合汇总；该阶段不下单。

新仓金额按下单时账户值自适应：

```text
standard_target = current_total_value * 0.95 / 3
cash_reserve = current_total_value * 0.05
buy_target = min(standard_target, max(0, available_cash - cash_reserve))
```

账户总资产变化只影响后续新仓目标，不强制再平衡已有持仓。

## 在聚宽中运行

1. 新建一个聚宽 Python 策略。
2. 将 `smart_trade_joinquant_resonance_reversal_etf.py` 的完整内容复制到策略编辑器。
3. 初始资金建议设置为 20,000 元，基准使用沪深300。
4. 标准成本设置为相对价格滑点 0.1%，ETF 买卖双边佣金 0.03%，单笔最低佣金
   5 元，ETF 卖出印花税为 0。
5. 双倍摩擦压力测试只把滑点提高到 0.2%、佣金率提高到 0.06%，最低佣金仍为
   5 元；不得同时改动策略规则或参数。

聚宽是收益验证的权威环境。本地测试只验证指标、事件、状态、订单路径和未来数据
边界，不能替代聚宽的撮合与收益结果。

## 冻结验证顺序

必须按以下顺序推进：

1. 在聚宽运行短区间冒烟回测，只检查初始化、日志、订单状态和无未来数据报错。
2. 只打开 2019-01-01 至 2021-12-31 的冻结训练期；2018 数据只能作为只读指标
   预热，不计入收益、不用于调参。
3. 根据设计规格中预先登记的收益、回撤、夏普、盈亏比、年度表现、交易样本、
   ETF 集中度和双倍摩擦门槛如实判断训练结果。
4. 训练记录完成后，必须再次取得用户确认，才可以依次打开验证窗口。

禁止使用本地收益替代聚宽结果，禁止用验证期或全周期结果调整参数，禁止删除训练期
表现较差的 ETF，禁止把观察指标升级为买卖资格、排序、仓位或卖出条件。训练门槛
未通过时也不得临时搜索相邻参数、延长共振窗口或增加指标来改善曲线。

完整业务规则、测试门槛和失败处理见
[`docs/strategy_spec.md`](docs/strategy_spec.md)。

## 非极值相对拐点观察（build 20260827.4）

该路径只记录未形成正式完整共振的增量候选：

- `HARD_BOLL_SOFT_OSC`：正式 BOLL 加相对 RSI 或 KDJ；
- `SOFT_ALL_THREE`：相对 BOLL、RSI、KDJ 三项齐全。

相对事件使用独立事件簿和 `RELATIVE:` 标识，不进入正式共振、排序、仓位、ATR
或订单。T 日 09:35 的相对事件与正式信号一样只使用截止 T-1 的完整日线；T 日及
后续到期日 15:30 才回顾性记录 1/3/5 个交易日的收盘结果。

初始化日志同时保留正式事件逻辑指纹，并新增独立的相对观察逻辑指纹；二者不能互相
替代。普通观察异常只写诊断且不会中断正式交易；`FutureDataError` 仍会让回测明确
失败。

### 聚宽冻结交付顺序

必须按以下三步执行：

1. 将 build `.4` 策略复制到聚宽，先做短区间冒烟；逐项核对初始化中的参数指纹、ETF
   池指纹、正式事件逻辑指纹、相对观察逻辑指纹四项值，无 `FutureDataError` 或其他未来
   数据问题，相对候选及其 1/3/5 日结果日志存在，且正式订单路径无异常并保持不变。
2. 冒烟通过后，只运行 2019-01-01 至 2021-12-31 的冻结训练回测；不得使用验证期
   调参。
3. 冻结训练完成后，用户须在独立聚宽研究环境（不是策略回测代码）先导出并保存唯一的
   交易日 manifest、冻结其原始 bytes 的小写 SHA-256；再导出完整 `20260827.3` 基线日志
   与 `20260827.4` 候选日志，才执行下列只读分析器命令。

聚宽研究环境导出 manifest 的示例（schema 与分析器合同完全一致）：

```python
import json
import jqdata

coverage_start = "2019-01-01"
coverage_end = "2021-12-31"
sessions = [
    session.isoformat()
    for session in jqdata.get_all_trade_days()
    if coverage_start <= session.isoformat() <= coverage_end
]
manifest = {
    "schema_version": 1,
    "market": "XSHG",
    "coverage_start": coverage_start,
    "coverage_end": coverage_end,
    "source": "JoinQuant get_all_trade_days",
    "sessions": sessions,
}
with open("joinquant_sessions_2019_2021.json", "w", encoding="utf-8", newline="\n") as stream:
    json.dump(manifest, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
```

先保存该文件，再在查看或分析 `.3/.4` 结果前运行并记录输出的小写值：

```powershell
(Get-FileHash .\joinquant_sessions_2019_2021.json -Algorithm SHA256).Hash.ToLowerInvariant()
```

这就是预先冻结的 hash。之后 manifest 的任何 bytes 变化都需要重新授权；不得在现场
重新计算 hash 后替换已预注册的值。

```powershell
python resonance_reversal_strategy/research/analyze_relative_turn_observations.py `
  --baseline-log D:\logs\resonance-20260827.3.log `
  --candidate-log D:\logs\resonance-20260827.4.log `
  --session-calendar .\joinquant_sessions_2019_2021.json `
  --session-calendar-sha256 <预先冻结的sha256> `
  --output D:\logs\relative-turn-report.json
```

分析器只读取用户显式提供的日志，拒绝 2022 年及以后观察记录，也不会搜索阈值、窗口
或 ETF。manifest 必须是只读、UTF-8 JSON，严格只含 `schema_version`、`market`、
`coverage_start`、`coverage_end`、`source`、`sessions` 六个字段；其大小最多 256 KiB，
会拒绝重复键、非 ISO 日期、不递增日期和不在 2019--2021 覆盖内的 session。manifest、
基线日志、候选日志与输出必须是四个不同的物理文件；分析器会拒绝同路径、符号/硬链接等
别名以及输出覆盖任何输入。

`--session-calendar-sha256` 接受 64 位十六进制的大小写形式并按大小写不敏感地比较，
但预注册记录必须使用上一步冻结的小写值。报告的 `session_calendar` 元数据固定包含
`schema_version`、`market`、`coverage_start`、`coverage_end`、`source`、`session_count`
和原始 manifest bytes 计算出的 `sha256`（小写）。每个正式和相对观察的 1/3/5 日结果
必须精确落在这个唯一 calendar 的对应交易日；缺少 session 证据、覆盖不完整或日期不一致
均 fail closed，只能得到失败的数据质量/门槛结果。

全部预注册门槛通过只代表可以提出下一份交易候选规格，不代表可以自动下单或进入验证期。
现在代码不包含真实 manifest、其冻结 hash 或聚宽平台结果，用户仍需按上述步骤导出。当前
也尚无真实聚宽 `.3/.4` 完整日志证据；本地测试不构成订单路径、期末资产或观察收益已通过
的证据。短区间聚宽冒烟同样不证明正式订单路径、收益或可以实盘。
