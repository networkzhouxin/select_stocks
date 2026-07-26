# Cross-Signal PTrade Deployment

## Frozen Scope

- Strategy version: `cross-v0.3.2`.
- JoinQuant source of truth: `smart_trade_joinquant_cross_signal_etf.py`.
- PTrade deployment file: `smart_trade_ptrade_cross_signal_etf.py`.
- JoinQuant remains the authority for performance. The PTrade backtest is a
  runtime smoke test only.
- This port contains no multi-factor weights, Tuesday/Thursday rotation,
  switch threshold, or multi-factor bear-market rules.

The formal release identity is printed once during initialization:

```text
[发布指纹] 构建=20260726.12 业务配置=1506a0e834fe 状态结构=4
```

The build identifies the copied deployment artifact. The business fingerprint
is calculated from the frozen strategy version, parameters, and normalized
nine-ETF pool, so the JoinQuant and PTrade files must print the same value.
The state schema is PTrade-only and does not participate in trading decisions.
Any mismatch from this documented identity requires a fresh local release
check before simulation or live trading continues.

State schema 4 deliberately rejects schema 3 risk state because the older
adapter could have written an unfinalized same-session daily value into
`highest_since_buy`. On the first start after this upgrade, held-position risk
facts are rebuilt from the broker delivery history and finalized historical
daily bars. Subsequent starts use the normal bounded schema-4 journal.

The indicator calculations, cross detection, buy/sell scoring, candidate
filtering, position sizing, minimum signal hold, and ATR stop formula are
frozen to the JoinQuant `cross-v0.3.2` mainline. Only platform access and live
order lifecycle handling differ.

After PTrade restores its persisted `g` object, a configuration lock rebuilds
`g.params` and `g.etf_pool` from the frozen source code and calls
`set_universe()` again. Old persisted configuration therefore cannot replace
the formal `cross-v0.3.2` parameters or nine-ETF pool after an upgrade or
server restart.

## Live Schedule

PTrade live mode registers four tasks, below the platform limit of five:

- `09:35`: run the complete cross-signal strategy using T-1 daily bars.
- A complete matching sell fill received through `on_trade_response` is the
  fast path. Once every pending sell is confirmed, the adapter immediately
  resumes the candidate list frozen at 09:35 instead of waiting for 09:36.
  Because PTrade portfolio cash and positions can lag a fill callback, the
  adapter excludes only fully sold codes from the stale holding snapshot and
  uses the greater of broker-reported cash and pre-sell cash plus confirmed
  sell proceeds. This compensation affects order continuity only; target
  sizing, ranking, and signal rules are unchanged.
- `09:36`: reconcile fills for both buy and sell orders submitted at 09:35
  through the official `get_trades()` API when `on_trade_response` was absent,
  delayed, or arrived before PTrade could attach an order ID. Only fills whose
  side and order ID exactly match a current pending order are accepted.
  Because `get_trades()` returns the current day's complete fill list, matched
  rows are grouped by order and applied as one cumulative broker fact. They
  are never added on top of the cumulative `Order.filled` value restored from
  `get_open_orders()`. Repeating the same query is therefore idempotent, while
  a later unique fill still advances the cumulative quantity.
  For an exact pending order that still has no matching fill, the same task
  calls `get_order(order_id)` and accepts only terminal status `5`, `6`, or
  `9` after validating the returned order ID, code, and cumulative quantity.
  A proven zero-fill failed buy releases the frozen backup queue; a proven
  zero-fill failed sell becomes retryable while its holding-risk state is
  retained. Missing, malformed, contradictory, or non-terminal results remain
  pending and fail closed. Status `8` is logged explicitly as "fully filled
  but waiting for trade details"; it remains pending because the order object
  alone cannot prove the exact fill price or fill value needed by the existing
  trade-state machine.
  Confirmed buys restore their actual fill quantity, fill price, buy date, ATR,
  and highest-close baseline through the existing callback state machine.
  Once all pending sells are confirmed, the adapter immediately resumes the
  buy evaluation already frozen at 09:35. It does not recalculate indicators,
  scores, ranking, or signals at 09:36. PTrade documents same-minute
  `get_trades()` results as cached from the first query, so the next minute is
  the earliest deterministic fallback rather than an arbitrary trading time.
- `10:35`: recheck ETFs that were halted at 09:35. For newly resumed ETFs,
  first reconcile fills for existing pending order IDs before refreshing
  `get_open_orders()`. A locally submitted order that disappears from the open
  list remains guarded until an exact fill or terminal callback proves its
  outcome; disappearance alone never releases a replacement buy.
  resumed holdings repeat the 09:35 ATR-stop and signal-sell checks using the
  current execution price and the same T-1 score, minimum-hold, trend, and
  risk-state guards. Newly resumed non-holdings receive their missing T-1
  score and join deferred buy execution after earlier sells are confirmed.
  The recovery pass is limited to the ETFs delayed by the 09:35 halt.
  It does not rerun already processed ETFs. Deferred scores are stored in
  pickle-eligible `g` fields with both the execution date and T-1 signal date;
  a date mismatch blocks execution.
- `10:36`: reconcile buy and sell fills for orders still pending after the
  10:35 pass with the same cumulative and idempotent order-level accounting;
  it reuses the same strict fill matching and exact-ID terminal-order query as
  09:36. If there is no pending order, it returns without calling
  `get_trades()` or `get_order(order_id)`. It does not read daily bars, recalculate
  indicators or scores, change ranking, or create a new trading decision.
- A buy submission exception, missing order ID, or terminal order response
  (`5`/`6`/`9`) with zero cumulative fills marks that ETF as a
  `零成交终态`. The code is `当日不再重试`; the adapter immediately continues
  through the same `冻结候选队列` and attempts the next qualified ETF without
  consuming the vacant slot. If broker cash has not synchronized yet, the
  existing 09:36/10:36 reconciliation path retries only that bounded backfill;
  this behavior `不新增定时任务`. Partial fills and normal fills keep their
  original lifecycle behavior.
- `after_trading_end` (normally around `15:30`): reconcile orders, print the
  position risk summary, and write the bounded state journal. A live
  same-session daily value is observation only: it is logged for diagnosis but
  cannot update `highest_since_buy`, because PTrade does not guarantee that the
  current daily period is already the final official close at callback time.
  This is not an additional `run_daily` thread task.
- The next `before_trading_start` first restores holding state and then reads
  the exact finalized T-1 daily bar. Only an exact-date, finite positive close
  may raise `highest_since_buy`. When volume is zero, the session is treated as
  suspended and the prior confirmed high is retained. A missing, stale-date,
  malformed, or failed T-1 response marks that holding unverified and blocks
  automatic exits and new buys until the final bar can be proved. This timing
  uses only information available before T trading and does not create a
  future function.

Initialization must prove the runtime mode with `is_trade()` before applying
mode-specific settings. Live mode receives only live platform parameters;
backtest mode receives only commission and slippage settings. If mode detection
raises, neither branch is configured and all `handle_data` trading is blocked.

Daily PTrade backtests execute scheduled work at the platform close regardless
of the requested time. That result must not be compared with the JoinQuant
09:35 performance result.

## Order Lifecycle Timing Diagnostics

PTrade emits one normalized `[订单生命周期]` record when an order is submitted,
partially filled, completed, rejected, cancelled, recovered from
`get_open_orders()`, or still pending after a 09:36/10:36 query. Every record
contains the source, side, ETF code, order ID, requested quantity, cumulative
fill, remaining quantity, elapsed time from submission, and terminal/raw
status. Elapsed time is reported as `未知` after a process restart when the
original submission time cannot be proved.
The corresponding Chinese field labels are `请求数量`, `累计成交`, `剩余数量`,
and `耗时`.

The source distinguishes `策略下单`, `成交主推`, `委托回报`,
`09:36主动核对`, `10:35主动核对`, `10:36主动核对`, and `get_open_orders`. Both active
reconciliation tasks emit `[订单核对汇总]` with `待核对买单`,
pending-sell count, matched buy/sell fill counts, and unresolved buy/sell
counts. `after_trading_end` emits
`[订单生命周期汇总]` with in-memory pending buy/sell counts, deferred-buy
state, and unknown-order-state guard.

These diagnostics do not add a scheduled task and do not change callback
matching, fill accumulation, retries, cash, positions, signals, or orders.
日志故障也与交易流程隔离：平台日志输出异常时仍尝试写入持久审计文件；
审计文件写入失败时平台日志继续输出，并且只提示一次，避免递归刷屏。

## Buy Filter Diagnostics

When no ETF survives the frozen buy filter, PTrade keeps the existing
`[cross-v0.3.2] 没有达到阈值的买入候选` line and adds:

- `[买入筛选汇总]`: the evaluation source, score count, pass count, and totals
  for score below threshold, existing/pending holdings, sell risk, missing
  fresh low-position evidence, blocked entry combinations, and buy
  prohibition.
- `[买入筛选明细]`: one complete line per rejected ETF with buy score, sell
  score, and every applicable rejection reason.

The source distinguishes `09:35主流程`, `成交主推`, `成交兜底`, and
`10:35复牌/卖单补偿`. These records only explain the existing filter. They do
not change candidate ordering, thresholds, signals, positions, cash, or order
submission.

## Data Boundary

- Signals use pre-adjusted daily bars ending at the proven previous trading
  day. Zero-volume daily rows are removed to match JoinQuant `skip_paused=True`.
- The `get_history` fallback accepts both the Python 3.11 long DataFrame and
  legacy code-column shape. Its index must be provably date-like so rows can be
  bounded by T-1; an integer, malformed, or otherwise unprovable index rejects
  the entire response instead of allowing an unbounded history window.
- The current-day snapshot price is used only for execution and ATR-stop
  evaluation. It never enters the T-1 signal calculation.
- In live mode, `hsTimeStamp` must be parseable to the second, belong to the
  same calendar date as the running process, not be later than the process
  clock, and be no more than 300 seconds old. A missing, malformed, future,
  prior-session, or older snapshot fails closed before its price can be used
  for an order or ATR comparison. This is an execution-data safety boundary,
  not a signal factor.
- If both PTrade trading-calendar APIs fail, the strategy submits no orders. It
  never guesses the previous trading day from weekdays.
- If the separate minimum-hold calendar query fails, normal signal sells are
  blocked instead of replacing five trading days with five calendar days.
- In live mode, a missing snapshot price or unknown halt status fails closed.

## Order Safety

- ETF codes from callbacks are normalized to `.SS` or `.SZ`.
- Every `on_trade_response` invocation writes an entry record before the live
  guard or any callback filter, followed by one bounded detail record per raw
  item. The detail includes the raw `order_id` (explicitly printed as `<空>`
  when absent), entrust number, business ID, direction, quantity, price,
  balance, entrust status (`status`), callback type (`real_type`), trade status
  (`real_status`), original entrust number for cancellation, rejection reason,
  and business time. These records are diagnostic only and do not relax
  pending-order matching.
- Buy and sell partial fills are accumulated from `on_trade_response`.
- A non-empty official `business_id` is recorded inside its matching pending
  order. A repeated push with the same成交编号 is ignored, so one broker fill
  cannot be accumulated twice. Callbacks without `business_id` retain the
  existing conservative order-ID matching path.
- Every submitted order must return a PTrade order ID. A `None` result is a
  submission failure and never creates a pending guard. Callbacks are applied
  only when their `order_id` matches the current pending order for that ETF.
- Cancellation trade pushes (`real_type="2"`) are not fills and never change
  filled quantity or strategy state.
- A sell submission does not erase `buy_date`, `entry_atr`, or
  `highest_since_buy`. State is cleared only after the requested quantity is
  fully filled.
- Submitted or partially filled sells do not release a holding slot for
  replacement buys. Confirmed proceeds may be used only after every pending
  sell has reached a terminal state, and a stale holding is excluded only when
  its requested sell quantity was fully filled. A complete trade callback
  resumes the frozen buy evaluation immediately; the 09:36 reconciliation is
  retained for an absent/delayed callback or an immediate attempt that could
  not submit an order. Otherwise the 10:35 task uses broker-confirmed cash and
  positions.
- A partially filled then cancelled sell preserves risk state for the residual
  position and releases the retry guard.
- Rejected orders release their guard without inventing a fill.
- At startup and after an intraday server restart, `get_open_orders()` rebuilds
  buy/sell guards. If broker order state cannot be queried, all new orders are
  blocked. `None` and malformed responses are also treated as unknown state,
  not as an empty order list.
- Multiple or opposite-side open orders for one ETF cannot be represented by a
  single guard, so reconciliation fails closed instead of discarding an order.
- `after_trading_end` reads `get_open_orders()` without cancelling or resubmitting anything. Every unfinished broker order is logged before the
  daily checkpoint is saved.
- A successful buy or sell submission writes the returned broker order ID.
- Malformed callback records that are not dictionaries are logged and skipped;
  they cannot terminate the callback loop or synthesize position state.
- Open-order requested and filled quantities must be finite, positive where
  required, and internally consistent. Malformed quantities block trading.
- PTrade may send an early callback with a blank `order_id`. It is ignored
  rather than guessed; the 10:35 task queries `get_open_orders()` and positions
  again before any deferred buy.
- Broker position quantity and `cost_basis` are always read from the live
  PTrade portfolio. Cost is not copied into strategy cache as a competing
  source of truth. A missing, zero, or non-finite broker cost blocks automatic
  exits.
- Persisted `buy_date`, entry ATR, and highest close are the primary cross-day
  risk state. If any field is missing, the adapter first attempts deterministic
  reconstruction and otherwise marks the holding unverified, blocking all
  automatic signal and ATR exits.
- `get_trades()` is the authoritative fallback for fills made by this strategy
  on the current day. It allows a same-day restart to reconstruct filled
  quantity, weighted entry price, buy date, T-1 entry ATR, and the initial
  highest-price baseline without waiting for the next delivery statement.
- `get_deliver()` is called only from documented lifecycle callbacks:
  `before_trading_start` and `after_trading_end`. Records from `20100101`
  through the proven T-1 date are replayed by signed quantity; the
  reconstructed open quantity must exactly match the current broker position.
  Under the dedicated-account operating contract, an existing in-pool holding
  is adopted even when its original buy came from the previously stopped
  strategy. Entry ATR is recalculated only from data available on the trading
  day before the actual buy date, and the trailing peak is rebuilt from the
  actual weighted fill price plus pre-adjusted non-zero-volume closing prices
  since entry.
- The adapter never uses the multi-factor fallback guesses such as
  `cost_basis * 2%`, `previous date - 10 days`, or an arbitrary 120-day peak.
  Quantity mismatch, missing fill price, missing calendar evidence, or
  incomplete price history leaves the position unverified.
- A failed takeover writes a stage-specific `[恢复诊断]` line.
  `delivery-replay` reports only ETF codes, dates, buy/sell direction,
  quantities, prices, aggregate replay counts, and field names; account,
  client, fund, and shareholder-account values are never emitted. Other
  stages distinguish broker facts, historical calendar lookup, entry ATR,
  weighted fill price, close-history reconstruction, and same-day handling.
  When the existing historical calendar path cannot prove T-1, the adapter
  calls documented `get_trading_day_by_date(buy_date, -1)` as an observation
  probe. Its log is marked `不参与交易判断=是`: the returned date is not used
  to verify state, calculate ATR, enable exits, or submit an order.
- Guojin PTrade may return documented calendar results as a NumPy Unicode
  array (`ndarray` with dtype such as `<U10`). Its elements are `numpy.str_`,
  which some pandas versions reject when passed directly to `Timestamp`.
  The adapter converts only that scalar type to native `str` before normal
  date parsing. The calendar contents, API priority, and fail-closed rules are
  unchanged; a usable `get_trade_days` result prevents all fallback probes.
- Historical delivery statements are account-wide rather than explicitly
  strategy-owned. Account takeover is therefore enabled only under this
  deployment's explicit operating contract: one account runs one active
  strategy at a time, the previous strategy is stopped before cross-signal is
  enabled, and the account is not traded manually. Existing holdings inside
  the frozen ETF pool are then owned by the active cross-signal strategy.
  Out-of-pool holdings remain unverified and block new buys instead of being
  sold or assigned risk state automatically.
- A restarted partially filled buy is verified only when its already-filled
  cost basis and every later fill price are positive and finite. Otherwise the
  resulting holding remains unverified; no zero/NaN baseline is synthesized.
- The explicit state store is a single bounded journal under PTrade's
  research path. Every envelope contains a state-schema version, monotonically
  increasing generation, producer strategy version, business-configuration fingerprint,
  broker position snapshot, SHA256 checksum, and protocol-4
  pickle payload. After a third complete generation is appended, a temporary
  journal containing the latest two valid generations is fully decoded and
  verified before an atomic same-directory replacement; no direct `os` call is used.
- Restore validates every complete journal record and selects the highest valid
  generation. A truncated tail never invalidates earlier complete records. On
  the next save, only the incomplete tail bytes are removed before a new record
  is appended. A checksum mismatch, unknown schema, fingerprint mismatch, or
  missing required field is rejected without partially applying state.
- Every broker position snapshot contains normalized ETF code, positive held
  quantity, and positive broker cost. Restore is refused unless the current
  broker position set, quantities, and costs still match the recorded snapshot.
- The anonymous journal identity is derived from the account and trade name
  so simulation and live instances cannot overwrite each other's state. The
  payload contains risk state, execution dates, deferred T-1 scores,
  sell-retry reasons, and halt/recovery state, but deliberately excludes strategy parameters and the
  ETF pool. Both identity values are mandatory; otherwise checkpointing fails
  closed instead of using a shared filename. PTrade rejects `get_trade_name()`
  during `initialize`, so the path is resolved and cached at the start of
  `before_trading_start`, before journal inspection and broker reconciliation.
- On each process start, validated PTrade-persisted `g` state is attempted first.
  It is accepted only when its state schema, business fingerprint, generation,
  complete per-position risk fields, and recorded code/quantity/cost snapshot
  match current broker holdings. Current broker positions remain the source of
  truth. A valid `g` state avoids an unnecessary delivery-history query. If a
  newer matching journal exists, the older `g` state is rejected so stale risk
  fields cannot replace fresher state. A journal whose recorded broker snapshot
  still matches and whose buy date, entry ATR, and highest close are complete
  for every current holding may restore those fields directly. Missing,
  incomplete, future-dated, or broker-mismatched state falls back to
  current-strategy fills, delivery records, and broker reconstruction; a
  matching journal may then fill only remaining gaps. If no source proves a
  holding, it remains unverified and exposure cannot increase.
- State journal writes run after the 09:35, 09:36, 10:35, and 10:36 tasks, from
  `after_trading_end`, and after order/trade callbacks. On restart, state is
  broker-validated before it can supply intraday continuity or old-position fallback.
- After one complete scan, the process caches only the verified journal tail:
  path, file length, latest generation, and payload digest. An unchanged state
  and broker snapshot does not append another record or increase the
  generation. A changed state appends from the cached tail without rescanning
  the full file. Any externally changed length or incomplete tail invalidates
  the cache and reopens the full validation-and-repair path.
- Successful compaction retains exactly the latest two complete generations,
  so the journal cannot grow without bound. If temporary-file writing,
  verification, or replacement fails, the original journal remains the recovery
  source and may temporarily contain more than two complete generations. The
  next changed-state save retries compaction.
- 恢复日志分成三个独立维度：`[PTrade框架g]` 说明普通 `g` 是否未提供、
  已接受、已拒绝或因台账更新而未采用；`[连续状态恢复]` 说明日内连续状态
  来自 `状态台账`、`PTrade持久状态` 还是无可用来源，并单独显示代次；
  `[持仓风险恢复]` 说明当前持仓的买入日期、ATR 和最高收盘价实际来自
  `状态台账`、`PTrade持久状态`、`当前策略成交`、`账户接管:交割单`、
  `混合恢复` 或 `未验证`。每只持仓继续单独报告数量、券商成本、买入日期、
  入场 ATR、最高收盘价、验证状态和证据来源。
- An unverified holding continues to block its own automatic ATR and signal
  exits. In addition, all new buys are blocked while any currently held ETF is
  unverified. Verified holdings retain their normal exit behavior, so recovery
  uncertainty cannot expand exposure or disable unrelated risk reduction.

## Observation-Only IOPV Log

For `513100.SS`, `513500.SS`, `513880.SS`, and `513050.SS`, the adapter writes
one `[IOPV观察]` line immediately before an actual buy submission. It reuses
the same cached `get_snapshot()` record that supplied the live buy price and
does not issue another market-data request.

The line records the callback time, code, execution reference price, positive
IOPV when available, descriptive premium percentage, raw `hsTimeStamp`, and
snapshot age. `有效=False` means that a positive price/IOPV pair was not
available. The observation return value is never consumed by candidate
filtering, ranking, position sizing, or order submission.

IOPV logging is failure-open and must never block or resize an order. Missing,
zero, stale, malformed, or exception-producing IOPV data only changes the log.
It does not reopen the rejected premium-filter experiment and must not be used
to select a threshold from validation or early live results.

## Platform Evidence

The adapter was checked against the official help bundled with the Guojin
PTrade client:

- `docs/帮助.html`: scheduling, `get_price`, `get_history`, `get_snapshot`,
  `get_stock_status`, `order`, `order_target`, and order/trade callbacks.
- `docs/财务数据.html`: bundled data-interface reference.
- `smart_trade_ptrade_multifactor_etf.py`: an already-running Guojin PTrade
  compatibility reference. Its strategy rules were not copied.

The public web search result `ptradeapi.com` describes itself as a personally
annotated copy of the original API documentation, so it is not treated as an
official authority for this port.

## PTrade 持久审计日志

实盘模式会把策略自身通过 `log.info`、`log.warning`、`log.error`、
`log.debug` 和 `log.critical` 输出的内容完整镜像到研究根目录下的单一文件：

```text
cross_signal_logs/cross_signal_v032_audit.log
```

平台控制台按用途分层：`INFO` 保留 `[交易日开始]`、五个处理阶段、候选与
交叉短摘要、实际买卖、订单生命周期、风险事件、`[交易日汇总]` 和
`[交易日结束]`；完整指标明细和逐只买入筛选明细使用带稳定标记的 `DEBUG`。
在 PTrade 界面取消勾选 `DEBUG` 可得到适合日常查看的简洁交易日志，需要排查
时重新勾选即可。

审计文件采用 UTF-8，仍镜像 `INFO`、`DEBUG`、`WARNING`、`ERROR` 和
`CRITICAL` 的时间戳、级别与完整消息，不缩减候选排名、完整指标明细、
状态恢复、委托、成交、停牌补偿、IOPV 观察或收盘汇总。它只能镜像本策略
主动调用日志接口产生的记录；
PTrade 平台自身在策略代码之外生成的服务器配置、调度或网关日志不在该文件中。

单文件硬上限为 `20 MB`。下一条日志会导致超限时，策略在同目录写入并校验
临时文件，只淘汰最旧的完整日志行，将文件压缩到约 `16 MB` 后再原子替换。
不会截断 UTF-8 字符或半条日志；替换失败时保留原文件，平台日志和交易流程
继续运行，并通过底层平台日志只提示一次审计文件写入失败。反过来，平台日志
接口异常也不会阻止审计文件写入或中断交易流程。该文件与只保存最近两代持仓
风险状态的状态台账相互独立，不能用审计日志替代状态恢复，也不能用状态台账
替代完整运行审计。

## PTrade 运行日志审计

部署或重启后的日志可先用仓库内的只读工具做结构化检查：

```powershell
python cross_signal_strategy/tools/audit_ptrade_runtime_log.py <日志文件> --date YYYY-MM-DD
```

工具按指定交易日检查初始化、状态恢复、09:35 主流程、条件性 10:35
复牌补偿、收盘汇总、委托及成交回报、`ERROR`、`WARNING`，以及 QDII
买入前是否已经记录 `[IOPV观察]`。它不会修改日志、策略或检查点，且
不读取行情数据、不调用 PTrade API，也不使用任何训练期或验证期数据。
日志包含多个交易日时必须使用 `--date` 逐日审计；工具不会把不同日期的
初始化、委托与收盘记录拼成一个貌似完整的交易日。

审计状态含义如下：

- `通过`：日志中存在该检查所要求的正面证据，且未发现对应失败证据。
- `失败`：发现明确的错误、未验证持仓、缺失的强制运行阶段，或 QDII
  买入前缺少 IOPV 观察等阻断事实。
- `需复核`：例如已经提交委托，但日志中没有后续成交、撤单或拒绝回报；
  不能仅凭当前日志证明最终状态。
- `条件未触发`：当日没有停牌补偿、没有委托或没有 QDII 买入，因此
  相应日志本来就不应出现；这不是失败。
- `证据不足`：日志片段、日期或格式不足以得出结论；不能视为已经通过。

证据边界：该工具只能审计日志里已经记录的事实，不能替代券商委托、
成交回报、持仓、交割单和检查点文件的人工核验，也不能证明未写入日志的
事件没有发生。凡是 `失败`、`需复核` 或 `证据不足`，均应先保留原始日志
并查明根因，再决定是否启用实盘资金。

## Deployment Check

1. Copy the complete PTrade deployment file into a new Guojin PTrade strategy.
2. Run a PTrade backtest only to confirm that the script starts and completes
   without an API or syntax error.
3. Use simulation trading before live capital. Confirm the 09:35 and 10:35
   task logs plus the approximately 15:30 `after_trading_end` log, callback
   code format, halt status, partial fills, and rejected orders. For every
   submitted QDII buy, confirm exactly one `[IOPV观察]` line appears before
   the `[买入]` submission log; both `有效=True` and `有效=False` must leave
   the submitted quantity unchanged.
4. In Guojin simulation, restart the strategy after 09:35 and before 10:35.
   Verify that `[PTrade框架g]`, `[连续状态恢复]`, and `[持仓风险恢复]`
   identify the framework state decision, journal generation, and actual
   position-risk source separately, and every held ETF is listed with the expected source
   and `已验证` status. Confirm that `execution_date`, deferred state, buy
   dates, entry ATR values, and trailing highs are unchanged.
5. In simulation only, back up and then append a truncated tail to the journal.
   Restart and verify that the last complete generation is selected and the
   next save removes only the truncated tail. Never perform this drill on the
   live journal. A checksum/schema error or any
   `未验证` line requires operator review before enabling live capital.
6. After a filled same-day simulation buy, make the journal unavailable and
   restart. Verify that `get_trades()` reconstructs the exact fill
   price/date/ATR and that new buys remain blocked until every existing holding
   is verified. Separately test first-start account takeover after stopping the
   previous strategy: every in-pool position must report
   `来源=账户接管:交割单`; no manual or second strategy may trade
   the account while cross-signal is active. If takeover remains unverified,
   retain all `[恢复交易日历]`, `[恢复交易日历探针]`, and
   `[恢复诊断]` lines. A valid `日期探针` is evidence for the
   next root-cause decision, not permission to trade.
7. Confirm broker-side ETF commission and minimum-fee settings separately.
   They are not strategy parameters and were not optimized here.
8. Keep the JoinQuant `cross-v0.3.2` file unchanged as the business-logic
   reference for future parity reviews.
