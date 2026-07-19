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
[发布指纹] 构建=20260720.2 业务配置=1506a0e834fe 状态结构=3
```

The build identifies the copied deployment artifact. The business fingerprint
is calculated from the frozen strategy version, parameters, and normalized
nine-ETF pool, so the JoinQuant and PTrade files must print the same value.
The state schema is PTrade-only and does not participate in trading decisions.
Any mismatch from this documented identity requires a fresh local release
check before simulation or live trading continues.

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

PTrade live mode registers two tasks, below the platform limit of five:

- `09:35`: run the complete cross-signal strategy using T-1 daily bars.
- `10:35`: recheck ETFs that were halted at 09:35. For newly resumed ETFs,
  resumed holdings repeat the 09:35 ATR-stop and signal-sell checks using the
  current execution price and the same T-1 score, minimum-hold, trend, and
  risk-state guards. Newly resumed non-holdings receive their missing T-1
  score and join deferred buy execution after earlier sells are confirmed.
  The recovery pass is limited to the ETFs delayed by the 09:35 halt.
  It does not rerun already processed ETFs. Deferred scores are stored in
  pickle-eligible `g` fields with both the execution date and T-1 signal date;
  a date mismatch blocks execution.
- `after_trading_end` (normally around `15:30`): use PTrade's official
  lifecycle callback to reconcile state, update the highest closing price
  since entry, print the position risk summary, and write the closing
  append-only state journal. This is not an additional `run_daily` thread task.

Initialization must prove the runtime mode with `is_trade()` before applying
mode-specific settings. Live mode receives only live platform parameters;
backtest mode receives only commission and slippage settings. If mode detection
raises, neither branch is configured and all `handle_data` trading is blocked.

Daily PTrade backtests execute scheduled work at the platform close regardless
of the requested time. That result must not be compared with the JoinQuant
09:35 performance result.

## Data Boundary

- Signals use pre-adjusted daily bars ending at the proven previous trading
  day. Zero-volume daily rows are removed to match JoinQuant `skip_paused=True`.
- The `get_history` fallback accepts both the Python 3.11 long DataFrame and
  legacy code-column shape. Its index must be provably date-like so rows can be
  bounded by T-1; an integer, malformed, or otherwise unprovable index rejects
  the entire response instead of allowing an unbounded history window.
- The current-day snapshot price is used only for execution and ATR-stop
  evaluation. It never enters the T-1 signal calculation.
- In live mode, `hsTimeStamp` must be parseable and have the same calendar date as the running process. A missing, malformed, or prior-session snapshot
  fails closed before its price can be used for an order or ATR comparison.
- If both PTrade trading-calendar APIs fail, the strategy submits no orders. It
  never guesses the previous trading day from weekdays.
- If the separate minimum-hold calendar query fails, normal signal sells are
  blocked instead of replacing five trading days with five calendar days.
- In live mode, a missing snapshot price or unknown halt status fails closed.

## Order Safety

- ETF codes from callbacks are normalized to `.SS` or `.SZ`.
- Buy and sell partial fills are accumulated from `on_trade_response`.
- Every submitted order must return a PTrade order ID. A `None` result is a
  submission failure and never creates a pending guard. Callbacks are applied
  only when their `order_id` matches the current pending order for that ETF.
- Cancellation trade pushes (`real_type="2"`) are not fills and never change
  filled quantity or strategy state.
- A sell submission does not erase `buy_date`, `entry_atr`, or
  `highest_since_buy`. State is cleared only after the requested quantity is
  fully filled.
- Submitted or partially filled sells do not release cash or holding slots for
  replacement buys. The 10:35 task uses broker-confirmed cash and positions.
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
- The explicit state store is a single append-only journal under PTrade's
  research path. Every envelope contains a state-schema version, monotonically
  increasing generation, producer strategy version, business-configuration fingerprint,
  broker position snapshot, SHA256 checksum, and protocol-4
  pickle payload. No `os` call or rename operation is required.
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
  truth. A valid `g` state avoids an unnecessary delivery-history query; any
  mismatch falls back to current-strategy fills, delivery records, and broker
  reconstruction. If a newer matching journal exists, the older `g` state is
  rejected so a stale highest close or ATR cannot replace fresher state. The
  journal may fill only holdings that broker history cannot prove. If no source
  proves a holding, it remains unverified and exposure cannot increase.
- State journal writes run after the 09:35 and 10:35 tasks, from
  `after_trading_end`, and after order/trade callbacks. On restart, state is
  broker-validated before it can supply intraday continuity or old-position fallback.
- After reconciliation, one `[状态恢复汇总]` source line and one line per
  held ETF report quantity, broker cost, buy date, entry ATR, highest close,
  `已验证`/`未验证` status, and evidence source. Displayed sources include
  `状态台账`, `PTrade持久状态`, `当前策略成交`,
  `账户接管:交割单`, and `未验证`.
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
   Verify that a `[状态恢复汇总]` line identifies the journal generation, and
   every held ETF is listed with the expected source
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
