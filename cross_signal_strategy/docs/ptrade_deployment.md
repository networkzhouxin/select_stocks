# Cross-Signal PTrade Deployment

## Frozen Scope

- Strategy version: `cross-v0.3.2`.
- JoinQuant source of truth: `smart_trade_joinquant_cross_signal_etf.py`.
- PTrade deployment file: `smart_trade_ptrade_cross_signal_etf.py`.
- JoinQuant remains the authority for performance. The PTrade backtest is a
  runtime smoke test only.
- This port contains no multi-factor weights, Tuesday/Thursday rotation,
  switch threshold, or multi-factor bear-market rules.

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
  checkpoint. This is not an additional `run_daily` thread task.

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
  The entry date must also reproduce an eligible frozen `cross-v0.3.2` T-1 buy
  signal. Entry ATR is recalculated only on that proven signal date, and the
  trailing peak is rebuilt from the actual weighted fill price plus
  pre-adjusted non-zero-volume closing prices since entry.
- The adapter never uses the multi-factor fallback guesses such as
  `cost_basis * 2%`, `previous date - 10 days`, or an arbitrary 120-day peak.
  Quantity mismatch, missing fill price, missing calendar evidence, ineligible
  entry signal, or incomplete price history leaves the position unverified.
- Historical delivery statements are account-wide rather than explicitly
  strategy-owned. Disaster reconstruction therefore assumes the PTrade trade
  uses a dedicated account, or at least that no other strategy/manual process
  trades the same ETF pool. A mixed account requires operator reconciliation;
  the explicit state checkpoint remains the authoritative ownership record.
- A restarted partially filled buy is verified only when its already-filled
  cost basis and every later fill price are positive and finite. Otherwise the
  resulting holding remains unverified; no zero/NaN baseline is synthesized.
- An explicit state checkpoint is written atomically to
  `cross_signal_v032_live_state_<identity>.pkl` under PTrade's research path.
  The anonymous identity is derived from the account and trade name so
  simulation and live instances cannot overwrite each other's state. The file
  contains risk state, execution dates, deferred T-1 scores, and halt/recovery
  state, but deliberately excludes strategy parameters and the ETF pool.
  Both the account identity and trade name are mandatory. If either value
  cannot be obtained, checkpointing fails closed instead of falling back to a
  partial-identity or shared filename.
  Its path is resolved and cached during initialize, the lifecycle phase where
  the required account, trade, and research-path APIs are documented.
- State checkpoints run after the 09:35 and 10:35 tasks, from
  `after_trading_end`, and after order/trade callbacks. On restart, the file is
  restored before broker order reconciliation and position verification. A
  version mismatch or malformed file is rejected instead of partially
  restoring state.

## Observation-Only IOPV Log

For `513100.SS`, `513500.SS`, `513880.SS`, and `513050.SS`, the adapter writes
one `[iopv-observe]` line immediately before an actual buy submission. It reuses
the same cached `get_snapshot()` record that supplied the live buy price and
does not issue another market-data request.

The line records the callback time, code, execution reference price, positive
IOPV when available, descriptive premium percentage, raw `hsTimeStamp`, and
snapshot age. `valid=False` means that a positive price/IOPV pair was not
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

## Deployment Check

1. Copy the complete PTrade deployment file into a new Guojin PTrade strategy.
2. Run a PTrade backtest only to confirm that the script starts and completes
   without an API or syntax error.
3. Use simulation trading before live capital. Confirm the 09:35 and 10:35
   task logs plus the approximately 15:30 `after_trading_end` log, callback
   code format, halt status, partial fills, and rejected orders. For every
   submitted QDII buy, confirm exactly one `[iopv-observe]` line appears before
   the `[buy]` submission log; both `valid=True` and `valid=False` must leave
   the submitted quantity unchanged.
4. In Guojin simulation, restart the strategy after 09:35 and before 10:35.
   Verify that the explicit state checkpoint restores `execution_date`,
   `deferred_signal_date`, `deferred_scores`, `paused_pool_codes`, buy dates,
   entry ATR values, and trailing highs. Confirm that the configuration lock
   still reports the formal parameters and nine-ETF pool after restoration.
5. Delete only a simulation copy of the explicit state checkpoint after a
   filled same-day buy, restart, and verify that `get_trades()` reconstructs
   the exact fill price/date/ATR. On a later day, test delivery reconstruction
   only in an account with no manual or second-strategy trades in the pool.
6. Confirm broker-side ETF commission and minimum-fee settings separately.
   They are not strategy parameters and were not optimized here.
7. Keep the JoinQuant `cross-v0.3.2` file unchanged as the business-logic
   reference for future parity reviews.
