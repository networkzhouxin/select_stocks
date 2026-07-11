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

PTrade live mode registers three tasks, below the platform limit of five:

- `09:35`: run the complete cross-signal strategy using T-1 daily bars.
- `10:35`: recheck ETFs that were halted at 09:35, add T-1 scores only for
  newly resumed ETFs, and execute deferred buys after earlier sells are
  confirmed. It does not rerun portfolio-wide stop or signal-sell decisions.
  Deferred scores are stored in pickle-eligible `g` fields with both the
  execution date and T-1 signal date; a date mismatch blocks execution.
- `15:30`: update the highest closing price since entry and print the position
  risk summary.

Daily PTrade backtests execute scheduled work at the platform close regardless
of the requested time. That result must not be compared with the JoinQuant
09:35 performance result.

## Data Boundary

- Signals use pre-adjusted daily bars ending at the proven previous trading
  day. Zero-volume daily rows are removed to match JoinQuant `skip_paused=True`.
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
- Persisted `buy_date`, entry ATR, and highest close must all exist for an old
  holding. If any field is missing, the holding is marked unverified and all
  automatic signal and ATR exits for it are blocked. The adapter never invents historical
  entry ATR, buy date, or trailing peak; the position then requires explicit
  operator reconciliation.
- A restarted partially filled buy is verified only when its already-filled
  cost basis and every later fill price are positive and finite. Otherwise the
  resulting holding remains unverified; no zero/NaN baseline is synthesized.
- An explicit state checkpoint is written atomically to
  `cross_signal_v032_live_state_<identity>.pkl` under PTrade's research path.
  The anonymous identity is derived from the account and trade name so
  simulation and live instances cannot overwrite each other's state. The file
  contains risk state, execution dates, deferred T-1 scores, and halt/recovery
  state, but deliberately excludes strategy parameters and the ETF pool.
  If the account/trade identity cannot be obtained, checkpointing fails closed
  instead of falling back to a shared filename.
- State checkpoints run after the 09:35, 10:35, and 15:30 tasks and after
  order/trade callbacks. On restart, the file is restored before broker order
  reconciliation and position verification. A version mismatch or malformed
  file is rejected instead of partially restoring state.

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
3. Use simulation trading before live capital. Confirm the 09:35, 10:35, and
   15:30 logs, callback code format, halt status, partial fills, and rejected
   orders.
4. In Guojin simulation, restart the strategy after 09:35 and before 10:35.
   Verify that the explicit state checkpoint restores `execution_date`,
   `deferred_signal_date`, `deferred_scores`, `paused_pool_codes`, buy dates,
   entry ATR values, and trailing highs. Confirm that the configuration lock
   still reports the formal parameters and nine-ETF pool after restoration.
5. Confirm broker-side ETF commission and minimum-fee settings separately.
   They are not strategy parameters and were not optimized here.
6. Keep the JoinQuant `cross-v0.3.2` file unchanged as the business-logic
   reference for future parity reviews.
