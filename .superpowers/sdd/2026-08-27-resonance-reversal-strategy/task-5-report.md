# Task 5 Report: Adaptive Entry Sizing and Frozen ATR Risk State

## Scope contract

- Target object: one prospective new ETF position and one existing ETF position's in-memory risk state.
- Target processing stage: pure order-time target-value calculation plus state/risk helper boundaries only.
- Allowed behavior: calculate a new-position target from current total value and available cash; keep entry ATR frozen; advance a highest-close anchor only upward; permit signal selling only after the buy date; reset only daily flags when the decision date changes; remove risk state only when actual amount is zero.
- Preserved: all frozen Tasks 1-4 public contracts, indicator/event/resonance logic, signal generation, existing-position rebalance behavior, platform scheduling, and all order submission.
- Must not propagate: no order is submitted, no tier/ATR-inverse/leverage sizing is introduced, no top-up/rebalance is added, no calendar logic is inferred, and ATR exits have no hold lock.

## Implemented

- `calc_buy_target_value` uses the frozen `target_exposure / max_holdings` target, then caps it at available cash after the exposure reserve.
- `calc_stop_state` uses frozen entry ATR and the close-only anchor with the required 2.5x ATR percentage, clamped to 5%-15%.
- `make_position_state` stores `buy_date`, `entry_atr`, `highest_close_anchor`, and `pending_exit`; `update_highest_close_anchor` only raises the anchor from a positive closing price.
- `can_signal_sell` is strictly `buy_date < decision_date`; it introduces no natural-day or trading-calendar computation.
- `reset_daily_state` resets only `sold_today` and `daily_attempted_buys` when `decision_date` changes, while pruning processed IDs on every call by `signal_date`.
- `clear_position_state_if_flat` deletes only when `actual_amount == 0`; a partial amount retains the complete state.

## TDD and verification evidence

- RED (risk/cash/holding helpers): `python -m pytest tests/test_resonance_reversal_strategy.py -k "buy_target or atr_stop or highest_anchor or signal_sell" -v` selected 8 new tests; all 8 failed with the expected missing-helper `AttributeError`.
- RED (daily/flat helpers): `python -m pytest tests/test_resonance_reversal_strategy.py -k "daily_state or flat" -v` selected the two new target tests; both failed with the expected missing-helper `AttributeError` (two pre-existing `flat` indicator tests passed).
- GREEN focused: `python -m pytest tests/test_resonance_reversal_strategy.py -k "buy_target or atr_stop or highest_anchor or signal_sell or daily_state or flat" -v` passed 12/12.
- Dedicated suite: `python -m pytest tests/test_resonance_reversal_strategy.py -v` passed 50/50 in 0.42s.
- Compile: `python -m py_compile resonance_reversal_strategy\\smart_trade_joinquant_resonance_reversal_etf.py` exited 0.
- Static review: `git diff --check` exited 0 (only Git LF-to-CRLF warnings); direct helper-call search finds no production caller yet, only the new direct behavior tests.

## Scope and control-flow review

| Condition | True behavior | False behavior | Protected behavior |
| --- | --- | --- | --- |
| `decision_date` changed | reset daily sell/buy-attempt sets | retain same-session sets | processed IDs are pruned in both cases |
| `actual_amount == 0` | remove that code's risk state, return `True` | retain state, return `False` | partial exits do not lose buy date, frozen ATR, anchor, or pending exit |
| `buy_date < decision_date` | signal sell is allowed | same-day signal sell is blocked | ATR state receives no hold lock |
| valid positive close | advance anchor only when higher | ignore non-positive/`None` close | entry ATR remains untouched |

No boolean or mode parameter was added. The only state cleanup predicate is the explicit `actual_amount == 0` contract. No branch enters an order, transaction, session, cache, lock, or resource-cleanup path. The direct-call search confirms these helpers remain unintegrated pure/state boundaries for a subsequent task.

## Concerns / non-goals

- No JoinQuant runtime/backtest or order submission was started: Task 5 intentionally supplies helpers only.
- The buy-date comparison relies on the supplied decision dates; the scheduled trading entrypoint remains responsible for trading-session context, as required.

## Fix Round 1: Reject NaN Highest-Close Anchor

- Finding addressed: `calc_stop_state` previously allowed `highest_close_anchor=NaN` through the nonpositive guard, returning a state with `raw_pct` and `stop_price` equal to `NaN`.
- Scope: only the existing invalid-input guard now also checks `pd.isna(highest_close_anchor)`; all valid-anchor arithmetic, ATR handling, clamping, and public interfaces remain unchanged.
- RED: `python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_stop_rejects_nan_highest_close_anchor" -v` selected 1 test and failed as expected: the old code returned `{'raw_pct': nan, 'stop_pct': 0.05, 'stop_price': nan}` instead of `None`.
- GREEN focused: `python -m pytest tests/test_resonance_reversal_strategy.py -k "atr_stop" -v` passed 4/4 in 0.38s, including all three pre-existing clamp cases and the NaN-anchor case.
- Dedicated suite: `python -m pytest tests/test_resonance_reversal_strategy.py -v` passed 51/51 in 0.43s.
- Compile/static: `python -m py_compile resonance_reversal_strategy\\smart_trade_joinquant_resonance_reversal_etf.py` exited 0; `git diff --check` exited 0 (only Git LF-to-CRLF warnings).
- Control flow: the new `pd.isna(highest_close_anchor)` disjunct shares the existing `return None` invalid-stop path. It adds no flag, caller, order, resource cleanup, or alternate valid-input branch.
