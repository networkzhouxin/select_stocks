# Binance Spot Dual-Baseline Quant Research System Design

Status: revised after the 2026-08-03 pre-implementation audit and frozen for
implementation planning. No historical stage has been activated.

## 1. Objective

Build a small-capital Binance Spot research and paper-trading system for a user
who expects to allocate only a few thousand CNY. The first version compares two
independent rule-based baselines on BTCUSDT and ETHUSDT using closed four-hour
bars. It must emphasize capital protection, reproducibility, and protection
against future-data leakage and overfitting.

This system is research software, not a promise of profit. The confirmed 10%
maximum-drawdown value is a rejection boundary for historical stages, not a
guarantee that live drawdown cannot exceed 10%.

## 2. Scope And Non-Goals

The first version includes:

- Binance Spot public historical and real-time market data.
- BTCUSDT and ETHUSDT only.
- One open crypto position at most; unused capital remains USDT.
- Two independent baselines, a common risk engine, a common backtester, and a
  common paper broker.
- Historical training, validation, final holdout, reporting, and a local
  Windows paper-trading runner.
- Local SQLite state plus HTML and CSV reports.

The first version excludes:

- Futures, perpetual contracts, margin, leverage, borrowing, and short selling.
- Real Binance orders, private account endpoints, API keys, withdrawals, and a
  live-trading entrypoint.
- Strategy mixing, voting, ensembling, fallback from one baseline to the other,
  and parameter optimization on reserved periods.
- Cloud deployment, Docker deployment, mobile push notifications, a custom GUI,
  tax calculations, and USDT yield or depeg modelling.
- Any runtime dependency on the existing ETF Cross-Signal package.

Real trading is a separate future milestone requiring a new design and explicit
user approval after the selected strategy passes the final holdout and the
paper-trading gate.

## 3. Global Invariants

- Use Binance Spot data only; never substitute futures or perpetual data.
- Use UTC+0 four-hour bars. Calendar-day labels and dataset boundaries are UTC.
- A signal may use only fully closed bars available at its decision time.
- A signal produced by bar T can execute no earlier than bar T+1.
- Baseline A and Baseline B share infrastructure but never signals, scores,
  positions, databases, or reports.
- Both baselines use identical data, cost, matching, risk, and benchmark rules.
- No strategy may hold BTC and ETH simultaneously.
- A higher-ranked alternative alone never forces a switch out of a valid
  holding.
- An ATR stop always overrides signal holding-period protections.
- No symbol may be sold and repurchased within the same four-hour bar.
- Source data is immutable. Derived caches, databases, and reports are written
  outside raw-data directories.

## 4. Data Contract And Time Isolation

### 4.1 Source

Use official Binance Spot klines for BTCUSDT and ETHUSDT. Persist the raw
response, request metadata, retrieval timestamp, symbol, interval, and a
cryptographic checksum. Crypto prices are unadjusted; stock-style corporate
action adjustment is not applicable.

Before the first experiment, a dedicated preregistration sealer audits both
symbols for:

- A common continuous history covering the proposed study periods.
- Exact four-hour UTC alignment.
- Unique opening times and a four-hour cadence.
- Finite OHLCV values, valid OHLC relationships, positive prices, nonnegative
  volume, and no duplicate or unexplained missing bars.
- A final closed-bar marker for streaming data.

The sealer is the only component allowed to open an unreleased stage's OHLCV. It
has no strategy, indicator, metric, or reporting imports and emits only
structural pass/fail, common coverage bounds, and content-addressed stage
manifests with ordered bar identities/checksums. It must not emit price, return,
volatility, signal, or other market-path-derived values.

After sealing, access is stage-specific. Training code can open only training
plus its warm-up. Once the validation entrant manifest and child bindings are
frozen, those children may read validation warm-up without consuming the token;
validation performance content remains unreadable until the coordinator's token
transaction atomically authorizes the children and admits the first bucket.
Holdout manifest/data remain sealed from every candidate, validation, selection,
report, cache, and log path until nomination. After nomination, a bound holdout
child may read its warm-up without consuming the token, but its first performance
read follows the same atomic token transaction. The sealer's earlier raw access
is a data-custody step, not candidate evaluation.

For REST data, a bar is closed only when its close time is earlier than the
verified Binance server time. For WebSocket data, require the explicit closed
flag. Do not use the local Windows clock alone to decide that a bar has
finished.

If the common-history contract fails, stop and ask the user to approve a revised
common start date. Do not silently shorten one symbol, splice third-party data,
or forward-fill a missing market bar.

### 4.2 Frozen Windows

Proposed windows, subject only to the initial common-history audit:

- Training/development performance:
  [2018-01-01T00:00:00Z, 2022-01-01T00:00:00Z).
- Validation performance:
  [2022-01-01T00:00:00Z, 2024-01-01T00:00:00Z).
- Final-holdout performance:
  [2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z).
- Indicator warm-up for each stage: exactly the 540 closed four-hour bars
  immediately preceding that stage's performance interval.
- Prospective paper observation: begins only after the final holdout passes.

The final holdout is a programmatically reserved historical interval, not a
claim that its market path was unknown to every human when this design was
written. Its purpose is to prevent candidate implementation, training,
validation, selection, and reporting code from reading that interval before
nomination, subject only to the nonrevealing sealer above. Only the prospective
paper qualification epoch is genuinely forward in calendar time. Reports must
use these descriptions and must not call the historical holdout a live or fully
blind test.

The initial audit converts these proposed UTC intervals and the exact ordered
warm-up bar identifiers into immutable stage manifests. After that audit,
neither a date nor a warm-up bar count remains adjustable.

The preregistered main evaluation balance is 500.00 USDT. Freeze that same
value in every stage manifest before training begins; it is the only balance
that can determine stage passage or candidate selection.

A performance bar belongs to a stage only when its open time is inside the
stage's half-open UTC interval and its Binance close time is earlier than the
exclusive interval end. A signal from the last admitted bar whose required
next-open fill lies at or beyond that end is canceled by the boundary rule
below.

Each historical stage is a separate experiment account:

- Start with exactly 500.00 USDT and no other asset.
- Carry no position, pending intent, ATR stop, highest close, Baseline A
  confirmation counter, or Baseline B holding age across a stage boundary.
- Warm-up values may calculate indicators and recent-cross flags for the first
  performance bars, but warm-up bars cannot create their own trades, returns,
  confirmation counts, or execution prices.
- Cancel an intent whose required fill time is outside the performance
  interval. It cannot move into the next stage.
- Immediately after the final admitted bar's regular close equity event, append
  exactly one non-fill terminal_valuation event. For every remaining asset j,
  let q_j be quantity and R_j that bar's raw close; terminal equity is
  C + sum_j(q_j * (R_j * (1 - s)) * (1 - c)), using Section 7's frozen s and c.
  If flat, it equals the final regular close equity. The strategy has at most one
  term; the dual-asset benchmark may have two. Do not create a strategy sell,
  closed trade, turnover, or carry into another stage.

Use chronological splits only. Random train/test splitting is prohibited.
Every loader receives a declared indicator_window and performance_window and
fails closed outside their union. Validation and final-holdout datasets require
separate manifests and explicit stage activation. The final-holdout manifest
cannot be activated until validation has nominated and fingerprinted exactly
one candidate.

## 5. Baseline A: 30/90-Day Trend Momentum

Baseline A is the deliberately simple reference model.

For a closed bar at index t:

- Thirty-day return r30 = close[t] / close[t-180] - 1.
- Ninety-day return r90 = close[t] / close[t-540] - 1.
- Thirty-day volatility vol30 is the population standard deviation of the most
  recent 180 four-hour log returns, multiplied by sqrt(180).
- Risk-adjusted score = ((r30 + r90) / 2) / vol30.
- A zero, negative, NaN, or nonfinite volatility makes the symbol ineligible.

A symbol is entry-eligible only when r30 and r90 are both strictly positive.
Eligible symbols rank by descending risk-adjusted score, with BTCUSDT as the
fixed deterministic tie-break.
The consecutive-top counter resets to zero whenever the symbol is ineligible or
is not top-ranked on a closed bar.

Entry requires the same symbol to be the top eligible symbol on two consecutive
closed bars. The order intent is then executed at the next bar open.

Holding and exit rules:

- Do not switch merely because the other symbol obtains a higher score.
- Exit when either r30 or r90 becomes nonpositive.
- If the current holding exits and the other symbol has completed fresh
  top-rank confirmation on the exit-signal bar and its immediately preceding
  bar, execute both actions in the same next-open event: sell and settle fees
  first, then size and buy the replacement from updated USDT. Cancel the buy if
  the sell does not fill.
- Otherwise remain in USDT.
- An exit resets that symbol's confirmation state; a later entry requires a
  fresh two-bar confirmation.

Baseline A has no additional indicator, profit target, rank-switch threshold,
or parameter search.

## 6. Baseline B: Four-Hour Cross-Signal Migration

Baseline B migrates the indicator and score semantics of the existing ETF
Cross-Signal mainline onto raw four-hour bars. It does not migrate ETF pools,
stock calendars, suspension handling, board-lot rules, platform APIs, or
portfolio sizing.

The provenance snapshot is:

- Source file:
  cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py
- Source strategy version: cross-v0.3.2
- Source Git commit: e30257631ae51c7131b994dd437520a15ec54f3b
- Source SHA-256:
  5c5eb4c3bc397133d2419e13165e28339125c114c85963cfda6150d535364c76

The migration copies semantics into the new module with a provenance note and
checked-in golden vectors. Production and normal tests must not import the ETF
module at runtime.

### 6.1 Raw Four-Hour Parameters

Periods remain raw bar counts; they are not multiplied by six:

- RSI: 6, 12, and 24.
- MACD: 12, 26, and 9.
- KDJ: 9, 3, and 3.
- Bollinger Bands: 20 and 2 standard deviations.
- Moving averages: 5, 10, 20, and 60.
- ATR: 14.
- ADX/DMI: 14, with strong-trend threshold 25.
- Volume averages: 5 and 20.
- Cross-recency window: 3 bars, equal to 12 hours.
- Buy threshold: 60.
- Strong-buy threshold: 70.
- Signal-sell threshold: 30.
- Sell-risk observation threshold: 18, log-only.
- New-buy RSI ceiling: RSI6 less than 85.
- Minimum signal holding period: 5 four-hour bars, equal to 20 hours.

The Binance application screenshot values such as MA7/25/99 and any exchange UI
defaults are observation-only and are not strategy parameters.

### 6.2 Frozen Indicator Implementation

Every Baseline B score evaluation receives exactly the most recent 120 closed
four-hour OHLCV bars ending at the decision bar. Reinitialize every recursive
calculation from the first bar in that slice, as the pinned source does. Do not
feed full-history EWM state into the calculation. The source's defensive
minimum is 110 bars, but a formal complete-history run is expected to supply all
120; otherwise log and skip the symbol.

Preserve these pandas-equivalent calculations:

Unless stated otherwise, pandas rolling operations use `min_periods` equal to
window length and `center = False`; EWM operations use `ignore_na = False`.
Preserve these calculations:

- RSI: delta = close.diff(); gain is positive delta and otherwise zero; loss is
  absolute negative delta and otherwise zero. Average gain and loss use EWM
  alpha = 1 / period, min_periods = period, adjust = True, and
  ignore_na = False. RSI is 100 - 100 / (1 + average_gain / average_loss). When
  average loss is zero, return 100 if average gain is positive and 50 if both
  are zero.
- MACD: fast and slow close EWM use their spans with min_periods = 0,
  adjust = False, and ignore_na = False. DIF is fast EMA minus slow EMA; DEA is
  DIF EWM with signal span and the same defaults; histogram is 2 multiplied by
  (DIF - DEA).
- KDJ: lowest low and highest high are rolling N-bar extrema. RSV is
  100 multiplied by (close - lowest) / (highest - lowest), with a zero range
  treated as NaN. K is RSV EWM with com = M1 - 1, min_periods = 0,
  adjust = False, and ignore_na = False; D applies the same EWM contract to K
  with com = M2 - 1; J = 3K - 2D.
- Bollinger Bands: rolling simple mean and sample standard deviation with
  ddof = 1; upper/lower are the mean plus/minus two standard deviations.
- ATR: true range is the pandas row maximum of high-low, absolute
  high-previous-close, and absolute low-previous-close with `skipna = True`.
  The first slice row therefore uses high-low when previous close is absent. ATR
  is the 14-bar simple rolling mean.
- DMI/ADX: `up_move = high.diff()` and `down_move = -low.diff()`.
  `plus_dm = up_move` only when up_move is strictly greater than down_move and
  strictly positive, otherwise zero. `minus_dm = down_move` only when down_move
  is strictly greater than up_move and strictly positive, otherwise zero; equal
  moves therefore produce zero in both series. Positive DI is 100 multiplied by
  rolling_sum_14(plus_dm) divided by rolling_sum_14(ATR), and negative DI is the
  corresponding minus_dm expression. DX is 100 multiplied by the absolute DI
  difference divided by the DI sum; ADX is the 14-bar simple rolling mean of DX.
  This deliberately preserves the source's nonstandard rolling sum of ATR as
  denominator and its pandas NaN/infinity behavior.
- MA5/10/20/60 and VOL5/20 are simple rolling means. MA20 slope is current MA20
  minus MA20 five bars earlier.

For a three-bar cross window, calculate fast minus slow and scan the three most
recent transitions from oldest to newest. A transition from less than or equal
to zero to strictly positive records above; a transition from greater than or
equal to zero to strictly negative records below. The latest recorded direction
in the window wins, so a newer opposite cross cancels the older direction.
Cross age is zero for the newest transition, one for the preceding transition,
and two for the oldest transition. A requested direction receives that age only
when it is the final direction after the complete scan; otherwise its age is
None. For the RSI group, `simultaneous` means that the current snapshot contains
at least one raw RSI upward flag and at least one raw RSI downward flag across
RSI6/12 and RSI6/24, even if the flags arose on different bars in the window.
That state makes the group direction neutral and awards neither RSI group's
cross points.

For the latest bar t, preserve these exact predicate boundaries from the pinned
source:

- lower_BOLL[t] <= close[t] <= middle_BOLL[t];
- close-near-MA20 means abs(close[t] / MA20[t] - 1) <= 0.05 when MA20[t] > 0;
- far-above-MA20 for buy scoring means close[t] / MA20[t] - 1 > 0.12;
- close-below-MA20, close-below-middle-BOLL, MA5-above-MA10,
  MA10-above-MA20, close-above-MA60, and VOL5-above-VOL20 all use strict
  inequalities;
- MA20 slope nonnegative means MA20[t] - MA20[t-5] >= 0;
- downside continuation means close[t] < MA60[t] and that MA20 slope < 0;
- volume confirmation means volume[t] > VOL20[t] and close[t] > close[t-1];
- the sell-risk overextension means close[t] / MA20[t] - 1 > 0.10 and
  RSI6[t] < RSI6[t-1];
- close below a falling MA10 means close[t] < MA10[t] and
  MA10[t] < MA10[t-1]; and
- fall back inside BOLL means close[t-1] > upper_BOLL[t-1] and
  close[t] <= upper_BOLL[t]. It has no lower-band condition.

Every recent-cross location predicate uses the frozen three-transition algorithm
above. A comparison whose required operand is NaN is false.

The pinned completeness guard requires the latest RSI6, RSI12, RSI24, DIF,
DEA, K, D, J, MA20, ATR, and ADX values to be present and not NaN. Other
derived fields are not part of that guard; a comparison involving their NaN is
simply false, as in the pinned source. Positive or negative infinity is not NaN
and is not an additional scoring-level rejection. The raw-data contract still
requires finite OHLCV values, and the common risk layer separately requires a
finite positive risk ATR. The latest close must be positive, and aggregate
volume across the latest five bars must be greater than zero.

### 6.3 Frozen Score Semantics

Buy reversal score:

- RSI6 crossing above RSI12: 12.
- RSI6 crossing above RSI24: 12.
- DIF crossing above DEA: 10.
- K crossing above D: 6.
- J crossing above D: 5.

The RSI upward scores count only when the RSI group direction is unambiguously
up within the three-bar cross window.

Buy location score:

- Close between lower and middle Bollinger Bands: 10.
- Close crossing above the middle Bollinger Band: 8.
- Close within 5% of MA20: 7.
- Close more than 12% above MA20: minus 10.

Buy trend score:

- MA5 above MA10: 6.
- MA10 above MA20: 6.
- MA20 five-bar slope nonnegative: 5.
- Close above MA60: 3.
- Close below MA60 while MA20 slope is negative: minus 15.

Buy volume score:

- Current volume above VOL20 while close rises: 6.
- VOL5 above VOL20: 4.

Total buy score is exactly
`max(0, reversal_score + location_score + trend_score + volume_score)`.
New entry also requires:

- Buy score at least 60.
- Sell score below 30.
- RSI6 below 85.
- At least one valid location state: lower-to-middle Bollinger location,
  middle-band upward cross, or close near MA20.
- Close not more than 12% above MA20.
- The frozen cross-v0.3.2 blocked-entry combination remains blocked. It reads
  the raw flags, not the neutralized RSI group direction: at least one of the
  raw RSI6/12 or RSI6/24 upward flags, the raw MACD upward flag, neither raw KDJ
  upward flag, volume_score > 0, and 0 < trend_score < 20. It can therefore
  block an entry even when simultaneous raw RSI down and up flags make the RSI
  score group neutral.

Sell reversal score:

- RSI6 crossing below RSI12: 12.
- RSI6 crossing below RSI24: 12.
- DIF crossing below DEA: 10.
- K crossing below D: 6.
- J crossing below D: 5.

The RSI downward scores count only when the RSI group direction is
unambiguously down within the three-bar cross window.

Sell risk score:

- close[t] / MA20[t] - 1 > 0.10 and RSI6[t] < RSI6[t-1]: 8.
- close[t] < MA10[t] and MA10[t] < MA10[t-1]: 10.
- close[t-1] > upper_BOLL[t-1] and close[t] <= upper_BOLL[t]: 6.

Total sell score is exactly `max(0, sell_reversal_score + sell_risk_score)`.

A normal signal exit requires all of:

- Minimum five-bar holding period completed.
- Sell score at least 30.
- At least one confirmation: close below MA20, close below the middle
  Bollinger Band, close below a falling MA10, downside continuation, or the
  far-above-MA20 RSI downturn.
- No strong ADX uptrend protection, unless a severe trend break is present.

Strong ADX uptrend protection exists only when ADX is at least 25, positive DI
is strictly greater than negative DI, and the five-bar MA20 slope is
nonnegative. ADX equal to 25 and slope equal to zero qualify; equal DI values do
not.

A severe trend break means close below MA20, close below a falling MA10, or
close below MA60 while the MA20 slope is negative. In those cases ADX
protection cannot suppress the signal exit.

For the five-bar minimum hold, the entry-fill bar becomes held bar one only
after that bar closes. The earliest normal sell signal is evaluated after held
bar five closes and fills at held bar six's open. ATR protection remains active
from entry and is not subject to this counter.

When buy score is at least 70 and sell score is below 30, preserve the holding
against a normal signal exit. This source-order branch is outcome-redundant with
the sell-below-30 requirement but remains part of log semantics: it suppresses
the later sell-risk observation log. Otherwise, a sell score from 18 through 29
is logged for observation only and cannot change the position.

Entry candidates rank by descending buy score, then descending reversal score,
then the fixed BTCUSDT tie-break. Baseline B adds no two-bar confirmation.
Ranking alone cannot replace an existing valid holding.

When a held Baseline B symbol exits normally and the other symbol satisfies all
entry rules on the same closed decision bar T, the decision may sell the held
symbol and conditionally buy the highest-ranked eligible other symbol at the
same T+1 execution event. The sold symbol is excluded. Sell, commission
settlement, fresh sizing, and dependent buy follow Sections 8 and 12. If the
other symbol is not eligible or the sell does not fill, remain in USDT.

The A-share zero-volume scaling rule, ETF maximum-hold rule, 95% base ratio,
trading-calendar logic, and suspension logic are explicitly excluded.

### 6.4 Behavioural Interpretation

Baseline A is intentionally trend-following: it will buy only after a positive
30/90-day trend already exists, and its two-bar confirmation adds entry lag.
Baseline B is a cross/reversal model, but its indicators and confirmation rules
also react after prices move. Neither baseline is described as eliminating
chasing or lag. Reports may show pre-entry and pre-exit returns to explain that
behaviour, but those diagnostics are observation-only and cannot add a filter.

## 7. Common Portfolio And Risk Rules

The common risk engine receives a DecisionPlan containing hold, one standalone
buy or sell, or an ordered sell with one dependent buy. Strategy code may name
the eligible symbol but cannot choose position size. It cannot create an
unordered pair of orders.

Risk ATR14 is strategy-independent. From fully closed bars ending at signal bar
T, calculate true range as the maximum of high minus low, absolute high minus
previous close, and absolute low minus previous close. Risk ATR14 is the simple
arithmetic mean of the latest 14 true ranges. The common indicator layer, not a
strategy, computes it identically for Baseline A and Baseline B. The value must
be finite and strictly positive; otherwise skip the buy.

Freeze this risk ATR in the DecisionPlan at T. It may not be recomputed from
T+1 or changed by the strategy. Baseline B's scoring ATR uses the same true
range and SMA14 definition inside its frozen 120-bar snapshot, while the common
risk value remains the sizing and stop authority for both baselines.

Use this price and settlement vocabulary everywhere; the unqualified term
`execution reference` is prohibited:

- Let adverse slippage s = 0.0005 and commission rate c = 0.001.
- Raw reference price R is the causal price before this model's slippage: the
  next admitted bar open for a normal historical fill; the bar open for a
  historical gap-through stop; the active stop for a historical intrabar stop;
  the first eligible ask for a paper buy; and the first eligible bid for a paper
  sell or paper stop.
- Synthetic fill price F is R * (1 + s) for a buy and R * (1 - s) for a sell.
  Slippage is applied exactly once.
- For filled base quantity q, precommission fill notional N = q * F,
  commission = N * c, and modelled slippage = q * abs(F - R).
- A buy changes cash by -(N + commission) and adds exactly q base units. A sell
  removes exactly q base units and changes cash by +(N - commission). Commission
  is always modelled in USDT and never reduces received base quantity.

At buy execution, calculate:

- account risk budget = equity_before_buy * 0.01;
- initial stop percentage p_stop =
  clamp(2.5 * frozen_risk_ATR14 / F, 0.05, 0.15);
- risk-sized notional N_risk =
  equity_before_buy * 0.01 / (p_stop + 0.003);
- allocation cap N_cap = equity_before_buy * 0.30; and
- candidate notional N_candidate =
  min(N_risk, N_cap, available_USDT / (1 + c)).

The unrounded quantity is N_candidate / F. Apply decimal downward quantization
and every applicable filter, then recalculate N, commission, cash sufficiency,
and the 30% cap from the resulting quantity. Skip a zero or nonconforming result.
Never round up or exceed risk. A dependent replacement buy uses only the cash
and equity committed after its parent sell and commission settle.

The 1% value is a modelled loss budget under the assumed stop fill. Gaps,
exchange outages, extreme slippage, and stablecoin or platform events can cause
larger real losses.

### 7.1 ATR Protective Stop

- ATR14 is frozen at entry.
- Initial highest close is the actual entry fill price.
- After each fully closed bar, highest close becomes the maximum of its previous
  value and that bar's close. Intrabar high is never used.
- Candidate stop =
  highest close multiplied by
  (1 - clamp(2.5 multiplied by entry ATR / highest close, 5%, 15%)).
- Active stop is the maximum of the previous active stop and candidate stop, so
  it can never loosen.
- The stop applicable to a bar is frozen before that bar begins.
- A new position creates its initial stop immediately after its next-open fill.
  That stop is active for the remainder of the entry bar, so the entry bar's
  later low may trigger a valid same-bar protective exit.

An ATR stop ignores Baseline B's five-bar minimum hold and strong-buy
protection. After a stop, the same symbol cannot re-enter during that bar.
There is no additional cooldown rule.

### 7.2 Cash And Equity Events

Let C_k be committed USDT cash and q_j,k the committed quantity of asset j
after event k. An equity mark never mutates cash, quantity, cost basis, signal,
or stop state.

E0 is the frozen starting equity immediately before the first admitted
performance-window open and before any benchmark initial allocation. Warm-up
bars create no equity event. Equity immediately after a fill is
Q_k = C_k + sum_j(q_j,k * M_j,k), where M is a causal raw mark retained for
attribution even when post-fill quantity becomes zero:

- A historical normal open fill uses that event's unadjusted open R. A historical
  gap stop uses raw open R, and an intrabar stop uses raw stop-trigger R.
- A paper buy uses the triggering bookTicker raw ask R. A paper normal sell,
  stop, or recovery_stop uses the triggering bookTicker raw bid R.
- A regular close uses that symbol's unadjusted raw close. terminal_valuation
  uses the frozen final raw close R, then subtracts terminal reserve separately.
- Every interim leg of a same-open benchmark allocation uses one raw-open price
  vector frozen before the first leg; no asset is repriced between legs.
- At an event without a new causal mark for an uninvolved asset, retain its prior
  mark. A completed strategy sell has zero post-fill quantity but still stores
  its sell-event R as M_current for the attribution transition.

With unchanged raw marks, the economic fill effect is modelled slippage plus
commission; independent Q18 operations may add a deterministic
accounting_rounding_residual under Section 9.3, which is reported separately and
never relabelled as friction. For each admitted performance bar i, regular close
equity E_i is close_cash_i + sum_j(close_quantity_j,i * raw_close_j,i).
The raw close is unadjusted and contains no hypothetical liquidation reserve.
Record each fill event in mandatory causal order and then exactly one regular
close event per admitted bar. The strategy one-position invariant does not apply
to the explicitly defined benchmark accounts in Section 9.3.

## 8. Cost And Matching Contract

### 8.1 Normal Signal Orders

- Use a taker/market execution model.
- A signal calculated after bar T closes fills at bar T+1 open.
- Backtest buy price = next open multiplied by 1.0005.
- Backtest sell price = next open multiplied by 0.9995.
- Charge 0.10% commission on every buy and every sell fill.
- Ignore BNB discounts, VIP reductions, and temporary zero-fee promotions.
- A complete buy and sell therefore models approximately 0.30% friction.

Backtest and paper orders are synthetic full-fill assumptions once all filters
pass; they do not claim to reproduce a real order book, depth impact, or partial
fills.

#### 8.1.1 WebSocket Protocol Health

Paper trading uses one persisted `logical_connection_epoch`; this is the only
schema field name for the logical public-stream epoch. Later prose saying
"logical epoch" or "connection epoch" is shorthand for that same field.
Subscribe with lowercase symbols and millisecond timestamps to exactly
`btcusdt@kline_4h`, `ethusdt@kline_4h`, `btcusdt@bookTicker`, and
`ethusdt@bookTicker`; timezone-offset Kline streams are prohibited. Normally one
physical socket carries all four streams.

Every physical socket implements the official protocol contract:

- Reply to each server ping as soon as possible with a pong carrying the exact
  ping payload. An unsolicited pong does not satisfy a received ping.
- Keep ping, pong, subscribe, unsubscribe, and other outbound control traffic
  within the official five-messages-per-second connection limit.
- Treat `serverShutdown` as an ingress-visible socket failure at receipt. No later
  frame from that physical socket may act; apply the role-aware outcome below and
  start a replacement where that lifecycle permits. The notice never proves an
  uninterrupted handoff.
- Persist socket open, close, error, ping, local pong-write success/failure,
  serverShutdown, subscription readiness, and retirement events. The protocol
  does not provide a separate pong acknowledgement to wait for.

For each physical socket, persist all_frame_watchdog_origin_monotonic. Its
initial value is the socket-open commit monotonic; after that, only a valid frame
whose interval passes this rule advances it to that frame's receipt. Also record
the last valid Kline and bookTicker receipt for every subscribed symbol. Because
the official server normally pings every 20 seconds and requires its pong within
one minute, absence of every valid inbound frame for more than 60 seconds is a
local transport safety rule, not an exchange market-data interval guarantee.

Before any inbound frame advances the origin, its serialized handler compares
the frame's monotonic receipt with the current origin. Exactly 60 seconds is
allowed, including for the first frame. If elapsed time is greater than 60
seconds, the same ingress transaction preserves the late raw frame and records
critical transport_silence_gap before any origin update. The late frame cannot
restore physical/logical health or participate in business, even if it acquires
the barrier before a delayed timer.

Apply failure by socket role and lifecycle. A required preregistration socket
failure atomically sets startup_lifecycle_state=failed and never creates
paper_mode_state. After activation, failure of the authoritative socket,
including a predecessor before atomic handoff, increments
connection_health_version and enters safe_paused; continue only through a new
logical_connection_epoch and explicit recovery. An unaccepted successor failure
while its authoritative predecessor remains healthy retires only that candidate
socket and permits another rollover attempt; it does not change logical health.
A failure of a planned_retired predecessor is audit-only. Receipt of
`serverShutdown`, close, transport error, malformed subscribed payload, or local
failure to write the required pong before its deadline uses this same role-aware
rule. If `serverShutdown` races the handoff transaction, ingress/barrier order
uniquely determines whether that socket was predecessor-authoritative or already
planned_retired when the failure applied.

Independently, freeze a local held-symbol bookTicker silence limit of 60 seconds.
The watchdog origin is the later of position-activation commit and the last
accepted held-symbol bookTicker monotonic receipt. Only a bookTicker accepted
under Section 8.1.2 advances this origin. Valid-but-stale or duplicate traffic may
maintain all-frame transport health but cannot advance the held-symbol origin;
conflicting, malformed, or otherwise invalid traffic advances neither. Elapsed
time exactly 60 seconds is allowed; greater than 60 seconds creates critical
stop_monitoring_gap before a late quote can be used. Quote processing itself
checks this interval, so a late frame cannot race and clear an overdue watchdog
event. It enters safe_paused and that frame cannot fill; recovery requires a
later post-recovery-barrier accepted quote. Without a position, stale bookTicker
data blocks new fills until a fresh accepted update arrives and never authorizes
a cached quote or widens an intent deadline.

#### 8.1.2 Logical Epochs, Rollover, And Message Identity

Because an official Spot stream connection is valid for only 24 hours,
establish a successor before hour 23 while the predecessor is healthy. At overlap
start, the predecessor remains authoritative, the successor is candidate-only,
handoff_watermark is frozen for each symbol, and successor_socket_id is bound.
A planned overlap preserves logical_connection_epoch only when the successor has
delivered a valid Kline frame and valid bookTicker frame for both symbols.
During overlap, both socket IDs feed one serialized ingestion queue and every
message retains origin socket.

Commit handoff acceptance only after all four successor subscriptions are ready
and an accepted successor-origin bookTicker for each symbol had u strictly
greater than that symbol's frozen handoff_watermark. Record
successor_progress_u; later shared-watermark movement does not erase that proof.
An equal-to-handoff message, or a successor message stale because the predecessor
advanced first, does not prove progress. The same transaction makes the
successor authoritative and marks the predecessor planned_retired. Its later
planned close/retirement is audit evidence and does not increment logical health
or pause the epoch; post-retirement frames cannot act.

Before handoff acceptance, a failed candidate successor is retired and may be
replaced while the predecessor remains healthy. If the authoritative predecessor
becomes unhealthy first, continuity is unproven: start a new
logical_connection_epoch and apply the lifecycle-aware connection-gap rule.
After acceptance, the same rule applies to an unhealthy authoritative successor.

`last_accepted_u` is scoped by `(logical_connection_epoch, symbol)`. A new epoch
starts without a watermark and its first valid bookTicker seeds one; an older
epoch's value remains audit data but is never compared with the new epoch.
Within one epoch and symbol:

- u greater than the watermark is accepted and advances it, regardless of which
  physical overlap socket delivered it;
- u lower than the watermark is stale and discarded;
- equal u with the same canonical `(b, B, a, A)` decimal payload is a duplicate;
  and
- equal u with a different canonical payload is critical
  book_ticker_payload_conflict. Retain both payloads and retire the affected
  physical connections because no REST source can reconstruct that historical
  best quote. A preregistration atomically fails without paper_mode_state; an
  activated epoch enters safe_paused and requires a new
  logical_connection_epoch plus explicit recovery.

The update ID is only a deduplication/staleness key. A noncontiguous increase is
accepted and does not itself prove a gap. For Kline frames, x=false updates are
persisted as raw evidence but never occupy the canonical closed-bar key or enter
signals. The first x=true payload for `(symbol, interval, open_time)` freezes the
closed record. A later x=true with the same canonical OHLCV, close time, and
trade count is a duplicate; a different x=true is critical
closed_kline_payload_conflict. Preserve every version and reconcile through
official REST. An x=false frame after closure is stale, logged, and ignored.

During preregistration, any closed_kline_payload_conflict atomically fails that
startup regardless of later REST resolution; it creates no paper_mode_state and
requires a new database/boundary. After activation, never rewrite a business
ledger that consumed a first closed version. If REST confirms that version,
retain it plus conflict evidence and require explicit recovery. If REST selects a different version or cannot resolve the conflict,
training discards/replays the affected run, consumed validation/holdout is
invalid and ends the research cycle, and any paper epoch whose signal,
DecisionPlan, or fill consumed it is invalidated and restarted under Section 13
with a new database. If no paper business decision consumed it, freeze the REST-
selected version, mark the bucket replay_only, and recover explicitly. Existing
intents, fills, cash, or reports are never silently reversed or overwritten.

Every received market or control frame, malformed payload, socket lifecycle or
local pong-write result, watchdog/clock event, and eligibility-affecting public
REST response or failure (server time, Kline, exchangeInfo, avgPrice, or ticker)
obtains a strictly increasing global ingress_seq under the ingestion barrier.
Its linearization point is the atomic ingress-record commit when the callback,
timer, or REST completion holds that barrier, not an unobservable network-stack
arrival. Sequence allocation and raw-record persistence are one transaction.
Market records use ingress_seq; no quote/Kline/exchange ID is activation authority.

The serialized queue applies committed ingress records strictly in sequence and
atomically advances last_applied_ingress_seq with each record's derived health/
data disposition. On restart, apply every committed-but-unapplied record before
activation or recovery. connection_health_version increments only when an
eligibility-relevant health state actually changes, not for every ingress. Every
state activation drains through its captured high water. Persist raw payload,
request clocks where applicable, process/socket identity, logical epoch,
monotonic and one-time corrected-UTC receipt, exchange identifier, and
immutable disposition. Never recalculate a stored corrected timestamp.

Only an accepted bookTicker is an execution-price source: R is its best ask for
a buy and best bid for a sell, requiring finite positive price and positive
displayed quantity. Do not race another stream or reuse a cached quote.

#### 8.1.3 Normal Paper Intent Window

For a normal paper intent, target_time is the intended next-bar open. Normally
both validated x=true Klines were received live in the same uninterrupted
logical epoch. A missing live close may instead be repaired by official REST and
bound to that decision epoch only when epoch/health stayed continuous, server
time proves closure, canonical identity matches, each REST evidence record has
been applied at or below the decision transaction's ingress high water, and
response plus decision commit finish no later than target_time plus 60 seconds.
Record evidence ingress_seq and binding. A late/unapplied repair, epoch change,
unhealthy interval, or unresolved conflict restores only state continuity,
receives permanent replay_only disposition, and cannot make the bucket executable.

An eligible quote must:

- belong to the decision's still-healthy logical epoch and process_epoch;
- have ingress_seq strictly greater than
  decision_commit_ingress_high_water_seq, monotonic receipt strictly after
  decision_commit_monotonic, and the committed connection_health_version still
  valid; and
- have its frozen corrected-UTC receipt at or after target_time and no later than
  target_time plus 60 seconds.

A fresh Binance server-time sample is a public request completed within the
preceding 60 seconds with round-trip time no greater than two seconds. Estimate
UTC offset against the local send/receive midpoint. Use corrected UTC only to
align target_time and test the UTC quote window. At decision commit, convert the
remaining interval to a process-local monotonic deadline. Persist target_time,
UTC expiry, window length, process_epoch, and commit clocks; never persist or
reuse an absolute monotonic value across a restart. A stale sample or backward/
forward wall-clock jump cannot widen the process-local deadline.

The first market- and time-eligible quote is authoritative. It may fill only
with already-applied causal filter evidence from Section 8.3; a failed attempt
cannot wait for later metadata or substitute another quote. Its serialized
handler reacquires the ingress barrier, drains every event through the then-
current high water, verifies that no intervening health/watchdog event invalidated
the intent or quote, and conditionally commits the fill while still holding the
barrier. Thus a disqualifying event linearized before fill commit wins, while an
event linearized after commit is causally later.

Apply Section 7's price vocabulary: paper buy F = raw ask * 1.0005 and paper
normal sell F = raw bid * 0.9995. Record signal and target times, live/REST Kline
evidence, decision and fill commit clocks/barriers/health versions, filter
evidence, quote receipt clocks, raw bid/ask, socket/logical epoch, u, ingress_seq,
and fill time.

If the deadline expires, logical epoch or health version changes, the clock
becomes stale, or the process restarts before the fill transaction commits,
transition the normal intent to missed. Never fill it from a later quote or
historical Kline open. A missed parent sell atomically cancels its dependent buy.

### 8.2 Stop Orders

#### 8.2.1 Historical Backtest Stop

For an existing position in bar T, use only the stop frozen before T begins:

- If T opens at or below the stop, sell at T open with adverse sell slippage.
- Otherwise, if T low touches or crosses the stop, sell at the stop with adverse
  sell slippage.
- If neither occurs, no stop fill exists.
- Update the highest closed price and next stop only after T closes.

At an open event, evaluate a gap-through-stop before any pending normal signal
sell. If both apply, create one stop-classified sell, cancel the duplicate
normal sell, and update cash once. A previously authorized buy of the other
symbol may proceed at that same open only after the sell fills and fees settle.
If the sell fails or is missed, cancel the replacement buy. A later intrabar
stop likewise cancels any unexecuted sell intent for that position.

This avoids assuming whether an unknown intrabar high occurred before or after
an intrabar low. A newly entered position follows the entry-bar exception in
Section 7.1.

#### 8.2.2 Prospective Paper Stop

Paper stop monitoring uses only serialized, accepted bookTicker best-bid updates
from Section 8.1. The first eligible best bid at or below the active stop
triggers one synthetic sell with R equal to that bid and F = R * 0.9995,
followed by the 0.10% commission. It never fills at the stop or at a price
learned later from a Kline.

Every state-activation commit shares the ingress barrier with all market,
control, health, and timer events. Before a decision, dependent-child activation,
entry stop, closed-bar stop update, or recovery activation commits, hold the
barrier, drain and apply every already linearized event, capture the greatest
assigned ingress_seq and current connection_health_version, commit the mutation
with process-local monotonic commit time, and then release the barrier. A later
linearized event receives a greater sequence. Therefore:

- decision_commit_ingress_high_water_seq is the decision transaction's ingress
  high-water value;
- a newly entered position sets stop_active_after_ingress_seq to the entry-fill
  transaction's ingress high water, not the entry quote's sequence;
- a closed-bar stop update uses its transaction ingress high water, not the
  Kline's sequence;
- a dependent child uses the parent-fill transaction ingress high water as
  activation_after_ingress_seq; and
- recovery uses recovery_commit_ingress_high_water_seq from its own transaction.

A quote may act on newly committed state only in the same process_epoch when its
ingress_seq is strictly greater than the applicable ingress barrier, its
monotonic receipt is strictly after that transaction's monotonic commit, and the
captured connection health remains valid. corrected UTC is used only for the
normal-intent target window, never to order a quote against a commit. A quote
linearized before but processed after commit cannot act retroactively. A rolled-
back transaction activates nothing.

If the first eligible next-open quote is at or below the stop while a normal
sell is pending, classify one sell as a stop and cancel the normal sell. A
dependent other-symbol buy activates only after the parent sell and fee commit;
it must wait for a new ask beyond activation_after_ingress_seq and remain inside
the original deadline.

Any connection-continuity loss, stale clock, process loss, overdue held-symbol
watchdog, or integrity failure while a position is open creates critical
stop_monitoring_gap and enters safe_paused. Backfill and deterministic state
replay may classify a proven historical crossing as missed_execution but never
create a retroactive fill. Explicit recovery follows Section 12; the replayed
active stop and first post-recovery-barrier held-symbol bid determine
recovery_stop or continued holding. Gap evidence remains permanent and resets
the clean paper clock.

Before any paper stop or recovery_stop fill, apply Section 8.3's already-fresh,
causally prior current paper market-order evidence to the complete retained
quantity. If the first triggering quote cannot produce a valid full-position
sell, required metadata is stale, or any applicable filter is unknown:

- persist one terminal skipped stop intent and its exact quote/filter evidence;
- create no fill and mutate no cash, quantity, stop, or equity;
- cancel or keep blocked every dependent buy without activating it;
- emit critical stop_execution_blocked, retain the position, and enter
  safe_paused; and
- create no repeated quote-by-quote stop intents until a new explicit recovery
  attempt establishes its own barrier and receives another quote.

The stop/recovery quote handler uses the same drain-recheck-commit ingress barrier
as a normal fill. On success, it conditionally commits intent, unique fill, fee,
position/equity mutation, fencing/account versions, fill high water, and health
version while the barrier remains held. Duplicate/competing quotes or a health
event linearized before commit cannot create a fill.

### 8.3 Quantity Filters

Query and record official Binance Spot exchangeInfo, symbol status, Spot and
market-order support, and filters. Historical filter history is not assumed to
exist. Both baselines use the same versioned execution contract and exact decimal
arithmetic. A quantity must satisfy the intersection of every applicable
LOT_SIZE and MARKET_LOT_SIZE minimum, maximum, and step grid after downward
quantization. Honor recorded disabled-zero semantics only where the official
filter definition permits them. Reject an unknown, contradictory, stale, or
uninterpretable field.

Historical runs use the frozen approximation
`market_filter_price_proxy = raw_reference_price_v1`. After selecting the
largest downward-quantized quantity not above the candidate quantity, define
filter_notional = quantity * R. Apply every recorded MIN_NOTIONAL and NOTIONAL
minimum/maximum whose market-order flag is active. Retain avgPriceMins and all
original fields in the manifest, but do not query a present-day rolling average
inside a historical run and do not use F as proxy. This is a deterministic
present-day feasibility approximation shared by both baselines, not a claim
about Binance's historical market-order decision.

For paper startup and every paper order or recovery attempt, the exact
exchangeInfo response used must have completed within the preceding 60 seconds
with request round-trip time no greater than two seconds. It must report the
symbol as TRADING with the required Spot/market capability. Persist request send,
receive, corrected server time, payload, and hash. The causal cut is paper
activation commit for startup and the triggering bookTicker for an order or
recovery. Startup evidence must be applied at or below activation ingress high
water; order/recovery evidence must be in the same process_epoch with
`evidence_ingress_seq < trigger_quote_ingress_seq`. A response merely received
but not committed/applied before the cut is ineligible, and a later REST response
cannot authorize an earlier quote.
Age exactly 60 seconds and RTT exactly two seconds are accepted; any larger value
fails closed.

Apply each active paper MIN_NOTIONAL or NOTIONAL market-order filter with its
own official public price proxy already prepared before the triggering quote:

- When that filter's avgPriceMins is positive, use `GET /api/v3/avgPrice`; require
  returned `mins` to equal avgPriceMins exactly, require
  `0 <= quote_corrected_utc - closeTime <= 60 seconds`, and require quote
  monotonic receipt minus response monotonic receipt to be between zero and 60
  seconds inclusive.
- When avgPriceMins is zero, use the official last-price meaning represented by
  a fresh `GET /api/v3/ticker/price` response; this endpoint's payload has no
  exchange timestamp, so its persisted request/receipt clocks provide causality
  and age.

Each request has the same maximum two-second RTT, 60-second receipt age,
same-process, applied-ingress, and strict evidence-before-quote requirements. If
two active filters need different proxies, calculate and record their notionals
separately; do not silently reuse one value. A missing required request clock or avgPrice closeTime,
interval mismatch, unknown flag, nonpositive price, stale response, or
contradictory result fails the attempt. Apply the quantity-grid intersection
first, then test every active notional boundary with its assigned proxy. Never
use F as a filter proxy.

The first market- and time-eligible quote ends that order attempt: filter pass
fills against that quote, while filter failure records a terminal skipped intent
and cannot wait for refreshed metadata or a later quote. A protective sell uses
Section 8.2.2's stronger stop_execution_blocked and safe_paused handling rather
than being silently abandoned.

For every attempted order record candidate and quantized quantity, R, F, every
filter proxy and notional, every tested boundary and market-application flag,
all causal evidence times/hashes, execution-contract hash, and final pass/skip
reason.

## 9. Evaluation And Selection Protocol

### 9.1 Hard Failure Rules

A baseline or candidate fails a stage if any of these occurs:

- Future data, an unclosed bar, or same-signal-bar execution affects a decision.
- Fees or slippage are omitted.
- A runtime, accounting, data-boundary, or quantity-filter error remains.
- Net return after costs or maximum drawdown is nonfinite.
- Net return after costs is not positive for the stage.
- Maximum drawdown exceeds 10% for the stage.
- Results cannot be reproduced from the recorded data, configuration, and code
  fingerprints.

Training is a development window, not machine-learning fitting. It may expose
an implementation defect. A fix is allowed inside this cycle only when it makes
the implementation conform to this frozen document. Record the defect,
recalculate every affected candidate and run fingerprint, discard every affected
training run, and replay each affected baseline from the training start. Only a
final clean training pass and its recomputed fingerprint may enter validation.
No reserved result may choose or tune a parameter, rule, indicator, or fallback.

Validation and final holdout each have a unique stage_activation_id and one
single-use cohort activation token. Before consumption, freeze an ordered
semantic_entrant_manifest: validation contains every one or two training-pass
baselines, whereas final holdout is a singleton. It binds each entrant's
candidate_strategy_fingerprint and stage-specific child run_fingerprint, but no
random run/database identity. Define activation_coordinator_fingerprint as the
SHA-256 of the stage-manifest hash, semantic_entrant_manifest hash, coordinator
protocol/schema version, every behavior-affecting coordinator executable-module
hash, and canonical serialization rules. It excludes stage_activation_id, token,
run_id, database_id/path, timestamps, and outcomes.

Separately create an immutable ordered_child_binding_record containing
stage_activation_id, token hash, activation_coordinator_fingerprint, and each
child's candidate fingerprint, run_id, run_fingerprint, and physical database_id;
its SHA-256 is cohort_activation_record_hash. This audit hash is identity-bearing
and is never treated as a stable semantic fingerprint or included in
canonical_report_hash. All child identities must exist before any child result
becomes visible. Store the token, manifests, binding record, and first-bucket
authorization in a separate append-only cohort-control SQLite database containing
no account state. Each isolated child database must verify that authorization;
no cross-database account transaction is implied.

Loading or calculating warm-up bars does not consume the token. The coordinator
must durably consume it in the same transaction that admits the first
performance bucket to the cohort and authorizes that bucket for every registered
child. A rolled-back transaction exposes no derived result and leaves the token
unconsumed. Once consumed, the stage may continue or recover only with the same
stage_activation_id, activation_coordinator_fingerprint,
cohort_activation_record_hash, and exact ordered child set; each child may resume
only its bound run_id, run_fingerprint, and database. Adding, removing,
replacing, or changing one child, or changing code, input data, configuration,
dependencies, metrics, matching, accounting, or execution semantics, invalidates
the complete stage and ends this research cycle. It does not authorize a clean
rerun or replacement token.

A presentation-only renderer is a separate artifact/process that is not imported
by any business-output module. Its renderer_fingerprint is stored beside, but
excluded from, candidate_strategy_fingerprint, run_fingerprint,
canonical_ledger_hash, and canonical_report_hash. It may regenerate HTML or CSV
from the immutable canonical ledger only when no signal, intent, fill, equity
event, metric value, selection result, or canonical report hash changes. A
semantic strategy change is not a fix to either frozen baseline. It ends this
research cycle and requires a new preregistered cycle with genuinely unused
validation and holdout periods. A validation or holdout period already viewed in
this cycle cannot become unseen again.

### 9.2 Fingerprints And Stage Progression

Before training, calculate a SHA-256 candidate_strategy_fingerprint for each
baseline from a canonical manifest containing the design revision hash;
baseline identity and semantic version; universe, bar, decision, and formal
starting-balance conventions; every frozen strategy, risk, cost, accounting,
metric, and selection contract; the baseline-specific executable strategy-module
hash; every executable shared indicator, strategy-support, risk, cost,
accounting, metric, and selection module hash that can affect business output;
canonical serialization and numeric-precision rules; and a
semantic_dependency_lock_hash covering the exact Python implementation/runtime,
decimal/libmpdec semantics, pandas, NumPy, and every other imported dependency
capable of changing candidate indicators, decisions, accounting, metrics, or
canonicalization. It excludes run_id, stage inputs and results, outcomes,
timestamps, process identity, and machine-specific presentation metadata.

A run_fingerprint contains candidate_strategy_fingerprint plus every frozen
behavior-affecting execution input: exact business executable source/build and
its business-source tree/commit identity, complete dependency lock and runtime
versions, broker-adapter version, stage and execution manifests, and a canonical
environment contract. The business-source identity is scoped to loaded business
inputs and therefore excludes a separately deployed presentation renderer. A
historical run also includes its complete immutable raw-data manifest. A
prospective paper run instead includes paper_startup_semantic_hash and its frozen
feed/environment contracts. It excludes nomination_record_hash,
final_holdout_pass_record_hash, paper_preregistration_record_hash, database/run/
epoch IDs, and every other audit identity. Future raw events append to
raw_event_manifest_root and the canonical ledger without changing
run_fingerprint. A separately deployed presentation renderer and its fingerprint
are not business execution inputs and are excluded. Also exclude PID, process
start time, absolute paths, temporary names, and other volatile metadata unless
the field is explicitly a frozen behavioral input.

At validation, holdout, and paper startup, recompute
candidate_strategy_fingerprint from the actually loaded business executable
modules, frozen manifest, and semantic_dependency_lock_hash and compare it with
the admitted/nominated candidate; never trust a supplied label alone. The
immutable
nomination record binds candidate_strategy_fingerprint, validation
stage_activation_id, activation_coordinator_fingerprint,
cohort_activation_record_hash, nominated child run_id/run_fingerprint/database_id,
canonical ledger/report hashes, and the complete ordered comparison path.
nomination_record_hash is SHA-256 of its canonical bytes with its own hash field
absent. It is an identity-bearing audit-chain hash, not a semantic input to a
later run or canonical report.

Cross-stage equality applies only to candidate_strategy_fingerprint, which
already commits the semantic_dependency_lock_hash. Any semantic dependency drift
is therefore a different candidate and is rejected. Holdout verifies the
complete nomination record but creates and verifies its own
stage-specific run_id, run_fingerprint, database_id, coordinator fingerprint, and
cohort activation record; paper does the same under its startup contract. A
validation run_fingerprint or database_id is evidence, never an expected paper or
holdout identity.

After a successful singleton holdout, write an immutable
final_holdout_pass_record binding nomination_record_hash, holdout
stage_activation_id, activation_coordinator_fingerprint,
cohort_activation_record_hash, child run_id/run_fingerprint/database_id,
candidate_strategy_fingerprint, canonical ledger/report hashes, every hard-gate
input/outcome, and explicit pass status. SHA-256 of its canonical bytes with the
hash field absent is final_holdout_pass_record_hash. Paper preregistration is
forbidden unless it
verifies this record against the original holdout ledger/report and binds that
hash.

1. Run both frozen baselines independently on 2018-2021.
2. If neither passes training, stop the research cycle without activating
   validation or final holdout. Otherwise, only the clean, recomputed
   training-pass fingerprints enter 2022-2023 validation.
3. Freeze one ordered validation cohort and all child identities, allocate its
   activation record, and run it once. The token mechanics in Section 9.1 apply;
   selection waits until every registered child reaches a terminal outcome.
4. If every entrant fails validation, stop the cycle without activating the
   final holdout or modifying a rule for a retry.
5. If only one baseline passes validation, nominate its frozen
   candidate_strategy_fingerprint.
6. If both pass, compare Sortino first, then Calmar, maximum drawdown, friction
   as a percentage of starting equity, and turnover in that order. Define
   relative_gap = abs(a - b) / max(abs(a), abs(b), 1e-12). Sortino or Calmar is
   eligible for comparison only when both baselines have finite, strictly
   positive values; otherwise skip that metric symmetrically. For an eligible
   ratio, higher wins when relative_gap is at least 10%; below 10% is a tie. For
   drawdown, friction, and turnover, compare only two finite nonnegative values;
   lower wins when relative_gap is at least 10%, otherwise continue.
7. If all eligible comparisons remain tied or skipped, nominate the simpler
   Baseline A candidate_strategy_fingerprint.
8. Freeze a singleton final-holdout cohort and run only the recomputed nominated
   fingerprint on [2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z). Its token is
   consumed by the first admitted performance bucket as specified in Section
   9.1.
9. If the candidate fails, stop the cycle. Do not inspect the holdout and then
   promote the validation runner-up.
10. If it passes, preserve the nomination and holdout evidence and begin
    prospective paper observation with that exact recomputed candidate
    fingerprint.

Only these two preregistered baselines exist in this research cycle. Record
every conforming training defect correction, discarded run, activation,
recovery, invalidation, and failure. Do not create a third candidate, reopen
validation, revise a baseline after a reserved result, or relabel a new run as a
recovery.

### 9.3 Metrics And Benchmarks

Record at minimum:

- Net total and annualized return.
- Maximum drawdown, drawdown interval, and drawdown duration.
- Annualized volatility and downside volatility.
- Sharpe, Sortino, and Calmar.
- Closed trades, win rate, profit factor, profit/loss ratio, and average and
  median trade return.
- Average and maximum holding duration.
- Average exposure, time in market, turnover, buy count, sell count, stop count,
  skipped orders, and missed executions.
- Commission, slippage, total friction, and friction as a percentage of both
  starting equity and gross profit.
- BTC and ETH attribution, calendar-period attribution, and the largest single
  trade's contribution to profit.

A closed trade starts with a filled buy and ends only with an actual filled
sell. terminal_valuation affects stage equity but does not close a trade. For a
closed trade:

- Entry cash outlay is buy notional plus buy commission.
- Exit cash receipt is sell notional minus sell commission.
- Net USDT profit is exit cash receipt minus entry cash outlay.
- Net trade return is net USDT profit divided by entry cash outlay.

A win has positive net profit, a loss has negative net profit, and zero is flat.
Win rate is wins divided by all closed trades, including flats in the
denominator. Gross winning profit is the sum of positive net USDT profits;
gross loss is the absolute sum of negative net USDT profits. Profit factor is
gross winning profit divided by gross loss. Profit/loss ratio is the arithmetic
mean winning net trade return divided by the absolute arithmetic mean losing net
trade return. A metric is nonfinite when it has no required observations or a
zero/nonfinite denominator.

For each actual fill, modelled slippage is q * abs(F - R). Total friction is all
actual-fill slippage and commissions plus, exactly once, the separately labelled
terminal reserve summed across every remaining asset j:
`sum_j(q_j * abs(R_j * (1 - s) - R_j) + q_j * (R_j * (1 - s)) * c)`.
terminal_valuation remains a non-fill event and creates no order, fill, sell,
turnover, holding duration, or closed trade. Independent Q18 rounding residuals
are accounting reconciliation items, not commission, slippage, terminal reserve,
or total friction. Friction as a percentage of gross profit uses gross winning
profit as its denominator and is nonfinite when gross winning profit is zero.

Use these same-maximum-crypto-allocation reference portfolios:

- 30% BTC plus 70% USDT.
- 30% ETH plus 70% USDT.
- 15% BTC plus 15% ETH plus 70% USDT.

Reference weights are determined only by the UTC calendar and never by a closed
signal. Allocate at the first admitted bar's open, then rebalance at the open of
the first admitted four-hour bar of every later UTC calendar month. Freeze one
raw-open vector R and mark pretrade equity from it. For each crypto asset with
weight w, freeze target quantity as the greatest valid downward-grid quantity
whose `target_quantity * R <= w * pretrade_equity`; never divide by F. The leg is
target quantity minus committed current quantity. Execute reductions first in
BTC, ETH order and settle their F-based proceeds/costs. Execute increases in BTC,
ETH order toward the already frozen targets, reducing an increase further only
when the settled available-cash-plus-buy-commission constraint or an applicable
filter requires it; never recompute pretrade equity or another target after a
leg. Apply the strategy's adverse F, commission, filter snapshot, and quantity
contract to every nonzero leg. Do not rebalance between those opens; use the same
terminal-valuation rule at stage end.

Benchmarks appear only in comparison reports. They cannot pass or fail a
baseline, nominate a candidate, or change a rule.

Do not present a comparison with a 100% crypto benchmark as a like-for-like
capital-risk comparison. Treat idle USDT as zero yield.

For each historical stage, formal passage, metrics, and selection use only the
frozen 500.00 USDT main ledger. In the same stage activation and single
chronological source-event pass, run isolated 300.00-USDT and 1000.00-USDT shadow
ledgers with identical frozen
strategy inputs, signal evaluations, source events, and
candidate_strategy_fingerprint. Each shadow advances its own portfolio decisions,
quantity/filter outcomes, and mutable account state under an explicit balance
tag. The 500-USDT ledger, canonical_ledger_hash, metrics, hard gates, and
selection inputs must be identical with the shadow engine enabled or disabled.
A diagnostics-disabled run is test-only and cannot issue a formal canonical
report; a valid formal report contains all three diagnostics, so its
canonical_report_hash covers them. Shadow return/drawdown outcomes cannot fail a
stage, order candidates, select a baseline, or change a rule.

Missing canonical shadow ledger/diagnostic payload, a shadow runtime/accounting
error, formal-ledger mutation, or source-pass/diagnostic inconsistency is instead
an implementation-completeness defect, never a main-ledger performance failure.
During training, a conforming
fix recalculates applicable fingerprints and requires the affected baseline plus
both shadows to replay from training start. Before a reserved token is consumed,
the defect blocks activation. After a validation/holdout token is consumed, it
invalidates that complete stage and ends the research cycle without rerun or
replacement token.

Let E0 be starting equity immediately before the first admitted performance
open, and E1 through En be the regular equity values at successive four-hour
closes. Define periodic return ri = Ei / E(i-1) - 1. The later
terminal_valuation is deliberately not a periodic return. Use population
denominators throughout:

- Mean periodic return mu = sum(ri) / n.
- Periodic volatility sigma = sqrt(sum((ri - mu)^2) / n).
- Periodic downside deviation at zero minimum acceptable return =
  sqrt(sum(min(ri, 0)^2) / n). The denominator is all n observations, not only
  the negative subset.
- Annualized volatility = sigma multiplied by sqrt(2190).
- Annualized downside volatility = downside deviation multiplied by sqrt(2190).
- Sharpe = sqrt(2190) multiplied by mu divided by sigma.
- Sortino = sqrt(2190) multiplied by mu divided by downside deviation.

If n is zero or a required denominator is zero or nonfinite, report the ratio as
nonfinite. No negative periodic return therefore makes downside deviation zero
and Sortino nonfinite. At selection time, Section 9.2 skips Sortino or Calmar for
both baselines unless both values are finite and strictly positive; a one-sided
nonfinite or nonpositive value is never an automatic win. The risk-free rate
and minimum acceptable return are both zero.

Always append terminal_valuation after En. Its general value is
`E_final = C + sum_j(q_j * R_j * (1 - s) * (1 - c))`; when flat it equals En.
Net total return is E_final / E0 - 1. Calculate CAGR from E0 to the terminal
event using exact UTC elapsed duration and 365.25 days per year. Calmar is CAGR
divided by maximum drawdown; report it as nonfinite when drawdown is zero or
either input is nonfinite. terminal_valuation affects net total return, CAGR,
Calmar, maximum drawdown, and total friction, but never periodic returns,
volatility, Sharpe, Sortino, turnover, trade counts, win rate, or
holding-duration statistics.

Turnover is the sum of the absolute F-based precommission notional of every
actual filled buy and sell, including both legs of a same-open replacement and
all stop fills, divided by the arithmetic mean of E1 through En. It is a raw
stage ratio, not annualized. terminal_valuation is excluded. The selection
friction percentage is total friction, including any terminal reserve, divided
by frozen starting equity.

The chronological drawdown stream starts with E0, then contains every post-fill
equity event, every regular close equity event, and terminal_valuation in causal
order. At each event, drawdown is one minus current equity divided by the
running peak including E0. Use event sequence to order equal timestamps; on tied
maximum drawdowns choose the earliest trough, with the latest preceding event at
the governing peak. Duration runs from that peak to recovery at or above it, or
to terminal if unrecovered. E0 must never be omitted merely because the first
fill occurs at the first admitted open.

At regular close i, crypto exposure is
`x_i = sum_j(q_j,i * raw_close_j,i) / E_i` when E_i is finite and positive.
Average exposure is sum(x_i) / n, and time in market is the count of regular
closes with any positive crypto quantity divided by n. Both are nonfinite when
n is zero. Closed-trade holding duration is exact UTC elapsed time from filled
buy to actual filled sell; report arithmetic mean and maximum, with an open
terminal position excluded.

For deterministic BTC/ETH attribution, between consecutive equity events assign
`q_j,previous * (M_j,current - M_j,previous)` to asset j, then subtract that
asset's actual-fill friction and any final terminal reserve. At each event,
calculate preliminary Q18 contributions. Define accounting_rounding_residual as
the persisted Q18 equity delta minus their Q18 sum, then assign it to the event's
affected asset; if both are affected, use BTC then ETH causal order. Persist this
residual separately as rounding only. It may be positive, negative, or zero,
never changes total friction, and absorbs non-distributive differences among
independently rounded fill, mark, cash, equity, and terminal calculations. Idle
USDT contributes zero, and the two asset totals, including residuals, must
reconcile exactly to E_final - E0. Group the same event contributions by UTC
calendar month and year; assign terminal reserve to the final period. Label
partial first/final periods. The largest single trade's contribution to profit is
the largest positive closed-trade net USDT profit divided by gross winning profit
and is nonfinite when there is no winning closed trade.

Do not add a minimum closed-trade count, minimum exposure, or minimum
time-in-market hard gate in this cycle. Always report those observations and
flag a passing stage with sparse trades or exposure as limited evidence; the
flag cannot alter passage, ranking, or selection. This is an explicit residual
risk to be considered before any later real-trading proposal.

The raw audit ledger retains all database, run, activation, and process identity.
Its immutable canonical business-ledger projection uses deterministic logical
event keys, frozen field order/decimal precision/UTC/event order, and excludes
random run_id, stage_activation_id, database/row IDs, host/process identity,
absolute paths, and other nonbehavioral audit metadata. Its SHA-256 value is
canonical_ledger_hash.

Every hashed canonical report payload contains canonical_ledger_hash,
candidate_strategy_fingerprint, the current run_fingerprint, metrics, gates,
diagnostics, and any ordered comparison path. A training payload adds its stable
stage-manifest hash. Validation and holdout add the stable stage-manifest and
semantic activation_coordinator_fingerprint. Paper has no cohort coordinator; it
adds paper_startup_semantic_hash and the frozen feed-contract semantic hash.

Store stage_activation_id, cohort_activation_record_hash,
nomination_record_hash, final_holdout_pass_record_hash,
paper_preregistration_record_hash, paper observation/database/run IDs, and
renderer_fingerprint beside the applicable payload. Bind them in the appropriate
audit records, but exclude them from canonical_report_hash. Also exclude
rendered_at timestamps, host/user names, absolute paths, temporary filenames,
HTML/CSV layout, and presentation-only metadata. A renderer may change excluded
fields, but must reproduce the same canonical_report_hash and all business
outputs.

### 9.4 Numeric And Canonical Serialization Contract

Freeze numeric_protocol_v1 as follows:

- Parse exchange prices, quantities, filters, cash, fees, and ledger values from
  strings into base-10 Decimal; binary float is forbidden in execution,
  filtering, accounting, metrics, hashes, and gates. An exchange decimal needing
  more than 18 fractional digits is unsupported and fails closed.
- Use Decimal context precision 50 and ROUND_HALF_EVEN. Q18 means quantize to
  exactly 18 fractional digits. Positive quantity/grid reductions alone use
  ROUND_FLOOR on exact integer step ticks.
- Persist buy F = Q18(R * (1 + s)) and sell F = Q18(R * (1 - s)); N =
  Q18(q * F); commission = Q18(N * c); slippage =
  Q18(q * abs(F - R)); and each named cash, equity,
  terminal, attribution, return, drawdown, friction, and metric result as Q18
  after its complete stated formula. Later formulas consume those persisted
  operands. Independent Q18 operations are not assumed to distribute over
  addition or subtraction; Section 9.3's accounting_rounding_residual is the sole
  reconciliation field. Normalize negative zero to positive zero.
- Baseline indicator and predicate calculations retain the pinned pandas/NumPy
  IEEE-754 float64 implementation and semantic_dependency_lock_hash; do not
  insert Decimal rounding into those predicates. Any finite score/value persisted
  into a business ledger is converted from its shortest round-trip float string
  to Q18. An indicator-only NaN, positive infinity, or negative infinity retained
  by a stated pinned predicate is serialized respectively as `FLOAT64:NAN`,
  `FLOAT64:POSITIVE_INFINITY`, or `FLOAT64:NEGATIVE_INFINITY`; it is never parsed
  as Decimal or admitted to later arithmetic. Predicate outcomes are computed
  before serialization. Any other nonfinite strategy input follows its stated
  fail/skip rule.
- Canonical finite decimals are JSON strings with a sign only when negative and
  exactly 18 fractional digits. Canonical nonfinite metrics are tagged strings
  `NONFINITE:NO_OBSERVATIONS`, `NONFINITE:ZERO_DENOMINATOR`, or
  `NONFINITE:INVALID_INPUT`; JSON NaN, Infinity, null, locale formatting,
  and scientific notation are forbidden.
- Canonical JSON uses UTF-8 without BOM, Unicode NFC strings, lexicographically
  ordered object keys, schema-ordered arrays, no insignificant whitespace, and
  explicit schema/numeric_protocol versions. Every record hash omits its own hash
  field.

Hard gates and selection compare canonical Q18 values. Therefore the smallest
positive canonical return is 0.000000000000000001, maximum drawdown exactly
0.100000000000000000 passes, and 0.100000000000000001 fails. The full protocol
and golden vectors are candidate/run fingerprint inputs.

## 10. Architecture

Create a new top-level sibling package named binance_spot_strategy. It must not
be nested under cross_signal_strategy.

Recommended boundaries:

- config: frozen symbols, bar interval, costs, risk, and stage manifests.
- domain: Bar, SignalIntent, DecisionPlan, ExecutionGroup, OrderIntent, Fill,
  Position, and EquitySnapshot.
- data: official Binance REST/WebSocket adapters, raw persistence, continuity
  checks, and stage-restricted loaders.
- indicators: pure indicator calculations.
- strategies/baseline_a: only the 30/90-day model.
- strategies/baseline_b: only the migrated cross-signal model.
- risk: sizing, one-position constraint, and ATR stop state.
- backtest: chronological event replay and the frozen fill contract.
- paper: real-time public-data simulated broker.
- state: SQLite transactions, schema versioning, checkpoints, and recovery.
- reporting: separate baseline reports and the frozen comparison process.
- entrypoints: historical backtest and paper runner only.

Tests live under tests/binance_spot and use synthetic data or checked-in small
fixtures. Existing ETF local loaders, ETF backtesters, JoinQuant/PTrade entry
files, stock-code conversion, board-lot rules, and G-drive data are not reused.

The data flow is:

closed-bar validation -> strategy intent -> common risk checks -> order intent
-> backtest or paper broker -> fill -> transactional state -> report.

Strategy modules know nothing about Binance APIs, SQLite, reports, or each
other. Broker adapters know nothing about indicator rules.

## 11. Local Windows Operation

- Provide a PowerShell launcher and Windows Task Scheduler setup.
- Start the paper runner at system startup or user login, subject to Section 12's
  global database-identity mutex and fencing claim. A duplicate launch exits
  without starting another event loop.
- Store state in local SQLite plus structured logs, HTML reports, and CSV.
- Never ask for or store a Binance API key; do not require a cloud server.

When the computer is offline or a paper Kline gap occurs, backfill official data
for indicator continuity. Section 8.1.3 alone permits a timely REST repair to
participate in an open execution window. Every common paper bucket obtains one
immutable execution_disposition: live_eligible only when admitted for executable
decision under Section 8; otherwise replay_only. No later frame, repair, restart,
or recovery may promote replay_only to live_eligible.

At nonexecuting replay start, hold the ingress barrier, apply all committed
ingress, and freeze `(replay_cutoff_close_time,
replay_cutoff_ingress_high_water_seq)` at the latest fully closed common bucket.
Atomically mark every unprocessed bucket at or before that cutoff replay_only.
Release the barrier and replay matching BTCUSDT/ETHUSDT buckets chronologically.
Before recovery commit, reacquire the barrier; if a newer fully closed common
bucket arrived, extend the cutoff and repeat. A late record for a close time at
or before a committed cutoff inherits replay_only permanently.

During replay:

- Recalculate indicators and advance Baseline A confirmation state. If an actual
  position remains, first test the stop frozen before each bar. Persist only the
  earliest provable crossing as stop-classified missed_execution, suppress that
  bar's lower-priority normal sell/replacement, and create no fill; later
  crossings are audit observations without duplicate misses.
- Then advance held-bar count, highest closed price, and next active stop from
  the close. Because actual quantity was retained, subsequent replay continues
  from that position and the stop may tighten but never loosen.
- Produce ReplayDecision evidence, never an executable DecisionPlan. On a bucket
  without a higher-priority stop miss, persist one missed_execution only when
  frozen rules would have created an expired order. An ordinary hold creates no
  fictitious miss. Never mutate cash or quantity.

With no position, commit replayed indicator/confirmation state. With a position,
commit replay cutoff, resulting stop/version, counters, and misses before
recovery_armed. In both cases set
decision_not_before_close_time = replay_cutoff_close_time. Section 12's strict
`bar_close_time > decision_not_before_close_time` check admits the first later
eligible bucket and rejects the cutoff/every backlog bucket. Neither a Kline
open nor a later quote is a retroactive fill.

The formal runner assumes the computer remains running and automatic sleep is
disabled. Downtime remains visible and applies Section 13's clocks.

## 12. State, Idempotency, And Failure Handling

Identify a bar by symbol, interval, and open time. Historical loaders fail closed
on a manifest gap. Paper requires validated BTCUSDT and ETHUSDT closes with the
same close time. Live/timely REST-bound buckets follow Section 8.1.3; every other
repair follows Section 11. Never invent, overwrite, or ignore a bar/conflict.

Every decision key has non-null (run_id, scope_type, scope_id,
candidate_strategy_fingerprint, run_fingerprint, bar_close_time). scope_type and
scope_id are respectively (training_stage, stage_id),
(reserved_stage, stage_activation_id) for validation/holdout, or
(paper_epoch, paper_observation_epoch_id). Never substitute NULL or reuse one
mode's identifier in another mode. Give each DecisionPlan execution_group_id. A
standalone order begins pending. In a replacement, the sell is pending parent
and buy is blocked child with depends_on_intent_id. Only committed parent fill/
fee settlement activates child; skipped, missed, or canceled parent atomically
cancels it.

Every intent/fill has an immutable ID. Legal intent transitions are
blocked -> pending or canceled, and pending -> filled, skipped, missed, or
canceled. Terminal states never change; blocked cannot fill/skip/miss directly.
Enforce UNIQUE(fill.intent_id) and conditional expected-state transitions.

### 12.1 Single Ownership, Ingress, And Transactions

Use a Windows `Global\` named mutex with a fixed DACL restricted to the intended
local runner identity, Administrators, and SYSTEM. Derive its key from the
resolved final file identity (volume/file ID, not spelling or 8.3/case/junction
alias) plus an immutable database UUID. Hold it before BEGIN IMMEDIATE claims the
owner row and increments process_epoch and monotonically increasing fencing_token.
A second process in any Windows session exits. After prior process termination
releases the mutex, a successor records takeover and increments the token;
wall-clock lease expiry alone never preempts a live owner.

Every write, checkpoint, owner heartbeat, and callback verifies fencing_token
and account-state version. Loss stops business processing; a stale writer cannot
commit after takeover. Each database has one serialized event loop and uses
BEGIN IMMEDIATE for mutations.

Global ingress_seq allocation and raw ingress insertion are atomic. The loop
applies committed records strictly in sequence and updates derived state plus
last_applied_ingress_seq in one transaction; duplicate application is
idempotent. Before any activation, decision, fill, or recovery, apply every
committed-but-unapplied record through captured high water. Change
connection_health_version only when eligibility-relevant health state changes.

A decision transaction holds the ingress barrier, drains/apply records, verifies
active, health version, live_eligible disposition, and
`bar_close_time > decision_not_before_close_time`, then atomically commits the
two-symbol inputs, signals, DecisionPlan, process/logical epochs,
decision_commit_ingress_high_water_seq, monotonic commit, health/account/fencing
versions, and checkpoint. Never wait for a quote inside a transaction.

A normal, stop, or recovery fill handler reacquires the ingress barrier around
final eligibility and commit. It drains all records through current high water,
verifies trigger ingress/monotonic ordering, current health/watchdog/filter
state, paper state, position/stop, fencing/account versions, and earliest-quote
identity. It conditionally commits intent/fill, source evidence, fee, position,
cash, equity, fill_commit_ingress_high_water_seq, health/version values, and
checkpoint before releasing the barrier. An earlier health event wins; a later
one is causally after the fill.

A parent sell fill also moves its blocked child pending and records
activation_after_ingress_seq and monotonic commit; size only from settled state
and a later ask. A normal buy/sell filter failure commits one terminal skipped
intent with no account mutation and does not pause an otherwise healthy runner;
a skipped parent sell atomically cancels its child. Only a protective stop or
recovery_stop filter failure commits stop_execution_blocked and safe_paused
under Section 8.2.2.

Persist UTC target/expiry, window length, process_epoch, commit monotonic value,
and inputs used to construct each process-local monotonic deadline. Never
serialize/reuse an absolute monotonic deadline; old-process pending intents miss.

### 12.2 Startup And Persistent Paper State Machines

Before paper activation, no paper_mode_state row exists. A separate
startup_lifecycle_state is exactly preregistered, activated, or failed. While
preregistered, feeds may append/apply raw ingress for health and startup evidence
only; strategy evaluation, intents, fills, E0, account mutation, and qualification
time are forbidden. Any preactivation condition described elsewhere as requiring
safe_paused instead atomically changes preregistered to failed and never creates
paper_mode_state. Phase-two commit alone changes preregistered to activated and
first creates paper_mode_state=active. A failed/restarted preregistration is
immutable and never reused.

After activation, paper_mode_state is exactly active, safe_paused, or
recovery_armed:

- active alone may evaluate an executable decision, activate/fill a normal
  intent, or perform ordinary stop monitoring.
- Any critical protocol, data conflict, integrity, reconciliation, clock,
  connection, held-watchdog, or stop-execution failure atomically enters
  safe_paused, records immutable incident, misses pending normal intents, cancels
  blocked children, and preserves cash, quantity, highest close, stop, and
  counters. Evidence ingestion continues; decisions/fills do not.
- Explicit recovery requires official repair and completed Section 11 replay,
  healthy new epoch/server clock, already-applied causal filters, DB/ledger/
  account/equity/quantity reconciliation, and no committed-but-unapplied ingress.
  Under the barrier it commits recovery_attempt_id, logical epoch,
  replay_cutoff_close_time and ingress high water, replayed_stop_version,
  decision_not_before_close_time, recovery_commit_ingress_high_water_seq,
  monotonic commit, health version, checksums, and recovery_armed. It never erases
  incident or clock reset.
- recovery_armed blocks normal work. With a position, the first eligible held bid
  beyond recovery ingress/monotonic barriers is consumed exactly once. At or
  below replayed stop it attempts full recovery_stop. Above stop it records
  continued holding. That transition holds the ingress barrier and, if quantity
  remains, sets ordinary_stop_active_after_ingress_seq to the transition's fill/
  state high water, so the recovery quote and earlier quotes cannot trigger an
  ordinary stop. It then returns active. Filter/health failure returns
  safe_paused and requires a new recovery attempt.
- If flat, require fresh accepted bookTicker for both symbols beyond recovery
  barrier. The active transition consumes that evidence, retains
  decision_not_before_close_time = replay_cutoff_close_time, and cannot evaluate
  a paused/backlog bucket; the first later live_eligible close may execute.

Store last fully processed live/replay buckets, every bucket disposition,
position/entry ATR/highest close/stop versions/held count/confirmation counters,
intents/dependencies and mode-neutral decision scope, startup semantic/audit
hashes and lifecycle, activation deadline, owner/fencing/process and
logical_connection_epoch values, socket roles/IDs/all-frame origins/handoff
watermarks/progress, held-symbol accepted-quote watchdog origin,
per-epoch/symbol u watermarks, global next/last-applied
ingress sequences, health version, every commit barrier/clock, UTC windows,
paper/incident/recovery IDs, replay cutoffs/decision floor, active/clean clock
accumulators/anchors, equity, and account version.

### 12.3 Paper Qualification Clock Accounting

Freeze clock_checkpoint_interval = 60 monotonic seconds in
paper_startup_semantic_manifest. Persist active_elapsed_ns and clean_elapsed_ns as
nonnegative integer durations, plus clock_anchor_process_epoch and the current
process-local clock_anchor_monotonic_ns. Absolute monotonic values are never
reused across a process_epoch.

Activation starts both accumulators at zero and sets the anchor to its monotonic
commit. While paper_mode_state=active and all eligibility health remains valid, a
serialized clock-checkpoint timer enters ingress at each 60-second monotonic
deadline; every lifecycle transition and qualification check also checkpoints.
Under the ingress
barrier, drain prior records and credit only the interval from the committed
anchor to the checkpoint/event monotonic. If a watchdog or other health failure
has an earlier causal expiry, credit at most through that expiry, not callback
delay. One transaction updates both elapsed totals and the next anchor.

Entering safe_paused first credits the eligible prefix to active_elapsed_ns,
then atomically sets clean_elapsed_ns=0 and removes the anchor. Recovery to active
starts a new anchor with clean still zero. Paused/recovery time earns neither
clock. A material defect invalidates both with the epoch. Any process restart
retains only committed active_elapsed_ns, discards uncommitted delta and all
offline time, resets clean_elapsed_ns to zero even when flat/incident-free, and
starts a new anchor only after reconciliation permits active. A crash before a
checkpoint therefore earns nothing after the prior commit; a crash after it
cannot double-credit the committed interval.

UTC timestamps are report/cross-check evidence only; wall-clock jumps never add,
remove, or restore qualification duration. The qualification transaction first
checkpoints through its own monotonic commit and requires active_elapsed_ns >=
60 * 24 hours in nanoseconds and clean_elapsed_ns >= 30 * 24 hours in
nanoseconds, plus every other Section 13 gate.

### 12.4 Restart Order

A new process acquires global mutex/fencing, loads committed state, applies all
committed-but-unapplied ingress in order, resolves old-process intents, and
checks startup/incident/position before enabling executable callbacks. A
restarted preregistration becomes failed and uses a new database/boundary. A
committed terminal fill remains terminal; historical replay retries deterministic
IDs idempotently.

Miss old-process pending normal intents and cancel blocked children. If parent
sell committed and child alone remained pending, miss child and retain cash. Any
restart with position records permanent stop_monitoring_gap and safe_paused. Any
unresolved incident, even flat, remains paused. Establish a healthy new epoch,
then explicitly recover. Only an activated, flat, incident-free, fully
reconciled restart with verified REST continuity, resolved intents, applied
ingress, and healthy epoch may return active for a bucket strictly after its
decision floor.

Historical Baseline A/B child runs use separate physical SQLite files. Shadows
use isolated ledger/account namespaces unable to mutate formal account. Each
paper preregistration/epoch has its own physical database. A separate append-only
cohort-control database contains no account state. Raw evidence may be shared
read-only; mutable state never is.

All runs record run_id and the non-null mode-neutral scope_type/scope_id defined
above, candidate_strategy_fingerprint, run_fingerprint, source commit,
configuration, raw/startup/feed/execution manifest hashes, UTC boundaries,
business executable artifact/environment, canonical_ledger_hash, and
canonical_report_hash. Store a presentation renderer_fingerprint separately when
rendered output exists.

## 13. Prospective Paper Gate

Paper startup is a two-phase activation at a preregistered future UTC four-hour
boundary.

During the final warm-up bar, phase one creates a new physical SQLite database
and freezes paper_startup_semantic_manifest before the target boundary. Recompute
the candidate fingerprint from loaded business code. The semantic manifest binds
candidate_strategy_fingerprint, exact business executable artifact/build and
adapter version, target boundary, activation_deadline = target boundary plus 60
seconds, feed/environment/numeric contracts, 60-second clock-checkpoint interval,
exact 500.00-USDT empty-account
convention, exactly 539 already closed warm-up identifiers/checksums per symbol,
and expected final warm-up identifiers. It contains no database, run, nomination,
holdout-pass, epoch, process, or other audit identity. SHA-256 of its canonical
bytes with its own hash field absent is paper_startup_semantic_hash; derive
run_fingerprint from that semantic hash under Section 9.2.

Separately create an immutable identity-bearing paper_preregistration_record that
binds paper_preregistration_id, database_id, run_id/run_fingerprint,
paper_startup_semantic_hash, nomination_record_hash,
final_holdout_pass_record_hash, target/deadline, and the exact empty-account
attestation. SHA-256 of its canonical bytes with its own hash field absent is
paper_preregistration_record_hash. This audit hash is excluded from
run_fingerprint and canonical_report_hash. Append
startup_lifecycle_state=preregistered, acquire Section 12's ownership fence, and
start healthy public feeds. This phase creates no paper_mode_state, E0, paper
epoch, performance bar, or qualification time.

At the target boundary, the final warm-up bar closes as the first performance
bar opens. Phase two is one bounded attempt. Both final warm-up x=true records
must be ingress-committed/applied live in the same uninterrupted healthy
logical_connection_epoch with corrected-UTC receipts no later than
activation_deadline, and the activation transaction itself must commit no later
than that deadline. Equality is accepted; any later time, one-symbol omission,
or x=false/other-stream health without both closes fails. A preregistered
performance-stream frame remains raw startup evidence and can never support
backdated activation. With both closes timely, hold the ingress barrier and
atomically verify all of these:

- The candidate_strategy_fingerprint recomputed from loaded business modules
  equals the nominated candidate and the candidate bound by the verified
  final_holdout_pass_record; the pass record's source ledger/report hashes, hard
  gates, explicit pass, and final_holdout_pass_record_hash all reconcile.
- The database still contains exactly 500.00 USDT, zero BTC and ETH, and no
  position, intent, fill, stop, confirmation counter, holding age, equity
  history, external cash flow, or imported state.
- The formal loader exposes exactly 540 ordered, fully closed four-hour warm-up
  bars per symbol ending at the target boundary, matched by close time and exact
  preregistered identity. They may initialize indicators and recent-cross state
  but create no decision, confirmation count, equity event, return, or fill.
- Owner fencing, WebSocket/server-clock health, source continuity, current
  exchangeInfo/filter readiness, executable artifact,
  paper_startup_semantic_hash, paper_preregistration_record_hash, and every bound
  audit-chain hash still match.

On success, one transaction changes startup_lifecycle_state from preregistered
to activated, creates a unique paper_observation_epoch_id and
paper_mode_state=active, sets decision_not_before_close_time to the final common
warm-up bar's close time, and commits E0 = 500.00 USDT with logical event time
immediately before the target performance open, plus the later
paper_activation_commit_time and activation ingress barrier. The final warm-up
bucket and any equal-time duplicate are therefore ineligible; the first
live_eligible performance close strictly after that floor may decide. The first
performance bucket has the preregistered boundary as open time. Qualification
clocks start at activation commit, not at E0's earlier exchange-event ordering
point. If any assertion fails or the activation transaction would commit after
activation_deadline, atomically set startup_lifecycle_state=failed and create
neither paper_mode_state, E0, epoch, nor clock. Preserve the failed startup
evidence; no late close can revive it, and another attempt requires a new
physical database and future boundary.

A qualifying candidate must:

- Reach active_elapsed_ns >= 60 * 24 hours under Section 12.3 in one valid paper
  epoch. Time in safe_paused/recovery_armed, offline time, and uncommitted
  process-local delta is excluded and extends calendar completion; a material
  software defect instead invalidates the epoch.
- Finish with clean_elapsed_ns >= 30 * 24 hours under Section 12.3 without a
  critical defect, duplicate signal, missed_execution, stop_monitoring_gap,
  missed state transition, unresolved data gap, reconciliation difference,
  pause, or process discontinuity in that clean interval. Incident/pause/restart
  resets clean_elapsed_ns to zero; neither paused state accrues time.
- Finish in active with no unresolved incident or reconciliation difference and
  with every scheduled four-hour bucket and required stream/clock/filter health
  interval accounted for. A miss or critical operational incident is
  disqualifying for the current clean-clock interval and resets that clock; it
  need not invalidate the whole epoch unless the defect rules below say so. It
  remains immutable audit evidence, and qualification is impossible until a new
  complete 30 * 24-hour clean interval has elapsed.
- Have the exact business executable artifact/build and adapter version pass
  scripted
  disconnect, duplicate-message, missing-bar, process-restart, duplicate-launch,
  filter-failure, and transaction-interruption exercises in isolated test
  databases. Deliberate fault injection never writes to the formal observation
  database.
- Reconcile every simulated fill, fee, position, and equity change against the
  immutable ingress, replay, and canonical ledgers.

Before changing code for a suspected defect, open a content-addressed impact
record with symptoms, causal evidence, affected components/outputs, and old
fingerprints. Before any resume, finalize it with the fix, tests, whether
historical signals/qualification could change, and recomputed new candidate,
artifact, adapter, and run fingerprints. If evidence cannot prove confinement to
paper-only infrastructure or presentation, classify it as shared candidate-core
by default.

Defects have these noninterchangeable consequences:

- A defect in shared candidate logic, indicators, risk, cost, accounting,
  metrics, selection, or any component that could change historical signals or
  qualification invalidates the nominated candidate. Validation and holdout may
  not be rerun after inspection; end this research cycle.
- A proven paper-only adapter, matching, timing, filter, storage,
  state-transition, fencing, or recovery defect that can change an intent, fill,
  fee, position, equity, or business report invalidates the complete paper
  epoch. After the fix, recomputed candidate_strategy_fingerprint must remain
  identical while adapter/artifact and run fingerprints identify the new build.
  Preserve the failed database, create a new physical database, use exactly
  500.00 USDT and a new two-phase 540-bar activation, and restart both the
  60-day active-time requirement and clean 30-day clock from zero. Never
  concatenate epochs.
- A presentation-only renderer defect may be corrected from the immutable
  canonical ledger without resetting clocks only when every business output,
  run_fingerprint, and canonical_report_hash remains unchanged; record the new
  renderer_fingerprint separately.
- An operational missed_execution caused by sleep, shutdown, network loss, or
  another external interruption, with no software or semantic change, preserves
  the epoch but atomically resets the clean 30-day clock and excludes paused/
  recovery time from the 60-day active-time accumulator. It prevents
  qualification until a complete new clean interval is earned, but its retained
  pre-interval evidence is not by itself a permanent epoch failure. A
  held-position interruption also follows Section 12's permanent gap and
  explicit recovery procedure.

Paper profitability is recorded but cannot tune thresholds, add indicators,
mix the two baselines, promote the validation runner-up, or relax a gate. Low
natural trade frequency does not justify a new minimum-trade rule or loosening
the strategy; report limited evidence and use historical replay plus isolated
deterministic fault tests to exercise rare paths.

## 14. Testing And Acceptance

Required automated coverage:

- Baseline A fixed vectors for 30/90-day returns, the most recent 180 four-hour
  log returns and sqrt(180) volatility scaling, positive-return eligibility, BTC
  tie-break, two-bar confirmation/reset, holding, ordinary exit, same-open
  replacement, and fresh stage state.
- Baseline B golden vectors for an exact rolling 120-bar reset on every decision,
  pandas rolling/EWM options, RSI zero-loss/flat cases, MACD, KDJ zero range,
  sample-standard-deviation Bollinger Bands, ATR, and the pinned nonstandard
  DMI/ADX denominator. Prove first-row true range equals high-low under pandas
  skipna=True, equal directional moves create zero DM, and denominator NaN/
  infinity behavior matches the pinned implementation.
- Cross-window vectors for previous==0/current>0, previous<0/current==0,
  previous==0/current<0, and previous>0/current==0; ages exactly zero/one/two;
  NaN transition suppression; multiple same-direction crosses; newest-opposite
  cancellation; and requested-direction age None when that direction is no
  longer final.
- RSI-group vectors where raw up/down flags occur on the same or different bars
  and pairs: reversal points become neutral, but raw upward flags remain visible
  to the blocked-entry predicate. Its truth table requires raw RSI up, raw MACD
  up, both raw KDJ-up flags false, volume_score > 0, and
  0 < trend_score < 20, including false boundaries at volume/trend zero and
  trend 20; simultaneous raw RSI down must not unblock it.
- Baseline B predicate and score vectors at and immediately to either side of
  inclusive lower/middle Bollinger location, inclusive 5% MA20 proximity,
  strict 12% buy and 10% sell overextension, RSI6 85, buy 60, strong buy 70,
  sell 18/30, strict MA/volume/rising-close comparisons, and slope zero. Cover
  both total-score zero clamps, fall-back-inside-BOLL's exact previous/current
  upper-band rule with no lower condition, ADX exactly 25, equal DI, and every
  severe-break confirmation.
- Baseline B exit vectors for ADX protection, source-order strong-buy log
  suppression, observation-only sell-risk logging, same-symbol exclusion,
  other-symbol replacement, and the exact counter: entry bar becomes held bar
  one only at close and earliest normal fill is held bar six's open. ATR exits
  are exempt.
- Common risk ATR vectors proving both baselines receive the same finite positive
  SMA14. Cover p_stop clamps exactly at 5%/15%, 1% budget, 30% cap,
  cash-plus-buy-fee cap, post-parent sizing, and downward quantity rounding
  without risk increase.
- Sealer/access-isolation tests proving the non-strategy sealer alone may inspect
  an unreleased stage and emits no price/path-derived value. Training opens only
  training plus warm-up; bound validation children may load warm-up without
  consuming the token, but their first performance-bucket read occurs only in
  the atomic token-consumption/authorization transaction. Holdout manifest/data
  remain unreadable until nomination; afterward warm-up may load without token
  and the first performance read uses the same atomic rule. Assert every denial
  occurs before file/query contents are returned.
- Historical boundary tests for fully closed bars, half-open stage membership,
  exact 540-bar warm-up and exclusion, fresh account state, two-symbol buckets,
  final-boundary cancellation, and rejection of future, same-signal-bar,
  unclosed, out-of-manifest, duplicate, or missing data.
- Training-defect tests proving every affected candidate/run fingerprint is
  recalculated and every affected baseline restarts from training start; only
  its final clean fingerprint can enter validation.
- Validation-cohort tests freezing the semantic entrant manifest and every
  ordered child candidate/run/database binding before token consumption.
  Random audit identities must change cohort_activation_record_hash but not
  activation_coordinator_fingerprint or canonical_report_hash; a coordinator
  protocol/module/serialization change must change the semantic fingerprint.
  The token commits once with the first performance bucket and authorizes all
  children before any result is visible; rollback consumes nothing; one-child
  crash recovers only the identical cohort/child bindings. Adding, removing, or
  changing a child invalidates the activation without a replacement token.
- Singleton-holdout activation tests for warm-up-only nonconsumption, atomic
  first-bucket consumption, exact recovery, pre-nomination rejection, forbidden
  runner-up promotion, and cycle termination after any post-consumption code,
  data, config, dependency, metric, accounting, matching, or execution change.
- Fingerprint mutation tests proving baseline-specific/shared semantic modules,
  numeric contracts, serialization, and the semantic_dependency_lock_hash alter
  candidate_strategy_fingerprint. A pandas, NumPy, Python, decimal/libmpdec, or
  other candidate-semantic dependency change must change that fingerprint and be
  rejected across training, validation, holdout, and paper. Run IDs, outcomes,
  volatile host/process fields, pure OS/host differences, broker-adapter changes,
  and presentation metadata do not alter candidate identity; applicable adapter,
  historical raw/stage/execution/environment, or paper startup semantic changes
  alter run_fingerprint. Changing
  nomination/pass/preregistration/database/run/epoch audit identities changes
  their audit hashes but not paper run_fingerprint or canonical_report_hash.
  Future feed events extend raw_event_manifest_root without changing the frozen
  paper run fingerprint. Holdout/paper recompute loaded code, never trust a label.
- Nomination/report tests bind cohort/nominated-child identities, ledger/report
  hashes, and ordered comparison path. nomination_record_hash and
  final_holdout_pass_record_hash use canonical bytes with their own hash fields
  absent; any bound-field mutation changes the applicable audit hash. Cross-stage
  checks require identical candidate_strategy_fingerprint while validation,
  holdout, and paper retain distinct correctly bound identities. Paper rejects a
  missing/forged/failed/mismatched pass or preregistration record. Its canonical
  payload contains paper_startup_semantic_hash, not a nonexistent coordinator or
  identity-bearing audit hash. Renderer-only changes preserve run_fingerprint,
  business values/clocks, and canonical_report_hash; any business payload change
  alters that hash.
- Historical matching tests for next-open R, gap-stop R, intrabar-stop R, both F
  sides, exactly-once slippage, F-based commission/cash, entry-bar stop,
  stop-before-normal-sell priority, one sell, and sell/fee settlement before a
  dependent buy.
- Equity vectors proving E0 precedes first fill; unchanged raw marks lose
  slippage plus commission subject only to the separately persisted
  accounting_rounding_residual; every admitted bar creates one regular close;
  and E0, fills, closes, and exactly one later terminal event form causal
  drawdown order. Include adversarial HALF_EVEN values producing positive and
  negative one-Q18-unit residuals.
- Terminal vectors for flat, one-asset strategy, and simultaneous BTC/ETH
  benchmark holdings. Verify
  `C + sum_j(q_j * R_j * (1-s) * (1-c))`, per-asset reserve exactly once, E0 in
  the running peak, and a terminal-created maximum drawdown. Terminal affects
  total return/CAGR/Calmar/MDD/friction but not periodic returns, Sharpe,
  Sortino, volatility, turnover, fills/trades, win rate, or holding duration.
- Metric vectors for population volatility/downside deviation, exact UTC CAGR,
  deterministic drawdown tie/duration, turnover, exposure, time in market,
  fill-to-fill holding, UTC grouping, largest-winner contribution, profit ratios,
  friction, and nonfinite denominators. Attribution vectors must cover historical
  open/gap/intrabar-stop R, paper ask/bid normal/stop/recovery R, regular raw
  close, raw terminal R plus separately deducted reserve, same-open replacement,
  and deterministic positive/negative/zero accounting_rounding_residual
  assignment; BTC+ETH including residuals must equal E_final-E0 while reported
  friction excludes the residual.
- numeric_protocol_v1 vectors for exact Decimal parsing, rejection beyond scale
  18, Q18 operation order, HALF_EVEN ties, FLOOR grid ticks, negative-zero
  normalization, shortest-float conversion, metric nonfinite tags, indicator-only
  FLOAT64 tags, non-distributive-operation residuals, UTF-8/NFC/key/array ordering,
  self-hash-field omission, and one-field hash mutation.
- Hard-gate boundary vectors proving 0.000000000000000000 return fails,
  0.000000000000000001 passes, 0.100000000000000000 drawdown passes, and
  0.100000000000000001 fails. Test nonfinite inputs and every other hard-failure
  category independently.
- Selection matrices for zero/one/two passers; symmetric skipping when either
  Sortino or Calmar is nonfinite, zero, or negative; both-positive finite ratios;
  one-sided invalid lower-is-better metrics; gaps just below and exactly 10%;
  all-skipped/tied Baseline A fallback; and final failure without runner-up.
- Quantity-filter tests for exact decimals, LOT_SIZE/MARKET_LOT_SIZE intersection,
  disabled-zero fields, contradictory/unknown fields, downward grids, active
  MIN_NOTIONAL/NOTIONAL bounds, and zero quantity. Historical notional must use
  quantity*R under raw_reference_price_v1, never F/current averages.
- Paper-filter tests for exchangeInfo/status/capability and each price proxy at
  age/RTT exactly and just beyond 60 seconds/two seconds; avgPriceMins positive
  with exact `/api/v3/avgPrice` mins/closeTime; avgPriceMins zero with
  `/api/v3/ticker/price`; multiple proxy requirements; future avgPrice closeTime
  or negative monotonic age; and missing/mismatched evidence. Evidence after the
  trigger quote is forbidden, and the first failed quote attempt may not wait
  for refresh or another quote. A normal buy/sell failure becomes terminal
  skipped without pausing a healthy runner (and a skipped parent cancels child);
  stop/recovery-stop failure alone commits stop_execution_blocked/safe_paused.
- WebSocket protocol tests for exact subscriptions/timestamps/pong payload,
  no pong-ACK assumption, control limit, serverShutdown, and all-frame health.
  A required preregistration-socket shutdown fails startup; an activated
  authoritative/pre-handoff-predecessor shutdown changes health, enters
  safe_paused, and requires a new epoch; an unaccepted-candidate shutdown retires
  only that candidate; and a planned-retired-predecessor shutdown is audit-only.
  Race shutdown against atomic handoff in both ingress orders and require one
  role-consistent outcome; a pre-fill authoritative shutdown prevents that fill.
  all_frame_watchdog_origin starts at socket-open commit; first and later frames
  at exactly 60 seconds pass, while just-over-60 fails even when the frame beats
  the timer. The late frame cannot advance origin or act. For a required
  preregistration socket this sets startup failed with no paper state; for an
  activated authoritative socket it commits transport_silence_gap/safe_paused
  and requires new logical_connection_epoch/recovery; an unaccepted successor
  alone is retired while a healthy predecessor continues. Also race pong-write,
  malformed/close/error, and fill commit under ordered ingress.
- Held-symbol watchdog tests at exactly and just over 60 seconds, including a
  late quote racing the timer. After handoff, valid-but-stale and duplicate
  successor traffic may maintain all-frame health but must not advance the held
  origin; more than 60 seconds without an accepted held-symbol quote commits
  stop_monitoring_gap and safe_paused before a frame can act. Only a newly
  accepted u advances that origin; conflict/malformed/invalid traffic never does.
  Flat stale data only blocks fills.
- Overlap tests freeze handoff_watermark and successor_progress_u. A successor
  above frozen handoff but stale below the shared watermark does not prove
  progress; an accepted one strictly above both does. Cover successor-first and
  predecessor stale/higher/equal messages, readiness, and noncontiguous u. A
  candidate successor failure while predecessor is healthy does not alter
  logical health; predecessor failure before acceptance forces a new epoch.
  Handoff atomically changes socket roles; planned predecessor close afterward
  is audit-only/no pause, whereas authoritative successor failure is a gap.
  Cover new-epoch seeding and equal-payload conflict recovery.
- Kline state tests for normal repeated x=false updates, first x=true freeze,
  later identical x=true duplicate, differing x=true conflict with immutable
  evidence and exact REST resolution, post-close x=false stale handling, and
  unresolved reconciliation remaining paused. For a conflict after first-version
  consumption, separately prove: no paper decision permits REST-selected
  replay_only recovery; a consumed decision without fill invalidates the paper
  epoch; a fill also invalidates it without reversing cash; a preregistration
  conflict sets startup failed without paper_mode_state; consumed validation/
  holdout ends the cycle; and training discards/replays.
- REST-repair tests for a causally timely close bound to an unchanged healthy
  decision epoch and committed within its 60-second window versus late,
  cross-epoch, unhealthy, or conflicting repair restricted to nonexecuting
  replay. Every server-time, Kline, exchangeInfo, avgPrice, and ticker response/
  failure receives ingress_seq; evidence must be committed and applied before
  its decision/activation/quote cut. Explicitly race a response received by the
  callback but not yet ingress-committed/applied. Record exact REST evidence and
  never use historical open as a fill.
- Normal-intent timing tests for target_time minus one millisecond, exactly
  target_time, exactly target_time plus 60 seconds, just after, stale server
  time, excessive RTT, wall-clock jumps, process-local monotonic nonextension,
  process/epoch/health changes, no cached quote, and bid/ask-only R.
- Ingress race tests for decision, entry-stop, closed-stop, child, and recovery
  commits. ingress_seq equal to a high water is ineligible; greater sequence with
  monotonic receipt before commit is ineligible; only both greater/after plus
  unchanged health may act. For normal, stop, and recovery fills, inject close,
  error, serverShutdown, malformed-frame, watchdog, and filter-expiry events
  between preliminary eligibility and commit; drain/recheck under the barrier
  must make the earlier event win. A postcommit event is causally later and
  rollback activates nothing.
- Paper-stop tests for first eligible best bid, raw R/adverse F, entry-quote
  exclusion, normal-sell conflict, one fill, full-quantity filters, one terminal
  skipped stop on filter failure, retained position/stop/equity, blocked child,
  no quote retry, and later recovery_stop only after a new explicit barrier.
- Nonexecuting-replay tests for chronological two-symbol bars, prebar stop
  crossing classified once as missed, held count/highest close/never-loosening
  stop advancement, Baseline A confirmation state, and no cash/quantity change or
  fill. Freeze both replay_cutoff_close_time and ingress high water, race older
  and newer concurrent arrivals, extend before recovery when required, and prove
  a bucket marked replay_only can never become live_eligible. Actionable expired
  orders create one miss; ordinary hold buckets and later crossings create none.
- Persistent-state tests for every legal active -> safe_paused -> recovery_armed
  path and illegal shortcut. Cover held first post-barrier bid above/equal/below
  replayed stop, consume it exactly once, establish a strictly later ordinary-
  stop barrier when quantity remains, and prove the recovery quote cannot trigger
  another stop. Cover flat two-symbol freshness,
  decision_not_before_close_time = replay cutoff rejecting backlog/equal buckets
  while accepting the first later live_eligible close, fresh-filter failure, new
  recovery failure, immutable incidents, unresolved flat-incident restart, and
  no normal decision/fill while paused or armed.
- Ownership tests for a `Global\` Windows mutex with fixed restricted DACL and key
  derived from resolved volume/file identity plus immutable database UUID.
  Prove case, 8.3, symlink/junction, and alternate-path aliases and different
  Windows sessions cannot acquire a second owner. Cover atomic process_epoch/
  fencing claim, duplicate Task Scheduler launch rejection, stale-writer denial
  on every write, owner loss, successor takeover, and open-position restart
  entering gap/recovery before executable callbacks.
- Transaction/restart tests for parent failure, child sizing/fresh quote,
  crash before/after each commit, and a committed ingress record followed by
  crash before derived-state application. Recovery applies strictly from
  last_applied_ingress_seq exactly once before any business callback. Also cover
  old-process deadline nonreuse and pending misses, parent-filled/child-pending
  restart, duplicate/concurrent callbacks, UNIQUE(fill.intent_id), account/
  fencing conflicts, rollback, and historical idempotency. Assert non-null,
  collision-free scope_type/scope_id keys for training stage, reserved activation,
  and paper epoch; cross-mode identifiers are never substituted.
- Historical shadow-ledger tests instrumenting exactly one activation,
  source-data read, and
  chronological pass. Each balance has independent state; the formal 500 ledger,
  canonical_ledger_hash, metrics, gates, and selection inputs are identical with
  shadow execution enabled/disabled. Shadow returns cannot affect gates or
  selection. Missing/crashed/mismatched shadows are completeness defects, not
  performance failures: training fixes force full affected replay, preactivation
  defects block token use, and post-consumption validation/holdout defects
  invalidate the stage/end the cycle without rerun.
- Sparse-evidence tests proving low/zero trades, exposure, or time in market adds
  the required warning but never an unstated hard gate or ranking change.
- Benchmark tests for E0, initial/monthly raw-open execution, one frozen R vector,
  target quantity derived from weight * pretrade equity divided by R rather than
  F, and no target/equity recomputation after a leg. Cover BTC-before-ETH
  reductions/increases, post-reduction cash-and-fee capping, filters/rounding,
  dual-asset terminal valuation/attribution, and nonparticipation in gates or
  selection.
- Two-phase paper-start tests separate paper_startup_semantic_hash from
  identity-bearing paper_preregistration_record_hash. Random database/run/
  nomination/pass IDs cannot alter paper run/report hashes; target, artifact,
  adapter, feed, numeric, balance, warm-up, deadline, or environment changes do.
  Verify 539 plus expected final IDs and both live x=true closes into exact 540,
  500.00 USDT/zero state, and loaded code. Both closes and activation at exactly
  target+60s pass; just after, one missing symbol, x=false/other healthy traffic,
  a preregistered protocol conflict, or restart atomically freezes failed with no
  paper_mode_state/E0/epoch/clock. Successful activation commits
  decision_not_before_close_time equal to final warm-up close: that bucket and an
  equal-time duplicate are rejected, while the first later live_eligible
  performance close is accepted. Late data cannot revive a failed startup; retry
  uses a new database/future boundary.
- Paper-clock tests use committed per-process monotonic intervals only. Cover
  exact and one-nanosecond-below 60*24 active and 30*24 clean thresholds,
  60-second checkpoints, earlier watchdog expiry, forward/backward wall jumps,
  safe_paused/recovery exclusion, incident atomic clean reset, crash immediately
  before/after checkpoint, no double credit, offline time zero, active total
  retained across same-epoch restart, and clean reset even for flat
  incident-free restart. Completion requires active/no unresolved issue and the
  exact business artifact/adapter that passed isolated fault tests.
- Defect-consequence tests requiring a content-addressed impact record and
  shared-core default when attribution is unproven. Shared defects end the cycle;
  proven paper-only material fixes preserve candidate fingerprint, change build/
  run identity, archive the failed database, and restart a new two-phase epoch
  plus both clocks; renderer-only fixes preserve business hash/clocks; external
  interruption preserves epoch but resets/excludes clocks as specified.
- Storage and architecture tests proving baseline child runs and paper epochs use
  separate physical databases, shadow namespaces cannot mutate formal state,
  baselines do not import each other, and the new package never imports
  cross_signal_strategy at runtime.
- Repeatability tests proving identical canonical inputs/code/config produce
  identical signals, intents, fills, equity events, ledgers, metrics, selection,
  and canonical_report_hash. Excluded renderer metadata leaves the hash stable;
  one business-field mutation changes it.

Normal tests must not read G-drive ETF data. Golden expectations for Baseline B
are checked-in values derived once from the pinned provenance snapshot, so
ordinary tests do not import or execute the ETF strategy.

The first-version milestone is complete only when:

- All automated tests above pass.
- Data, stage/cohort, fingerprint, execution, artifact, and software manifests
  reproduce immutable ledgers and canonical reports.
- Each approved stage is activated and run in order without a forbidden retry.
- HTML/CSV output shows equity, returns, drawdowns, trades, exposure, attribution,
  costs, terminal reserve, skipped/missed actions, incidents, replay/recovery,
  diagnostics, and evidence-limit flags.
- The formal paper database satisfies Section 13's two-phase startup, active-time,
  clean-time, final-state, and reconciliation gates.
- No private API, real-order entrypoint, key configuration, or hidden live mode
  exists.

## 15. Delivery Sequence

1. Create the isolated package and domain contracts.
2. Implement immutable data ingestion and stage isolation.
3. Implement and golden-test Baseline A and Baseline B.
4. Implement the common risk and matching engines.
5. Implement SQLite state, reporting, and deterministic backtests.
6. Complete the training-stage comparison and freeze eligible versions.
7. Consume the single validation cohort activation and nominate one candidate.
8. Consume the singleton final-holdout activation and preserve its pass record.
9. Implement and operate the public-data paper runner.
10. After both the 60-day active-time and final continuous 30-day clean-time
    paper gates pass, stop and request a separate real-trading design decision.

No implementation step may use a later stage to revise an earlier frozen rule.

## 16. Authoritative External References

The design was checked on 2026-08-03 against Binance's official Spot sources:

- [Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
- [Spot WebSocket market streams](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)
- [Spot symbol and exchange filters](https://developers.binance.com/en/docs/products/spot/filters)
- [Binance transaction-fee explanation](https://www.binance.com/en/academy/articles/how-to-calculate-transaction-fees-on-binance)

Exchange endpoints, payloads, filters, rate limits, and actual account fees may
change. Recheck these official references when writing the implementation and
again before any future real-trading design. The frozen 0.10% per-side fee in
this research contract is a conservative modelling choice, not a claim about a
particular account's current fee tier.
