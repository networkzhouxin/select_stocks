# Binance Spot Dual-Baseline Quant Research System Design

Status: approved in conversation on 2026-08-03 and frozen for implementation planning.

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

Before the first experiment, audit both symbols for:

- A common continuous history covering the proposed study periods.
- Exact four-hour UTC alignment.
- Unique opening times and a four-hour cadence.
- Finite OHLCV values, valid OHLC relationships, positive prices, nonnegative
  volume, and no duplicate or unexplained missing bars.
- A final closed-bar marker for streaming data.

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
- At the stage end, value an open position at the last close after reserving the
  modelled adverse sell slippage and sell commission. Label this
  terminal_valuation; do not invent a strategy sell or carry the position
  forward.

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

- RSI: delta = close.diff(); gain is positive delta and otherwise zero; loss is
  absolute negative delta and otherwise zero. Average gain and loss use EWM
  alpha = 1 / period, min_periods = period, and adjust = True. RSI is
  100 - 100 / (1 + average_gain / average_loss). When average loss is zero,
  return 100 if average gain is positive and 50 if both are zero.
- MACD: fast and slow close EWM use their spans with adjust = False. DIF is
  fast EMA minus slow EMA; DEA is DIF EWM with signal span and adjust = False;
  histogram is 2 multiplied by (DIF - DEA).
- KDJ: lowest low and highest high are rolling N-bar extrema. RSV is
  100 multiplied by (close - lowest) / (highest - lowest), with a zero range
  treated as NaN. K is RSV EWM with com = M1 - 1 and adjust = False; D is K EWM
  with com = M2 - 1 and adjust = False; J = 3K - 2D.
- Bollinger Bands: rolling simple mean and sample standard deviation with
  ddof = 1; upper/lower are the mean plus/minus two standard deviations.
- ATR: true range is the row maximum of high-low, absolute high-previous-close,
  and absolute low-previous-close. ATR is the 14-bar simple rolling mean.
- DMI/ADX: positive and negative directional movement follow the pinned source
  comparisons. Sum each over 14 bars. Divide by the rolling 14-bar sum of the
  ATR series, multiply by 100, calculate DX from the absolute DI difference
  divided by the DI sum, and take a 14-bar simple rolling mean for ADX. This
  deliberately preserves the source's nonstandard denominator.
- MA5/10/20/60 and VOL5/20 are simple rolling means. MA20 slope is current MA20
  minus MA20 five bars earlier.

For a three-bar cross window, calculate fast minus slow and scan the three most
recent transitions from oldest to newest. A transition from less than or equal
to zero to strictly positive records above; a transition from greater than or
equal to zero to strictly negative records below. The last recorded direction
in the window wins. A prior cross in the opposite direction therefore cancels
the older direction. For the RSI group, any simultaneous up and down result
across RSI6/12 and RSI6/24 makes the group direction neutral and awards neither
group's RSI cross points.

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

Total buy score is the nonnegative sum of those groups. New entry also requires:

- Buy score at least 60.
- Sell score below 30.
- RSI6 below 85.
- At least one valid location state: lower-to-middle Bollinger location,
  middle-band upward cross, or close near MA20.
- Close not more than 12% above MA20.
- The frozen cross-v0.3.2 blocked-entry combination remains blocked: RSI upward
  cross plus MACD upward cross, no KDJ upward cross, positive volume score, and
  trend score strictly between 0 and 20.

Sell reversal score:

- RSI6 crossing below RSI12: 12.
- RSI6 crossing below RSI24: 12.
- DIF crossing below DEA: 10.
- K crossing below D: 6.
- J crossing below D: 5.

The RSI downward scores count only when the RSI group direction is
unambiguously down within the three-bar cross window.

Sell risk score:

- More than 10% above MA20 while RSI6 turns down: 8.
- Close below a falling MA10: 10.
- Prior close above the upper Bollinger Band and current close back inside: 6.

Total sell score is the nonnegative sum of the sell reversal and sell risk
groups.

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
against a normal signal exit. A sell score from 18 through 29 is logged for
observation only and cannot change the position.

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
risk value remains the sizing and stop authority for both baselines. At
execution:

- Backtest execution reference is T+1 open multiplied by 1.0005.
- Paper execution reference is the first eligible fresh ask defined in Section
  8.1, multiplied by 1.0005.
- Account risk budget = current equity multiplied by 1%.
- Initial stop percentage =
  clamp(2.5 multiplied by frozen risk ATR14 divided by execution reference,
  5%, 15%).
- Modelled round-trip trading friction = 0.30%.
- Risk-sized trade notional =
  equity multiplied by 1% divided by
  (initial stop percentage plus 0.30%).
- Final trade notional is the smaller of risk-sized notional, 30% of equity,
  and available USDT divided by 1.001 so the 0.10% buy commission cannot
  overdraw cash.

Trade notional means fill price multiplied by base-asset quantity before
commission. Charge commission as its USDT-equivalent cash amount for normalized
research accounting, and calculate received base quantity from trade notional
divided by fill price. Quantity is always rounded down. Recalculate trade
notional after rounding; if it fails the frozen quantity or notional contract,
skip it. Never round up and exceed risk.

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

Paper trading uses one logical Binance Spot public WebSocket connection_epoch
carrying the UTC Kline and bookTicker streams for both symbols. Normally it has
one physical socket. Because an official Spot stream connection is valid for
only 24 hours, establish a successor before hour 23 while the predecessor is
still healthy.

A planned rollover preserves the logical connection_epoch only when the
successor is fully subscribed and has delivered fresh Kline and bookTicker
messages for both symbols before the predecessor retires. During the overlap,
messages from both socket_id values enter one serialized ingestion queue.
Deduplicate closed Klines by bar key. For bookTicker, persist last_accepted_u
per symbol across all physical sockets and accept a message only when its
official update ID u is strictly greater. Discard u less than or equal to that
watermark as duplicate or stale; among acceptable updates, local receive order
is authoritative. If no continuous overlap exists, treat it as a real
connection gap, increment the logical epoch, and apply the missed-execution or
stop-monitoring-gap rules.

The ingestion queue assigns every accepted message a strictly increasing global
stream_seq, timestamps it immediately, and persists its raw payload, socket_id,
logical connection_epoch, and update identifier. Only bookTicker is an
execution-price source: use its best ask for a buy and best bid for a sell,
requiring a finite positive price and positive displayed quantity. Do not race
it against a trade stream or reuse a cached quote.

For a normal paper intent, target_time is the intended next-bar open. An
eligible quote must:

- Belong to the same uninterrupted logical connection_epoch as the two
  closed-bar inputs and completed decision.
- Have stream_seq greater than decision_committed_seq and be received after the
  decision transaction commits.
- Have corrected receive time at or after target_time and no later than
  target_time plus 60 seconds according to a local UTC receive clock whose
  offset has passed a fresh Binance server-time check.

A fresh server-time check is a public server-time request completed within the
preceding 60 seconds with round-trip time no greater than two seconds. Estimate
the UTC offset against the local send/receive midpoint and apply that offset to
receive timestamps. If no qualifying sample exists, treat the clock as stale;
do not widen the deadline.

The first eligible quote is authoritative. Paper buy fill price is best ask
multiplied by 1.0005; paper normal sell fill price is best bid multiplied by
0.9995. Record signal time, target time, Kline receive times, decision commit
time, quote receive time, stream identifiers, raw bid/ask, and simulated fill
time.

If the deadline expires, the connection epoch changes, the clock check becomes
stale, or the process restarts before a fill commits, transition the normal
intent to missed. Never fill it from a later quote or a historically retrieved
Kline open. A dependent buy is canceled when its parent sell is missed.

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

Paper stop monitoring uses only serialized bookTicker best-bid updates from the
connection defined in Section 8.1. The first eligible best bid less than or
equal to the active stop triggers one synthetic sell at that observed bid
multiplied by 0.9995, followed by the 0.10% commission. It never fills at a stop
price learned later from a closed Kline.

For an existing position, the triggering quote must be in the uninterrupted
monitoring connection_epoch and have stream_seq greater than
stop_active_after_seq. A closed-Kline stop update records its commit sequence as
the new stop_active_after_seq and never applies to an earlier quote. For a new
entry, stop monitoring begins only after the entry fill transaction commits;
stop_active_after_seq equals the entry quote's stream_seq. Thus the entry quote
cannot also manufacture a stop fill.

If the first eligible quote at a next-open event is at or below the stop while a
normal sell is pending, classify the single sell as a stop and cancel the normal
sell. Activate any dependent other-symbol buy only after that sell and its fee
commit; the child must wait for a new other-symbol ask with stream_seq greater
than the parent fill and still inside the original 60-second deadline.

Any disconnect, stale clock, or process loss while a position is open creates a
critical stop_monitoring_gap and enters safe paused mode. Backfilled trades or
Klines may classify a proven historical stop crossing as missed_execution, but
must never create a retroactive fill. After explicit recovery, the first new
eligible bid at or below the still-active stop may create a recovery_stop fill
at that bid with adverse slippage; if it is above the stop, keep the position.
Either outcome preserves the monitoring-gap record and resets the clean paper
clock.

The triggering quote, stop intent, fill, fee, position mutation, equity
mutation, and checkpoint commit atomically under the conditional state rules in
Section 12. Duplicate or competing quotes cannot create a second stop fill.

### 8.3 Quantity Filters

Paper trading queries and records the current official Binance Spot symbol
filters. Historical Binance filter history is not assumed to exist. Formal
backtests use one frozen, versioned execution-contract snapshot for both
baselines and clearly label it as a present-day feasibility constraint, not a
claim about historical exchange rules.

Apply every current public filter relevant to a synthetic market order. At
minimum, evaluate LOT_SIZE, MARKET_LOT_SIZE, and any active MIN_NOTIONAL or
NOTIONAL constraint returned by the frozen execution-contract snapshot. If an
applicable filter cannot be interpreted, skip the order instead of assuming it
passes. Use decimal arithmetic and downward quantization.

## 9. Evaluation And Selection Protocol

### 9.1 Hard Failure Rules

A baseline or candidate fails a stage if any of these occurs:

- Future data, an unclosed bar, or same-signal-bar execution affects a decision.
- Fees or slippage are omitted.
- A runtime, accounting, data-boundary, or quantity-filter error remains.
- Net return after costs is not positive for the stage.
- Maximum drawdown exceeds 10% for the stage.
- Results cannot be reproduced from the recorded data, configuration, and code
  fingerprints.

Training is a development window, not machine-learning fitting. It may expose
an implementation defect. A fix is allowed inside this cycle only when it makes
the implementation conform to this frozen document; discard the affected run
and replay it from the stage start. No reserved result may choose or tune a
parameter, rule, indicator, or fallback.

A semantic strategy change is not a fix to either frozen baseline. It ends this
research cycle and requires a new preregistered cycle with genuinely unused
validation and holdout periods. A validation or holdout period already viewed
in this cycle cannot become unseen again.

### 9.2 Stage Progression

1. Run both frozen baselines independently on 2018-2021.
2. If neither passes training, stop the research cycle without activating
   validation or final holdout. Otherwise, only training-pass baselines enter
   2022-2023 validation.
3. Run validation once without changing either baseline.
4. If every entrant fails validation, stop the cycle without activating the
   final holdout or modifying a rule for a retry.
5. If only one baseline passes validation, fingerprint it as the nominated
   candidate.
6. If both pass, compare finite positive Sortino first, then finite positive
   Calmar, maximum drawdown, friction as a percentage of starting equity, and
   turnover in that order. For every numeric comparison define relative_gap =
   abs(a - b) / max(abs(a), abs(b), 1e-12); a gap below 10% is a tie. If exactly
   one Sortino or Calmar is finite and positive, that baseline wins that
   comparison. If neither is finite and positive, move to the next metric. For
   two finite positive ratios, higher wins when the gap is at least 10%. For
   drawdown, friction, and turnover, lower wins when the gap is at least 10%.
7. If all comparisons remain effectively tied, fingerprint and nominate the
   simpler Baseline A.
8. Run only the nominated fingerprint once on
   [2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z).
9. If the candidate fails, stop the cycle. Do not inspect the holdout and then
   promote the validation runner-up.
10. If it passes, freeze that fingerprint and begin prospective paper
    observation.

Only these two preregistered baselines exist in this research cycle. Record
every conforming defect correction, discarded run, and failure. Do not create a
third candidate, reopen validation, or revise a baseline after a reserved result.

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

Modelled slippage is the absolute difference between the unadjusted execution
reference and synthetic fill price multiplied by filled quantity. Total
friction is slippage plus commissions over all fills, plus the separately
identified adverse slippage and commission reserve in terminal_valuation.
Friction as a percentage of gross profit uses gross winning profit as its
denominator and is nonfinite when gross winning profit is zero.

Use these same-maximum-crypto-allocation reference portfolios:

- 30% BTC plus 70% USDT.
- 30% ETH plus 70% USDT.
- 15% BTC plus 15% ETH plus 70% USDT.

Reference weights are determined only by the UTC calendar and never by a closed
signal. Allocate at the first admitted bar's open, then rebalance at the open of
the first admitted four-hour bar of every later UTC calendar month. Mark
pretrade equity at that open and compute all target notionals from that same
equity. Execute reductions first in BTC, ETH order, settle their costs, then
execute increases in BTC, ETH order. Apply the strategy's slippage, commission,
filter snapshot, and downward rounding contract to every leg. Do not rebalance
between those opens; use the same terminal-valuation rule at stage end.

Benchmarks appear only in comparison reports. They cannot pass or fail a
baseline, nominate a candidate, or change a rule.

Do not present a comparison with a 100% crypto benchmark as a like-for-like
capital-risk comparison. Treat idle USDT as zero yield.

Formal passage, metrics, and selection use only the frozen 500.00 USDT main
balance. Also report unchanged-rule feasibility at 300, 500, and 1000 USDT to
expose minimum-notional and rounding effects for small capital. The 500-USDT
case must reproduce the formal run; 300 and 1000 USDT are diagnostics only and
cannot fail a stage, order candidates, select a baseline, or change a rule.

Let E0 be starting equity and E1 through En be the regular equity values at
successive four-hour closes. Define each periodic return as
ri = Ei / E(i-1) - 1. Use population denominators throughout:

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
nonfinite. In particular, no negative return makes Sortino nonfinite and causes
selection to move to Calmar. The risk-free rate and minimum acceptable return
are both zero.

Calculate CAGR from exact UTC elapsed duration using 365.25 days per year.
Calmar is CAGR divided by maximum drawdown; report it as nonfinite when drawdown
is zero or either input is nonfinite.

Turnover is the sum of the absolute precommission notional of every filled buy
and sell, including both legs of a same-open replacement and all stop fills,
divided by the arithmetic mean of regular four-hour close equity. It is a raw
stage ratio, not annualized. A terminal_valuation is not a fill and is excluded.
The selection friction percentage is total commission plus modelled slippage
divided by the frozen starting equity.

Record equity after every fill and at every four-hour close. Compute maximum
drawdown from the combined chronological fill-and-close equity events, while
Sharpe, Sortino, volatility, downside volatility, and turnover's denominator
use only the regular four-hour close series.

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
- Start the paper runner at system startup or user login.
- Store state in a local SQLite database and write local structured logs,
  HTML reports, and CSV exports.
- The first version never asks for or stores a Binance API key.
- Do not require a cloud server.

When the computer is offline, backfill market data after restart for indicator
continuity, but never create a simulated historical fill that could not have
been placed at the time. With no open position, record an affected normal intent
as missed_execution and resume from the next eligible closed-bar decision. If a
position was open at any point in the outage, apply the stop_monitoring_gap and
safe-pause recovery contract in Section 8.2 instead of automatically resuming.

The formal paper-observation clock assumes the computer remains running and
automatic sleep is disabled. Intentional or accidental downtime remains visible
in the stability report.

## 12. State, Idempotency, And Failure Handling

- Identify a bar by symbol, interval, and open time.
- Do not evaluate a portfolio bucket until validated closed BTCUSDT and ETHUSDT
  bars with the same close time are both present. This two-symbol watermark is
  mandatory even when only one symbol is held. If one bar is missing, pause the
  whole bucket and repair data before ranking or signalling.
- Identify a portfolio decision uniquely by run_id, stage_id, strategy version,
  configuration hash, execution-contract hash, and bar close time.
- Give every DecisionPlan an execution_group_id. A standalone order begins
  pending. For an ordered replacement, the sell is the parent pending intent
  and the buy is a blocked child carrying depends_on_intent_id. The child can
  become pending only after the parent fill and fee settle; any skipped, missed,
  or canceled parent atomically cancels the child.
- Give every intent and fill an immutable identifier. Allowed intent transitions
  are blocked -> pending or canceled, and pending -> filled, skipped, missed, or
  canceled. Terminal states never transition again; blocked cannot fill,
  skip, or become missed directly.
- Enforce a one-to-zero-or-one fill relationship with UNIQUE(fill.intent_id).
  Every terminal transition uses a conditional update that succeeds only from
  the expected nonterminal state. A callback may not invent a new fill ID to
  bypass that constraint.
- Process market events through one serialized portfolio event loop per
  database. Use SQLite BEGIN IMMEDIATE plus an account-state version check for
  every mutation, so competing quote callbacks cannot overwrite cash, position,
  or equity.
- Commit the two-symbol bucket inputs, signals, complete DecisionPlan, process
  and connection epochs, decision_committed_seq, and checkpoint in one atomic
  decision transaction. Never keep a transaction open while awaiting a quote.
- A normal fill transaction conditionally moves its pending intent to filled
  and commits the source quote, fill, fee, position, cash, equity, account-state
  version, and checkpoint together. A parent sell fill also moves its blocked
  child to pending and records the child's activation sequence in that same
  transaction. Size that child only from updated cash and its later eligible ask.
- A paper stop transaction conditionally verifies the active position and state
  version, creates the stop intent and its unique fill, and commits the source
  quote, fee, position, cash, equity, and checkpoint atomically.
- Store the last fully processed portfolio bucket and the active position,
  entry ATR, highest closed price, active stop, held-bar count, Baseline A
  confirmation counters, all blocked or pending intents and dependencies,
  process and logical connection epochs, physical socket IDs, per-symbol
  last_accepted_u values, global quote sequences, deadlines, paper equity, and
  account-state version.
- On restart, load only committed state. A committed terminal fill remains
  terminal. Historical replay may retry the same deterministic event and IDs
  idempotently. Paper execution may not fill an intent from an earlier process
  epoch: mark an old pending normal intent missed before consuming new quotes;
  cancel its blocked child. If a parent sell was already committed and its child
  alone remained pending, mark that child missed and retain the settled cash.
  An open position also creates the stop_monitoring_gap required by Section 8.2.
- If a Kline gap is detected, pause decisions, backfill by REST, validate the
  repaired sequence, and then resume.
- Never invent a bar or silently ignore an unresolved gap.
- If database integrity, state reconciliation, clock freshness, or equity
  reconciliation fails, enter safe paused mode and emit a critical error.
- A critical error must not be converted into a buy, sell, or synthetic fill.

Historical Baseline A and Baseline B runs use separate physical SQLite files.
The selected paper candidate also uses its own physical database. They may read
the same immutable raw-bar files but never share mutable state.

All runs record run_id, stage_id, strategy version, source commit,
configuration hash, raw-data manifest hash, execution-contract version,
start/end time, and software environment summary.

## 13. Prospective Paper Gate

The selected final-holdout candidate must:

- Run for at least 60 calendar days.
- Finish with at least 30 consecutive days without a critical defect, duplicate
  signal, missed_execution, stop_monitoring_gap, missed state transition,
  unresolved data gap, or unexplained reconciliation difference.
- Pass scripted disconnect, duplicate-message, missing-bar, process restart,
  and transaction-interruption exercises.
- Reconcile every simulated fill, fee, position, and equity change.

If a defect changes signals, matching, risk, or reported returns, fix it and
restart the consecutive 30-day clock. Any missed_execution, including one
caused by sleep, shutdown, or network loss, also resets that clock even when no
software defect exists. Paper profitability is recorded but
cannot tune thresholds, add indicators, mix the two baselines, or promote the
validation runner-up.

Low natural trade frequency does not justify loosening the strategy. Historical
replay and deterministic fault tests exercise rare paths.

## 14. Testing And Acceptance

Required automated coverage:

- Baseline A return, volatility, ranking, tie-break, two-bar confirmation,
  holding, exit, replacement, and reset cases.
- Baseline B golden vectors for the exact 120-bar reset semantics, every frozen
  indicator formula, required-field NaN guard, cross flag and age, score group,
  entry filter, blocked combination, sell confirmation, ADX/DI/slope boundary,
  severe-break override, replacement, and exact minimum holding-period count.
- Common risk ATR fixed vectors and proof that both baselines receive the same
  finite SMA14 value from the same closed OHLC input.
- Closed-bar enforcement, Binance server-time freshness, stage date and bar
  membership, fresh stage state, warm-up exclusion, cross-boundary intent
  cancellation, terminal valuation, and deliberate future-data rejection.
- Backtest next-open matching, adverse slippage, commissions, gap-through-stop,
  intrabar-stop touch, entry-bar stop activation, stop-versus-normal-sell
  priority, stop-update ordering, and same-open sell-before-buy replacement.
- One-percent risk sizing, 30% cap, one-position limit, available-cash and buy-fee
  caps, decimal rounding, frozen public filters, unknown-filter rejection, and
  below-minimum order rejection.
- Paper quote tests for bookTicker-only bid/ask selection, planned overlapping
  24-hour socket rollover, failed rollover, update-ID deduplication, rejection
  when a newer-socket u arrives before an older-socket u, logical
  connection_epoch and stream_seq ordering, no cached quote, target_time minus
  one millisecond, exactly target_time, exactly target_time plus 60 seconds, just
  after the deadline, stale clocks, and historical-open rejection.
- Paper stop tests for live best-bid triggering, entry-quote exclusion, one-fill
  enforcement, normal-sell priority conflict, disconnect with confirmed and
  unknown historical crossings, recovery_stop, and no retroactive Kline fill.
- Execution-group tests for parent failure cancellation, parent sell settlement
  before child sizing, a fresh child quote, crash before parent commit, crash
  after parent commit but before child fill, and restart from each state.
- Duplicate messages, concurrent quote callbacks, out-of-order bars, missing
  bars, two-symbol bucket barriers, duplicate decisions, UNIQUE(fill.intent_id),
  conditional terminal transitions, account-version conflicts, SQLite rollback,
  and serialized recovery.
- Same-symbol same-bar re-entry rejection and legitimate other-symbol
  exit-before-buy ordering.
- Selection and reporting tests for zero, one, and two stage passers;
  fixed-vector Sharpe, Sortino, Calmar, turnover, profit factor, profit/loss
  ratio, friction, and terminal valuation; zero denominators; the exact 10%
  boundary;
  Baseline A's final tie-break; candidate fingerprinting; one-use holdout;
  final failure without runner-up promotion; and 500-USDT selection isolation
  from 300/1000-USDT diagnostics.
- Benchmark tests for initial and monthly open execution, sell-before-buy symbol
  ordering, filters, costs, rounding, terminal valuation, and nonselection use.
- Storage tests proving Baseline A, Baseline B, and the selected paper run use
  separate physical SQLite databases.
- Architecture tests proving the baselines do not import one another and the
  new package does not import cross_signal_strategy at runtime.
- Repeatability: identical data, code, and configuration produce identical
  signals, fills, equity series, and report hashes.

Normal tests must not read G-drive ETF data. Golden expectations for Baseline B
are checked-in values derived once from the pinned provenance snapshot, so
ordinary tests do not import or execute the ETF strategy.

The first-version milestone is complete only when:

- All automated tests pass.
- Data and execution manifests reproduce the reports.
- Each approved stage is run in order.
- HTML/CSV output shows equity, returns, drawdowns, trades, exposure, costs,
  skipped actions, missed executions, and errors.
- No private API, real-order entrypoint, key configuration, or hidden live mode
  exists.

## 15. Delivery Sequence

1. Create the isolated package and domain contracts.
2. Implement immutable data ingestion and stage isolation.
3. Implement and golden-test Baseline A and Baseline B.
4. Implement the common risk and matching engines.
5. Implement SQLite state, reporting, and deterministic backtests.
6. Complete the training-stage comparison and freeze eligible versions.
7. Run validation once and nominate the single candidate.
8. Run the nominated candidate once on the final holdout.
9. Implement and operate the public-data paper runner.
10. After the 60/30-day paper gate, stop and request a separate real-trading
    design decision.

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
