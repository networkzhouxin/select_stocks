# Cross-Signal Age-2 Half-Decay Candidate Design

Status: frozen design awaiting written-spec review. User approved the fixed
direction in conversation on 2026-08-17. No implementation or backtest result
may change the rule below.

## Objective And Hypothesis

Test one soft alternative to the already-rejected all-or-nothing removal of
older crosses. The official `cross_window=3` lets a bullish cross contribute
its full reversal points at ages 0, 1, and 2. The fixed hypothesis is that the
oldest still-valid bullish cross may retain useful reversal information but
deserves less influence when the remaining buy conditions arrive late.

The candidate therefore changes only the buy-side reversal contribution of an
active age-2 bullish cross to 50% of its official point value. Age-0 and age-1
crosses remain at 100%. No nearby decay coefficient will be tried.

## Alternatives Considered

1. **Age-2 half-decay, buy side only (selected).** This is the smallest soft
   intervention aimed at the reported late-entry mechanism. It preserves some
   information from age-2 crosses after `cross_window=2` showed that deleting
   them entirely is harmful.
2. **Linear decay across every age (`1`, `2/3`, `1/3`).** Rejected because it
   also weakens age-1 evidence and introduces two effective changes instead of
   isolating the oldest-cross question.
3. **Apply the same decay to sell crosses or overwrite the formal strategy.**
   Rejected because weakening old bearish crosses can delay exits further, and
   overwriting `cross-v0.3.3` before a training gate would destroy the clean
   baseline comparison.

## Frozen Rule

For buy-side reversal scoring only:

| Bullish cross | Official points | Candidate points at age 0/1 | Candidate points at age 2 |
|---|---:|---:|---:|
| RSI6 above RSI12 | 12 | 12 | 6 |
| RSI6 above RSI24 | 12 | 12 | 6 |
| MACD DIF above DEA | 10 | 10 | 5 |
| KDJ K above D | 6 | 6 | 3 |
| KDJ J above D | 5 | 5 | 2.5 |

RSI group-direction behavior remains official: RSI bullish points contribute
only when the active RSI group direction is up. A recent opposite cross that
makes the RSI group mixed continues to suppress the RSI bullish contribution.

The candidate recomputes:

`candidate_buy_score = max(0, candidate_reversal_score + official_location_score + official_trend_score + official_volume_score)`

Everything else remains unchanged, including:

- `cross_window=3` and all indicator periods;
- buy threshold, overheat rule, position-location filter, entry-combo block,
  ranking, and tie-breaks;
- all sell scores, price confirmations, ADX protection, minimum hold, and ATR
  rules;
- ETF pool, capital, sizing, fees, slippage, 09:35 execution, and T-1 signal
  timing;
- portfolio ATR-stress sizing adopted in `cross-v0.3.3`.

## Evidence Boundary

- Development and comparison: only 2019-01-01 through 2021-12-31 from the
  immutable approved training root.
- 2018 warm-up may calculate rolling indicators only; it cannot contribute to
  performance statistics or rule selection.
- Reserved validation, stress, recent-market, full-period, PTrade simulation,
  and live cases are prohibited during this experiment.
- JoinQuant remains the performance authority. Local replay is a direction and
  path-change gate only.
- The 2026 `513100` and `513050` cases motivated the question but do not select
  the rule, coefficient, ETF exception, or result interpretation.

## Architecture

Stage 1 adds an isolated research adapter that wraps the frozen official local
signal adapter. It consumes the already-T-1-safe score snapshot, adjusts only
the five bullish-cross point contributions listed above, returns defensive
copies, and leaves the formal JoinQuant/PTrade files untouched.

The A/B runner reuses identical training dates, market data, broker, execution,
fees, ETF pool, and planner for baseline and candidate. It reports aggregate
metrics, annual returns, buy/sell counts, and filled-order-day differences by
year.

Stage 2 occurs only if the local gate passes. It creates a separate JoinQuant
candidate file named for the experiment; it does not overwrite the official
`smart_trade_joinquant_cross_signal_etf.py`. The user then runs only the
2019-2021 JoinQuant training backtest and supplies the result/logs available at
that time.

## Pre-Registered Local Gate

The candidate passes only if all conditions hold against the local official
`cross-v0.3.3` baseline:

1. at least one filled-order day changes in each of 2019, 2020, and 2021;
2. total return and annualized return are strictly higher;
3. maximum drawdown does not worsen;
4. Sharpe, Sortino, win rate, and profit/loss ratio do not worsen;
5. annual return does not worsen in any of 2019, 2020, or 2021.

If any condition fails, reject the candidate locally, do not create a
JoinQuant candidate, do not inspect validation periods, and do not try another
decay coefficient, selected indicator, ETF/year exception, or threshold
compensation.

## Tests And Safety Checks

Tests must be written and observed failing before implementation. They must
prove:

- age-0 and age-1 bullish contributions are unchanged;
- only age-2 bullish contributions are multiplied by exactly `0.5`;
- RSI mixed-direction semantics remain official;
- location, trend, volume, sell scores, parameters, and input snapshots remain
  unchanged;
- the wrapped score remains bound to the same T-1 `signal_date` and
  `max_data_date`;
- baseline and candidate use identical training dates and approved roots;
- the strict gate checks all aggregate, annual, and path-change conditions;
- no formal strategy or multi-factor file is modified by Stage 1.

## Reporting And Stop Condition

Record the hypothesis, changed files, baseline and candidate training metrics,
annual results, order-path changes, future-function/data-boundary audit,
interpretation, and next step. A failed experiment must be appended to the
failed-experiment and research-budget records rather than silently discarded.

Passing the local gate authorizes only preparation of the separate JoinQuant
training candidate. It does not authorize validation, formal adoption, PTrade
sync, or live use.
