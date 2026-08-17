# Cross-Signal Extreme-Lag Attribution Design

Status: pre-registered 2026-08-17, user-approved for Step 0 only. This design
authorizes a read-only 2019-2021 attribution report. It does not authorize a
strategy candidate, parameter search, validation-period read, or mainline
change.

## Objective

Determine whether the official `cross-v0.3.3` path contains a repeated,
cross-year class of extreme late entries or late exits that is material enough
to justify one future structural candidate. Long-term training performance
remains the primary objective. The report must not optimize for bottom buying,
top selling, or the comfort of any single live trade.

## Scope And Evidence Boundary

- Strategy scope: `cross_signal_strategy` only. Production multi-factor files
  are out of scope.
- Signal and replay period: 2019-01-01 through 2021-12-31 only.
- Warm-up: the approved read-only 2018 warm-up root may supply indicator
  lookback, but its rows cannot enter returns or selection evidence.
- Market-data root: only the approved immutable cross-signal training root may
  be read. No files may be written below a source-data root.
- Live cases such as the 2026 `513100` entry and `513050` exit may explain the
  question but cannot select a metric, threshold, parameter, ETF exception, or
  candidate rule.
- Reserved validation, pressure, recent-market, and full-period results are
  prohibited during Step 0.
- JoinQuant remains the authority for strategy performance. The local replay
  is used only for path-aligned attribution.

## Non-Goals

- Do not change `cross_window`, score weights, buy/sell thresholds, indicator
  periods, ETF pool, sizing, minimum hold, ATR rules, or execution time.
- Do not test age decay, faster MACD/KDJ exits, profit-giveback exits,
  profit-gated direct sells, profit-tier stops, or other closed research
  families.
- Do not claim that entry or exit timing can identify historical bottoms or
  tops in real time.
- Do not create a candidate automatically from the report. Any candidate needs
  a separate pre-registered design and explicit user approval.

## Unit Of Analysis

Use each official filled buy episode and its corresponding official filled
exit as the primary unit. Bind episodes to actual fills and position state, not
candidate signals. Preserve official code, date, side, quantity, and fill-path
identity before interpreting timing metrics.

If local filled-order dates or sides are not aligned with the official
JoinQuant path for 2019-2021, stop the attribution. Investigate the replay or
data path; do not change strategy rules to force alignment.

## Entry-Lag Measurements

For every official filled buy, record:

1. Every contributing bullish cross and its age (`0`, `1`, or `2` completed
   trading sessions) at the frozen T-1 signal date.
2. Reversal-score contribution by age, plus the share of reversal score coming
   from age-two crosses. This is descriptive; no decay formula is calculated.
3. Trading-session distance from the earliest contributing bullish cross to
   the filled buy.
4. Price extension from the close on the earliest contributing cross session
   to the actual fill, normalized by the entry ATR frozen from T-1 data.
5. T-day execution gap from T-1 close to the fill, normalized by the same ATR.
6. Post-entry 5-session MAE and MFE, used only after the episode exists for
   evaluation. These forward excursions must never feed a signal or order.

The report must present continuous distributions and annual summaries. It
must not invent an "extreme" cutoff after inspecting the data.

## Exit-Lag Measurements

For every official closed episode, record:

1. The first eligible session after minimum hold where `sell_score >= 30`.
2. Whether price-structure confirmation was absent, present, or blocked by an
   official protection rule on that first high-score session.
3. Trading-session delay from the first eligible high-score session to the
   actual filled exit.
4. Profit at the first eligible high-score session, official peak closing-price
   profit, and official exit profit.
5. Giveback from peak profit to official exit profit and the incremental
   giveback after the first eligible high-score session.
6. Post-exit 3-session and 5-session returns, used only as retrospective
   evaluation and never as an order input.

ATR exits and ordinary signal exits must be reported separately. Episodes with
unverified fill or state evidence must be retained as missing/unusable rather
than silently imputed.

## Annual And Concentration Reporting

Report every measurement for:

- the full 2019-2021 training window;
- 2019, 2020, and 2021 separately;
- each ETF code;
- ordinary signal exits and ATR exits separately where applicable.

Include counts, missing counts, median, interquartile range, and clearly named
tail observations. Tail rows are examples, not thresholds. A pattern is not
eligible for later mechanism design if its direction reverses between training
years or if its apparent effect is dominated by one ETF or a few trades.

## Step 0 Stop/Continue Decision

Step 0 ends with one of two outcomes:

- **Stop:** timing discomfort exists but is not directionally consistent across
  2019, 2020, and 2021; the effect is concentrated; order-path alignment is not
  proved; or the data cannot distinguish the proposed mechanism. No candidate
  is opened.
- **Eligible for a separate design:** one continuous timing measurement has a
  consistent adverse relationship with forward trade quality across all three
  training years, is not dominated by one ETF or a few episodes, and identifies
  a market mechanism not already closed by the experiment ledger. This outcome
  authorizes only a new design discussion, not implementation.

No ranking of multiple candidate rules is permitted in Step 0.

## Planned Artifacts After Separate Implementation Approval

- `cross_signal_strategy/research/extreme_lag_attribution.py`: read-only metric
  computation and annual/concentration summaries.
- `tests/test_cross_signal_extreme_lag_attribution.py`: tests written before
  implementation for fill binding, cross-age accounting, trading-session
  delays, ATR normalization, missing-state retention, date/root guards, and
  forward-evaluation isolation.
- `cross_signal_strategy/reports/extreme_lag_attribution_2019_2021.md`: generated
  human-readable report.
- Optional machine-readable CSV/JSON output under
  `cross_signal_strategy/reports/`, never under a market-data root.

## Test-First And Safety Requirements

Before implementing each metric group, add a failing focused test. Tests must
prove:

- all signal inputs stop at T-1;
- T-day prices are execution-only;
- forward MAE/MFE and post-exit returns are evaluation-only outputs;
- only approved roots and 2019-2021 performance dates are accepted;
- 2018 rows can serve only as warm-up;
- actual fills/positions bind episodes;
- missing or contradictory state remains explicit;
- no official JoinQuant/PTrade strategy or multi-factor file is imported as a
  writable target or modified.

## Review Gate

Implementation may begin only after the user reviews this written specification
and explicitly approves proceeding. Any material change to metrics, scope,
stop conditions, or evidence boundaries requires renewed design approval.
