# IOPV Data Quality Audit

Date: 2026-07-11

## Scope

- Dataset: read-only `G:\financial\history_data\cross_signal_train_2019_2021`
- Window: 2019-01-01 through 2021-12-31 only
- Universe: the frozen 12-ETF cross-signal pool
- Grain: one ETF, trade date, and one-minute timestamp
- Rows inspected: 2,029,422
- Strategy files changed: none

The audit is observation-only. It does not define a premium threshold, alter a
signal, or use a reserved validation period.

## Structural Checks

| Check | Result |
|---|---:|
| Duplicate `(code, date, time)` rows | 0 |
| Missing IOPV rows | 222,039 |
| Non-positive finite IOPV rows | 0 |
| Infinite IOPV rows | 0 |
| Valid positive IOPV rows | 1,807,383 (89.06%) |
| Trading days represented across code-years | 8,421 |
| Days with a 09:35 bar | 8,421 (100.00%) |
| Days with valid 09:35 IOPV | 7,515 (89.24%) |
| Executable 09:35 days with valid IOPV | 7,206 / 8,046 (89.56%) |

Most complete sessions contain 241 rows. The 09:30 IOPV is normally blank;
later minute rows carry IOPV when the source has coverage.

## Coverage By Year

| Year | 09:35 valid / represented | Rate | Executable 09:35 valid / executable | Rate |
|---|---:|---:|---:|---:|
| 2019 | 2,031 / 2,589 | 78.45% | 1,943 / 2,464 | 78.86% |
| 2020 | 2,577 / 2,916 | 88.37% | 2,465 / 2,784 | 88.54% |
| 2021 | 2,907 / 2,916 | 99.69% | 2,798 / 2,798 | 100.00% |

The missingness is not random and cannot be attributed only to fund listings or
individual suspensions:

- In 2019, the ten ETFs already listed for the full year share the same 50
  missing 09:35 dates.
- In 2020, all twelve ETFs share the same 27 missing 09:35 dates.
- Example: `510300` traded normally on 2019-07-23, including 1,516,600 units and
  62 trades at 09:35, but IOPV is blank for all 241 rows that day.
- Example: `513100` traded normally on 2020-03-17 and 2020-03-18, but IOPV is
  blank for every minute on both dates.

This is high-confidence cross-sectional source-level missingness. The exact
upstream cause is not documented in the isolated dataset and remains unknown.

## QDII 09:35 Evidence

The following figures use only 09:35 rows that pass the local executability
rule: the minute exists and `volume` and `num_trades` are not both zero.

| ETF | Year | Samples | Median | P01 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| 159920 | 2019 | 194 | -0.13% | -0.76% | 0.22% | 0.47% |
| 159920 | 2020 | 216 | -0.14% | -0.97% | 0.79% | 1.26% |
| 159920 | 2021 | 243 | -0.07% | -0.39% | 0.62% | 1.30% |
| 513050 | 2019 | 194 | -0.15% | -1.57% | 2.11% | 2.36% |
| 513050 | 2020 | 215 | 0.34% | -1.96% | 4.52% | 6.57% |
| 513050 | 2021 | 243 | 0.98% | -1.88% | 10.13% | 12.51% |
| 513100 | 2019 | 192 | -0.15% | -1.18% | 1.09% | 1.82% |
| 513100 | 2020 | 215 | 6.78% | -0.71% | 22.34% | 24.86% |
| 513100 | 2021 | 243 | -0.11% | -0.97% | 0.94% | 1.31% |
| 513500 | 2019 | 184 | -0.21% | -1.33% | 1.11% | 1.37% |
| 513500 | 2020 | 215 | 0.05% | -1.34% | 4.75% | 7.04% |
| 513500 | 2021 | 243 | 0.04% | -0.90% | 0.87% | 0.94% |
| 513880 | 2019 | 36 | -0.40% | -1.24% | 0.23% | 0.30% |
| 513880 | 2020 | 140 | -0.37% | -1.78% | 1.68% | 4.06% |
| 513880 | 2021 | 132 | -0.24% | -0.87% | 1.40% | 3.93% |

Two local `513100` closing observations match contemporaneous fund-manager
announcements to rounding:

- 2020-02-07: market close 3.709, IOPV 3.431, local premium 8.10%. The official
  announcement reported 3.709 versus 3.4311 and 8.10%.
- 2020-09-21: market close 4.667, IOPV 3.810, local premium 22.49%. The official
  announcement reported a 22.5% premium.

Sources: [2020-02-08 fund-manager announcement](https://www.sse.com.cn/disclosure/fund/announcement/c/2020-02-10/513100_20200208_1.pdf),
[2020-09-22 fund-manager announcement](https://www.sse.com.cn/disclosure/fund/announcement/c/2020-09-22/513100_20200922_1.pdf).

These matches support price/IOPV unit consistency and the reality of the large
2020 premium observations. They do not prove every historical row is correct.

## Suspension And Stale-Price Behavior

There are 216,447 zero-volume and zero-trade minute rows. Of those, 185,434 have
valid IOPV and 20,576 show an IOPV change from the preceding minute. Therefore,
IOPV can continue moving while the secondary-market price is stale and no local
fill is justified. Premium values from those rows are descriptive only and are
excluded from the executable 09:35 table.

## Point-In-Time Limitation

The isolated data README documents minute fields but does not document whether
a row labelled `09:35` represents information available at exactly 09:35:00 or
the completed 09:35 minute. Using that row as an exact 09:35 decision input
could therefore introduce same-minute look-ahead. The current local backtester
may keep using it as an execution-price proxy, but a future premium filter must
not consume it as a signal until timestamp semantics are proved.

Daily NAV is also unsuitable for a 09:35 same-day decision because it is not a
point-in-time morning value. QDII IOPV is only an indicative reference and can
differ from actual fund NAV because of overseas market hours, exchange rates,
quota constraints, and calculation methodology.

## Decision

Classification: **usable for descriptive training diagnostics, not yet safe as
a trading rule input**.

The data is strong enough to establish that historical A-share ETF transaction
prices include genuine premium/discount behavior and to identify high-premium
episodes. It is not strong enough to tune or adopt a premium threshold because:

1. 2019-2020 IOPV missingness is material and regime-dependent.
2. Missing rows cannot be reconstructed from later daily NAV without leakage.
3. Exact 09:35 point-in-time availability is unproved.
4. The standard JoinQuant strategy interface still needs a point-in-time IOPV
   availability probe before local research can be reproduced on the authority
   platform.

The next legitimate step is a platform/data-source capability probe, not a
premium-factor backtest. No strategy experiment or research threshold is opened
by this audit.
