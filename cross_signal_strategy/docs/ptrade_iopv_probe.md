# PTrade IOPV Capability Probe

Date: 2026-07-13

## Purpose

This is a capability-only platform check for the four QDII ETFs in the frozen
PTrade cross-signal pool. The probe places no orders. It checks whether the
Guojin PTrade trading module actually supplies a positive, current IOPV value
through `get_snapshot()` and the related publication metadata through
`get_etf_info()`.

The official local PTrade documentation states that:

- `get_snapshot()` is available only in the trading module, returns real-time
  snapshots, and includes `iopv` as the ETF indicative net asset value.
- `get_etf_info()` is available only in the PTrade client stock-trading module
  and returns `publish`, `nav_pre`, and `nav_percu`. Some counter connections
  are explicitly unsupported.

Documentation proves the API contract, not that Guojin's current connection
publishes valid values for every QDII ETF. The isolated probe tests that final
platform fact.

## Safety Boundary

- Run `archive/probes/smart_trade_ptrade_cross_signal_iopv_probe.py` as a separate temporary
  simulation/live strategy. Do not merge it into the official strategy.
- Do not run it alongside the live cross-signal strategy because both alter the
  strategy universe and consume scheduled callbacks.
- The probe uses three callbacks, below PTrade's combined five-callback limit.
- It must not define or tune a premium threshold.
- The probe must not be used as validation-period performance evidence. Its
  current-market values must not change any 2019-2021-selected rule.
- PTrade daily backtest is not suitable for this check because its callbacks do
  not reproduce the requested intraday execution time.

## Run Procedure

1. Create a temporary PTrade strategy containing only the probe file.
2. Run it for one normal trading session in simulation or live mode.
3. Stop it after the 09:36 callback.
4. Retain only log lines beginning with `[ptrade-iopv-`.

The probe checks `09:34`, `09:35`, and `09:36` so that a value can be compared
with its `hsTimeStamp`. It does not calculate a premium or classify any value
as acceptable for trading.

## Evidence Gate

Point-in-time IOPV is operationally available only if all of the following are
observed for a target ETF:

1. `get_snapshot()` returns a record rather than an empty mapping.
2. `iopv_positive=True` when `publish=1`.
3. `hsTimeStamp` is present and `age_seconds` is consistent with a current
   quote at each callback.
4. The result is repeatable rather than a single transient response.

If IOPV is zero, missing, stale, or unavailable on the Guojin connection, the
strategy must continue without an IOPV input. Even if the gate passes, this
only establishes live data feasibility. The already rejected training premium
filter remains rejected, and no trading-rule experiment is reopened.
