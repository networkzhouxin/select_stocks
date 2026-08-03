# Binance Spot Strategy M1

M1 contains a synthetic, offline foundation only. It freezes numeric and
canonical JSON protocols, immutable domain contracts, semantic runtime
evidence, and strict candidate/run identity primitives.

The Task 6 identity golden vectors are synthetic protocol fixtures. They prove
that canonical candidate and run payloads hash reproducibly, but they are not a
formal Baseline A or Baseline B candidate: executable strategy modules do not
exist in M1.

Run the focused identity verification with the frozen interpreter:

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_fingerprints.py" -v
```

Deferred scope is explicit:

- no JoinQuant, PTrade, ETF, or other pre-existing strategy code was imported;
- no market data was read, including training, validation, or holdout data;
- no Binance network API, live feed, order placement, or account access exists;
- no strategy, execution, broker, persistence, audit, or state machine exists;
- no backtest, training result, validation result, holdout result, or paper
  qualification was produced; and
- no formal candidate fingerprint or real Baseline A/B fingerprint was
  generated.

M1 runtime attestation is deliberately limited. It freezes runtime versions,
the exact `python.exe` and `_decimal` artifacts, and each distribution's raw
`RECORD` bytes. It does not prove the actual content of every installed
dependency file or standard-library file.

Before any formal candidate is generated or any training begins, a complete
source-execution and dependency-content attestation protocol must be frozen
and pass.

This package is not a complete Binance strategy and is not ready for real
trading.
