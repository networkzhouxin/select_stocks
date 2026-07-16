# PTrade Resilient State Checkpoint Design

## Scope

Strengthen restart and redeployment recovery for the formal cross-signal PTrade
adapter. This change is limited to PTrade persistence, broker reconciliation,
recovery gating, and diagnostics. It must not change JoinQuant code, ETF pool,
indicators, thresholds, signal dates, position sizing, or buy/sell rules.

## Problem

The adapter currently has three recovery layers: PTrade's persisted non-private
`g` fields, one explicit pickle checkpoint, and deterministic reconstruction
from broker positions/trades. The explicit checkpoint is now written directly
because PTrade forbids `os`, so a process failure during the write can damage
the only file. The checkpoint also couples compatibility to the strategy
version, and an unverified existing holding does not prevent unrelated new
buys.

## Design

### Dual-slot checkpoint

Use two account-and-trade-isolated files with `.a` and `.b` slots. Each file is
an envelope containing a state-schema version, monotonically increasing
generation, producer strategy version, SHA256 checksum, and protocol-4 pickle
bytes for the state payload. Each save overwrites only the older slot. Restore
validates both slots and chooses the highest valid generation; a damaged newest
slot therefore falls back to the previous valid slot without filesystem rename
operations.

The old single-file checkpoint remains a read-only migration source. If no
valid dual-slot file exists, restore the validated legacy payload and write the
next checkpoint in dual-slot format. The legacy file is not deleted.

### Compatibility

Introduce `LIVE_STATE_SCHEMA_VERSION` independently of `STRATEGY_VERSION`.
Compatible strategy releases may restore the same state schema even when the
producer strategy version differs. An unknown schema, checksum mismatch,
malformed payload, or missing required field is rejected. No partial state is
applied.

### Recovery safety gate

Existing verified holdings continue to receive normal ATR and signal exits.
An unverified holding remains untouched automatically because its entry facts
cannot be proved. While any current holding is unverified, all new buy
submissions are blocked so the strategy cannot expand exposure in an
incompletely recovered portfolio. Broker order-state uncertainty continues to
block all trading as before.

### Recovery diagnostics

At startup, log the checkpoint source and generation. After broker
reconciliation, emit one summary per current holding with broker quantity,
broker cost, buy date, entry ATR, highest close, verification status, and
recovery source. Sources are `checkpoint-a`, `checkpoint-b`, `legacy`,
`ptrade-g`, `get-trades`, `get-deliver`, or `unverified`. Source metadata is
private runtime state and is not part of the business checkpoint.

## Failure Behavior

- One corrupt dual-slot file: restore the other valid slot.
- Both slots invalid but legacy valid: restore legacy and migrate on save.
- Every checkpoint invalid or absent: retain PTrade-restored `g` facts, then
  attempt deterministic broker reconstruction.
- Broker evidence cannot prove a held position: mark it unverified, block its
  automatic exits, and block every new buy.
- Open-order query fails or is malformed: preserve the existing global trading
  block.

## Testing

Tests must be written and observed failing before implementation. Coverage must
include alternating generations, newest-slot corruption fallback, checksum
rejection, protocol 4, cross-strategy-version compatibility, unknown-schema
rejection, legacy migration, no partial restore, unverified-position buy
blocking, verified-position exit continuity, and startup recovery summaries.
The complete PTrade test module and full repository test suite must pass.

## Deployment

Deploy under the same PTrade account and trade name. In simulation, verify a
normal restart and a restart after deliberately corrupting only the newest slot.
Any `UNVERIFIED` holding or state/checksum error requires operator review before
live capital is enabled.
