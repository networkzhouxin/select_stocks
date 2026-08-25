"""Order-free CLI for the blocked RSI low-turn prospective observer."""

import argparse
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from numbers import Real
import sys
import tempfile
from typing import Mapping
from zoneinfo import ZoneInfo


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from cross_signal_strategy.research.rsi_low_turn_outcomes import (
    DOUBLED_FRICTION,
    EventOutcomeRecord,
    HORIZONS,
    MaturedLabel,
    NOMINAL_FRICTION,
    RoundTripResult,
    build_summary,
    calculate_round_trip,
    mature_event_labels,
)
from cross_signal_strategy.research.rsi_low_turn_shadow import detect_rsi_low_turn
from cross_signal_strategy.research.rsi_low_turn_source import (
    ApprovedFuturePriceSource,
    MIN_COLLECTION_START,
    SHANGHAI,
    SourceContractError,
    load_arrival_input,
    load_manifest,
    prior_source_session_from_snapshot,
)
from cross_signal_strategy.research.rsi_low_turn_store import (
    STATE_MARKER_FILE,
    ShadowStore,
    SourceSnapshotBatch,
    SourceRewriteError,
)


FROZEN_ETF_CODES = (
    "159915", "512100", "159928", "513100", "513500", "513880", "513050",
    "518880", "159985",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--data-root", type=Path, required=True)
    collect.add_argument("--approved-root", type=Path, required=True)
    collect.add_argument("--state-dir", type=Path, required=True)
    collect.add_argument("--as-of", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--state-dir", type=Path, required=True)
    summarize.add_argument("--generated-at", required=True)
    return parser


def collect(data_root: Path, approved_root: Path, state_dir: Path, as_of: str) -> None:
    observed_at = _parse_shanghai_datetime(as_of, "as_of")
    manifest = load_manifest(data_root, approved_root)
    state_path = Path(state_dir).resolve()
    if state_path == manifest.root or state_path.is_relative_to(manifest.root):
        raise SourceContractError("state dir must be outside the approved data root")

    preflight_store = ShadowStore(state_path, create=False)
    preflight_store.validate_collect_integrity()
    has_marker = (state_path / STATE_MARKER_FILE).exists()
    if has_marker:
        preflight_store.require_complete_initial_collection(_required_snapshot_paths(), FROZEN_ETF_CODES)
    source_history = preflight_store.load_source_snapshots()
    existing_events = preflight_store.replay_validated_events(
        FROZEN_ETF_CODES, require_exact=has_marker,
    )
    _event_outcome_records(
        existing_events, preflight_store.load_labels(), source_history,
    )
    before_snapshots = _read_source_snapshots(
        manifest.root, manifest.daily_subdir, manifest.minute_subdir,
    )
    snapshot_batch = preflight_store.prepare_source_snapshot_batch(before_snapshots, observed_at)
    source_contents = dict(before_snapshots)
    prior_sessions = {
        prior_source_session_from_snapshot(
            source_contents[f"daily/{code}.csv"], code, observed_at,
        )
        for code in FROZEN_ETF_CODES
    }
    if len(prior_sessions) != 1:
        raise SourceContractError("ETF files do not share one consistent prior source session")
    prior_session = next(iter(prior_sessions))
    inputs = tuple(
        load_arrival_input(
            manifest.root,
            manifest.root,
            code,
            observed_at,
            expected_prior_session=prior_session,
            source_contents=source_contents,
        )
        for code in FROZEN_ETF_CODES
    )
    labels = _matured_existing_labels(
        existing_events, manifest.root, observed_at, source_contents,
    )
    after_snapshots = _read_source_snapshots(
        manifest.root, manifest.daily_subdir, manifest.minute_subdir,
    )
    if after_snapshots != before_snapshots:
        raise SourceRewriteError("source changed during collection")
    _validate_input_snapshot_hashes(inputs, before_snapshots)
    decisions = tuple(detect_rsi_low_turn(item) for item in inputs)
    for decision in decisions:
        preflight_store.validate_evaluation(decision, observed_at)
    available_snapshots = (*source_history, *snapshot_batch.records)
    _validate_new_label_provenance(existing_events, labels, available_snapshots)
    for event_id, horizon, payload in labels:
        preflight_store.validate_label(event_id, horizon, payload)

    store = ShadowStore(state_path)
    _record_source_snapshots(store, snapshot_batch)
    evaluation_results = [
        store.record_evaluation(decision, observed_at) for decision in decisions
    ]
    label_results = [
        store.append_label(event_id, horizon, payload)
        for event_id, horizon, payload in labels
    ]
    store.write_state_marker()
    print(
        "收集完成："
        f"evaluations={sum(result.written for result in evaluation_results)} "
        f"matured_labels={sum(result.written for result in label_results)} "
        "orders_disabled=True"
    )


def summarize(state_dir: Path, generated_at: str) -> None:
    timestamp = _parse_aware_datetime(generated_at, "generated_at")
    state_path = Path(state_dir).resolve()
    if not state_path.is_dir():
        raise SourceRewriteError("state directory is required")
    store = ShadowStore(state_path, create=False)
    store.require_complete_initial_collection(_required_snapshot_paths(), FROZEN_ETF_CODES)
    events = store.replay_validated_events(FROZEN_ETF_CODES)
    summary = build_summary(
        _event_outcome_records(
            events, store.load_labels(), store.load_source_snapshots(),
        ),
        MIN_COLLECTION_START,
        timestamp,
    )
    _atomic_write_summary(state_path, summary)
    print(
        "汇总完成："
        f"matured_five_day_events={summary['counts']['matured_five_day_events']} "
        f"status={summary['status']} orders_disabled=True"
    )


def _read_source_snapshots(
    root: Path, daily_subdir: str, minute_subdir: str,
) -> tuple[tuple[str, bytes], ...]:
    paths = [root / "manifest.json"]
    paths.extend(root / daily_subdir / f"{code}.csv" for code in FROZEN_ETF_CODES)
    paths.extend(root / minute_subdir / f"{code}.csv" for code in FROZEN_ETF_CODES)
    snapshots = []
    for path in paths:
        try:
            snapshots.append((path.relative_to(root).as_posix(), path.read_bytes()))
        except OSError as exc:
            raise SourceContractError(f"source snapshot is unreadable: {path.name}") from exc
    return tuple(snapshots)


def _validate_input_snapshot_hashes(
    inputs: tuple[object, ...], snapshots: tuple[tuple[str, bytes], ...],
) -> None:
    hashes = {relative_path: hashlib.sha256(content).hexdigest() for relative_path, content in snapshots}
    for item in inputs:
        code = getattr(item, "code")
        expected = (
            hashes["manifest.json"], hashes[f"daily/{code}.csv"],
            hashes[f"minute_0935/{code}.csv"],
        )
        if getattr(item, "source_hashes") != expected:
            raise SourceRewriteError("input source hashes do not match collection snapshot")


def _record_source_snapshots(
    store: ShadowStore, batch: SourceSnapshotBatch,
) -> None:
    store.write_source_snapshot_batch(batch)


def _required_snapshot_paths() -> tuple[str, ...]:
    return (
        "manifest.json",
        *(f"daily/{code}.csv" for code in FROZEN_ETF_CODES),
        *(f"minute_0935/{code}.csv" for code in FROZEN_ETF_CODES),
    )


def _matured_existing_labels(
    events: tuple[Mapping[str, object], ...],
    root: Path,
    as_of: datetime,
    source_contents: Mapping[str, bytes],
) -> tuple[tuple[str, int, Mapping[str, object]], ...]:
    source = ApprovedFuturePriceSource(
        root, root, source_contents=source_contents, collected_at=as_of,
    )
    labels = []
    for event in events:
        for label in mature_event_labels(event, source, as_of):
            if label.status == "matured":
                labels.append((label.event_id, label.horizon, asdict(label)))
    return tuple(labels)


def _event_outcome_records(
    events: tuple[Mapping[str, object], ...],
    labels: tuple[Mapping[str, object], ...],
    source_snapshots: tuple[Mapping[str, object], ...] = (),
) -> tuple[EventOutcomeRecord, ...]:
    event_ids = set()
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise SourceRewriteError("stored event is invalid")
        if event_id in event_ids:
            raise SourceRewriteError(f"duplicate event {event_id}")
        event_ids.add(event_id)

    labels_by_event: dict[str, dict[int, MaturedLabel]] = {}
    validated_labels = []
    for record in labels:
        event_id = record.get("event_id")
        horizon = record.get("horizon")
        payload = record.get("payload")
        if not isinstance(event_id, str) or not isinstance(horizon, int) or not isinstance(payload, Mapping):
            raise SourceRewriteError("stored label is invalid")
        if event_id not in event_ids:
            raise SourceRewriteError(f"dangling label {event_id}")
        if horizon not in HORIZONS:
            raise SourceRewriteError(f"unsupported horizon {horizon}")
        event_labels = labels_by_event.setdefault(event_id, {})
        if horizon in event_labels:
            raise SourceRewriteError(f"duplicate label for event_id {event_id} horizon {horizon}")
        event_labels[horizon] = None
        validated_labels.append((event_id, horizon, payload))
    for event_id, horizon, payload in validated_labels:
        event = next(item for item in events if item.get("event_id") == event_id)
        labels_by_event[event_id][horizon] = _matured_label(
            event, horizon, payload, source_snapshots,
        )

    records = []
    for event in events:
        event_id = event.get("event_id")
        code = event.get("code")
        arrival_date = event.get("arrival_date")
        if not isinstance(event_id, str) or not isinstance(code, str) or not isinstance(arrival_date, str):
            raise SourceRewriteError("stored event is invalid")
        try:
            arrival = date.fromisoformat(arrival_date)
        except ValueError as exc:
            raise SourceRewriteError("stored event arrival date is invalid") from exc
        records.append(EventOutcomeRecord(event_id, code, arrival, labels_by_event.get(event_id, {})))
    return tuple(records)


def _matured_label(
    event: Mapping[str, object],
    horizon: int,
    payload: Mapping[str, object],
    source_snapshots: tuple[Mapping[str, object], ...] = (),
) -> MaturedLabel:
    event_id = event.get("event_id")
    if not isinstance(event_id, str):
        raise SourceRewriteError("stored event is invalid")
    base_fields = {
        "event_id", "horizon", "status", "exit_price", "nominal", "doubled",
        "mfe", "mae",
    }
    provenance_fields = {
        "target_timestamp", "available_at", "collected_at", "source_relative_path",
        "source_sha256", "source_byte_length",
    }
    if not base_fields.issubset(payload):
        raise SourceRewriteError("stored label fields are invalid")
    if payload.get("event_id") != event_id or payload.get("horizon") != horizon:
        raise SourceRewriteError("stored label identity is invalid")
    if payload.get("status") != "matured":
        raise SourceRewriteError("stored label status is invalid")
    if not _strict_real(payload.get("exit_price")):
        raise SourceRewriteError("stored label numeric type is invalid")
    exit_price = _positive_finite_number(payload.get("exit_price"), "stored label exit price")
    nominal = _round_trip(payload.get("nominal"))
    doubled = _round_trip(payload.get("doubled"))
    if nominal is None or doubled is None:
        raise SourceRewriteError("stored label round trip is required")
    code = event.get("code")
    entry_open = event.get("entry_open")
    if not isinstance(code, str):
        raise SourceRewriteError("stored event is invalid")
    entry = _positive_finite_number(entry_open, "stored event entry price")
    expected_nominal = calculate_round_trip(code, entry, exit_price, NOMINAL_FRICTION)
    expected_doubled = calculate_round_trip(code, entry, exit_price, DOUBLED_FRICTION)
    if nominal != expected_nominal or doubled != expected_doubled:
        raise SourceRewriteError("stored label round trip does not match recomputation")
    if payload.get("mfe") is not None or payload.get("mae") is not None:
        raise SourceRewriteError("stored label MFE/MAE must remain unproved")
    if set(payload) != base_fields | provenance_fields:
        if set(payload) == base_fields:
            raise SourceRewriteError("stored label provenance is required")
        raise SourceRewriteError("stored label fields are invalid")
    provenance = _validated_label_provenance(event, payload, source_snapshots)
    return MaturedLabel(
        event_id=event_id,
        horizon=horizon,
        status="matured",
        exit_price=exit_price,
        nominal=nominal,
        doubled=doubled,
        mfe=None,
        mae=None,
        target_timestamp=provenance[0],
        available_at=provenance[1],
        collected_at=provenance[2],
        source_relative_path=provenance[3],
        source_sha256=provenance[4],
        source_byte_length=provenance[5],
    )


def _validate_new_label_provenance(
    events: tuple[Mapping[str, object], ...],
    labels: tuple[tuple[str, int, Mapping[str, object]], ...],
    source_snapshots: tuple[Mapping[str, object], ...],
) -> None:
    events_by_id = {event["event_id"]: event for event in events}
    for event_id, horizon, payload in labels:
        event = events_by_id.get(event_id)
        if event is None:
            raise SourceRewriteError(f"dangling label {event_id}")
        _matured_label(event, horizon, payload, source_snapshots)


def _validated_label_provenance(
    event: Mapping[str, object],
    payload: Mapping[str, object],
    source_snapshots: tuple[Mapping[str, object], ...],
) -> tuple[datetime, datetime, datetime, str, str, int]:
    try:
        target = _stored_aware_datetime(payload.get("target_timestamp"))
        available = _stored_aware_datetime(payload.get("available_at"))
        collected = _stored_aware_datetime(payload.get("collected_at"))
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("stored label provenance timestamp is invalid") from exc
    if (
        target.astimezone(SHANGHAI).timetz().replace(tzinfo=None).isoformat() != "09:35:00"
        or available > target
        or available > collected
    ):
        raise SourceRewriteError("stored label provenance is not point-in-time valid")
    arrival_value = event.get("arrival_date")
    try:
        arrival = date.fromisoformat(arrival_value)
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("stored event arrival date is invalid") from exc
    if target.date() <= arrival:
        raise SourceRewriteError("stored label provenance is not a future session")

    code = event.get("code")
    expected_path = f"minute_0935/{code}.csv"
    relative_path = payload.get("source_relative_path")
    digest = payload.get("source_sha256")
    byte_length = payload.get("source_byte_length")
    if (
        relative_path != expected_path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
        or not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 0
    ):
        raise SourceRewriteError("stored label provenance is invalid")
    matched = False
    for snapshot in source_snapshots:
        if (
            snapshot.get("relative_path") == relative_path
            and snapshot.get("sha256") == digest
            and snapshot.get("byte_length") == byte_length
        ):
            try:
                observed = _stored_aware_datetime(snapshot.get("observed_at"))
            except (TypeError, ValueError) as exc:
                raise SourceRewriteError("stored label provenance history is invalid") from exc
            if observed <= collected:
                matched = True
                break
    if not matched:
        raise SourceRewriteError("stored label provenance does not match source history")
    return target, available, collected, relative_path, digest.lower(), byte_length


def _stored_aware_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _round_trip(value: object) -> RoundTripResult | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SourceRewriteError("stored round trip is invalid")
    if set(value) != {
        "amount", "buy_exec_price", "sell_exec_price", "buy_commission",
        "sell_commission", "net_pnl", "net_return",
    }:
        raise SourceRewriteError("stored round trip is invalid")
    try:
        amount = value["amount"]
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise ValueError("amount")
        numeric_fields = (
            "buy_exec_price", "sell_exec_price", "buy_commission", "sell_commission",
            "net_pnl", "net_return",
        )
        if not all(_strict_real(value[field]) for field in numeric_fields):
            raise SourceRewriteError("stored label numeric type is invalid")
        return RoundTripResult(
            amount,
            _finite_number(value["buy_exec_price"]), _finite_number(value["sell_exec_price"]),
            _finite_number(value["buy_commission"]), _finite_number(value["sell_commission"]),
            _finite_number(value["net_pnl"]), _finite_number(value["net_return"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceRewriteError("stored round trip is invalid") from exc


def _strict_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise SourceRewriteError(f"stored label {key} is invalid")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return _finite_number(value)
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("stored label number is invalid") from exc


def _positive_finite_number(value: object, message: str) -> float:
    try:
        number = _finite_number(value)
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError(f"{message} is invalid") from exc
    if number <= 0.0:
        raise SourceRewriteError(f"{message} is invalid")
    return number


def _finite_number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite")
    return number


def _atomic_write_summary(state_dir: Path, summary: Mapping[str, object]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=state_dir,
            prefix=".summary-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(state_dir / "summary.json")
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _parse_shanghai_datetime(value: str, label: str) -> datetime:
    parsed = _parse_aware_datetime(value, label)
    return parsed.astimezone(SHANGHAI)


def _parse_aware_datetime(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SourceContractError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceContractError(f"{label} must be timezone-aware")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            collect(args.data_root, args.approved_root, args.state_dir, args.as_of)
        else:
            summarize(args.state_dir, args.generated_at)
    except (SourceContractError, SourceRewriteError) as exc:
        print(f"错误：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
