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
    existing_observations = preflight_store.load_future_observations()
    existing_labels = preflight_store.load_labels()
    _event_outcome_records(
        existing_events, existing_observations, existing_labels, source_history,
        allow_missing_labels=True,
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
    observations = _matured_existing_observations(
        existing_events, existing_observations, manifest.root, observed_at,
        source_contents, available_snapshots,
    )
    all_observations = (*existing_observations, *(
        {"event_id": event_id, "horizon": horizon, "payload": payload}
        for event_id, horizon, payload in observations
    ))
    labels = _labels_for_observations(
        existing_events, all_observations, existing_labels, available_snapshots,
    )
    _event_outcome_records(
        existing_events,
        tuple(all_observations),
        (*existing_labels, *(
            {"event_id": event_id, "horizon": horizon, "payload": payload}
            for event_id, horizon, payload in labels
        )),
        available_snapshots,
    )
    for event_id, horizon, payload in observations:
        preflight_store.validate_future_observation(event_id, horizon, payload)
    for event_id, horizon, payload in labels:
        preflight_store.validate_label(event_id, horizon, payload)

    store = ShadowStore(state_path)
    _record_source_snapshots(store, snapshot_batch)
    evaluation_results = [
        store.record_evaluation(decision, observed_at) for decision in decisions
    ]
    observation_results = [
        store.append_future_observation(event_id, horizon, payload)
        for event_id, horizon, payload in observations
    ]
    label_results = [
        store.append_label(event_id, horizon, payload)
        for event_id, horizon, payload in labels
    ]
    store.write_state_marker()
    print(
        "收集完成："
        f"evaluations={sum(result.written for result in evaluation_results)} "
        f"future_observations={sum(result.written for result in observation_results)} "
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
            events, store.load_future_observations(), store.load_labels(),
            store.load_source_snapshots(),
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


def _matured_existing_observations(
    events: tuple[Mapping[str, object], ...],
    existing: tuple[Mapping[str, object], ...],
    root: Path,
    as_of: datetime,
    source_contents: Mapping[str, bytes],
    source_snapshots: tuple[Mapping[str, object], ...],
) -> tuple[tuple[str, int, Mapping[str, object]], ...]:
    source = ApprovedFuturePriceSource(
        root, root, source_contents=source_contents, collected_at=as_of,
    )
    existing_keys = {(item["event_id"], item["horizon"]) for item in existing}
    observations = []
    for event in events:
        for horizon in HORIZONS:
            key = (event["event_id"], horizon)
            if key in existing_keys:
                continue
            observation = source.observation_for(event, horizon, as_of, source_snapshots)
            if observation is not None:
                observations.append((event["event_id"], horizon, _json_ready(asdict(observation))))
    return tuple(observations)


def _labels_for_observations(
    events: tuple[Mapping[str, object], ...],
    observations: tuple[Mapping[str, object], ...],
    existing_labels: tuple[Mapping[str, object], ...],
    source_snapshots: tuple[Mapping[str, object], ...],
) -> tuple[tuple[str, int, Mapping[str, object]], ...]:
    existing_keys = {(item["event_id"], item["horizon"]) for item in existing_labels}
    events_by_id = {event["event_id"]: event for event in events}
    labels = []
    for record in observations:
        key = (record["event_id"], record["horizon"])
        if key in existing_keys:
            continue
        event = events_by_id.get(record["event_id"])
        if event is None:
            raise SourceRewriteError(f"dangling future observation {record['event_id']}")
        observation = _validated_future_observation(
            event, record["horizon"], record["payload"], source_snapshots,
        )
        labels.append((key[0], key[1], _label_payload_from_observation(event, observation)))
    return tuple(labels)


def _event_outcome_records(
    events: tuple[Mapping[str, object], ...],
    observations: tuple[Mapping[str, object], ...],
    labels: tuple[Mapping[str, object], ...],
    source_snapshots: tuple[Mapping[str, object], ...] = (),
    *,
    allow_missing_labels: bool = False,
) -> tuple[EventOutcomeRecord, ...]:
    event_ids = set()
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise SourceRewriteError("stored event is invalid")
        if event_id in event_ids:
            raise SourceRewriteError(f"duplicate event {event_id}")
        event_ids.add(event_id)

    events_by_id = {event["event_id"]: event for event in events}
    observations_by_key = {}
    for record in observations:
        event_id = record.get("event_id")
        horizon = record.get("horizon")
        payload = record.get("payload")
        if not isinstance(event_id, str) or not isinstance(horizon, int) or not isinstance(payload, Mapping):
            raise SourceRewriteError("stored future observation is invalid")
        if event_id not in event_ids:
            raise SourceRewriteError(f"dangling future observation {event_id}")
        if horizon not in HORIZONS:
            raise SourceRewriteError(f"unsupported horizon {horizon}")
        key = (event_id, horizon)
        if key in observations_by_key:
            raise SourceRewriteError(f"duplicate future observation for event_id {event_id} horizon {horizon}")
        observations_by_key[key] = _validated_future_observation(
            events_by_id[event_id], horizon, payload, source_snapshots,
        )

    label_records = {}
    for record in labels:
        event_id, horizon, payload = (
            record.get("event_id"), record.get("horizon"), record.get("payload"),
        )
        if not isinstance(event_id, str) or not isinstance(horizon, int) or not isinstance(payload, Mapping):
            raise SourceRewriteError("stored label is invalid")
        key = (event_id, horizon)
        if event_id not in event_ids:
            raise SourceRewriteError(f"dangling label {event_id}")
        if horizon not in HORIZONS:
            raise SourceRewriteError(f"unsupported horizon {horizon}")
        if key not in observations_by_key:
            raise SourceRewriteError("stored label has no future observation")
        if key in label_records:
            raise SourceRewriteError(f"duplicate label for event_id {event_id} horizon {horizon}")
        label_records[key] = payload
    missing = set(observations_by_key) - set(label_records)
    if missing and not allow_missing_labels:
        raise SourceRewriteError("future observation has no matured label")

    labels_by_event: dict[str, dict[int, MaturedLabel]] = {}
    for key, payload in label_records.items():
        event_id, horizon = key
        labels_by_event.setdefault(event_id, {})[horizon] = _matured_label(
            events_by_id[event_id], horizon, payload, observations_by_key[key],
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


def _validated_future_observation(
    event: Mapping[str, object],
    horizon: int,
    payload: Mapping[str, object],
    source_snapshots: tuple[Mapping[str, object], ...],
) -> Mapping[str, object]:
    expected = {
        "event_id", "code", "arrival_date", "horizon", "target_date",
        "target_timestamp", "future_sessions", "minute", "daily_snapshot",
        "minute_snapshot",
    }
    if set(payload) != expected:
        raise SourceRewriteError("future observation fields are invalid")
    if (
        payload.get("event_id") != event.get("event_id")
        or payload.get("code") != event.get("code")
        or payload.get("arrival_date") != event.get("arrival_date")
        or payload.get("horizon") != horizon
    ):
        raise SourceRewriteError("future observation event identity is invalid")
    try:
        arrival = date.fromisoformat(payload["arrival_date"])
        target_date = date.fromisoformat(payload["target_date"])
        target = _stored_aware_datetime(payload["target_timestamp"])
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("future observation timestamp is invalid") from exc
    sessions = payload.get("future_sessions")
    if not isinstance(sessions, list) or len(sessions) != horizon:
        raise SourceRewriteError("future observation session proof is invalid")
    session_dates = []
    collected_values = []
    for identity_name in ("daily_snapshot", "minute_snapshot"):
        identity = payload.get(identity_name)
        if not isinstance(identity, Mapping):
            raise SourceRewriteError("future observation snapshot identity is invalid")
        expected_path = (
            f"daily/{event['code']}.csv" if identity_name == "daily_snapshot"
            else f"minute_0935/{event['code']}.csv"
        )
        if identity.get("relative_path") != expected_path:
            raise SourceRewriteError("future observation snapshot identity is invalid")
        try:
            observed = _stored_aware_datetime(identity.get("observed_at"))
            collected = _stored_aware_datetime(identity.get("collected_at"))
        except (TypeError, ValueError) as exc:
            raise SourceRewriteError("future observation snapshot identity is invalid") from exc
        if observed > collected:
            raise SourceRewriteError("future observation snapshot identity is not point-in-time valid")
        collected_values.append(collected)
        if not any(
            snapshot.get("relative_path") == identity.get("relative_path")
            and snapshot.get("sha256") == identity.get("sha256")
            and snapshot.get("byte_length") == identity.get("byte_length")
            and _stored_aware_datetime(snapshot.get("observed_at")) == observed
            for snapshot in source_snapshots
        ):
            raise SourceRewriteError("future observation snapshot identity does not match source history")
    if collected_values[0] != collected_values[1]:
        raise SourceRewriteError("future observation snapshots were not collected together")
    collected = collected_values[0]
    for session in sessions:
        if not isinstance(session, Mapping) or set(session) != {"date", "available_at", "source"}:
            raise SourceRewriteError("future observation session proof is invalid")
        try:
            session_date = date.fromisoformat(session.get("date"))
            available = _stored_aware_datetime(session.get("available_at"))
        except (TypeError, ValueError) as exc:
            raise SourceRewriteError("future observation session proof is invalid") from exc
        if (
            session_date <= arrival
            or available > collected
            or not isinstance(session.get("source"), str)
            or not session["source"].strip()
        ):
            raise SourceRewriteError("future observation session proof is invalid")
        session_dates.append(session_date)
    if session_dates != sorted(set(session_dates)) or session_dates[-1] != target_date:
        raise SourceRewriteError("future observation session proof is invalid")
    expected_target = datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=35), SHANGHAI)
    if target != expected_target:
        raise SourceRewriteError("future observation target is invalid")
    minute = payload.get("minute")
    if not isinstance(minute, Mapping) or set(minute) != {
        "timestamp", "open", "volume", "num_trades", "available_at", "source",
    }:
        raise SourceRewriteError("future observation minute proof is invalid")
    try:
        minute_timestamp = _stored_aware_datetime(minute.get("timestamp"))
        minute_available = _stored_aware_datetime(minute.get("available_at"))
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("future observation minute proof is invalid") from exc
    if (
        minute_timestamp != target
        or minute_available > target
        or minute_available > collected
        or not all(_strict_real(minute.get(key)) and float(minute[key]) > 0 for key in ("open", "volume", "num_trades"))
        or not isinstance(minute.get("source"), str)
        or not minute["source"].strip()
    ):
        raise SourceRewriteError("future observation minute proof is invalid")
    return payload


def _matured_label(
    event: Mapping[str, object],
    horizon: int,
    payload: Mapping[str, object],
    observation: Mapping[str, object],
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
    minute = observation.get("minute")
    minute_snapshot = observation.get("minute_snapshot")
    if not isinstance(minute, Mapping) or not isinstance(minute_snapshot, Mapping):
        raise SourceRewriteError("future observation minute proof is invalid")
    expected_provenance = {
        "target_timestamp": observation.get("target_timestamp"),
        "available_at": minute.get("available_at"),
        "collected_at": minute_snapshot.get("collected_at"),
        "source_relative_path": minute_snapshot.get("relative_path"),
        "source_sha256": minute_snapshot.get("sha256"),
        "source_byte_length": minute_snapshot.get("byte_length"),
    }
    if (
        exit_price != float(minute.get("open"))
        or any(payload.get(key) != value for key, value in expected_provenance.items())
    ):
        raise SourceRewriteError("stored label does not match future observation")
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
    provenance = (
        _stored_aware_datetime(expected_provenance["target_timestamp"]),
        _stored_aware_datetime(expected_provenance["available_at"]),
        _stored_aware_datetime(expected_provenance["collected_at"]),
        expected_provenance["source_relative_path"],
        expected_provenance["source_sha256"],
        expected_provenance["source_byte_length"],
    )
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


def _label_payload_from_observation(
    event: Mapping[str, object], observation: Mapping[str, object],
) -> Mapping[str, object]:
    minute = observation["minute"]
    minute_snapshot = observation["minute_snapshot"]
    exit_price = float(minute["open"])
    code = event["code"]
    entry = float(event["entry_open"])
    return {
        "event_id": event["event_id"],
        "horizon": observation["horizon"],
        "status": "matured",
        "exit_price": exit_price,
        "nominal": asdict(calculate_round_trip(code, entry, exit_price, NOMINAL_FRICTION)),
        "doubled": asdict(calculate_round_trip(code, entry, exit_price, DOUBLED_FRICTION)),
        "mfe": None,
        "mae": None,
        "target_timestamp": observation["target_timestamp"],
        "available_at": minute["available_at"],
        "collected_at": minute_snapshot["collected_at"],
        "source_relative_path": minute_snapshot["relative_path"],
        "source_sha256": minute_snapshot["sha256"],
        "source_byte_length": minute_snapshot["byte_length"],
    }


def _stored_aware_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _json_ready(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


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
