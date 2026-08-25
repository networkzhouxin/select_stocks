"""Append-only local state for the order-free RSI low-turn shadow observer."""

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

from cross_signal_strategy.research.rsi_low_turn_shadow import SignalDecision, VERSION


EVALUATIONS_FILE = "evaluations.jsonl"
EVENTS_FILE = "events.jsonl"
HASHES_FILE = "source_hashes.jsonl"
LABELS_FILE = "labels.jsonl"
STATE_MARKER_FILE = "observer_state.json"
STATE_MARKER = {
    "observer": "rsi_low_turn_prospective_shadow",
    "schema_version": 1,
}


@dataclass(frozen=True)
class RecordResult:
    written: bool
    event_created: bool
    reason: str


@dataclass(frozen=True)
class SourceSnapshotBatch:
    records: tuple[Mapping[str, object], ...]


class SourceRewriteError(RuntimeError):
    """Raised when an append-only key is supplied with different content."""


class ShadowStore:
    def __init__(self, state_dir: Path, create: bool = True):
        self.state_dir = Path(state_dir).resolve()
        if create:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        elif self.state_dir.exists() and not self.state_dir.is_dir():
            raise SourceRewriteError("state path is not a directory")

    def record_source_snapshot(
        self, relative_path: str, content: bytes, observed_at: datetime,
    ) -> bool:
        batch = self.prepare_source_snapshot_batch(((relative_path, content),), observed_at)
        return bool(self.write_source_snapshot_batch(batch))

    def load_source_snapshots(self) -> tuple[Mapping[str, object], ...]:
        records = tuple(self._read_records(HASHES_FILE))
        keys = set()
        for record in records:
            path = _validated_source_snapshot(record)
            key = (path, record["sha256"], record["byte_length"])
            if key in keys:
                raise SourceRewriteError(f"duplicate source snapshot for {path}")
            keys.add(key)
        return records

    def prepare_source_snapshot_batch(
        self, snapshots: tuple[tuple[str, bytes], ...], observed_at: datetime,
    ) -> SourceSnapshotBatch:
        history = self.load_source_snapshots()
        latest = _latest_snapshot_by_path(history)
        records = []
        for relative_path, content in snapshots:
            record = _source_snapshot_record(relative_path, content, observed_at)
            prior = latest.get(record["relative_path"])
            if prior is not None:
                prior_length = prior["byte_length"]
                if len(content) < prior_length or hashlib.sha256(content[:prior_length]).hexdigest() != prior["sha256"]:
                    raise SourceRewriteError(f"source hash changed for {record['relative_path']}")
            if prior is None or (
                record["sha256"], record["byte_length"]
            ) != (prior["sha256"], prior["byte_length"]):
                records.append(record)
                latest[record["relative_path"]] = record
        return SourceSnapshotBatch(tuple(records))

    def write_source_snapshot_batch(self, batch: SourceSnapshotBatch) -> tuple[bool, ...]:
        if not isinstance(batch, SourceSnapshotBatch):
            raise TypeError("batch must be a SourceSnapshotBatch")
        history = self.load_source_snapshots()
        existing = {
            (item["relative_path"], item["sha256"], item["byte_length"])
            for item in history
        }
        results = []
        for record in batch.records:
            _validated_source_snapshot(record)
            key = (record["relative_path"], record["sha256"], record["byte_length"])
            if key in existing:
                results.append(False)
                continue
            self._append_record(HASHES_FILE, record)
            existing.add(key)
            results.append(True)
        return tuple(results)

    def validate_collect_integrity(self) -> None:
        self.load_source_snapshots()
        self.load_evaluations()
        self.load_events()
        self.load_labels()
        marker = self.state_dir / STATE_MARKER_FILE
        if marker.exists():
            self.require_state_marker()
        elif self.state_dir.exists():
            allowed = {HASHES_FILE, EVALUATIONS_FILE, EVENTS_FILE, LABELS_FILE}
            if any(path.name not in allowed or not path.is_file() for path in self.state_dir.iterdir()):
                raise SourceRewriteError("observer state marker is required")

    def require_complete_initial_collection(
        self, source_paths: tuple[str, ...], codes: tuple[str, ...],
    ) -> None:
        self.require_state_marker()
        observed_paths = {record["relative_path"] for record in self.load_source_snapshots()}
        if set(source_paths) != observed_paths:
            raise SourceRewriteError("observer state initial source snapshots are incomplete")
        evaluation_codes = {record["code"] for record in self.load_evaluations()}
        if not set(codes).issubset(evaluation_codes):
            raise SourceRewriteError("observer state initial evaluations are incomplete")

    def validate_evaluation(self, decision: SignalDecision, observed_at: datetime) -> None:
        record = _evaluation_record(decision, observed_at)
        matching = [
            item for item in self.load_evaluations()
            if item.get("event_id") == decision.event_id
        ]
        if matching and any(_canonical(item) != _canonical(record) for item in matching):
            raise SourceRewriteError(f"conflicting evaluation for event_id {decision.event_id}")
        if _evaluation_requires_event(record):
            event = _event_record_from_evaluation(record)
            for existing in self.load_events():
                if existing.get("event_id") == event["event_id"] and _canonical(existing) != _canonical(event):
                    raise SourceRewriteError(f"conflicting event for event_id {event['event_id']}")

    def validate_label(self, event_id: str, horizon: int, payload: Mapping[str, object]) -> None:
        record = _label_record(event_id, horizon, payload)
        matching = [
            item for item in self.load_labels()
            if item.get("event_id") == event_id and item.get("horizon") == horizon
        ]
        if matching and any(_canonical(item) != _canonical(record) for item in matching):
            raise SourceRewriteError(f"conflicting label for event_id {event_id} horizon {horizon}")

    def write_state_marker(self) -> None:
        path = self.state_dir / STATE_MARKER_FILE
        if path.exists():
            self.require_state_marker()
            return
        path.write_text(_canonical(STATE_MARKER) + "\n", encoding="utf-8", newline="\n")

    def require_state_marker(self) -> None:
        path = self.state_dir / STATE_MARKER_FILE
        if not path.is_file():
            raise SourceRewriteError("observer state marker is required")
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SourceRewriteError("observer state marker is invalid") from exc
        if _canonical(value) != _canonical(STATE_MARKER):
            raise SourceRewriteError("observer state marker is invalid")

    def record_evaluation(
        self, decision: SignalDecision, observed_at: datetime
    ) -> RecordResult:
        observed = _as_aware_datetime(observed_at)
        arrival = _as_aware_datetime(decision.item.arrival_dt)
        if observed < arrival:
            raise ValueError("observed_at cannot precede arrival_dt")

        record = _evaluation_record(decision, observed)
        existing = self.load_evaluations()
        matching = [item for item in existing if item.get("event_id") == decision.event_id]
        if matching:
            if any(_canonical(item) != _canonical(record) for item in matching):
                raise SourceRewriteError(
                    f"conflicting evaluation for event_id {decision.event_id}"
                )
            event_created = self._reconcile_event(matching[0])
            return RecordResult(
                False,
                event_created,
                "event_recovered" if event_created else "duplicate_evaluation",
            )

        late_import = observed > arrival
        active_before = self._episode_is_active_before(decision)
        self._append_record(EVALUATIONS_FILE, record)

        if late_import:
            return RecordResult(True, False, "late_import")
        if not decision.signal_detected:
            return RecordResult(True, False, "no_signal")
        if active_before:
            return RecordResult(True, False, "same_active_episode")
        if not decision.valid_event:
            return RecordResult(True, False, "invalid_event")

        self._reconcile_event(record)
        return RecordResult(True, True, "event_created")

    def append_label(
        self, event_id: str, horizon: int, payload: Mapping[str, object]
    ) -> RecordResult:
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        record = _label_record(event_id, horizon, payload)
        existing = self.load_labels()
        matching = [
            item
            for item in existing
            if item.get("event_id") == event_id and item.get("horizon") == horizon
        ]
        if matching:
            if any(_canonical(item) != _canonical(record) for item in matching):
                raise SourceRewriteError(
                    f"conflicting label for event_id {event_id} horizon {horizon}"
                )
            return RecordResult(False, False, "duplicate_label")

        self._append_record(LABELS_FILE, record)
        return RecordResult(True, False, "label_appended")

    def load_events(self) -> tuple[Mapping[str, object], ...]:
        return self._load_unique_keyed_records(EVENTS_FILE, "event_id", "event", reject_duplicates=True)

    def load_labels(self) -> tuple[Mapping[str, object], ...]:
        records = tuple(self._read_records(LABELS_FILE))
        keys: dict[tuple[str, int], Mapping[str, object]] = {}
        for record in records:
            event_id, horizon = record.get("event_id"), record.get("horizon")
            if not isinstance(event_id, str) or not isinstance(horizon, int):
                raise SourceRewriteError("invalid label key in labels.jsonl")
            key = (event_id, horizon)
            if key in keys:
                if _canonical(keys[key]) != _canonical(record):
                    raise SourceRewriteError(f"conflicting label for event_id {event_id} horizon {horizon}")
                raise SourceRewriteError(f"duplicate label for event_id {event_id} horizon {horizon}")
            keys[key] = record
        return records

    def load_evaluations(self) -> tuple[Mapping[str, object], ...]:
        records = tuple(self._read_records(EVALUATIONS_FILE))
        ids = set()
        for record in records:
            event_id = _validated_evaluation(record)
            if event_id in ids:
                raise SourceRewriteError(f"duplicate evaluation for event_id {event_id}")
            ids.add(event_id)
        return records

    def _episode_is_active_before(self, decision: SignalDecision) -> bool:
        return self._episode_is_active_before_record(
            decision.item.code, _as_aware_datetime(decision.item.arrival_dt)
        )

    def _episode_is_active_before_record(self, code: str, arrival: datetime) -> bool:
        records = []
        for item in self.load_evaluations():
            if item.get("code") != code:
                continue
            item_arrival = _parse_datetime(item.get("arrival_dt"))
            item_observed = _parse_datetime(item.get("observed_at"))
            if item_arrival >= arrival or item_observed > item_arrival:
                continue
            records.append(item)
        records.sort(key=lambda item: item["arrival_dt"])

        active = False
        for item in records:
            active = bool(item["signal_detected"])
        return active

    def _reconcile_event(self, evaluation: Mapping[str, object]) -> bool:
        if not _evaluation_requires_event(evaluation):
            return False
        code = evaluation.get("code")
        if not isinstance(code, str):
            raise SourceRewriteError("stored evaluation code is invalid")
        arrival = _parse_datetime(evaluation.get("arrival_dt"))
        if self._episode_is_active_before_record(code, arrival):
            return False

        event = _event_record_from_evaluation(evaluation)
        event_id = event["event_id"]
        matching = [
            item
            for item in self._load_unique_keyed_records(EVENTS_FILE, "event_id", "event", reject_duplicates=True)
            if item.get("event_id") == event_id
        ]
        if matching:
            if any(_canonical(item) != _canonical(event) for item in matching):
                raise SourceRewriteError(f"conflicting event for event_id {event_id}")
            return False
        self._append_record(EVENTS_FILE, event)
        return True

    def _append_unique(
        self,
        filename: str,
        record: Mapping[str, object],
        key: tuple[str, str],
        label: str,
    ) -> bool:
        matching = [
            item
            for item in self._read_records(filename)
            if (str(item.get("relative_path")), str(item.get("observed_at"))) == key
        ]
        if matching:
            if any(_canonical(item) != _canonical(record) for item in matching):
                raise SourceRewriteError(f"conflicting {label} for {key[0]}")
            return False
        self._append_record(filename, record)
        return True

    def _load_unique_keyed_records(
        self, filename: str, key_name: str, label: str, reject_duplicates: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        unique: dict[str, dict[str, object]] = {}
        for record in self._read_records(filename):
            key = record.get(key_name)
            if not isinstance(key, str) or not key:
                raise SourceRewriteError(f"invalid {label} key in {filename}")
            prior = unique.get(key)
            if prior is None:
                unique[key] = record
            elif reject_duplicates or _canonical(prior) != _canonical(record):
                raise SourceRewriteError(f"conflicting {label} for {key_name} {key}")
        return tuple(unique.values())

    def _append_record(self, filename: str, record: Mapping[str, object]) -> None:
        path = self.state_dir / filename
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(record))
            handle.write("\n")

    def _read_records(self, filename: str) -> list[dict[str, object]]:
        path = self.state_dir / filename
        if not path.exists():
            return []
        records = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line, parse_constant=_reject_json_constant)
                    if not isinstance(record, dict):
                        raise SourceRewriteError(
                            f"invalid JSONL in {filename} at line {line_number}: record is not an object"
                        )
                    records.append(record)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, SourceRewriteError):
                raise
            raise SourceRewriteError(f"invalid JSONL in {filename}") from exc
        return records


def _evaluation_record(decision: SignalDecision, observed_at: datetime) -> dict[str, object]:
    item = asdict(decision.item)
    return {
        "event_id": decision.event_id,
        "code": decision.item.code,
        "arrival_dt": _iso_datetime(decision.item.arrival_dt),
        "signal_date": decision.item.signal_date.isoformat(),
        "observed_at": _iso_datetime(observed_at),
        "signal_detected": decision.signal_detected,
        "valid_event": decision.valid_event,
        "reasons": list(decision.reasons),
        "item": _json_value(item),
    }


def _source_snapshot_record(
    relative_path: str, content: bytes, observed_at: datetime,
) -> dict[str, object]:
    if not isinstance(content, bytes):
        raise TypeError("source snapshot content must be bytes")
    return {
        "relative_path": _validate_relative_path(relative_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_length": len(content),
        "observed_at": _iso_datetime(observed_at),
    }


def _validated_source_snapshot(record: Mapping[str, object]) -> str:
    path = _validate_relative_path(record.get("relative_path"))
    digest = _validate_sha256(record.get("sha256"))
    length = record.get("byte_length")
    _parse_datetime(record.get("observed_at"))
    if set(record) != {"relative_path", "sha256", "byte_length", "observed_at"}:
        raise SourceRewriteError(f"invalid source snapshot fields for {path}")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise SourceRewriteError(f"invalid source snapshot length for {path}")
    if not digest:
        raise SourceRewriteError(f"invalid source snapshot digest for {path}")
    return path


def _latest_snapshot_by_path(
    history: tuple[Mapping[str, object], ...],
) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for record in history:
        latest[record["relative_path"]] = record
    return latest


def _validated_evaluation(record: Mapping[str, object]) -> str:
    required = {
        "event_id", "code", "arrival_dt", "signal_date", "observed_at",
        "signal_detected", "valid_event", "reasons", "item",
    }
    if set(record) != required:
        raise SourceRewriteError("stored evaluation fields are invalid")
    event_id = record["event_id"]
    code = record["code"]
    if not isinstance(event_id, str) or not isinstance(code, str) or not code:
        raise SourceRewriteError("stored evaluation identity is invalid")
    arrival = _parse_datetime(record["arrival_dt"])
    observed = _parse_datetime(record["observed_at"])
    try:
        signal_date = date.fromisoformat(record["signal_date"])
    except (TypeError, ValueError) as exc:
        raise SourceRewriteError("stored evaluation signal date is invalid") from exc
    expected = hashlib.sha256(
        "|".join([VERSION, code, arrival.date().isoformat(), signal_date.isoformat()]).encode("utf-8")
    ).hexdigest()
    if event_id != expected or observed < arrival:
        raise SourceRewriteError("stored evaluation identity is invalid")
    if not isinstance(record["signal_detected"], bool) or not isinstance(record["valid_event"], bool):
        raise SourceRewriteError("stored evaluation decision is invalid")
    if record["valid_event"] and not record["signal_detected"]:
        raise SourceRewriteError("stored evaluation decision is invalid")
    if not isinstance(record["reasons"], list) or not all(isinstance(value, str) for value in record["reasons"]):
        raise SourceRewriteError("stored evaluation reasons are invalid")
    item = record["item"]
    if not isinstance(item, Mapping):
        raise SourceRewriteError("stored evaluation item is invalid")
    if item.get("code") != code or item.get("arrival_dt") != record["arrival_dt"]:
        raise SourceRewriteError("stored evaluation item identity is invalid")
    if item.get("signal_date") != record["signal_date"]:
        raise SourceRewriteError("stored evaluation item identity is invalid")
    hashes = item.get("source_hashes")
    if not isinstance(hashes, list) or not all(isinstance(value, str) and len(value) == 64 for value in hashes):
        raise SourceRewriteError("stored evaluation source hashes are invalid")
    return event_id


def _label_record(event_id: str, horizon: int, payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return {
        "event_id": event_id,
        "horizon": horizon,
        "payload": _json_value(payload),
    }


def _evaluation_requires_event(evaluation: Mapping[str, object]) -> bool:
    arrival = _parse_datetime(evaluation.get("arrival_dt"))
    observed = _parse_datetime(evaluation.get("observed_at"))
    return (
        observed <= arrival
        and evaluation.get("signal_detected") is True
        and evaluation.get("valid_event") is True
    )


def _event_record_from_evaluation(evaluation: Mapping[str, object]) -> dict[str, object]:
    item = evaluation.get("item")
    if not isinstance(item, Mapping):
        raise SourceRewriteError("stored evaluation input is invalid")
    arrival = _parse_datetime(evaluation.get("arrival_dt"))
    event_id = evaluation.get("event_id")
    code = evaluation.get("code")
    if not isinstance(event_id, str) or not isinstance(code, str):
        raise SourceRewriteError("stored evaluation identity is invalid")
    return {
        "event_id": event_id,
        "code": code,
        "arrival_dt": _iso_datetime(arrival),
        "arrival_date": arrival.date().isoformat(),
        "signal_date": evaluation.get("signal_date"),
        "observed_at": evaluation.get("observed_at"),
        "entry_open": item.get("entry_open"),
        "price_proved": item.get("price_proved"),
        "price_reason": item.get("price_reason"),
        "background": item.get("background"),
        "source_hashes": item.get("source_hashes"),
        "decision": {
            "event_id": event_id,
            "signal_detected": evaluation.get("signal_detected"),
            "valid_event": evaluation.get("valid_event"),
            "reasons": evaluation.get("reasons"),
        },
        "input": _json_value(item),
    }


def _validate_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative_path must be a non-empty relative path")
    normalized = relative_path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or PureWindowsPath(relative_path).is_absolute() or ".." in posix.parts:
        raise ValueError("relative_path must not escape the source root")
    return posix.as_posix()


def _validate_sha256(sha256: str) -> str:
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError("sha256 must be a 64-character hexadecimal digest") from exc
    return sha256.lower()


def _as_aware_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetimes")
    return value


def _iso_datetime(value: datetime) -> str:
    return _as_aware_datetime(value).isoformat()


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise SourceRewriteError("stored timestamp is invalid")
    return _as_aware_datetime(datetime.fromisoformat(value))


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _iso_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"value is not JSON-serializable: {type(value).__name__}")


def _canonical(record: Mapping[str, object]) -> str:
    return json.dumps(
        _json_value(record),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
