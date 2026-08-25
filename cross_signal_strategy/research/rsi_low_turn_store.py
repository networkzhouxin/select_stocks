"""Append-only local state for the order-free RSI low-turn shadow observer."""

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

from cross_signal_strategy.research.rsi_low_turn_shadow import SignalDecision


EVALUATIONS_FILE = "evaluations.jsonl"
EVENTS_FILE = "events.jsonl"
HASHES_FILE = "source_hashes.jsonl"
LABELS_FILE = "labels.jsonl"


@dataclass(frozen=True)
class RecordResult:
    written: bool
    event_created: bool
    reason: str


class SourceRewriteError(RuntimeError):
    """Raised when an append-only key is supplied with different content."""


class ShadowStore:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def record_source_hash(
        self, relative_path: str, sha256: str, observed_at: datetime
    ) -> bool:
        path = _validate_relative_path(relative_path)
        digest = _validate_sha256(sha256)
        observed = _iso_datetime(observed_at)
        record = {
            "relative_path": path,
            "sha256": digest,
            "observed_at": observed,
        }
        existing = self._read_records(HASHES_FILE)
        matching_path = [item for item in existing if item.get("relative_path") == path]
        if matching_path and any(item.get("sha256") != digest for item in matching_path):
            raise SourceRewriteError(f"source hash changed for {path}")
        return self._append_unique(HASHES_FILE, record, (path, observed), "source hash")

    def record_evaluation(
        self, decision: SignalDecision, observed_at: datetime
    ) -> RecordResult:
        observed = _as_aware_datetime(observed_at)
        arrival = _as_aware_datetime(decision.item.arrival_dt)
        if observed < arrival:
            raise ValueError("observed_at cannot precede arrival_dt")

        record = _evaluation_record(decision, observed)
        existing = self._read_records(EVALUATIONS_FILE)
        matching = [item for item in existing if item.get("event_id") == decision.event_id]
        if matching:
            if _canonical(matching[0]) != _canonical(record):
                raise SourceRewriteError(
                    f"conflicting evaluation for event_id {decision.event_id}"
                )
            return RecordResult(False, False, "duplicate_evaluation")

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

        self._append_record(EVENTS_FILE, _event_record(decision, observed))
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

        record = {
            "event_id": event_id,
            "horizon": horizon,
            "payload": _json_value(payload),
        }
        existing = self._read_records(LABELS_FILE)
        matching = [
            item
            for item in existing
            if item.get("event_id") == event_id and item.get("horizon") == horizon
        ]
        if matching:
            if _canonical(matching[0]) != _canonical(record):
                raise SourceRewriteError(
                    f"conflicting label for event_id {event_id} horizon {horizon}"
                )
            return RecordResult(False, False, "duplicate_label")

        self._append_record(LABELS_FILE, record)
        return RecordResult(True, False, "label_appended")

    def load_events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._read_records(EVENTS_FILE))

    def load_labels(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._read_records(LABELS_FILE))

    def _episode_is_active_before(self, decision: SignalDecision) -> bool:
        arrival = _as_aware_datetime(decision.item.arrival_dt)
        records = []
        for item in self._read_records(EVALUATIONS_FILE):
            if item.get("code") != decision.item.code:
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

    def _append_unique(
        self,
        filename: str,
        record: Mapping[str, object],
        key: tuple[str, str],
        label: str,
    ) -> bool:
        for item in self._read_records(filename):
            item_key = (str(item.get("relative_path")), str(item.get("observed_at")))
            if item_key != key:
                continue
            if _canonical(item) != _canonical(record):
                raise SourceRewriteError(f"conflicting {label} for {key[0]}")
            return False
        self._append_record(filename, record)
        return True

    def _append_record(self, filename: str, record: Mapping[str, object]) -> None:
        path = self.state_dir / filename
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(record))
            handle.write("\n")

    def _read_records(self, filename: str) -> list[dict[str, object]]:
        path = self.state_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


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


def _event_record(decision: SignalDecision, observed_at: datetime) -> dict[str, object]:
    return {
        "event_id": decision.event_id,
        "code": decision.item.code,
        "arrival_dt": _iso_datetime(decision.item.arrival_dt),
        "signal_date": decision.item.signal_date.isoformat(),
        "observed_at": _iso_datetime(observed_at),
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
