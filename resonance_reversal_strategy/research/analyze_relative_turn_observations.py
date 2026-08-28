"""Read-only validation and descriptive analysis for relative observation logs."""

import argparse
import hashlib
import html
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile
import weakref
from collections import Counter
from datetime import date, datetime, time
from typing import NamedTuple


TRAIN_START = date(2019, 1, 1)
TRAIN_END = date(2021, 12, 31)
SESSION_MANIFEST_SCHEMA_VERSION = 1
SESSION_MANIFEST_MARKET = "XSHG"
SESSION_MANIFEST_SOURCE = "JoinQuant get_all_trade_days"
CANDIDATE_BUILD = "20260827.4"
BASELINE_BUILD = "20260827.3"
RELATIVE_OBSERVATION_FINGERPRINT = "f47d32b87be6d926"
BASELINE_FILLED_COUNT = 138
BASELINE_FINAL_ASSET = 23856.40
PARAMETER_FINGERPRINT = "e1227fbd8b4a884e"
POOL_FINGERPRINT = "9123995edeb1ed84"
FORMAL_EVENT_FINGERPRINT = "1c0b8a22f48c97c3"
LOG_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<time>\d{2}:\d{2}:\d{2}))?\s+-"
)
BRANCHES = ("HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE")
DIRECTIONS = ("BUY_TURN", "SELL_TURN")
HORIZONS = (1, 3, 5)
INDICATORS = frozenset(("BOLL", "KDJ", "RSI"))
SOURCES = frozenset(("HARD", "RELATIVE"))
TRADING_LOG_TIME = time(9, 35)
AFTER_CLOSE_LOG_TIME = time(15, 30)
STRICT_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
STRUCTURED_EVENT_TOKENS = frozenset((
    "relative_resonance_observation", "observation_outcome",
    "strategy_initialized", "resonance_decision", "order_transition",
    "portfolio_summary", "signal_snapshot",
))
STRUCTURED_EVENT_FIELD_RE = re.compile(
    r'"event"\s*:\s*"(?:%s)"' % "|".join(
        re.escape(token) for token in sorted(STRUCTURED_EVENT_TOKENS)
    ),
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MAX_SESSION_MANIFEST_BYTES = 256 * 1024


class SessionCalendarMetadata(NamedTuple):
    schema_version: int
    market: str
    coverage_start: str
    coverage_end: str
    source: str
    session_count: int


def _build_session_manifest_api():
    issued_snapshots = weakref.WeakKeyDictionary()

    class ValidatedSessionManifest:
        """An immutable manifest capability that only this module can issue."""

        __slots__ = ("sessions", "sha256", "metadata", "__weakref__")

        def __init__(self, *args, **kwargs):
            raise TypeError("validated session manifest is required")

        def __setattr__(self, name, value):
            raise AttributeError("validated session manifest is immutable")

        def __delattr__(self, name):
            raise AttributeError("validated session manifest is immutable")

    def issue(sessions, sha256, metadata):
        if (type(sessions) is not tuple or type(sha256) is not str
                or type(metadata) is not SessionCalendarMetadata):
            raise TypeError("validated session manifest is required")
        capability = object.__new__(ValidatedSessionManifest)
        object.__setattr__(capability, "sessions", sessions)
        object.__setattr__(capability, "sha256", sha256)
        object.__setattr__(capability, "metadata", metadata)
        issued_snapshots[capability] = (sessions, sha256, metadata)
        return capability

    def has_exact_field_types(capability):
        if type(capability) is not ValidatedSessionManifest:
            return False
        try:
            sessions = capability.sessions
            sha256 = capability.sha256
            metadata = capability.metadata
        except AttributeError:
            return False
        if (type(sessions) is not tuple or type(sha256) is not str
                or type(metadata) is not SessionCalendarMetadata
                or type(metadata.schema_version) is not int
                or type(metadata.market) is not str
                or type(metadata.coverage_start) is not str
                or type(metadata.coverage_end) is not str
                or type(metadata.source) is not str
                or type(metadata.session_count) is not int):
            return False
        return all(type(session) is date for session in sessions)

    def is_issued_snapshot(capability):
        if not has_exact_field_types(capability):
            return False
        snapshot = issued_snapshots.get(capability)
        if snapshot is None:
            return False
        return snapshot == (
            capability.sessions, capability.sha256, capability.metadata,
        )

    def _validate_session_manifest_bytes(raw_bytes, expected_sha256):
        """Validate one frozen manifest from its original UTF-8 bytes."""
        if type(raw_bytes) is not bytes:
            raise ValueError("session calendar manifest must be bytes")
        if type(expected_sha256) is not str or not SHA256_RE.fullmatch(expected_sha256):
            raise ValueError("session calendar sha256 must be 64 hexadecimal characters")
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError("session calendar sha256 mismatch")
        try:
            payload = json.loads(
                raw_bytes.decode("utf-8"), parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid session calendar manifest") from exc
        required = {
            "schema_version", "market", "coverage_start", "coverage_end", "source", "sessions",
        }
        if type(payload) is not dict or set(payload) != required:
            raise ValueError("invalid session calendar manifest schema")
        if (type(payload["schema_version"]) is not int
                or type(payload["market"]) is not str
                or type(payload["coverage_start"]) is not str
                or type(payload["coverage_end"]) is not str
                or type(payload["source"]) is not str
                or payload["schema_version"] != SESSION_MANIFEST_SCHEMA_VERSION
                or payload["market"] != SESSION_MANIFEST_MARKET
                or payload["source"] != SESSION_MANIFEST_SOURCE
                or payload["coverage_start"] != TRAIN_START.isoformat()
                or payload["coverage_end"] != TRAIN_END.isoformat()):
            raise ValueError("invalid session calendar manifest schema")
        sessions = payload["sessions"]
        if type(sessions) is not list or not sessions:
            raise ValueError("invalid session calendar manifest sessions")
        normalized = []
        previous = None
        for value in sessions:
            if type(value) is not str or not _valid_text(value):
                raise ValueError("invalid session calendar manifest session")
            try:
                session = _calendar_date(value)
            except ValueError as exc:
                raise ValueError("invalid session calendar manifest session") from exc
            if not TRAIN_START <= session <= TRAIN_END:
                raise ValueError("session calendar manifest session outside coverage")
            if previous is not None and session <= previous:
                raise ValueError("session calendar manifest sessions must be strictly increasing")
            normalized.append(session)
            previous = session
        return issue(
            tuple(normalized), str(actual_sha256),
            SessionCalendarMetadata(
                int(payload["schema_version"]), str(payload["market"]),
                str(payload["coverage_start"]), str(payload["coverage_end"]),
                str(payload["source"]), int(len(normalized)),
            ),
        )

    def require(manifest):
        if (not is_issued_snapshot(manifest)
                or not SHA256_RE.fullmatch(manifest.sha256)):
            raise TypeError("validated session manifest is required")
        if (manifest.metadata.schema_version != SESSION_MANIFEST_SCHEMA_VERSION
                or manifest.metadata.market != SESSION_MANIFEST_MARKET
                or manifest.metadata.coverage_start != TRAIN_START.isoformat()
                or manifest.metadata.coverage_end != TRAIN_END.isoformat()
                or manifest.metadata.source != SESSION_MANIFEST_SOURCE
                or manifest.metadata.session_count != len(manifest.sessions)):
            raise TypeError("validated session manifest is required")
        previous = None
        for session in manifest.sessions:
            if not TRAIN_START <= session <= TRAIN_END:
                raise TypeError("validated session manifest is required")
            if previous is not None and session <= previous:
                raise TypeError("validated session manifest is required")
            previous = session
        return manifest

    return _validate_session_manifest_bytes, require


validate_session_calendar_manifest, _require_validated_session_manifest = (
    _build_session_manifest_api()
)
del _build_session_manifest_api


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def parse_joinquant_log_line(line, ordinal):
    """Return one finite JSON object from a JoinQuant log line, if present."""
    text = html.unescape(line.strip())
    payload_start = text.find("{")
    if payload_start < 0:
        return None
    match = LOG_TIMESTAMP_RE.match(text)
    try:
        payload = json.loads(
            text[payload_start:], parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except json.JSONDecodeError as exc:
        if match and STRUCTURED_EVENT_FIELD_RE.search(text[payload_start:]):
            payload = {"_parse_error": "invalid structured JSON: %s" % exc.msg}
        else:
            return None
    except ValueError as exc:
        payload = {"_parse_error": str(exc)}
    if not isinstance(payload, dict):
        return None
    result = dict(payload)
    result["_log_date"] = match.group("date") if match else None
    result["_log_timestamp"] = (
        match.group("date") + "T" + match.group("time")
        if match and match.group("time") else None
    )
    result["_ordinal"] = int(ordinal)
    return result


def load_log_records(paths):
    """Read explicit log paths without changing the input files."""
    records = []
    ordinal = 0
    normalized_paths = sorted(
        (pathlib.Path(path_value).expanduser().resolve(strict=True) for path_value in paths),
        key=lambda path: str(path),
    )
    for source_index, path_value in enumerate(normalized_paths):
        with pathlib.Path(path_value).open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                ordinal += 1
                record = parse_joinquant_log_line(line, ordinal)
                if record is not None:
                    record["_source_path"] = str(path_value)
                    record["_source_index"] = source_index
                    record["_source_line"] = line_number
                    records.append(record)
    return records


def _calendar_date(value):
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        raise ValueError("date must not contain a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("date must be an ISO string")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be complete ISO date")
    return date.fromisoformat(value)


def _reject_duplicate_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def read_session_calendar_manifest_bytes(path_value):
    """Read only the bounded manifest bytes needed for its frozen contract."""
    path = pathlib.Path(path_value)
    with path.open("rb") as stream:
        raw_bytes = stream.read(MAX_SESSION_MANIFEST_BYTES + 1)
    if len(raw_bytes) > MAX_SESSION_MANIFEST_BYTES:
        raise ValueError("session calendar manifest exceeds maximum size")
    return raw_bytes


def _is_training_date(value, label, errors):
    try:
        parsed = _calendar_date(value)
    except ValueError:
        errors.append("invalid %s date: %r" % (label, value))
        return None
    if parsed is None:
        errors.append("missing %s date" % label)
        return None
    if not TRAIN_START <= parsed <= TRAIN_END:
        errors.append("%s outside 2019-2021: %s" % (label, parsed))
    return parsed


def _strict_log_timestamp(record, label, errors):
    value = record.get("_log_timestamp")
    if type(value) is not str or not STRICT_TIMESTAMP_RE.fullmatch(value):
        errors.append("invalid %s log timestamp" % label)
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        errors.append("invalid %s log timestamp" % label)
        return None


def _parse_strict_log_timestamp(record):
    value = record.get("_log_timestamp")
    if type(value) is not str or not STRICT_TIMESTAMP_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _validate_registration_log_timestamp(record, label, signal_date, errors):
    timestamp = _strict_log_timestamp(record, label, errors)
    if timestamp is None or signal_date is None:
        return timestamp
    if timestamp.date() <= signal_date:
        errors.append("%s log timestamp must follow signal date" % label)
    if timestamp.time() < TRADING_LOG_TIME:
        errors.append("%s log timestamp before 09:35" % label)
    return timestamp


def _validate_outcome_log_timestamp(record, label, closing_date, errors):
    timestamp = _strict_log_timestamp(record, label, errors)
    if timestamp is None or closing_date is None:
        return timestamp
    if timestamp.date() != closing_date:
        errors.append("%s log date mismatch" % label)
    if timestamp.time() < AFTER_CLOSE_LOG_TIME:
        errors.append("%s log timestamp before 15:30" % label)
    return timestamp


def _valid_text(value):
    """Accept only non-empty Unicode scalar strings for contract text fields."""
    if type(value) is not str or not value.strip():
        return False
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def build_relative_observation_id(record):
    """Reproduce the frozen Task4 relative-observation ID without strategy imports."""
    parts = [
        "RELATIVE", record["branch"], record["direction"], record["code"],
    ]
    source_map = record["hard_or_relative_source_by_indicator"]
    date_map = record["supporter_event_dates"]
    for indicator in sorted(record["supporters"]):
        parts.append("%s:%s:%s" % (
            indicator, source_map[indicator], _calendar_date(date_map[indicator]),
        ))
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return "RELATIVE:" + digest


def _expected_relative_observation_id(record):
    supporters = record.get("supporters")
    sources = record.get("hard_or_relative_source_by_indicator")
    dates = record.get("supporter_event_dates")
    if (not _valid_text(record.get("code"))
            or record.get("direction") not in DIRECTIONS
            or record.get("branch") not in BRANCHES
            or not isinstance(supporters, (list, tuple))
            or not supporters
            or any(not _valid_text(item) for item in supporters)
            or len(set(supporters)) != len(supporters)
            or not isinstance(sources, dict)
            or not isinstance(dates, dict)
            or set(sources) != set(supporters)
            or set(dates) != set(supporters)
            or any(not _valid_text(value) or value not in SOURCES
                   for value in sources.values())):
        return None
    try:
        if any(_calendar_date(dates[indicator]) is None for indicator in supporters):
            return None
        return build_relative_observation_id(record)
    except (KeyError, TypeError, ValueError):
        return None


def _is_formal_registration_record(record):
    return (record.get("event") == "resonance_decision"
            and record.get("accepted") is True
            and record.get("reason") == "COMPLETE_RESONANCE")


def _classify_record_namespace(record, errors):
    """Classify one record once; conflicting records enter neither contract."""
    if (record.get("event") == "observation_outcome"
            and record.get("_outcome_payload_valid") is False):
        return "invalid"
    relative_id = record.get("relative_observation_id")
    kind = record.get("observation_kind")
    resonance_id = record.get("resonance_id")
    event = record.get("event")
    relative_id_valid = _valid_text(relative_id) and relative_id.startswith("RELATIVE:")
    relative_resonance = _valid_text(resonance_id) and resonance_id.startswith("RELATIVE:")
    relative_marker = (
        "relative_observation_id" in record
        or "observation_kind" in record
        or relative_resonance
    )
    if relative_marker:
        valid = relative_id_valid and kind == "RELATIVE_RESONANCE"
        if event == "observation_outcome":
            valid = valid and resonance_id == relative_id
        elif event == "relative_resonance_observation":
            valid = valid and (resonance_id in (None, ""))
        else:
            valid = False
        if not valid:
            if relative_resonance and relative_id is None and kind is None:
                errors.append("relative namespace record missing relative markers: %s" % resonance_id)
            else:
                errors.append("invalid or conflicting relative record namespace")
            return "invalid"
        return "relative"
    if _is_formal_registration_record(record):
        if _valid_text(resonance_id) and not resonance_id.startswith("RELATIVE:"):
            return "formal"
        errors.append("invalid formal record namespace")
        return "invalid"
    if event == "observation_outcome":
        if _valid_text(resonance_id) and not resonance_id.startswith("RELATIVE:"):
            return "formal"
        errors.append("invalid formal outcome namespace")
        return "invalid"
    return "other"


def _is_relative_record(record):
    return record.get("_record_namespace") == "relative"


def _matching_relative_record(record):
    return (
        _is_relative_record(record)
        and record.get("build") == CANDIDATE_BUILD
        and record.get("relative_observation_fingerprint") == RELATIVE_OBSERVATION_FINGERPRINT
    )


def _text(record, field, label, errors):
    value = record.get(field)
    if not _valid_text(value):
        errors.append("missing or invalid %s: %r" % (label, value))
        return None
    return value


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def _validate_support_contract(record, prefix, signal_date, session_evidence,
                               session_calendar, errors):
    supporters = record.get("supporters")
    if (not isinstance(supporters, (list, tuple)) or not supporters
            or any(not _valid_text(item) for item in supporters)
            or len(set(supporters)) != len(supporters)
            or not set(supporters).issubset(INDICATORS)):
        errors.append("invalid %s supporters" % prefix)
        return
    support_set = set(supporters)
    dates = record.get("supporter_event_dates")
    sources = record.get("hard_or_relative_source_by_indicator")
    if not isinstance(dates, dict) or set(dates) != support_set:
        errors.append("invalid %s supporter_event_dates" % prefix)
    else:
        has_signal_date_evidence = False
        supported_dates = []
        for indicator in sorted(dates):
            supported_date = _is_training_date(
                dates[indicator], "%s supporter_event_dates" % prefix, errors,
            )
            if supported_date is not None:
                supported_dates.append(supported_date)
            if (supported_date is not None and signal_date is not None
                    and supported_date > signal_date):
                errors.append("%s supporter date after signal date" % prefix)
            if supported_date is not None and supported_date == signal_date:
                has_signal_date_evidence = True
        if signal_date is not None and not has_signal_date_evidence:
            errors.append("%s supporters lack signal-date evidence" % prefix)
        if (signal_date is not None and len(supported_dates) == len(dates)
                and any(value < signal_date for value in supported_dates)):
            if signal_date not in session_calendar or session_calendar.index(signal_date) == 0:
                errors.append("%s supporter window unverifiable in session calendar" % prefix)
            else:
                previous_session = session_calendar[session_calendar.index(signal_date) - 1]
                if session_evidence.get(signal_date) != previous_session:
                    errors.append("%s supporter window session evidence mismatch" % prefix)
                allowed_dates = {signal_date, previous_session}
                if any(value not in allowed_dates for value in supported_dates):
                    errors.append("invalid %s supporter trading-session window" % prefix)
    if (not isinstance(sources, dict) or set(sources) != support_set
            or any(not _valid_text(value) or value not in SOURCES
                   for value in sources.values())):
        errors.append("invalid %s hard_or_relative_source_by_indicator" % prefix)
        return
    branch = record.get("branch")
    valid = {
        "HARD_BOLL_SOFT_OSC": (
            ({"BOLL", "RSI"}, {"BOLL": "HARD", "RSI": "RELATIVE"}),
            ({"BOLL", "KDJ"}, {"BOLL": "HARD", "KDJ": "RELATIVE"}),
            ({"BOLL", "KDJ", "RSI"}, {
                "BOLL": "HARD", "KDJ": "RELATIVE", "RSI": "RELATIVE",
            }),
        ),
        "SOFT_ALL_THREE": (
            ({"BOLL", "KDJ", "RSI"}, {
                "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
            }),
        ),
    }
    if _valid_text(branch) and branch in valid and not any(support_set == expected_set and sources == expected_sources
                                   for expected_set, expected_sources in valid[branch]):
        errors.append("impossible %s branch/supporters/source contract" % prefix)


def _validate_candidate_initializations(records, errors):
    initialized = [record for record in records if record.get("event") == "strategy_initialized"]
    if not initialized:
        errors.append("missing candidate strategy_initialized record")
        return False
    expected = {
        "build": CANDIDATE_BUILD,
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
    }
    valid = True
    for record in initialized:
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                errors.append("candidate initialization %s mismatch: %r" % (
                    field, record.get(field),
                ))
                valid = False
    return valid


def _validate_baseline_initializations(records, errors):
    initialized = [record for record in records if record.get("event") == "strategy_initialized"]
    if not initialized:
        errors.append("missing baseline strategy_initialized record")
        return False
    expected = {
        "build": BASELINE_BUILD,
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
    }
    valid = True
    for record in initialized:
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                errors.append("baseline initialization %s mismatch: %r" % (field, record.get(field)))
                valid = False
    return valid


def _validated_session_calendar(records, role, initialization_valid, errors):
    """Use unconditional after-close portfolio summaries as session evidence."""
    if not initialization_valid:
        errors.append("%s session calendar unavailable: invalid initialization" % role)
        return ()
    evidence = {}
    unusable_dates = set()
    for record in records:
        if record.get("event") != "portfolio_summary":
            continue
        closing_date = _is_training_date(
            record.get("closing_date"), "%s session summary closing" % role, errors,
        )
        timestamp = _strict_log_timestamp(record, "%s session summary" % role, errors)
        usable = closing_date is not None and timestamp is not None
        if timestamp is not None and closing_date is not None:
            if timestamp.date() != closing_date:
                errors.append("%s session summary log date mismatch" % role)
                usable = False
            if timestamp.time() < AFTER_CLOSE_LOG_TIME:
                errors.append("%s session summary log timestamp before 15:30" % role)
                usable = False
        if closing_date is None:
            continue
        public = {key: value for key, value in record.items() if not key.startswith("_")}
        previous = evidence.get(closing_date)
        if previous is not None and previous != public:
            errors.append("conflicting %s session calendar evidence: %s" % (role, closing_date))
            unusable_dates.add(closing_date)
        elif usable:
            evidence[closing_date] = public
        else:
            unusable_dates.add(closing_date)
    for closing_date in unusable_dates:
        evidence.pop(closing_date, None)
    if not evidence:
        errors.append("missing %s portfolio_summary session evidence" % role)
    return tuple(sorted(evidence))


def _validate_session_evidence(records, role, summary_calendar, manifest_calendar, errors):
    """Evidence may expose a missing summary, but cannot create a session."""
    dates = set()
    for record in records:
        event = record.get("event")
        if event == "signal_snapshot":
            try:
                decision_date = _calendar_date(record.get("decision_date"))
                signal_date = _calendar_date(record.get("signal_date"))
            except ValueError:
                decision_date = signal_date = None
            if decision_date is not None:
                dates.add(decision_date)
            if signal_date is not None:
                dates.add(signal_date)
        if event == "order_transition" and record.get("outcome") == "FILLED":
            timestamp = _parse_strict_log_timestamp(record)
            if timestamp is not None:
                dates.add(timestamp.date())
        if event == "observation_outcome":
            payload = record.get("outcome")
            if type(payload) is not dict:
                continue
            try:
                closing_date = _calendar_date(payload.get("closing_date"))
            except ValueError:
                closing_date = None
            if closing_date is not None:
                dates.add(closing_date)
    for session_date in sorted(dates):
        if session_date not in manifest_calendar:
            errors.append("%s session evidence absent from manifest: %s" % (role, session_date))
        if session_date not in summary_calendar:
            errors.append("%s session evidence missing summary: %s" % (role, session_date))


def _validate_baseline_observation_namespaces(records, errors):
    for record in records:
        event = record.get("event")
        namespace = record.get("_record_namespace")
        if event == "relative_resonance_observation":
            errors.append("baseline relative observation registration is not allowed")
        elif event == "observation_outcome" and namespace != "formal":
            errors.append("baseline non-formal observation outcome is not allowed")
        elif _is_formal_registration_record(record) and namespace != "formal":
            errors.append("baseline non-formal observation registration is not allowed")


def _expected_outcome_session(event_date, horizon, calendar):
    if event_date is None or event_date not in calendar:
        return None
    index = calendar.index(event_date) + horizon
    return calendar[index] if index < len(calendar) else None


def _validate_outcome_sessions(registrations, outcomes, calendar, role, horizons, errors):
    for resonance_id, registration in registrations.items():
        try:
            event_date = _calendar_date(registration.get("signal_date"))
        except ValueError:
            continue
        for horizon in horizons:
            expected = _expected_outcome_session(event_date, horizon, calendar)
            if expected is None:
                errors.append("%s session calendar missing horizon %s coverage: %s" % (
                    role, horizon, resonance_id,
                ))
                continue
            record = outcomes.get((resonance_id, horizon))
            payload = record.get("outcome") if record is not None else None
            if type(payload) is not dict:
                continue
            try:
                closing_date = _calendar_date(payload.get("closing_date"))
            except ValueError:
                continue
            if closing_date != expected:
                errors.append("%s outcome closing session mismatch: %s/%s" % (
                    role, resonance_id, horizon,
                ))


def _candidate_session_evidence(records, errors):
    """Build explicit candidate decision-date to prior-signal-date evidence."""
    evidence = {}
    unusable_decisions = set()
    expected_fingerprints = {
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
    }
    for record in records:
        if record.get("event") != "signal_snapshot":
            continue
        decision_date = _is_training_date(
            record.get("decision_date"), "candidate signal_snapshot decision", errors,
        )
        signal_date = _is_training_date(
            record.get("signal_date"), "candidate signal_snapshot signal", errors,
        )
        timestamp = _strict_log_timestamp(record, "candidate signal_snapshot", errors)
        usable = decision_date is not None and signal_date is not None and timestamp is not None
        if record.get("build") != CANDIDATE_BUILD:
            errors.append("candidate signal_snapshot build mismatch: %r" % record.get("build"))
            usable = False
        for field, expected in expected_fingerprints.items():
            if record.get(field) != expected:
                errors.append("candidate signal_snapshot %s mismatch: %r" % (
                    field, record.get(field),
                ))
                usable = False
        if not _valid_text(record.get("code")):
            errors.append("invalid candidate signal_snapshot code")
            usable = False
        if type(record.get("valid")) is not bool:
            errors.append("invalid candidate signal_snapshot valid")
            usable = False
        if decision_date is not None and signal_date is not None and decision_date <= signal_date:
            errors.append("candidate signal_snapshot signal date must precede decision date")
            usable = False
        if timestamp is not None and decision_date is not None and timestamp.date() != decision_date:
            errors.append("candidate signal_snapshot log date mismatch")
            usable = False
        if timestamp is not None and timestamp.time() != TRADING_LOG_TIME:
            errors.append("candidate signal_snapshot log timestamp must equal 09:35")
            usable = False
        if decision_date is None:
            continue
        if not usable:
            unusable_decisions.add(decision_date)
            continue
        identity = (signal_date, record.get("build"), *(
            record.get(field) for field in sorted(expected_fingerprints)
        ))
        previous = evidence.get(decision_date)
        if previous is not None and previous != identity:
            errors.append("conflicting candidate signal_snapshot session evidence: %s" % decision_date)
            unusable_decisions.add(decision_date)
            continue
        evidence[decision_date] = identity
    for decision_date in unusable_decisions:
        evidence.pop(decision_date, None)
    return {decision_date: identity[0] for decision_date, identity in evidence.items()}


def _validate_relative_common(record, errors):
    observation_id = _text(record, "relative_observation_id", "relative observation id", errors)
    if observation_id is not None and not observation_id.startswith("RELATIVE:"):
        errors.append("invalid relative observation id: %s" % observation_id)
    if record.get("build") != CANDIDATE_BUILD:
        errors.append("relative build mismatch: %r" % record.get("build"))
    if record.get("relative_observation_fingerprint") != RELATIVE_OBSERVATION_FINGERPRINT:
        errors.append("relative fingerprint mismatch: %r" % record.get("relative_observation_fingerprint"))
    if not _valid_text(record.get("observation_kind")):
        errors.append("invalid relative observation_kind: %r" % record.get("observation_kind"))
    elif record.get("observation_kind") != "RELATIVE_RESONANCE":
        errors.append("invalid relative observation_kind: %r" % record.get("observation_kind"))
    return observation_id


def _validate_relative_registration(record, session_evidence, session_calendar, errors):
    observation_id = _validate_relative_common(record, errors)
    if record.get("branch") not in BRANCHES:
        errors.append("invalid branch: %r" % record.get("branch"))
    if record.get("direction") not in DIRECTIONS:
        errors.append("invalid candidate direction: %r" % record.get("direction"))
    for field, expected in (
            ("parameter_fingerprint", PARAMETER_FINGERPRINT),
            ("pool_fingerprint", POOL_FINGERPRINT),
            ("event_logic_fingerprint", FORMAL_EVENT_FINGERPRINT),
            ("relative_observation_fingerprint", RELATIVE_OBSERVATION_FINGERPRINT)):
        if record.get(field) != expected:
            errors.append("registration %s mismatch: %r" % (field, record.get(field)))
    _text(record, "code", "candidate code", errors)
    signal_date = _is_training_date(record.get("signal_date"), "signal", errors)
    registration_timestamp = _validate_registration_log_timestamp(
        record, "relative registration", signal_date, errors,
    )
    expires_date = _is_training_date(record.get("expires_date"), "expires", errors)
    if (signal_date is not None and expires_date is not None
            and expires_date < signal_date):
        errors.append("expired candidate: %s" % observation_id)
    _validate_support_contract(
        record, "candidate", signal_date, session_evidence, session_calendar, errors,
    )
    if (registration_timestamp is not None and signal_date is not None
            and session_evidence.get(registration_timestamp.date()) != signal_date):
        errors.append("relative registration session evidence mismatch")
    if registration_timestamp is not None and expires_date is not None:
        decision_date = registration_timestamp.date()
        if expires_date not in session_calendar:
            errors.append("relative registration expires date absent from manifest")
        if expires_date != decision_date:
            errors.append("relative registration expires_date must equal decision session")
    event_close = _finite_number(record.get("event_close"))
    if event_close is None or event_close <= 0:
        errors.append("invalid candidate event_close")
    expected_id = _expected_relative_observation_id(record)
    if (observation_id is not None and expected_id is not None
            and observation_id != expected_id):
        errors.append("relative observation id digest mismatch: %s" % observation_id)
    return observation_id


def _validate_relative_outcome_shape(record, errors):
    observation_id = _validate_relative_common(record, errors)
    resonance_id = _text(record, "resonance_id", "relative outcome resonance id", errors)
    if observation_id is not None and resonance_id != observation_id:
        errors.append("relative outcome resonance_id mismatch: %s" % observation_id)
    horizon = record.get("horizon")
    if (isinstance(horizon, bool) or not isinstance(horizon, int)
            or horizon not in HORIZONS):
        errors.append("invalid relative horizon: %r" % horizon)
        horizon = None
    _text(record, "code", "relative outcome code", errors)
    if record.get("branch") not in BRANCHES:
        errors.append("invalid relative outcome branch: %r" % record.get("branch"))
    if record.get("direction") not in DIRECTIONS:
        errors.append("invalid relative outcome direction: %r" % record.get("direction"))
    _is_training_date(record.get("event_date"), "relative outcome event", errors)
    supporters = record.get("supporters")
    if (not isinstance(supporters, (list, tuple)) or not supporters
            or any(not _valid_text(item) for item in supporters)
            or len(set(supporters)) != len(supporters)):
        errors.append("invalid relative outcome supporters")
    outcome = record.get("outcome")
    if not isinstance(outcome, dict):
        errors.append("invalid relative outcome payload")
        _strict_log_timestamp(record, "relative outcome", errors)
    else:
        closing_date = _is_training_date(
            outcome.get("closing_date"), "relative outcome closing", errors,
        )
        if outcome.get("status") == "RECORDED":
            _validate_outcome_log_timestamp(record, "relative outcome", closing_date, errors)
        else:
            _strict_log_timestamp(record, "relative outcome", errors)
    if isinstance(outcome, dict) and outcome.get("status") == "RECORDED":
        closing_price = _finite_number(outcome.get("closing_price"))
        if closing_price is None or closing_price <= 0:
            errors.append("invalid relative outcome closing_price")
        if _finite_number(outcome.get("return")) is None:
            errors.append("invalid relative outcome return")
        if _finite_number(outcome.get("direction_adjusted_return")) is None:
            errors.append("invalid relative outcome direction_adjusted_return")
    return observation_id, horizon


def _same_supporters(left, right):
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return tuple(left or ()) == tuple(right or ())


def _validate_relative_identity(registration, outcome, errors):
    observation_id = registration.get("relative_observation_id")
    for field in ("observation_kind", "code", "direction", "branch"):
        if outcome.get(field) != registration.get(field):
            errors.append("relative outcome %s mismatch: %s" % (field, observation_id))
    if str(outcome.get("event_date"))[:10] != str(registration.get("signal_date"))[:10]:
        errors.append("relative outcome event_date mismatch: %s" % observation_id)
    if not _same_supporters(outcome.get("supporters"), registration.get("supporters")):
        errors.append("relative outcome supporters mismatch: %s" % observation_id)
    for field in ("supporter_event_dates", "hard_or_relative_source_by_indicator"):
        if field in outcome and outcome.get(field) != registration.get(field):
            errors.append("relative outcome %s mismatch: %s" % (field, observation_id))
    if outcome.get("resonance_id") != observation_id:
        errors.append("relative outcome resonance_id mismatch: %s" % observation_id)
    payload = outcome.get("outcome")
    if isinstance(payload, dict) and payload.get("status") == "RECORDED":
        raw = _finite_number(payload.get("return"))
        adjusted = _finite_number(payload.get("direction_adjusted_return"))
        close = _finite_number(payload.get("closing_price"))
        event_close = _finite_number(registration.get("event_close"))
        expected_raw = (close / event_close - 1.0 if close is not None and event_close is not None
                        and close > 0 and event_close > 0 else None)
        if raw is not None and expected_raw is not None and not math.isclose(
                raw, expected_raw, rel_tol=0.0, abs_tol=1e-12):
            errors.append("relative outcome return mismatch: %s" % observation_id)
        expected = (expected_raw if registration.get("direction") == "BUY_TURN"
                    else (-expected_raw if expected_raw is not None else None))
        if adjusted is not None and expected is not None and not math.isclose(
                adjusted, expected, rel_tol=0.0, abs_tol=1e-12):
            errors.append("relative outcome direction_adjusted_return mismatch: %s" % observation_id)


def _validate_relative_recorded_closing_order(registrations, outcomes, errors):
    for observation_id, registration in registrations.items():
        records = [outcomes.get((observation_id, horizon)) for horizon in HORIZONS]
        if not all(record is not None for record in records):
            continue
        payloads = [record.get("outcome") for record in records]
        if not all(type(payload) is dict and payload.get("status") == "RECORDED"
                   for payload in payloads):
            continue
        try:
            signal_date = _calendar_date(registration.get("signal_date"))
            closing_dates = [_calendar_date(payload["closing_date"]) for payload in payloads]
        except (KeyError, ValueError):
            continue
        if (signal_date is not None and all(value is not None for value in closing_dates)
                and not (signal_date < closing_dates[0] < closing_dates[1] < closing_dates[2])):
            errors.append("relative RECORDED closing dates are not strictly ordered: %s" % observation_id)
        registration_timestamp = _parse_strict_log_timestamp(registration)
        outcome_timestamps = [_parse_strict_log_timestamp(record) for record in records]
        if (registration_timestamp is not None and all(value is not None for value in outcome_timestamps)
                and not (registration_timestamp < outcome_timestamps[0] < outcome_timestamps[1]
                         < outcome_timestamps[2])):
            errors.append("relative outcome log timestamps are not strictly increasing: %s" % observation_id)


def _validate_filled_orders(records, role, errors):
    for record in records:
        if not (record.get("event") == "order_transition" and record.get("outcome") == "FILLED"):
            continue
        timestamp = _strict_log_timestamp(record, "filled order", errors)
        if timestamp is not None and not (TRAIN_START <= timestamp.date() <= TRAIN_END):
            errors.append("invalid filled order log timestamp in %s" % role)
        if timestamp is not None and timestamp.time() != TRADING_LOG_TIME:
            errors.append("filled order log timestamp outside 09:35 in %s" % role)
        if (record.get("side") not in ("BUY", "SELL")
                or not _valid_text(record.get("code"))):
            errors.append("invalid filled order identity in %s" % role)
        for field in ("before_amount", "after_amount"):
            value = record.get(field)
            if type(value) is not int or value < 0:
                errors.append("invalid filled order %s in %s" % (field, role))


def _valid_baseline_registration(record):
    if (not _valid_text(record.get("resonance_id"))
            or not _valid_text(record.get("code"))
            or record.get("direction") not in DIRECTIONS):
        return False
    try:
        signal_date = _calendar_date(record.get("signal_date"))
    except ValueError:
        return False
    timestamp = _parse_strict_log_timestamp(record)
    return (signal_date is not None and TRAIN_START <= signal_date <= TRAIN_END
            and timestamp is not None and timestamp.date() > signal_date
            and timestamp.time() >= TRADING_LOG_TIME)


def _valid_baseline_recorded_horizon(registration, outcome_record, horizon, session_calendar=()):
    if outcome_record is None or outcome_record.get("code") != registration.get("code"):
        return False
    if ("direction" in outcome_record
            and outcome_record.get("direction") != registration.get("direction")):
        return False
    payload = outcome_record.get("outcome")
    if type(payload) is not dict or payload.get("status") != "RECORDED":
        return False
    if _finite_number(payload.get("return")) is None:
        return False
    try:
        signal_date = _calendar_date(registration.get("signal_date"))
        event_date = _calendar_date(outcome_record.get("event_date"))
        closing_date = _calendar_date(payload.get("closing_date"))
    except ValueError:
        return False
    registration_timestamp = _parse_strict_log_timestamp(registration)
    outcome_timestamp = _parse_strict_log_timestamp(outcome_record)
    return (signal_date is not None and event_date == signal_date
            and closing_date is not None and TRAIN_START <= closing_date <= TRAIN_END
            and closing_date == _expected_outcome_session(signal_date, horizon, session_calendar)
            and closing_date > signal_date and registration_timestamp is not None
            and outcome_timestamp is not None and outcome_timestamp.date() == closing_date
            and outcome_timestamp.time() >= AFTER_CLOSE_LOG_TIME
            and outcome_timestamp > registration_timestamp)


def _validate_baseline(records, errors, initialization_valid, session_calendar):
    registrations = {}
    registration_replicas = {}
    outcomes = {}
    invalid_registration_ids = set()
    invalid_horizon_ids = set()
    for record in records:
        if (record.get("_record_namespace") != "formal"
                or not _is_formal_registration_record(record)):
            continue
        resonance_id = _text(record, "resonance_id", "formal resonance id", errors)
        _text(record, "code", "formal code", errors)
        if record.get("direction") not in DIRECTIONS:
            errors.append("invalid formal direction: %r" % record.get("direction"))
        signal_date = _is_training_date(record.get("signal_date"), "formal signal", errors)
        _validate_registration_log_timestamp(record, "formal registration", signal_date, errors)
        if resonance_id is not None:
            registration_replicas.setdefault(resonance_id, []).append(record)
    for resonance_id, replicas in registration_replicas.items():
        public_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in replicas
        ]
        if any(public_record != public_records[0] for public_record in public_records[1:]):
            errors.append("duplicate formal registration: %s" % resonance_id)
            invalid_registration_ids.add(resonance_id)
            continue
        valid_replicas = [record for record in replicas if _valid_baseline_registration(record)]
        if valid_replicas:
            registrations[resonance_id] = min(valid_replicas, key=_sort_key)
    for record in records:
        if not _is_formal_registration_record(record):
            continue
        resonance_id = record.get("resonance_id")
        if resonance_id in registrations and (
                record.get("_record_namespace") != "formal"
                or not _valid_baseline_registration(record)):
            invalid_registration_ids.add(resonance_id)
    for record in records:
        if record.get("event") != "observation_outcome":
            continue
        resonance_id = record.get("resonance_id")
        if type(resonance_id) is not str or resonance_id not in registrations:
            continue
        horizon = record.get("horizon")
        if (record.get("_record_namespace") != "formal"
                or type(record.get("outcome")) is not dict
                or isinstance(horizon, bool) or not isinstance(horizon, int)
                or horizon not in HORIZONS):
            invalid_horizon_ids.add(resonance_id)
    for record in records:
        if record.get("_record_namespace") != "formal" or record.get("event") != "observation_outcome":
            continue
        resonance_id = _text(record, "resonance_id", "formal outcome resonance id", errors)
        horizon = record.get("horizon")
        if (isinstance(horizon, bool) or not isinstance(horizon, int)
                or horizon not in HORIZONS):
            errors.append("invalid formal horizon: %r" % horizon)
            if resonance_id in registrations:
                invalid_horizon_ids.add(resonance_id)
            continue
        _text(record, "code", "formal outcome code", errors)
        _is_training_date(record.get("event_date"), "formal outcome event", errors)
        outcome = record.get("outcome")
        if not isinstance(outcome, dict):
            errors.append("invalid formal outcome payload")
            _strict_log_timestamp(record, "formal outcome", errors)
        else:
            closing_date = _is_training_date(
                outcome.get("closing_date"), "formal outcome closing", errors,
            )
            _validate_outcome_log_timestamp(record, "formal outcome", closing_date, errors)
        if isinstance(outcome, dict) and outcome.get("status") == "RECORDED":
            if _finite_number(outcome.get("return")) is None:
                errors.append("invalid formal outcome return")
        if resonance_id not in registrations:
            errors.append("orphan formal outcome: %s" % resonance_id)
            continue
        registration = registrations[resonance_id]
        if record.get("code") != registration.get("code"):
            errors.append("formal outcome code mismatch: %s" % resonance_id)
        if str(record.get("event_date"))[:10] != str(registration.get("signal_date"))[:10]:
            errors.append("formal outcome event_date mismatch: %s" % resonance_id)
        if "direction" in record and record.get("direction") != registration.get("direction"):
            errors.append("formal outcome direction mismatch: %s" % resonance_id)
        if horizon == 5 and type(outcome) is dict:
            try:
                closing_date = _calendar_date(outcome.get("closing_date"))
                signal_date = _calendar_date(registration.get("signal_date"))
            except ValueError:
                closing_date = signal_date = None
            if (closing_date is not None and signal_date is not None
                    and closing_date <= signal_date):
                errors.append("formal horizon 5 closing must follow signal date: %s" % resonance_id)
            registration_timestamp = _parse_strict_log_timestamp(registration)
            outcome_timestamp = _parse_strict_log_timestamp(record)
            if (registration_timestamp is not None and outcome_timestamp is not None
                    and outcome_timestamp <= registration_timestamp):
                errors.append("formal horizon 5 log timestamp must follow registration: %s" % resonance_id)
        key = (resonance_id, horizon)
        if key in outcomes:
            errors.append("duplicate formal outcome: %s/%s" % key)
            invalid_horizon_ids.add(resonance_id)
        else:
            outcomes[key] = record
    formal_missing_outcome_count = 0
    for resonance_id in registrations:
        for horizon in HORIZONS:
            record = outcomes.get((resonance_id, horizon))
            payload = record.get("outcome") if record is not None else None
            if record is None:
                errors.append("missing formal horizon %s: %s" % (horizon, resonance_id))
            elif type(payload) is not dict or payload.get("status") != "RECORDED":
                errors.append("formal horizon %s is not RECORDED: %s" % (horizon, resonance_id))
        five_day = outcomes.get((resonance_id, 5))
        payload = five_day.get("outcome") if five_day is not None else None
        comparable = (isinstance(payload, dict) and payload.get("status") == "RECORDED"
                      and _finite_number(payload.get("return")) is not None)
        if not comparable:
            formal_missing_outcome_count += 1
            errors.append("formal comparison incomplete: %s" % resonance_id)
    _validate_outcome_sessions(
        registrations, outcomes, session_calendar, "formal", HORIZONS, errors,
    )
    canonical_registrations = {}
    if initialization_valid:
        for resonance_id, registration in registrations.items():
            if (resonance_id not in invalid_registration_ids
                    and resonance_id not in invalid_horizon_ids
                    and _valid_baseline_registration(registration)
                    and all(_valid_baseline_recorded_horizon(
                        registration, outcomes.get((resonance_id, horizon)),
                        horizon, session_calendar,
                    ) for horizon in HORIZONS)):
                canonical_registrations[resonance_id] = registration
    _validate_filled_orders(records, "baseline", errors)
    return registrations, outcomes, formal_missing_outcome_count, canonical_registrations


def _filter_candidate_records(records):
    selected = []
    ignored = Counter()
    for record in records:
        if _matching_relative_record(record) or (
                record.get("event") == "strategy_initialized"
                and record.get("build") == CANDIDATE_BUILD
                and record.get("relative_observation_fingerprint") == RELATIVE_OBSERVATION_FINGERPRINT):
            selected.append(record)
        elif _is_relative_record(record):
            ignored["relative_build_mismatch" if record.get("build") != CANDIDATE_BUILD
                    else "relative_fingerprint_mismatch"] += 1
        elif record.get("event") == "observation_outcome":
            ignored["formal_observation_outcome"] += 1
        elif record.get("event") == "strategy_initialized":
            ignored["initialization_mismatch"] += 1
    return selected, dict(sorted(ignored.items()))


def lower_quartile(values):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * 0.25
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_returns(values, errors=None, label="return"):
    values = tuple(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "hit_rate": None, "q1": None}
    try:
        mean = math.fsum(values) / len(values)
        median = statistics.median(values)
        q1 = lower_quartile(values)
    except (OverflowError, ValueError):
        mean = median = q1 = None
    if not all(value is not None and math.isfinite(value) for value in (mean, median, q1)):
        if errors is not None:
            errors.append("non-finite aggregate: %s" % label)
        mean = median = q1 = None
    return {"count": len(values), "mean": mean, "median": median,
            "hit_rate": sum(value > 0 for value in values) / len(values), "q1": q1}


def extract_filled_order_path(records):
    return tuple((record.get("_log_date"), record.get("side"), record.get("code"),
                  record.get("before_amount"), record.get("after_amount"))
                 for record in records if record.get("event") == "order_transition"
                 and record.get("outcome") == "FILLED")


def _validated_final_asset(records, role, errors):
    frozen_summaries = [record for record in records if record.get("event") == "portfolio_summary"
                        and record.get("closing_date") == TRAIN_END.isoformat()]
    if not frozen_summaries:
        return None
    canonical = None
    timestamps_valid = True
    for record in frozen_summaries:
        timestamp = _strict_log_timestamp(record, "%s frozen portfolio summary" % role, errors)
        if timestamp is None:
            timestamps_valid = False
        elif timestamp.date() != TRAIN_END:
            errors.append("%s frozen portfolio summary log date mismatch" % role)
            timestamps_valid = False
        elif timestamp.time() < AFTER_CLOSE_LOG_TIME:
            errors.append("%s frozen portfolio summary log timestamp before 15:30" % role)
            timestamps_valid = False
        value = _finite_number(record.get("total_value"))
        if value is None:
            errors.append("invalid %s frozen portfolio summary total_value" % role)
            continue
        public = {key: item for key, item in record.items() if not key.startswith("_")}
        if canonical is None:
            canonical = (public, value)
        elif public != canonical[0]:
            errors.append("conflicting %s frozen portfolio summary" % role)
    return canonical[1] if timestamps_valid and canonical is not None and not any(
        error.startswith("conflicting %s frozen portfolio summary" % role)
        or error.startswith("invalid %s frozen portfolio summary" % role)
        for error in errors) else None


def extract_final_asset(records):
    """Backward-compatible inspection helper; audited calls use the function above."""
    return _validated_final_asset(records, "inspection", [])


def _sort_key(record):
    return (str(record.get("_log_timestamp") or "9999-12-31T23:59:59"),
            str(record.get("_source_path") or ""),
            int(record.get("_source_line") or record.get("_input_index") or 0))


def _grouped_summaries(values_by_group, errors=None):
    return {group: {"horizon_%d" % horizon: summarize_returns(
                        values_by_group[group][horizon], errors, "%s/horizon_%d" % (group, horizon))
                    for horizon in HORIZONS} for group in values_by_group}


def _scope_summary(registrations, outcomes, relative):
    direction_counts = {direction: 0 for direction in DIRECTIONS}
    year_counts = {str(year): 0 for year in (2019, 2020, 2021)}
    etf_counts = Counter()
    returns_by_horizon = {horizon: [] for horizon in HORIZONS}
    for resonance_id in sorted(registrations):
        registration = registrations[resonance_id]
        direction = registration.get("direction")
        if direction in direction_counts:
            direction_counts[direction] += 1
        try:
            signal_date = _calendar_date(registration.get("signal_date"))
        except ValueError:
            signal_date = None
        if signal_date is not None and str(signal_date.year) in year_counts:
            year_counts[str(signal_date.year)] += 1
        code = registration.get("code")
        if _valid_text(code):
            etf_counts[code] += 1
        for horizon in HORIZONS:
            record = outcomes.get((resonance_id, horizon))
            payload = record.get("outcome") if record is not None else None
            if type(payload) is not dict or payload.get("status") != "RECORDED":
                continue
            raw_return = _finite_number(payload.get("return"))
            if raw_return is None:
                continue
            value = (_finite_number(payload.get("direction_adjusted_return"))
                     if relative else
                     (raw_return if direction == "BUY_TURN" else -raw_return))
            if value is not None:
                returns_by_horizon[horizon].append(value)
    return {
        "candidate_count": len(registrations),
        "direction_counts": direction_counts,
        "year_counts": year_counts,
        "etf_counts": dict(sorted(etf_counts.items())),
        "horizon_1": summarize_returns(returns_by_horizon[1]),
        "horizon_3": summarize_returns(returns_by_horizon[3]),
        "horizon_5": summarize_returns(returns_by_horizon[5]),
    }


def _formal_five_day_returns(registrations, outcomes):
    values = []
    for (resonance_id, horizon), record in outcomes.items():
        if horizon != 5:
            continue
        outcome = record.get("outcome")
        if type(outcome) is not dict:
            continue
        value = _finite_number(outcome.get("return"))
        if outcome.get("status") == "RECORDED" and value is not None:
            values.append(value if registrations[resonance_id].get("direction") == "BUY_TURN" else -value)
    return values


def _normalized_timeline(records):
    """Preserve source-line order; only canonical source paths break file ties."""
    normalized = []
    for index, record in enumerate(records):
        item = dict(record)
        item.setdefault("_input_index", index)
        normalized.append(item)
    return sorted(normalized, key=_sort_key)


def _has_cross_file_filled_timestamp(records, role, errors):
    sources_by_timestamp = {}
    for record in records:
        if record.get("event") != "order_transition" or record.get("outcome") != "FILLED":
            continue
        source = record.get("_source_path")
        timestamp = record.get("_log_timestamp")
        if source and timestamp:
            sources_by_timestamp.setdefault(timestamp, set()).add(source)
    ambiguous = False
    for timestamp, sources in sorted(sources_by_timestamp.items()):
        if len(sources) > 1:
            ambiguous = True
            errors.append("ambiguous filled order timestamp across files in %s: %s" % (
                role, timestamp,
            ))
    return ambiguous


def _overlap_key(record, label, errors):
    code = record.get("code")
    direction = record.get("direction")
    if not _valid_text(code):
        errors.append("invalid %s code" % label)
        return None
    if direction not in DIRECTIONS:
        errors.append("invalid %s direction" % label)
        return None
    try:
        signal_date = _calendar_date(record.get("signal_date"))
    except ValueError:
        errors.append("invalid %s signal date" % label)
        return None
    if signal_date is None or not TRAIN_START <= signal_date <= TRAIN_END:
        errors.append("invalid %s signal date" % label)
        return None
    return code, direction, signal_date.isoformat()


def _relative_identity_is_scalar(record):
    supporters = record.get("supporters")
    return (
        _valid_text(record.get("code"))
        and _valid_text(record.get("direction")) and record.get("direction") in DIRECTIONS
        and _valid_text(record.get("branch")) and record.get("branch") in BRANCHES
        and isinstance(supporters, (list, tuple))
        and all(_valid_text(item) for item in supporters)
    )


def _audit_outcome_closing_date(record, errors):
    if record.get("event") != "observation_outcome":
        return
    payload = record.get("outcome")
    if type(payload) is not dict:
        record["_outcome_payload_valid"] = False
        errors.append("invalid outcome payload")
        return
    record["_outcome_payload_valid"] = True
    _is_training_date(payload.get("closing_date"), "outcome closing", errors)


def _audit_log_timestamp_training_window(record, label, errors):
    timestamp = _strict_log_timestamp(record, label, errors)
    if timestamp is not None and not TRAIN_START <= timestamp.date() <= TRAIN_END:
        errors.append("%s log timestamp outside 2019-2021: %s" % (label, timestamp.date()))


def _has_relative_markers(record):
    resonance_id = record.get("resonance_id")
    return (record.get("relative_observation_id") is not None
            or record.get("observation_kind") == "RELATIVE_RESONANCE"
            or (_valid_text(resonance_id) and resonance_id.startswith("RELATIVE:")))


def _audit_candidate_formal_observations(records, errors):
    for record in records:
        if record.get("event") == "resonance_decision":
            _audit_log_timestamp_training_window(record, "candidate formal decision", errors)
            _is_training_date(
                record.get("decision_date"), "candidate formal decision", errors,
            )
            _is_training_date(record.get("signal_date"), "candidate formal signal", errors)
            expires_date = record.get("expires_date")
            if _is_formal_registration_record(record) or expires_date not in (None, ""):
                _is_training_date(expires_date, "candidate formal expires", errors)
        elif (record.get("event") == "observation_outcome"
              and not _has_relative_markers(record)):
            _is_training_date(
                record.get("event_date"), "candidate formal outcome event", errors,
            )
            _audit_log_timestamp_training_window(record, "candidate formal outcome", errors)


def analyze_records(candidate_records, baseline_records, session_manifest):
    """Validate immutable log contracts and return a deterministic report."""
    session_manifest = _require_validated_session_manifest(session_manifest)
    session_calendar = session_manifest.sessions
    candidate_records = _normalized_timeline(candidate_records)
    baseline_records = _normalized_timeline(baseline_records)
    errors = []
    for record in candidate_records + baseline_records:
        if record.get("_parse_error"):
            source = record.get("_source_path")
            line = record.get("_source_line")
            location = "%s:%s" % (source, line) if source and line else "input record"
            errors.append("parse error at %s: %s" % (location, record["_parse_error"]))
        _audit_outcome_closing_date(record, errors)
        record["_record_namespace"] = _classify_record_namespace(record, errors)
    _audit_candidate_formal_observations(candidate_records, errors)
    candidate_order_ambiguous = _has_cross_file_filled_timestamp(
        candidate_records, "candidate", errors,
    )
    baseline_order_ambiguous = _has_cross_file_filled_timestamp(
        baseline_records, "baseline", errors,
    )
    candidate_initialization_valid = _validate_candidate_initializations(candidate_records, errors)
    baseline_initialization_valid = _validate_baseline_initializations(baseline_records, errors)
    candidate_session_calendar = _validated_session_calendar(
        candidate_records, "candidate", candidate_initialization_valid, errors,
    )
    baseline_session_calendar = _validated_session_calendar(
        baseline_records, "baseline", baseline_initialization_valid, errors,
    )
    if candidate_session_calendar != session_calendar:
        errors.append("candidate portfolio_summary sessions differ from manifest")
    if baseline_session_calendar != session_calendar:
        errors.append("baseline portfolio_summary sessions differ from manifest")
    _validate_session_evidence(
        candidate_records, "candidate", candidate_session_calendar, session_calendar, errors,
    )
    _validate_session_evidence(
        baseline_records, "baseline", baseline_session_calendar, session_calendar, errors,
    )
    _validate_baseline_observation_namespaces(baseline_records, errors)
    candidate_session_evidence = _candidate_session_evidence(candidate_records, errors)
    _validate_filled_orders(candidate_records, "candidate", errors)
    for record in candidate_records:
        if record.get("event") == "relative_resonance_observation":
            _validate_relative_registration(
                record, candidate_session_evidence, session_calendar, errors,
            )
        elif (record.get("event") == "observation_outcome" and (
                record.get("relative_observation_id") is not None
                or record.get("observation_kind") == "RELATIVE_RESONANCE"
                or (isinstance(record.get("resonance_id"), str)
                    and record.get("resonance_id").startswith("RELATIVE:")))):
            _validate_relative_outcome_shape(record, errors)
    (formal_registrations, formal_outcomes, formal_missing_outcome_count,
     canonical_formal_registrations) = _validate_baseline(
         baseline_records, errors, baseline_initialization_valid, session_calendar,
     )
    canonical_formal_outcomes = {
        key: record for key, record in formal_outcomes.items()
        if key[0] in canonical_formal_registrations
    }
    selected, ignored_record_counts = _filter_candidate_records(candidate_records)
    registrations = {}
    candidates = []
    for record in sorted(selected, key=_sort_key):
        if record.get("event") != "relative_resonance_observation":
            continue
        observation_id = record.get("relative_observation_id")
        if not isinstance(observation_id, str) or not _relative_identity_is_scalar(record):
            continue
        if observation_id in registrations:
            errors.append("duplicate relative candidate: %s" % observation_id)
            continue
        registrations[observation_id] = record
        candidates.append(record)
    outcomes = {}
    for record in sorted(selected, key=_sort_key):
        if record.get("event") != "observation_outcome":
            continue
        observation_id = record.get("relative_observation_id")
        horizon = record.get("horizon")
        if (not isinstance(observation_id, str) or isinstance(horizon, bool)
                or not isinstance(horizon, int) or horizon not in HORIZONS):
            continue
        if observation_id not in registrations:
            errors.append("orphan relative outcome: %s" % observation_id)
            continue
        _validate_relative_identity(registrations[observation_id], record, errors)
        key = (observation_id, horizon)
        if key in outcomes:
            errors.append("duplicate relative outcome: %s/%s" % key)
            continue
        outcomes[key] = record
    _validate_outcome_sessions(
        registrations, outcomes, session_calendar, "relative", HORIZONS, errors,
    )
    _validate_relative_recorded_closing_order(registrations, outcomes, errors)
    year_counts = {"2019": 0, "2020": 0, "2021": 0}
    direction_counts = {direction: 0 for direction in DIRECTIONS}
    etf_counts = Counter()
    returns_by_horizon = {horizon: [] for horizon in HORIZONS}
    by_branch = {branch: {horizon: [] for horizon in HORIZONS} for branch in BRANCHES}
    by_direction = {direction: {horizon: [] for horizon in HORIZONS} for direction in DIRECTIONS}
    five_day_2021 = []
    positive_by_etf = {}
    missing_outcome_count = 0
    for candidate in candidates:
        observation_id = candidate["relative_observation_id"]
        try:
            signal_date = _calendar_date(candidate.get("signal_date"))
        except ValueError:
            signal_date = None
        if signal_date is not None and str(signal_date.year) in year_counts:
            year_counts[str(signal_date.year)] += 1
        direction = candidate.get("direction")
        if direction in direction_counts:
            direction_counts[direction] += 1
        code = candidate.get("code")
        if isinstance(code, str):
            etf_counts[code] += 1
        for horizon in HORIZONS:
            record = outcomes.get((observation_id, horizon))
            outcome = record.get("outcome") if record is not None else None
            value = _finite_number(outcome.get("direction_adjusted_return")
                                   if isinstance(outcome, dict) else None)
            if (not isinstance(outcome, dict) or outcome.get("status") != "RECORDED"
                    or value is None):
                missing_outcome_count += 1
                continue
            returns_by_horizon[horizon].append(value)
            if candidate.get("branch") in by_branch:
                by_branch[candidate["branch"]][horizon].append(value)
            if direction in by_direction:
                by_direction[direction][horizon].append(value)
            if horizon == 5:
                if signal_date is not None and signal_date.year == 2021:
                    five_day_2021.append(value)
                if value > 0 and isinstance(code, str):
                    previous = positive_by_etf.get(code, 0.0)
                    try:
                        combined = math.fsum((previous, value))
                    except OverflowError:
                        combined = math.inf
                    if not math.isfinite(combined):
                        errors.append("non-finite aggregate: positive contribution")
                    positive_by_etf[code] = combined
    formal_keys = set()
    for record in canonical_formal_registrations.values():
        key = _overlap_key(record, "baseline formal overlap", errors)
        if key is not None:
            formal_keys.add(key)
    relative_keys = set()
    for record in candidates:
        key = _overlap_key(record, "relative overlap", errors)
        if key is not None:
            relative_keys.add(key)
    formal_overlap_count = len(formal_keys & relative_keys)
    try:
        total_positive = math.fsum(positive_by_etf.values())
    except OverflowError:
        total_positive = math.inf
        errors.append("non-finite aggregate: positive contribution")
    if not math.isfinite(total_positive) or any(
            not math.isfinite(value) for value in positive_by_etf.values()):
        errors.append("non-finite aggregate: positive contribution")
        max_positive_contribution = None
    else:
        max_positive_contribution = (max(positive_by_etf.values()) / total_positive
                                     if total_positive > 0 else None)
    candidate_path = extract_filled_order_path(candidate_records)
    baseline_path = extract_filled_order_path(baseline_records)
    candidate_asset = _validated_final_asset(candidate_records, "candidate", errors)
    baseline_asset = _validated_final_asset(baseline_records, "baseline", errors)
    horizon_5 = summarize_returns(returns_by_horizon[5], errors, "horizon_5")
    year_2021 = summarize_returns(five_day_2021, errors, "year_2021_horizon_5")
    formal_horizon_5 = summarize_returns(
        _formal_five_day_returns(
            canonical_formal_registrations, canonical_formal_outcomes,
        ),
        errors, "formal_horizon_5",
    )
    formal_order_path_exact = (not candidate_order_ambiguous and not baseline_order_ambiguous
                               and len(candidate_path) == BASELINE_FILLED_COUNT
                               and len(baseline_path) == BASELINE_FILLED_COUNT
                               and candidate_path == baseline_path)
    final_asset_exact = (candidate_asset is not None and baseline_asset is not None
                         and math.isclose(candidate_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
                         and math.isclose(baseline_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
                         and math.isclose(candidate_asset, baseline_asset, abs_tol=0.01))
    grouped_by_branch = _grouped_summaries(by_branch, errors)
    grouped_by_direction = _grouped_summaries(by_direction, errors)
    branch_registrations = {
        branch: {
            observation_id: record for observation_id, record in registrations.items()
            if record.get("branch") == branch
        }
        for branch in BRANCHES
    }
    scope_summaries = {
        "formal_resonance": _scope_summary(
            canonical_formal_registrations, canonical_formal_outcomes, False,
        ),
        "relative_total": _scope_summary(registrations, outcomes, True),
        "HARD_BOLL_SOFT_OSC": _scope_summary(
            branch_registrations["HARD_BOLL_SOFT_OSC"], outcomes, True,
        ),
        "SOFT_ALL_THREE": _scope_summary(
            branch_registrations["SOFT_ALL_THREE"], outcomes, True,
        ),
    }
    horizon_1 = summarize_returns(returns_by_horizon[1], errors, "horizon_1")
    horizon_3 = summarize_returns(returns_by_horizon[3], errors, "horizon_3")
    errors = sorted(set(errors))
    data_quality_complete = not errors and formal_overlap_count == 0 and missing_outcome_count == 0
    gates = {
        "candidate_count_at_least_30": len(candidates) >= 30,
        "each_training_year_at_least_5": all(year_counts[str(year)] >= 5 for year in (2019, 2020, 2021)),
        "horizon_5_median_positive": horizon_5["median"] is not None and horizon_5["median"] > 0,
        "horizon_5_hit_rate_above_half": horizon_5["hit_rate"] is not None and horizon_5["hit_rate"] > 0.5,
        "year_2021_median_nonnegative": year_2021["median"] is not None and year_2021["median"] >= 0,
        "horizon_5_q1_not_worse_than_formal": (horizon_5["q1"] is not None
                                                  and formal_horizon_5["q1"] is not None
                                                  and horizon_5["q1"] >= formal_horizon_5["q1"]),
        "single_etf_positive_contribution_at_most_half": (max_positive_contribution is not None
                                                             and max_positive_contribution <= 0.5),
        "formal_order_path_exact": formal_order_path_exact,
        "final_asset_exact": final_asset_exact,
        "data_quality_complete": data_quality_complete,
    }
    return {
        "session_calendar": dict(
            session_manifest.metadata._asdict(), sha256=session_manifest.sha256,
        ),
        "data_quality": {"errors": errors, "relative_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
                         "formal_overlap_count": formal_overlap_count,
                         "missing_outcome_count": missing_outcome_count,
                         "formal_missing_outcome_count": formal_missing_outcome_count,
                         "ignored_record_counts": ignored_record_counts},
        "metrics": {"candidate_count": len(candidates), "year_counts": year_counts,
                    "direction_counts": direction_counts, "etf_counts": dict(sorted(etf_counts.items())),
                    "by_branch": grouped_by_branch,
                    "by_direction": grouped_by_direction,
                    "scope_summaries": scope_summaries,
                    "horizon_1": horizon_1,
                    "horizon_3": horizon_3,
                    "horizon_5": horizon_5, "year_2021_horizon_5": year_2021,
                    "formal_horizon_5": formal_horizon_5,
                    "max_positive_contribution_by_etf": max_positive_contribution,
                    "filled_path_count": len(candidate_path), "final_asset": candidate_asset},
        "gates": gates, "continue_candidate": all(gates.values()),
    }


def _normalized_input_paths(paths):
    normalized = []
    for path_value in paths:
        path = pathlib.Path(path_value).expanduser().resolve(strict=False)
        if not path.is_file():
            raise ValueError("input log must be a file: %s" % path)
        normalized.append(path)
    return normalized


def _output_conflicts_with_input(output_path, input_paths):
    for input_path in input_paths:
        if _paths_alias(output_path, input_path):
            return True
    return False


def _input_paths_have_alias(paths):
    return any(
        _paths_alias(left_path, right_path)
        for index, left_path in enumerate(paths)
        for right_path in paths[index + 1:]
    )


def _paths_alias(left_path, right_path):
    if left_path == right_path:
        return True
    try:
        return os.path.samefile(left_path, right_path)
    except OSError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-log", action="append")
    parser.add_argument("--baseline-log", action="append")
    parser.add_argument("--session-calendar")
    parser.add_argument("--session-calendar-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if not args.session_calendar:
            raise ValueError("session calendar is required")
        if not args.session_calendar_sha256:
            raise ValueError("session calendar sha256 is required")
        manifest_path = pathlib.Path(args.session_calendar).expanduser().resolve(strict=False)
        if not manifest_path.is_file():
            raise ValueError("session calendar manifest must be a file: %s" % manifest_path)
        session_manifest = validate_session_calendar_manifest(
            read_session_calendar_manifest_bytes(manifest_path),
            args.session_calendar_sha256,
        )
        if not args.candidate_log:
            raise ValueError("candidate log is required")
        if not args.baseline_log:
            raise ValueError("baseline log is required")
        if not args.output:
            raise ValueError("output is required")
        candidate_paths = _normalized_input_paths(args.candidate_log)
        baseline_paths = _normalized_input_paths(args.baseline_log)
        if any(_paths_alias(manifest_path, path) for path in candidate_paths + baseline_paths):
            raise ValueError("session calendar must not match an input log")
        if _input_paths_have_alias(candidate_paths):
            raise ValueError("candidate log paths must be distinct")
        if _input_paths_have_alias(baseline_paths):
            raise ValueError("baseline log paths must be distinct")
        if any(_paths_alias(candidate_path, baseline_path)
               for candidate_path in candidate_paths for baseline_path in baseline_paths):
            raise ValueError("candidate and baseline logs must be distinct")
        output_path = pathlib.Path(args.output).expanduser().resolve(strict=False)
        if _output_conflicts_with_input(
                output_path, candidate_paths + baseline_paths + [manifest_path]):
            raise ValueError("output path must not match an input log")
        candidate_records = load_log_records(candidate_paths)
        baseline_records = load_log_records(baseline_paths)
    except (OSError, ValueError) as exc:
        print("input error: %s" % exc, file=sys.stderr)
        return 2
    report = analyze_records(candidate_records, baseline_records, session_manifest)
    temporary_path = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_path.parent,
                prefix=".%s." % output_path.name, suffix=".tmp", delete=False) as stream:
            temporary_path = pathlib.Path(stream.name)
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except (OSError, UnicodeError):
                pass
        print("output error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
