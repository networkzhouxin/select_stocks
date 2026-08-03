"""Deterministic canonical JSON encoding for hashed protocol records."""

from collections.abc import Mapping
import json
import unicodedata

from .numeric_v1 import Q18, format_q18


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented by canonical JSON v1."""


def _normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalJsonError(
            "canonical JSON strings must contain only Unicode scalar values"
        ) from exc
    return normalized


def _normalize(value: object) -> object:
    if type(value) is str:
        return _normalize_string(value)
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is Q18:
        return format_q18(value)
    if type(value) in (list, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, object]] = []
        normalized_keys: set[str] = set()
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJsonError(
                    "canonical JSON object keys must be exact strings"
                )
            normalized_key = _normalize_string(key)
            if normalized_key in normalized_keys:
                raise CanonicalJsonError(
                    "canonical JSON object keys collide after NFC normalization"
                )
            normalized_keys.add(normalized_key)
            normalized_items.append((normalized_key, _normalize(item)))
        return dict(sorted(normalized_items))
    raise CanonicalJsonError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def _normalized_snapshot(payload: object) -> object:
    try:
        return _normalize(payload)
    except CanonicalJsonError:
        raise
    except (RecursionError, UnicodeEncodeError) as exc:
        raise CanonicalJsonError(
            "canonical JSON payload cannot be normalized"
        ) from exc


def _encode_snapshot(normalized: object) -> bytes:
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (ValueError, RecursionError, UnicodeEncodeError) as exc:
        raise CanonicalJsonError(
            "canonical JSON payload cannot be encoded"
        ) from exc


def canonical_json_bytes(payload: object) -> bytes:
    """Encode an approved value as canonical UTF-8 JSON bytes."""

    return _encode_snapshot(_normalized_snapshot(payload))


def canonical_hashed_payload_bytes(payload: object) -> bytes:
    """Encode one normalized snapshot of a versioned top-level mapping."""

    if not isinstance(payload, Mapping):
        raise CanonicalJsonError("hashed payload must be a mapping")
    snapshot = _normalized_snapshot(payload)
    if type(snapshot) is not dict:
        raise CanonicalJsonError("hashed payload must normalize to an object")
    schema_version = snapshot.get("schema_version")
    if type(schema_version) is not str or not schema_version:
        raise CanonicalJsonError(
            "hashed payload requires a nonempty schema_version"
        )
    numeric_protocol_version = snapshot.get("numeric_protocol_version")
    if (
        type(numeric_protocol_version) is not str
        or numeric_protocol_version != "numeric_protocol_v1"
    ):
        raise CanonicalJsonError(
            "hashed payload requires numeric_protocol_version "
            "numeric_protocol_v1"
        )
    return _encode_snapshot(snapshot)


__all__ = (
    "CanonicalJsonError",
    "canonical_hashed_payload_bytes",
    "canonical_json_bytes",
)
