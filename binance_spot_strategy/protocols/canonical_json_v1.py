"""Deterministic canonical JSON encoding for hashed protocol records."""

from collections.abc import Mapping
import json
import unicodedata

from .numeric_v1 import Q18, format_q18


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented by canonical JSON v1."""


def _normalize(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Q18):
        return format_q18(value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, object]] = []
        normalized_keys: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(
                    "canonical JSON object keys must be strings"
                )
            normalized_key = unicodedata.normalize("NFC", key)
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


def canonical_json_bytes(payload: object) -> bytes:
    """Encode an approved value as canonical UTF-8 JSON bytes."""

    normalized = _normalize(payload)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hashed_payload_bytes(payload: object) -> bytes:
    """Encode a versioned top-level mapping intended for hashing."""

    if not isinstance(payload, Mapping):
        raise CanonicalJsonError("hashed payload must be a mapping")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise CanonicalJsonError(
            "hashed payload requires a nonempty schema_version"
        )
    if payload.get("numeric_protocol_version") != "numeric_protocol_v1":
        raise CanonicalJsonError(
            "hashed payload requires numeric_protocol_version "
            "numeric_protocol_v1"
        )
    return canonical_json_bytes(payload)


__all__ = (
    "CanonicalJsonError",
    "canonical_hashed_payload_bytes",
    "canonical_json_bytes",
)
