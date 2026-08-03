"""Strict SHA-256 helpers used by deterministic identity protocols."""

from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from binance_spot_strategy.protocols import canonical_hashed_payload_bytes


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTF8_BOM = b"\xef\xbb\xbf"


def sha256_bytes(data: bytes) -> str:
    """Hash an exact byte sequence as lowercase hexadecimal SHA-256."""

    if not isinstance(data, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return sha256(data).hexdigest()


def sha256_raw_file(path: str | Path) -> str:
    """Hash file bytes without text decoding or newline conversion."""

    return sha256_bytes(Path(path).read_bytes())


def sha256_tracked_text(path: str | Path) -> str:
    """Hash UTF-8 tracked text after Git-style newline normalization."""

    raw = Path(path).read_bytes()
    if raw.startswith(_UTF8_BOM):
        raise ValueError("tracked text must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("tracked text must be valid UTF-8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalized.encode("utf-8"))


def require_sha256(value: object, path: str = "sha256") -> str:
    """Return a digest only if it uses canonical lowercase hex encoding."""

    if not isinstance(value, str):
        raise TypeError(f"{path} must be a lowercase SHA-256 digest")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def hash_canonical_payload(payload: Any) -> str:
    """Hash a payload through canonical JSON v1."""

    return sha256_bytes(canonical_hashed_payload_bytes(payload))


__all__ = (
    "hash_canonical_payload",
    "require_sha256",
    "sha256_bytes",
    "sha256_raw_file",
    "sha256_tracked_text",
)
