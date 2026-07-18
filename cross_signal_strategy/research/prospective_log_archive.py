# -*- coding: utf-8 -*-
"""Archive future PTrade logs without opening a strategy experiment.

The archive is deliberately limited to release identity, timestamps, session
boundaries, and event counts.  It does not calculate prices, returns, signal
quality, or any market statistic.  Raw bytes are retained under their SHA256
digest so a later pre-registered study can prove which evidence existed before
its hypothesis was opened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable


PROTOCOL_START = date(2026, 7, 18)
MANIFEST_SCHEMA_VERSION = 1


class LogProtocolError(ValueError):
    """Raised when an input violates the frozen prospective protocol."""


class LogIdentityError(LogProtocolError):
    """Raised when a log cannot be bound to the frozen formal release."""


@dataclass(frozen=True)
class LogFileRecord:
    source_name: str
    archive_name: str
    sha256: str
    size_bytes: int
    first_timestamp: str
    last_timestamp: str
    execution_dates: tuple[str, ...]
    signal_dates: tuple[str, ...]
    release_identity_count: int
    execution_count: int
    buy_order_count: int
    sell_order_count: int
    buy_fill_count: int
    sell_fill_count: int
    iopv_observation_count: int
    halt_recovery_count: int


_TIMESTAMP_RE = re.compile(r"(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+-")
_RELEASE_RE = re.compile(
    r"\[发布指纹\]\s*构建=([^\s]+)\s+业务配置=([0-9a-fA-F]+)\s+状态结构=(\d+)"
)
_EXECUTION_RE = re.compile(
    r"\[(cross-v[^\]]+)\]\s*执行日期=(\d{4}-\d{2}-\d{2})\s+"
    r"信号日期=(\d{4}-\d{2}-\d{2})\s+是否调仓=(?:是|否|True|False)"
)
_BUY_ORDER_RE = re.compile(r"\[买入\]\s+\d{6}\.(?:SS|SZ)\s+买入评分=")
_SELL_ORDER_RE = re.compile(r"\[卖出\]\s+\d{6}\.(?:SS|SZ)\s+原因=")
_BUY_FILL_RE = re.compile(r"\[成交回报\]\s+买入\s+\d{6}\.(?:SS|SZ)\s+")
_SELL_FILL_RE = re.compile(r"\[成交回报\]\s+卖出\s+\d{6}\.(?:SS|SZ)\s+")


def _decode_log(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise LogProtocolError("log encoding must be UTF-8 or GB18030")


def _timestamp_text(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def inspect_log_bytes(
    source_name: str,
    raw: bytes,
    *,
    expected_version: str,
    expected_build: str,
    expected_fingerprint: str,
    protocol_start: date = PROTOCOL_START,
) -> LogFileRecord:
    """Validate one exported log and return non-performance metadata only."""
    if not isinstance(raw, bytes) or not raw:
        raise LogProtocolError("log input must contain raw bytes")
    text = _decode_log(raw)
    timestamps = [_timestamp_text(value) for value in _TIMESTAMP_RE.findall(text)]
    if not timestamps:
        raise LogProtocolError("log contains no platform timestamps")
    timestamp_days = [
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S").date()
        for value in timestamps
    ]
    if min(timestamp_days) < protocol_start:
        raise LogProtocolError(
            "log predates protocol start %s" % protocol_start.isoformat()
        )

    releases = _RELEASE_RE.findall(text)
    if not releases:
        raise LogIdentityError("release identity is missing")
    for build, fingerprint, _state_schema in releases:
        if build != expected_build:
            raise LogIdentityError(
                "deployment build mismatch: expected=%s actual=%s"
                % (expected_build, build)
            )
        if fingerprint.lower() != expected_fingerprint.lower():
            raise LogIdentityError(
                "business fingerprint mismatch: expected=%s actual=%s"
                % (expected_fingerprint, fingerprint)
            )

    executions = _EXECUTION_RE.findall(text)
    if not executions:
        raise LogIdentityError("release identity has no executable strategy session")
    versions = {version for version, _execution, _signal in executions}
    if versions != {expected_version}:
        raise LogIdentityError(
            "strategy version mismatch: expected=%s actual=%s"
            % (expected_version, ",".join(sorted(versions)))
        )

    digest = sha256(raw).hexdigest()
    execution_dates = tuple(sorted({execution for _, execution, _ in executions}))
    signal_dates = tuple(sorted({signal for _, _, signal in executions}))
    return LogFileRecord(
        source_name=Path(str(source_name)).name,
        archive_name=digest + ".log",
        sha256=digest,
        size_bytes=len(raw),
        first_timestamp=min(timestamps),
        last_timestamp=max(timestamps),
        execution_dates=execution_dates,
        signal_dates=signal_dates,
        release_identity_count=len(releases),
        execution_count=len(executions),
        buy_order_count=len(_BUY_ORDER_RE.findall(text)),
        sell_order_count=len(_SELL_ORDER_RE.findall(text)),
        buy_fill_count=len(_BUY_FILL_RE.findall(text)),
        sell_fill_count=len(_SELL_FILL_RE.findall(text)),
        iopv_observation_count=text.count("[IOPV观察]"),
        halt_recovery_count=text.count("[复牌补偿]"),
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _record_payload(record: LogFileRecord) -> dict:
    payload = asdict(record)
    payload["execution_dates"] = list(record.execution_dates)
    payload["signal_dates"] = list(record.signal_dates)
    return payload


def _manifest(
    records: Iterable[LogFileRecord],
    *,
    expected_version: str,
    expected_build: str,
    expected_fingerprint: str,
    protocol_start: date,
) -> dict:
    ordered = sorted(records, key=lambda item: item.sha256)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_start": protocol_start.isoformat(),
        "release": {
            "strategy_version": expected_version,
            "deployment_build": expected_build,
            "business_fingerprint": expected_fingerprint,
        },
        "files": [_record_payload(record) for record in ordered],
    }


def _load_existing_records(
    manifest_path: Path,
    raw_root: Path,
    *,
    expected_version: str,
    expected_build: str,
    expected_fingerprint: str,
    protocol_start: date,
) -> dict[str, LogFileRecord]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_release = {
        "strategy_version": expected_version,
        "deployment_build": expected_build,
        "business_fingerprint": expected_fingerprint,
    }
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise LogProtocolError("existing manifest schema does not match")
    if payload.get("protocol_start") != protocol_start.isoformat():
        raise LogProtocolError("existing manifest protocol start does not match")
    if payload.get("release") != expected_release:
        raise LogIdentityError("existing manifest release identity does not match")

    records = {}
    for item in payload.get("files", []):
        restored = dict(item)
        restored["execution_dates"] = tuple(restored.get("execution_dates", ()))
        restored["signal_dates"] = tuple(restored.get("signal_dates", ()))
        record = LogFileRecord(**restored)
        expected_name = record.sha256 + ".log"
        if record.archive_name != expected_name:
            raise LogProtocolError("existing manifest has a noncanonical archive name")
        raw_path = raw_root / expected_name
        if not raw_path.is_file():
            raise LogProtocolError(
                "existing raw evidence is missing: %s" % expected_name
            )
        raw = raw_path.read_bytes()
        if sha256(raw).hexdigest() != record.sha256 or len(raw) != record.size_bytes:
            raise LogProtocolError(
                "existing raw evidence failed digest verification: %s" % expected_name
            )
        records[record.sha256] = record
    return records


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    if path.is_file() and path.read_bytes() == raw:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def archive_log_bundle(
    log_paths: Iterable[Path],
    archive_root: Path,
    *,
    expected_version: str,
    expected_build: str,
    expected_fingerprint: str,
    protocol_start: date = PROTOCOL_START,
) -> dict:
    """Copy validated logs into an append-only, content-addressed archive."""
    root = Path(archive_root).resolve()
    sources = [Path(path).resolve() for path in log_paths]
    if not sources:
        raise LogProtocolError("at least one source log is required")
    for source in sources:
        if _is_relative_to(source, root):
            raise LogProtocolError("source logs must remain outside the archive root")
        if not source.is_file():
            raise LogProtocolError("source log does not exist: %s" % source)

    inspected = []
    raw_by_digest = {}
    for source in sources:
        raw = source.read_bytes()
        record = inspect_log_bytes(
            source.name,
            raw,
            expected_version=expected_version,
            expected_build=expected_build,
            expected_fingerprint=expected_fingerprint,
            protocol_start=protocol_start,
        )
        inspected.append(record)
        raw_by_digest[record.sha256] = raw

    raw_root = root / "raw"
    manifest_path = root / "manifest.json"
    existing = _load_existing_records(
        manifest_path,
        raw_root,
        expected_version=expected_version,
        expected_build=expected_build,
        expected_fingerprint=expected_fingerprint,
        protocol_start=protocol_start,
    )
    records = dict(existing)
    for record in inspected:
        records.setdefault(record.sha256, record)

    raw_root.mkdir(parents=True, exist_ok=True)
    for digest, raw in raw_by_digest.items():
        destination = raw_root / (digest + ".log")
        if destination.exists():
            if destination.read_bytes() != raw:
                raise LogProtocolError("archived digest path contains different bytes")
            continue
        _atomic_write_bytes(destination, raw)

    manifest = _manifest(
        records.values(),
        expected_version=expected_version,
        expected_build=expected_build,
        expected_fingerprint=expected_fingerprint,
        protocol_start=protocol_start,
    )
    encoded = (json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")
    _atomic_write_bytes(manifest_path, encoded)
    return manifest
