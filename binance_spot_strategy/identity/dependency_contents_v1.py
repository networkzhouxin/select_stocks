"""Complete immutable content identity for the frozen CPython runtime."""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import os
import re
from typing import Any, NoReturn
import stat
import sys
import sysconfig
import weakref
import unicodedata

from binance_spot_strategy.protocols import canonical_json_bytes

from .dependency_lock_v1 import (
    load_semantic_dependency_lock,
    semantic_dependency_lock_hash,
    verify_current_runtime,
)
from .digests import hash_canonical_payload, require_sha256, sha256_bytes



class DependencyContentLockError(ValueError):
    """Raised when content-lock evidence is malformed or does not verify."""


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCK_PATH = _PACKAGE_ROOT / "config" / "dependency_contents.lock.json"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "numeric_protocol_version",
        "semantic_dependency_lock_sha256",
        "trees",
    }
)
_TREE_KEYS = frozenset({"owner_kind", "owner_name", "owner_version", "files"})
_FILE_KEYS = frozenset(
    {"logical_path", "artifact_kind", "size_bytes", "sha256"}
)
_ARTIFACT_KINDS = frozenset({"runtime_binary", "source", "extension", "data"})
_DRIVE_RE = re.compile(r"[A-Za-z]:")


def _error(path: str, reason: str) -> NoReturn:
    raise DependencyContentLockError(
        f"invalid dependency content lock at {path}: {reason}"
    )


def _exact_object(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict:
        _error(path, "must be an object")
    assert isinstance(value, dict)
    if not all(type(key) is str for key in value):
        _error(path, "object keys must be exact strings")
    actual = frozenset(value)
    if actual != keys:
        _error(
            path,
            f"keys differ (missing={sorted(keys - actual)}, extra={sorted(actual - keys)})",
        )
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        _error(path, "must be a nonempty exact string")
    assert isinstance(value, str)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DependencyContentLockError(
            f"invalid dependency content lock at {path}: "
            "must contain only Unicode scalar values"
        ) from exc
    if unicodedata.normalize("NFC", value) != value:
        _error(path, "must use NFC normalization")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        _error(path, "must be an exact integer")
    assert isinstance(value, int)
    return value


def _digest(value: object, path: str) -> str:
    try:
        return require_sha256(value, path)
    except (TypeError, ValueError) as exc:
        _error(path, str(exc))


def _logical_path(value: object, path: str) -> str:
    logical = _string(value, path)
    if (
        logical.startswith("/")
        or logical.startswith("\\")
        or "\\" in logical
        or _DRIVE_RE.match(logical) is not None
    ):
        _error(path, "must be a prefix-relative forward-slash path")
    parts = logical.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _error(path, "must not contain empty, dot, or dot-dot segments")
    if any(":" in part for part in parts):
        _error(path, "must not contain a drive or alternate stream")
    return logical


def _path_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


@dataclass(frozen=True, slots=True)
class ContentFileV1:
    logical_path: str
    artifact_kind: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _logical_path(self.logical_path, "ContentFileV1.logical_path")
        artifact_kind = _string(
            self.artifact_kind, "ContentFileV1.artifact_kind"
        )
        if artifact_kind not in _ARTIFACT_KINDS:
            _error("ContentFileV1.artifact_kind", "is not an approved artifact kind")
        if _integer(self.size_bytes, "ContentFileV1.size_bytes") < 0:
            _error("ContentFileV1.size_bytes", "must be nonnegative")
        _digest(self.sha256, "ContentFileV1.sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_path": self.logical_path,
            "artifact_kind": self.artifact_kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ContentTreeV1:
    owner_kind: str
    owner_name: str
    owner_version: str
    files: tuple[ContentFileV1, ...]

    def __post_init__(self) -> None:
        _string(self.owner_kind, "ContentTreeV1.owner_kind")
        _string(self.owner_name, "ContentTreeV1.owner_name")
        _string(self.owner_version, "ContentTreeV1.owner_version")
        if type(self.files) is not tuple:
            _error("ContentTreeV1.files", "must be an exact tuple")
        if not self.files:
            _error("ContentTreeV1.files", "must not be empty")
        if not all(type(item) is ContentFileV1 for item in self.files):
            _error("ContentTreeV1.files", "must contain exact ContentFileV1 values")
        paths = [item.logical_path for item in self.files]
        if paths != sorted(paths):
            _error("ContentTreeV1.files", "must be sorted by logical_path")
        keys = [_path_key(item) for item in paths]
        if len(keys) != len(set(keys)):
            _error("ContentTreeV1.files", "contains duplicate or colliding paths")

    def to_payload(self) -> dict[str, object]:
        return {
            "owner_kind": self.owner_kind,
            "owner_name": self.owner_name,
            "owner_version": self.owner_version,
            "files": [item.to_payload() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class DependencyContentLockV1:
    schema_version: str
    numeric_protocol_version: str
    semantic_dependency_lock_sha256: str
    trees: tuple[ContentTreeV1, ...]

    def __post_init__(self) -> None:
        schema = _string(self.schema_version, "DependencyContentLockV1.schema_version")
        if schema != "dependency_content_lock_v1":
            _error(
                "DependencyContentLockV1.schema_version",
                "must equal dependency_content_lock_v1",
            )
        numeric = _string(
            self.numeric_protocol_version,
            "DependencyContentLockV1.numeric_protocol_version",
        )
        if numeric != "numeric_protocol_v1":
            _error(
                "DependencyContentLockV1.numeric_protocol_version",
                "must equal numeric_protocol_v1",
            )
        _digest(
            self.semantic_dependency_lock_sha256,
            "DependencyContentLockV1.semantic_dependency_lock_sha256",
        )
        if type(self.trees) is not tuple:
            _error("DependencyContentLockV1.trees", "must be an exact tuple")
        if not all(type(tree) is ContentTreeV1 for tree in self.trees):
            _error(
                "DependencyContentLockV1.trees",
                "must contain exact ContentTreeV1 values",
            )
        _validate_tree_contract(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "numeric_protocol_version": self.numeric_protocol_version,
            "semantic_dependency_lock_sha256": self.semantic_dependency_lock_sha256,
            "trees": [tree.to_payload() for tree in self.trees],
        }


def _validate_tree_contract(lock: DependencyContentLockV1) -> None:
    semantic = load_semantic_dependency_lock()
    expected_semantic_hash = semantic_dependency_lock_hash(semantic)
    if lock.semantic_dependency_lock_sha256 != expected_semantic_hash:
        _error(
            "semantic_dependency_lock_sha256",
            "does not bind the canonical semantic dependency lock",
        )
    expected = [
        ("cpython_runtime", "CPython", semantic["python"]["version"]),
        ("cpython_stdlib", "stdlib", semantic["python"]["version"]),
    ]
    expected.extend(
        ("distribution", item["name"], item["version"])
        for item in semantic["distributions"]
    )
    actual = [
        (tree.owner_kind, tree.owner_name, tree.owner_version)
        for tree in lock.trees
    ]
    if actual != expected:
        _error("trees", "must contain the exact ordered owner set")
    identities: set[str] = set()
    for tree in lock.trees:
        for item in tree.files:
            key = _path_key(item.logical_path)
            if key in identities:
                _error("trees", "contains a duplicate or colliding global file identity")
            identities.add(key)


def _file_from_payload(value: object, path: str) -> ContentFileV1:
    item = _exact_object(value, _FILE_KEYS, path)
    try:
        return ContentFileV1(
            logical_path=item["logical_path"],
            artifact_kind=item["artifact_kind"],
            size_bytes=item["size_bytes"],
            sha256=item["sha256"],
        )
    except DependencyContentLockError as exc:
        raise DependencyContentLockError(f"{exc} (payload {path})") from exc


def _tree_from_payload(value: object, path: str) -> ContentTreeV1:
    item = _exact_object(value, _TREE_KEYS, path)
    files = item["files"]
    if type(files) is not list:
        _error(f"{path}.files", "must be an array")
    assert isinstance(files, list)
    return ContentTreeV1(
        owner_kind=item["owner_kind"],
        owner_name=item["owner_name"],
        owner_version=item["owner_version"],
        files=tuple(
            _file_from_payload(file_item, f"{path}.files[{index}]")
            for index, file_item in enumerate(files)
        ),
    )


def _lock_from_payload(value: object) -> DependencyContentLockV1:
    root = _exact_object(value, _TOP_LEVEL_KEYS, "$")
    trees = root["trees"]
    if type(trees) is not list:
        _error("trees", "must be an array")
    assert isinstance(trees, list)
    return DependencyContentLockV1(
        schema_version=root["schema_version"],
        numeric_protocol_version=root["numeric_protocol_version"],
        semantic_dependency_lock_sha256=root["semantic_dependency_lock_sha256"],
        trees=tuple(
            _tree_from_payload(tree, f"trees[{index}]")
            for index, tree in enumerate(trees)
        ),
    )


def _reject_json_number(_: str) -> NoReturn:
    raise DependencyContentLockError(
        "invalid dependency content lock JSON: floats are forbidden"
    )


def _reject_json_constant(_: str) -> NoReturn:
    raise DependencyContentLockError(
        "invalid dependency content lock JSON: nonfinite numbers are forbidden"
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        try:
            key.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DependencyContentLockError(
                "invalid dependency content lock JSON: object key is not Unicode scalar text"
            ) from exc
        if key in result:
            raise DependencyContentLockError(
                "invalid dependency content lock JSON: duplicate object key"
            )
        result[key] = value
    return result


def _reject_nulls(value: object) -> None:
    if value is None:
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: null is forbidden"
        )
    if type(value) is dict:
        assert isinstance(value, dict)
        for item in value.values():
            _reject_nulls(item)
    elif type(value) is list:
        assert isinstance(value, list)
        for item in value:
            _reject_nulls(item)


def load_dependency_content_lock(
    path: str | Path | None = None,
) -> DependencyContentLockV1:
    """Load one strict dependency-content lock without changing it."""

    lock_path = _DEFAULT_LOCK_PATH if path is None else Path(path)
    try:
        raw = lock_path.read_bytes()
    except OSError as exc:
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: unable to read lock"
        ) from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: UTF-8 BOM is forbidden"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: invalid UTF-8"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_constant,
        )
        _reject_nulls(value)
        return _lock_from_payload(value)
    except DependencyContentLockError:
        raise
    except json.JSONDecodeError as exc:
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: malformed document"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise DependencyContentLockError(
            "invalid dependency content lock JSON: parser limit exceeded"
        ) from exc


def dependency_content_lock_hash(lock: DependencyContentLockV1) -> str:
    """Hash an exact validated content-lock value through canonical JSON v1."""

    if type(lock) is not DependencyContentLockV1:
        _error("$", "must be an exact DependencyContentLockV1")
    _validate_tree_contract(lock)
    return hash_canonical_payload(lock.to_payload())


class _OpaqueVerified:
    __slots__ = ("__weakref__",)

    def __new__(cls, *args: object, **kwargs: object):
        raise TypeError("verified content values are issued only by the verifier")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified content values cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("verified content values cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("verified content values cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("verified content values cannot be copied")


class VerifiedContentTreeV1(_OpaqueVerified):
    __slots__ = ("_owner_key", "_tree_hash")

    def __init_subclass__(cls, **kwargs: object) -> NoReturn:
        raise TypeError("verified content values cannot be subclassed")


class VerifiedDependencyContentV1(_OpaqueVerified):
    __slots__ = ("_content_lock_hash", "_trees")

    def __init_subclass__(cls, **kwargs: object) -> NoReturn:
        raise TypeError("verified content values cannot be subclassed")


def _create_receipt_authority():
    registry: weakref.WeakKeyDictionary[object, int] = weakref.WeakKeyDictionary()

    def issue(
        receipt_type: type[VerifiedContentTreeV1] | type[VerifiedDependencyContentV1],
        attributes: tuple[tuple[str, object], ...],
    ) -> VerifiedContentTreeV1 | VerifiedDependencyContentV1:
        if receipt_type not in (VerifiedContentTreeV1, VerifiedDependencyContentV1):
            raise TypeError("unsupported verified receipt type")
        receipt = object.__new__(receipt_type)
        for name, value in attributes:
            object.__setattr__(receipt, name, value)
        registry[receipt] = os.getpid()
        return receipt

    def require_dependency(
        value: object,
    ) -> VerifiedDependencyContentV1:
        if type(value) is not VerifiedDependencyContentV1:
            raise DependencyContentLockError(
                "untrusted verified dependency content receipt"
            )
        try:
            issuer_pid = registry[value]
        except (KeyError, TypeError) as exc:
            raise DependencyContentLockError(
                "untrusted verified dependency content receipt"
            ) from exc
        if issuer_pid != os.getpid():
            raise DependencyContentLockError(
                "foreign-process verified dependency content receipt"
            )
        return value

    return issue, require_dependency


_issue_verified_receipt, _require_verified_dependency_content = (
    _create_receipt_authority()
)
del _create_receipt_authority

@dataclass(frozen=True, slots=True)
class _DistributionInputV1:
    name: str
    version: str
    install_root: Path
    record_path: Path

    def __post_init__(self) -> None:
        _string(self.name, "distribution.name")
        _string(self.version, "distribution.version")
        if not isinstance(self.install_root, Path) or not isinstance(
            self.record_path, Path
        ):
            raise TypeError("distribution paths must be pathlib.Path values")


@dataclass(frozen=True, slots=True)
class _ScanEnvironmentV1:
    prefix: Path
    stdlib_root: Path
    site_packages_roots: tuple[Path, ...]
    implementation: str
    python_version: str
    distributions: tuple[_DistributionInputV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, Path) or not isinstance(
            self.stdlib_root, Path
        ):
            raise TypeError("scan roots must be pathlib.Path values")
        if type(self.site_packages_roots) is not tuple or not all(
            isinstance(item, Path) for item in self.site_packages_roots
        ):
            raise TypeError("site_packages_roots must be an exact tuple of paths")
        _string(self.implementation, "environment.implementation")
        _string(self.python_version, "environment.python_version")
        if type(self.distributions) is not tuple or not all(
            type(item) is _DistributionInputV1 for item in self.distributions
        ):
            raise TypeError("distributions must be an exact input tuple")


@dataclass(frozen=True, slots=True)
class _SnapshotV1:
    logical_path: str
    file_id: tuple[int, int]
    size_bytes: int
    sha256: str
    data: bytes | None = None


_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class _SnapshotSessionV1:
    __slots__ = ("_windows_leases", "_portable_streams", "_captured")

    def __init__(self) -> None:
        self._windows_leases: list[tuple[int, Path]] = []
        self._portable_streams: list[tuple[object, Path]] = []
        self._captured: dict[Path, os.stat_result] = {}

    def open_stream(self, path: Path):
        if sys.platform != "win32":
            stream = path.open("rb", buffering=0)
            self._portable_streams.append((stream, path))
            return stream, False

        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x80000000,
            0x00000001,
            None,
            3,
            0x08000000,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in (None, invalid_handle):
            raise OSError(ctypes.get_last_error(), "unable to lock snapshot file")

        duplicate_handle = kernel32.DuplicateHandle
        duplicate_handle.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        )
        duplicate_handle.restype = ctypes.c_int
        current_process = kernel32.GetCurrentProcess()
        duplicate = ctypes.c_void_p()
        if not duplicate_handle(
            current_process,
            handle,
            current_process,
            ctypes.byref(duplicate),
            0,
            False,
            0x00000002,
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "unable to retain snapshot handle")
        assert duplicate.value is not None
        self._windows_leases.append((int(duplicate.value), path))
        try:
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
            stream = os.fdopen(descriptor, "rb", buffering=0)
        except Exception:
            kernel32.CloseHandle(handle)
            raise
        return stream, True

    def require_unseen(self, path: Path) -> None:
        if path in self._captured:
            raise DependencyContentLockError(
                "dependency content scan failed: multiple distribution owners"
            )


    def capture(self, path: Path, information: os.stat_result) -> None:
        if path in self._captured:
            raise DependencyContentLockError(
                "dependency content scan failed: file hashed more than once"
            )
        self._captured[path] = information

    def validate(self) -> None:
        for path, captured in self._captured.items():
            try:
                current = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise DependencyContentLockError(
                    "dependency content scan failed: controlled file changed after hashing"
                ) from exc
            if any(
                getattr(captured, field) != getattr(current, field)
                for field in _STABLE_STAT_FIELDS
            ):
                logical = path.name
                raise DependencyContentLockError(
                    f"dependency content scan failed: final snapshot drift {logical}"
                )

    def close(self) -> None:
        for stream, _ in reversed(self._portable_streams):
            try:
                stream.close()
            except OSError:
                pass
        self._portable_streams.clear()
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            for handle, _ in reversed(self._windows_leases):
                kernel32.CloseHandle(ctypes.c_void_p(handle))
        self._windows_leases.clear()
        self._captured.clear()


_ACTIVE_SNAPSHOT_SESSION: ContextVar[_SnapshotSessionV1 | None] = ContextVar(
    "dependency_content_snapshot_session", default=None
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _canonical_root(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        information = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DependencyContentLockError(
            f"dependency content scan failed: unavailable {label}"
        ) from exc
    if not stat.S_ISDIR(information.st_mode) or _is_reparse(information):
        raise DependencyContentLockError(
            f"dependency content scan failed: unsafe {label}"
        )
    return resolved


def _is_reparse(information: os.stat_result) -> bool:
    attributes = getattr(information, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _root_identity(path: Path) -> tuple[int, int]:
    try:
        information = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DependencyContentLockError(
            "dependency content scan failed: root identity unavailable"
        ) from exc
    if not stat.S_ISDIR(information.st_mode) or _is_reparse(information):
        raise DependencyContentLockError(
            "dependency content scan failed: root identity is unsafe"
        )
    return information.st_dev, information.st_ino


def _assert_root_identity(path: Path, expected: tuple[int, int]) -> None:
    if _root_identity(path) != expected:
        raise DependencyContentLockError(
            "dependency content scan failed: root identity changed"
        )


def _check_components(path: Path, prefix: Path) -> None:
    try:
        relative = path.relative_to(prefix)
    except ValueError as exc:
        raise DependencyContentLockError(
            "dependency content scan failed: final prefix escape"
        ) from exc
    current = prefix
    for component in relative.parts:
        if component == "..":
            current = current.parent
            if not _is_relative_to(current, prefix):
                raise DependencyContentLockError(
                    "dependency content scan failed: final prefix escape"
                )
            continue
        if component == ".":
            continue
        current = current / component
        try:
            information = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise DependencyContentLockError(
                "dependency content scan failed: controlled path unavailable"
            ) from exc
        if stat.S_ISLNK(information.st_mode) or _is_reparse(information):
            raise DependencyContentLockError(
                "dependency content scan failed: symlink or reparse alias"
            )


def _prefix_logical(path: Path, prefix: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DependencyContentLockError(
            "dependency content scan failed: controlled file unavailable"
        ) from exc
    if not _is_relative_to(resolved, prefix):
        raise DependencyContentLockError(
            "dependency content scan failed: final prefix escape"
        )
    logical = resolved.relative_to(prefix).as_posix()
    return _logical_path(logical, "filesystem.logical_path")


def _controlled_file_identity(
    path: Path, prefix: Path
) -> tuple[tuple[int, int], str]:
    _check_components(path, prefix)
    logical = _prefix_logical(path, prefix)
    try:
        before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or _is_reparse(before)
        ):
            raise DependencyContentLockError(
                f"dependency content scan failed: unsafe file {logical}"
            )
        if before.st_nlink != 1:
            raise DependencyContentLockError(
                f"dependency content scan failed: hardlink alias {logical}"
            )
        with path.open("rb", buffering=0) as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise DependencyContentLockError(
                    f"dependency content scan failed: file identity changed {logical}"
                )
        after = path.stat(follow_symlinks=False)
    except DependencyContentLockError:
        raise
    except OSError as exc:
        raise DependencyContentLockError(
            f"dependency content scan failed: unable to identify {logical}"
        ) from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise DependencyContentLockError(
            f"dependency content scan failed: file changed during scan {logical}"
        )
    return (before.st_dev, before.st_ino), logical


def _snapshot_file(path: Path, prefix: Path, keep_data: bool = False) -> _SnapshotV1:
    _check_components(path, prefix)
    logical = _prefix_logical(path, prefix)
    try:
        before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or _is_reparse(before)
        ):
            raise DependencyContentLockError(
                f"dependency content scan failed: unsafe file {logical}"
            )
        if before.st_nlink != 1:
            raise DependencyContentLockError(
                f"dependency content scan failed: hardlink alias {logical}"
            )
        active_session = _ACTIVE_SNAPSHOT_SESSION.get()
        if active_session is not None:
            active_session.require_unseen(path)
        digest = sha256()
        chunks: list[bytes] | None = [] if keep_data else None
        session = _ACTIVE_SNAPSHOT_SESSION.get()
        if session is None:
            stream = path.open("rb", buffering=0)
            close_after_read = True
        else:
            stream, close_after_read = session.open_stream(path)
        try:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise DependencyContentLockError(
                    f"dependency content scan failed: file identity changed {logical}"
                )
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        finally:
            if close_after_read:
                stream.close()
        after = path.stat(follow_symlinks=False)
        if session is not None:
            session.capture(path, after)
    except DependencyContentLockError:
        raise
    except OSError as exc:
        raise DependencyContentLockError(
            f"dependency content scan failed: unable to snapshot {logical}"
        ) from exc
    if any(getattr(before, field) != getattr(after, field) for field in _STABLE_STAT_FIELDS):
        raise DependencyContentLockError(
            f"dependency content scan failed: file changed during scan {logical}"
        )
    return _SnapshotV1(
        logical_path=logical,
        file_id=(before.st_dev, before.st_ino),
        size_bytes=before.st_size,
        sha256=digest.hexdigest(),
        data=b"".join(chunks) if chunks is not None else None,
    )


def _cache_excluded(logical_path: str) -> bool:
    parts = logical_path.split("/")
    name = parts[-1].casefold()
    return "__pycache__" in (part.casefold() for part in parts) or name.endswith(
        (".pyc", ".pyo")
    )


def _artifact_kind(logical_path: str) -> str:
    suffix = Path(logical_path).suffix.casefold()
    if suffix in (".exe", ".dll", ".lib"):
        return "runtime_binary"
    if suffix in (".pyd", ".so"):
        return "extension"
    if suffix in (".py", ".pyi", ".pyx", ".pxd", ".c", ".h"):
        return "source"
    return "data"


def _content_file(snapshot: _SnapshotV1) -> ContentFileV1:
    return ContentFileV1(
        logical_path=snapshot.logical_path,
        artifact_kind=_artifact_kind(snapshot.logical_path),
        size_bytes=snapshot.size_bytes,
        sha256=snapshot.sha256,
    )


def _enumerate_regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    try:
        for current_text, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_text)
            for name in tuple(directory_names):
                candidate = current / name
                information = candidate.stat(follow_symlinks=False)
                if stat.S_ISLNK(information.st_mode) or _is_reparse(information):
                    raise DependencyContentLockError(
                        "dependency content scan failed: symlink or reparse directory alias"
                    )
            for name in file_names:
                files.append(current / name)
    except DependencyContentLockError:
        raise
    except OSError as exc:
        raise DependencyContentLockError(
            "dependency content scan failed: directory enumeration failed"
        ) from exc
    return tuple(files)


def _record_target(raw_path: str, install_root: Path, prefix: Path) -> Path:
    _string(raw_path, "RECORD.path")
    if (
        raw_path.startswith(("/", "\\"))
        or "\\" in raw_path
        or _DRIVE_RE.match(raw_path) is not None
    ):
        raise DependencyContentLockError(
            "dependency RECORD failed: invalid raw path"
        )
    parts = raw_path.split("/")
    if any(part in ("", ".") for part in parts):
        raise DependencyContentLockError(
            "dependency RECORD failed: invalid raw path segment"
        )
    candidate = install_root.joinpath(*parts)
    _check_components(candidate, prefix)
    try:
        target = candidate.resolve(strict=True)
    except OSError as exc:
        raise DependencyContentLockError(
            "dependency RECORD failed: controlled file unavailable"
        ) from exc
    if not _is_relative_to(target, prefix):
        raise DependencyContentLockError(
            "dependency RECORD failed: final prefix escape"
        )
    return target


def _record_digest(value: str) -> bytes:
    if not value.startswith("sha256="):
        raise DependencyContentLockError(
            "dependency RECORD failed: hash algorithm must be sha256"
        )
    encoded = value[7:]
    if re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded) is None:
        raise DependencyContentLockError(
            "dependency RECORD failed: noncanonical URL-safe SHA-256"
        )
    try:
        decoded = base64.b64decode(encoded + "=", altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise DependencyContentLockError(
            "dependency RECORD failed: malformed URL-safe SHA-256"
        ) from exc
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if len(decoded) != 32 or canonical != encoded:
        raise DependencyContentLockError(
            "dependency RECORD failed: noncanonical URL-safe SHA-256"
        )
    return decoded


def _record_size(value: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise DependencyContentLockError(
            "dependency RECORD failed: size must be a canonical integer"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise DependencyContentLockError(
            "dependency RECORD failed: size is invalid"
        ) from exc


def _parse_distribution(
    distribution: _DistributionInputV1,
    prefix: Path,
) -> tuple[tuple[_SnapshotV1, ...], tuple[Path, ...]]:
    install_root = _canonical_root(distribution.install_root, "distribution root")
    if not _is_relative_to(install_root, prefix):
        raise DependencyContentLockError(
            "dependency RECORD failed: installation root escape"
        )
    try:
        record_relative = distribution.record_path.relative_to(
            distribution.install_root
        )
    except ValueError as exc:
        raise DependencyContentLockError(
            "dependency RECORD failed: RECORD outside installation root"
        ) from exc
    record_candidate = install_root.joinpath(*record_relative.parts)
    _check_components(record_candidate, prefix)
    record_path = record_candidate.resolve(strict=True)
    if not _is_relative_to(record_path, install_root):
        raise DependencyContentLockError(
            "dependency RECORD failed: RECORD outside installation root"
        )
    record_snapshot = _snapshot_file(record_path, prefix, keep_data=True)
    assert record_snapshot.data is not None
    if record_snapshot.data.startswith(b"\xef\xbb\xbf"):
        raise DependencyContentLockError("dependency RECORD failed: BOM is forbidden")
    try:
        text = record_snapshot.data.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DependencyContentLockError(
            "dependency RECORD failed: malformed UTF-8 CSV"
        ) from exc
    snapshots: list[_SnapshotV1] = []
    identities: set[str] = set()
    root_candidates: set[Path] = set()
    saw_record = False
    for index, row in enumerate(rows):
        if len(row) != 3 or not all(type(field) is str for field in row):
            raise DependencyContentLockError(
                f"dependency RECORD failed: row {index} must have exactly three columns"
            )
        raw_path, record_hash, record_size = row
        target = _record_target(raw_path, install_root, prefix)
        logical = _prefix_logical(target, prefix)
        key = _path_key(logical)
        if key in identities:
            raise DependencyContentLockError(
                "dependency RECORD failed: duplicate or NFC/casefold-colliding identity"
            )
        identities.add(key)
        if target == record_path:
            snapshot = record_snapshot
            saw_record = True
        else:
            snapshot = _snapshot_file(target, prefix)
        if bool(record_hash) != bool(record_size):
            raise DependencyContentLockError(
                "dependency RECORD failed: hash and size must be omitted together"
            )
        if record_hash:
            expected_digest = _record_digest(record_hash)
            expected_size = _record_size(record_size)
            if expected_digest.hex() != snapshot.sha256 or expected_size != snapshot.size_bytes:
                raise DependencyContentLockError(
                    f"dependency RECORD failed: content mismatch {logical}"
                )
        snapshots.append(snapshot)
        if _is_relative_to(target, install_root):
            relative = target.relative_to(install_root)
            if relative.parts and relative.parts[0].casefold() != "__pycache__":
                root_candidates.add(install_root / relative.parts[0])
    if not saw_record:
        raise DependencyContentLockError(
            "dependency RECORD failed: RECORD self row is missing"
        )
    return tuple(snapshots), tuple(sorted(root_candidates, key=lambda item: str(item)))


def _scan_environment_once(environment: _ScanEnvironmentV1) -> DependencyContentLockV1:
    if type(environment) is not _ScanEnvironmentV1:
        raise TypeError("environment must be an exact _ScanEnvironmentV1")
    prefix = _canonical_root(environment.prefix, "base prefix")
    stdlib = _canonical_root(environment.stdlib_root, "stdlib root")
    if not _is_relative_to(stdlib, prefix):
        raise DependencyContentLockError(
            "dependency content scan failed: stdlib root escape"
        )
    site_roots = tuple(
        _canonical_root(item, "site-packages root")
        for item in environment.site_packages_roots
    )
    if any(not _is_relative_to(item, stdlib) for item in site_roots):
        raise DependencyContentLockError(
            "dependency content scan failed: site-packages root escape"
        )
    root_ids = {root: _root_identity(root) for root in (prefix, stdlib, *site_roots)}
    semantic = load_semantic_dependency_lock()
    expected_distributions = tuple(
        (item["name"], item["version"]) for item in semantic["distributions"]
    )
    actual_distributions = tuple(
        (item.name, item.version) for item in environment.distributions
    )
    if actual_distributions != expected_distributions:
        raise DependencyContentLockError(
            "dependency content scan failed: distribution set differs from semantic lock"
        )

    distribution_snapshots: list[tuple[_SnapshotV1, ...]] = []
    owned_file_ids: dict[tuple[int, int], str] = {}
    owned_logical: dict[str, str] = {}
    package_roots: dict[str, tuple[str, Path, set[tuple[int, int]]]] = {}
    for distribution in environment.distributions:
        snapshots, roots = _parse_distribution(distribution, prefix)
        distribution_snapshots.append(snapshots)
        snapshot_ids = {item.file_id for item in snapshots}
        for snapshot in snapshots:
            prior = owned_file_ids.get(snapshot.file_id)
            logical_prior = owned_logical.get(_path_key(snapshot.logical_path))
            if prior is not None or logical_prior is not None:
                raise DependencyContentLockError(
                    "dependency content scan failed: multiple distribution owners"
                )
            owned_file_ids[snapshot.file_id] = distribution.name
            owned_logical[_path_key(snapshot.logical_path)] = distribution.name
        for package_root in roots:
            root_key = _path_key(_prefix_logical(package_root, prefix))
            prior_root = package_roots.get(root_key)
            if prior_root is not None and prior_root[0] != distribution.name:
                raise DependencyContentLockError(
                    "dependency content scan failed: ambiguous namespace ownership"
                )
            package_roots[root_key] = (distribution.name, package_root, snapshot_ids)

    for owner, package_root, declared_ids in package_roots.values():
        if package_root.is_dir():
            candidates = _enumerate_regular_files(package_root)
        else:
            candidates = (package_root,)
        for candidate in candidates:
            file_id, logical = _controlled_file_identity(candidate, prefix)
            if file_id not in declared_ids:
                raise DependencyContentLockError(
                    f"dependency content scan failed: undeclared distribution file {logical}"
                )

    def catch_all(root: Path, excluded_roots: tuple[Path, ...]) -> tuple[_SnapshotV1, ...]:
        collected: list[_SnapshotV1] = []
        for candidate in _enumerate_regular_files(root):
            resolved = candidate.resolve(strict=True)
            if any(_is_relative_to(resolved, excluded) for excluded in excluded_roots):
                continue
            logical = _prefix_logical(candidate, prefix)
            if _cache_excluded(logical):
                continue
            file_id, _ = _controlled_file_identity(candidate, prefix)
            if file_id in owned_file_ids:
                continue
            snapshot = _snapshot_file(candidate, prefix)
            collected.append(snapshot)
        return tuple(collected)

    runtime_snapshots = catch_all(prefix, (stdlib,))
    stdlib_snapshots = catch_all(stdlib, site_roots)
    all_snapshots = (
        *runtime_snapshots,
        *stdlib_snapshots,
        *(item for group in distribution_snapshots for item in group),
    )
    seen_ids: set[tuple[int, int]] = set()
    seen_paths: set[str] = set()
    for snapshot in all_snapshots:
        key = _path_key(snapshot.logical_path)
        if snapshot.file_id in seen_ids or key in seen_paths:
            raise DependencyContentLockError(
                "dependency content scan failed: duplicate owner or file alias"
            )
        seen_ids.add(snapshot.file_id)
        seen_paths.add(key)
    for root, identity in root_ids.items():
        _assert_root_identity(root, identity)

    trees = [
        ContentTreeV1(
            "cpython_runtime",
            "CPython",
            environment.python_version,
            tuple(sorted((_content_file(item) for item in runtime_snapshots), key=lambda item: item.logical_path)),
        ),
        ContentTreeV1(
            "cpython_stdlib",
            "stdlib",
            environment.python_version,
            tuple(sorted((_content_file(item) for item in stdlib_snapshots), key=lambda item: item.logical_path)),
        ),
    ]
    for distribution, snapshots in zip(
        environment.distributions, distribution_snapshots, strict=True
    ):
        trees.append(
            ContentTreeV1(
                "distribution",
                distribution.name,
                distribution.version,
                tuple(sorted((_content_file(item) for item in snapshots), key=lambda item: item.logical_path)),
            )
        )
    return DependencyContentLockV1(
        "dependency_content_lock_v1",
        "numeric_protocol_v1",
        semantic_dependency_lock_hash(semantic),
        tuple(trees),
    )


def _scan_environment(environment: _ScanEnvironmentV1) -> DependencyContentLockV1:
    if _ACTIVE_SNAPSHOT_SESSION.get() is not None:
        raise DependencyContentLockError(
            "dependency content scan failed: nested snapshot session"
        )
    session = _SnapshotSessionV1()
    token = _ACTIVE_SNAPSHOT_SESSION.set(session)
    try:
        lock = _scan_environment_once(environment)
        session.validate()
        return lock
    finally:
        try:
            session.close()
        finally:
            _ACTIVE_SNAPSHOT_SESSION.reset(token)



def _build_dependency_content_lock(
    environment: _ScanEnvironmentV1,
) -> DependencyContentLockV1:
    return _scan_environment(environment)


def _verify_dependency_contents_in_environment(
    lock: DependencyContentLockV1,
    environment: _ScanEnvironmentV1,
    _issue_receipt=_issue_verified_receipt,
) -> VerifiedDependencyContentV1:
    if type(lock) is not DependencyContentLockV1:
        _error("$", "must be an exact DependencyContentLockV1")
    actual = _scan_environment(environment)
    if lock != actual:
        expected_by_path = {
            item.logical_path: item
            for tree in lock.trees
            for item in tree.files
        }
        actual_by_path = {
            item.logical_path: item
            for tree in actual.trees
            for item in tree.files
        }
        paths = sorted(set(expected_by_path) | set(actual_by_path))
        drift = [
            path
            for path in paths
            if expected_by_path.get(path) != actual_by_path.get(path)
        ]
        detail = drift[0] if drift else "owner metadata"
        raise DependencyContentLockError(f"dependency content drift: {detail}")
    tree_receipts_list: list[VerifiedContentTreeV1] = []
    for tree in actual.trees:
        payload = {
            "schema_version": "verified_content_tree_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
            "tree": tree.to_payload(),
        }
        tree_receipt = _issue_receipt(
            VerifiedContentTreeV1,
            (
                ("_owner_key", (tree.owner_kind, tree.owner_name)),
                ("_tree_hash", hash_canonical_payload(payload)),
            ),
        )
        if type(tree_receipt) is not VerifiedContentTreeV1:
            raise AssertionError("receipt authority returned the wrong tree type")
        tree_receipts_list.append(tree_receipt)
    tree_receipts = tuple(tree_receipts_list)
    receipt = _issue_receipt(
        VerifiedDependencyContentV1,
        (
            ("_content_lock_hash", dependency_content_lock_hash(lock)),
            ("_trees", tree_receipts),
        ),
    )
    if type(receipt) is not VerifiedDependencyContentV1:
        raise AssertionError("receipt authority returned the wrong dependency type")
    return receipt


del _issue_verified_receipt



def _default_environment() -> _ScanEnvironmentV1:
    semantic = load_semantic_dependency_lock()
    prefix = Path(sys.base_prefix)
    stdlib = Path(sysconfig.get_path("stdlib"))
    site_roots: list[Path] = []
    for name in ("purelib", "platlib"):
        root = Path(sysconfig.get_path(name))
        if root not in site_roots:
            site_roots.append(root)
    distributions: list[_DistributionInputV1] = []
    for item in semantic["distributions"]:
        try:
            installed = metadata.distribution(item["name"])
            files = installed.files
            if type(files) is not list:
                raise DependencyContentLockError(
                    "dependency content scan failed: distribution inventory unavailable"
                )
            records = [
                path
                for path in files
                if str(path).replace("\\", "/").endswith(".dist-info/RECORD")
            ]
            if len(records) != 1:
                raise DependencyContentLockError(
                    "dependency content scan failed: exact RECORD unavailable"
                )
            install_root = Path(installed.locate_file(Path(".")))
            record_path = Path(installed.locate_file(records[0]))
            installed_version = installed.version
        except DependencyContentLockError:
            raise
        except Exception as exc:
            raise DependencyContentLockError(
                "dependency content scan failed: installed distribution unavailable"
            ) from exc
        if type(installed_version) is not str or installed_version != item["version"]:
            raise DependencyContentLockError(
                "dependency content scan failed: distribution version drift"
            )
        distributions.append(
            _DistributionInputV1(
                item["name"], item["version"], install_root, record_path
            )
        )
    return _ScanEnvironmentV1(
        prefix=prefix,
        stdlib_root=stdlib,
        site_packages_roots=tuple(site_roots),
        implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        distributions=tuple(distributions),
    )


def verify_current_dependency_contents(
    lock: DependencyContentLockV1,
) -> VerifiedDependencyContentV1:
    """Verify every locked byte using roots derived from this interpreter."""

    semantic = load_semantic_dependency_lock()
    verify_current_runtime(semantic)
    return _verify_dependency_contents_in_environment(lock, _default_environment())


def generate_dependency_content_lock(
    output_path: Path,
) -> DependencyContentLockV1:
    """Audit the current interpreter and exclusively create one candidate lock."""

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    if output_path.exists():
        raise DependencyContentLockError(
            "dependency content generation requires a nonexistent output"
        )
    semantic = load_semantic_dependency_lock()
    verify_current_runtime(semantic)
    lock = _build_dependency_content_lock(_default_environment())
    payload = json.dumps(
        lock.to_payload(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ).encode("utf-8") + b"\n"
    try:
        with output_path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise DependencyContentLockError(
            "dependency content generation requires a nonexistent output"
        ) from exc
    except OSError as exc:
        raise DependencyContentLockError(
            "dependency content generation failed"
        ) from exc
    return lock


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "generate":
        generate_dependency_content_lock(arguments.output)
        return 0
    raise AssertionError("unreachable")


__all__ = (
    "ContentFileV1",
    "ContentTreeV1",
    "DependencyContentLockError",
    "DependencyContentLockV1",
    "VerifiedContentTreeV1",
    "VerifiedDependencyContentV1",
    "dependency_content_lock_hash",
    "load_dependency_content_lock",
    "verify_current_dependency_contents",
)


if __name__ == "__main__":
    raise SystemExit(_main())
