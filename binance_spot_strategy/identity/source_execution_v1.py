"""Same-process attested source execution for the Binance Spot M2 gate.

The stable records in this module are serializable evidence.  The session and
admission classes are deliberately process-local capabilities backed by hidden
issuer registries and retained file handles.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Callable, Iterator, NoReturn
import unicodedata
import weakref

from .dependency_contents_v1 import (
    ContentTreeV1,
    DependencyContentLockError,
    DependencyContentLockV1,
    VerifiedDependencyContentV1,
    _ScanEnvironmentV1,
    _SnapshotSessionV1,
    _check_components,
    _default_environment,
    _is_reparse,
    _require_verified_dependency_content,
    _verify_dependency_contents_in_environment,
    dependency_content_lock_hash,
    verify_current_dependency_contents,
)
from binance_spot_strategy.protocols import canonical_json_bytes

from .digests import hash_canonical_payload, require_sha256, sha256_bytes


class SourceExecutionValidationError(ValueError):
    """Raised when source-execution policy or live evidence fails closed."""


_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "dependency_content_lock_sha256",
        "bootstrap_launcher_raw_sha256",
        "bootstrap_modules",
        "entrypoints",
        "business_modules",
        "dependency_owner_tree_sha256s",
    }
)
_MODULE_KEYS = frozenset(
    {
        "role",
        "module_name",
        "logical_path",
        "tracked_source_sha256",
        "raw_artifact_sha256",
    }
)
_ENTRYPOINT_KEYS = frozenset({"mode", "module_name", "callable_name"})
_ALLOWED_MODES = frozenset(
    {"task2-synthetic", "verify-m2", "seal-historical-data"}
)
_PRODUCTION_MODES = frozenset({"verify-m2", "seal-historical-data"})
_M2_EXECUTION_OWNER_IDENTITIES = (
    ("cpython_runtime", "CPython"),
    ("cpython_stdlib", "stdlib"),
    ("distribution", "numpy"),
    ("distribution", "python-dateutil"),
    ("distribution", "six"),
    ("distribution", "tzdata"),
)
_FORBIDDEN_DEPENDENCY_ROOTS = ("pandas", "pyarrow")
_BOOTSTRAP_IDENTITIES = (
    (
        "binance_spot_strategy.identity.dependency_contents_v1",
        "binance_spot_strategy/identity/dependency_contents_v1.py",
    ),
    (
        "binance_spot_strategy.identity.source_execution_v1",
        "binance_spot_strategy/identity/source_execution_v1.py",
    ),
)
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DRIVE_RE = re.compile(r"[A-Za-z]:")
_UTF8_BOM = b"\xef\xbb\xbf"


def _fail(path: str, reason: str) -> NoReturn:
    raise SourceExecutionValidationError(
        f"invalid source execution evidence at {path}: {reason}"
    )


def _exact_dict(value: object, keys: frozenset[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an exact object")
    assert isinstance(value, dict)
    if not all(type(key) is str for key in value):
        _fail(path, "object keys must be exact strings")
    actual = frozenset(value)
    if actual != keys:
        _fail(
            path,
            f"keys differ (missing={sorted(keys - actual)}, extra={sorted(actual - keys)})",
        )
    return value


def _exact_list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an exact array")
    assert isinstance(value, list)
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a nonempty exact string")
    assert isinstance(value, str)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SourceExecutionValidationError(
            f"invalid source execution evidence at {path}: invalid Unicode"
        ) from exc
    if unicodedata.normalize("NFC", value) != value:
        _fail(path, "must use NFC normalization")
    return value


def _digest(value: object, path: str) -> str:
    try:
        return require_sha256(value, path)
    except (TypeError, ValueError) as exc:
        _fail(path, str(exc))


def _module_name(value: object, path: str) -> str:
    name = _string(value, path)
    if _MODULE_RE.fullmatch(name) is None:
        _fail(path, "must be a dotted ASCII Python module name")
    return name


def _callable_name(value: object, path: str) -> str:
    name = _string(value, path)
    if _IDENTIFIER_RE.fullmatch(name) is None:
        _fail(path, "must be one ASCII Python identifier")
    return name


def _logical_source_path(value: object, path: str) -> str:
    logical = _string(value, path)
    if (
        logical.startswith(("/", "\\"))
        or "\\" in logical
        or _DRIVE_RE.match(logical) is not None
    ):
        _fail(path, "must be a repository-relative forward-slash path")
    parts = logical.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail(path, "must not contain empty, dot, or dot-dot segments")
    if any(":" in part for part in parts):
        _fail(path, "must not contain drives or alternate streams")
    if not logical.endswith(".py") or logical.endswith((".pyc", ".pyo")):
        _fail(path, "must identify one Python source artifact")
    return logical


def _identity_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


@dataclass(frozen=True, slots=True)
class BusinessModuleBindingV1:
    role: str
    module_name: str
    logical_path: str
    tracked_source_sha256: str
    raw_artifact_sha256: str

    def __post_init__(self) -> None:
        role = _string(self.role, "BusinessModuleBindingV1.role")
        if role not in ("bootstrap", "business"):
            _fail("BusinessModuleBindingV1.role", "must be bootstrap or business")
        _module_name(self.module_name, "BusinessModuleBindingV1.module_name")
        _logical_source_path(
            self.logical_path, "BusinessModuleBindingV1.logical_path"
        )
        _digest(
            self.tracked_source_sha256,
            "BusinessModuleBindingV1.tracked_source_sha256",
        )
        _digest(
            self.raw_artifact_sha256,
            "BusinessModuleBindingV1.raw_artifact_sha256",
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "module_name": self.module_name,
            "logical_path": self.logical_path,
            "tracked_source_sha256": self.tracked_source_sha256,
            "raw_artifact_sha256": self.raw_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class EntrypointBindingV1:
    mode: str
    module_name: str
    callable_name: str

    def __post_init__(self) -> None:
        mode = _string(self.mode, "EntrypointBindingV1.mode")
        if mode not in _ALLOWED_MODES:
            _fail("EntrypointBindingV1.mode", "is not an internal literal mode")
        _module_name(self.module_name, "EntrypointBindingV1.module_name")
        _callable_name(self.callable_name, "EntrypointBindingV1.callable_name")

    def to_payload(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "module_name": self.module_name,
            "callable_name": self.callable_name,
        }


@dataclass(frozen=True, slots=True)
class SourceExecutionPolicyV1:
    schema_version: str
    dependency_content_lock_sha256: str
    bootstrap_launcher_raw_sha256: str
    bootstrap_modules: tuple[BusinessModuleBindingV1, ...]
    entrypoints: tuple[EntrypointBindingV1, ...]
    business_modules: tuple[BusinessModuleBindingV1, ...]
    dependency_owner_tree_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if _string(self.schema_version, "SourceExecutionPolicyV1.schema_version") != "source_execution_policy_v1":
            _fail(
                "SourceExecutionPolicyV1.schema_version",
                "must equal source_execution_policy_v1",
            )
        _digest(
            self.dependency_content_lock_sha256,
            "SourceExecutionPolicyV1.dependency_content_lock_sha256",
        )
        _digest(
            self.bootstrap_launcher_raw_sha256,
            "SourceExecutionPolicyV1.bootstrap_launcher_raw_sha256",
        )
        self._validate_modules()
        self._validate_entrypoints()
        if type(self.dependency_owner_tree_sha256s) is not tuple:
            _fail(
                "SourceExecutionPolicyV1.dependency_owner_tree_sha256s",
                "must be an exact tuple",
            )
        if not self.dependency_owner_tree_sha256s:
            _fail(
                "SourceExecutionPolicyV1.dependency_owner_tree_sha256s",
                "must not be empty",
            )
        if not all(type(item) is str for item in self.dependency_owner_tree_sha256s):
            _fail(
                "SourceExecutionPolicyV1.dependency_owner_tree_sha256s",
                "must contain exact strings",
            )
        for index, digest in enumerate(self.dependency_owner_tree_sha256s):
            _digest(digest, f"dependency_owner_tree_sha256s[{index}]")
        if len(set(self.dependency_owner_tree_sha256s)) != len(
            self.dependency_owner_tree_sha256s
        ):
            _fail(
                "SourceExecutionPolicyV1.dependency_owner_tree_sha256s",
                "must be unique",
            )

    def _validate_modules(self) -> None:
        if type(self.bootstrap_modules) is not tuple or not all(
            type(item) is BusinessModuleBindingV1 for item in self.bootstrap_modules
        ):
            _fail(
                "SourceExecutionPolicyV1.bootstrap_modules",
                "must contain exact module bindings",
            )
        if tuple(
            (item.module_name, item.logical_path) for item in self.bootstrap_modules
        ) != _BOOTSTRAP_IDENTITIES:
            _fail(
                "SourceExecutionPolicyV1.bootstrap_modules",
                "must contain both exact bootstrap helpers in order",
            )
        if any(item.role != "bootstrap" for item in self.bootstrap_modules):
            _fail(
                "SourceExecutionPolicyV1.bootstrap_modules",
                "must contain only bootstrap roles",
            )
        if type(self.business_modules) is not tuple or not self.business_modules:
            _fail(
                "SourceExecutionPolicyV1.business_modules",
                "must be a nonempty exact tuple",
            )
        if not all(type(item) is BusinessModuleBindingV1 for item in self.business_modules):
            _fail(
                "SourceExecutionPolicyV1.business_modules",
                "must contain exact module bindings",
            )
        if any(item.role != "business" for item in self.business_modules):
            _fail(
                "SourceExecutionPolicyV1.business_modules",
                "must contain only business roles",
            )
        names = [item.module_name for item in self.business_modules]
        if names != sorted(names):
            _fail("SourceExecutionPolicyV1.business_modules", "must be sorted")
        keys = [_identity_key(item) for item in names]
        paths = [_identity_key(item.logical_path) for item in self.business_modules]
        if len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
            _fail(
                "SourceExecutionPolicyV1.business_modules",
                "contains duplicate or colliding identities",
            )

    def _validate_entrypoints(self) -> None:
        if type(self.entrypoints) is not tuple or not self.entrypoints:
            _fail(
                "SourceExecutionPolicyV1.entrypoints",
                "must be a nonempty exact tuple",
            )
        if not all(type(item) is EntrypointBindingV1 for item in self.entrypoints):
            _fail(
                "SourceExecutionPolicyV1.entrypoints",
                "must contain exact entrypoint bindings",
            )
        order = [
            (item.mode, item.module_name, item.callable_name)
            for item in self.entrypoints
        ]
        if order != sorted(order):
            _fail("SourceExecutionPolicyV1.entrypoints", "must be sorted")
        mode_keys = [_identity_key(item.mode) for item in self.entrypoints]
        if len(mode_keys) != len(set(mode_keys)):
            _fail(
                "SourceExecutionPolicyV1.entrypoints",
                "contains duplicate or colliding modes",
            )
        modules = {item.module_name for item in self.business_modules}
        if any(item.module_name not in modules for item in self.entrypoints):
            _fail(
                "SourceExecutionPolicyV1.entrypoints",
                "must bind an exact business module",
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dependency_content_lock_sha256": self.dependency_content_lock_sha256,
            "bootstrap_launcher_raw_sha256": self.bootstrap_launcher_raw_sha256,
            "bootstrap_modules": [item.to_payload() for item in self.bootstrap_modules],
            "entrypoints": [item.to_payload() for item in self.entrypoints],
            "business_modules": [item.to_payload() for item in self.business_modules],
            "dependency_owner_tree_sha256s": list(
                self.dependency_owner_tree_sha256s
            ),
        }


@dataclass(frozen=True, slots=True)
class ExecutionClosureV1:
    schema_version: str
    execution_policy_sha256: str
    dependency_content_lock_sha256: str
    business_source_tree_sha256: str
    business_artifact_manifest_sha256: str
    business_modules: tuple[BusinessModuleBindingV1, ...]

    def __post_init__(self) -> None:
        if _string(self.schema_version, "ExecutionClosureV1.schema_version") != "execution_closure_v1":
            _fail(
                "ExecutionClosureV1.schema_version",
                "must equal execution_closure_v1",
            )
        for name in (
            "execution_policy_sha256",
            "dependency_content_lock_sha256",
            "business_source_tree_sha256",
            "business_artifact_manifest_sha256",
        ):
            _digest(getattr(self, name), f"ExecutionClosureV1.{name}")
        if type(self.business_modules) is not tuple or not all(
            type(item) is BusinessModuleBindingV1 for item in self.business_modules
        ):
            _fail(
                "ExecutionClosureV1.business_modules",
                "must contain exact module bindings",
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_policy_sha256": self.execution_policy_sha256,
            "dependency_content_lock_sha256": self.dependency_content_lock_sha256,
            "business_source_tree_sha256": self.business_source_tree_sha256,
            "business_artifact_manifest_sha256": self.business_artifact_manifest_sha256,
            "business_modules": [item.to_payload() for item in self.business_modules],
        }


def _binding_from_payload(value: object, path: str) -> BusinessModuleBindingV1:
    item = _exact_dict(value, _MODULE_KEYS, path)
    return BusinessModuleBindingV1(
        role=_string(item["role"], f"{path}.role"),
        module_name=_module_name(item["module_name"], f"{path}.module_name"),
        logical_path=_logical_source_path(
            item["logical_path"], f"{path}.logical_path"
        ),
        tracked_source_sha256=_digest(
            item["tracked_source_sha256"], f"{path}.tracked_source_sha256"
        ),
        raw_artifact_sha256=_digest(
            item["raw_artifact_sha256"], f"{path}.raw_artifact_sha256"
        ),
    )


def _entrypoint_from_payload(value: object, path: str) -> EntrypointBindingV1:
    item = _exact_dict(value, _ENTRYPOINT_KEYS, path)
    return EntrypointBindingV1(
        mode=_string(item["mode"], f"{path}.mode"),
        module_name=_module_name(item["module_name"], f"{path}.module_name"),
        callable_name=_callable_name(
            item["callable_name"], f"{path}.callable_name"
        ),
    )


def source_execution_policy_from_payload(payload: object) -> SourceExecutionPolicyV1:
    """Parse one strict source-execution policy payload."""

    root = _exact_dict(payload, _POLICY_KEYS, "$")
    bootstrap = _exact_list(root["bootstrap_modules"], "$.bootstrap_modules")
    entrypoints = _exact_list(root["entrypoints"], "$.entrypoints")
    business = _exact_list(root["business_modules"], "$.business_modules")
    owners = _exact_list(
        root["dependency_owner_tree_sha256s"],
        "$.dependency_owner_tree_sha256s",
    )
    return SourceExecutionPolicyV1(
        schema_version=_string(root["schema_version"], "$.schema_version"),
        dependency_content_lock_sha256=_digest(
            root["dependency_content_lock_sha256"],
            "$.dependency_content_lock_sha256",
        ),
        bootstrap_launcher_raw_sha256=_digest(
            root["bootstrap_launcher_raw_sha256"],
            "$.bootstrap_launcher_raw_sha256",
        ),
        bootstrap_modules=tuple(
            _binding_from_payload(item, f"$.bootstrap_modules[{index}]")
            for index, item in enumerate(bootstrap)
        ),
        entrypoints=tuple(
            _entrypoint_from_payload(item, f"$.entrypoints[{index}]")
            for index, item in enumerate(entrypoints)
        ),
        business_modules=tuple(
            _binding_from_payload(item, f"$.business_modules[{index}]")
            for index, item in enumerate(business)
        ),
        dependency_owner_tree_sha256s=tuple(
            _digest(item, f"$.dependency_owner_tree_sha256s[{index}]")
            for index, item in enumerate(owners)
        ),
    )


def source_execution_policy_hash(policy: SourceExecutionPolicyV1) -> str:
    if type(policy) is not SourceExecutionPolicyV1:
        _fail("$", "must be an exact SourceExecutionPolicyV1")
    return sha256_bytes(canonical_json_bytes(policy.to_payload()))


def _tracked_sha256(raw: bytes) -> str:
    if raw.startswith(_UTF8_BOM):
        raise SourceExecutionValidationError("source digest failed: UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceExecutionValidationError("source digest failed: invalid UTF-8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalized.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _PinnedSnapshotV1:
    logical_path: str
    path: Path
    file_id: tuple[int, int]
    size_bytes: int
    raw_sha256: str
    data: bytes | None


_AUTHORIZED_OPEN_DEPTH: ContextVar[int] = ContextVar(
    "source_execution_authorized_open_depth", default=0
)


@contextmanager
def _authorized_open() -> Iterator[None]:
    token = _AUTHORIZED_OPEN_DEPTH.set(_AUTHORIZED_OPEN_DEPTH.get() + 1)
    try:
        yield
    finally:
        _AUTHORIZED_OPEN_DEPTH.reset(token)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_root(path: Path, label: str) -> Path:
    try:
        information = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SourceExecutionValidationError(f"{label} unavailable") from exc
    if (
        not stat.S_ISDIR(information.st_mode)
        or stat.S_ISLNK(information.st_mode)
        or _is_reparse(information)
    ):
        raise SourceExecutionValidationError(f"unsafe {label}")
    return resolved


def _read_pinned_snapshot(
    session: _SnapshotSessionV1,
    path: Path,
    root: Path,
    logical_path: str,
    *,
    keep_data: bool,
) -> _PinnedSnapshotV1:
    _check_components(path, root)
    try:
        before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or _is_reparse(before)
            or before.st_nlink != 1
        ):
            raise SourceExecutionValidationError("unsafe controlled artifact")
        stream, close_after = session.open_stream(path)
        digest = sha256()
        chunks: list[bytes] | None = [] if keep_data else None
        try:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise SourceExecutionValidationError("controlled file identity changed")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        finally:
            if close_after:
                stream.close()
        after = path.stat(follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise SourceExecutionValidationError("controlled file changed during read")
        session.capture(path, after)
    except (SourceExecutionValidationError, DependencyContentLockError):
        raise
    except OSError as exc:
        raise SourceExecutionValidationError("controlled artifact unavailable") from exc
    data = b"".join(chunks) if chunks is not None else None
    return _PinnedSnapshotV1(
        logical_path=logical_path,
        path=path,
        file_id=(before.st_dev, before.st_ino),
        size_bytes=before.st_size,
        raw_sha256=digest.hexdigest(),
        data=data,
    )


class _ControlledSourceLoader(importlib.abc.Loader):
    def __init__(self, guard: "_PersistentImportGuardV1", fullname: str, snapshot: _PinnedSnapshotV1, package: bool) -> None:
        self._guard = guard
        self._fullname = fullname
        self._snapshot = snapshot
        self._package = package

    def create_module(self, spec):
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        if not self._guard.active:
            raise SourceExecutionValidationError("attested import guard is inactive")
        raw = self._snapshot.data
        if type(raw) is not bytes:
            raise SourceExecutionValidationError("source snapshot unavailable")
        module.__file__ = self._snapshot.logical_path
        module.__loader__ = self
        module.__package__ = self._fullname if self._package else self._fullname.rpartition(".")[0]
        if self._package:
            module.__path__ = [str(self._snapshot.path.parent)]
        code = compile(raw, self._snapshot.logical_path, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _PersistentImportGuardV1(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        repository_root: Path,
        dependency_root: Path,
        snapshots_by_path: dict[Path, _PinnedSnapshotV1],
        business_by_name: dict[str, _PinnedSnapshotV1],
    ) -> None:
        self.repository_root = repository_root
        self.dependency_root = dependency_root
        self.snapshots_by_path = snapshots_by_path
        self.business_by_name = business_by_name
        self.loaded_names: set[str] = set()
        self.active = True

    def protects(self, value: object) -> bool:
        if not self.active or _AUTHORIZED_OPEN_DEPTH.get() > 0:
            return False
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        try:
            path = Path(os.fsdecode(value)).resolve(strict=False)
        except (TypeError, ValueError, OSError):
            return False
        return _under(path, self.repository_root) or _under(path, self.dependency_root)

    def find_spec(self, fullname: str, path=None, target=None):
        if not self.active:
            raise SourceExecutionValidationError("attested import guard is inactive")
        if _is_forbidden_dependency_name(fullname):
            raise SourceExecutionValidationError(
                "forbidden pandas or pyarrow import rejected before resolver delegation"
            )
        if fullname in self.business_by_name:
            snapshot = self.business_by_name[fullname]
            is_package = snapshot.logical_path.endswith("/__init__.py")
            loader = _ControlledSourceLoader(self, fullname, snapshot, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=snapshot.logical_path,
                is_package=is_package,
            )
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            raise SourceExecutionValidationError(f"undeclared import rejected: {fullname}")
        if spec.origin in ("built-in", "frozen"):
            if fullname not in sys.builtin_module_names and fullname not in sys.stdlib_module_names:
                raise SourceExecutionValidationError("unreviewed builtin/frozen import")
            return spec
        if spec.origin is None:
            raise SourceExecutionValidationError("namespace loader rejected")
        try:
            origin = Path(spec.origin).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise SourceExecutionValidationError("import origin unavailable") from exc
        snapshot = self.snapshots_by_path.get(origin)
        if snapshot is None:
            raise SourceExecutionValidationError(f"undeclared import rejected: {fullname}")
        suffix = origin.suffix.casefold()
        if suffix in (".pyc", ".pyo"):
            raise SourceExecutionValidationError("bytecode-only import rejected")
        if suffix == ".py":
            is_package = origin.name == "__init__.py"
            loader = _ControlledSourceLoader(self, fullname, snapshot, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=snapshot.logical_path,
                is_package=is_package,
            )
        allowed_extension = any(
            str(origin).casefold().endswith(item.casefold())
            for item in importlib.machinery.EXTENSION_SUFFIXES
        )
        if not allowed_extension:
            raise SourceExecutionValidationError("custom or memory loader rejected")
        if not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            raise SourceExecutionValidationError("custom extension loader rejected")
        return spec


_ACTIVE_GUARDS: weakref.WeakSet[_PersistentImportGuardV1] = weakref.WeakSet()
_AUDIT_HOOK_INSTALLED = False


def _audit_hook(event: str, args: tuple[object, ...]) -> None:
    if event not in ("open", "os.open") or not args:
        return
    for guard in tuple(_ACTIVE_GUARDS):
        if guard.protects(args[0]):
            raise PermissionError("attested guard denied direct protected-root access")


def _install_guard(guard: _PersistentImportGuardV1) -> None:
    global _AUDIT_HOOK_INSTALLED
    if not _AUDIT_HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _AUDIT_HOOK_INSTALLED = True
    _ACTIVE_GUARDS.add(guard)
    sys.meta_path.insert(0, guard)


def _remove_guard(guard: _PersistentImportGuardV1) -> None:
    guard.active = False
    _ACTIVE_GUARDS.discard(guard)
    while guard in sys.meta_path:
        sys.meta_path.remove(guard)


class _OpaqueExecutionCapability:
    __slots__ = ("__weakref__",)

    def __new__(cls, *args: object, **kwargs: object):
        raise TypeError("attested execution capabilities are verifier-issued only")

    def __reduce__(self) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be copied")


class AttestedExecutionSessionV1(_OpaqueExecutionCapability):
    __slots__ = (
        "_closure",
        "_entrypoint_result",
        "_guard",
        "_pinned_session",
        "_verified_content",
        "_policy",
        "_content_lock",
        "_environment",
        "_repository_root",
        "_source_snapshots",
        "_dependency_snapshots",
        "_loaded_module_names",
        "_test_only",
    )

    def __init_subclass__(cls, **kwargs: object) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be subclassed")

    @property
    def closure(self) -> ExecutionClosureV1:
        return self._closure

    @property
    def entrypoint_result(self) -> object:
        return self._entrypoint_result


class AttestedExecutionAdmissionV1(_OpaqueExecutionCapability):
    __slots__ = ("_session_ref", "_guard_ref", "_issuer_pid")

    def __init_subclass__(cls, **kwargs: object) -> NoReturn:
        raise TypeError("attested execution capabilities cannot be subclassed")


def _execution_owner_trees(
    lock: DependencyContentLockV1,
) -> tuple[ContentTreeV1, ...]:
    by_identity = {
        (tree.owner_kind, tree.owner_name): tree for tree in lock.trees
    }
    if len(by_identity) != len(lock.trees):
        raise SourceExecutionValidationError("duplicate dependency owner tree identity")
    try:
        return tuple(by_identity[identity] for identity in _M2_EXECUTION_OWNER_IDENTITIES)
    except KeyError as exc:
        raise SourceExecutionValidationError(
            "required M2 dependency owner tree is missing"
        ) from exc


def _is_forbidden_dependency_name(fullname: object) -> bool:
    return type(fullname) is str and any(
        fullname == root or fullname.startswith(root + ".")
        for root in _FORBIDDEN_DEPENDENCY_ROOTS
    )


def _reject_prefilled_forbidden_modules() -> None:
    prefilled = sorted(
        name
        for name in sys.modules
        if _is_forbidden_dependency_name(name)
    )
    if prefilled:
        raise SourceExecutionValidationError(
            "prefilled pandas or pyarrow module rejected"
        )


def _tree_identity(tree: ContentTreeV1) -> str:
    return hash_canonical_payload(
        {
            "schema_version": "dependency_owner_tree_binding_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
            "tree": tree.to_payload(),
        }
    )


def _validate_policy_against_lock(
    policy: SourceExecutionPolicyV1, lock: DependencyContentLockV1
) -> None:
    if type(lock) is not DependencyContentLockV1:
        _fail("content_lock", "must be an exact DependencyContentLockV1")
    if dependency_content_lock_hash(lock) != policy.dependency_content_lock_sha256:
        raise SourceExecutionValidationError("dependency content lock digest mismatch")
    admitted_owner_trees = tuple(
        _tree_identity(tree) for tree in _execution_owner_trees(lock)
    )
    if admitted_owner_trees != policy.dependency_owner_tree_sha256s:
        raise SourceExecutionValidationError(
            "dependency owner trees are not the exact ordered M2 subset; pandas is excluded"
        )


def _verify_binding(snapshot: _PinnedSnapshotV1, binding: BusinessModuleBindingV1) -> None:
    raw = snapshot.data
    if type(raw) is not bytes:
        raise SourceExecutionValidationError("source snapshot bytes unavailable")
    if snapshot.raw_sha256 != binding.raw_artifact_sha256:
        raise SourceExecutionValidationError("source raw digest mismatch")
    if _tracked_sha256(raw) != binding.tracked_source_sha256:
        raise SourceExecutionValidationError("source tracked digest mismatch")


def _build_closure(policy: SourceExecutionPolicyV1) -> ExecutionClosureV1:
    source_payload = {
        "schema_version": "business_source_tree_v1",
        "numeric_protocol_version": "numeric_protocol_v1",
        "modules": [
            {
                "role": item.role,
                "module_name": item.module_name,
                "logical_path": item.logical_path,
                "tracked_source_sha256": item.tracked_source_sha256,
            }
            for item in policy.business_modules
        ],
    }
    artifact_payload = {
        "schema_version": "business_artifact_manifest_v1",
        "numeric_protocol_version": "numeric_protocol_v1",
        "modules": [item.to_payload() for item in policy.business_modules],
    }
    return ExecutionClosureV1(
        schema_version="execution_closure_v1",
        execution_policy_sha256=source_execution_policy_hash(policy),
        dependency_content_lock_sha256=policy.dependency_content_lock_sha256,
        business_source_tree_sha256=hash_canonical_payload(source_payload),
        business_artifact_manifest_sha256=hash_canonical_payload(artifact_payload),
        business_modules=policy.business_modules,
    )


def _create_capability_endpoints():
    sessions: weakref.WeakKeyDictionary[object, int] = weakref.WeakKeyDictionary()
    admissions: weakref.WeakKeyDictionary[object, tuple[AttestedExecutionSessionV1, int]] = weakref.WeakKeyDictionary()

    def register_session(session: AttestedExecutionSessionV1) -> None:
        sessions[session] = os.getpid()

    def require_session(value: object) -> AttestedExecutionSessionV1:
        if type(value) is not AttestedExecutionSessionV1:
            raise SourceExecutionValidationError("untrusted attested execution session")
        try:
            pid = sessions[value]
        except (KeyError, TypeError) as exc:
            raise SourceExecutionValidationError("untrusted attested execution session") from exc
        if pid != os.getpid():
            raise SourceExecutionValidationError("foreign-process attested execution session")
        assert isinstance(value, AttestedExecutionSessionV1)
        if not value._guard.active:
            raise SourceExecutionValidationError("attested execution session is inactive")
        return value

    def register_admission(
        admission: AttestedExecutionAdmissionV1,
        session: AttestedExecutionSessionV1,
    ) -> None:
        admissions[admission] = (session, os.getpid())

    def require_admission(value: object) -> AttestedExecutionAdmissionV1:
        if type(value) is not AttestedExecutionAdmissionV1:
            raise SourceExecutionValidationError("untrusted M2 admission")
        try:
            session, pid = admissions[value]
        except (KeyError, TypeError) as exc:
            raise SourceExecutionValidationError("untrusted M2 admission") from exc
        if pid != os.getpid():
            raise SourceExecutionValidationError("foreign-process M2 admission")
        require_session(session)
        assert isinstance(value, AttestedExecutionAdmissionV1)
        if value._session_ref() is not session or value._guard_ref() is not session._guard:
            raise SourceExecutionValidationError("M2 admission issuer identity changed")
        return value

    def unregister_session(session: AttestedExecutionSessionV1) -> None:
        sessions.pop(session, None)
        for admission, state in tuple(admissions.items()):
            if state[0] is session:
                admissions.pop(admission, None)

    endpoints = tuple(
        lru_cache(maxsize=0)(item)
        for item in (
            register_session,
            require_session,
            register_admission,
            require_admission,
            unregister_session,
        )
    )
    for endpoint in endpoints:
        del endpoint.__wrapped__
    return endpoints


(
    _register_session,
    _require_session,
    _register_admission,
    _require_admission,
    _unregister_session,
) = _create_capability_endpoints()
del _create_capability_endpoints


def _pin_dependency_tree(
    lock: DependencyContentLockV1,
    environment: _ScanEnvironmentV1,
    retained: _SnapshotSessionV1,
) -> tuple[dict[Path, _PinnedSnapshotV1], dict[str, _PinnedSnapshotV1]]:
    root = _safe_root(environment.prefix, "dependency root")
    by_path: dict[Path, _PinnedSnapshotV1] = {}
    by_logical: dict[str, _PinnedSnapshotV1] = {}
    for tree in _execution_owner_trees(lock):
        for item in tree.files:
            path = root.joinpath(*item.logical_path.split("/"))
            snapshot = _read_pinned_snapshot(
                retained,
                path,
                root,
                item.logical_path,
                keep_data=item.artifact_kind in ("source", "data", "runtime_binary"),
            )
            if snapshot.size_bytes != item.size_bytes or snapshot.raw_sha256 != item.sha256:
                raise SourceExecutionValidationError("dependency snapshot digest mismatch")
            resolved = path.resolve(strict=True)
            if resolved in by_path:
                raise SourceExecutionValidationError("duplicate dependency file identity")
            by_path[resolved] = snapshot
            by_logical[item.logical_path] = snapshot
    retained.validate()
    return by_path, by_logical


def _pin_repository_sources(
    policy: SourceExecutionPolicyV1,
    repository_root: Path,
    retained: _SnapshotSessionV1,
) -> tuple[dict[Path, _PinnedSnapshotV1], dict[str, _PinnedSnapshotV1]]:
    root = _safe_root(repository_root, "repository root")
    bindings = (*policy.bootstrap_modules, *policy.business_modules)
    snapshots_by_path: dict[Path, _PinnedSnapshotV1] = {}
    snapshots_by_module: dict[str, _PinnedSnapshotV1] = {}
    for binding in bindings:
        path = root.joinpath(*binding.logical_path.split("/"))
        snapshot = _read_pinned_snapshot(
            retained,
            path,
            root,
            binding.logical_path,
            keep_data=True,
        )
        _verify_binding(snapshot, binding)
        resolved = path.resolve(strict=True)
        if resolved in snapshots_by_path:
            raise SourceExecutionValidationError("duplicate repository source identity")
        snapshots_by_path[resolved] = snapshot
        snapshots_by_module[binding.module_name] = snapshot

    launcher = root / "binance_spot_strategy" / "attested_launcher.py"
    launcher_snapshot = _read_pinned_snapshot(
        retained,
        launcher,
        root,
        "binance_spot_strategy/attested_launcher.py",
        keep_data=False,
    )
    if launcher_snapshot.raw_sha256 != policy.bootstrap_launcher_raw_sha256:
        raise SourceExecutionValidationError("bootstrap launcher digest mismatch")
    snapshots_by_path[launcher.resolve(strict=True)] = launcher_snapshot
    retained.validate()
    return snapshots_by_path, snapshots_by_module


def _execute_business_modules(
    policy: SourceExecutionPolicyV1,
    guard: _PersistentImportGuardV1,
) -> tuple[tuple[str, ...], object]:
    # The private Task 2 engine is itself imported by the unit-test process;
    # the direct launcher separately enforces that bootstrap helpers were not
    # normally imported.  Business modules must always start absent.
    governed = {item.module_name for item in policy.business_modules}
    prefilled = sorted(name for name in governed if name in sys.modules)
    if prefilled:
        raise SourceExecutionValidationError("prefilled governed module rejected")

    loaded: list[str] = []
    try:
        for binding in policy.business_modules:
            snapshot = guard.business_by_name[binding.module_name]
            is_package = snapshot.logical_path.endswith("/__init__.py")
            loader = _ControlledSourceLoader(
                guard, binding.module_name, snapshot, is_package
            )
            spec = importlib.util.spec_from_loader(
                binding.module_name,
                loader,
                origin=snapshot.logical_path,
                is_package=is_package,
            )
            if spec is None:
                raise SourceExecutionValidationError("unable to build controlled spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[binding.module_name] = module
            loaded.append(binding.module_name)
            loader.exec_module(module)
            if getattr(module, "__file__", None) != snapshot.logical_path:
                raise SourceExecutionValidationError("module __file__ provenance changed")
        result: object = None
        if len(policy.entrypoints) == 1 and policy.entrypoints[0].mode == "task2-synthetic":
            entrypoint = policy.entrypoints[0]
            module = sys.modules[entrypoint.module_name]
            target = getattr(module, entrypoint.callable_name, None)
            if not callable(target):
                raise SourceExecutionValidationError("entrypoint callable unavailable")
            result = target()
        return tuple(loaded), result
    except Exception:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        raise


def _establish(
    policy: SourceExecutionPolicyV1,
    lock: DependencyContentLockV1,
    environment: _ScanEnvironmentV1,
    repository_root: Path,
    verified: VerifiedDependencyContentV1,
    *,
    test_only: bool,
) -> AttestedExecutionSessionV1:
    if type(policy) is not SourceExecutionPolicyV1:
        _fail("policy", "must be an exact SourceExecutionPolicyV1")
    _reject_prefilled_forbidden_modules()
    _validate_policy_against_lock(policy, lock)
    _require_verified_dependency_content(verified)
    if test_only and any(item.mode in _PRODUCTION_MODES for item in policy.entrypoints):
        raise SourceExecutionValidationError("production_policy_not_frozen")

    retained = _SnapshotSessionV1()
    guard: _PersistentImportGuardV1 | None = None
    loaded: tuple[str, ...] = ()
    try:
        with _authorized_open():
            dependency_by_path, dependency_by_logical = _pin_dependency_tree(
                lock, environment, retained
            )
            repository_by_path, repository_by_module = _pin_repository_sources(
                policy, repository_root, retained
            )
        source_by_name = {
            item.module_name: repository_by_module[item.module_name]
            for item in policy.business_modules
        }
        all_paths = dict(dependency_by_path)
        all_paths.update(repository_by_path)
        guard = _PersistentImportGuardV1(
            _safe_root(repository_root, "repository root"),
            _safe_root(environment.prefix, "dependency root"),
            all_paths,
            source_by_name,
        )
        _install_guard(guard)
        loaded, entrypoint_result = _execute_business_modules(policy, guard)
        closure = _build_closure(policy)
        session = object.__new__(AttestedExecutionSessionV1)
        object.__setattr__(session, "_closure", closure)
        object.__setattr__(session, "_entrypoint_result", entrypoint_result)
        object.__setattr__(session, "_guard", guard)
        object.__setattr__(session, "_pinned_session", retained)
        object.__setattr__(session, "_verified_content", verified)
        object.__setattr__(session, "_policy", policy)
        object.__setattr__(session, "_content_lock", lock)
        object.__setattr__(session, "_environment", environment)
        object.__setattr__(session, "_repository_root", Path(repository_root))
        object.__setattr__(session, "_source_snapshots", repository_by_module)
        object.__setattr__(session, "_dependency_snapshots", dependency_by_logical)
        object.__setattr__(session, "_loaded_module_names", loaded)
        object.__setattr__(session, "_test_only", test_only)
        _register_session(session)
        return session
    except Exception:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        if guard is not None:
            _remove_guard(guard)
        retained.close()
        raise


def establish_attested_execution(
    policy: SourceExecutionPolicyV1,
    content_lock: DependencyContentLockV1,
) -> AttestedExecutionSessionV1:
    """Perform Task 1's real full scan and establish a current-process session."""

    if type(policy) is not SourceExecutionPolicyV1:
        _fail("policy", "must be an exact SourceExecutionPolicyV1")
    if type(content_lock) is not DependencyContentLockV1:
        _fail("content_lock", "must be an exact DependencyContentLockV1")
    try:
        verified = verify_current_dependency_contents(content_lock)
        environment = _default_environment()
    except DependencyContentLockError as exc:
        raise SourceExecutionValidationError("dependency content verification failed") from exc
    repository_root = Path(__file__).resolve().parents[2]
    return _establish(
        policy,
        content_lock,
        environment,
        repository_root,
        verified,
        test_only=False,
    )


def _establish_attested_execution_for_test(
    policy: SourceExecutionPolicyV1,
    content_lock: DependencyContentLockV1,
    environment: _ScanEnvironmentV1,
    repository_root: Path,
) -> AttestedExecutionSessionV1:
    """Exercise the real verifier against synthetic roots; production modes fail."""

    if type(environment) is not _ScanEnvironmentV1 or not isinstance(repository_root, Path):
        raise TypeError("synthetic establishment requires exact private adapters")
    try:
        verified = _verify_dependency_contents_in_environment(
            content_lock, environment
        )
    except DependencyContentLockError as exc:
        raise SourceExecutionValidationError("dependency content verification failed") from exc
    return _establish(
        policy,
        content_lock,
        environment,
        repository_root,
        verified,
        test_only=True,
    )


def _rescan_sources(session: AttestedExecutionSessionV1) -> None:
    root = _safe_root(session._repository_root, "repository root")
    bindings = (*session._policy.bootstrap_modules, *session._policy.business_modules)
    with _authorized_open():
        for binding in bindings:
            path = root.joinpath(*binding.logical_path.split("/"))
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise SourceExecutionValidationError("source rescan unavailable") from exc
            if sha256_bytes(raw) != binding.raw_artifact_sha256:
                raise SourceExecutionValidationError("source changed before admission")
            if _tracked_sha256(raw) != binding.tracked_source_sha256:
                raise SourceExecutionValidationError("tracked source changed before admission")


def verify_execution_closure(
    closure: ExecutionClosureV1,
    policy: SourceExecutionPolicyV1,
    session: AttestedExecutionSessionV1,
) -> None:
    if type(closure) is not ExecutionClosureV1:
        _fail("closure", "must be an exact ExecutionClosureV1")
    if type(policy) is not SourceExecutionPolicyV1:
        _fail("policy", "must be an exact SourceExecutionPolicyV1")
    authentic = _require_session(session)
    expected = _build_closure(policy)
    if closure != expected or authentic._closure != expected:
        raise SourceExecutionValidationError("execution closure mismatch")


def issue_m2_admission(
    session: AttestedExecutionSessionV1,
) -> AttestedExecutionAdmissionV1:
    authentic = _require_session(session)
    try:
        authentic._pinned_session.validate()
        with _authorized_open():
            if authentic._test_only:
                verified = _verify_dependency_contents_in_environment(
                    authentic._content_lock, authentic._environment
                )
            else:
                verified = verify_current_dependency_contents(authentic._content_lock)
        _require_verified_dependency_content(verified)
    except DependencyContentLockError as exc:
        raise SourceExecutionValidationError("dependency content changed before admission") from exc
    _rescan_sources(authentic)
    verify_execution_closure(authentic._closure, authentic._policy, authentic)
    admission = object.__new__(AttestedExecutionAdmissionV1)
    object.__setattr__(admission, "_session_ref", weakref.ref(authentic))
    object.__setattr__(admission, "_guard_ref", weakref.ref(authentic._guard))
    object.__setattr__(admission, "_issuer_pid", os.getpid())
    _register_admission(admission, authentic)
    return admission


def require_attested_m2_admission(value: object) -> AttestedExecutionAdmissionV1:
    """Require one live, authentic, same-process Task 2 admission capability."""

    return _require_admission(value)


def _verified_resource_bytes(
    session: AttestedExecutionSessionV1,
    owner_or_module: str,
    logical_path: str,
) -> bytes:
    authentic = _require_session(session)
    _string(owner_or_module, "resource.owner_or_module")
    logical = _logical_source_path(logical_path, "resource.logical_path") if logical_path.endswith(".py") else _string(logical_path, "resource.logical_path")
    if "\\" in logical or logical.startswith(("/", "\\")) or any(
        part in ("", ".", "..") for part in logical.split("/")
    ):
        raise SourceExecutionValidationError("invalid verified resource path")
    snapshot = authentic._source_snapshots.get(owner_or_module)
    if snapshot is not None and snapshot.logical_path != logical:
        snapshot = None
    if snapshot is None:
        snapshot = authentic._dependency_snapshots.get(logical)
    if snapshot is None or type(snapshot.data) is not bytes:
        raise SourceExecutionValidationError("verified resource is unavailable")
    return snapshot.data


def _deactivate_attested_execution_for_test(
    session: AttestedExecutionSessionV1,
) -> None:
    if type(session) is not AttestedExecutionSessionV1:
        raise TypeError("test deactivation requires an exact session")
    if not getattr(session, "_test_only", False):
        raise SourceExecutionValidationError("production session cannot be deactivated")
    for name in reversed(session._loaded_module_names):
        sys.modules.pop(name, None)
    _remove_guard(session._guard)
    session._pinned_session.close()
    _unregister_session(session)


__all__ = (
    "AttestedExecutionAdmissionV1",
    "AttestedExecutionSessionV1",
    "BusinessModuleBindingV1",
    "EntrypointBindingV1",
    "ExecutionClosureV1",
    "SourceExecutionPolicyV1",
    "SourceExecutionValidationError",
    "establish_attested_execution",
    "issue_m2_admission",
    "require_attested_m2_admission",
    "source_execution_policy_from_payload",
    "source_execution_policy_hash",
    "verify_execution_closure",
)
