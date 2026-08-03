"""Strict semantic dependency lock schema and offline runtime verification."""

from __future__ import annotations

import _decimal
import decimal
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, NoReturn
import unicodedata

from binance_spot_strategy.protocols import numeric_context

from .digests import (
    hash_canonical_payload,
    require_sha256,
    sha256_raw_file,
    sha256_tracked_text,
)


class DependencyLockError(ValueError):
    """Raised when the semantic lock is malformed or runtime evidence drifts."""


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_LOCK_PATH = _PACKAGE_ROOT / "config" / "semantic_dependencies.lock.json"
_DESIGN_PATH = (
    _REPOSITORY_ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-03-binance-spot-dual-baseline-design.md"
)
_REQUIREMENTS_PATH = _PACKAGE_ROOT / "requirements.lock.txt"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "numeric_protocol_version",
        "design_revision_sha256",
        "requirements_lock_sha256",
        "source_hash_convention",
        "python",
        "float64",
        "decimal",
        "unicode",
        "distributions",
    }
)
_SOURCE_KEYS = frozenset(
    {"digest_encoding", "logical_path", "runtime_artifact", "tracked_source"}
)
_PYTHON_KEYS = frozenset(
    {
        "implementation",
        "version",
        "version_info",
        "cache_tag",
        "compiler",
        "build",
        "byteorder",
        "executable_sha256",
    }
)
_FLOAT64_KEYS = frozenset(
    {"radix", "mant_dig", "dig", "rounds", "max_exp", "min_exp"}
)
_DECIMAL_KEYS = frozenset(
    {
        "module_version",
        "libmpdec_version",
        "have_contextvar",
        "extension_sha256",
        "context",
    }
)
_CONTEXT_KEYS = frozenset(
    {"precision", "rounding", "emin", "emax", "capitals", "clamp", "traps"}
)
_UNICODE_KEYS = frozenset({"database_version", "normalization"})
_DISTRIBUTION_KEYS = frozenset({"name", "version", "record_sha256"})
_DISTRIBUTION_NAMES = (
    "numpy",
    "pandas",
    "python-dateutil",
    "six",
    "tzdata",
)
_SOURCE_CONVENTION = {
    "digest_encoding": "lowercase_hex",
    "logical_path": "repo_relative_posix",
    "runtime_artifact": "sha256_raw_bytes",
    "tracked_source": "sha256_git_blob_bytes",
}


def _schema_error(path: str, reason: str) -> NoReturn:
    raise DependencyLockError(f"invalid semantic dependency lock at {path}: {reason}")


def _require_exact_object(
    value: object, expected_keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict:
        _schema_error(path, "must be an object")
    assert isinstance(value, dict)
    if not all(type(key) is str for key in value):
        _schema_error(path, "object keys must be strings")
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        _schema_error(path, f"keys differ (missing={missing}, extra={extra})")
    return value


def _require_string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        _schema_error(path, "must be a nonempty string")
    assert isinstance(value, str)
    return value


def _require_integer(value: object, path: str) -> int:
    if type(value) is not int:
        _schema_error(path, "must be an integer")
    assert isinstance(value, int)
    return value


def _require_boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _schema_error(path, "must be a boolean")
    assert isinstance(value, bool)
    return value


def _require_list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        _schema_error(path, "must be an array")
    assert isinstance(value, list)
    return value


def _validate_digest(value: object, path: str) -> None:
    try:
        require_sha256(value, path)
    except (TypeError, ValueError) as exc:
        _schema_error(path, str(exc))


def _validate_semantic_dependency_lock(lock: object) -> dict[str, Any]:
    root = _require_exact_object(lock, _TOP_LEVEL_KEYS, "$.")
    schema_version = _require_string(root["schema_version"], "schema_version")
    if schema_version != "semantic_dependency_lock_v1":
        _schema_error("schema_version", "must equal semantic_dependency_lock_v1")
    numeric_protocol_version = _require_string(
        root["numeric_protocol_version"], "numeric_protocol_version"
    )
    if numeric_protocol_version != "numeric_protocol_v1":
        _schema_error("numeric_protocol_version", "must equal numeric_protocol_v1")
    _validate_digest(root["design_revision_sha256"], "design_revision_sha256")
    _validate_digest(root["requirements_lock_sha256"], "requirements_lock_sha256")

    source = _require_exact_object(
        root["source_hash_convention"], _SOURCE_KEYS, "source_hash_convention"
    )
    for field in sorted(_SOURCE_KEYS):
        _require_string(source[field], f"source_hash_convention.{field}")
    if source != _SOURCE_CONVENTION:
        _schema_error(
            "source_hash_convention", "must equal the source hash convention v1"
        )

    python = _require_exact_object(root["python"], _PYTHON_KEYS, "python")
    for field in ("implementation", "version", "cache_tag", "compiler", "byteorder"):
        _require_string(python[field], f"python.{field}")
    version_info = _require_list(python["version_info"], "python.version_info")
    if len(version_info) != 5:
        _schema_error("python.version_info", "must contain five values")
    for index in (0, 1, 2, 4):
        _require_integer(version_info[index], f"python.version_info[{index}]")
    _require_string(version_info[3], "python.version_info[3]")
    build = _require_list(python["build"], "python.build")
    if len(build) != 2:
        _schema_error("python.build", "must contain two values")
    for index, value in enumerate(build):
        _require_string(value, f"python.build[{index}]")
    _validate_digest(python["executable_sha256"], "python.executable_sha256")

    float64 = _require_exact_object(root["float64"], _FLOAT64_KEYS, "float64")
    for field in sorted(_FLOAT64_KEYS):
        _require_integer(float64[field], f"float64.{field}")

    decimal_lock = _require_exact_object(root["decimal"], _DECIMAL_KEYS, "decimal")
    _require_string(decimal_lock["module_version"], "decimal.module_version")
    _require_string(decimal_lock["libmpdec_version"], "decimal.libmpdec_version")
    _require_boolean(decimal_lock["have_contextvar"], "decimal.have_contextvar")
    _validate_digest(decimal_lock["extension_sha256"], "decimal.extension_sha256")
    context = _require_exact_object(
        decimal_lock["context"], _CONTEXT_KEYS, "decimal.context"
    )
    for field in ("precision", "emin", "emax", "capitals", "clamp"):
        _require_integer(context[field], f"decimal.context.{field}")
    _require_string(context["rounding"], "decimal.context.rounding")
    traps = _require_list(context["traps"], "decimal.context.traps")
    if not traps:
        _schema_error("decimal.context.traps", "must not be empty")
    for index, trap in enumerate(traps):
        _require_string(trap, f"decimal.context.traps[{index}]")
    if len(set(traps)) != len(traps):
        _schema_error("decimal.context.traps", "must not contain duplicates")

    unicode_lock = _require_exact_object(root["unicode"], _UNICODE_KEYS, "unicode")
    _require_string(unicode_lock["database_version"], "unicode.database_version")
    normalization = _require_string(
        unicode_lock["normalization"], "unicode.normalization"
    )
    if normalization != "NFC":
        _schema_error("unicode.normalization", "must equal NFC")

    distributions = _require_list(root["distributions"], "distributions")
    if len(distributions) != len(_DISTRIBUTION_NAMES):
        _schema_error(
            "distributions", "must contain the exact ordered distribution set"
        )
    actual_names: list[str] = []
    for index, distribution in enumerate(distributions):
        path = f"distributions[{index}]"
        item = _require_exact_object(distribution, _DISTRIBUTION_KEYS, path)
        actual_names.append(_require_string(item["name"], f"{path}.name"))
        _require_string(item["version"], f"{path}.version")
        _validate_digest(item["record_sha256"], f"{path}.record_sha256")
    if tuple(actual_names) != _DISTRIBUTION_NAMES:
        _schema_error(
            "distributions", "must contain the exact ordered distribution names"
        )
    return root


def _reject_json_number(_: str) -> NoReturn:
    raise DependencyLockError(
        "invalid semantic dependency lock JSON: floats are forbidden"
    )


def _reject_json_constant(_: str) -> NoReturn:
    raise DependencyLockError(
        "invalid semantic dependency lock JSON: nonfinite numbers are forbidden"
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DependencyLockError(
                f"invalid semantic dependency lock JSON: duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nulls(value: object, path: str = "$") -> None:
    if value is None:
        raise DependencyLockError(
            f"invalid semantic dependency lock JSON: null at {path}"
        )
    if type(value) is dict:
        assert isinstance(value, dict)
        for key, item in value.items():
            _reject_nulls(item, f"{path}.{key}")
    elif type(value) is list:
        assert isinstance(value, list)
        for index, item in enumerate(value):
            _reject_nulls(item, f"{path}[{index}]")


def load_semantic_dependency_lock(path: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly validate a semantic dependency lock JSON document."""

    lock_path = _DEFAULT_LOCK_PATH if path is None else Path(path)
    try:
        raw = lock_path.read_bytes()
    except OSError as exc:
        raise DependencyLockError(
            "invalid semantic dependency lock JSON: unable to read lock"
        ) from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DependencyLockError(
            "invalid semantic dependency lock JSON: UTF-8 BOM is forbidden"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DependencyLockError(
            "invalid semantic dependency lock JSON: invalid UTF-8"
        ) from exc
    try:
        lock = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_constant,
        )
    except DependencyLockError:
        raise
    except json.JSONDecodeError as exc:
        raise DependencyLockError(
            "invalid semantic dependency lock JSON: malformed document"
        ) from exc
    _reject_nulls(lock)
    return _validate_semantic_dependency_lock(lock)


def semantic_dependency_lock_hash(lock: object) -> str:
    """Validate and hash a lock through canonical JSON v1."""

    validated = _validate_semantic_dependency_lock(lock)
    return hash_canonical_payload(validated)


def _distribution_record_hash(distribution: Any) -> str | None:
    try:
        files = distribution.files
        if type(files) is not list:
            return None
        records = [
            item
            for item in files
            if str(item).replace("\\", "/").endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            return None
        record_path = distribution.locate_file(records[0])
        return sha256_raw_file(record_path)
    except Exception:
        return None


def _compare(
    drift_paths: set[str], path: str, expected: object, actual: object
) -> None:
    if actual != expected or type(actual) is not type(expected):
        drift_paths.add(path)


def verify_current_runtime(lock: object) -> None:
    """Fail closed if local semantic runtime evidence differs from the lock."""

    validated = _validate_semantic_dependency_lock(lock)
    drift_paths: set[str] = set()

    try:
        design_hash = sha256_tracked_text(_DESIGN_PATH)
    except (OSError, TypeError, ValueError):
        design_hash = None
    _compare(
        drift_paths,
        "design_revision_sha256",
        validated["design_revision_sha256"],
        design_hash,
    )
    try:
        requirements_hash = sha256_tracked_text(_REQUIREMENTS_PATH)
    except (OSError, TypeError, ValueError):
        requirements_hash = None
    _compare(
        drift_paths,
        "requirements_lock_sha256",
        validated["requirements_lock_sha256"],
        requirements_hash,
    )

    python_lock = validated["python"]
    python_actual = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info),
        "cache_tag": sys.implementation.cache_tag,
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
        "byteorder": sys.byteorder,
    }
    for field, actual in python_actual.items():
        _compare(drift_paths, f"python.{field}", python_lock[field], actual)
    try:
        executable_hash = sha256_raw_file(sys.executable)
    except (OSError, TypeError, ValueError):
        executable_hash = None
    _compare(
        drift_paths,
        "python.executable_sha256",
        python_lock["executable_sha256"],
        executable_hash,
    )

    float_lock = validated["float64"]
    float_actual = {
        "radix": sys.float_info.radix,
        "mant_dig": sys.float_info.mant_dig,
        "dig": sys.float_info.dig,
        "rounds": sys.float_info.rounds,
        "max_exp": sys.float_info.max_exp,
        "min_exp": sys.float_info.min_exp,
    }
    for field, actual in float_actual.items():
        _compare(drift_paths, f"float64.{field}", float_lock[field], actual)

    decimal_lock = validated["decimal"]
    _compare(
        drift_paths,
        "decimal.module_version",
        decimal_lock["module_version"],
        decimal.__version__,
    )
    _compare(
        drift_paths,
        "decimal.libmpdec_version",
        decimal_lock["libmpdec_version"],
        decimal.__libmpdec_version__,
    )
    _compare(
        drift_paths,
        "decimal.have_contextvar",
        decimal_lock["have_contextvar"],
        decimal.HAVE_CONTEXTVAR,
    )
    try:
        extension_hash = sha256_raw_file(_decimal.__file__)
    except (OSError, TypeError, ValueError):
        extension_hash = None
    _compare(
        drift_paths,
        "decimal.extension_sha256",
        decimal_lock["extension_sha256"],
        extension_hash,
    )
    context = numeric_context()
    context_actual = {
        "precision": context.prec,
        "rounding": context.rounding,
        "emin": context.Emin,
        "emax": context.Emax,
        "capitals": context.capitals,
        "clamp": context.clamp,
        "traps": sorted(
            signal.__name__
            for signal, enabled in context.traps.items()
            if enabled
        ),
    }
    context_lock = decimal_lock["context"]
    for field, actual in context_actual.items():
        _compare(
            drift_paths,
            f"decimal.context.{field}",
            context_lock[field],
            actual,
        )

    unicode_lock = validated["unicode"]
    _compare(
        drift_paths,
        "unicode.database_version",
        unicode_lock["database_version"],
        unicodedata.unidata_version,
    )
    _compare(
        drift_paths,
        "unicode.normalization",
        unicode_lock["normalization"],
        "NFC",
    )

    for distribution_lock in validated["distributions"]:
        name = distribution_lock["name"]
        version_path = f"distributions.{name}.version"
        record_path = f"distributions.{name}.record_sha256"
        try:
            installed = metadata.distribution(name)
        except Exception:
            drift_paths.update((version_path, record_path))
            continue
        try:
            installed_version = installed.version
        except Exception:
            installed_version = None
        _compare(
            drift_paths,
            version_path,
            distribution_lock["version"],
            installed_version,
        )
        record_hash = _distribution_record_hash(installed)
        _compare(
            drift_paths,
            record_path,
            distribution_lock["record_sha256"],
            record_hash,
        )

    if drift_paths:
        raise DependencyLockError(
            "semantic dependency drift: " + ", ".join(sorted(drift_paths))
        )


__all__ = (
    "DependencyLockError",
    "load_semantic_dependency_lock",
    "semantic_dependency_lock_hash",
    "verify_current_runtime",
)
