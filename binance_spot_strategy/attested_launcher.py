"""Direct-source isolated launcher for the Binance Spot M2 attestation gate.

Task 2 freezes the bootstrap mechanics but intentionally has no production
module policy.  Task 9 supplies the reviewed module lock and enables the two
reserved production modes.  Until then both modes fail closed before package
or business imports.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import sys
from types import MappingProxyType


_ALLOWED_MODES = frozenset({"verify-m2", "seal-historical-data"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class _BootstrapFailure(RuntimeError):
    pass


def _fail(code: str) -> int:
    sys.stderr.write(code + "\n")
    return 2


def _require_isolated_runtime() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
    ):
        raise _BootstrapFailure("isolated_runtime_required")
    sys.dont_write_bytecode = True
    if not sys.dont_write_bytecode:
        raise _BootstrapFailure("isolated_runtime_required")


def _read_compile_exec_once(
    repository_root: Path,
    logical_path: str,
    expected_raw_sha256: str,
    private_name: str,
) -> MappingProxyType:
    """Verify, compile, and execute one reviewed helper from one byte buffer.

    This narrow primitive is intentionally not reached before Task 9 supplies
    the final reviewed policy.  It never imports the helper by its normal name.
    """

    if type(logical_path) is not str or type(expected_raw_sha256) is not str:
        raise _BootstrapFailure("bootstrap_policy_invalid")
    if _SHA256_RE.fullmatch(expected_raw_sha256) is None:
        raise _BootstrapFailure("bootstrap_policy_invalid")
    parts = logical_path.split("/")
    if (
        not logical_path.endswith(".py")
        or "\\" in logical_path
        or logical_path.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise _BootstrapFailure("bootstrap_policy_invalid")
    root = repository_root.resolve(strict=True)
    path = root.joinpath(*parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        with resolved.open("rb", buffering=0) as stream:
            raw = stream.read()
    except (OSError, ValueError) as exc:
        raise _BootstrapFailure("bootstrap_source_unavailable") from exc
    if sha256(raw).hexdigest() != expected_raw_sha256:
        raise _BootstrapFailure("bootstrap_source_digest_mismatch")
    namespace = {
        "__name__": private_name,
        "__file__": logical_path,
        "__package__": "",
        "__builtins__": __builtins__,
    }
    code = compile(raw, logical_path, "exec", dont_inherit=True)
    exec(code, namespace)
    return MappingProxyType(namespace)


def main(argv: tuple[str, ...] | None = None) -> int:
    try:
        _require_isolated_runtime()
    except _BootstrapFailure as exc:
        return _fail(str(exc))
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or type(arguments[0]) is not str:
        return _fail("unsupported_mode")
    mode = arguments[0]
    if mode not in _ALLOWED_MODES:
        return _fail("unsupported_mode")
    # The final source policy and both literal entrypoint bindings are frozen
    # only after Tasks 3-8 exist.  Issuing a session/admission earlier would
    # turn a caller-supplied label into evidence, so Task 2 always stops here.
    return _fail("production_policy_not_frozen")


if __name__ == "__main__":
    raise SystemExit(main())
