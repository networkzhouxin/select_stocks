"""Deterministic identity and runtime-evidence interfaces."""

from .dependency_lock_v1 import (
    DependencyLockError,
    load_semantic_dependency_lock,
    semantic_dependency_lock_hash,
    verify_current_runtime,
)
from .digests import (
    hash_canonical_payload,
    require_sha256,
    sha256_bytes,
    sha256_raw_file,
    sha256_tracked_text,
)

__all__ = (
    "DependencyLockError",
    "hash_canonical_payload",
    "load_semantic_dependency_lock",
    "require_sha256",
    "semantic_dependency_lock_hash",
    "sha256_bytes",
    "sha256_raw_file",
    "sha256_tracked_text",
    "verify_current_runtime",
)
