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
from .manifests_v1 import (
    CandidateManifestV1,
    ContractDigest,
    HistoricalRunManifestV1,
    ManifestValidationError,
    ModuleDigest,
    PaperRunManifestV1,
    RunCommonV1,
    candidate_manifest_from_payload,
    candidate_strategy_fingerprint,
    run_fingerprint,
    run_manifest_from_payload,
)

__all__ = (
    "CandidateManifestV1",
    "ContractDigest",
    "DependencyLockError",
    "HistoricalRunManifestV1",
    "ManifestValidationError",
    "ModuleDigest",
    "PaperRunManifestV1",
    "RunCommonV1",
    "candidate_manifest_from_payload",
    "candidate_strategy_fingerprint",
    "hash_canonical_payload",
    "load_semantic_dependency_lock",
    "require_sha256",
    "run_fingerprint",
    "run_manifest_from_payload",
    "semantic_dependency_lock_hash",
    "sha256_bytes",
    "sha256_raw_file",
    "sha256_tracked_text",
    "verify_current_runtime",
)
