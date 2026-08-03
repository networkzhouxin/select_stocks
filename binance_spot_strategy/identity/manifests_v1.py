"""Strict synthetic candidate and run identity manifests for M1."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, NoReturn
import unicodedata

from binance_spot_strategy.protocols import Q18, format_q18, parse_canonical_q18

from .digests import hash_canonical_payload, require_sha256


class ManifestValidationError(ValueError):
    """Raised when a candidate or run manifest violates schema v1."""


_CONTRACT_ROLE_ORDER = (
    "strategy",
    "risk",
    "cost",
    "accounting",
    "metric",
    "selection",
)
_SHARED_MODULE_ROLE_ORDER = (
    "indicator",
    "strategy_support",
    "risk",
    "cost",
    "accounting",
    "metric",
    "selection",
)
_MODULE_ROLES = frozenset(
    (*_SHARED_MODULE_ROLE_ORDER, "baseline_strategy", "broker_adapter")
)
_SEMANTIC_VERSION_RE = re.compile(
    r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?\Z"
)
_SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "numeric_protocol_version",
        "design_revision_sha256",
        "baseline",
        "conventions",
        "contracts",
        "baseline_strategy_module",
        "shared_modules",
        "protocols",
    }
)
_BASELINE_KEYS = frozenset({"id", "semantic_version"})
_CONVENTION_KEYS = frozenset(
    {"universe", "bar_interval", "decision_timing", "formal_starting_balance"}
)
_CONTRACT_KEYS = frozenset({"role", "sha256"})
_MODULE_KEYS = frozenset({"role", "logical_name", "sha256"})
_PROTOCOL_KEYS = frozenset(
    {
        "canonical_json_version",
        "canonical_protocol_sha256",
        "numeric_protocol_sha256",
        "golden_vectors_sha256",
        "semantic_dependency_lock_sha256",
    }
)
_RUN_KEYS = frozenset(
    {
        "schema_version",
        "numeric_protocol_version",
        "candidate_strategy_fingerprint",
        "business_artifact_manifest_sha256",
        "business_source_tree_sha256",
        "source_commit",
        "dependency_lock_sha256",
        "runtime_manifest_sha256",
        "broker_adapter",
        "stage_manifest_sha256",
        "execution_manifest_sha256",
        "environment_contract_sha256",
        "input",
    }
)
_HISTORICAL_INPUT_KEYS = frozenset(
    {"kind", "immutable_raw_data_manifest_sha256"}
)
_PAPER_INPUT_KEYS = frozenset(
    {"kind", "paper_startup_semantic_hash", "feed_contract_sha256"}
)


def _validation_error(path: str, reason: str) -> NoReturn:
    raise ManifestValidationError(f"invalid identity manifest at {path}: {reason}")


def _require_exact_dict(
    value: object, expected_keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict:
        _validation_error(path, "must be an object")
    assert isinstance(value, dict)
    if not all(type(key) is str for key in value):
        _validation_error(path, "object keys must be strings")
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        _validation_error(path, f"keys differ (missing={missing}, extra={extra})")
    return value


def _require_list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        _validation_error(path, "must be an array")
    assert isinstance(value, list)
    return value


def _require_nonempty_string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        _validation_error(path, "must be a nonempty string")
    assert isinstance(value, str)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ManifestValidationError(
            f"invalid identity manifest at {path}: must contain only Unicode scalar values"
        ) from exc
    return value


def _require_constant(value: object, expected: str, path: str) -> str:
    actual = _require_nonempty_string(value, path)
    if actual != expected:
        _validation_error(path, f"must equal {expected}")
    return actual


def _require_digest(value: object, path: str) -> str:
    try:
        return require_sha256(value, path)
    except (TypeError, ValueError) as exc:
        _validation_error(path, str(exc))


def _require_source_commit(value: object, path: str) -> str:
    commit = _require_nonempty_string(value, path)
    if _SOURCE_COMMIT_RE.fullmatch(commit) is None:
        _validation_error(path, "must be 40 lowercase hexadecimal characters")
    return commit


def _require_logical_name(value: object, path: str) -> str:
    logical_name = _require_nonempty_string(value, path)
    if unicodedata.normalize("NFC", logical_name) != logical_name:
        _validation_error(path, "must already use NFC normalization")
    return logical_name


@dataclass(frozen=True, slots=True)
class ContractDigest:
    """One frozen semantic-contract digest in its declared role."""

    role: str
    sha256: str

    def __post_init__(self) -> None:
        role = _require_nonempty_string(self.role, "ContractDigest.role")
        if role not in _CONTRACT_ROLE_ORDER:
            _validation_error("ContractDigest.role", "is not a contract role")
        _require_digest(self.sha256, "ContractDigest.sha256")


@dataclass(frozen=True, slots=True)
class ModuleDigest:
    """One frozen logical module name and tracked-source digest."""

    role: str
    logical_name: str
    sha256: str

    def __post_init__(self) -> None:
        role = _require_nonempty_string(self.role, "ModuleDigest.role")
        if role not in _MODULE_ROLES:
            _validation_error("ModuleDigest.role", "is not a module role")
        _require_logical_name(self.logical_name, "ModuleDigest.logical_name")
        _require_digest(self.sha256, "ModuleDigest.sha256")


def _validate_contracts(contracts: object) -> tuple[ContractDigest, ...]:
    if type(contracts) is not tuple:
        _validation_error("CandidateManifestV1.contracts", "must be a tuple")
    assert isinstance(contracts, tuple)
    if len(contracts) != len(_CONTRACT_ROLE_ORDER):
        _validation_error(
            "CandidateManifestV1.contracts", "must contain exactly six entries"
        )
    if not all(type(item) is ContractDigest for item in contracts):
        _validation_error(
            "CandidateManifestV1.contracts", "must contain ContractDigest values"
        )
    roles = tuple(item.role for item in contracts)
    if roles != _CONTRACT_ROLE_ORDER:
        _validation_error(
            "CandidateManifestV1.contracts", "uses the wrong contract role order"
        )
    return contracts


def _validate_shared_modules(modules: object) -> tuple[ModuleDigest, ...]:
    if type(modules) is not tuple:
        _validation_error("CandidateManifestV1.shared_modules", "must be a tuple")
    assert isinstance(modules, tuple)
    if not all(type(item) is ModuleDigest for item in modules):
        _validation_error(
            "CandidateManifestV1.shared_modules", "must contain ModuleDigest values"
        )

    role_indexes = {role: index for index, role in enumerate(_SHARED_MODULE_ROLE_ORDER)}
    present_roles: set[str] = set()
    previous_role_index = -1
    previous_name_by_role: dict[str, str] = {}
    for index, module in enumerate(modules):
        if module.role not in role_indexes:
            _validation_error(
                f"CandidateManifestV1.shared_modules[{index}].role",
                "must be a shared module role",
            )
        role_index = role_indexes[module.role]
        if role_index < previous_role_index:
            _validation_error(
                "CandidateManifestV1.shared_modules",
                "must be grouped in shared module role order",
            )
        previous_role_index = role_index
        present_roles.add(module.role)
        previous_name = previous_name_by_role.get(module.role)
        if previous_name is not None and module.logical_name <= previous_name:
            _validation_error(
                "CandidateManifestV1.shared_modules",
                "logical names within a role must be unique and strictly ordered",
            )
        previous_name_by_role[module.role] = module.logical_name
    if present_roles != set(_SHARED_MODULE_ROLE_ORDER):
        _validation_error(
            "CandidateManifestV1.shared_modules",
            "must contain at least one module for every shared role",
        )
    return modules


@dataclass(frozen=True, slots=True)
class CandidateManifestV1:
    """Complete synthetic strategy-semantic identity input for M1."""

    schema_version: str
    numeric_protocol_version: str
    design_revision_sha256: str
    baseline_id: str
    baseline_semantic_version: str
    universe: tuple[str, ...]
    bar_interval: str
    decision_timing: str
    formal_starting_balance: Q18
    contracts: tuple[ContractDigest, ...]
    baseline_strategy_module: ModuleDigest
    shared_modules: tuple[ModuleDigest, ...]
    canonical_json_version: str
    canonical_protocol_sha256: str
    numeric_protocol_sha256: str
    golden_vectors_sha256: str
    semantic_dependency_lock_sha256: str

    def __post_init__(self) -> None:
        _require_constant(
            self.schema_version, "candidate_manifest_v1", "CandidateManifestV1.schema_version"
        )
        _require_constant(
            self.numeric_protocol_version,
            "numeric_protocol_v1",
            "CandidateManifestV1.numeric_protocol_version",
        )
        _require_digest(
            self.design_revision_sha256, "CandidateManifestV1.design_revision_sha256"
        )
        baseline_id = _require_nonempty_string(
            self.baseline_id, "CandidateManifestV1.baseline_id"
        )
        if baseline_id not in ("baseline_a", "baseline_b"):
            _validation_error(
                "CandidateManifestV1.baseline_id", "must be baseline_a or baseline_b"
            )
        semantic_version = _require_nonempty_string(
            self.baseline_semantic_version,
            "CandidateManifestV1.baseline_semantic_version",
        )
        if _SEMANTIC_VERSION_RE.fullmatch(semantic_version) is None:
            _validation_error(
                "CandidateManifestV1.baseline_semantic_version",
                "must be a semantic version",
            )
        if type(self.universe) is not tuple:
            _validation_error(
                "CandidateManifestV1.universe",
                "must be a tuple",
            )
        if not all(type(symbol) is str for symbol in self.universe):
            _validation_error(
                "CandidateManifestV1.universe", "must contain exact strings"
            )
        if self.universe != ("BTCUSDT", "ETHUSDT"):
            _validation_error(
                "CandidateManifestV1.universe",
                "must equal ('BTCUSDT', 'ETHUSDT')",
            )
        _require_constant(
            self.bar_interval, "4h", "CandidateManifestV1.bar_interval"
        )
        _require_constant(
            self.decision_timing,
            "closed_bar_only",
            "CandidateManifestV1.decision_timing",
        )
        if type(self.formal_starting_balance) is not Q18:
            _validation_error(
                "CandidateManifestV1.formal_starting_balance", "must be Q18"
            )
        if format_q18(self.formal_starting_balance) != "500.000000000000000000":
            _validation_error(
                "CandidateManifestV1.formal_starting_balance",
                "must equal canonical Q18 500",
            )
        _validate_contracts(self.contracts)
        if type(self.baseline_strategy_module) is not ModuleDigest:
            _validation_error(
                "CandidateManifestV1.baseline_strategy_module",
                "must be ModuleDigest",
            )
        if self.baseline_strategy_module.role != "baseline_strategy":
            _validation_error(
                "CandidateManifestV1.baseline_strategy_module.role",
                "must equal baseline_strategy",
            )
        _validate_shared_modules(self.shared_modules)
        _require_constant(
            self.canonical_json_version,
            "canonical_json_v1",
            "CandidateManifestV1.canonical_json_version",
        )
        for name in (
            "canonical_protocol_sha256",
            "numeric_protocol_sha256",
            "golden_vectors_sha256",
            "semantic_dependency_lock_sha256",
        ):
            _require_digest(getattr(self, name), f"CandidateManifestV1.{name}")

    def to_payload(self) -> dict[str, object]:
        """Return only the frozen business-semantic candidate fields."""

        return {
            "schema_version": self.schema_version,
            "numeric_protocol_version": self.numeric_protocol_version,
            "design_revision_sha256": self.design_revision_sha256,
            "baseline": {
                "id": self.baseline_id,
                "semantic_version": self.baseline_semantic_version,
            },
            "conventions": {
                "universe": list(self.universe),
                "bar_interval": self.bar_interval,
                "decision_timing": self.decision_timing,
                "formal_starting_balance": format_q18(self.formal_starting_balance),
            },
            "contracts": [
                {"role": item.role, "sha256": item.sha256}
                for item in self.contracts
            ],
            "baseline_strategy_module": {
                "role": self.baseline_strategy_module.role,
                "logical_name": self.baseline_strategy_module.logical_name,
                "sha256": self.baseline_strategy_module.sha256,
            },
            "shared_modules": [
                {
                    "role": item.role,
                    "logical_name": item.logical_name,
                    "sha256": item.sha256,
                }
                for item in self.shared_modules
            ],
            "protocols": {
                "canonical_json_version": self.canonical_json_version,
                "canonical_protocol_sha256": self.canonical_protocol_sha256,
                "numeric_protocol_sha256": self.numeric_protocol_sha256,
                "golden_vectors_sha256": self.golden_vectors_sha256,
                "semantic_dependency_lock_sha256": self.semantic_dependency_lock_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class RunCommonV1:
    """Frozen fields common to historical and paper run identities."""

    schema_version: str
    numeric_protocol_version: str
    candidate_strategy_fingerprint: str
    business_artifact_manifest_sha256: str
    business_source_tree_sha256: str
    source_commit: str
    dependency_lock_sha256: str
    runtime_manifest_sha256: str
    broker_adapter: ModuleDigest
    stage_manifest_sha256: str
    execution_manifest_sha256: str
    environment_contract_sha256: str

    def __post_init__(self) -> None:
        schema_version = _require_nonempty_string(
            self.schema_version, "RunCommonV1.schema_version"
        )
        if schema_version not in (
            "historical_run_manifest_v1",
            "paper_run_manifest_v1",
        ):
            _validation_error("RunCommonV1.schema_version", "is not a run schema v1")
        _require_constant(
            self.numeric_protocol_version,
            "numeric_protocol_v1",
            "RunCommonV1.numeric_protocol_version",
        )
        for name in (
            "candidate_strategy_fingerprint",
            "business_artifact_manifest_sha256",
            "business_source_tree_sha256",
            "dependency_lock_sha256",
            "runtime_manifest_sha256",
            "stage_manifest_sha256",
            "execution_manifest_sha256",
            "environment_contract_sha256",
        ):
            _require_digest(getattr(self, name), f"RunCommonV1.{name}")
        _require_source_commit(self.source_commit, "RunCommonV1.source_commit")
        if type(self.broker_adapter) is not ModuleDigest:
            _validation_error("RunCommonV1.broker_adapter", "must be ModuleDigest")
        if self.broker_adapter.role != "broker_adapter":
            _validation_error(
                "RunCommonV1.broker_adapter.role", "must equal broker_adapter"
            )

    def _common_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "numeric_protocol_version": self.numeric_protocol_version,
            "candidate_strategy_fingerprint": self.candidate_strategy_fingerprint,
            "business_artifact_manifest_sha256": self.business_artifact_manifest_sha256,
            "business_source_tree_sha256": self.business_source_tree_sha256,
            "source_commit": self.source_commit,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "broker_adapter": {
                "role": self.broker_adapter.role,
                "logical_name": self.broker_adapter.logical_name,
                "sha256": self.broker_adapter.sha256,
            },
            "stage_manifest_sha256": self.stage_manifest_sha256,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
        }


@dataclass(frozen=True, slots=True)
class HistoricalRunManifestV1(RunCommonV1):
    """Historical-run identity with an immutable raw-data manifest digest."""

    immutable_raw_data_manifest_sha256: str

    def __post_init__(self) -> None:
        RunCommonV1.__post_init__(self)
        _require_constant(
            self.schema_version,
            "historical_run_manifest_v1",
            "HistoricalRunManifestV1.schema_version",
        )
        _require_digest(
            self.immutable_raw_data_manifest_sha256,
            "HistoricalRunManifestV1.immutable_raw_data_manifest_sha256",
        )

    def to_payload(self) -> dict[str, object]:
        """Return only fields that define a historical run identity."""

        payload = self._common_payload()
        payload["input"] = {
            "kind": "historical",
            "immutable_raw_data_manifest_sha256": self.immutable_raw_data_manifest_sha256,
        }
        return payload


@dataclass(frozen=True, slots=True)
class PaperRunManifestV1(RunCommonV1):
    """Paper-run identity with startup and feed contract digests."""

    paper_startup_semantic_hash: str
    feed_contract_sha256: str

    def __post_init__(self) -> None:
        RunCommonV1.__post_init__(self)
        _require_constant(
            self.schema_version,
            "paper_run_manifest_v1",
            "PaperRunManifestV1.schema_version",
        )
        _require_digest(
            self.paper_startup_semantic_hash,
            "PaperRunManifestV1.paper_startup_semantic_hash",
        )
        _require_digest(
            self.feed_contract_sha256, "PaperRunManifestV1.feed_contract_sha256"
        )

    def to_payload(self) -> dict[str, object]:
        """Return only fields that define a paper run identity."""

        payload = self._common_payload()
        payload["input"] = {
            "kind": "paper",
            "paper_startup_semantic_hash": self.paper_startup_semantic_hash,
            "feed_contract_sha256": self.feed_contract_sha256,
        }
        return payload


def _parse_contract(value: object, path: str) -> ContractDigest:
    item = _require_exact_dict(value, _CONTRACT_KEYS, path)
    return ContractDigest(
        role=_require_nonempty_string(item["role"], f"{path}.role"),
        sha256=_require_digest(item["sha256"], f"{path}.sha256"),
    )


def _parse_module(value: object, path: str) -> ModuleDigest:
    item = _require_exact_dict(value, _MODULE_KEYS, path)
    return ModuleDigest(
        role=_require_nonempty_string(item["role"], f"{path}.role"),
        logical_name=_require_logical_name(
            item["logical_name"], f"{path}.logical_name"
        ),
        sha256=_require_digest(item["sha256"], f"{path}.sha256"),
    )


def candidate_manifest_from_payload(payload: object) -> CandidateManifestV1:
    """Parse and validate one exact candidate_manifest_v1 object."""

    root = _require_exact_dict(payload, _CANDIDATE_KEYS, "$")
    baseline = _require_exact_dict(root["baseline"], _BASELINE_KEYS, "baseline")
    conventions = _require_exact_dict(
        root["conventions"], _CONVENTION_KEYS, "conventions"
    )
    raw_universe = _require_list(conventions["universe"], "conventions.universe")
    universe = tuple(
        _require_nonempty_string(item, f"conventions.universe[{index}]")
        for index, item in enumerate(raw_universe)
    )
    raw_contracts = _require_list(root["contracts"], "contracts")
    contracts = tuple(
        _parse_contract(item, f"contracts[{index}]")
        for index, item in enumerate(raw_contracts)
    )
    raw_shared_modules = _require_list(root["shared_modules"], "shared_modules")
    shared_modules = tuple(
        _parse_module(item, f"shared_modules[{index}]")
        for index, item in enumerate(raw_shared_modules)
    )
    protocols = _require_exact_dict(
        root["protocols"], _PROTOCOL_KEYS, "protocols"
    )
    try:
        starting_balance = parse_canonical_q18(
            conventions["formal_starting_balance"]
        )
    except (TypeError, ValueError) as exc:
        _validation_error("conventions.formal_starting_balance", str(exc))

    return CandidateManifestV1(
        schema_version=_require_constant(
            root["schema_version"], "candidate_manifest_v1", "schema_version"
        ),
        numeric_protocol_version=_require_constant(
            root["numeric_protocol_version"],
            "numeric_protocol_v1",
            "numeric_protocol_version",
        ),
        design_revision_sha256=_require_digest(
            root["design_revision_sha256"], "design_revision_sha256"
        ),
        baseline_id=_require_nonempty_string(baseline["id"], "baseline.id"),
        baseline_semantic_version=_require_nonempty_string(
            baseline["semantic_version"], "baseline.semantic_version"
        ),
        universe=universe,
        bar_interval=_require_nonempty_string(
            conventions["bar_interval"], "conventions.bar_interval"
        ),
        decision_timing=_require_nonempty_string(
            conventions["decision_timing"], "conventions.decision_timing"
        ),
        formal_starting_balance=starting_balance,
        contracts=contracts,
        baseline_strategy_module=_parse_module(
            root["baseline_strategy_module"], "baseline_strategy_module"
        ),
        shared_modules=shared_modules,
        canonical_json_version=_require_nonempty_string(
            protocols["canonical_json_version"],
            "protocols.canonical_json_version",
        ),
        canonical_protocol_sha256=_require_digest(
            protocols["canonical_protocol_sha256"],
            "protocols.canonical_protocol_sha256",
        ),
        numeric_protocol_sha256=_require_digest(
            protocols["numeric_protocol_sha256"],
            "protocols.numeric_protocol_sha256",
        ),
        golden_vectors_sha256=_require_digest(
            protocols["golden_vectors_sha256"],
            "protocols.golden_vectors_sha256",
        ),
        semantic_dependency_lock_sha256=_require_digest(
            protocols["semantic_dependency_lock_sha256"],
            "protocols.semantic_dependency_lock_sha256",
        ),
    )


def _run_common_kwargs(root: dict[str, Any]) -> dict[str, object]:
    return {
        "schema_version": _require_nonempty_string(
            root["schema_version"], "schema_version"
        ),
        "numeric_protocol_version": _require_constant(
            root["numeric_protocol_version"],
            "numeric_protocol_v1",
            "numeric_protocol_version",
        ),
        "candidate_strategy_fingerprint": _require_digest(
            root["candidate_strategy_fingerprint"],
            "candidate_strategy_fingerprint",
        ),
        "business_artifact_manifest_sha256": _require_digest(
            root["business_artifact_manifest_sha256"],
            "business_artifact_manifest_sha256",
        ),
        "business_source_tree_sha256": _require_digest(
            root["business_source_tree_sha256"], "business_source_tree_sha256"
        ),
        "source_commit": _require_source_commit(
            root["source_commit"], "source_commit"
        ),
        "dependency_lock_sha256": _require_digest(
            root["dependency_lock_sha256"], "dependency_lock_sha256"
        ),
        "runtime_manifest_sha256": _require_digest(
            root["runtime_manifest_sha256"], "runtime_manifest_sha256"
        ),
        "broker_adapter": _parse_module(root["broker_adapter"], "broker_adapter"),
        "stage_manifest_sha256": _require_digest(
            root["stage_manifest_sha256"], "stage_manifest_sha256"
        ),
        "execution_manifest_sha256": _require_digest(
            root["execution_manifest_sha256"], "execution_manifest_sha256"
        ),
        "environment_contract_sha256": _require_digest(
            root["environment_contract_sha256"], "environment_contract_sha256"
        ),
    }


def run_manifest_from_payload(
    payload: object,
) -> HistoricalRunManifestV1 | PaperRunManifestV1:
    """Parse a run only when schema_version and input.kind form an exact pair."""

    root = _require_exact_dict(payload, _RUN_KEYS, "$")
    schema_version = _require_nonempty_string(
        root["schema_version"], "schema_version"
    )
    raw_input = root["input"]
    if type(raw_input) is not dict:
        _validation_error("input", "must be an object")
    assert isinstance(raw_input, dict)
    if not all(type(key) is str for key in raw_input):
        _validation_error("input", "object keys must be strings")
    if "kind" not in raw_input:
        _validation_error("input", "requires kind")
    kind = _require_nonempty_string(raw_input["kind"], "input.kind")
    pair = (schema_version, kind)
    common = _run_common_kwargs(root)
    if pair == ("historical_run_manifest_v1", "historical"):
        run_input = _require_exact_dict(
            raw_input, _HISTORICAL_INPUT_KEYS, "input"
        )
        return HistoricalRunManifestV1(
            **common,
            immutable_raw_data_manifest_sha256=_require_digest(
                run_input["immutable_raw_data_manifest_sha256"],
                "input.immutable_raw_data_manifest_sha256",
            ),
        )
    if pair == ("paper_run_manifest_v1", "paper"):
        run_input = _require_exact_dict(raw_input, _PAPER_INPUT_KEYS, "input")
        return PaperRunManifestV1(
            **common,
            paper_startup_semantic_hash=_require_digest(
                run_input["paper_startup_semantic_hash"],
                "input.paper_startup_semantic_hash",
            ),
            feed_contract_sha256=_require_digest(
                run_input["feed_contract_sha256"], "input.feed_contract_sha256"
            ),
        )
    _validation_error(
        "(schema_version,input.kind)",
        "must be the historical/historical or paper/paper v1 pair",
    )


def candidate_strategy_fingerprint(manifest: CandidateManifestV1) -> str:
    """Hash exactly one validated CandidateManifestV1 payload."""

    if type(manifest) is not CandidateManifestV1:
        raise TypeError("candidate fingerprint requires exact CandidateManifestV1")
    return hash_canonical_payload(CandidateManifestV1.to_payload(manifest))


def run_fingerprint(
    manifest: HistoricalRunManifestV1 | PaperRunManifestV1,
) -> str:
    """Hash exactly one validated historical or paper run manifest payload."""

    if type(manifest) is HistoricalRunManifestV1:
        return hash_canonical_payload(HistoricalRunManifestV1.to_payload(manifest))
    if type(manifest) is PaperRunManifestV1:
        return hash_canonical_payload(PaperRunManifestV1.to_payload(manifest))
    raise TypeError(
        "run fingerprint requires exact HistoricalRunManifestV1 or PaperRunManifestV1"
    )


__all__ = (
    "CandidateManifestV1",
    "ContractDigest",
    "HistoricalRunManifestV1",
    "ManifestValidationError",
    "ModuleDigest",
    "PaperRunManifestV1",
    "RunCommonV1",
    "candidate_manifest_from_payload",
    "candidate_strategy_fingerprint",
    "run_fingerprint",
    "run_manifest_from_payload",
)
