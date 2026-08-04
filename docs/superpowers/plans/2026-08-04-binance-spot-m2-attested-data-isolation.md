# Binance Spot M2 Attested Data Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the M1 source-execution and dependency-content evidence gap, then build an immutable, content-addressed Binance Spot four-hour data custody path whose only M2 release path is the exact training warm-up and performance dataset.

**Architecture:** M2 has two sequential gates. The attestation gate proves installed dependency payloads and the exact loaded business-source closure under an isolated launcher. The data gate records official public REST responses before parsing, normalizes each closed kline into an immutable object, seals nonrevealing stage manifests, releases only a physically stage-scoped training bundle, and composes validation/holdout access with a deny-all authority until later activation milestones exist.

**Tech Stack:** CPython 3.13.6, standard-library `unittest`, `urllib.request`, `importlib.metadata`, `csv`, `Decimal`, immutable dataclasses, SHA-256, UTF-8 canonical JSON v1, NumPy 2.2.6, pandas 3.0.1.

## Scope Decomposition

M2 produces one independently testable offline milestone:

1. M2.0 attested runtime: verify every behavior-capable distribution payload named by the locked `RECORD` files and prove the exact business modules loaded from repository source.
2. M2.1 immutable contracts and storage: strict raw-response, closure-evidence, bar-reference, raw-data-manifest, and stage-manifest protocols plus a create-only object store.
3. M2.2 public REST custody adapter: a no-key, allowlisted market-data client tested only through injected transports.
4. M2.3 preregistration sealer: full structural audit of both symbols with no market-path-derived output.
5. M2.4 stage isolation: a successful training-only loader and fail-before-read validation/holdout denial.

M2.5 is an operational gate, not part of the authorized implementation run. The implementation may create the public-data custody entrypoint, but nobody may execute it against Binance or any real data root until the user separately approves an absolute data root and the one-time custody run.

The following remain outside M2: indicators, Baseline A/B, sizing, ATR state, matching, fees, backtests, SQLite, metrics, reports, validation/holdout token transactions, nomination, paper feeds, API keys, private endpoints, order placement, and real trading.

## Official Interface Recheck

The implementation plan was rechecked on 2026-08-04 against Binance's official Spot documentation:

- Public market-data-only REST origin: `https://data-api.binance.vision`.
- Allowed endpoints in M2: `GET /api/v3/time` and `GET /api/v3/klines` only.
- Kline request: exact `symbol`, `interval=4h`, `startTime`, `endTime`, `timeZone=0`, and `limit=1000` query parameters.
- Timestamps use Binance's default millisecond representation; do not send `X-MBX-TIME-UNIT`.
- Klines are uniquely identified by open time; the response supplies raw open time, OHLCV strings, and close time.
- Public market-data-only URLs require no API key. M2 must reject authorization headers, signatures, account endpoints, and endpoint overrides.

References:

- `https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md`
- `https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md`

## File Structure

```text
binance_spot_strategy/
|-- README.md                                      # Update M2 scope and gates
|-- attested_launcher.py                          # Direct-source isolated launcher
|-- config/
|   |-- dependency_contents.lock.json             # Runtime/stdlib/distribution bytes
|   `-- m2_business_modules.lock.json             # M2 loaded-source allowlist/digests
|-- data/
|   |-- __init__.py                               # Narrow public data interfaces
|   |-- http_records_v1.py                        # Immutable request/receipt evidence
|   |-- transport_v1.py                           # Allowlisted public REST I/O only
|   |-- raw_archive_v1.py                         # Create-only raw-byte archive
|   |-- kline_codec_v1.py                         # Exact server-time/kline decoding
|   |-- acquisition_v1.py                         # Frozen paging and causal collection
|   |-- raw_manifest_v1.py                        # Complete attempt/page input closure
|   |-- stage_manifest_v1.py                      # Nonrevealing stage refs and hashes
|   |-- preregistration_sealer_v1.py              # Full-custody structural audit
|   |-- stage_access_v1.py                        # Trusted reserved-stage authority port
|   `-- stage_loader_v1.py                        # Released-view snapshot loading
|-- entrypoints/
|   |-- __init__.py
|   `-- seal_historical_data.py                   # Public-data custody CLI; never auto-run
`-- identity/
    |-- __init__.py                               # Export M2 attestation interfaces
    |-- dependency_contents_v1.py                 # Complete runtime/dependency content lock
    `-- source_execution_v1.py                    # Controlled loaded-source/artifact proof

tests/binance_spot/
|-- fixtures/
|   |-- m2_business_artifact_v1.json
|   |-- m2_data_manifests_v1.json
|   `-- raw_api/
|       |-- klines_page.json
|       `-- server_time.json
|-- test_acquisition.py
|-- test_binance_public_transport.py
|-- test_data_contracts.py
|-- test_dependency_contents.py
|-- test_kline_codec.py
|-- test_m2_acceptance.py
|-- test_raw_archive.py
|-- test_source_execution.py
|-- test_stage_access.py
|-- test_stage_loader.py
`-- test_stage_sealer.py
```

No production cache, downloaded response, generated bar object, stage bundle, secret, or absolute data-root path may be committed to Git.

## Global Constraints

- Execute implementation in `D:/test/select_stocks/.worktrees/binance-spot-m2-data-isolation` on branch `codex/binance-spot-m2-data-isolation`, created only after this plan is approved and only from baseline `d14d0ac834c13e98873dd42fe76c735be9026d85`.
- Before Task 1, verify that exact HEAD and a clean tracked baseline. Any drift requires a new read-only review; never let a changed checkout be silently absorbed into generated locks.
- Preserve the approved design identity `18adf5fd56177354e5e4194a416ae1c9de5a4371ae6c6872783cb08d7a3026a0` and canonical/numeric protocol versions from M1.
- Preserve CPython 3.13.6, Decimal 1.70/libmpdec 4.0.0, Unicode 15.1.0, NumPy 2.2.6, pandas 3.0.1, python-dateutil 2.9.0.post0, six 1.17.0, and tzdata 2025.3.
- Use exactly `D:/Programs/Python/Python313/python.exe -B -m unittest`; never call bare `python`, never install pytest, and never treat the unrelated pytest-style suite as executed.
- Write every behavioral test before production code, record the expected RED cause, implement the minimum GREEN behavior, rerun all prior Binance Spot tests, review, then commit the task separately.
- Normal and focused tests must not open a socket, contact Binance, read a G-drive path, inspect an external market-data root, or import `cross_signal_strategy`.
- The future custody CLI has no default data root. It requires an absolute repository-external path and rejects repository roots, relative paths, symlinks, junctions/reparse points, credentials, endpoint overrides, and pre-existing unknown layout files.
- The object store exposes no overwrite, update, rename-over-existing, cleanup, or delete API. Existing digest paths are accepted only when their bytes and sizes already match.
- Official data collection is sequential and fail-fast. M2 performs no automatic HTTP retry; an operator may rerun later because persisted objects and receipts are idempotent.
- Use Binance server time obtained before each kline request as closure evidence. A bar is eligible only when `close_time_ms < verified_server_time_ms`; equality fails. Local Windows time is audit metadata only.
- Freeze one four-hour interval as `14_400_000` milliseconds. Every open is aligned to UTC hours 00/04/08/12/16/20, and `close_time_ms == open_time_ms + 14_400_000 - 1`.
- Warm-up means exactly 540 paired BTC/ETH buckets immediately preceding each performance window: 540 bars per symbol, not 540 bars total.
- Expected performance bucket counts are training 8,766, validation 4,380, and holdout 5,658. Every bucket contains exactly one BTCUSDT bar and one ETHUSDT bar with identical open/close times.
- Stage windows remain half-open and fixed. If common coverage fails, emit only the allowed structural failure and stop for a separately approved design/config revision; never shorten, forward-fill, splice, or silently change a window.
- `stage_manifest_sha256` binds the stage contract, windows, balance, and ordered bucket references. `immutable_raw_data_manifest_sha256` binds only that stage's warm-up plus performance input closure. Neither digest represents the all-stage custody catalog.
- Validation warm-up overlaps the end of training and holdout warm-up overlaps the end of validation. Isolation protects stage-labelled access and reserved performance content; it does not falsely claim overlapping historical bars were never present in an earlier released stage.
- The sealer may import config, domain, protocols, identity, and data support only. It must not import strategies, indicators, risk, backtest, state, reporting, metrics, or presentation code.
- The custody sealer may persist full ordered identities/checksums and all stage refs only inside custody. Its public result may return structural status, common coverage bounds/counts, and training raw/stage/catalog/release-record hashes only. It must not return or log OHLCV, validation/holdout refs, returns, volatility, indicators, signals, ranks, metrics, paths, or any other market-path-derived value.
- Training is the only successful M2 loader composition. Validation/holdout use a trusted `ReservedStageAccessAuthority` port whose M2 production implementation is deny-all. Callers cannot authorize access with a boolean, token string, `StageActivationId`, fake hash, or injected public constructor.
- Authorization occurs before catalog lookup, cache lookup, file open, SQL/query, mmap, decompression, iteration, or parsing. All bytes are read, hashed, parsed, and validated as one snapshot before any immutable result is returned; never yield partial results.
- Do not create a formal candidate fingerprint, run fingerprint, activation token, nomination, validation result, holdout result, training metric, backtest, or paper state in M2. Synthetic golden hashes remain explicitly synthetic.
- Do not execute the real custody entrypoint during implementation, testing, review, or merge. A separate user authorization is mandatory after the offline M2 milestone passes.
- Treat transport, acquisition, codecs, raw manifests, the preregistration sealer, and the custody CLI as one custody TCB. Every module in that TCB receives the same strategy/indicator/metric/reporting import denylist and the same no-OHLCV-output tests; production codecs require an opaque custody capability and are not made safe merely by omission from `__all__`.
- A released stage view contains byte-copied, stage-local canonical bar records, its own manifests/catalog, one release record, and one root pointer only. It never contains or links to raw HTTP bodies, page receipts, acquisition-chain records, the custody catalog, another stage object, or any path/capability that can resolve those objects.
- Pin both external roots by Windows volume serial plus file ID, retain directory handles, revalidate every component before writes/publication, and claim a single-owner lock before transport construction. Release uses exclusive byte copies only; symlink, junction, mount/reparse, hardlink, reflink, alias, nested-root, and concurrent-owner paths fail closed.
- Public sealer/CLI output is a positive allowlist: structural status, common bounds/counts, and training stage/raw/catalog/release-record hashes only. Validation/holdout hashes, refs, object keys, relative/absolute paths, cache keys, exception text, `repr`, `args`, `__cause__`, and `__context__` never cross the custody capability.


---

### Task 1: Freeze And Verify Complete Runtime/Dependency Content

**Files:**
- Create: `binance_spot_strategy/config/dependency_contents.lock.json`
- Create: `binance_spot_strategy/identity/dependency_contents_v1.py`
- Modify: `binance_spot_strategy/identity/__init__.py`
- Create: `tests/binance_spot/test_dependency_contents.py`

**Interfaces:**
- Consumes: `load_semantic_dependency_lock()`, the exact CPython base prefix/stdlib roots, and the five locked installed distributions.
- Produces: strict immutable `ContentFileV1`, `ContentTreeV1`, and `DependencyContentLockV1`, plus runtime-only opaque `VerifiedContentTreeV1`/`VerifiedDependencyContentV1` issued only by the real verifier. Verified values have no public constructor/parser/serializer/pickle/reduce/equality-by-fields or cross-process representation.
- Produces: `load_dependency_content_lock(path: str | Path | None = None) -> DependencyContentLockV1`.
- Produces: `dependency_content_lock_hash(lock: DependencyContentLockV1) -> str`.
- Produces: `verify_current_dependency_contents(lock: DependencyContentLockV1) -> VerifiedDependencyContentV1`.
- Produces private audit API `generate_dependency_content_lock(output_path: Path) -> DependencyContentLockV1`; it accepts one explicit nonexistent output and is never exported from `identity.__init__`.
- Keeps `semantic_dependency_lock_v1` and all M1 golden hashes unchanged; the new lock composes with M1 rather than changing its schema.

The canonical content record is exact and contains no machine root:

```python
@dataclass(frozen=True, slots=True)
class ContentFileV1:
    logical_path: str
    artifact_kind: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True, slots=True)
class ContentTreeV1:
    owner_kind: str
    owner_name: str
    owner_version: str
    files: tuple[ContentFileV1, ...]
```

Owner order is exact: CPython runtime tree, CPython standard library excluding `site-packages`, then normalized distribution names in semantic-lock order. The runtime tree is every regular file under the canonical `sys.base_prefix` outside the canonical stdlib and `site-packages` subtrees, excluding only `__pycache__`, `.pyc`, and `.pyo`; this necessarily includes `python.exe`, `pythonw.exe` when present, `python3.dll`, `python313.dll`, `DLLs/` including stdlib `.pyd` files, runtime-owned support DLLs, `pyvenv.cfg`, and every other prefix-owned regular file. The stdlib tree is every regular file under its canonical root with the same cache exclusion and with `site-packages` removed.

`artifact_kind` is one of `runtime_binary`, `source`, `extension`, or `data`. Every listed file, including RECORD-self and generated cache entries present in a distribution inventory, receives a freshly computed raw SHA-256 and exact byte size in the content lock even when its raw RECORD row legitimately omits both fields. Windows system DLLs outside `sys.base_prefix` are the declared OS trust boundary, not silently classified as runtime content.

- [ ] **Step 1: Write failing schema, path, and canonical-hash tests**

Create `tests/binance_spot/test_dependency_contents.py` with a small checked-in synthetic payload and temporary trees. Reject missing/unknown keys, bool-as-int, str/digest subclasses, JSON null/float/nonfinite, duplicate keys, BOM, invalid UTF-8, lone surrogates, duplicate or unsorted files, NFC/casefold path collisions, forged/subclassed/pickled verified values, and a verified receipt from another process.

Canonical content-lock logical paths reject absolute roots, drives, UNC prefixes, backslashes, empty/`.`/`..` segments. Raw distribution RECORD paths are a separate input grammar: controlled `..` is accepted only when handle-based resolution from the exact owning distribution installation root (the parent containing its `.dist-info` directory and used by `Distribution.locate_file`) remains inside pinned `sys.base_prefix`. Convert the final file identity to one prefix-relative, forward-slash, no-`..` canonical logical path before hashing. Prove insertion order/hash seed stability and one-field hash mutation.

- [ ] **Step 2: Run schema tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_dependency_contents.py" -v
```

Expected: import error for missing `binance_spot_strategy.identity.dependency_contents_v1`.

- [ ] **Step 3: Write failing filesystem and RECORD verification tests**

Use temporary synthetic runtime, stdlib, `.dist-info`, package, and prefix-level `Scripts` roots. Include NumPy-style `../../Scripts/f2py.exe` and `../../Scripts/numpy-config.exe` relative to the owning distribution installation root: both resolve inside the prefix, canonicalize without `..`, and belong only to NumPy. Mutate a `.py`, `.pyd`, script, or data file while preserving size/mtime; add/delete/rename; use symlink/junction/hardlink/root escape; or swap bytes. Every invalid case fails without absolute-path leakage.

Strictly parse each distribution RECORD as three CSV columns. Hash and size must either both be empty where RECORD permits omission or both be present; present hashes must be canonical URL-safe SHA-256 and present sizes canonical integers, and both must match the one current snapshot. Reject one-sided omission, duplicate/NFC-colliding canonical file identities, malformed/noncanonical URL-safe base64, non-SHA-256 algorithms, bool/noncanonical sizes, final prefix escape, multiple distribution owners, and RECORD/content-lock mismatch. Precompute distribution-owned file IDs from all five resolved RECORDs; remove them from runtime/stdlib catch-all ownership. Every included file has exactly one owner, while RECORD and content lock cross-bind the same bytes.

- [ ] **Step 4: Implement strict content-lock parsing and current-tree verification**

The default production verifier derives roots from the running interpreter, `sys.base_prefix`, `sysconfig`, and `importlib.metadata`; callers cannot substitute a root. It verifies the complete runtime, stdlib, and five-distribution sets above and only then returns `VerifiedDependencyContentV1`. Test-only private adapters may supply temporary roots.

For each distribution, require exact ownership of every resolved RECORD file plus undeclared-file rejection under its unambiguous package/`.dist-info` roots. Ambiguous namespace or cross-distribution ownership fails. After distribution ownership is frozen, enumerate runtime/stdlib catch-all files excluding those file IDs; reject symlink/reparse/hardlink alias, root/file-ID change, unowned executable, duplicate owner, or uncovered controlled file. Hash each snapshot once. Only this real scan can mint opaque `VerifiedDependencyContentV1`.

The checked-in lock payload has these top-level fields:

```python
{
    "schema_version": "dependency_content_lock_v1",
    "numeric_protocol_version": "numeric_protocol_v1",
    "semantic_dependency_lock_sha256": semantic_lock_hash,
    "trees": [tree.to_payload() for tree in trees],
}
```

The generator writes with exclusive create. Runtime verification never rewrites or updates the lock.

- [ ] **Step 5: Generate and review the current-machine candidate lock**
Use this exact reviewed generation command from the clean baseline:

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m binance_spot_strategy.identity.dependency_contents_v1 generate --output binance_spot_strategy/config/dependency_contents.lock.json
```


Run the exact frozen interpreter once to write `binance_spot_strategy/config/dependency_contents.lock.json`. Review owner roots, file counts, path ordering, artifact kinds, all distribution RECORD ownership, and the exclusion list. Reject the environment and stop if editable installs, ambiguous namespace ownership, unexplained unhashed files, root escapes, or unowned executable artifacts are found; do not weaken the contract to make generation pass.

Then run `verify_current_dependency_contents(load_dependency_content_lock())` and assert repeatable verification of CPython runtime, stdlib, NumPy, pandas, python-dateutil, six, and tzdata.

- [ ] **Step 6: Run complete regression and static checks**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_dependency_contents.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
```

Expected: all prior 118 M1 tests plus Task 1 tests pass; installed runtime/dependency files remain untouched.

- [ ] **Step 7: Commit Task 1**

```powershell
git add binance_spot_strategy/config/dependency_contents.lock.json binance_spot_strategy/identity tests/binance_spot/test_dependency_contents.py
git commit -m "feat(binance-spot): lock runtime dependency contents"
```

### Task 2: Same-Process Controlled Source Execution And Artifact Closure

**Files:**
- Create: `binance_spot_strategy/attested_launcher.py`
- Create: `binance_spot_strategy/identity/source_execution_v1.py`
- Modify: `binance_spot_strategy/identity/__init__.py`
- Create: `tests/binance_spot/test_source_execution.py`

**Interfaces:**
- Consumes: Task 1 `DependencyContentLockV1` plus an exact `SourceExecutionPolicyV1`.
- Produces: immutable `BusinessModuleBindingV1`, `SourceExecutionPolicyV1`, and stable `ExecutionClosureV1`.
- Produces: runtime-only opaque `AttestedExecutionSessionV1` and `AttestedExecutionAdmissionV1`; neither has a public constructor, parser, serializer, equality-by-fields, pickle/reduce hook, or cross-process representation.
- Produces: `source_execution_policy_from_payload(payload: object) -> SourceExecutionPolicyV1` and `source_execution_policy_hash(policy: SourceExecutionPolicyV1) -> str`.
- Produces: `establish_attested_execution(policy: SourceExecutionPolicyV1, content_lock: DependencyContentLockV1) -> AttestedExecutionSessionV1`; it internally performs the real full scan and accepts no caller-supplied verified value.
- Produces: `issue_m2_admission(session: AttestedExecutionSessionV1) -> AttestedExecutionAdmissionV1`.
- Produces: `verify_execution_closure(closure: ExecutionClosureV1, policy: SourceExecutionPolicyV1, session: AttestedExecutionSessionV1) -> None`.
- Produces: `require_attested_m2_admission(value: object) -> AttestedExecutionAdmissionV1`; M1 synthetic candidate/run objects and supplied digest labels are rejected.

The stable module and closure fields are:

```python
@dataclass(frozen=True, slots=True)
class BusinessModuleBindingV1:
    role: str
    module_name: str
    logical_path: str
    tracked_source_sha256: str
    raw_artifact_sha256: str

@dataclass(frozen=True, slots=True)
class EntrypointBindingV1:
    mode: str
    module_name: str
    callable_name: str

@dataclass(frozen=True, slots=True)
class SourceExecutionPolicyV1:
    schema_version: str
    dependency_content_lock_sha256: str
    bootstrap_launcher_raw_sha256: str
    bootstrap_modules: tuple[BusinessModuleBindingV1, ...]
    entrypoints: tuple[EntrypointBindingV1, ...]
    business_modules: tuple[BusinessModuleBindingV1, ...]
    dependency_owner_tree_sha256s: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ExecutionClosureV1:
    schema_version: str
    execution_policy_sha256: str
    dependency_content_lock_sha256: str
    business_source_tree_sha256: str
    business_artifact_manifest_sha256: str
    business_modules: tuple[BusinessModuleBindingV1, ...]
```

Absolute paths, PID, hostname, scan time, temporary cache roots, random IDs, and presentation-only code are audit metadata and never enter the stable closure.

`AttestedExecutionSessionV1` owns the actual `VerifiedDependencyContentV1`, stable closure, live persistent import guard, retained native-file handles, current process identity, and an unexported issuer capability. `AttestedExecutionAdmissionV1` is minted only from that live session and refers back to its exact issuer/guard identity. Stable closure fields remain serializable evidence; session/admission capabilities do not.

Dependency admission mode is frozen to complete verified-owner trees: a dependency source, native module, or package-data file is admissible only when its exact file identity, logical path, size, and digest are entries in one of `dependency_owner_tree_sha256s`. Task 1 continues to verify the complete seven-owner content lock, including pandas, but the M2 execution policy admits only the exact ordered runtime, stdlib, NumPy, python-dateutil, six, and tzdata owner trees. The pandas owner tree is deliberately excluded from `dependency_owner_tree_sha256s`; `pandas`, every `pandas.*` descendant, `pyarrow`, and every `pyarrow.*` descendant must fail before resolver delegation, byte reads, module creation, or native loading. This permits legitimate lazy imports only inside the execution-admitted owner trees while rejecting every resolver result outside them. Business and bootstrap modules remain an exact per-module allowlist. Executing pandas/PyArrow is deferred to a separately approved M3 attestation decision and must never be inferred from their installation in the shared interpreter.

- [ ] **Step 1: Write failing policy/schema tests**

Reject missing/unknown fields, invalid versions, bool-as-int, subclasses, duplicate/NFC-colliding modes/modules, unsorted entrypoints/modules/owner trees, mode not in the internal literal allowlist, bad role, escaping path, invalid callable, omitted launcher/bootstrap helper, and digest mismatch. Prove checkout-root independence and raw/tracked hash semantics. Synthetic policy uses one synthetic mode; final Task 9 policy contains exact ordered `verify-m2` and `seal-historical-data` bindings and the exact six-tree M2 execution subset, with the verified pandas tree absent.

- [ ] **Step 2: Write failing controlled-loader and TOCTOU tests**

Create synthetic packages and attack them with repository shadow modules, altered `sys.path`, prefilled false `sys.modules`, custom `MetaPathFinder`, memory/zip/namespace loaders, sourceless `.pyc`, a malicious timestamp-valid cache, symlink/junction escape, source exchange between verify and exec, post-exec disk mutation, forged `module.__file__`, a dependency import outside verified owner trees, and an unlisted extension/DLL load. Add direct, descendant, prefilled-`sys.modules`, and hostile-finder attempts for both `pandas` and `pyarrow`; prove they fail before any package byte, package-data byte, extension, or DLL becomes visible.

Prove a business/dependency `.py` execution snapshot is read once and that same immutable buffer is used for raw hash, tracked hash, `compile`, and `exec`. A later seal/admission rescan is a separate verification read and can never replace the executed buffer. Verifying one process and executing another is forbidden.

Add bootstrap-normal-import, forged/pickled/cross-process admission, dead/replaced guard, content change before admission, and Windows native swap-and-restore attacks. For native files loaded after launcher start, keep a `CreateFileW` read handle that denies write/delete sharing across hash and delegated load, bind volume/file ID, restrict DLL search roots, and retain all reviewed `.pyd`/owned-DLL handles for the session lifetime. Assert swap/delete/alias/load-from-another-path fails. CPython and OS loader images already resident before launcher start remain explicit local TCB, not retrospectively attested memory pages.

- [ ] **Step 3: Run Task 2 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_source_execution.py" -v
```

Expected: import error for missing `source_execution_v1` or launcher.

- [ ] **Step 4: Implement the isolated bootstrap and persistent import guard**

Invoke the launcher as a direct source script with exact flags `-I -S -B`. It verifies `sys.flags.isolated`, disables bytecode, prevents `.pth` and user-site execution, derives repository/runtime roots, and admits only the policy's exact internal mode and entry callable.

The launcher contains the minimal strict bootstrap-policy parser, SHA-256/path verifier, and direct-source loader needed before package import. It reads `dependency_contents_v1.py` and `source_execution_v1.py` exactly once, compares their raw bytes to the reviewed bootstrap bindings, and `compile`/`exec`s those same buffers in private module namespaces. A normal import of either helper before the persistent guard exists is a hard failure, preventing circular self-attestation.

The local bootstrap TCB is stated exactly: the already-running CPython/OS loader and file/hash primitives, reviewed launcher bytes, and reviewed bootstrap bindings. The protocol proves reproducibility relative to that TCB; it does not claim TPM, secure boot, code signing, remote attestation, or proof of pre-launch in-memory pages.

Before any business/dependency import, `establish_attested_execution` runs Task 1's real verifier across the complete content lock, including pandas, and receives its private-issued capability. The controlled loader then uses one execution snapshot per source and retains the guard. Pin every admitted dependency source/native/package-data file with a read handle denying write/delete sharing for the session lifetime; bind file ID/digest and restrict DLL search roots. Install a persistent audit hook that denies Python `open`/`os.open`/pathlib/import-resource access under protected roots except the module-private verified resource reader. Builtin/frozen modules require exact reviewed names/loader kinds. Reject pyc, zip/custom/namespace loaders, unsafe DLL search, undeclared business import, or any direct native package-data I/O observed by the offline trace. Do not add a `six.moves` or generic custom-loader exception in M2: the approved trace must not request it, and any future real requirement for it is another stop-and-revise event.

Task 2 tests the engine with private synthetic policies only. The literal `verify-m2` mode is reserved but must fail `production_policy_not_frozen` until Task 9 creates the final M2 module lock and enables production session/admission issuance. No Task 2 or Task 8 helper can mint a production admission early, and command-line input never selects an arbitrary module/callable.

- [ ] **Step 5: Build and verify stable closure identities**

The execution policy binds the verified dependency-content-lock hash, launcher, exact bootstrap and business modules, both literal entrypoint mode/module/callable bindings, and complete admitted dependency owner trees. The closure binds policy, dependency content, business source tree, and raw business artifact hashes. Candidate labels and caller-supplied digests are never evidence.

`issue_m2_admission` proves live session/guard/issuer identity, verifies every retained dependency handle/file ID/digest, reruns complete set/business-source verification, and verifies closure. The guard repeats source/native checks before lazy load; package-data reads are served only from the verified handle snapshot/resource reader. Test post-admission package-data mutation through `open`, `Path.read_bytes`, import resources, alias paths, and a fake native direct-read adapter; every route either returns the originally verified bytes or fails before data is visible. If current packages require unmediated native package-data I/O or the environment cannot retain the complete handle set, stop and revise the attestation design.

`require_attested_m2_admission` verifies the same current-process issuer and live guard immediately before any stage-view capability is issued. Explicitly reject `CandidateManifestV1`, `RunCommonV1`, synthetic M1 fingerprints, dataclass lookalikes, pickled/serialized values, all-zero hashes, dead sessions, and a closure receipt from another process.

- [ ] **Step 6: Run an offline import trace for execution-admitted dependencies**

Under the controlled launcher, import the exact module names `numpy`, `dateutil`, `six`, `tzdata`, and the existing M1 Binance Spot modules without reading market data. Record the actual trace for review while policy identity binds their complete execution-admitted owner trees. Do not import pandas as part of the successful M2 trace. In separate negative probes, require `pandas`, every `pandas.*` descendant, `pyarrow`, and every `pyarrow.*` descendant to fail before `PathFinder`, module execution, protected-root reads, or native-image loading; assert that none remains in `sys.modules` or the process image set. The Task 1 receipt must still prove the unchanged full content lock, including pandas. The successful trace must observe no `six.moves`; if dynamic code generation, an unowned namespace, an unlisted DLL/plugin, or any required custom loader is observed, stop and revise the attestation design rather than adding a generic exception. Pandas/PyArrow execution attestation is deferred to M3, where it requires a separately reviewed negative-optional-dependency rule or complete owner-tree lock before any training use.

- [ ] **Step 7: Run focused and complete suites**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_source_execution.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
```

- [ ] **Step 8: Commit Task 2**

```powershell
git add binance_spot_strategy/attested_launcher.py binance_spot_strategy/identity tests/binance_spot/test_source_execution.py
git commit -m "feat(binance-spot): control attested source execution"
```

### Task 3: Immutable Public-HTTP Evidence Contracts

**Files:**
- Create: `binance_spot_strategy/data/__init__.py`
- Create: `binance_spot_strategy/data/http_records_v1.py`
- Create: `tests/binance_spot/test_data_contracts.py`

**Interfaces:**
- Consumes: M1 canonical timestamp, SHA-256, canonical JSON, `Symbol`, and `BarInterval` contracts.
- Produces: `RestRequestV1`, complete `HttpResponseBytes`, overflow `HttpResponsePrefixV1`, `RestExchangeReceiptV1`, `RestFailureReceiptV1`, and `ServerTimeEvidenceV1`.
- Produces: strict `to_payload`, `from_payload`, and canonical hash functions for every identity-bearing receipt.

Use these exact public shapes:

```python
@dataclass(frozen=True, slots=True)
class RestRequestV1:
    request_seq: int
    attempt_no: int
    method: str
    scheme: str
    host: str
    path: str
    ordered_query: tuple[tuple[str, str], ...]
    ordered_headers: tuple[tuple[str, str], ...]
    started_at_utc: datetime

@dataclass(frozen=True, slots=True)
class HttpResponseBytes:
    status_code: int
    ordered_headers: tuple[tuple[str, str], ...]
    body: bytes

@dataclass(frozen=True, slots=True)
class HttpResponsePrefixV1:
    status_code: int
    ordered_headers: tuple[tuple[str, str], ...]
    captured_prefix: bytes

@dataclass(frozen=True, slots=True)
class RestExchangeReceiptV1:
    request: RestRequestV1
    retrieved_at_utc: datetime
    status_code: int
    ordered_response_headers: tuple[tuple[str, str], ...]
    body_sha256: str
    body_size: int

@dataclass(frozen=True, slots=True)
class RestFailureReceiptV1:
    request: RestRequestV1
    failed_at_utc: datetime
    failure_code: str
    capture_kind: str
    status_code_or_zero: int
    ordered_response_headers: tuple[tuple[str, str], ...]
    captured_body_sha256: str
    captured_body_size: int

@dataclass(frozen=True, slots=True)
class ServerTimeEvidenceV1:
    exchange_receipt_sha256: str
    request_seq: int
    server_time_ms: int
```

`capture_kind` is exactly `none` or `prefix`. For `none`, status is zero and the digest/size are the canonical empty-bytes values; for `prefix`, the receipt binds the exact captured prefix and original HTTP status/headers without presenting it as a complete response.

Freeze `failure_code` to: `dns_failure`, `connect_failure`, `tls_failure`, `timeout`, `response_body_too_large`, `redirect_status`, `http_status`, `content_type`, `content_encoding`, `server_time_decode`, `kline_decode`, `closure_failure`, `page_sequence`, `empty_page`, `archive_integrity`, `root_identity`, and `interrupted_run`. No exception-derived value is accepted. A failure receipt never serializes an exception message, credential-bearing URL, body excerpt, or path.

Response receipts bind every response header returned by `HTTPMessage.raw_items()` in received order. Lowercase ASCII header names, preserve values after rejecting CR/LF/NUL, allow repeated response field names because multiplicity and array order are evidence identity, and never sort, merge, or select a machine-dependent subset. After persistence, semantic validation rejects more than one `Content-Type` or `Content-Encoding` field rather than choosing or merging values; it then accepts exactly media type `application/json` with at most `charset=utf-8` and absent or `identity` content encoding. Request headers remain unique and duplicate/casefold-colliding request names are rejected.

- [ ] **Step 1: Write failing exact-type and allowlist tests**

Test exact builtins, frozen tuples, canonical millisecond UTC, positive sequence/attempt numbers, `GET`, `https`, `data-api.binance.vision`, and only `/api/v3/time` or `/api/v3/klines`. Reject bool-as-int, every subclass, duplicate headers/query keys, NFC/casefold collisions, invalid header bytes, fragments, userinfo, endpoint override, and unexpected query fields.

Reject `Authorization`, `Proxy-Authorization`, `Cookie`, `X-MBX-APIKEY`, `signature`, `timestamp`, and every account/order/private endpoint. Require `Accept: application/json` and `Accept-Encoding: identity`; do not accept transparent compression.

- [ ] **Step 2: Write failing receipt/golden-hash tests**

Assert complete response and overflow prefix are exact immutable `bytes`, while receipts contain only digest and size. Test status boundaries, response header order/case/repeated names, invalid names/values, ambiguous repeated content headers, naive/sub-millisecond timestamps, negative server time, `serverTime` bool, capture-kind invariants, and failure-code injection. One-field mutations must change the receipt hash; audit timestamps change receipt identity but never data semantics outside the receipt closure.

- [ ] **Step 3: Run Task 3 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_data_contracts.py" -v
```

Expected: import error for missing `binance_spot_strategy.data.http_records_v1`.

- [ ] **Step 4: Implement strict immutable HTTP evidence**

Snapshot external mappings/iterables once into exact builtins before validation. Preserve schema-defined query/header order; never sort a caller's malformed input into validity. Canonical receipt payloads include `schema_version`, `numeric_protocol_version`, exact request metadata, retrieval/failure timestamp, status, all normalized response headers, capture kind, body/prefix digest, and size, with their own hash absent.
Keep `HttpResponseBytes` out of `data.__all__`; the full-custody acquisition path imports it from its defining module. Export only safe immutable record types and validation errors from the narrow package surface.

- [ ] **Step 5: Run focused and complete tests, then commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_data_contracts.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data tests/binance_spot/test_data_contracts.py
git commit -m "feat(binance-spot): define public data evidence"
```

### Task 4: Create-Only Content-Addressed Raw Archive

**Files:**
- Create: `binance_spot_strategy/data/raw_archive_v1.py`
- Create: `tests/binance_spot/test_raw_archive.py`

**Interfaces:**
- Consumes: exact raw HTTP entity-body bytes and Task 3 canonical receipt payloads.
- Produces: immutable `ContentRefV1(kind, sha256, size_bytes)`.
- Produces: custody-only `RawArchiveWriterV1.put_blob(raw_bytes: bytes) -> ContentRefV1`, `put_record(payload: Mapping[str, object]) -> ContentRefV1`, and `persist_exchange(request: RestRequestV1, response: HttpResponseBytes, retrieved_at_utc: datetime) -> RestExchangeReceiptV1`.
- Produces runtime-only opaque `VerifiedExternalRootV1`, `RootOwnerLeaseV1`, `CustodyArchiveSessionV1`, and named `CustodyStageWritersV1`; none is serializable, publicly constructible, iterable, positional, or reducible to a path.
- Produces internal `pin_external_root(path: Path, purpose: str) -> VerifiedExternalRootV1` and `claim_root_owner(root: VerifiedExternalRootV1) -> RootOwnerLeaseV1`.
- Produces internal `open_custody_archive_session(root: VerifiedExternalRootV1, owner: RootOwnerLeaseV1) -> CustodyArchiveSessionV1`; the session alone issues the reader/writer capabilities below and keeps root/component handles plus the owner lease alive.
- Produces internal `open_custody_stage_writers(session: CustodyArchiveSessionV1) -> CustodyStageWritersV1`. The factory derives exactly three handle-relative child roots `stage-views/training`, `stage-views/validation`, and `stage-views/holdout` from the pinned custody session, embeds the matching immutable stage label in each writer, and rejects reused file IDs/backing roots or a pre-existing wrong-label layout before content access.
- Produces internal `open_training_release_writer(root: VerifiedExternalRootV1, owner: RootOwnerLeaseV1) -> ContentAddressedStageViewWriterV1`; it accepts only external purpose `training_release`, embeds label `training_release`, retains the pinned handles/lease, and has no raw-body API.
- Produces: `persist_incomplete_response(request: RestRequestV1, response: HttpResponsePrefixV1, failed_at_utc: datetime) -> RestFailureReceiptV1`, `persist_failure(request: RestRequestV1, failure_code: str, failed_at_utc: datetime) -> RestFailureReceiptV1`, and `persist_server_time_evidence(evidence: ServerTimeEvidenceV1) -> ContentRefV1`.
- Produces custody-only `CustodySnapshotReaderV1.read_blob_once(ref: ContentRefV1) -> bytes` and `read_record_once(ref: ContentRefV1) -> bytes`. Each returns one immutable snapshot and never a path, stream, iterator, mmap, or reusable file object.
- Produces stage-only `ContentAddressedStageViewWriterV1.put_bar_object(raw_record: bytes) -> ContentRefV1`, `put_manifest(payload: Mapping[str, object]) -> ContentRefV1`, and `publish_root(root_record_ref: ContentRefV1) -> ContentRefV1`. Custody-stage roots point to their `StageViewCatalogV1`; the external training-release root points to `TrainingReleaseRecordV1`.
- No package-root export exposes a writer, full reader, root handle, raw body, or path.

- [ ] **Step 1: Write failing path, mutation, and crash tests**
External `purpose` is exactly `custody` or `training_release`. Pinning canonicalizes Windows drive/UNC/8.3/case aliases, walks every component without following reparse points, records volume serial/file ID, and rejects repository/worktree/nested/same-identity external roots. The three fixed internal custody-stage children are created only handle-relative by `open_custody_stage_writers`; they are not caller-supplied external roots and do not weaken that rejection. Owner claim is exclusive by layout identity and occurs before any transport is constructed.


Use `TemporaryDirectory` only. Assert identical bytes return the same ref; an existing matching object is rehashed before reuse. Reject a truncated/corrupted existing object, same digest path with different bytes, filename/content mismatch, non-bytes input, traversal, absolute logical key, symlink/junction/reparse component, hardlink/file-ID alias, non-regular file, unknown layout marker, root identity change, concurrent owner, and an archive root inside the repository. Assert the named custody-writer factory cannot return duplicate/shared file identities, swapped labels, a missing child, or a positional/reordered collection; the external training-release writer can never be substituted for one of the three custody writers.

Simulate interruption after partial write. A partial digest-named object must never be returned as valid; the next operation fails closed rather than overwriting it. Failure receipts remain reachable from the acquisition manifest and cannot be silently removed.

- [ ] **Step 2: Run Task 4 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_raw_archive.py" -v
```

Expected: import error for missing `raw_archive_v1`.

- [ ] **Step 3: Implement exact-byte write-once storage**

Store raw HTTP entity bodies/prefixes under `blobs/sha256/<first-two-hex>/<digest>` and canonical records under `records/sha256/<first-two-hex>/<digest>`. Create the versioned layout marker and every final object with exclusive creation through retained no-follow handles. Flush and fsync before returning. If a path exists, read once, recompute size/hash/file identity, and accept only identical content. Expose no delete, overwrite, rename-over-existing, or repair method.

Persist complete response bytes before parsing or interpreting status/header/content. Then persist the exchange receipt referring to body digest/size. Persist a bounded overflow prefix before its failure receipt, and persist body-less transport failures before returning. All body/prefix/receipt bytes are immutable custody evidence even when an attempt is not selected.

- [ ] **Step 4: Prove one-snapshot reads and capability separation**

Use a stateful reader spy and replace-after-open/root-junction-swap attacks to prove hash and parse consumers receive the same byte snapshot and never reopen by path. Bind root/component volume and file IDs, retain directory handles, and recheck before each publish. Assert the safe `data` package export omits all writer/read capabilities, raw response bytes, and root paths.

- [ ] **Step 5: Run focused and complete tests, then commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_raw_archive.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data/raw_archive_v1.py tests/binance_spot/test_raw_archive.py
git commit -m "feat(binance-spot): add immutable raw archive"
```

### Task 5: Allowlisted Public REST Transport And Causal Acquisition

**Files:**
- Create: `binance_spot_strategy/data/transport_v1.py`
- Create: `binance_spot_strategy/data/acquisition_v1.py`
- Create: `tests/binance_spot/test_binance_public_transport.py`
- Create: `tests/binance_spot/test_acquisition.py`

**Interfaces:**
- Consumes: Task 3 requests/responses, Task 4 custody-only archive session, an injected audit clock, `ServerTimeEvidenceDecoderV1`, and `PageEvidenceDecoderV1`.
- Produces: `PublicRestTransport.get(request: RestRequestV1) -> HttpResponseBytes | HttpResponsePrefixV1` and production `UrllibPublicRestTransport`; only DNS/TLS/connect/timeout failures have no HTTP capture.
- Produces: immutable `HistoricalAcquisitionPlanV1` with exact symbols, interval, UTC bounds, page limit, origin, endpoint, and timeout.
- Produces: immutable `AcquisitionChainGenesisV1`, `AcquisitionChainRecordV1`, `AcquisitionRunStartV1`, `KlinePageAttemptStartV1`, strict `PageAttemptOutcomeV1`, `SelectedPageEvidenceV1`, `AcquisitionRunRecordV1`, `AcquisitionPassV1`, and `AcquisitionFailureV1`.
- Produces: `ServerTimeEvidenceDecoderV1.decode(raw_body: bytes, response_receipt: RestExchangeReceiptV1) -> ServerTimeEvidenceV1`.
- Produces: `PageEvidenceDecoderV1.decode_and_archive_selected_page(raw_body: bytes, symbol: Symbol, interval: BarInterval, requested_start_ms: int, requested_end_time_inclusive_ms: int, response_receipt: RestExchangeReceiptV1, server_time_evidence: ServerTimeEvidenceV1, attempt_start_sha256: str, archive: RawArchiveWriterV1) -> SelectedPageEvidenceV1`.
- Produces: `acquire_historical_archive(plan: HistoricalAcquisitionPlanV1, transport: PublicRestTransport, archive: RawArchiveWriterV1, clock: AuditClock, server_time_decoder: ServerTimeEvidenceDecoderV1, page_decoder: PageEvidenceDecoderV1) -> AcquisitionPassV1 | AcquisitionFailureV1`.

Freeze these exact acquisition-chain and result fields:

```python
@dataclass(frozen=True, slots=True)
class AcquisitionChainGenesisV1:
    schema_version: str
    numeric_protocol_version: str
    custody_layout_id: str
    plan_sha256: str

@dataclass(frozen=True, slots=True)
class AcquisitionChainRecordV1:
    schema_version: str
    numeric_protocol_version: str
    genesis_sha256: str
    previous_chain_record_sha256: str
    chain_ordinal: int
    run_ordinal: int
    event_kind: str
    event_payload_sha256: str

@dataclass(frozen=True, slots=True)
class AcquisitionRunStartV1:
    schema_version: str
    numeric_protocol_version: str
    genesis_sha256: str
    plan_sha256: str
    run_ordinal: int

@dataclass(frozen=True, slots=True)
class KlinePageAttemptStartV1:
    schema_version: str
    numeric_protocol_version: str
    genesis_sha256: str
    plan_sha256: str
    run_ordinal: int
    symbol: Symbol
    page_ordinal: int
    attempt_no: int
    interval: BarInterval
    requested_start_ms: int
    requested_end_time_inclusive_ms: int

@dataclass(frozen=True, slots=True)
class PageAttemptTimeFailureV1:
    schema_version: str
    numeric_protocol_version: str
    outcome_kind: str
    attempt_start_sha256: str
    server_time_exchange_or_failure_receipt_sha256: str
    server_time_request_seq: int
    failure_code: str

@dataclass(frozen=True, slots=True)
class PageAttemptKlineFailureV1:
    schema_version: str
    numeric_protocol_version: str
    outcome_kind: str
    attempt_start_sha256: str
    server_time_receipt_sha256: str
    server_time_evidence_sha256: str
    server_time_request_seq: int
    kline_exchange_or_failure_receipt_sha256: str
    kline_request_seq: int
    failure_code: str

@dataclass(frozen=True, slots=True)
class PageAttemptSuccessV1:
    schema_version: str
    numeric_protocol_version: str
    outcome_kind: str
    attempt_start_sha256: str
    server_time_receipt_sha256: str
    server_time_evidence_sha256: str
    server_time_request_seq: int
    kline_exchange_receipt_sha256: str
    kline_request_seq: int
    selected_page_evidence_sha256: str

PageAttemptOutcomeV1 = (
    PageAttemptTimeFailureV1
    | PageAttemptKlineFailureV1
    | PageAttemptSuccessV1
)

@dataclass(frozen=True, slots=True)
class SelectedPageEvidenceV1:
    schema_version: str
    numeric_protocol_version: str
    attempt_start_sha256: str
    symbol: Symbol
    interval: BarInterval
    requested_start_ms: int
    requested_end_time_inclusive_ms: int
    first_open_time_ms: int
    last_open_time_ms: int
    raw_body_sha256: str
    raw_body_size: int
    ordered_kline_record_refs: tuple[ContentRefV1, ...]

@dataclass(frozen=True, slots=True)
class AcquisitionRunRecordV1:
    schema_version: str
    numeric_protocol_version: str
    genesis_sha256: str
    plan_sha256: str
    run_ordinal: int
    run_status: str
    ordered_page_attempt_outcome_sha256s: tuple[str, ...]
    selected_pages: tuple[SelectedPageEvidenceV1, ...]
    failure_code: str

@dataclass(frozen=True, slots=True)
class AcquisitionPassV1:
    schema_version: str
    numeric_protocol_version: str
    plan_sha256: str
    run_ordinal: int
    run_record_sha256: str
    acquisition_chain_head_sha256: str
    selected_pages: tuple[SelectedPageEvidenceV1, ...]

@dataclass(frozen=True, slots=True)
class AcquisitionFailureV1:
    schema_version: str
    numeric_protocol_version: str
    plan_sha256: str
    run_ordinal: int
    run_record_sha256: str
    acquisition_chain_head_sha256: str
    failure_code: str
```

`PageAttemptOutcomeV1` is a strict discriminated union. `outcome_kind` is exactly `time_failure`, `kline_failure`, or `success` for its corresponding exact dataclass; no other union member, null, empty string, zero hash, or sentinel stands in for unavailable evidence. A time failure has no server-time evidence or kline fields, and a kline failure has no selected-page field.

`AcquisitionChainRecordV1` is the sole linear journal. Its first record points to the genesis hash; every later record points to the immediately preceding chain-record hash. `event_kind` is exactly `run_start`, `attempt_start`, `attempt_complete`, `run_terminal`, or `interrupted_recovery`. Its `event_payload_sha256` binds respectively an `AcquisitionRunStartV1`, `KlinePageAttemptStartV1`, exact `PageAttemptOutcomeV1`, terminal `AcquisitionRunRecordV1`, or interrupted `AcquisitionRunRecordV1`. Chain ordinal is globally contiguous and run ordinal never decreases.

`AcquisitionRunRecordV1.run_status` is exactly `pass`, `fail`, or `interrupted`; pass uses failure code `none`. Before any network I/O, persist `AcquisitionRunStartV1` and append `run_start`, then persist the exact `KlinePageAttemptStartV1` and append `attempt_start`. Both start records bind the same genesis. The run record lists every completed attempt outcome hash in occurrence order. A pass/fail run must match every attempt start to exactly one outcome; only an interrupted run may end with one unmatched final attempt start. Persist the terminal run payload and append its terminal chain record before returning a result.

On restart, a nonterminal chain head is closed under the single-owner lock by one `interrupted` recovery run payload/event before any new run or network. It can never become selected. Orphan immutable objects remain custody-only and are never inferred into validity. Tests inject a crash after every persistence boundary and prove the next successful head cannot omit or reorder any completed failure/interrupted event. The final custody manifest traverses the exact chain from successful head to genesis.

Freeze the production plan to:

```python
PUBLIC_MARKET_DATA_ORIGIN = "https://data-api.binance.vision"
SERVER_TIME_PATH = "/api/v3/time"
KLINES_PATH = "/api/v3/klines"
MAX_RESPONSE_BODY_BYTES = 4_194_304
MAX_CAPTURE_PREFIX_BYTES = 4_194_305
KLINE_LIMIT = 1000
HTTP_TIMEOUT_SECONDS = 15
INTERVAL_MILLISECONDS = 14_400_000
```

The planned historical open-time range is `[2017-10-03T00:00:00Z, 2026-08-01T00:00:00Z)`: exactly the earliest proposed training warm-up through the final holdout boundary. This is a derived request range, not proof of common coverage.

- [ ] **Step 1: Write failing transport allowlist tests**

Patch the urllib opener; never use a real socket. Assert exact GET URL encoding, query order, `Accept: application/json`, `Accept-Encoding: identity`, fixed host/path, disabled environment proxies, fixed timeout, and exact response bytes. Install a no-redirect handler: every 3xx, including same-origin, is returned as the original response capture and later fails structurally.

Catch `urllib.error.HTTPError` as an HTTP response and return its status/headers/body rather than losing 3xx/4xx/5xx evidence. Read at most `MAX_CAPTURE_PREFIX_BYTES`: EOF at or below `MAX_RESPONSE_BODY_BYTES` returns complete `HttpResponseBytes`; the extra byte returns `HttpResponsePrefixV1`. The transport never validates status, content type, content encoding, or JSON, never retries/sleeps, and never reads an account endpoint. Acquisition persists the complete body or overflow prefix first, then validates status/header/content and records structural failure.

- [ ] **Step 2: Write failing server-time-before-page acquisition tests**

Use a scripted fake transport and archive spy. Prove this run-level prelude once and the page-level sequence for every page:

```text
once per run: persist AcquisitionRunStartV1, then append run_start
for each page:
    persist KlinePageAttemptStartV1, then append attempt_start
    send GET /api/v3/time
    persist its complete body/overflow prefix when present, then its exchange/failure receipt
    if time transport/status/header/body/decode fails:
        persist PageAttemptTimeFailureV1, append attempt_complete, persist failed run, append run_terminal, stop
    strictly decode and persist ServerTimeEvidenceV1
    send the immediately adjacent GET /api/v3/klines
    persist its complete body/overflow prefix when present, then its exchange/failure receipt
    if kline transport/status/header/body/decode/page validation fails:
        persist PageAttemptKlineFailureV1, append attempt_complete, persist failed run, append run_terminal, stop
    decode and validate the whole page before selecting any row
    persist every canonical per-bar record after full-page validation
    persist SelectedPageEvidenceV1 bound to attempt_start_sha256
    persist PageAttemptSuccessV1, then append attempt_complete
after the final page, persist passed run and append run_terminal
```

Task 5 tests inject deterministic fake server-time/page decoders; Task 6 supplies both production strict implementations and a wiring integration test. The acquisition loop owns ordering, persistence, chain/run identity, and fail-fast behavior; decoders own exact payload semantics.

Crash tests cover every boundary, including after attempt start but before `/time`, after a `/time` body/prefix, after its receipt, on server-time decode failure, after server-time evidence but before the kline request, after a kline body/prefix, and before each outcome/chain/terminal write. Recovery never fabricates missing server evidence, kline fields, or a selected page.

Require one-to-one binding of plan hash, run ordinal, symbol, page ordinal, attempt number, bounds, start/outcome hashes, and receipts. Require `kline_request_seq == server_time_request_seq + 1` with no intervening business request and globally increasing sequences within the run. Test cross-run/page receipt replay, local clock jumps, server-time rollback, HTTP 429/500, redirect, overflow, network failure, invalid time body, malformed/truncated kline body, duplicate/retrograde/empty/overlap/range-escape page, and an extra unclosed bar. No failed attempt becomes selected.

- [ ] **Step 3: Run Task 5 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_binance_public_transport.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_acquisition.py" -v
```

Expected: missing transport/acquisition imports.

- [ ] **Step 4: Implement sequential fail-fast acquisition**

For each symbol in exact `BTCUSDT`, `ETHUSDT` order, request pages from planned start with `symbol`, `interval=4h`, `startTime`, `endTime`, `timeZone=0`, and `limit=1000`. Advance `next_start_ms = last_open_time_ms + 14_400_000`; never infer progress from row count. The plan field is `exclusive_end_ms`; each request/evidence field is explicitly `requested_end_time_inclusive_ms = exclusive_end_ms - 1`.

Before every page, acquire/persist/decode one server-time response and bind it to the immediately following Kline request. No automatic retry is allowed. Any error persists a terminal failed run record that transitively binds every earlier run/attempt and stops before custody/stage manifest publication.

A later operator rerun against the same recognized root starts from the historical range beginning, increments `run_ordinal`, points to the prior chain head, and may reuse only rehashed byte-identical objects. A different plan or broken/missing predecessor chain fails closed. The single-owner root lock is claimed and root identities are pinned before transport construction.

Add tests that mutate any old failed receipt, reorder/omit a prior run, reuse a server-time receipt across run/page, or crash between response/receipt/attempt/run publication. Final success must traverse the exact create-only chain to genesis and reject every mutation or omission.

- [ ] **Step 5: Prove no normal test can reach the network**

Add a suite guard patching `socket.socket`, `socket.create_connection`, urllib openers, and DNS lookup to raise. All Binance Spot tests, including the future custody CLI tests, must remain green through fake transports.

- [ ] **Step 6: Run complete tests and commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_binance_public_transport.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_acquisition.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data/transport_v1.py binance_spot_strategy/data/acquisition_v1.py tests/binance_spot/test_binance_public_transport.py tests/binance_spot/test_acquisition.py
git commit -m "feat(binance-spot): acquire public kline evidence"
```

### Task 6: Strict Kline Codec And Complete Custody-Acquisition Manifest

**Files:**
- Create: `binance_spot_strategy/data/kline_codec_v1.py`
- Create: `binance_spot_strategy/data/raw_manifest_v1.py`
- Create: `tests/binance_spot/fixtures/raw_api/server_time.json`
- Create: `tests/binance_spot/fixtures/raw_api/klines_page.json`
- Create: `tests/binance_spot/test_kline_codec.py`

**Interfaces:**
- Consumes: exact raw server-time/kline response bytes, their receipts, symbol/interval request context, and preceding `ServerTimeEvidenceV1`.
- Produces: strict production implementations of Task 5 `ServerTimeEvidenceDecoderV1` and `PageEvidenceDecoderV1`.
- Produces: immutable `KlineRecordObjectV1` retaining only the canonical per-bar source record; raw-page/receipt/server-time provenance remains in custody-only page evidence.
- Produces: strict `CustodyAcquisitionManifestV1`, `custody_acquisition_manifest_from_payload`, and `custody_acquisition_manifest_hash`.

Freeze the canonical per-bar object and full custody closure:

```python
@dataclass(frozen=True, slots=True)
class KlineRecordObjectV1:
    schema_version: str
    numeric_protocol_version: str
    symbol: Symbol
    interval: BarInterval
    raw_kline: tuple[int, str, str, str, str, str, int, str, int, str, str, str]

@dataclass(frozen=True, slots=True)
class CustodyAcquisitionManifestV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    acquisition_plan_sha256: str
    source_origin: str
    symbols: tuple[Symbol, Symbol]
    interval: BarInterval
    acquisition_chain_genesis_sha256: str
    acquisition_chain_head_sha256: str
    successful_run_record_sha256: str
    selected_pages: tuple[SelectedPageEvidenceV1, ...]
```

`kline_record_object_to_payload`, `kline_record_object_from_payload`, and `kline_record_object_hash` use the exact canonical payload below. Its canonical bytes are persisted through `put_record`; the resulting `ContentRefV1.sha256` must equal `kline_record_object_hash`.

- [ ] **Step 1: Write failing server-time and 12-field kline tests**

Require server-time body to be an exact object with exact positive integer `serverTime`; reject duplicate keys, bool, float, negative, null, unknown/missing field, BOM, and malformed UTF-8/JSON.

Require each kline to be an exact 12-element array. Open/close times and trade count are exact ints; decimal fields are exact strings accepted by M1 `parse_finite_decimal`; the ignore field is preserved as an exact string. Reject float/bool/null/subclass, scientific notation, whitespace, leading plus, over-18 scale, zero/negative price, negative volume/trade count, OHLC envelope failure, duplicate identity, symbol/interval relabel, and any row outside the requested page bounds.

Assert UTC grid and exact close formula. A bar passes closure only when `close_time_ms < server_time_ms`; test server time one millisecond after, equal to, and one millisecond before close. If any row in a page is unclosed or invalid, reject the entire page and select none of it.

- [ ] **Step 2: Freeze per-bar semantic checksum and raw manifest goldens**

Hash each bar from this canonical payload so a Binance row cannot be relabelled between symbols or intervals:

```python
{
    "schema_version": "binance_spot_kline_record_v1",
    "numeric_protocol_version": "numeric_protocol_v1",
    "symbol": symbol.value,
    "interval": interval.value,
    "raw_kline": list(raw_kline),
}
```

The custody-acquisition manifest validates the successful run record and traverses its append-only acquisition chain to genesis. Through the ordered attempt-start/outcome and selected-page records it binds every earlier pass/fail/interrupted attempt, requested bounds, server-time/response receipts, raw body/prefix refs, and ordered per-bar record refs. Arrays are schema order: reject reordering rather than sorting it into validity.

- [ ] **Step 3: Run Task 6 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_kline_codec.py" -v
```

Expected: missing codec/raw-manifest imports.

- [ ] **Step 4: Implement strict one-snapshot decoding and manifest validation**

Decode server-time and each raw body once with duplicate-key and nonfinite rejection. Preserve original decimal strings in `KlineRecordObjectV1`; construct the existing domain `Bar` only after exact Decimal validation. Bind every selected page to its exact `KlinePageAttemptStartV1`, matching `PageAttemptSuccessV1`, immediately preceding server-time evidence, request pair, plan/run/symbol/page identity, and bounds.

Validate the complete page in memory first. If any row fails, persist no per-bar object and select no part of the page. After full success, canonicalize and exclusive-create every `KlineRecordObjectV1`, verify returned refs, then persist `SelectedPageEvidenceV1`. Unknown fields, selected non-200 receipts, missing/replayed attempts, range mismatch, overlap, gap, BTC/ETH order swap, body digest drift, row-digest drift, or chain omission fail closed.

Add a production-wiring integration test that feeds fake raw responses through the real server-time/page decoders and archive, proving every selected record hash resolves to the exact canonical per-bar bytes without reopening the raw page.

The custody-acquisition manifest and raw-page provenance are full-custody closure only. They are never used directly as a future run's `immutable_raw_data_manifest_sha256`, never copied to a stage release, and never placed in `KlineBarRefV1`. Task 7 derives stage identity solely from exact stage-union per-bar objects. The custody manifest contains no strategy result, activation, candidate/run identity, or metric.

- [ ] **Step 5: Run complete tests and commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_kline_codec.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data/kline_codec_v1.py binance_spot_strategy/data/raw_manifest_v1.py tests/binance_spot/fixtures/raw_api tests/binance_spot/test_kline_codec.py
git commit -m "feat(binance-spot): freeze raw kline manifests"
```

### Task 7: Nonrevealing Preregistration Sealer And Stage Manifests

**Files:**
- Create: `binance_spot_strategy/data/stage_manifest_v1.py`
- Create: `binance_spot_strategy/data/preregistration_sealer_v1.py`
- Create: `tests/binance_spot/fixtures/m2_data_manifests_v1.json`
- Create: `tests/binance_spot/test_stage_sealer.py`

**Interfaces:**
- Consumes: Task 6 `CustodyAcquisitionManifestV1` and a custody-only archive capability.
- Produces: strict `KlineBarRefV1`, `StageBucketRefV1`, `StageRawDataManifestV1`, `HistoricalStageManifestV1`, custody-only `StageSealBindingV1`/`PreregistrationSealManifestV1`, releasable `TrainingReleaseCatalogV1`/`TrainingReleaseRecordV1`, and public `SealPassV1`/`SealFailureV1`.
- Produces strict `to_payload`, `from_payload`, and canonical hash functions for every identity-bearing manifest/catalog.
- Produces: `seal_preregistered_stages(reader: CustodySnapshotReaderV1, custody_manifest_sha256: str, custody_stage_writers: CustodyStageWritersV1, training_release_writer: ContentAddressedStageViewWriterV1) -> SealPassV1 | SealFailureV1`.
- Produces physically separate byte-copied training, validation, and holdout custody views plus one external training-only release; no link, shared object root, or cross-view resolver is permitted.

Freeze these exact stage-local shapes:

```python
@dataclass(frozen=True, slots=True)
class KlineBarRefV1:
    symbol: Symbol
    interval: BarInterval
    open_time: datetime
    close_time: datetime
    kline_record_sha256: str
    kline_record_size: int

@dataclass(frozen=True, slots=True)
class StageBucketRefV1:
    open_time: datetime
    close_time: datetime
    bars: tuple[KlineBarRefV1, KlineBarRefV1]

@dataclass(frozen=True, slots=True)
class StageRawDataManifestV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    stage_name: str
    source_contract_sha256: str
    symbols: tuple[Symbol, Symbol]
    interval: BarInterval
    indicator_start: datetime
    performance_start: datetime
    performance_end: datetime
    warmup_buckets: tuple[StageBucketRefV1, ...]
    performance_buckets: tuple[StageBucketRefV1, ...]

@dataclass(frozen=True, slots=True)
class HistoricalStageManifestV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    stage_name: str
    stage_raw_data_manifest_sha256: str
    symbols: tuple[Symbol, Symbol]
    interval: BarInterval
    indicator_start: datetime
    performance_start: datetime
    performance_end: datetime
    warmup_count: int
    initial_balance_q18: str
    common_coverage_start: datetime
    common_coverage_end: datetime
    warmup_buckets: tuple[StageBucketRefV1, ...]
    performance_buckets: tuple[StageBucketRefV1, ...]

@dataclass(frozen=True, slots=True)
class StageViewCatalogV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    stage_name: str
    stage_raw_data_manifest_ref: ContentRefV1
    stage_manifest_ref: ContentRefV1
    ordered_bar_record_refs: tuple[ContentRefV1, ...]

@dataclass(frozen=True, slots=True)
class StageSealBindingV1:
    stage_name: str
    stage_raw_data_manifest_sha256: str
    stage_manifest_sha256: str
    custody_view_root_sha256: str

@dataclass(frozen=True, slots=True)
class PreregistrationSealManifestV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    custody_acquisition_manifest_sha256: str
    acquisition_chain_head_sha256: str
    common_coverage_start: datetime
    common_coverage_end: datetime
    stage_bindings: tuple[StageSealBindingV1, StageSealBindingV1, StageSealBindingV1]

@dataclass(frozen=True, slots=True)
class TrainingReleaseCatalogV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    stage_name: str
    stage_raw_data_manifest_ref: ContentRefV1
    stage_manifest_ref: ContentRefV1
    ordered_bar_record_refs: tuple[ContentRefV1, ...]

@dataclass(frozen=True, slots=True)
class TrainingReleaseRecordV1:
    schema_version: str
    numeric_protocol_version: str
    design_revision: str
    stage_name: str
    source_contract_sha256: str
    training_raw_data_manifest_sha256: str
    training_stage_manifest_sha256: str
    training_release_catalog_sha256: str
    training_release_root_identity_sha256: str

@dataclass(frozen=True, slots=True)
class SealPassV1:
    schema_version: str
    numeric_protocol_version: str
    status: str
    common_coverage_start: datetime
    common_coverage_end: datetime
    training_warmup_count: int
    training_performance_count: int
    training_raw_data_manifest_sha256: str
    training_stage_manifest_sha256: str
    training_release_catalog_sha256: str
    training_release_record_sha256: str

@dataclass(frozen=True, slots=True)
class SealFailureV1:
    schema_version: str
    numeric_protocol_version: str
    status: str
    failure_code: str
    common_bounds_known: bool
    common_coverage_start: datetime | None
    common_coverage_end: datetime | None
    symbol: Symbol | None
    expected_open_time: datetime | None
    observed_open_time: datetime | None
    source_checksum_sha256: str | None
```

`StageRawDataManifestV1` binds only the exact stage warm-up/performance union of per-bar canonical objects; its hash is the future `immutable_raw_data_manifest_sha256`. It contains no raw-response, page, receipt, server-time, acquisition-chain, custody-manifest, or other-stage hash. Custody-only records retain the auditable mapping from each per-bar hash back to its page evidence.

`HistoricalStageManifestV1` freezes exact symbols/interval/windows, 540 warm-up count, `500.000000000000000000` USDT Q18 balance, common bounds, and ordered buckets without OHLCV. `StageViewCatalogV1` is custody-only and its exact stage name must match its binding. `TrainingReleaseCatalogV1.stage_name` is exactly `training`; it contains only the two training manifests and deduplicated first-occurrence ordered training bar refs. Its hash identifies the exact released data closure, but is not by itself authority to load that closure.

`TrainingReleaseRecordV1.stage_name` is exactly `training`. It binds the source contract and the three exact training closure hashes to a canonical path-free root-identity audit payload hash derived from the retained volume serial/file ID/layout identity. Its own canonical hash is the final release success/approval identity. Store the record content-addressed after the release-tree scan and publish a single create-only root pointer to that record last; the pointer never targets the catalog directly.

`PreregistrationSealManifestV1` and validation/holdout `StageSealBindingV1` values are persisted inside custody and never returned, logged, or copied to training. `SealPassV1.status` is exactly `pass`; it is returned only after the release-record pointer is durably published and repeats that record hash. `SealFailureV1.status` is exactly `fail`. Failure optional fields must be jointly consistent with `common_bounds_known` and its allowlisted structural code; they contain no stage ref or path.
Strict parser/hash tests cover missing/unknown keys, schema-defined versus forbidden null, float/nonfinite, bool-as-int, subclasses, BOM/UTF-8, NFC/casefold collisions, wrong schema/numeric/design version, stage relabel, hash-type swaps, tuple/order changes, duplicate bucket/ref, symbol swap, root-identity mismatch, catalog/record substitution, and every one-field mutation. A valid holdout manifest/catalog can never be relabelled or supplied as training.

Generate pages that straddle both training/validation and validation/holdout boundaries. Modify only a reserved row and prove the earlier stage raw/stage/catalog hashes remain byte-identical while the custody closure and affected reserved stage change. Page-level provenance is deliberately outside every stage identity.


- [ ] **Step 1: Write failing structural audit tests**

Generate synthetic full-window data in memory. Test exact UTC 4h alignment, close `open + 4h - 1ms`, unique opens, four-hour cadence, finite positive OHLC, nonnegative volume, valid OHLC envelope, strict REST closure, and matching BTC/ETH identity sequences.

Reject one-symbol and both-symbol gaps, duplicate equal records, same-identity checksum conflicts, 3h59m/4h+1ms cadence, off-grid UTC hour, open/close mismatch, out-of-window row, NaN/Infinity, zero/negative price, negative volume, and envelope failure. Never forward-fill or silently crop.

Test exact stage counts:

```python
self.assertEqual(len(training.performance_buckets), 8766)
self.assertEqual(len(validation.performance_buckets), 4380)
self.assertEqual(len(holdout.performance_buckets), 5658)
self.assertEqual(len(training.warmup_buckets), 540)
self.assertEqual(len(validation.warmup_buckets), 540)
self.assertEqual(len(holdout.warmup_buckets), 540)
```

Test 539/541 warm-up, wrong first/last warm-up identity, internal gap, mixed performance row, open exactly at start/end, close exactly at end, and one-millisecond boundary probes. The warm-up first open is exactly stage start minus 90 days and last close is stage start minus one millisecond.

- [ ] **Step 2: Write failing nonleakage and publication tests**

Embed distinct training, validation, and holdout sentinel OHLCV strings, including reserved rows in a cross-boundary raw page. Recursively inspect public outcomes, diagnostics, logs, stdout/stderr, exception `str/repr/args/__cause__/__context__`, the release record, and the full byte tree of the training release. No reserved sentinel, validation/holdout hash/ref, raw page bytes, receipt, custody manifest/catalog, object key, or path may appear.

Output keys reject `open`, `high`, `low`, `close`, `volume`, `price`, `return`, `volatility`, `indicator`, `signal`, `rank`, and `metric` except explicitly allowed structural `open_time`/`close_time`. Use known reserved hash/path/cache sentinels to prove the positive public whitelist, not only forbidden field-name filtering.

On any audit failure, no seal root or stage publication pointer exists. Intermediate immutable evidence may remain custody-only but cannot be mistaken for success. Common-history failure returns only the defined structural failure fields.

- [ ] **Step 3: Run Task 7 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_sealer.py" -v
```

Expected: missing stage-manifest/sealer imports.

- [ ] **Step 4: Implement stage-scoped manifests and atomic seal publication**

Audit the complete custody snapshot before returning anything. Derive the three stage-local raw/stage manifests only after full success. Consume only the named `CustodyStageWritersV1.training`, `.validation`, and `.holdout` capabilities and reverify their embedded labels/distinct file identities before writes. Materialize each view through exclusive byte copies of canonical per-bar records, then its own manifests/catalog, and publish the root last. Prohibit raw HTTP/page evidence, full custody/pre-registration records, links, absolute/relative custody paths, and cross-stage object resolvers.

Publish validation/holdout views only in the fixed handle-relative custody children. Byte-copy the already verified training allowlist into the distinct pinned release root and scan the resulting content closure before creating `TrainingReleaseRecordV1`. The release content namespace may contain only catalog-referenced bar/manifests, the catalog, that release record, and one root pointer; outside it, only the exact pinned layout marker and live owner-lease metadata are allowed and neither enters release identity. Any extra/missing object, label mismatch, duplicate/shared backing identity, or swapped custody writer fails before publication.

Persist the release record, rescan/reverify its referenced closure and root identity through retained handles, then publish its record ref as the release root last. A failed scan publishes neither custody seal pointer nor training release pointer.

`PreregistrationSealManifestV1` binds the custody acquisition/chain, common bounds, and exact ordered training/validation/holdout bindings. Exclusive custody publication of its canonical hash is the internal success marker. The public pass returns only the training whitelist, including the release-record hash but not the root-identity payload/hash. Existing pointers/objects must match byte-for-byte; no replacement is allowed.

If common coverage cannot support the frozen first warm-up or final boundary, return `common_history_failed` and stop. The implementation does not calculate an alternative start or mutate `STAGE_WINDOWS`.

- [ ] **Step 5: Enforce import and API isolation**

AST/import tests apply the denylist to transport, acquisition, codecs, raw manifest, sealer, and custody entrypoint. They import no strategy, indicator, risk, backtest, state, reporting, metric, presentation, pandas, pyarrow, NumPy, or `cross_signal_strategy` module. Production decoder/sealer constructors require the opaque custody archive session. `binance_spot_strategy.data.__all__` excludes every custody reader/writer/session, raw kline value, sealer capability, validation/holdout hash/view root, pre-registration seal, and full acquisition catalog.

- [ ] **Step 6: Run complete tests and commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_sealer.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data/stage_manifest_v1.py binance_spot_strategy/data/preregistration_sealer_v1.py tests/binance_spot/fixtures/m2_data_manifests_v1.json tests/binance_spot/test_stage_sealer.py
git commit -m "feat(binance-spot): seal isolated stage manifests"
```

### Task 8: Attested Training Loader And Reserved-Stage Deny Boundary

**Files:**
- Create: `binance_spot_strategy/data/stage_access_v1.py`
- Create: `binance_spot_strategy/data/stage_loader_v1.py`
- Create: `tests/binance_spot/test_stage_access.py`
- Create: `tests/binance_spot/test_stage_loader.py`

**Interfaces:**
- Consumes: Task 2 `AttestedExecutionAdmissionV1`, a physically training-scoped released root, an opaque approved release, exact `TrainingReleaseRecordV1` closure, `TrainingStageId`, declared indicator/performance windows, and expected manifest hashes.
- Produces: runtime-only opaque `VerifiedTrainingReleaseRootV1`, `ApprovedTrainingReleaseV1`, and `TrainingReleasedViewV1`; none is publicly constructible, serializable, picklable, exported, or comparable by fields.
- Produces no public/production `ApprovedTrainingReleaseV1` factory in M2. The only M2 issuer is private test composition that recognizes the one committed synthetic release-record golden hash; it cannot accept an arbitrary caller string/record/`SealPassV1`.
- Produces internal `issue_training_released_view(admission: AttestedExecutionAdmissionV1, verified_root: VerifiedTrainingReleaseRootV1, approved_release: ApprovedTrainingReleaseV1) -> TrainingReleasedViewV1`.
- Produces: immutable `StageBarBucketV1` and `TrainingStageDatasetV1(indicator_buckets, performance_buckets)`; there is no combined `all_bars` accessor.
- Produces: `TrainingStageLoader.load_exact_stage(admission: AttestedExecutionAdmissionV1, stage_id: TrainingStageId, view: TrainingReleasedViewV1, expected_stage_manifest_sha256: str, indicator_window: tuple[datetime, datetime], performance_window: tuple[datetime, datetime]) -> TrainingStageDatasetV1`.
- Produces immutable `ReservedStageAccessRequestV1(stage_activation_id, stage_name, partition_name, expected_stage_manifest_sha256, indicator_window, performance_window)`.
- Produces: `ReservedStageAccessAuthority.require_access(request: ReservedStageAccessRequestV1) -> None` and `DenyAllReservedStageAccessAuthority`; M2's public composition provides no permissive implementation or authority injection point.

`issue_training_released_view` first calls `require_attested_m2_admission` and verifies the current live session/guard/content receipt, then authenticates the current-process `ApprovedTrainingReleaseV1` before any root content read. It rechecks the pinned root volume/file ID, reads the single root pointer and release record each exactly once under retained handles, requires the pointer/record hash to equal the approved hash, verifies `training_release_root_identity_sha256` against the current canonical path-free root audit payload, and verifies the record's exact source-contract/raw/stage/catalog hashes. Only then does it parse `TrainingReleaseCatalogV1` and verify every referenced object. The view binds that approved record hash, an internal training-only one-snapshot reader, catalog/manifest refs, and root identity; it exposes no root/path/general content lookup and cannot resolve an object absent from the catalog.

- [ ] **Step 1: Write failing fail-before-read access tests**

Use spies for admission/session/guard/approval verification, release record, catalog, cache, file open, object read, query, mmap, decompression, and iterator creation. Reject before every content spy is called when admission is absent/foreign/stale/serialized, approval is absent/forged/serialized/cross-process or supplied as a naked expected hash/record/`SealPassV1`, scope/stage/partition type is wrong, root/file identity changed, a training request names a validation/holdout ref, a caller supplies `authorized=True`, a fake token/hash/`StageActivationId`, or a validation proof is replayed for holdout.

Prepopulate a fake reserved-stage cache and prove authorization still occurs before cache lookup. Error codes/logs do not reveal sealed paths, manifest payload, OHLCV, or cache keys.

- [ ] **Step 2: Write failing full-snapshot training validation tests**

Accept only exact training indicator and performance windows from frozen config. Reject a missing/extra/replaced root pointer, release-record hash drift, root-identity drift, source-contract drift, raw/stage/catalog hash-type swap or mismatch, 539/541 warm-up, 8765/8767 performance buckets, first/last/internal gap, disorder, duplicate, extra symbol, missing symbol, mismatched open/close, out-of-catalog/manifest object, object checksum drift, manifest mutation, relabelled reserved manifest, stage raw hash drift, raw-page/custody object, and any validation/holdout-only object.

Simulate the last object being corrupt and prove no earlier bucket was returned or yielded. Read each object once, hash and parse the same byte snapshot, validate the complete dataset, then return immutable tuples.

- [ ] **Step 3: Run Task 8 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_access.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_loader.py" -v
```

Expected: missing access/loader imports.

- [ ] **Step 4: Implement trusted composition and training-only loading**

Task 8 exercises the builder only with a private synthetic session/admission issuer and a test-only approval issuer closure that recognizes the exact committed synthetic `TrainingReleaseRecordV1` golden hash. The issuer is not exported or reachable from CLI input and rejects all caller-created strings, records, and pass objects. The builder verifies admission and approval before record/catalog/root content, supplies the loader only with the opaque training view, validates declared windows against frozen config without opening bar content, then validates the complete record/catalog/manifest/object snapshot, separates warm-up from performance, and returns exact paired `Bar` buckets.

No production admission issuer or final builder composition exists in Task 8. Task 9 first freezes the complete M2 policy/module lock, then enables production session/admission issuance and reruns these tests through that final composition.

M2 intentionally has no production approval issuer for an arbitrary or real training release. A real M2.5 pass reports the release-record hash and stops. A later M3 plan plus explicit user approval must freeze that exact record hash in reviewed configuration before any production training loader can receive an `ApprovedTrainingReleaseV1`.

The reserved loader calls `ReservedStageAccessAuthority` before receiving any store/cache capability. M2 production construction always uses `DenyAllReservedStageAccessAuthority`. The future coordinator port may eventually authorize bound warm-up or one performance bucket, but M2 provides no grant factory, token issuer, activation-success object, SQLite implementation, or public authority constructor.

- [ ] **Step 5: Document deferred activation acceptance**

Tests and docstrings explicitly defer validation/holdout warm-up-without-token, atomic first performance bucket, cohort-wide child authorization, rollback, recovery, nomination, and post-consumption invalidation to their later SQLite/control-plane milestones. M2 claims only a typed port, deny-all production composition, physical stage-view separation, and denial before content access.

- [ ] **Step 6: Run complete tests and commit**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_access.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_stage_loader.py" -v
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
git diff --check
git add binance_spot_strategy/data/stage_access_v1.py binance_spot_strategy/data/stage_loader_v1.py tests/binance_spot/test_stage_access.py tests/binance_spot/test_stage_loader.py
git commit -m "feat(binance-spot): isolate training stage reads"
```

### Task 9: Offline End-To-End Gate, Custody CLI, And M2 Artifact Freeze

**Files:**
- Create: `binance_spot_strategy/config/m2_business_modules.lock.json`
- Create: `binance_spot_strategy/entrypoints/__init__.py`
- Create: `binance_spot_strategy/entrypoints/seal_historical_data.py`
- Modify: `binance_spot_strategy/attested_launcher.py`
- Modify: `binance_spot_strategy/README.md`
- Create: `tests/binance_spot/fixtures/m2_business_artifact_v1.json`
- Create: `tests/binance_spot/test_m2_acceptance.py`

**Interfaces:**
- Consumes: Tasks 1-8 and deterministic generated synthetic full-window data.
- Produces: reviewed M2 source-execution policy/module lock, synthetic artifact/stage golden hashes, and a public-data custody CLI that has no default or automatic execution path.
- Produces: one test-only offline composition that returns existing `SealPassV1` and `TrainingStageDatasetV1` values; no additional production acceptance-result protocol is introduced.
- Produces side-effect-free `seal_historical_data.main(argv: Sequence[str]) -> int`; dependency injection is private test support and cannot be selected by CLI input.

- [ ] **Step 1: Write failing offline end-to-end acceptance tests**

In memory, generate exact two-symbol four-hour data from `2017-10-03T00:00:00Z` to `2026-08-01T00:00:00Z`, with deterministic per-stage sentinel OHLCV strings and preceding server-time evidence. Run archive, acquisition chain, custody manifest, sealer, training release record, final-policy attested session/admission, the private exact-golden synthetic approval issuer, and training loader.

Use a private test-only root adapter with fixed nonproduction volume serial, file ID, and layout identities so the synthetic `TrainingReleaseRecordV1` hash is reproducible. The golden fixture binds those explicit synthetic identities. Production root pinning has no adapter injection point and accepts only live Win32 identities; Task 4 separately exercises its real temporary-root behavior without freezing machine-specific IDs into a golden.

Assert exact frozen manifest/artifact hashes, 540 training indicator buckets, 8,766 training performance buckets, two bars per bucket, cross-boundary reserved-row mutation independence, no raw-page/custody/validation/holdout bytes or hashes in the released tree, and reserved-stage denial with zero content operations. Exercise fail-to-rerun and interrupted-to-recovery acquisition chains through a final successful custody manifest. Do not calculate an indicator, signal, position, trade, return, drawdown, or metric.

- [ ] **Step 2: Write failing CLI safety tests**

The CLI requires distinct absolute `--custody-root` and `--training-release-root` values, both outside every repository/worktree and free of symlink/junction/reparse/alias components. It also requires exact acknowledgement `BINANCE_SPOT_M2_SEALER_CUSTODY_ONLY`. Reject missing arguments, relative/equal/nested or same-file-ID roots, repository roots, nonempty unknown layouts, endpoint/base-url override, credentials, symbols/interval/window override, retry option, and any private/order/account flag before transport construction.

Importing `entrypoints` or `seal_historical_data` must perform zero filesystem mutation, socket, DNS, transport construction, or output; only an `if __name__ == "__main__"` guard calls `main`. Patch transport, socket, DNS, environment proxy, filesystem, stdout/stderr, attestation, root pinning, and owner lock. Prove argument, acknowledgement, attestation, root identity, or owner failure causes zero network and zero stage-content operations. `--help` and no-argument invocations perform no network.

Freeze construction order: parse fixed arguments -> verify acknowledgement -> establish final Task 2 session/admission -> pin distinct external custody/release handles and file IDs -> claim single-owner locks -> open the custody session -> derive the three fixed named custody-stage writers -> open the external training-release writer -> validate recognized layouts/chain -> construct production transport -> acquire/seal/release-record publication -> print only the public training whitelist. Recheck root identity before every write and both publications; inject junction swaps after validation, before first write, and before custody-seal/release-record root publication. The CLI never issues `ApprovedTrainingReleaseV1` and never invokes the training loader.

- [ ] **Step 3: Run Task 9 tests and verify RED**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_m2_acceptance.py" -v
```

Expected: missing entrypoint, M2 module lock, or acceptance composition.

- [ ] **Step 4: Implement the custody-only composition root**

Implement the side-effect-free CLI and internal composition, but keep both `verify-m2` and `seal-historical-data` production modes disabled with `production_policy_not_frozen`. The entrypoint accepts no strategy, report, database, API key, account, order, arbitrary endpoint, public transport factory, or path resolver.

The CLI has no resume/repair/delete mode. A normal failed run publishes its terminal acquisition-chain record but no seal pointer; an interrupted run is closed by the exact recovery record on the next separately authorized invocation. A rerun is a fresh full acquisition run in the same verified append-only chain and may reuse only rehashed identical objects.

- [ ] **Step 5: Freeze and verify the M2 business-module policy**

Create `m2_business_modules.lock.json` from the exact launcher, two bootstrap helpers, ordered M2 business modules, both literal entrypoint bindings, and only the exact execution-admitted runtime, stdlib, NumPy, python-dateutil, six, and tzdata owner trees after all source is present. The fully verified pandas owner tree remains absent from the M2 execution policy, and both pandas and PyArrow roots remain fail-closed. The lock excludes itself and audit-only source commit to avoid recursive tracked identity; the future run identity separately binds the source commit.

Review the raw launcher/bootstrap/module hashes, then enable only literal `verify-m2` and `seal-historical-data`. Establish the production `AttestedExecutionSessionV1`, issue an admission, and run Task 8's actual training-view composition against only the committed synthetic release through the private exact-golden test approval issuer. Compare the same-process closure to `m2_business_artifact_v1.json`; forged/synthetic/cross-process admissions and forged/arbitrary release approvals remain rejected. No real release is loader-approved in M2.

M1 candidate/run fixtures remain untouched and synthetic. The README states that formal admission requires the new content lock and execution closure; `CandidateManifestV1` and M1 synthetic fingerprints are never training-ready. Actual Baseline modules and formal V2 candidate identity remain M3 work.

- [ ] **Step 6: Run the complete offline verification matrix**

```powershell
& 'D:\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests/binance_spot -p "test_*.py" -v
& 'D:\Programs\Python\Python313\python.exe' -I -S -B binance_spot_strategy/attested_launcher.py verify-m2
git diff --check
rg -n "api[_-]?key|secret|signature|create_order|order/place|/api/v3/order|cross_signal_strategy|G:\\" binance_spot_strategy tests/binance_spot
```

Expected: every Binance Spot test passes; final-policy session/admission and M2 closure match; source search contains only explicit rejection/test assertions, documentation, and no credential/order implementation or runtime Cross-Signal/G-drive dependency. Import/CLI tests prove no accidental network. No real network or external data root was touched.

- [ ] **Step 7: Run two independent review gates**

Dispatch one specification reviewer against design Sections 4, 9.2, 10, 14, and 15, and one adversarial reviewer against attestation/session forgery, bootstrap bypass, native/data TOCTOU, cross-page stage contamination, acquisition-chain omission, data leakage, stage escape, raw mutation, root/reparse alias, cache-before-auth, partial-return, and accidental-network attacks. Resolve every Critical/Important finding and rerun focused plus complete tests. Record limitations: local reproducibility is not TPM/remote attestation; already-running CPython/OS loader pages and Windows system DLLs are local TCB; physical WORM depends on approved storage/ACL/backup policy.

- [ ] **Step 8: Update README and commit the offline M2 milestone**

README must state exactly what passed and what did not occur: attestation protocol and current M2 closure passed; synthetic full-window seal/load passed; no Binance request, real stage manifest, formal candidate, training/backtest result, validation/holdout activation, paper run, API key, or order path exists.
This plan is committed as the approval artifact before the implementation worktree is created, so Task 9 does not stage it again.


```powershell
git add binance_spot_strategy tests/binance_spot
git commit -m "feat(binance-spot): complete offline M2 data isolation"
```

## M2 Acceptance Traceability

| Frozen requirement | M2 evidence | Expected outcome after implementation |
|---|---|---|
| Complete runtime/dependency content | Task 1 exact content lock and current-tree verification | Locally attested by M2 verification |
| Actual loaded business bytes | Task 2 same-buffer compile/exec session/guard closure | Attested for M2; Baseline closure waits for M3 modules |
| Official raw response + metadata | Tasks 3-6 body/prefix archive, receipts, append-only runs, and custody manifest | Offline behavior proven through fake transport |
| Verified server-time closure | Tasks 5-6 exact request-pair evidence and strict `<` test | Offline behavior proven; real proof waits for authorized custody run |
| Common continuous BTC/ETH history | Task 7 structural audit | Synthetic proof only; real result remains unknown until M2.5 |
| Exact windows and 540 warm-up | Tasks 7-8 golden stage manifests and loader | Synthetic proof completed |
| Unreleased-stage isolation | Tasks 7-8 stage-local views, trusted authority port, deny-all composition | M2 training-only boundary proven |
| Validation warm-up and atomic first bucket | Later cohort-control/SQLite milestone | Explicitly deferred; M2 cannot claim pass |
| Holdout nomination/token rules | Later nomination/holdout milestone | Explicitly deferred; M2 cannot claim pass |
| Baseline A/B, risk, matching, backtest | M3-M5 | Not implemented in M2 |
| Public-data custody run | Separate user authorization below | Still not executed |

## M2.5 Separate Real-Data Custody Authorization Gate

Do not execute the custody entrypoint as part of this plan. After the offline M2 commit and user review, stop and request all of the following in one explicit authorization:

1. Exact absolute full-custody root.
2. Exact absolute training-release root, distinct from and not nested under custody.
3. Approval to contact only official public `GET /api/v3/time` and `GET /api/v3/klines` on `data-api.binance.vision` without credentials.
4. Approval of the custody-root filesystem/ACL/backup policy; content addressing detects mutation but is not physical WORM by itself.
5. Confirmation that only the dedicated custody/sealer TCB may inspect unreleased validation/holdout OHLCV and may emit only the frozen nonrevealing public summary.

Immediately before that separately authorized run, recheck the official Binance endpoint, payload, time, limit, and rate-limit documentation. Construct the exact command only after the two approved roots are known; do not place guessed/example roots in an executable command.

If the real sealer fails common coverage, cadence, duplicate, closure, checksum, or any other structural rule, preserve the evidence, publish no seal root, and stop for a user-approved design/config revision. Never change dates, reduce warm-up, fill a bar, swap source, or rerun a forbidden evaluation stage.

If it passes, release only the training stage view and report only structural status/common bounds/counts plus training raw/stage/catalog/release-record hashes. Stop without issuing a training approval or opening the release through the loader. Do not expose validation/holdout refs and do not calculate strategy indicators or performance. The next design/plan cycle is M3: after explicit user approval freezes that exact release-record hash, independently implement and golden-test Baseline A and Baseline B against the approved released training loader while preserving separate candidate identities.
