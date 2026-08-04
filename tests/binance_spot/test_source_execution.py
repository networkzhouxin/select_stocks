from __future__ import annotations

from copy import deepcopy
import ctypes
from ctypes import wintypes
from dataclasses import FrozenInstanceError
import hashlib
import importlib.abc
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unicodedata
import unittest
from unittest import mock

from binance_spot_strategy.identity import (
    AttestedExecutionAdmissionV1,
    AttestedExecutionSessionV1,
    BusinessModuleBindingV1,
    EntrypointBindingV1,
    ExecutionClosureV1,
    SourceExecutionPolicyV1,
    SourceExecutionValidationError,
    establish_attested_execution,
    issue_m2_admission,
    require_attested_m2_admission,
    source_execution_policy_from_payload,
    source_execution_policy_hash,
    verify_execution_closure,
)
from binance_spot_strategy.identity.digests import hash_canonical_payload
from binance_spot_strategy.identity.manifests_v1 import (
    CandidateManifestV1,
    RunCommonV1,
)
from binance_spot_strategy.identity.source_execution_v1 import (
    _deactivate_attested_execution_for_test,
    _establish_attested_execution_for_test,
    _verified_resource_bytes,
)
from tests.binance_spot.test_dependency_contents import (
    SyntheticContentEnvironment,
)
from binance_spot_strategy.identity.dependency_contents_v1 import (
    _build_dependency_content_lock,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPOSITORY_ROOT / "binance_spot_strategy" / "attested_launcher.py"
DEPENDENCY_HELPER = (
    REPOSITORY_ROOT
    / "binance_spot_strategy"
    / "identity"
    / "dependency_contents_v1.py"
)
SOURCE_HELPER = (
    REPOSITORY_ROOT
    / "binance_spot_strategy"
    / "identity"
    / "source_execution_v1.py"
)


class StringSubclass(str):
    pass


def raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tracked_sha(raw: bytes) -> str:
    text = raw.decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return raw_sha(normalized.encode("utf-8"))


M2_EXECUTION_OWNER_NAMES = (
    "CPython",
    "stdlib",
    "numpy",
    "python-dateutil",
    "six",
    "tzdata",
)


def owner_tree_sha256(tree) -> str:
    return hash_canonical_payload(
        {
            "schema_version": "dependency_owner_tree_binding_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
            "tree": tree.to_payload(),
        }
    )


def binding_payload(
    role: str,
    module_name: str,
    logical_path: str,
    raw: bytes,
) -> dict[str, str]:
    return {
        "role": role,
        "module_name": module_name,
        "logical_path": logical_path,
        "tracked_source_sha256": tracked_sha(raw),
        "raw_artifact_sha256": raw_sha(raw),
    }


class SyntheticExecutionFixture:
    def __init__(
        self,
        directory: str,
        *,
        entry_source: bytes = b"def main():\n    return 'synthetic-ok'\n",
    ) -> None:
        self.root = Path(directory)
        self.repository = self.root / "checkout"
        self.repository.mkdir()
        self.content = SyntheticContentEnvironment(
            str(self.root / "content-environment")
        )
        self.lock = _build_dependency_content_lock(self.content.environment)

        files = {
            "binance_spot_strategy/attested_launcher.py": LAUNCHER_PATH.read_bytes(),
            "binance_spot_strategy/identity/dependency_contents_v1.py": (
                DEPENDENCY_HELPER.read_bytes()
            ),
            "binance_spot_strategy/identity/source_execution_v1.py": (
                SOURCE_HELPER.read_bytes()
            ),
            "synthetic_app/__init__.py": b"PACKAGE_MARKER = 'controlled'\n",
            "synthetic_app/entry.py": entry_source,
        }
        for logical_path, raw in files.items():
            target = self.repository / Path(logical_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

        trees_by_name = {tree.owner_name: tree for tree in self.lock.trees}
        tree_hashes = [
            owner_tree_sha256(trees_by_name[name])
            for name in M2_EXECUTION_OWNER_NAMES
        ]
        self.payload: dict[str, object] = {
            "schema_version": "source_execution_policy_v1",
            "dependency_content_lock_sha256": hash_canonical_payload(
                self.lock.to_payload()
            ),
            "bootstrap_launcher_raw_sha256": raw_sha(
                files["binance_spot_strategy/attested_launcher.py"]
            ),
            "bootstrap_modules": [
                binding_payload(
                    "bootstrap",
                    "binance_spot_strategy.identity.dependency_contents_v1",
                    "binance_spot_strategy/identity/dependency_contents_v1.py",
                    files[
                        "binance_spot_strategy/identity/dependency_contents_v1.py"
                    ],
                ),
                binding_payload(
                    "bootstrap",
                    "binance_spot_strategy.identity.source_execution_v1",
                    "binance_spot_strategy/identity/source_execution_v1.py",
                    files[
                        "binance_spot_strategy/identity/source_execution_v1.py"
                    ],
                ),
            ],
            "entrypoints": [
                {
                    "mode": "task2-synthetic",
                    "module_name": "synthetic_app.entry",
                    "callable_name": "main",
                }
            ],
            "business_modules": [
                binding_payload(
                    "business",
                    "synthetic_app",
                    "synthetic_app/__init__.py",
                    files["synthetic_app/__init__.py"],
                ),
                binding_payload(
                    "business",
                    "synthetic_app.entry",
                    "synthetic_app/entry.py",
                    files["synthetic_app/entry.py"],
                ),
            ],
            "dependency_owner_tree_sha256s": tree_hashes,
        }
        self.policy = source_execution_policy_from_payload(self.payload)
        self.sessions: list[AttestedExecutionSessionV1] = []

    def establish(self) -> AttestedExecutionSessionV1:
        session = _establish_attested_execution_for_test(
            self.policy,
            self.lock,
            self.content.environment,
            self.repository,
        )
        self.sessions.append(session)
        return session

    def close(self) -> None:
        for session in reversed(self.sessions):
            _deactivate_attested_execution_for_test(session)


def loaded_process_image_paths() -> frozenset[str]:
    if sys.platform != "win32":
        return frozenset()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = (
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    capacity = 4096
    modules = (wintypes.HMODULE * capacity)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(
        process,
        modules,
        ctypes.sizeof(modules),
        ctypes.byref(needed),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    count = min(capacity, needed.value // ctypes.sizeof(wintypes.HMODULE))
    paths: set[str] = set()
    for index in range(count):
        buffer = ctypes.create_unicode_buffer(32768)
        length = psapi.GetModuleFileNameExW(
            process, modules[index], buffer, len(buffer)
        )
        if length:
            paths.add(buffer.value.casefold())
    return frozenset(paths)


def forbidden_process_images() -> frozenset[str]:
    return frozenset(
        path
        for path in loaded_process_image_paths()
        if "pandas" in path or "pyarrow" in path
    )


class SourceExecutionPolicyTests(unittest.TestCase):
    def _fixture(self, directory: str) -> SyntheticExecutionFixture:
        return SyntheticExecutionFixture(directory)

    def test_policy_parses_to_exact_frozen_records(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            policy = fixture.policy
        self.assertIs(type(policy), SourceExecutionPolicyV1)
        self.assertIs(type(policy.bootstrap_modules), tuple)
        self.assertIs(type(policy.bootstrap_modules[0]), BusinessModuleBindingV1)
        self.assertIs(type(policy.entrypoints[0]), EntrypointBindingV1)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            policy.schema_version = "changed"

    def test_missing_unknown_and_nonexact_fields_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            payload = fixture.payload
            targets = (
                (),
                ("bootstrap_modules", 0),
                ("entrypoints", 0),
                ("business_modules", 0),
            )
            for target in targets:
                for operation in ("missing", "unknown"):
                    mutated = deepcopy(payload)
                    current = mutated
                    for component in target:
                        current = current[component]
                    assert isinstance(current, dict)
                    if operation == "missing":
                        current.pop(next(iter(current)))
                    else:
                        current["unknown"] = "rejected"
                    with self.subTest(target=target, operation=operation):
                        with self.assertRaises(SourceExecutionValidationError):
                            source_execution_policy_from_payload(mutated)

            for field in (
                "schema_version",
                "dependency_content_lock_sha256",
                "bootstrap_launcher_raw_sha256",
            ):
                mutated = deepcopy(payload)
                mutated[field] = StringSubclass(str(mutated[field]))
                with self.subTest(field=field):
                    with self.assertRaises(SourceExecutionValidationError):
                        source_execution_policy_from_payload(mutated)

    def test_versions_roles_names_paths_callables_and_modes_are_literal(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            mutations: list[tuple[str, dict[str, object]]] = []
            for value in ("v2", "", True, 1):
                item = deepcopy(fixture.payload)
                item["schema_version"] = value
                mutations.append(("version", item))
            for field, value in (
                ("role", "strategy"),
                ("module_name", "bad-name"),
                ("logical_path", "../escape.py"),
                ("logical_path", "C:/escape.py"),
            ):
                item = deepcopy(fixture.payload)
                item["business_modules"][0][field] = value
                mutations.append((field, item))
            for field, value in (
                ("mode", "caller-selected"),
                ("module_name", "bad/module"),
                ("callable_name", "bad.call"),
            ):
                item = deepcopy(fixture.payload)
                item["entrypoints"][0][field] = value
                mutations.append((field, item))
            for label, item in mutations:
                with self.subTest(label=label):
                    with self.assertRaises(SourceExecutionValidationError):
                        source_execution_policy_from_payload(item)

    def test_order_duplicates_and_nfc_casefold_collisions_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            variants: list[dict[str, object]] = []

            reversed_business = deepcopy(fixture.payload)
            reversed_business["business_modules"].reverse()
            variants.append(reversed_business)

            duplicate_business = deepcopy(fixture.payload)
            duplicate_business["business_modules"].append(
                deepcopy(duplicate_business["business_modules"][-1])
            )
            variants.append(duplicate_business)

            colliding_business = deepcopy(fixture.payload)
            colliding_business["business_modules"][0]["module_name"] = "SYNTHETIC_APP.ENTRY"
            variants.append(colliding_business)

            duplicate_mode = deepcopy(fixture.payload)
            duplicate_mode["entrypoints"].append(
                deepcopy(duplicate_mode["entrypoints"][0])
            )
            variants.append(duplicate_mode)

            for index, item in enumerate(variants):
                with self.subTest(index=index):
                    with self.assertRaises(SourceExecutionValidationError):
                        source_execution_policy_from_payload(item)

    def test_policy_admits_exact_ordered_six_tree_subset_and_excludes_pandas(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            trees_by_name = {tree.owner_name: tree for tree in fixture.lock.trees}
            self.assertEqual(
                tuple(tree.owner_name for tree in fixture.lock.trees),
                (
                    "CPython",
                    "stdlib",
                    "numpy",
                    "pandas",
                    "python-dateutil",
                    "six",
                    "tzdata",
                ),
            )
            self.assertEqual(
                fixture.policy.dependency_owner_tree_sha256s,
                tuple(
                    owner_tree_sha256(trees_by_name[name])
                    for name in M2_EXECUTION_OWNER_NAMES
                ),
            )
            self.assertNotIn(
                owner_tree_sha256(trees_by_name["pandas"]),
                fixture.policy.dependency_owner_tree_sha256s,
            )

            reordered = deepcopy(fixture.payload)
            reordered["dependency_owner_tree_sha256s"].reverse()
            reordered_policy = source_execution_policy_from_payload(reordered)
            with self.assertRaisesRegex(
                SourceExecutionValidationError, "ordered|owner trees"
            ):
                _establish_attested_execution_for_test(
                    reordered_policy,
                    fixture.lock,
                    fixture.content.environment,
                    fixture.repository,
                )

            with_pandas = deepcopy(fixture.payload)
            with_pandas["dependency_owner_tree_sha256s"].insert(
                3, owner_tree_sha256(trees_by_name["pandas"])
            )
            with_pandas_policy = source_execution_policy_from_payload(with_pandas)
            with self.assertRaisesRegex(
                SourceExecutionValidationError, "pandas|owner trees"
            ):
                _establish_attested_execution_for_test(
                    with_pandas_policy,
                    fixture.lock,
                    fixture.content.environment,
                    fixture.repository,
                )

    def test_both_reviewed_bootstrap_helpers_are_mandatory_and_exact(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            variants = []
            missing = deepcopy(fixture.payload)
            missing["bootstrap_modules"].pop()
            variants.append(missing)
            relabelled = deepcopy(fixture.payload)
            relabelled["bootstrap_modules"][0]["module_name"] = "synthetic.bootstrap"
            variants.append(relabelled)
            misplaced = deepcopy(fixture.payload)
            misplaced["bootstrap_modules"][0]["logical_path"] = "other.py"
            variants.append(misplaced)
            for item in variants:
                with self.assertRaises(SourceExecutionValidationError):
                    source_execution_policy_from_payload(item)

    def test_policy_hash_is_checkout_root_independent_and_mutation_sensitive(self) -> None:
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first = self._fixture(first_dir)
            second = self._fixture(second_dir)
            self.assertEqual(
                source_execution_policy_hash(first.policy),
                source_execution_policy_hash(second.policy),
            )
            mutated = deepcopy(first.payload)
            mutated["entrypoints"][0]["callable_name"] = "other"
            changed = source_execution_policy_from_payload(mutated)
            self.assertNotEqual(
                source_execution_policy_hash(first.policy),
                source_execution_policy_hash(changed),
            )


class ControlledSourceExecutionTests(unittest.TestCase):
    def test_raw_and_tracked_hashes_use_one_crlf_execution_snapshot(self) -> None:
        source = b"VALUE = 'crlf'\r\ndef main():\r\n    return VALUE\r\n"
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory, entry_source=source)
            try:
                binding = fixture.policy.business_modules[-1]
                self.assertNotEqual(
                    binding.raw_artifact_sha256,
                    binding.tracked_source_sha256,
                )
                session = fixture.establish()
                self.assertEqual(session.entrypoint_result, "crlf")
                expected = raw_sha(source)
                actual = next(
                    item.raw_artifact_sha256
                    for item in session.closure.business_modules
                    if item.module_name == "synthetic_app.entry"
                )
                self.assertEqual(actual, expected)
            finally:
                fixture.close()

    def test_shadow_sys_path_and_meta_path_finder_cannot_replace_business_source(self) -> None:
        class MaliciousFinder(importlib.abc.MetaPathFinder):
            called = False

            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("synthetic_app"):
                    self.called = True
                    raise AssertionError("custom finder reached")
                return None

        with TemporaryDirectory() as directory, TemporaryDirectory() as shadow_dir:
            fixture = SyntheticExecutionFixture(directory)
            shadow = Path(shadow_dir) / "synthetic_app"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("", encoding="utf-8")
            (shadow / "entry.py").write_text(
                "def main(): return 'shadow'", encoding="utf-8"
            )
            finder = MaliciousFinder()
            sys.path.insert(0, shadow_dir)
            sys.meta_path.insert(0, finder)
            try:
                session = fixture.establish()
                self.assertEqual(session.entrypoint_result, "synthetic-ok")
                self.assertFalse(finder.called)
            finally:
                fixture.close()
                sys.meta_path.remove(finder)
                sys.path.remove(shadow_dir)

    def test_prefilled_business_sys_modules_entry_fails_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            false_module = types.ModuleType("synthetic_app.entry")
            sys.modules["synthetic_app.entry"] = false_module
            try:
                with self.assertRaisesRegex(
                    SourceExecutionValidationError, "prefilled"
                ):
                    fixture.establish()
            finally:
                sys.modules.pop("synthetic_app.entry", None)
                sys.modules.pop("synthetic_app", None)
                fixture.close()

    def test_unlisted_dependency_import_is_rejected_despite_importable_shadow(self) -> None:
        source = b"import evil_dependency\ndef main():\n    return evil_dependency.VALUE\n"
        with TemporaryDirectory() as directory, TemporaryDirectory() as shadow_dir:
            fixture = SyntheticExecutionFixture(directory, entry_source=source)
            Path(shadow_dir, "evil_dependency.py").write_text(
                "VALUE = 'escaped'\n", encoding="utf-8"
            )
            sys.path.insert(0, shadow_dir)
            try:
                with self.assertRaisesRegex(
                    SourceExecutionValidationError, "undeclared import"
                ):
                    fixture.establish()
            finally:
                fixture.close()
                sys.path.remove(shadow_dir)
                sys.modules.pop("evil_dependency", None)
                sys.modules.pop("synthetic_app", None)
                sys.modules.pop("synthetic_app.entry", None)

    def test_pandas_and_pyarrow_names_fail_before_resolver_bytes_exec_or_native_load(self) -> None:
        class HostileFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def __init__(self, package_file: Path, data_file: Path) -> None:
                self.package_file = package_file
                self.data_file = data_file
                self.called = False
                self.package_bytes_read = False
                self.package_data_read = False
                self.module_created = False
                self.module_executed = False
                self.native_load_attempted = False

            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "pandas"
                    or fullname.startswith("pandas.")
                    or fullname == "pyarrow"
                    or fullname.startswith("pyarrow.")
                ):
                    self.called = True
                    self.package_file.read_bytes()
                    self.package_bytes_read = True
                    self.data_file.read_bytes()
                    self.package_data_read = True
                    self.native_load_attempted = True
                    return importlib.util.spec_from_loader(fullname, self)
                return None

            def create_module(self, spec):
                self.module_created = True
                return None

            def exec_module(self, module):
                self.module_executed = True

        forbidden_names = ("pandas", "pandas.core", "pyarrow", "pyarrow.lib")
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            hostile_root = Path(directory) / "hostile"
            hostile_root.mkdir()
            package_file = hostile_root / "package.py"
            package_file.write_bytes(b"HOSTILE = True\n")
            data_file = hostile_root / "package-data.bin"
            data_file.write_bytes(b"hostile-data")
            hostile = HostileFinder(package_file, data_file)
            try:
                session = fixture.establish()
                sys.meta_path.append(hostile)
                before_images = forbidden_process_images()
                for fullname in forbidden_names:
                    with self.subTest(fullname=fullname):
                        with mock.patch.object(
                            importlib.machinery.PathFinder,
                            "find_spec",
                            wraps=importlib.machinery.PathFinder.find_spec,
                        ) as path_finder:
                            with self.assertRaisesRegex(
                                SourceExecutionValidationError,
                                "pandas|pyarrow|forbidden",
                            ):
                                session._guard.find_spec(fullname)
                            with self.assertRaisesRegex(
                                SourceExecutionValidationError,
                                "pandas|pyarrow|forbidden",
                            ):
                                importlib.import_module(fullname)
                        path_finder.assert_not_called()
                        root_name = fullname.partition(".")[0]
                        self.assertNotIn(root_name, sys.modules)
                        self.assertNotIn(fullname, sys.modules)
                self.assertFalse(hostile.called)
                self.assertFalse(hostile.package_bytes_read)
                self.assertFalse(hostile.package_data_read)
                self.assertFalse(hostile.module_created)
                self.assertFalse(hostile.module_executed)
                self.assertFalse(hostile.native_load_attempted)
                self.assertEqual(forbidden_process_images(), before_images)
            finally:
                while hostile in sys.meta_path:
                    sys.meta_path.remove(hostile)
                fixture.close()
                for fullname in forbidden_names:
                    sys.modules.pop(fullname, None)

    def test_prefilled_pandas_and_pyarrow_direct_and_descendant_modules_fail_closed(self) -> None:
        forbidden_names = ("pandas", "pandas.core", "pyarrow", "pyarrow.lib")
        for fullname in forbidden_names:
            with self.subTest(fullname=fullname), TemporaryDirectory() as directory:
                fixture = SyntheticExecutionFixture(directory)
                sentinel = types.ModuleType(fullname)
                sys.modules[fullname] = sentinel
                try:
                    before_images = forbidden_process_images()
                    with self.assertRaisesRegex(
                        SourceExecutionValidationError,
                        "prefilled|pandas|pyarrow|forbidden",
                    ):
                        fixture.establish()
                    self.assertIs(sys.modules.get(fullname), sentinel)
                    self.assertEqual(forbidden_process_images(), before_images)
                finally:
                    sys.modules.pop(fullname, None)
                    fixture.close()
                self.assertNotIn(fullname, sys.modules)

    def test_session_pins_only_the_six_execution_admitted_owner_trees(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                trees_by_name = {tree.owner_name: tree for tree in fixture.lock.trees}
                admitted_paths = set(session._dependency_snapshots)
                for owner_name in M2_EXECUTION_OWNER_NAMES:
                    with self.subTest(owner_name=owner_name):
                        self.assertTrue(
                            {item.logical_path for item in trees_by_name[owner_name].files}
                            <= admitted_paths
                        )
                self.assertTrue(
                    {item.logical_path for item in trees_by_name["pandas"].files}
                    .isdisjoint(admitted_paths)
                )
            finally:
                fixture.close()

    def test_source_digest_mismatch_and_post_execution_change_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            target = fixture.repository / "synthetic_app" / "entry.py"
            target.write_bytes(b"def main():\n    return 'changed-before'\n")
            with self.assertRaisesRegex(SourceExecutionValidationError, "digest"):
                fixture.establish()
            fixture.close()

        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                target = fixture.repository / "synthetic_app" / "entry.py"
                try:
                    target.write_bytes(b"def main():\n    return 'changed-after'\n")
                except PermissionError:
                    admission = issue_m2_admission(session)
                    self.assertIs(
                        require_attested_m2_admission(admission), admission
                    )
                else:
                    with self.assertRaises(SourceExecutionValidationError):
                        issue_m2_admission(session)
            finally:
                fixture.close()

    def test_guard_blocks_direct_source_and_package_data_reads(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                target = fixture.repository / "synthetic_app" / "entry.py"
                with self.assertRaisesRegex(PermissionError, "attested guard"):
                    target.read_bytes()
                self.assertEqual(
                    _verified_resource_bytes(
                        session, "synthetic_app.entry", "synthetic_app/entry.py"
                    ),
                    b"def main():\n    return 'synthetic-ok'\n",
                )
                with self.assertRaises(SourceExecutionValidationError):
                    _verified_resource_bytes(
                        session, "synthetic_app.entry", "../escape.py"
                    )
            finally:
                fixture.close()

    @unittest.skipUnless(sys.platform == "win32", "Windows retained-handle contract")
    def test_dependency_native_handle_denies_swap_delete_and_restore(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            native = fixture.content.prefix / "python3.dll"
            original = native.read_bytes()
            replacement = fixture.root / "replacement.dll"
            replacement.write_bytes(b"replacement")
            try:
                fixture.establish()
                with self.assertRaises(PermissionError):
                    os.replace(replacement, native)
                with self.assertRaises(PermissionError):
                    native.unlink()
                self.assertEqual(
                    _verified_resource_bytes(
                        fixture.sessions[-1], "CPython", "python3.dll"
                    ),
                    original,
                )
            finally:
                fixture.close()


class AttestedAdmissionTests(unittest.TestCase):
    def test_closure_is_stable_exact_and_verified_against_live_session(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                closure = session.closure
                self.assertIs(type(closure), ExecutionClosureV1)
                self.assertEqual(
                    closure.execution_policy_sha256,
                    source_execution_policy_hash(fixture.policy),
                )
                verify_execution_closure(closure, fixture.policy, session)
                forged = ExecutionClosureV1(
                    schema_version=closure.schema_version,
                    execution_policy_sha256="0" * 64,
                    dependency_content_lock_sha256=closure.dependency_content_lock_sha256,
                    business_source_tree_sha256=closure.business_source_tree_sha256,
                    business_artifact_manifest_sha256=closure.business_artifact_manifest_sha256,
                    business_modules=closure.business_modules,
                )
                with self.assertRaises(SourceExecutionValidationError):
                    verify_execution_closure(forged, fixture.policy, session)
            finally:
                fixture.close()

    def test_session_and_admission_are_opaque_nonserializable_capabilities(self) -> None:
        for value_type in (AttestedExecutionSessionV1, AttestedExecutionAdmissionV1):
            with self.subTest(value_type=value_type.__name__):
                with self.assertRaises(TypeError):
                    value_type()
                with self.assertRaises(TypeError):
                    type("Forged", (value_type,), {})
                forged = object.__new__(value_type)
                with self.assertRaises(TypeError):
                    pickle.dumps(forged)
                self.assertFalse(hasattr(forged, "to_payload"))

    def test_only_live_same_process_session_can_issue_and_require_admission(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                admission = issue_m2_admission(session)
                self.assertIs(
                    require_attested_m2_admission(admission), admission
                )

                forged = object.__new__(AttestedExecutionAdmissionV1)
                with self.assertRaises(SourceExecutionValidationError):
                    require_attested_m2_admission(forged)
                for value in (
                    "0" * 64,
                    {"verified": True},
                    object.__new__(CandidateManifestV1),
                    object.__new__(RunCommonV1),
                ):
                    with self.subTest(value_type=type(value).__name__):
                        with self.assertRaises(SourceExecutionValidationError):
                            require_attested_m2_admission(value)

                _deactivate_attested_execution_for_test(session)
                with self.assertRaisesRegex(
                    SourceExecutionValidationError, "inactive|untrusted"
                ):
                    require_attested_m2_admission(admission)
            finally:
                fixture.close()

    def test_exact_type_forge_with_copied_fields_and_cross_process_material_fail(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            try:
                session = fixture.establish()
                admission = issue_m2_admission(session)
                forged = object.__new__(AttestedExecutionAdmissionV1)
                for slot in AttestedExecutionAdmissionV1.__slots__:
                    if slot == "__weakref__":
                        continue
                    try:
                        object.__setattr__(forged, slot, getattr(admission, slot))
                    except AttributeError:
                        pass
                with self.assertRaises(SourceExecutionValidationError):
                    require_attested_m2_admission(forged)

                script = "import os; print(os.getpid())"
                completed = subprocess.run(
                    [sys.executable, "-B", "-c", script],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertNotEqual(int(completed.stdout.strip()), os.getpid())
                cross_process = object.__new__(AttestedExecutionAdmissionV1)
                with self.assertRaises(SourceExecutionValidationError):
                    require_attested_m2_admission(cross_process)
            finally:
                fixture.close()

    def test_test_issuer_cannot_mint_reserved_production_admission(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            payload = deepcopy(fixture.payload)
            payload["entrypoints"] = [
                {
                    "mode": "verify-m2",
                    "module_name": "synthetic_app.entry",
                    "callable_name": "main",
                }
            ]
            policy = source_execution_policy_from_payload(payload)
            with self.assertRaisesRegex(
                SourceExecutionValidationError, "production_policy_not_frozen"
            ):
                _establish_attested_execution_for_test(
                    policy,
                    fixture.lock,
                    fixture.content.environment,
                    fixture.repository,
                )


class IsolatedLauncherTests(unittest.TestCase):
    def test_launcher_requires_exact_isolated_runtime_and_blocks_production_mode(self) -> None:
        plain = subprocess.run(
            [sys.executable, "-B", str(LAUNCHER_PATH), "verify-m2"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(plain.returncode, 0)
        self.assertIn("isolated_runtime_required", plain.stderr)

        isolated = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(LAUNCHER_PATH),
                "verify-m2",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(isolated.returncode, 0)
        self.assertIn("production_policy_not_frozen", isolated.stderr)
        self.assertNotIn(str(REPOSITORY_ROOT), isolated.stderr)

    def test_launcher_rejects_arbitrary_mode_without_importing_business_package(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(LAUNCHER_PATH),
                "caller.module:callable",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unsupported_mode", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_public_establishment_runs_real_task1_verifier_before_policy_failure(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticExecutionFixture(directory)
            with self.assertRaises(SourceExecutionValidationError):
                establish_attested_execution(fixture.policy, fixture.lock)


if __name__ == "__main__":
    unittest.main()
