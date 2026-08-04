from __future__ import annotations

from copy import deepcopy
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from binance_spot_strategy.identity.dependency_contents_v1 import (
    ContentFileV1,
    ContentTreeV1,
    DependencyContentLockError,
    DependencyContentLockV1,
    VerifiedContentTreeV1,
    VerifiedDependencyContentV1,
    dependency_content_lock_hash,
    load_dependency_content_lock,
    verify_current_dependency_contents,
)
from binance_spot_strategy.identity.dependency_contents_v1 import (
    _DistributionInputV1,
    _ScanEnvironmentV1,
    _build_dependency_content_lock,
    _snapshot_file,
    generate_dependency_content_lock,
    _verify_dependency_contents_in_environment,
)
from binance_spot_strategy.identity.dependency_lock_v1 import (
    load_semantic_dependency_lock,
    semantic_dependency_lock_hash,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


def synthetic_payload() -> dict[str, object]:
    semantic = load_semantic_dependency_lock()
    distributions = semantic["distributions"]
    trees = [
        {
            "owner_kind": "cpython_runtime",
            "owner_name": "CPython",
            "owner_version": semantic["python"]["version"],
            "files": [
                {
                    "logical_path": "python.exe",
                    "artifact_kind": "runtime_binary",
                    "size_bytes": 3,
                    "sha256": SHA_A,
                }
            ],
        },
        {
            "owner_kind": "cpython_stdlib",
            "owner_name": "stdlib",
            "owner_version": semantic["python"]["version"],
            "files": [
                {
                    "logical_path": "Lib/os.py",
                    "artifact_kind": "source",
                    "size_bytes": 4,
                    "sha256": SHA_B,
                }
            ],
        },
    ]
    for index, distribution in enumerate(distributions):
        trees.append(
            {
                "owner_kind": "distribution",
                "owner_name": distribution["name"],
                "owner_version": distribution["version"],
                "files": [
                    {
                        "logical_path": f"Lib/site-packages/p{index}.py",
                        "artifact_kind": "source",
                        "size_bytes": index + 1,
                        "sha256": format(index + 1, "064x"),
                    }
                ],
            }
        )
    return {
        "schema_version": "dependency_content_lock_v1",
        "numeric_protocol_version": "numeric_protocol_v1",
        "semantic_dependency_lock_sha256": semantic_dependency_lock_hash(semantic),
        "trees": trees,
    }


class DependencyContentSchemaTests(unittest.TestCase):
    def _write(self, directory: str, raw: bytes) -> Path:
        path = Path(directory, "dependency.lock.json")
        path.write_bytes(raw)
        return path

    def _load_payload(self, payload: object) -> DependencyContentLockV1:
        with TemporaryDirectory() as directory:
            path = self._write(
                directory,
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
            return load_dependency_content_lock(path)

    def test_checked_synthetic_payload_parses_to_strict_immutable_records(self) -> None:
        lock = self._load_payload(synthetic_payload())

        self.assertIs(type(lock), DependencyContentLockV1)
        self.assertIs(type(lock.trees), tuple)
        self.assertIs(type(lock.trees[0]), ContentTreeV1)
        self.assertIs(type(lock.trees[0].files), tuple)
        self.assertIs(type(lock.trees[0].files[0]), ContentFileV1)
        with self.assertRaises((AttributeError, TypeError)):
            lock.trees[0].files[0].size_bytes = 9

    def test_missing_and_unknown_keys_are_rejected_at_every_level(self) -> None:
        payload = synthetic_payload()
        targets = (payload, payload["trees"][0], payload["trees"][0]["files"][0])
        for target in targets:
            for operation in ("missing", "unknown"):
                mutated = deepcopy(payload)
                if target is payload:
                    current = mutated
                elif target is payload["trees"][0]:
                    current = mutated["trees"][0]
                else:
                    current = mutated["trees"][0]["files"][0]
                if operation == "missing":
                    current.pop(next(iter(current)))
                else:
                    current["unknown"] = "rejected"
                with self.subTest(level=len(current), operation=operation):
                    with self.assertRaises(DependencyContentLockError):
                        self._load_payload(mutated)

    def test_exact_scalar_types_and_canonical_digests_are_required(self) -> None:
        payload = synthetic_payload()
        mutations = []
        for field in ("schema_version", "numeric_protocol_version"):
            mutated = deepcopy(payload)
            mutated[field] = StringSubclass(mutated[field])
            mutations.append(mutated)
        mutated = deepcopy(payload)
        mutated["semantic_dependency_lock_sha256"] = StringSubclass(
            mutated["semantic_dependency_lock_sha256"]
        )
        mutations.append(mutated)
        mutated = deepcopy(payload)
        mutated["trees"][0]["files"][0]["sha256"] = StringSubclass(SHA_A)
        mutations.append(mutated)
        for value in (True, IntSubclass(3), -1):
            mutated = deepcopy(payload)
            mutated["trees"][0]["files"][0]["size_bytes"] = value
            mutations.append(mutated)
        for mutated in mutations:
            with self.subTest(mutation=mutated):
                with self.assertRaises(DependencyContentLockError):
                    dependency_content_lock_hash(mutated)

    def test_json_rejects_duplicate_null_float_nonfinite_bom_utf8_and_surrogate(self) -> None:
        documents = (
            b'{"schema_version":"x","schema_version":"y"}',
            b'{"value":null}',
            b'{"value":1.5}',
            b'{"value":NaN}',
            b"\xef\xbb\xbf{}",
            b"\xff",
            b'{"value":"\\ud800"}',
        )
        with TemporaryDirectory() as directory:
            for index, raw in enumerate(documents):
                with self.subTest(index=index):
                    with self.assertRaises(DependencyContentLockError):
                        load_dependency_content_lock(self._write(directory, raw))

    def test_canonical_logical_paths_reject_nonportable_or_ambiguous_forms(self) -> None:
        invalid = (
            "",
            ".",
            "..",
            "/absolute",
            "C:/drive",
            "C:\\drive",
            "\\\\server\\share",
            "a\\b",
            "a//b",
            "a/./b",
            "a/../b",
            "a/",
        )
        for logical_path in invalid:
            payload = synthetic_payload()
            payload["trees"][0]["files"][0]["logical_path"] = logical_path
            with self.subTest(logical_path=logical_path):
                with self.assertRaises(DependencyContentLockError):
                    self._load_payload(payload)

    def test_files_must_be_strictly_sorted_unique_and_collision_free(self) -> None:
        base = synthetic_payload()["trees"][0]["files"][0]
        variants = (
            [deepcopy(base), deepcopy(base)],
            [dict(base, logical_path="z.py"), dict(base, logical_path="a.py")],
            [dict(base, logical_path="A.py"), dict(base, logical_path="a.py")],
            [dict(base, logical_path="Caf\u00e9.py"), dict(base, logical_path="Cafe\u0301.py")],
        )
        for files in variants:
            payload = synthetic_payload()
            payload["trees"][0]["files"] = files
            with self.subTest(paths=[item["logical_path"] for item in files]):
                with self.assertRaises(DependencyContentLockError):
                    self._load_payload(payload)

    def test_tree_order_and_global_file_identity_are_strict(self) -> None:
        reordered = synthetic_payload()
        reordered["trees"][0], reordered["trees"][1] = (
            reordered["trees"][1],
            reordered["trees"][0],
        )
        duplicate_owner = synthetic_payload()
        duplicate_owner["trees"][2]["files"][0]["logical_path"] = "python.exe"
        for payload in (reordered, duplicate_owner):
            with self.assertRaises(DependencyContentLockError):
                self._load_payload(payload)

    def test_hash_is_insertion_order_stable_and_changes_for_one_field(self) -> None:
        payload = synthetic_payload()
        first = self._load_payload(payload)
        reversed_payload = {key: payload[key] for key in reversed(tuple(payload))}
        second = self._load_payload(reversed_payload)
        self.assertEqual(
            dependency_content_lock_hash(first), dependency_content_lock_hash(second)
        )

        changed_payload = deepcopy(payload)
        changed_payload["trees"][0]["files"][0]["size_bytes"] += 1
        changed = self._load_payload(changed_payload)
        self.assertNotEqual(
            dependency_content_lock_hash(first), dependency_content_lock_hash(changed)
        )

    def test_hash_is_stable_across_python_hash_seeds(self) -> None:
        script = """
from binance_spot_strategy.identity.dependency_contents_v1 import load_dependency_content_lock, dependency_content_lock_hash
print(dependency_content_lock_hash(load_dependency_content_lock()))
"""
        repository_root = Path(__file__).resolve().parents[2]
        outputs = []
        for seed in ("1", "2", "3"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=repository_root,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs, [outputs[0]] * len(outputs))

    def test_verified_receipts_cannot_be_constructed_subclassed_or_pickled(self) -> None:
        for verified_type in (VerifiedContentTreeV1, VerifiedDependencyContentV1):
            with self.subTest(verified_type=verified_type.__name__):
                with self.assertRaises(TypeError):
                    verified_type()
                with self.assertRaises(TypeError):
                    type("Forged", (verified_type,), {})
                with self.assertRaises(TypeError):
                    pickle.dumps(object.__new__(verified_type))


    def test_public_identity_exports_content_contract_but_not_generator(self) -> None:
        from binance_spot_strategy import identity

        public_names = (
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
        for name in public_names:
            with self.subTest(name=name):
                self.assertIn(name, identity.__all__)
                self.assertTrue(hasattr(identity, name))
class SyntheticContentEnvironment:
    def __init__(self, directory: str) -> None:
        self.root = Path(directory)
        self.prefix = self.root / "prefix"
        self.stdlib = self.prefix / "Lib"
        self.site_packages = self.stdlib / "site-packages"
        self.scripts = self.prefix / "Scripts"
        self.site_packages.mkdir(parents=True)
        self.scripts.mkdir()
        (self.prefix / "python.exe").write_bytes(b"exe")
        (self.prefix / "python3.dll").write_bytes(b"dll")
        (self.prefix / "LICENSE.txt").write_bytes(b"license")
        (self.stdlib / "os.py").write_bytes(b"stdlib")
        semantic = load_semantic_dependency_lock()
        self.distributions: list[_DistributionInputV1] = []
        self.records: dict[str, Path] = {}
        for index, item in enumerate(semantic["distributions"]):
            package_name = f"locked_package_{index}"
            package_root = self.site_packages / package_name
            package_root.mkdir()
            package_file = package_root / "module.py"
            package_file.write_bytes(f"package-{index}".encode("ascii"))
            dist_info = self.site_packages / (
                item["name"].replace("-", "_") + f"-{item['version']}.dist-info"
            )
            dist_info.mkdir()
            record = dist_info / "RECORD"
            rows = [self.record_row(package_file)]
            if item["name"] == "numpy":
                f2py = self.scripts / "f2py.exe"
                numpy_config = self.scripts / "numpy-config.exe"
                f2py.write_bytes(b"f2py")
                numpy_config.write_bytes(b"numpy-config")
                rows.extend(
                    (
                        self.record_row(f2py, "../../Scripts/f2py.exe"),
                        self.record_row(
                            numpy_config, "../../Scripts/numpy-config.exe"
                        ),
                    )
                )
            record_relative = record.relative_to(self.site_packages).as_posix()
            rows.append((record_relative, "", ""))
            self.write_record(record, rows)
            self.records[item["name"]] = record
            self.distributions.append(
                _DistributionInputV1(
                    name=item["name"],
                    version=item["version"],
                    install_root=self.site_packages,
                    record_path=record,
                )
            )
        self.environment = _ScanEnvironmentV1(
            prefix=self.prefix,
            stdlib_root=self.stdlib,
            site_packages_roots=(self.site_packages,),
            implementation="CPython",
            python_version=semantic["python"]["version"],
            distributions=tuple(self.distributions),
        )

    def record_row(
        self, path: Path, raw_path: str | None = None
    ) -> tuple[str, str, str]:
        data = path.read_bytes()
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        relative = (
            raw_path
            if raw_path is not None
            else path.relative_to(self.site_packages).as_posix()
        )
        return relative, "sha256=" + encoded.decode("ascii"), str(len(data))

    @staticmethod
    def write_record(path: Path, rows: list[tuple[str, ...]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(rows)


class DependencyContentFilesystemTests(unittest.TestCase):
    def test_complete_synthetic_trees_verify_and_assign_numpy_scripts_once(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            lock = _build_dependency_content_lock(fixture.environment)
            receipt = _verify_dependency_contents_in_environment(
                lock, fixture.environment
            )

        self.assertIs(type(receipt), VerifiedDependencyContentV1)
        numpy_tree = next(tree for tree in lock.trees if tree.owner_name == "numpy")
        runtime_tree = lock.trees[0]
        self.assertIn(
            "Scripts/f2py.exe", {item.logical_path for item in numpy_tree.files}
        )
        self.assertIn(
            "Scripts/numpy-config.exe",
            {item.logical_path for item in numpy_tree.files},
        )
        self.assertNotIn(
            "Scripts/f2py.exe", {item.logical_path for item in runtime_tree.files}
        )
        kinds = {
            item.logical_path: item.artifact_kind
            for tree in lock.trees
            for item in tree.files
        }
        self.assertEqual(kinds["python.exe"], "runtime_binary")
        self.assertEqual(kinds["Lib/os.py"], "source")

    def test_same_size_same_mtime_byte_swap_is_detected_without_path_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            lock = _build_dependency_content_lock(fixture.environment)
            target = fixture.stdlib / "os.py"
            original = target.stat()
            target.write_bytes(b"STDLIB")
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
            with self.assertRaises(DependencyContentLockError) as raised:
                _verify_dependency_contents_in_environment(lock, fixture.environment)
            message = str(raised.exception)
            self.assertIn("Lib/os.py", message)
            self.assertNotIn(str(fixture.prefix), message)

    def test_add_delete_and_rename_each_fail_closed(self) -> None:
        operations = (
            lambda f: (f.stdlib / "added.py").write_bytes(b"added"),
            lambda f: (f.prefix / "LICENSE.txt").unlink(),
            lambda f: (f.stdlib / "os.py").rename(f.stdlib / "renamed.py"),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with TemporaryDirectory() as directory:
                    fixture = SyntheticContentEnvironment(directory)
                    lock = _build_dependency_content_lock(fixture.environment)
                    operation(fixture)
                    with self.assertRaises(DependencyContentLockError):
                        _verify_dependency_contents_in_environment(
                            lock, fixture.environment
                        )

    def test_undeclared_distribution_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            lock = _build_dependency_content_lock(fixture.environment)
            (fixture.site_packages / "locked_package_1" / "undeclared.dat").write_bytes(
                b"undeclared"
            )
            with self.assertRaises(DependencyContentLockError) as raised:
                _verify_dependency_contents_in_environment(lock, fixture.environment)
        self.assertIn("undeclared distribution file", str(raised.exception))
        self.assertNotIn(str(fixture.prefix), str(raised.exception))

    def test_hardlink_alias_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            target = fixture.stdlib / "os.py"
            alias = fixture.stdlib / "alias.py"
            os.link(target, alias)
            with self.assertRaises(DependencyContentLockError) as raised:
                _build_dependency_content_lock(fixture.environment)
        self.assertIn("alias", str(raised.exception).lower())
        self.assertNotIn(str(fixture.prefix), str(raised.exception))

    def test_record_path_symlink_and_final_prefix_escape_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            external = fixture.root / "external.py"
            external.write_bytes(b"external")
            record = fixture.records["numpy"]
            rows = [
                fixture.record_row(external, "../../../external.py"),
                (record.relative_to(fixture.site_packages).as_posix(), "", ""),
            ]
            fixture.write_record(record, rows)
            with self.assertRaises(DependencyContentLockError) as raised:
                _build_dependency_content_lock(fixture.environment)
            self.assertNotIn(str(fixture.root), str(raised.exception))

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            target = fixture.site_packages / "locked_package_0" / "module.py"
            link = fixture.site_packages / "locked_package_0" / "linked.py"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            record = fixture.records["numpy"]
            fixture.write_record(
                record,
                [
                    fixture.record_row(link),
                    (record.relative_to(fixture.site_packages).as_posix(), "", ""),
                ],
            )
            with self.assertRaises(DependencyContentLockError):
                _build_dependency_content_lock(fixture.environment)

    def test_record_rows_are_strict_and_cross_bind_current_bytes(self) -> None:
        invalid_rows = (
            ("locked_package_0/module.py", "", "9"),
            ("locked_package_0/module.py", "sha256=AAAA", "9"),
            ("locked_package_0/module.py", "md5=" + "a" * 22, "9"),
            ("locked_package_0/module.py", "sha256=" + "A" * 43, "09"),
            ("locked_package_0/module.py", "sha256=" + "A" * 43, "9"),
            ("locked_package_0/module.py", "sha256=" + "A" * 43),
        )
        for invalid in invalid_rows:
            with self.subTest(invalid=invalid):
                with TemporaryDirectory() as directory:
                    fixture = SyntheticContentEnvironment(directory)
                    record = fixture.records["numpy"]
                    fixture.write_record(record, [invalid])
                    with self.assertRaises(DependencyContentLockError):
                        _build_dependency_content_lock(fixture.environment)

    def test_record_duplicate_collision_and_multiple_owners_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            path = fixture.site_packages / "locked_package_0" / "module.py"
            row = fixture.record_row(path)
            record = fixture.records["numpy"]
            fixture.write_record(record, [row, row])
            with self.assertRaises(DependencyContentLockError):
                _build_dependency_content_lock(fixture.environment)

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            shared = fixture.site_packages / "locked_package_0" / "module.py"
            second_record = fixture.records["pandas"]
            fixture.write_record(
                second_record,
                [
                    fixture.record_row(shared),
                    (
                        second_record.relative_to(fixture.site_packages).as_posix(),
                        "",
                        "",
                    ),
                ],
            )
            with self.assertRaises(DependencyContentLockError) as raised:
                _build_dependency_content_lock(fixture.environment)
        self.assertIn("multiple", str(raised.exception))

    def test_content_lock_mismatch_fails_instead_of_rewriting(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            lock = _build_dependency_content_lock(fixture.environment)
            first_tree = lock.trees[0]
            first_file = first_tree.files[0]
            changed_file = ContentFileV1(
                logical_path=first_file.logical_path,
                artifact_kind=first_file.artifact_kind,
                size_bytes=first_file.size_bytes,
                sha256=("0" if first_file.sha256[0] != "0" else "1")
                + first_file.sha256[1:],
            )
            changed_tree = ContentTreeV1(
                first_tree.owner_kind,
                first_tree.owner_name,
                first_tree.owner_version,
                (changed_file,) + first_tree.files[1:],
            )
            changed_lock = DependencyContentLockV1(
                lock.schema_version,
                lock.numeric_protocol_version,
                lock.semantic_dependency_lock_sha256,
                (changed_tree,) + lock.trees[1:],
            )
            with self.assertRaises(DependencyContentLockError):
                _verify_dependency_contents_in_environment(
                    changed_lock, fixture.environment
                )


    def test_each_controlled_file_is_hashed_only_once_per_scan(self) -> None:
        from binance_spot_strategy.identity import dependency_contents_v1 as module

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            with patch.object(
                module, "_snapshot_file", wraps=module._snapshot_file
            ) as snapshot_file:
                _build_dependency_content_lock(fixture.environment)
        logical_calls = [
            Path(call.args[0]).resolve()
            for call in snapshot_file.call_args_list
        ]
        self.assertEqual(
            len(logical_calls),
            len(set(logical_calls)),
            "each controlled path must be read and hashed once",
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows junction contract")
    def test_junction_alias_is_rejected_without_absolute_path_leakage(self) -> None:
        import _winapi

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            target = fixture.root / "junction-target"
            target.mkdir()
            (target / "payload.py").write_bytes(b"payload")
            junction = fixture.site_packages / "locked_package_0" / "junction"
            _winapi.CreateJunction(str(target), str(junction))
            with self.assertRaises(DependencyContentLockError) as raised:
                _build_dependency_content_lock(fixture.environment)
            message = str(raised.exception)
            self.assertIn("reparse", message)
            self.assertNotIn(str(fixture.root), message)
    def test_generator_uses_exclusive_create_and_never_updates_existing_lock(self) -> None:
        from binance_spot_strategy.identity import dependency_contents_v1 as module

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            output = Path(directory) / "candidate.json"
            with patch.object(
                module, "_default_environment", return_value=fixture.environment
            ):
                generated = generate_dependency_content_lock(output)
                original = output.read_bytes()
                with self.assertRaises(DependencyContentLockError):
                    generate_dependency_content_lock(output)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(
                dependency_content_lock_hash(load_dependency_content_lock(output)),
                dependency_content_lock_hash(generated),
            )

    def test_checked_in_lock_verifies_current_contents_repeatably(self) -> None:
        lock = load_dependency_content_lock()
        first = verify_current_dependency_contents(lock)
        second = verify_current_dependency_contents(lock)
        self.assertIs(type(first), VerifiedDependencyContentV1)
        self.assertIs(type(second), VerifiedDependencyContentV1)
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
    @unittest.skipUnless(sys.platform == "win32", "Windows junction contract")
    def test_record_path_rejects_junction_even_when_target_stays_in_prefix(self) -> None:
        import _winapi

        with TemporaryDirectory() as directory:
            fixture = SyntheticContentEnvironment(directory)
            target = fixture.prefix / "alias-target"
            target.mkdir()
            target_file = target / "module.py"
            target_file.write_bytes(b"alias")
            junction = fixture.site_packages / "junction-owned"
            _winapi.CreateJunction(str(target), str(junction))
            record = fixture.records["numpy"]
            fixture.write_record(
                record,
                [
                    fixture.record_row(target_file, "junction-owned/module.py"),
                    (
                        record.relative_to(fixture.site_packages).as_posix(),
                        "",
                        "",
                    ),
                ],
            )
            with self.assertRaises(DependencyContentLockError) as raised:
                _build_dependency_content_lock(fixture.environment)
        self.assertIn("reparse", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
