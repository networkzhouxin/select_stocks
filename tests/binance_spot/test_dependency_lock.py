from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from binance_spot_strategy.identity.dependency_lock_v1 import (
    DependencyLockError,
    load_semantic_dependency_lock,
    semantic_dependency_lock_hash,
    verify_current_runtime,
)
from binance_spot_strategy.identity.digests import (
    hash_canonical_payload,
    require_sha256,
    sha256_bytes,
    sha256_raw_file,
    sha256_tracked_text,
)


EXPECTED_LOCK_HASH = (
    "7d7c3124979b3a217b401e1e975fab8d97f837b4e7c9d732cfbeb1ea1cc74ca8"
)


class AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True


class DigestStringSubclass(str):
    pass


class DigestTests(unittest.TestCase):
    def test_sha256_bytes_and_raw_file_hash_exact_bytes(self) -> None:
        expected = (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        self.assertEqual(sha256_bytes(b"abc"), expected)
        with TemporaryDirectory() as directory:
            path = Path(directory, "raw.bin")
            path.write_bytes(b"abc")
            self.assertEqual(sha256_raw_file(path), expected)

    def test_require_sha256_rejects_noncanonical_digests(self) -> None:
        canonical = "a" * 64
        self.assertEqual(require_sha256(canonical), canonical)
        for invalid in (
            canonical.upper(),
            "a" * 63,
            "g" * 64,
            123,
            DigestStringSubclass(canonical),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    require_sha256(invalid)

    def test_tracked_text_normalizes_newlines_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            cr = root / "cr.txt"
            lf.write_bytes("alpha\nβeta\n".encode("utf-8"))
            crlf.write_bytes("alpha\r\nβeta\r\n".encode("utf-8"))
            cr.write_bytes("alpha\rβeta\r".encode("utf-8"))
            expected = sha256_bytes("alpha\nβeta\n".encode("utf-8"))
            self.assertEqual(sha256_tracked_text(lf), expected)
            self.assertEqual(sha256_tracked_text(crlf), expected)
            self.assertEqual(sha256_tracked_text(cr), expected)

    def test_tracked_text_rejects_bom_and_invalid_utf8(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "bom.txt"
            invalid = root / "invalid.txt"
            bom.write_bytes(b"\xef\xbb\xbfcontent")
            invalid.write_bytes(b"\xff")
            for path in (bom, invalid):
                with self.subTest(path=path.name):
                    with self.assertRaises(ValueError):
                        sha256_tracked_text(path)

    def test_hash_canonical_payload_uses_canonical_protocol(self) -> None:
        first = {
            "schema_version": "example_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
            "b": 2,
            "a": 1,
        }
        second = {
            "a": 1,
            "b": 2,
            "numeric_protocol_version": "numeric_protocol_v1",
            "schema_version": "example_v1",
        }
        self.assertEqual(hash_canonical_payload(first), hash_canonical_payload(second))


class DependencyLockTests(unittest.TestCase):
    def _write_lock(self, directory: str, raw: bytes) -> Path:
        path = Path(directory, "semantic_dependencies.lock.json")
        path.write_bytes(raw)
        return path

    def test_checked_in_lock_has_frozen_semantic_hash(self) -> None:
        lock = load_semantic_dependency_lock()
        self.assertEqual(semantic_dependency_lock_hash(lock), EXPECTED_LOCK_HASH)

    def test_current_runtime_matches_checked_in_lock(self) -> None:
        verify_current_runtime(load_semantic_dependency_lock())

    def test_version_change_is_schema_valid_but_reports_stable_drift_path(self) -> None:
        lock = deepcopy(load_semantic_dependency_lock())
        lock["python"]["version"] = "3.13.5"
        with self.assertRaisesRegex(DependencyLockError, r"python\.version"):
            verify_current_runtime(lock)

    def test_all_drift_paths_are_sorted_into_one_error(self) -> None:
        lock = deepcopy(load_semantic_dependency_lock())
        lock["python"]["version"] = "3.13.5"
        lock["unicode"]["database_version"] = "0.0.0"
        with self.assertRaises(DependencyLockError) as raised:
            verify_current_runtime(lock)
        self.assertEqual(
            str(raised.exception),
            "semantic dependency drift: python.version, unicode.database_version",
        )

    def test_schema_rejects_unknown_keys_at_every_nested_level(self) -> None:
        original = load_semantic_dependency_lock()
        mutations = (
            (original, "unknown"),
            (original["source_hash_convention"], "unknown"),
            (original["python"], "unknown"),
            (original["float64"], "unknown"),
            (original["decimal"], "unknown"),
            (original["decimal"]["context"], "unknown"),
            (original["unicode"], "unknown"),
            (original["distributions"][0], "unknown"),
        )
        for target_template, key in mutations:
            lock = deepcopy(original)
            if target_template is original:
                target = lock
            elif target_template is original["source_hash_convention"]:
                target = lock["source_hash_convention"]
            elif target_template is original["python"]:
                target = lock["python"]
            elif target_template is original["float64"]:
                target = lock["float64"]
            elif target_template is original["decimal"]:
                target = lock["decimal"]
            elif target_template is original["decimal"]["context"]:
                target = lock["decimal"]["context"]
            elif target_template is original["unicode"]:
                target = lock["unicode"]
            else:
                target = lock["distributions"][0]
            target[key] = "rejected"
            with self.subTest(target=tuple(target.keys())):
                with self.assertRaises(DependencyLockError):
                    semantic_dependency_lock_hash(lock)

    def test_json_loader_rejects_duplicate_null_float_and_nonfinite(self) -> None:
        invalid_documents = (
            b'{"schema_version":"x","schema_version":"y"}',
            b'{"value":null}',
            b'{"value":1.5}',
            b'{"value":NaN}',
            b'{"value":Infinity}',
        )
        with TemporaryDirectory() as directory:
            for index, raw in enumerate(invalid_documents):
                path = self._write_lock(directory, raw)
                with self.subTest(index=index):
                    with self.assertRaises(DependencyLockError):
                        load_semantic_dependency_lock(path)

    def test_schema_rejects_bool_for_int_tuple_for_array_and_bad_digest(self) -> None:
        original = load_semantic_dependency_lock()
        mutations = []
        bool_as_int = deepcopy(original)
        bool_as_int["float64"]["radix"] = True
        mutations.append(bool_as_int)
        tuple_array = deepcopy(original)
        tuple_array["python"]["version_info"] = tuple(
            tuple_array["python"]["version_info"]
        )
        mutations.append(tuple_array)
        bad_digest = deepcopy(original)
        bad_digest["design_revision_sha256"] = "A" * 64
        mutations.append(bad_digest)
        for lock in mutations:
            with self.subTest(lock=lock):
                with self.assertRaises(DependencyLockError):
                    semantic_dependency_lock_hash(lock)

    def test_schema_rejects_reordered_or_unexpected_distributions(self) -> None:
        original = load_semantic_dependency_lock()
        reversed_lock = deepcopy(original)
        reversed_lock["distributions"].reverse()
        missing_lock = deepcopy(original)
        missing_lock["distributions"].pop()
        for lock in (reversed_lock, missing_lock):
            with self.assertRaises(DependencyLockError):
                semantic_dependency_lock_hash(lock)

    def test_hash_and_verify_share_strict_validation(self) -> None:
        lock = deepcopy(load_semantic_dependency_lock())
        lock["unexpected"] = "field"
        for operation in (semantic_dependency_lock_hash, verify_current_runtime):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(DependencyLockError):
                    operation(lock)

    def test_constant_and_digest_boundaries_reject_equality_bypasses(self) -> None:
        original = load_semantic_dependency_lock()
        mutations = []

        for field in ("schema_version", "numeric_protocol_version"):
            non_string = deepcopy(original)
            non_string[field] = AlwaysEqual()
            mutations.append(non_string)
            string_subclass = deepcopy(original)
            string_subclass[field] = DigestStringSubclass(string_subclass[field])
            mutations.append(string_subclass)

        for field in original["source_hash_convention"]:
            lock = deepcopy(original)
            lock["source_hash_convention"][field] = DigestStringSubclass(
                lock["source_hash_convention"][field]
            )
            mutations.append(lock)

        normalization = deepcopy(original)
        normalization["unicode"]["normalization"] = DigestStringSubclass("NFC")
        mutations.append(normalization)

        digest_paths = (
            ("design_revision_sha256",),
            ("requirements_lock_sha256",),
            ("python", "executable_sha256"),
            ("decimal", "extension_sha256"),
            ("distributions", 0, "record_sha256"),
        )
        for path in digest_paths:
            lock = deepcopy(original)
            target = lock
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = DigestStringSubclass(target[path[-1]])
            mutations.append(lock)

        for index, lock in enumerate(mutations):
            for operation in (semantic_dependency_lock_hash, verify_current_runtime):
                with self.subTest(index=index, operation=operation.__name__):
                    with self.assertRaises(DependencyLockError):
                        operation(lock)

    def test_missing_distribution_record_fails_closed_at_record_path(self) -> None:
        import importlib.metadata

        real_distribution = importlib.metadata.distribution
        real_numpy = real_distribution("numpy")

        class MissingRecordDistribution:
            version = real_numpy.version
            files = [Path("numpy-2.2.6.dist-info/METADATA")]

            @staticmethod
            def locate_file(path: object) -> Path:
                return Path(path)

        def distribution(name: str):
            if name == "numpy":
                return MissingRecordDistribution()
            return real_distribution(name)

        with patch(
            "binance_spot_strategy.identity.dependency_lock_v1.metadata.distribution",
            side_effect=distribution,
        ):
            with self.assertRaises(DependencyLockError) as raised:
                verify_current_runtime(load_semantic_dependency_lock())
        self.assertIn("distributions.numpy.record_sha256", str(raised.exception))

    def test_duplicate_distribution_record_fails_closed_at_record_path(self) -> None:
        import importlib.metadata

        real_distribution = importlib.metadata.distribution
        real_numpy = real_distribution("numpy")

        class DuplicateRecordDistribution:
            version = real_numpy.version
            files = [
                Path("numpy-2.2.6.dist-info/RECORD"),
                Path("copy.dist-info/RECORD"),
            ]

            @staticmethod
            def locate_file(path: object) -> Path:
                return Path(path)

        def distribution(name: str):
            if name == "numpy":
                return DuplicateRecordDistribution()
            return real_distribution(name)

        with patch(
            "binance_spot_strategy.identity.dependency_lock_v1.metadata.distribution",
            side_effect=distribution,
        ):
            with self.assertRaises(DependencyLockError) as raised:
                verify_current_runtime(load_semantic_dependency_lock())
        self.assertIn("distributions.numpy.record_sha256", str(raised.exception))

    def test_distribution_version_property_error_is_collected_without_os_text(self) -> None:
        import importlib.metadata

        real_distribution = importlib.metadata.distribution
        real_numpy = real_distribution("numpy")

        class BrokenVersionDistribution:
            files = real_numpy.files

            @property
            def version(self) -> str:
                raise OSError("VOLATILE C:\\private\\python")

            @staticmethod
            def locate_file(path: object) -> Path:
                return real_numpy.locate_file(path)

        def distribution(name: str):
            if name == "numpy":
                return BrokenVersionDistribution()
            return real_distribution(name)

        lock = deepcopy(load_semantic_dependency_lock())
        lock["unicode"]["database_version"] = "0.0.0"
        with patch(
            "binance_spot_strategy.identity.dependency_lock_v1.metadata.distribution",
            side_effect=distribution,
        ):
            with self.assertRaises(DependencyLockError) as raised:
                verify_current_runtime(lock)
        self.assertEqual(
            str(raised.exception),
            "semantic dependency drift: distributions.numpy.version, "
            "unicode.database_version",
        )
        self.assertNotIn("VOLATILE", str(raised.exception))
        self.assertNotIn("private", str(raised.exception))

    def test_distribution_files_property_error_maps_to_record_path(self) -> None:
        import importlib.metadata

        real_distribution = importlib.metadata.distribution
        real_numpy = real_distribution("numpy")

        class BrokenFilesDistribution:
            version = real_numpy.version

            @property
            def files(self) -> list[Path]:
                raise OSError("VOLATILE files")

        def distribution(name: str):
            if name == "numpy":
                return BrokenFilesDistribution()
            return real_distribution(name)

        with patch(
            "binance_spot_strategy.identity.dependency_lock_v1.metadata.distribution",
            side_effect=distribution,
        ):
            with self.assertRaises(DependencyLockError) as raised:
                verify_current_runtime(load_semantic_dependency_lock())
        self.assertEqual(
            str(raised.exception),
            "semantic dependency drift: distributions.numpy.record_sha256",
        )
        self.assertNotIn("VOLATILE", str(raised.exception))

    def test_distribution_locate_file_error_maps_to_record_path(self) -> None:
        import importlib.metadata

        real_distribution = importlib.metadata.distribution
        real_numpy = real_distribution("numpy")

        class BrokenLocateDistribution:
            version = real_numpy.version
            files = real_numpy.files

            @staticmethod
            def locate_file(path: object) -> Path:
                raise OSError("VOLATILE locate")

        def distribution(name: str):
            if name == "numpy":
                return BrokenLocateDistribution()
            return real_distribution(name)

        with patch(
            "binance_spot_strategy.identity.dependency_lock_v1.metadata.distribution",
            side_effect=distribution,
        ):
            with self.assertRaises(DependencyLockError) as raised:
                verify_current_runtime(load_semantic_dependency_lock())
        self.assertEqual(
            str(raised.exception),
            "semantic dependency drift: distributions.numpy.record_sha256",
        )
        self.assertNotIn("VOLATILE", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
