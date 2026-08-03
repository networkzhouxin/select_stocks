from collections import UserDict
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_records_v1.json"
with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
    GOLDEN_RECORDS = json.load(fixture_file)

from binance_spot_strategy.protocols import (
    CanonicalJsonError,
    Q18,
    canonical_hashed_payload_bytes,
    canonical_json_bytes,
    quantize_q18,
)


class UnsupportedObject:
    def __str__(self) -> str:
        return "must-not-be-serialized"


class AlwaysEqualStr(str):
    def __eq__(self, other: object) -> bool:
        return True


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


class Q18Subclass(Q18):
    pass


class SplitViewMapping(Mapping):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(("actual",))

    def __len__(self) -> int:
        return 1

    def get(self, key: str, default: object = None) -> object:
        if key == "schema_version":
            return "split_v1"
        if key == "numeric_protocol_version":
            return "numeric_protocol_v1"
        return default

    def items(self):
        return (("actual", 1),)


class ItemsOnlyVersionMapping(Mapping):
    def __init__(self) -> None:
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise AssertionError("hashed encoding must not call get or __getitem__")

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 3

    def items(self):
        self.items_calls += 1
        if self.items_calls != 1:
            raise AssertionError("items must be observed exactly once")
        return (
            ("schema_version", "snapshot_v1"),
            ("numeric_protocol_version", "numeric_protocol_v1"),
            ("value", 1),
        )


class CanonicalJsonTests(unittest.TestCase):
    def test_fixture_declares_the_approved_protocol_versions(self) -> None:
        self.assertEqual(
            GOLDEN_RECORDS["schema_version"],
            "canonical_records_fixture_v1",
        )
        self.assertEqual(
            GOLDEN_RECORDS["numeric_protocol_version"],
            "numeric_protocol_v1",
        )

    def test_primary_golden_record_matches_exact_utf8_and_sha256(self) -> None:
        payload = {
            "name": "Cafe\u0301",
            "items": ["ETHUSDT", "BTCUSDT"],
            "amount": quantize_q18(Decimal("1.005")),
            "schema_version": "test_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
        }

        actual = canonical_hashed_payload_bytes(payload)

        self.assertEqual(
            actual,
            GOLDEN_RECORDS["primary_expected_utf8"].encode("utf-8"),
        )
        self.assertEqual(
            hashlib.sha256(actual).hexdigest(),
            GOLDEN_RECORDS["primary_expected_sha256"],
        )

    def test_unicode_golden_record_normalizes_keys_and_values_to_nfc(self) -> None:
        payload = {
            "z": ["ETHUSDT", "BTCUSDT"],
            "e\u0301": "Cafe\u0301",
            "a": quantize_q18(Decimal("1")),
            "schema_version": "golden_v1",
            "numeric_protocol_version": "numeric_protocol_v1",
        }

        actual = canonical_hashed_payload_bytes(payload)

        self.assertEqual(
            actual,
            GOLDEN_RECORDS["unicode_expected_utf8"].encode("utf-8"),
        )
        self.assertEqual(
            hashlib.sha256(actual).hexdigest(),
            GOLDEN_RECORDS["unicode_expected_sha256"],
        )

    def test_rejects_each_forbidden_value_type(self) -> None:
        for value in (None, 0.1, Decimal("1.0"), b"bytes", {"set"}):
            with self.subTest(value=value):
                with self.assertRaises(CanonicalJsonError):
                    canonical_json_bytes(
                        {
                            "schema_version": "test_v1",
                            "numeric_protocol_version": "numeric_protocol_v1",
                            "value": value,
                        }
                    )

    def test_rejects_keys_that_collide_after_nfc_normalization(self) -> None:
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({"é": 1, "e\u0301": 2})

    def test_rejects_non_string_mapping_keys(self) -> None:
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({1: "value"})

    def test_rejects_unsupported_object_instead_of_stringifying_it(self) -> None:
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({"value": UnsupportedObject()})

    def test_preserves_array_and_tuple_order(self) -> None:
        forward = canonical_json_bytes(
            {"items": ["ETHUSDT", "BTCUSDT"], "tuple": (2, 1)}
        )
        reversed_order = canonical_json_bytes(
            {"items": ["BTCUSDT", "ETHUSDT"], "tuple": (1, 2)}
        )

        self.assertNotEqual(forward, reversed_order)
        self.assertIn(b'"items":["ETHUSDT","BTCUSDT"]', forward)
        self.assertIn(b'"tuple":[2,1]', forward)

    def test_accepts_string_key_mapping_implementations(self) -> None:
        actual = canonical_json_bytes(UserDict({"b": 2, "a": True}))

        self.assertEqual(actual, b'{"a":true,"b":2}')

    def test_hashed_payload_requires_approved_top_level_versions(self) -> None:
        invalid_payloads = (
            {"numeric_protocol_version": "numeric_protocol_v1"},
            {
                "schema_version": "",
                "numeric_protocol_version": "numeric_protocol_v1",
            },
            {
                "schema_version": "test_v1",
                "numeric_protocol_version": "numeric_protocol_v2",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(CanonicalJsonError):
                    canonical_hashed_payload_bytes(payload)

    def test_hashed_payload_validates_and_encodes_one_normalized_snapshot(self) -> None:
        with self.assertRaises(CanonicalJsonError):
            canonical_hashed_payload_bytes(SplitViewMapping())

        payload = ItemsOnlyVersionMapping()
        self.assertEqual(
            canonical_hashed_payload_bytes(payload),
            b'{"numeric_protocol_version":"numeric_protocol_v1","schema_version":"snapshot_v1","value":1}',
        )
        self.assertEqual(payload.items_calls, 1)

    def test_hashed_versions_require_exact_builtin_strings(self) -> None:
        payload = {
            "schema_version": "test_v1",
            "numeric_protocol_version": AlwaysEqualStr("wrong_numeric"),
        }
        with self.assertRaises(CanonicalJsonError):
            canonical_hashed_payload_bytes(payload)

    def test_hashed_versions_validate_raw_types_before_normalization(self) -> None:
        q18_version = quantize_q18(Decimal("1"))
        for field in ("schema_version", "numeric_protocol_version"):
            payload = {
                "schema_version": "test_v1",
                "numeric_protocol_version": "numeric_protocol_v1",
            }
            payload[field] = q18_version
            with self.subTest(field=field):
                with self.assertRaises(CanonicalJsonError):
                    canonical_hashed_payload_bytes(payload)

    def test_builtin_value_and_key_subclasses_never_reach_json_encoder(self) -> None:
        forbidden_values = (
            StringSubclass("value"),
            IntSubclass(1),
            ListSubclass([1]),
            TupleSubclass((1,)),
            Q18Subclass(Decimal("1.000000000000000000")),
        )
        for value in forbidden_values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(CanonicalJsonError):
                    canonical_json_bytes({"value": value})
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({StringSubclass("key"): "value"})

    def test_lone_surrogates_are_reported_as_canonical_json_errors(self) -> None:
        for payload in (
            {"value": "\ud800"},
            {"\ud800": "value"},
        ):
            with self.subTest(position=tuple(payload)):
                with self.assertRaises(CanonicalJsonError):
                    canonical_json_bytes(payload)

    def test_recursive_payload_is_reported_as_canonical_json_error(self) -> None:
        recursive = []
        recursive.append(recursive)
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({"value": recursive})

    def test_encoder_integer_limit_is_reported_as_canonical_json_error(self) -> None:
        huge_integer = 10**5000
        with self.assertRaises(CanonicalJsonError):
            canonical_json_bytes({"value": huge_integer})

    def test_hashed_payload_preserves_hash_fields(self) -> None:
        actual = canonical_hashed_payload_bytes(
            {
                "schema_version": "test_v1",
                "numeric_protocol_version": "numeric_protocol_v1",
                "payload_sha256": "already-present",
            }
        )

        self.assertIn(b'"payload_sha256":"already-present"', actual)

    def test_emitted_bytes_have_no_bom_whitespace_or_newline(self) -> None:
        actual = canonical_json_bytes({"b": False, "a": 1})

        self.assertEqual(actual, b'{"a":1,"b":false}')
        self.assertFalse(actual.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", actual)
        self.assertNotIn(b"\r", actual)


if __name__ == "__main__":
    unittest.main()
