from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from decimal import (
    Decimal,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    localcontext,
)
import json
from pathlib import Path
import unittest

import numpy as np


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "numeric_protocol_v1.json"
with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
    VECTORS = json.load(fixture_file)

from binance_spot_strategy.protocols import (
    IndicatorNonfiniteTag,
    MetricNonfiniteTag,
    NumericProtocolError,
    Q18,
    canonicalize_float64,
    floor_positive_to_step,
    format_q18,
    numeric_context,
    parse_canonical_q18,
    parse_finite_decimal,
    quantize_q18,
)


class NumericProtocolTests(unittest.TestCase):
    def test_fixture_declares_the_approved_protocol_versions(self) -> None:
        self.assertEqual(VECTORS["schema_version"], "numeric_protocol_vectors_v1")
        self.assertEqual(VECTORS["numeric_protocol_version"], "numeric_protocol_v1")

    def test_numeric_context_is_fresh_and_has_only_the_approved_traps(self) -> None:
        first = numeric_context()
        second = numeric_context()

        self.assertIsNot(first, second)
        self.assertEqual(first.prec, 50)
        self.assertEqual(first.rounding, ROUND_HALF_EVEN)
        self.assertEqual(first.Emin, -999999)
        self.assertEqual(first.Emax, 999999)
        self.assertEqual(first.capitals, 1)
        self.assertEqual(first.clamp, 0)
        self.assertEqual(
            {signal for signal, enabled in first.traps.items() if enabled},
            {DivisionByZero, FloatOperation, InvalidOperation, Overflow},
        )

    def test_parse_finite_decimal_preserves_valid_exchange_text_exactly(self) -> None:
        self.assertEqual(
            parse_finite_decimal("1.230000000000000001"),
            Decimal("1.230000000000000001"),
        )

    def test_parse_finite_decimal_rejects_non_exchange_grammar(self) -> None:
        for invalid in (
            "1.2300000000000000000",
            " 1.2",
            "+1.2",
            "1e-8",
            "1,2",
            "1.",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(NumericProtocolError):
                    parse_finite_decimal(invalid)

    def test_q18_vectors_use_half_even_and_normalize_negative_zero(self) -> None:
        for case in VECTORS["q18_cases"]:
            with self.subTest(case=case):
                actual = quantize_q18(Decimal(case["input"]))
                self.assertEqual(format_q18(actual), case["expected"])

    def test_q18_is_immutable_and_rejects_noncanonical_decimal_values(self) -> None:
        value = quantize_q18(Decimal("1"))

        with self.assertRaises(FrozenInstanceError):
            value.value = Decimal("2.000000000000000000")
        for invalid in (
            Decimal("1.0"),
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-0.000000000000000000"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(NumericProtocolError):
                    Q18(invalid)

    def test_canonical_q18_parser_requires_canonical_text(self) -> None:
        self.assertEqual(
            format_q18(parse_canonical_q18("1.230000000000000000")),
            "1.230000000000000000",
        )
        for invalid in (
            "1.23",
            "01.230000000000000000",
            "-0.000000000000000000",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(NumericProtocolError):
                    parse_canonical_q18(invalid)

    def test_q18_entry_points_reject_float_values(self) -> None:
        with self.assertRaises(TypeError):
            quantize_q18(0.1)
        with self.assertRaises(TypeError):
            quantize_q18(np.float64(0.1))
        with self.assertRaises(TypeError):
            floor_positive_to_step(1.0, Decimal("0.01"))

    def test_grid_floor_vectors_use_exact_integer_ticks(self) -> None:
        for case in VECTORS["floor_cases"]:
            with self.subTest(case=case):
                actual = floor_positive_to_step(
                    Decimal(case["value"]),
                    Decimal(case["step"]),
                )
                self.assertEqual(format_q18(actual), case["expected"])

    def test_grid_floor_rejects_nonpositive_nonfinite_and_over_scale_operands(self) -> None:
        invalid_pairs = (
            (Decimal("0"), Decimal("0.01")),
            (Decimal("-1"), Decimal("0.01")),
            (Decimal("1"), Decimal("0")),
            (Decimal("1"), Decimal("-0.01")),
            (Decimal("NaN"), Decimal("0.01")),
            (Decimal("1"), Decimal("Infinity")),
            (Decimal("1.0000000000000000000"), Decimal("0.01")),
            (Decimal("1"), Decimal("0.0100000000000000000")),
        )
        for value, step in invalid_pairs:
            with self.subTest(value=value, step=step):
                with self.assertRaises(NumericProtocolError):
                    floor_positive_to_step(value, step)

    def test_quantization_is_independent_of_caller_context_across_threads(self) -> None:
        def quantize_tie(_: int) -> str:
            with localcontext() as caller_context:
                caller_context.prec = 3
                caller_context.rounding = ROUND_DOWN
                return format_q18(
                    quantize_q18(Decimal("1.0000000000000000015"))
                )

        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = tuple(executor.map(quantize_tie, range(8)))

        self.assertEqual(actual, ("1.000000000000000002",) * 8)

    def test_float64_bridge_matches_finite_literal_vectors(self) -> None:
        for case in VECTORS["finite_float_cases"]:
            for value in (case["input"], np.float64(case["input"])):
                with self.subTest(case=case, value_type=type(value)):
                    actual = canonicalize_float64(value)
                    self.assertIsInstance(actual, Q18)
                    self.assertEqual(format_q18(actual), case["expected"])

    def test_float64_bridge_rejects_every_other_numeric_type(self) -> None:
        for invalid in (1, Decimal("0.1"), np.float32(0.1)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    canonicalize_float64(invalid)

    def test_nonfinite_tag_values_and_float64_mapping_are_exact(self) -> None:
        self.assertEqual(
            tuple(tag.value for tag in IndicatorNonfiniteTag),
            (
                "FLOAT64:NAN",
                "FLOAT64:POSITIVE_INFINITY",
                "FLOAT64:NEGATIVE_INFINITY",
            ),
        )
        self.assertEqual(
            tuple(tag.value for tag in MetricNonfiniteTag),
            (
                "NONFINITE:NO_OBSERVATIONS",
                "NONFINITE:ZERO_DENOMINATOR",
                "NONFINITE:INVALID_INPUT",
            ),
        )
        self.assertIs(canonicalize_float64(float("nan")), IndicatorNonfiniteTag.NAN)
        self.assertIs(
            canonicalize_float64(float("inf")),
            IndicatorNonfiniteTag.POSITIVE_INFINITY,
        )
        self.assertIs(
            canonicalize_float64(np.float64("-inf")),
            IndicatorNonfiniteTag.NEGATIVE_INFINITY,
        )

    def test_format_q18_rejects_raw_decimal(self) -> None:
        with self.assertRaises(TypeError):
            format_q18(Decimal("1.000000000000000000"))


if __name__ == "__main__":
    unittest.main()
