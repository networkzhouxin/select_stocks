"""Deterministic Decimal protocol for Binance Spot M1 records."""

from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)
import re


Q18_QUANTUM = Decimal("0.000000000000000001")
EXCHANGE_DECIMAL_RE = re.compile(r"-?\d+(?:\.\d{1,18})?\Z")
CANONICAL_Q18_RE = re.compile(r"-?(?:0|[1-9]\d*)\.\d{18}\Z")


class NumericProtocolError(ValueError):
    """Raised when a value violates numeric protocol v1."""


def numeric_context() -> Context:
    """Return a fresh Decimal context isolated from caller state."""

    context = Context(
        prec=50,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        traps=[],
    )
    for signal in context.traps:
        context.traps[signal] = False
    for signal in (DivisionByZero, FloatOperation, InvalidOperation, Overflow):
        context.traps[signal] = True
    context.clear_flags()
    return context


@dataclass(frozen=True, slots=True)
class Q18:
    """A finite Decimal encoded with exactly 18 fractional digits."""

    value: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            raise TypeError("Q18 value must be Decimal")
        if not self.value.is_finite():
            raise NumericProtocolError("Q18 value must be finite")
        if self.value.as_tuple().exponent != -18:
            raise NumericProtocolError("Q18 value must have exponent -18")
        if self.value.is_zero() and self.value.is_signed():
            raise NumericProtocolError("Q18 value must not be negative zero")


def parse_finite_decimal(text: str) -> Decimal:
    """Parse exchange decimal text without rounding or scale relaxation."""

    if not isinstance(text, str):
        raise TypeError("exchange decimal input must be str")
    if EXCHANGE_DECIMAL_RE.fullmatch(text) is None:
        raise NumericProtocolError("invalid exchange decimal text")
    return Decimal(text)


def quantize_q18(value: Decimal) -> Q18:
    """Round a finite Decimal to Q18 with protocol-local HALF_EVEN rules."""

    if not isinstance(value, Decimal):
        raise TypeError("Q18 quantization input must be Decimal")
    try:
        with localcontext(numeric_context()):
            quantized = value.quantize(Q18_QUANTUM)
    except DecimalException as exc:
        raise NumericProtocolError("Decimal cannot be quantized to Q18") from exc
    if not quantized.is_finite():
        raise NumericProtocolError("Q18 value must be finite")
    if quantized.is_zero() and quantized.is_signed():
        quantized = quantized.copy_abs()
    return Q18(quantized)


def parse_canonical_q18(text: str) -> Q18:
    """Parse an already-canonical Q18 string without rounding it."""

    if not isinstance(text, str):
        raise TypeError("canonical Q18 input must be str")
    if CANONICAL_Q18_RE.fullmatch(text) is None:
        raise NumericProtocolError("invalid canonical Q18 text")
    value = Decimal(text)
    if value.is_zero() and value.is_signed():
        raise NumericProtocolError("canonical Q18 text must not be negative zero")
    return Q18(value)


def format_q18(value: Q18) -> str:
    """Format only a validated Q18 value as fixed-point text."""

    if not isinstance(value, Q18):
        raise TypeError("Q18 formatter input must be Q18")
    return format(value.value, "f")


def _decimal_coefficient_and_exponent(value: Decimal) -> tuple[int, int]:
    sign, digits, exponent = value.as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    return coefficient, int(exponent)


def _digits_from_nonnegative_integer(value: int) -> tuple[int, ...]:
    if value == 0:
        return (0,)
    digits = []
    while value:
        value, digit = divmod(value, 10)
        digits.append(digit)
    return tuple(reversed(digits))


def _require_positive_grid_operand(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal")
    if not value.is_finite():
        raise NumericProtocolError(f"{name} must be finite")
    if value <= 0:
        raise NumericProtocolError(f"{name} must be positive")
    if value.as_tuple().exponent < -18:
        raise NumericProtocolError(f"{name} scale must not exceed 18")
    return value


def floor_positive_to_step(value: Decimal, step: Decimal) -> Q18:
    """Floor a positive Decimal to a positive step using integer ticks."""

    checked_value = _require_positive_grid_operand(value, "value")
    checked_step = _require_positive_grid_operand(step, "step")
    if checked_value < checked_step:
        return quantize_q18(Decimal(0))
    maximum_integer_digits = (
        numeric_context().prec + Q18_QUANTUM.as_tuple().exponent
    )
    if checked_value.adjusted() >= maximum_integer_digits:
        raise NumericProtocolError("value cannot be represented as Q18")
    value_coefficient, value_exponent = _decimal_coefficient_and_exponent(
        checked_value
    )
    step_coefficient, step_exponent = _decimal_coefficient_and_exponent(
        checked_step
    )
    common_exponent = min(value_exponent, step_exponent)
    aligned_value = value_coefficient * 10 ** (value_exponent - common_exponent)
    aligned_step = step_coefficient * 10 ** (step_exponent - common_exponent)
    floored_coefficient = (aligned_value // aligned_step) * aligned_step
    exact_floor = Decimal(
        (
            0,
            _digits_from_nonnegative_integer(floored_coefficient),
            common_exponent,
        )
    )
    return quantize_q18(exact_floor)


__all__ = (
    "NumericProtocolError",
    "Q18",
    "floor_positive_to_step",
    "format_q18",
    "numeric_context",
    "parse_canonical_q18",
    "parse_finite_decimal",
    "quantize_q18",
)
