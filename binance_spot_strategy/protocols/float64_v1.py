"""Explicit bridge from binary float64 values into numeric protocol v1."""

from decimal import Decimal
from enum import StrEnum
import math

import numpy as np

from .numeric_v1 import Q18, quantize_q18


class IndicatorNonfiniteTag(StrEnum):
    NAN = "FLOAT64:NAN"
    POSITIVE_INFINITY = "FLOAT64:POSITIVE_INFINITY"
    NEGATIVE_INFINITY = "FLOAT64:NEGATIVE_INFINITY"


class MetricNonfiniteTag(StrEnum):
    NO_OBSERVATIONS = "NONFINITE:NO_OBSERVATIONS"
    ZERO_DENOMINATOR = "NONFINITE:ZERO_DENOMINATOR"
    INVALID_INPUT = "NONFINITE:INVALID_INPUT"


def canonicalize_float64(value: float | np.float64) -> Q18 | IndicatorNonfiniteTag:
    """Map exactly Python float or NumPy float64 to Q18 or a stable tag."""

    if type(value) not in (float, np.float64):
        raise TypeError("float64 bridge accepts only float or numpy.float64")
    float_value = float(value)
    if math.isnan(float_value):
        return IndicatorNonfiniteTag.NAN
    if math.isinf(float_value):
        if math.copysign(1.0, float_value) > 0:
            return IndicatorNonfiniteTag.POSITIVE_INFINITY
        return IndicatorNonfiniteTag.NEGATIVE_INFINITY
    return quantize_q18(Decimal(repr(float_value)))


__all__ = (
    "IndicatorNonfiniteTag",
    "MetricNonfiniteTag",
    "canonicalize_float64",
)
