"""
Number utilities for numeric precision handling.

This module provides functions for decimal places, significant digits, and rounding,
used in number comparison.
"""

import math

# Default epsilon for number comparison
DEFAULT_NUMBER_EPSILON = 1e-10


def decimal_places(x: float) -> int:
    """
    Infer the number of decimal places from a float's canonical representation.

    Unlike significant digits, this can be derived from a float by formatting
    and stripping trailing zeros.

    Args:
        x: Float value

    Returns:
        Number of digits after the decimal point (0 for integers)

    Examples:
        9.8 -> 1, 9.87 -> 2, 500.0 -> 0, 0.00123 -> 5
    """
    if x == 0 or math.isnan(x) or math.isinf(x):
        return 0
    s = format(x, ".15g")
    if "e" in s.lower():
        # Scientific notation: use f format with limited precision
        s = format(x, ".15f").rstrip("0").rstrip(".")
    else:
        s = s.rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def round_to_decimal_places(x: float, n: int) -> float:
    """Round a float to n decimal places."""
    if n < 0:
        return x
    return round(x, n)
