"""Numeric precision helpers: decimal-place inference, significant digits, and epsilon-aware rounding."""

import math

# Default epsilon for number comparison
DEFAULT_NUMBER_EPSILON = 1e-10


def decimal_places(x: int | float | str) -> int:
    """
    Infer the number of decimal places.

    Accepts a float **or** a string.  When a string is provided, trailing
    zeros are preserved (e.g. ``"0.50"`` → 2).  When a float is provided,
    trailing zeros are indistinguishable so we strip them (``0.5`` → 1).

    Args:
        x: Float or string representation of a number

    Returns:
        Number of digits after the decimal point (0 for integers)

    Examples:
        ``9.8`` → 1, ``"0.50"`` → 2, ``500.0`` → 0, ``0.00123`` → 5
    """
    if isinstance(x, str):
        s = x.strip()
        if "." in s:
            return len(s.split(".")[1])
        return 0

    if x == 0 or math.isnan(x) or math.isinf(x):
        return 0
    s = format(x, ".15g")
    if "e" in s.lower():
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
