"""
Number comparison for answer comparators.

Provides compare_numbers used by NormalizedMatchComparator and other comparators.
"""

from prkit.prkit_evaluation.utils.number_utils import (
    DEFAULT_NUMBER_EPSILON,
    decimal_places,
    round_to_decimal_places,
)


def compare_numbers(
    a: float, b: float, epsilon: float = DEFAULT_NUMBER_EPSILON
) -> bool:
    """
    Compare two floats with epsilon tolerance and decimal-place rounding.

    Uses the same logic as _compare_number in category_match:
    - Epsilon tolerance when abs(diff) is small
    - Decimal-place rounding: round a to b's precision if a has more decimals

    Args:
        a: First number
        b: Second number (ground truth)
        epsilon: Tolerance for floating-point comparison

    Returns:
        True if numbers are considered equal
    """
    gt_decimals = decimal_places(b)
    a_decimals = decimal_places(a)
    if a_decimals > gt_decimals:
        a = round_to_decimal_places(a, gt_decimals)
    return abs(a - b) < epsilon
