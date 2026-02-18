"""
Same-answer-type comparison functions for physical reasoning evaluation.

This module consolidates category-specific comparison logic used when comparing
two values of the same answer type (number vs number, formula vs formula, etc.).
These functions are used by CategoryComparator and can be reused by other
comparison strategies.
"""

from typing import Callable, Optional, Tuple, Union

from sympy import sympify
from sympy.core.sympify import SympifyError

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_evaluation.utils.number_utils import (
    decimal_places,
    round_to_decimal_places,
    DEFAULT_NUMBER_EPSILON,
)


# Type alias for category comparison function: (predicted_norm, ground_truth_norm) -> bool
CategoryCompareFn = Callable[[Union[float, str], Union[float, str]], bool]


def compare_number(
    predicted_norm: Union[float, Answer],
    ground_truth_norm: Union[float, Answer],
    epsilon: float = DEFAULT_NUMBER_EPSILON,
) -> bool:
    """
    Compare two normalized numbers.

    Uses:
    1. Epsilon tolerance when abs(diff) is small.
    2. Decimal-place rounding: if predicted has more decimal places than ground
       truth, round predicted to ground truth's precision before comparing.

    Args:
        predicted_norm: Normalized predicted value (float)
        ground_truth_norm: Normalized ground truth value (float)
        epsilon: Tolerance for floating-point comparison

    Returns:
        True if numbers are considered equal
    """
    if isinstance(predicted_norm, Answer):
        predicted_norm = predicted_norm.value
    if isinstance(ground_truth_norm, Answer):
        ground_truth_norm = ground_truth_norm.value

    pred, gt = predicted_norm, ground_truth_norm

    # Round predicted to ground truth's decimal places if predicted has more
    gt_decimals = decimal_places(ground_truth_norm)
    pred_decimals = decimal_places(predicted_norm)
    if pred_decimals > gt_decimals:
        pred = round_to_decimal_places(pred, gt_decimals)

    return abs(pred - gt) < epsilon


def compare_plain_text(
    predicted_norm: Union[str, Answer],
    ground_truth_norm: Union[str, Answer],
) -> bool:
    """Compare two normalized strings."""
    if isinstance(predicted_norm, Answer):
        predicted_norm = predicted_norm.value
    if isinstance(ground_truth_norm, Answer):
        ground_truth_norm = ground_truth_norm.value

    return predicted_norm == ground_truth_norm


def parse_physical_quantity(s: str) -> Tuple[Optional[float], str]:
    """
    Parse a normalized physical quantity string into (numeric_value, unit).

    Format is typically "number unit" (e.g. "-10000 A/s"). Returns (None, full_str)
    if the numeric part cannot be parsed.

    Args:
        s: Normalized physical quantity string

    Returns:
        Tuple of (parsed_float or None, unit_string)
    """
    s = str(s).strip()
    parts = s.split(None, 1)  # split on first whitespace
    num_str = parts[0]
    unit = parts[1].strip() if len(parts) > 1 else ""
    try:
        # Use same logic as normalize_number for fractions (e.g. "500/11")
        if "/" in num_str and num_str.count("/") == 1:
            n, d = num_str.split("/", 1)
            num_val = float(n.strip()) / float(d.strip())
        else:
            num_val = float(num_str.replace(",", ""))
        return (num_val, unit)
    except (ValueError, ZeroDivisionError):
        return (None, s)


def compare_physical_quantity(
    predicted_norm: Union[str, Answer],
    ground_truth_norm: Union[str, Answer],
    epsilon: float = DEFAULT_NUMBER_EPSILON,
) -> bool:
    """
    Compare two normalized physical quantities (number + units).

    When units are the same, compares only the numeric part using compare_number
    (epsilon tolerance and decimal-place rounding). When units differ, falls
    back to plain text comparison (unit conversion not yet implemented).

    Args:
        predicted_norm: Normalized predicted answer (from _normalize_physical_quantity)
        ground_truth_norm: Normalized ground truth (from _normalize_physical_quantity)
        Both are typically in the form "number unit" (e.g. "-10000 A/s").
        epsilon: Tolerance for numeric comparison when units match
    """
    if isinstance(predicted_norm, Answer):
        pred_num = predicted_norm.value
        pred_unit = predicted_norm.unit
        pred_string = (
            f"{pred_num} {pred_unit}" if pred_unit is not None else str(pred_num)
        )
    else:
        pred_num, pred_unit = parse_physical_quantity(predicted_norm)
        pred_string = predicted_norm

    if isinstance(ground_truth_norm, Answer):
        gt_num = ground_truth_norm.value
        gt_unit = ground_truth_norm.unit
        gt_string = (
            f"{gt_num} {gt_unit}" if gt_unit is not None else str(gt_num)
        )
    else:
        gt_num, gt_unit = parse_physical_quantity(ground_truth_norm)
        gt_string = ground_truth_norm

    # When units are the same, compare only the numeric part
    if pred_unit == gt_unit and pred_num is not None and gt_num is not None:
        return compare_number(pred_num, gt_num, epsilon)

    # Units differ or parse failed: fall back to plain text
    return compare_plain_text(pred_string, gt_string)


def _formula_to_sympify(s: str) -> str:
    """
    Convert a formula string to sympify-compatible format.

    Replaces ^ with ** so that expressions like "x^2" parse correctly.
    Normalized output from latex2sympy already uses **, but raw strings may use ^.
    """
    return str(s).strip().replace("^", "**")


def compare_formula(
    predicted_norm: Union[str, Answer],
    ground_truth_norm: Union[str, Answer],
) -> bool:
    """
    Compare two normalized formula expressions using SymPy.

    Uses sympy.sympify to parse expressions and sympy's .equals() to check
    mathematical equivalence. Structurally equivalent but superficially
    different forms (e.g. x+y vs y+x, (x+1)**2 vs x**2+2*x+1) are treated
    as equal. Falls back to plain text comparison if parsing fails.

    Args:
        predicted_norm: Normalized predicted answer (SymPy string from normalize_expression)
        ground_truth_norm: Normalized ground truth (SymPy string from normalize_expression)

    Returns:
        True if expressions are mathematically equivalent, False otherwise.
    """
    if isinstance(predicted_norm, Answer):
        pred_expression = predicted_norm.value
    else:
        pred_expression = predicted_norm
    if isinstance(ground_truth_norm, Answer):
        gt_expression = ground_truth_norm.value
    else:
        gt_expression = ground_truth_norm

    try:
        pred_sym = sympify(_formula_to_sympify(pred_expression))
        gt_sym = sympify(_formula_to_sympify(gt_expression))
        return pred_sym.equals(gt_sym)
    except (SympifyError, ValueError, TypeError, AttributeError):
        return compare_plain_text(pred_expression, gt_expression)
