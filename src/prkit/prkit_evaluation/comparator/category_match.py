"""
Category-Based Comparator for answer comparison.

This comparator normalizes answers first, then applies category-specific comparison
strategies. For the same category, a dedicated comparison function is used;
for different categories, both answers are normalized as text and compared.

Category-specific comparison details can be customized by subclassing and
overriding _get_category_comparators() or by passing custom comparators.
"""

from typing import Callable, Dict, Optional, Tuple, Union

from sympy import sympify

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.normalization import normalize_answer, normalize_text

from prkit.prkit_evaluation.utils.number_utils import (
    decimal_places,
    round_to_decimal_places,
    DEFAULT_NUMBER_EPSILON,
)

from .base import BaseComparator

# Type alias for category comparison function: (predicted_norm, ground_truth_norm) -> bool
CategoryCompareFn = Callable[[Union[float, str], Union[float, str]], bool]


def _same_comparison_category(
    cat1: AnswerCategory, cat2: AnswerCategory
) -> bool:
    """True if both answers should be compared using the same comparison strategy."""
    return cat1 == cat2


def _compare_number(
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


def _compare_plain_text(
    predicted_norm: Union[str, Answer], 
    ground_truth_norm: Union[str, Answer]
) -> bool:
    """Compare two normalized strings."""
    if isinstance(predicted_norm, Answer):
        predicted_norm = predicted_norm.value
    if isinstance(ground_truth_norm, Answer):
        ground_truth_norm = ground_truth_norm.value
    
    return predicted_norm == ground_truth_norm


def _parse_physical_quantity(s: str) -> Tuple[Optional[float], str]:
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


def _compare_physical_quantity(
    predicted_norm: Union[str, Answer],
    ground_truth_norm: Union[str, Answer],
    epsilon: float = DEFAULT_NUMBER_EPSILON,
) -> bool:
    """
    Compare two normalized physical quantities (number + units).

    When units are the same, compares only the numeric part using _compare_number
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
        pred_num, pred_unit = _parse_physical_quantity(predicted_norm)
        pred_string = predicted_norm

    if isinstance(ground_truth_norm, Answer):
        gt_num = ground_truth_norm.value
        gt_unit = ground_truth_norm.unit
        gt_string = (
            f"{gt_num} {gt_unit}" if gt_unit is not None else str(gt_num)
        )
    else:
        gt_num, gt_unit = _parse_physical_quantity(ground_truth_norm)
        gt_string = ground_truth_norm
    
    # When units are the same, compare only the numeric part
    if pred_unit == gt_unit and pred_num is not None and gt_num is not None:
        return _compare_number(pred_num, gt_num, epsilon)

    # Units differ or parse failed: fall back to plain text
    return _compare_plain_text(pred_string, gt_string)


def _formula_to_sympify(s: str) -> str:
    """
    Convert a formula string to sympify-compatible format.

    Replaces ^ with ** so that expressions like "x^2" parse correctly.
    Normalized output from latex2sympy already uses **, but raw strings may use ^.
    """
    return str(s).strip().replace("^", "**")


def _compare_formula(
    predicted_norm: Union[str, Answer],
    ground_truth_norm: Union[str, Answer]
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
    except Exception:
        return _compare_plain_text(pred_expression, gt_expression)


class CategoryComparator(BaseComparator):
    """
    Comparator that normalizes answers first, then applies category-specific
    comparison strategies.

    Category-specific comparators can be customized via:
    1. Subclassing and overriding _get_category_comparators()
    2. Passing custom_comparators to __init__()

    Categories: number, equation, physical_quantity, formula, text
    """

    # Default comparators per category (placeholders; customize as needed)
    DEFAULT_COMPARATORS: Dict[AnswerCategory, CategoryCompareFn] = {
        AnswerCategory.NUMBER: _compare_number,
        AnswerCategory.EQUATION: _compare_plain_text,
        AnswerCategory.PHYSICAL_QUANTITY: _compare_physical_quantity,
        AnswerCategory.FORMULA: _compare_formula,
        AnswerCategory.TEXT: _compare_plain_text,
        AnswerCategory.OPTION: _compare_plain_text,
    }

    def __init__(
        self,
        custom_comparators: Optional[Dict[AnswerCategory, CategoryCompareFn]] = None,
    ):
        """
        Args:
            custom_comparators: Optional dict mapping category name to comparison
                function. Merged with defaults; custom entries override defaults.
        """
        self._comparators = dict(self.DEFAULT_COMPARATORS)
        if custom_comparators:
            self._comparators.update(custom_comparators)

    def _get_category_comparators(self) -> Dict[AnswerCategory, CategoryCompareFn]:
        """
        Return the category -> comparator mapping.

        Override this in subclasses to customize category-based comparison
        without passing custom_comparators at init.
        """
        return self._comparators

    def _normalize_answer(
        self,
        answer_str: str,
    ) -> Tuple[AnswerCategory, Union[float, str]]:
        """
        Normalize an answer string based on its category.

        Args:
            answer_str: The answer string to normalize

        Returns:
            Tuple of (category, normalized_value)
        """
        return normalize_answer(answer_str)

    def _normalize_as_text(
        self,
        answer_str: str
    ) -> str:
        """
        Normalize an answer string as text (strip whitespace).

        Used when comparing answers from different categories.

        Args:
            answer_str: The answer string to normalize as text

        Returns:
            Normalized text string
        """
        return normalize_text(answer_str)

    def _compare_by_category(
        self,
        category: AnswerCategory,
        predicted_norm: Union[float, str],
        ground_truth_norm: Union[float, str],
    ) -> bool:
        """
        Compare two normalized values using the category-specific strategy.

        Args:
            category: The shared category of both answers
            predicted_norm: Normalized predicted/student answer
            ground_truth_norm: Normalized ground truth answer

        Returns:
            True if answers match according to category rules, False otherwise
        """
        comparators = self._get_category_comparators()
        comparator = comparators.get(category)
        if comparator is None:
            return _compare_plain_text(predicted_norm, ground_truth_norm)
        return comparator(predicted_norm, ground_truth_norm)

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
    ) -> bool:
        """
        Compare two answers after normalization.

        Accepts either raw strings (e.g. from JSON) or Answer objects. For option
        answers (both Answer with answer_category=OPTION), uses case-insensitive
        exact match. Otherwise, normalizes both and applies category-specific
        comparison.

        Args:
            answer1: First answer to compare (string or Answer)
            answer2: Second answer to compare (string or Answer)

        Returns:
            True if answers match after category-based comparison, False otherwise
        """
        if isinstance(answer1, Answer) and isinstance(answer2, Answer):
            cat1 = answer1.answer_category
            cat2 = answer2.answer_category
            if _same_comparison_category(cat1, cat2):
                return self._compare_by_category(cat1, answer1.value, answer2.value)
            else:
                text1 = self._normalize_as_text(answer1.value)
                text2 = self._normalize_as_text(answer2.value)
                return text1 == text2
        else:
            ans1_str = (
                str(answer1.value) if isinstance(answer1, Answer) else str(answer1)
            )
            ans2_str = (
                str(answer2.value) if isinstance(answer2, Answer) else str(answer2)
            )

            cat1, predicted_norm = self._normalize_answer(ans1_str)
            cat2, ground_truth_norm = self._normalize_answer(ans2_str)

            if not _same_comparison_category(cat1, cat2):
                text1 = self._normalize_as_text(ans1_str)
                text2 = self._normalize_as_text(ans2_str)
                return text1 == text2

            return self._compare_by_category(cat1, predicted_norm, ground_truth_norm)

    def accuracy_score(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
    ) -> float:
        """
        Compute accuracy score for category-based comparison.

        Returns 1.0 if answers match, 0.0 otherwise. Accepts either raw strings
        (e.g. from JSON) or Answer objects.

        Args:
            answer1: First answer to compare (string or Answer)
            answer2: Second answer to compare (string or Answer)

        Returns:
            1.0 if answers match after category-based comparison, 0.0 otherwise
        """
        is_match = self.compare(answer1, answer2)
        return 1.0 if is_match else 0.0
