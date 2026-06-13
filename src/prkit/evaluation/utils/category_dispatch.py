"""
Dispatch normalized prediction vs ground truth to the right same-type compare function.

This is a small helper used by :class:`~prkit.evaluation.comparator.smart_match.SmartMatchComparator`,
:class:`~prkit.evaluation.comparator.category_match.CategoryComparator`, and
:class:`~prkit.evaluation.comparator.typed_llm.TypedLLMComparator` where the SmartMatch-style
path applies: pick a per-:class:`~prkit.core.domain.answer_category.AnswerCategory` compare
callable from a mapping, with optional logging and plain-text fallback on unexpected failures.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Union

from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.compare_same_type import compare_plain_text
from prkit.evaluation.utils.normalization import normalize_text

SameCategoryCompareFn = Callable[[Union[float, str], Union[float, str]], bool]


def compare_by_category(
    category: AnswerCategory,
    predicted_norm: Union[float, str],
    ground_truth_norm: Union[float, str],
    compare_fn_by_category: Mapping[AnswerCategory, SameCategoryCompareFn],
    logger: Optional[object] = None,
) -> bool:
    """Compare two normalized values using the category-specific compare function.

    For :attr:`AnswerCategory.TEXT`, applies :func:`~prkit.evaluation.utils.normalization.normalize_text`
    to both sides before dispatching. Unknown categories fall back to
    :func:`~prkit.evaluation.utils.compare_same_type.compare_plain_text`.

    On compare-function exception, logs a warning when *logger* is provided, then
    falls back to plain-text comparison (SmartMatch behavior).

    Args:
        category: Shared answer category for both sides.
        predicted_norm: Normalized model answer.
        ground_truth_norm: Normalized reference answer.
        compare_fn_by_category: Mapping from category to a same-type compare callable
            (e.g. :data:`SmartMatchComparator.DEFAULT_COMPARATORS`).
        logger: Optional logger with ``warning`` for fallback diagnostics.
    """
    if category == AnswerCategory.TEXT:
        predicted_norm = normalize_text(str(predicted_norm))
        ground_truth_norm = normalize_text(str(ground_truth_norm))
    compare_fn = compare_fn_by_category.get(category, compare_plain_text)
    try:
        return compare_fn(predicted_norm, ground_truth_norm)
    except Exception as e:
        if logger is not None:
            logger.warning(
                f"{category} comparator failed: {e}. "
                "Falling back to plain text comparison."
            )
        return compare_plain_text(predicted_norm, ground_truth_norm)
