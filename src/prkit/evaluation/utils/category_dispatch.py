"""Dispatch normalized prediction/ground-truth to the per-category compare function."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from prkit.core.domain.answer import AnswerValue
from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.compare_same_type import compare_plain_text
from prkit.evaluation.utils.normalization import normalize_text

SameCategoryCompareFn = Callable[[AnswerValue, AnswerValue], bool]


def compare_by_category(
    category: AnswerCategory,
    predicted_norm: AnswerValue,
    ground_truth_norm: AnswerValue,
    compare_fn_by_category: Mapping[AnswerCategory, SameCategoryCompareFn],
    logger: logging.Logger | None = None,
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
