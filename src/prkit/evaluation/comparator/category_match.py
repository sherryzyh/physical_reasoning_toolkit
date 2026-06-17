"""
Category-Based Comparator for same-type answer comparison.

``compare`` / ``accuracy_score`` resolve same-type pairs via the shared
:func:`compare_by_category` dispatch. Cross-type pairs are compared as normalized
plain text only (GT-as-substring semantics via :func:`compare_plain_text`).

For cross-type deterministic matching (e.g. PQ vs NUMBER), see
:class:`SmartMatchComparator`.

Subclass hooks: overriding ``_comparators`` affects :meth:`_compare_by_category`
and :meth:`compare`.
"""

from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain.answer import Answer
from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.answer_utils import same_comparison_category
from prkit.evaluation.utils.category_dispatch import (
    SameCategoryCompareFn,
    compare_by_category,
)
from prkit.evaluation.utils.compare_same_type import (
    compare_formula,
    compare_number,
    compare_option,
    compare_physical_quantity,
    compare_plain_text,
)
from prkit.evaluation.utils.normalization import normalize_answer, normalize_text

from .base import BaseComparator


def _typed_category_and_value(
    answer: str | Answer,
) -> tuple[AnswerCategory | None, float | str]:
    """Return ``(category, normalized_value)`` for *answer*, or ``(None, raw_text)`` when normalization fails."""
    if isinstance(answer, Answer):
        return answer.answer_category, str(answer.value)
    try:
        category, normalized = normalize_answer(answer)
        return category, normalized
    except (ValueError, TypeError, RuntimeError):
        return None, str(answer).strip()


class CategoryComparator(BaseComparator):
    """
    Comparator restricted to same-type category dispatch plus plain-text
    fallback for cross-type pairs.

    ``answer1`` is treated as the model prediction and ``answer2`` as ground truth,
    matching :class:`TypedLLMComparator`. Optional ``kwargs`` (e.g. ``question``,
    ``symbolic_answer_is_expression``) are accepted for API compatibility with
    the LLM judge but are ignored — this comparator does not call an LLM.
    """

    DEFAULT_COMPARATORS: dict[AnswerCategory, SameCategoryCompareFn] = {
        AnswerCategory.NUMBER: compare_number,
        AnswerCategory.EQUATION: compare_plain_text,
        AnswerCategory.PHYSICAL_QUANTITY: compare_physical_quantity,
        AnswerCategory.FORMULA: compare_formula,
        AnswerCategory.TEXT: compare_plain_text,
        AnswerCategory.OPTION: compare_option,
    }

    def __init__(self) -> None:
        """Initialize with default category comparators."""
        super().__init__()
        self._comparators = dict(self.DEFAULT_COMPARATORS)
        self.logger = PRKitLogger.get_logger(__name__)

    def _compare_by_category(
        self,
        category: AnswerCategory,
        predicted_norm: float | str,
        ground_truth_norm: float | str,
    ) -> bool:
        """Compare two normalized values using the category-specific strategy."""
        return compare_by_category(
            category,
            predicted_norm,
            ground_truth_norm,
            self._comparators,
            self.logger,
        )

    def compare(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> bool:
        """
        True when same-type :func:`compare_by_category` matches, or when
        cross-type plain-text comparison matches after normalization.

        Cross-type pairs are not graded with category-specific logic; they are
        compared only as normalized text. For richer cross-type rules, use
        :class:`SmartMatchComparator`.
        """
        pred_cat, pred_value = _typed_category_and_value(answer1)
        gt_cat, gt_value = _typed_category_and_value(answer2)
        if pred_cat is None or gt_cat is None:
            return False

        if same_comparison_category(gt_cat, pred_cat):
            return self._compare_by_category(gt_cat, pred_value, gt_value)

        pred_text = normalize_text(str(pred_value))
        gt_text = normalize_text(str(gt_value))
        return compare_plain_text(pred_text, gt_text)

    def accuracy_score(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> float:
        """
        1.0 if :meth:`compare` is True, else 0.0.
        """
        is_match = self.compare(answer1, answer2, **kwargs)
        return 1.0 if is_match else 0.0
