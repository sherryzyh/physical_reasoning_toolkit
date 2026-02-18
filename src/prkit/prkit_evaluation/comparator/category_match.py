"""
Category-Based Comparator for answer comparison.

This comparator normalizes answers first, then applies category-specific comparison
strategies. For the same category, a dedicated comparison function is used;
for different categories, both answers are normalized as text and compared.

Category-specific comparison details can be customized by subclassing and
overriding _get_category_comparators().
"""

from typing import Union

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category
from prkit.prkit_evaluation.utils.normalization import normalize_answer, normalize_text
from prkit.prkit_evaluation.utils.compare_by_type import (
    compare_number,
    compare_plain_text,
    compare_physical_quantity,
    compare_formula,
)

from .base import BaseComparator


class CategoryComparator(BaseComparator):
    """
    Comparator that normalizes answers first, then applies category-specific
    comparison strategies.

    Category-specific comparators can be customized by subclassing and
    overriding _get_category_comparators().

    Categories: number, equation, physical_quantity, formula, text
    """

    # Default comparators per category (placeholders; customize as needed)
    DEFAULT_COMPARATORS = {
        AnswerCategory.NUMBER: compare_number,
        AnswerCategory.EQUATION: compare_plain_text,
        AnswerCategory.PHYSICAL_QUANTITY: compare_physical_quantity,
        AnswerCategory.FORMULA: compare_formula,
        AnswerCategory.TEXT: compare_plain_text,
        AnswerCategory.OPTION: compare_plain_text,
    }

    def __init__(self):
        """Initialize with default category comparators."""
        self._comparators = dict(self.DEFAULT_COMPARATORS)
        self.logger = PRKitLogger.get_logger(__name__)


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
        if category == AnswerCategory.TEXT:
            predicted_norm = normalize_text(str(predicted_norm))
            ground_truth_norm = normalize_text(str(ground_truth_norm))
        comparator = self._comparators.get(category, compare_plain_text)
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
        
        if isinstance(answer1, Answer):
            pred_norm = str(answer1.value)
            pred_string = pred_norm
            pred_cat = answer1.answer_category
            self.logger.debug(f"model answer (answer object): {pred_norm} ({pred_cat})")
        else:
            pred_cat, pred_norm = normalize_answer(answer1)
            pred_string = str(answer1)
            self.logger.debug(f"model answer (answer string): {pred_norm} ({pred_cat})")

        if isinstance(answer2, Answer):
            gt_norm = str(answer2.value)
            gt_string = gt_norm
            gt_cat = answer2.answer_category
            self.logger.debug(f"gt answer (answer object): {gt_norm} ({gt_cat})")
        else:
            gt_cat, gt_norm = normalize_answer(answer2)
            gt_string = str(answer2)
            self.logger.debug(f"gt answer (answer string): {gt_norm} ({gt_cat})")
            
        if same_comparison_category(gt_cat, pred_cat):
            return self._compare_by_category(gt_cat, pred_norm, gt_norm)
        else:
            pred_norm = normalize_text(pred_string)
            gt_norm = normalize_text(gt_string)
            return pred_norm == gt_norm


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
