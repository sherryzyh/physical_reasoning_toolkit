"""
Normalized Match Comparator for answer comparison.

This comparator performs preprocessing and normalization before comparison:
- number: normalized to float format, compared for equality
- physical_quantity: number + units (placeholder comparison; user will add later)
- equation, formula, text: normalized and compared as identical strings
"""

from typing import Tuple, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory

from prkit.prkit_evaluation.utils.normalization import normalize_answer, normalize_text
from prkit.prkit_evaluation.utils.number_utils import DEFAULT_NUMBER_EPSILON

from .base import BaseComparator


def _same_comparison_category(
    cat1: AnswerCategory, cat2: AnswerCategory
) -> bool:
    """True if both answers should be compared using the same comparison strategy."""
    return cat1 == cat2


class NormalizedMatchComparator(BaseComparator):
    """Comparator that normalizes answers before exact matching."""

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

    def _normalize_as_text(self, answer_str: str) -> str:
        """
        Normalize an answer string as text (strip whitespace).
        
        This is used when comparing answers from different categories.
        
        Args:
            answer_str: The answer string to normalize as text
            
        Returns:
            Normalized text string
        """
        return normalize_text(answer_str)

    def compare(self, answer1: Answer, answer2: Answer) -> bool:
        """
        Compare two answers after normalization.
        
        If the answers are in different categories, both are normalized as text
        and compared as strings.
        
        Args:
            answer1: First answer to compare (typically predicted/student answer)
            answer2: Second answer to compare (typically ground truth/correct answer)
            
        Returns:
            True if answers match after normalization, False otherwise
        """
        # Option answers: direct comparison (case-insensitive)
        if (
            answer1.answer_category == AnswerCategory.OPTION
            and answer2.answer_category == AnswerCategory.OPTION
        ):
            return (
                str(answer1.value).strip().upper()
                == str(answer2.value).strip().upper()
            )

        # Convert both answers to strings
        ans1_str = str(answer1.value)
        ans2_str = str(answer2.value)

        # Use auto-detection normalization (categorizes as number, equation,
        # physical_quantity, formula, or text)
        cat1, norm1 = self._normalize_answer(ans1_str)
        cat2, norm2 = self._normalize_answer(ans2_str)
        
        # If categories differ, treat both as text and compare as strings
        if not _same_comparison_category(cat1, cat2):
            # Normalize both as text and compare
            text1 = self._normalize_as_text(ans1_str)
            text2 = self._normalize_as_text(ans2_str)
            return text1 == text2
        
        # Compare based on category
        if cat1 == AnswerCategory.NUMBER:
            if not (isinstance(norm1, float) and isinstance(norm2, float)):
                return False
            return abs(norm1 - norm2) < DEFAULT_NUMBER_EPSILON
        else:
            # equation, formula, physical_quantity, text: identical string comparison
            if not (isinstance(norm1, str) and isinstance(norm2, str)):
                return False
            return norm1 == norm2

    def accuracy_score(self, answer1: Answer, answer2: Answer) -> float:
        """
        Compute accuracy score for normalized match comparison.
        
        Returns 1.0 if answers match after normalization, 0.0 otherwise.
        
        Args:
            answer1: First answer to compare
            answer2: Second answer to compare
            
        Returns:
            1.0 if answers match after normalization, 0.0 otherwise
        """
        is_match = self.compare(answer1, answer2)
        return 1.0 if is_match else 0.0
