"""
Normalized Match Comparator for answer comparison.

This comparator performs preprocessing and normalization before comparison:
- number: normalized to float format, compared for equality
- physical_quantity: number + units (placeholder comparison; user will add later)
- equation, formula, text: normalized and compared as identical strings

Accepts either Answer objects or raw strings.
"""

from typing import Any, Tuple, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory

from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category, to_str
from prkit.prkit_evaluation.utils.normalization import normalize_answer, normalize_text
from prkit.prkit_evaluation.utils.number_utils import DEFAULT_NUMBER_EPSILON

from .base import BaseComparator


class NormalizedMatchComparator(BaseComparator):
    """Comparator that normalizes answers before exact matching."""

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        **kwargs: Any
    ) -> bool:
        """
        Compare two answers after normalization.
        
        If the answers are in different categories, both are normalized as text
        and compared as strings.
        
        Args:
            answer1: First answer to compare (string or Answer)
            answer2: Second answer to compare (string or Answer)
            
        Returns:
            True if answers match after normalization, False otherwise
        """
        ans1_str = to_str(answer1)
        ans2_str = to_str(answer2)

        # Option answers: direct comparison (case-insensitive) when both are Answer with OPTION
        if isinstance(answer1, Answer) and isinstance(answer2, Answer):
            if (
                answer1.answer_category == AnswerCategory.OPTION
                and answer2.answer_category == AnswerCategory.OPTION
            ):
                return ans1_str.strip().upper() == ans2_str.strip().upper()

        # Use auto-detection normalization (categorizes as number, equation,
        # physical_quantity, formula, or text)
        cat1, norm1 = normalize_answer(ans1_str)
        cat2, norm2 = normalize_answer(ans2_str)
        
        # If categories differ, treat both as text and compare as strings
        if not same_comparison_category(cat1, cat2):
            # Normalize both as text and compare
            text1 = normalize_text(ans1_str)
            text2 = normalize_text(ans2_str)
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

    def accuracy_score(self, answer1: Union[str, Answer], answer2: Union[str, Answer]) -> float:
        """
        Compute accuracy score for normalized match comparison.
        
        Returns 1.0 if answers match after normalization, 0.0 otherwise.
        
        Args:
            answer1: First answer to compare (string or Answer)
            answer2: Second answer to compare (string or Answer)
            
        Returns:
            1.0 if answers match after normalization, 0.0 otherwise
        """
        return 1.0 if self.compare(answer1, answer2) else 0.0
