"""
Normalized Match Comparator for answer comparison.

This comparator performs preprocessing and normalization before comparison:
- Numbers: normalized to float format (handles integers, decimals, scientific notation)
- Expression formulas: LaTeX expressions normalized to a single format
- Text: whitespace stripped
"""

import math
from typing import Literal, Tuple, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_type import AnswerType

from prkit.prkit_evaluation.utils.normalization import (
    normalize_answer,
    normalize_expression,
    normalize_text,
)

from .base import BaseComparator


class NormalizedMatchComparator(BaseComparator):
    """Comparator that normalizes answers before exact matching."""

    def _normalize_answer(
        self,
        answer_str: str,
    ) -> Tuple[Literal["number", "expression", "text"], Union[float, str]]:
        """
        Normalize an answer string based on its category.
        
        Args:
            answer_str: The answer string to normalize
            
        Returns:
            Tuple of (category, normalized_value)
        """
        return normalize_answer(answer_str)
    
    def _normalize_expression(self, answer_str: str) -> str:
        """
        Normalize an expression string (handles both LaTeX and plain string formats).
        
        This is used when the answer type is already known to be SYMBOLIC/expression.
        It will normalize LaTeX delimiters if present, or just normalize whitespace
        if the expression is in plain string format.
        
        Args:
            answer_str: The expression string to normalize
            
        Returns:
            Normalized expression string
        """
        normalized, _ = normalize_expression(answer_str)
        return normalized

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
        # Convert both answers to strings
        ans1_str = str(answer1.value)
        ans2_str = str(answer2.value)
        
        # If both answers are known to be SYMBOLIC (expression), normalize as expressions
        # This handles both LaTeX-formatted and plain string expressions
        if answer1.answer_type == AnswerType.SYMBOLIC and answer2.answer_type == AnswerType.SYMBOLIC:
            norm1 = self._normalize_expression(ans1_str)
            norm2 = self._normalize_expression(ans2_str)
            return norm1 == norm2
        
        # Otherwise, use auto-detection normalization
        cat1, norm1 = self._normalize_answer(ans1_str)
        cat2, norm2 = self._normalize_answer(ans2_str)
        
        # If categories are different, treat both as text
        if cat1 != cat2:
            # Normalize both as text and compare
            text1 = self._normalize_as_text(ans1_str)
            text2 = self._normalize_as_text(ans2_str)
            return text1 == text2
        
        # Compare based on category
        if cat1 == "number":
            if not (isinstance(norm1, float) and isinstance(norm2, float)):
                return False
            # Use a small epsilon for floating point comparison
            return abs(norm1 - norm2) < 1e-10
        else:
            # For expression and text, compare strings
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
