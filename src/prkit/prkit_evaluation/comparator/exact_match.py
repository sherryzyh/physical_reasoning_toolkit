"""
Exact Match Comparator for answer comparison.

This comparator performs exact string matching between two answers.
Accepts either Answer objects or raw strings.
"""

from typing import Any, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_evaluation.utils.answer_utils import to_str

from .base import BaseComparator


class ExactMatchComparator(BaseComparator):
    """Comparator that performs exact string matching between answers."""

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        **kwargs: Any
    ) -> bool:
        """
        Compare two answers exactly.
        
        Args:
            answer1: First answer to compare (string or Answer)
            answer2: Second answer to compare (string or Answer)
            
        Returns:
            True if answers match exactly, False otherwise
        """
        return to_str(answer1) == to_str(answer2)

    def accuracy_score(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        **kwargs: Any
    ) -> float:
        """
        Compute accuracy score for exact match comparison.
        
        Returns 1.0 if answers match exactly, 0.0 otherwise.
        
        Args:
            answer1: First answer to compare
            answer2: Second answer to compare
            
        Returns:
            1.0 if answers match exactly, 0.0 otherwise
        """
        return 1.0 if self.compare(answer1, answer2) else 0.0
