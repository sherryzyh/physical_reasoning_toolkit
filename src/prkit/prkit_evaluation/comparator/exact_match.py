"""
Exact Match Comparator for answer comparison.

This comparator performs exact string matching between two answers.
"""

from prkit.prkit_core.domain.answer import Answer

from .base import BaseComparator


class ExactMatchComparator(BaseComparator):
    """Comparator that performs exact string matching between answers."""

    def compare(self, answer1: Answer, answer2: Answer) -> bool:
        """
        Compare two answers exactly.
        
        Args:
            answer1: First answer to compare (typically predicted/student answer)
            answer2: Second answer to compare (typically ground truth/correct answer)
            
        Returns:
            True if answers match exactly, False otherwise
        """
        # Convert both answers to strings and compare
        ans1_str = str(answer1.value).strip()
        ans2_str = str(answer2.value).strip()
        
        return ans1_str == ans2_str

    def accuracy_score(self, answer1: Answer, answer2: Answer) -> float:
        """
        Compute accuracy score for exact match comparison.
        
        Returns 1.0 if answers match exactly, 0.0 otherwise.
        
        Args:
            answer1: First answer to compare
            answer2: Second answer to compare
            
        Returns:
            1.0 if answers match exactly, 0.0 otherwise
        """
        is_match = self.compare(answer1, answer2)
        return 1.0 if is_match else 0.0
