"""
Base comparator class for answer comparison strategies.

This module provides the foundation for all answer comparators
in the physical reasoning toolkit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from prkit.prkit_core.domain.answer import Answer


class BaseComparator(ABC):
    """Base class for answer comparison strategies."""

    @abstractmethod
    def compare(self, answer1: Answer, answer2: Answer) -> Union[bool, float]:
        """
        Compare two answers and return comparison result.
        
        For exact match comparators, returns a boolean (True/False).
        For distance-based comparators, returns a numeric value (distance).
        
        Args:
            answer1: First answer to compare (typically predicted/student answer)
            answer2: Second answer to compare (typically ground truth/correct answer)
            
        Returns:
            Comparison result:
            - bool: True if answers match exactly, False otherwise
            - float: Numeric distance/score for distance-based comparison
        """
        pass

    @abstractmethod
    def accuracy_score(self, answer1: Answer, answer2: Answer) -> float:
        """
        Compute a normalized accuracy score between two answers in [0, 1].
        
        For exact match comparators, returns 1.0 if equal, 0.0 otherwise.
        For distance-based comparators, scales the distance to [0, 1].
        
        Args:
            answer1: First answer to compare
            answer2: Second answer to compare
            
        Returns:
            Accuracy score in [0, 1] where:
            - 1.0 means perfect match
            - 0.0 means no match
            - Values in between indicate partial accuracy
        """
        pass

    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """
        Check if this comparator can handle the given answer types.
        
        Default implementation returns True. Subclasses can override
        to restrict which answer types they can handle.
        
        Args:
            answer1: First answer to check
            answer2: Second answer to check
            
        Returns:
            True if this comparator can handle the answer types, False otherwise
        """
        return True
