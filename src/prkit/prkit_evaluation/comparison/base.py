"""
Base comparator class for answer comparison strategies.

This module provides the foundation for all answer comparators
in the physical reasoning toolkit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from prkit.prkit_core.models.answer import Answer


class BaseComparator(ABC):
    """Base class for answer comparison strategies."""

    @abstractmethod
    def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Compare two answers and return comparison results.

        Args:
            answer1: First answer to compare
            answer2: Second answer to compare

        Returns:
            Dictionary containing comparison results with keys:
            - is_equal: Boolean indicating if answers are equivalent
            - details: Additional comparison details
        """
        pass

    @abstractmethod
    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """Check if this comparator can handle the given answer types."""
        pass
