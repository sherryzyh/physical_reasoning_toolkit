"""
Base evaluator class for physical reasoning evaluation.

This module provides the foundation for all evaluators that use
different comparators to evaluate answers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_evaluation.comparator.base import BaseComparator


class BaseEvaluator(ABC):
    """Base class for evaluators that use comparators."""

    def __init__(self, comparator: Optional[BaseComparator] = None):
        """
        Initialize the evaluator with a comparator.
        
        Args:
            comparator: Comparator instance to use for comparing answers.
                       If None, a default comparator will be used.
        """
        self.comparator = comparator

    @abstractmethod
    def evaluate(
        self,
        predicted_answer: Answer,
        ground_truth_answer: Answer,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a predicted answer against a ground truth answer.
        
        Args:
            predicted_answer: The predicted/student answer to evaluate
            ground_truth_answer: The ground truth/correct answer
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary containing evaluation results with keys such as:
            - accuracy_score: Accuracy score in [0, 1]
            - comparison_result: Raw comparison result from comparator
            - details: Additional evaluation details
        """
        pass

    def set_comparator(self, comparator: BaseComparator) -> None:
        """
        Set or change the comparator used by this evaluator.
        
        Args:
            comparator: Comparator instance to use
        """
        self.comparator = comparator

    def get_comparator(self) -> Optional[BaseComparator]:
        """
        Get the current comparator used by this evaluator.
        
        Returns:
            The current comparator instance, or None if not set
        """
        return self.comparator
