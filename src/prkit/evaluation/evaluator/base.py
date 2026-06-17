"""Abstract base evaluator for physical reasoning evaluation, delegating comparisons to a :class:`BaseComparator`.

.. deprecated::
    The evaluator stack is superseded by :class:`prkit.scoring.SemanticsScorer`
    (the :class:`prkit.api.Scorer` / :class:`prkit.api.Verdict` contract).
    Constructing an evaluator emits a ``DeprecationWarning``; this stack will be
    removed in a future release. See ``prkit/CONTRACT.md``.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

from prkit.core.domain.answer import Answer
from prkit.evaluation.comparator.base import DEPRECATION_HINT, BaseComparator


class BaseEvaluator(ABC):
    """Base class for evaluators that use comparators.

    .. deprecated:: superseded by :class:`prkit.scoring.SemanticsScorer`.
    """

    def __init__(self, comparator: BaseComparator | None = None):
        """
        Initialize the evaluator with a comparator.

        Args:
            comparator: Comparator instance to use for comparing answers.
                       If None, a default comparator will be used.
        """
        warnings.warn(
            f"{type(self).__name__} is deprecated and will be removed in a future "
            f"release; {DEPRECATION_HINT}.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.comparator = comparator

    @abstractmethod
    def evaluate(
        self, predicted_answer: Answer, ground_truth_answer: Answer, **kwargs: Any
    ) -> dict[str, Any]:
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

    def get_comparator(self) -> BaseComparator | None:
        """
        Get the current comparator used by this evaluator.

        Returns:
            The current comparator instance, or None if not set
        """
        return self.comparator
