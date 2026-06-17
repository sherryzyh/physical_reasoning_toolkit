"""Abstract base class for all answer comparators in PRKit.

.. deprecated::
    The comparator/evaluator stacks are superseded by the unified, version-stamped
    :class:`prkit.scoring.SemanticsScorer` (the :class:`prkit.api.Scorer` /
    :class:`prkit.api.Verdict` contract), which wraps the deterministic semantics
    comparison engine. Constructing any comparator emits a ``DeprecationWarning``;
    these classes will be removed in a future release. See ``prkit/CONTRACT.md``.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

from prkit.core.domain.answer import Answer

#: Shared pointer to the replacement, reused by the evaluator stack.
DEPRECATION_HINT = (
    "use prkit.scoring.SemanticsScorer (the prkit.api.Scorer / Verdict contract) "
    "instead; see prkit/CONTRACT.md"
)


class BaseComparator(ABC):
    """Base class for answer comparison strategies.

    .. deprecated:: superseded by :class:`prkit.scoring.SemanticsScorer`.
    """

    def __init__(self) -> None:
        warnings.warn(
            f"{type(self).__name__} is deprecated and will be removed in a future "
            f"release; {DEPRECATION_HINT}.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def compare(
        self, answer1: str | Answer, answer2: str | Answer, **kwargs: Any
    ) -> Any:
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
    def accuracy_score(
        self, answer1: str | Answer, answer2: str | Answer, **kwargs: Any
    ) -> float:
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
