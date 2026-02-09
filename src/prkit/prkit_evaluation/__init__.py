"""
Physical Reasoning Evaluation Package.

This package provides tools for evaluating answers in physical reasoning tasks,
including comparators for comparing answers and evaluators for performing
evaluations.
"""

from prkit.prkit_evaluation.comparator import BaseComparator, ExactMatchComparator
from prkit.prkit_evaluation.evaluator import AccuracyEvaluator, BaseEvaluator

__all__ = [
    "BaseComparator",
    "ExactMatchComparator",
    "BaseEvaluator",
    "AccuracyEvaluator",
]
