"""
Evaluator module for physical reasoning evaluation.

This module provides evaluators that use different comparators to evaluate
answers in physical reasoning tasks.
"""

from .accuracy import AccuracyEvaluator
from .base import BaseEvaluator

__all__ = [
    "BaseEvaluator",
    "AccuracyEvaluator",
]
