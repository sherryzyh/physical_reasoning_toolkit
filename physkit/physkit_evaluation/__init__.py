"""
Physical Reasoning Evaluation Toolkit

A comprehensive toolkit for evaluating physical reasoning models with support for
multiple answer types and evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "Physical Reasoning Team"

from .metrics import AccuracyMetric
from .comparison import BaseComparator, SymbolicComparator, NumericalComparator, TextualComparator, SmartAnswerComparator

__all__ = [
    "AccuracyMetric",
    "BaseComparator",
    "SymbolicComparator",
    "NumericalComparator",
    "TextualComparator",
    "SmartAnswerComparator",
]
