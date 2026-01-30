"""
Physical Reasoning Evaluation Toolkit

A comprehensive toolkit for evaluating physical reasoning models with support for
multiple answer types and evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .comparison import (
    BaseComparator,
    NumericalComparator,
    SmartAnswerComparator,
    SymbolicComparator,
    TextualComparator,
)
from .metrics import AccuracyMetric

__all__ = [
    "AccuracyMetric",
    "BaseComparator",
    "SymbolicComparator",
    "NumericalComparator",
    "TextualComparator",
    "SmartAnswerComparator",
]
