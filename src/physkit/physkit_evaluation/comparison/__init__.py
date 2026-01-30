"""
Comparison strategies for different answer types.

This module provides comparison strategies for evaluating the similarity
or equivalence of different types of answers.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .base import BaseComparator
from .symbolic import SymbolicComparator
from .numerical import NumericalComparator
from .textual import TextualComparator
from .option import OptionComparator
from .answer_comparator import SmartAnswerComparator

__all__ = [
    "BaseComparator",
    "SymbolicComparator",
    "NumericalComparator",
    "TextualComparator",
    "OptionComparator",
    "SmartAnswerComparator",
]
