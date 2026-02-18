"""
Comparator module for answer comparison strategies.

This module provides various comparators for comparing answers in physical
reasoning evaluation tasks.
"""

from .base import BaseComparator
from .category_match import CategoryComparator
from .exact_match import ExactMatchComparator
from .normalized_match import NormalizedMatchComparator
from .smart_match import SmartMatchComparator

__all__ = [
    "BaseComparator",
    "CategoryComparator",
    "ExactMatchComparator",
    "NormalizedMatchComparator",
    "SmartMatchComparator",
]
