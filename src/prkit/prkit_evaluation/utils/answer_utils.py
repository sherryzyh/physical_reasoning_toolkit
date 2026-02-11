"""
Answer utilities for comparison.

This module provides functions for converting Answer objects or raw strings
to strings, and helpers for comparison logic.
"""

from typing import Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory


def to_str(a: Union[str, Answer]) -> str:
    """Extract string from Answer or return str as-is, with leading/trailing whitespace stripped."""
    return str(a.value).strip() if isinstance(a, Answer) else str(a).strip()


def same_comparison_category(
    cat1: AnswerCategory, cat2: AnswerCategory
) -> bool:
    """True if both answers should be compared using the same comparison strategy."""
    return cat1 == cat2
