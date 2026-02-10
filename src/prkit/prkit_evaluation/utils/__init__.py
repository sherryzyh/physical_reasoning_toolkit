"""
Utility functions for evaluation.

This module provides utility functions used across the evaluation package,
including answer normalization, formatting, and other helper functions.
"""

from prkit.prkit_core.domain.answer_category import AnswerCategory

from .normalization import (
    classify_expression,
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)
from .number_utils import (
    count_significant_digits,
    decimal_places,
    round_to_decimal_places,
    round_to_sig_figs,
)

__all__ = [
    "AnswerCategory",
    "classify_expression",
    "count_significant_digits",
    "decimal_places",
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
    "round_to_decimal_places",
    "round_to_sig_figs",
]
