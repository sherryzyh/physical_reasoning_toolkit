"""
Utility functions for evaluation.

This module provides utility functions used across the evaluation package,
including answer normalization, formatting, and other helper functions.
"""

from .normalization import (
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)

__all__ = [
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
]
