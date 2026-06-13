"""Backward-compatible aliases for the legacy v1 normalization module.

The toolkit now maintains a single normalization implementation in
``prkit.prkit_evaluation.utils.normalization``. This module re-exports the
legacy v1 public helpers and internal utility names so older imports continue
to work while fixes land in one place.
"""

from .normalization import (
    _FRAC_LATEX_PATTERN,
    _extract_math_content,
    _format_numeric_value,
    _match_balanced_braces,
    _normalize_physical_quantity,
    _normalize_symbolic_expression,
    _parse_exponent,
    _parse_numeric_base,
    _starts_with_latex_delimiter,
    classify_expression,
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)

__all__ = [
    "_FRAC_LATEX_PATTERN",
    "_extract_math_content",
    "_format_numeric_value",
    "_match_balanced_braces",
    "_normalize_physical_quantity",
    "_normalize_symbolic_expression",
    "_parse_exponent",
    "_parse_numeric_base",
    "_starts_with_latex_delimiter",
    "classify_expression",
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
]
