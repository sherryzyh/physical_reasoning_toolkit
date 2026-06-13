"""Deprecated backward-compatible aliases for the legacy v1 normalization module.

.. deprecated::
    The toolkit maintains a single normalization implementation in
    :mod:`prkit.evaluation.utils.normalization`. This shim re-exports the legacy
    v1 helpers so older imports keep working, but it will be removed in a future
    release. Import directly from ``prkit.evaluation.utils.normalization`` instead.
"""

import warnings

warnings.warn(
    "prkit.evaluation.utils.normalization_v1 is deprecated and will be removed in a "
    "future release; import from prkit.evaluation.utils.normalization instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
