"""Deprecated backward-compatible aliases for the legacy v2 normalization module.

.. deprecated::
    The toolkit maintains a single normalization implementation in
    :mod:`prkit.evaluation.utils.normalization`. This shim re-exports the legacy
    v2 helpers, constants, and utility names so older imports keep working, but it
    will be removed in a future release. Import directly from
    ``prkit.evaluation.utils.normalization`` instead.
"""

import warnings

from .normalization import (
    _FRAC_LATEX_PATTERN,
    _FRACTION_RE,
    _FRACTION_TOKEN,
    _NUM_TOKEN,
    _NUMERIC_PREFIX_RE,
    _POWER_RE,
    _POWER_TOKEN,
    _QUANTITY_PATTERN,
    _SCI_10_RE,
    _SCI_10_TOKEN,
    _SCI_E_TOKEN,
    _SIGNED_NUM_OR_E_TOKEN,
    _SIGNED_NUM_TOKEN,
    _SIMPLE_NUMBER_RE,
    _SUPERSCRIPT_TRANSLATION,
    _UNICODE_WHITESPACE,
    _UNIT_ALIASES,
    _UNIT_TO_BASE,
    _canonicalize_quantity_string,
    _canonicalize_unit_alias,
    _evaluate_numeric_expression,
    _extract_math_content,
    _format_numeric_value,
    _match_balanced_braces,
    _normalize_physical_quantity,
    _normalize_symbolic_expression,
    _normalize_unit_expression,
    _parse_exponent,
    _parse_numeric_base,
    _parse_unit_expression,
    _replace_superscript_exponents,
    _split_numeric_and_unit,
    _split_unit_exponent,
    _starts_with_latex_delimiter,
    _try_parse_number_only,
    _try_parse_physical_quantity,
    classify_expression,
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)

warnings.warn(
    "prkit.evaluation.utils.normalization_v2 is deprecated and will be removed in a "
    "future release; import from prkit.evaluation.utils.normalization instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "_FRAC_LATEX_PATTERN",
    "_FRACTION_RE",
    "_FRACTION_TOKEN",
    "_NUMERIC_PREFIX_RE",
    "_NUM_TOKEN",
    "_POWER_RE",
    "_POWER_TOKEN",
    "_QUANTITY_PATTERN",
    "_SCI_10_RE",
    "_SCI_10_TOKEN",
    "_SCI_E_TOKEN",
    "_SIGNED_NUM_OR_E_TOKEN",
    "_SIGNED_NUM_TOKEN",
    "_SIMPLE_NUMBER_RE",
    "_SUPERSCRIPT_TRANSLATION",
    "_UNICODE_WHITESPACE",
    "_UNIT_ALIASES",
    "_UNIT_TO_BASE",
    "_canonicalize_quantity_string",
    "_canonicalize_unit_alias",
    "_evaluate_numeric_expression",
    "_extract_math_content",
    "_format_numeric_value",
    "_match_balanced_braces",
    "_normalize_physical_quantity",
    "_normalize_symbolic_expression",
    "_normalize_unit_expression",
    "_parse_exponent",
    "_parse_numeric_base",
    "_parse_unit_expression",
    "_replace_superscript_exponents",
    "_split_numeric_and_unit",
    "_split_unit_exponent",
    "_starts_with_latex_delimiter",
    "_try_parse_number_only",
    "_try_parse_physical_quantity",
    "classify_expression",
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
]
