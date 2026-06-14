"""Backward-compatible evaluation normalization wrappers.

`prkit.semantics` owns the normalization logic. This module preserves the
legacy evaluation-facing API by re-exporting semantics normalization helpers and mapping
its answer kinds back to legacy `AnswerCategory` / string labels.
"""

from __future__ import annotations

import re

from latex2sympy2_extended import latex2sympy

from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.latex_symbol_preprocess import _preprocess_latex
from prkit.semantics.normalization.atomic_kinds import (
    NormalizedAtomicKind,
)
from prkit.semantics.normalization.atomic_normalization import (
    normalize_answer as _semantics_normalize_answer,
)
from prkit.semantics.normalization.atomic_normalization import (
    normalize_expression as _semantics_normalize_expression,
)
from prkit.semantics.normalization.atomic_normalization import (
    normalize_number,
)
from prkit.semantics.normalization.math_text_normalization import (
    _UNICODE_WHITESPACE,
    _extract_math_content,
    _match_balanced_braces,
    _normalize_unicode,
    _starts_with_latex_delimiter,
    normalize_text,
)
from prkit.semantics.normalization.physical_quantity_normalization import (
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
    _UNIT_ALIASES,
    _UNIT_TO_BASE,
    _canonicalize_quantity_string,
    _canonicalize_unit_alias,
    _evaluate_numeric_expression,
    _format_numeric_value,
    _normalize_physical_quantity,
    _normalize_unit_expression,
    _parse_exponent,
    _parse_numeric_base,
    _parse_unit_expression,
    _replace_superscript_exponents,
    _split_numeric_and_unit,
    _split_unit_exponent,
    _try_parse_number_only,
    _try_parse_physical_quantity,
)


def _legacy_expression_kind(kind: NormalizedAtomicKind) -> str:
    """Convert a semantics ``NormalizedAtomicKind`` to the legacy string label used by the evaluation API."""
    mapping = {
        NormalizedAtomicKind.NUMBER: "number",
        NormalizedAtomicKind.PHYSICAL_QUANTITY: "physical_quantity",
        NormalizedAtomicKind.RELATION: "equation",
        NormalizedAtomicKind.EXPRESSION: "formula",
    }
    return mapping.get(kind, "formula")


def _legacy_answer_category(kind: NormalizedAtomicKind) -> AnswerCategory:
    """Convert a semantics ``NormalizedAtomicKind`` to the evaluation-layer ``AnswerCategory``."""
    mapping = {
        NormalizedAtomicKind.NUMBER: AnswerCategory.NUMBER,
        NormalizedAtomicKind.PHYSICAL_QUANTITY: AnswerCategory.PHYSICAL_QUANTITY,
        NormalizedAtomicKind.RELATION: AnswerCategory.EQUATION,
        NormalizedAtomicKind.EXPRESSION: AnswerCategory.FORMULA,
        NormalizedAtomicKind.TEXT: AnswerCategory.TEXT,
    }
    return mapping[kind]


def classify_expression(clean_str: str) -> str:
    """Classify a cleaned expression using the legacy string labels."""

    from prkit.semantics.normalization.expression_normalization import (
        classify_expression as _classify_expression,
    )

    return _legacy_expression_kind(_classify_expression(clean_str))


def normalize_expression(
    answer_str: str,
) -> tuple[float | str, bool, str]:
    """Normalize an expression while preserving legacy category labels."""

    normalized, success, kind = _semantics_normalize_expression(answer_str)
    return normalized, success, _legacy_expression_kind(kind)


def normalize_answer(
    answer_str: str,
) -> tuple[AnswerCategory, float | str]:
    """Normalize an answer string via the semantics-owned implementation."""

    kind, normalized = _semantics_normalize_answer(answer_str)
    return _legacy_answer_category(kind), normalized


def _normalize_symbolic_expression(
    clean_math: str, had_latex_patterns: bool
) -> tuple[str, bool]:
    """Legacy-compatible symbolic normalization helper for tests and aliases."""

    if had_latex_patterns:
        preprocessed = _preprocess_latex(clean_math)
        try:
            symbolic_expr = latex2sympy(preprocessed)
            normalized = re.sub(r"\s+", " ", str(symbolic_expr)).strip()
            return normalized, True
        except Exception:
            return preprocessed, False

    normalized = re.sub(r"\s+", " ", clean_math).strip()
    return normalized, True


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
    "_normalize_unicode",
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
    "latex2sympy",
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
]
