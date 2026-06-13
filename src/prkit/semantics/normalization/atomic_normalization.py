"""Atomic answer normalization built from focused helper modules."""

from __future__ import annotations

import functools
import math

from .atomic_kinds import NormalizedAtomicKind
from .expression_normalization import (
    _normalize_symbolic_expression,
    classify_expression,
)
from .math_text_normalization import (
    _contains_latex_commands,
    _extract_math_content,
    _looks_like_math_expression,
    _starts_with_latex_delimiter,
    _unicode_math_to_latex,
    normalize_text,
)
from .physical_quantity_normalization import (
    _normalize_physical_quantity,
    _try_parse_number_only,
    _try_parse_physical_quantity,
    normalize_number,
)


def normalize_expression(
    answer_str: str,
) -> tuple[float | str, bool, NormalizedAtomicKind]:
    """Normalize an answer surface that should be interpreted as math.

    The return value packages three pieces of information:

    - the normalized surface,
    - whether normalization confidently succeeded, and
    - the resulting atomic kind.

    This function is used as a second-stage rescue after the caller has
    already decided that plain-text normalization is insufficient.
    """

    clean_math, had_latex_patterns = _extract_math_content(answer_str)
    if not had_latex_patterns and _contains_latex_commands(clean_math):
        had_latex_patterns = True

    expr_kind = classify_expression(clean_math)
    if expr_kind == NormalizedAtomicKind.PHYSICAL_QUANTITY:
        return (
            _normalize_physical_quantity(clean_math),
            True,
            NormalizedAtomicKind.PHYSICAL_QUANTITY,
        )

    if expr_kind == NormalizedAtomicKind.EXPRESSION:
        rescued_number = _try_parse_number_only(clean_math)
        if rescued_number is not None:
            return rescued_number, True, NormalizedAtomicKind.NUMBER
        rescued_quantity = _try_parse_physical_quantity(clean_math)
        if rescued_quantity is not None:
            return rescued_quantity, True, NormalizedAtomicKind.PHYSICAL_QUANTITY

    normalized, success = _normalize_symbolic_expression(clean_math, had_latex_patterns)
    return normalized, success, expr_kind


@functools.lru_cache(maxsize=8192)
def normalize_answer(
    answer_str: str,
) -> tuple[NormalizedAtomicKind, float | str]:
    """Normalize a raw atomic answer into a coarse kind plus payload.

    The helper intentionally stops short of building full
    ``PhysicsAnswerSemantics`` objects. Instead it classifies a single
    answer surface as number, quantity, symbolic expression/relation, or
    plain text so the higher-level answer normalizer can attach the
    remaining protocol metadata.
    """

    normalized_number = normalize_number(answer_str)
    if not (isinstance(normalized_number, float) and math.isnan(normalized_number)):
        return NormalizedAtomicKind.NUMBER, normalized_number

    if not _starts_with_latex_delimiter(answer_str):
        clean_str, had_latex = _extract_math_content(answer_str)
        expr_kind = classify_expression(clean_str)
        if expr_kind == NormalizedAtomicKind.PHYSICAL_QUANTITY:
            return (
                NormalizedAtomicKind.PHYSICAL_QUANTITY,
                _normalize_physical_quantity(clean_str),
            )
        if expr_kind == NormalizedAtomicKind.RELATION and _looks_like_math_expression(
            clean_str
        ):
            effective_had_latex = had_latex or _contains_latex_commands(clean_str)
            normalized, success = _normalize_symbolic_expression(
                clean_str, effective_had_latex
            )
            if success:
                return NormalizedAtomicKind.RELATION, normalized

        expression_str = _unicode_math_to_latex(answer_str)
        if _contains_latex_commands(expression_str):
            normalized_expr, success, expr_kind = normalize_expression(expression_str)
            if success:
                return expr_kind, normalized_expr
        return NormalizedAtomicKind.TEXT, normalize_text(clean_str)

    expression_str = _unicode_math_to_latex(answer_str)
    normalized_expr, success, expr_kind = normalize_expression(expression_str)
    if success:
        return expr_kind, normalized_expr

    clean_str, _ = _extract_math_content(answer_str)
    return NormalizedAtomicKind.TEXT, normalize_text(clean_str)


__all__ = [
    "normalize_answer",
    "normalize_expression",
]
