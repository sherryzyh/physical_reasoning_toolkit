"""Shared evaluation helpers: answer normalization, comparison dispatch, and numeric utilities."""

from prkit.core.domain.answer_category import AnswerCategory

from .category_dispatch import compare_by_category
from .compare_same_type import (
    compare_formula,
    compare_number,
    compare_physical_quantity,
    compare_plain_text,
)
from .normalization import (
    classify_expression,
    normalize_answer,
    normalize_expression,
    normalize_number,
    normalize_text,
)
from .number_utils import (
    decimal_places,
    round_to_decimal_places,
)

__all__ = [
    "AnswerCategory",
    "classify_expression",
    "compare_by_category",
    "compare_formula",
    "compare_number",
    "compare_physical_quantity",
    "compare_plain_text",
    "decimal_places",
    "normalize_answer",
    "normalize_expression",
    "normalize_number",
    "normalize_text",
    "round_to_decimal_places",
]
