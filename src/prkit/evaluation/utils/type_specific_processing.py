"""Type-focused processing helpers shared across comparators."""

from __future__ import annotations

import re

from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.normalization import normalize_answer
from prkit.semantics.normalization.physical_quantity_normalization import (
    parse_physical_quantity as _parse_semantics_physical_quantity,
)

_EQ_PATTERN = re.compile(r"^Eq\((.+),\s*(.+)\)$", re.DOTALL)


def parse_physical_quantity(s: str) -> tuple[float | None, str, str]:
    """Parse normalized physical quantity as ``(numeric_value, unit, num_str)``."""
    return _parse_semantics_physical_quantity(s)


def extract_rhs_and_category(
    norm_value: float | str,
    category: AnswerCategory,
) -> tuple[float | str, AnswerCategory]:
    """Extract equation RHS and re-normalize when input is equation-like."""
    if category != AnswerCategory.EQUATION:
        s = str(norm_value)
        if "=" not in s:
            return norm_value, category

    s = str(norm_value)
    rhs: str | None = None
    eq_m = _EQ_PATTERN.match(s)
    if eq_m:
        rhs = eq_m.group(2).strip()
    elif "=" in s:
        rhs = s.rsplit("=", 1)[1].strip()

    if rhs:
        new_cat, new_norm = normalize_answer(rhs)
        return new_norm, new_cat
    return norm_value, category
