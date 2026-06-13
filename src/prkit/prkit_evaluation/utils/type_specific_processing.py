"""Type-focused processing helpers shared across comparators."""

from __future__ import annotations

import re
from typing import Optional, Union

from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_semantics.normalization.physical_quantity_normalization import (
    parse_physical_quantity as _parse_semantics_physical_quantity,
)
from prkit.prkit_evaluation.utils.normalization import normalize_answer

_EQ_PATTERN = re.compile(r"^Eq\((.+),\s*(.+)\)$", re.DOTALL)


def parse_physical_quantity(s: str) -> tuple[Optional[float], str, str]:
    """Parse normalized physical quantity as ``(numeric_value, unit, num_str)``."""
    return _parse_semantics_physical_quantity(s)


def extract_rhs_and_category(
    norm_value: Union[float, str],
    category: AnswerCategory,
) -> tuple[Union[float, str], AnswerCategory]:
    """Extract equation RHS and re-normalize when input is equation-like."""
    if category != AnswerCategory.EQUATION:
        s = str(norm_value)
        if "=" not in s:
            return norm_value, category

    s = str(norm_value)
    rhs: Optional[str] = None
    eq_m = _EQ_PATTERN.match(s)
    if eq_m:
        rhs = eq_m.group(2).strip()
    elif "=" in s:
        rhs = s.rsplit("=", 1)[1].strip()

    if rhs:
        new_cat, new_norm = normalize_answer(rhs)
        return new_norm, new_cat
    return norm_value, category
