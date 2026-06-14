"""Helpers for cross-category answer matching (e.g. formula vs text, equation vs number)."""

from __future__ import annotations

import re

from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.compare_same_type import (
    compare_formula,
    compare_physical_quantity,
)
from prkit.evaluation.utils.normalization import normalize_answer
from prkit.evaluation.utils.type_specific_processing import extract_rhs_and_category

_EQ_PATTERN = re.compile(r"^Eq\((.+),\s*(.+)\)$", re.DOTALL)


def split_respecting_parens(s: str, delimiter: str = ",") -> list[str]:
    """Split on delimiter only at depth-0 (outside nested parentheses)."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(depth - 1, 0)
        if ch == delimiter and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def expand_gt_set(gt_norm: float | str) -> list[str]:
    """Split a SymPy-style set literal ``{a, b, …}`` into individual candidate strings."""
    s = str(gt_norm).strip()
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        parts = split_respecting_parens(inner)
        return [p.strip() for p in parts if p.strip()]
    return [s]


def strip_unbalanced_parens(s: str) -> str:
    """Strip leading/trailing unmatched parentheses and brackets."""
    s = s.strip()
    for _open, _close in [("(", ")"), ("[", "]")]:
        while s.endswith(_close) and s.count(_close) > s.count(_open):
            s = s[:-1].rstrip()
        while s.startswith(_open) and s.count(_open) > s.count(_close):
            s = s[1:].lstrip()
    return s


def extract_formula_candidates(text: str) -> list[str]:
    """Split *text* on commas/semicolons and collect RHS substrings for formula matching."""
    candidates: list[str] = []
    separators = re.compile(r"[,;，；]")
    parts = separators.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidates.append(part)
        cleaned = strip_unbalanced_parens(part)
        if cleaned != part:
            candidates.append(cleaned)
        if "=" in part:
            rhs = part.rsplit("=", 1)[1].strip()
            if rhs:
                rhs_clean = strip_unbalanced_parens(rhs)
                candidates.append(rhs)
                if rhs_clean != rhs:
                    candidates.append(rhs_clean)
    return candidates


def compare_text_against_formula_or_equation_gt(
    pred_text: str,
    gt_norm: float | str,
) -> bool:
    """Try symbolic/quantity matching of formula candidates extracted from text."""
    gt_norm_str = str(gt_norm)
    eq_m = _EQ_PATTERN.match(gt_norm_str)
    if eq_m:
        gt_norm_str = eq_m.group(2).strip()

    gt_norms = expand_gt_set(gt_norm_str)
    candidates = extract_formula_candidates(pred_text)
    for cand_str in candidates:
        cand_str = cand_str.strip()
        if not cand_str:
            continue
        try:
            cand_cat, cand_norm = normalize_answer(cand_str)
        except (ValueError, TypeError, RuntimeError):
            continue
        if cand_cat == AnswerCategory.EQUATION:
            cand_norm, cand_cat = extract_rhs_and_category(cand_norm, cand_cat)
        if cand_cat == AnswerCategory.TEXT:
            continue
        for gn in gt_norms:
            try:
                if compare_formula(cand_norm, gn):
                    return True
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass
            if cand_cat == AnswerCategory.PHYSICAL_QUANTITY:
                try:
                    if compare_physical_quantity(cand_norm, gn):
                        return True
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass
    return False
