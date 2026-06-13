"""
Smart-Match Comparator for answer comparison.

``compare`` / ``accuracy_score`` first attempt same-type comparison, then
try equation RHS extraction to reduce equations to their value types, and
finally attempt deterministic cross-type matching for pairs that neither
path could resolve.

A match is ``True`` when any of the three paths resolves to ``True``.
If none resolves the pair, this comparator returns ``False`` / score ``0.0``.
"""

import re
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain.answer import Answer
from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.answer_utils import same_comparison_category
from prkit.evaluation.utils.category_dispatch import (
    SameCategoryCompareFn,
    compare_by_category,
)
from prkit.evaluation.utils.compare_cross_type import (
    compare_text_against_formula_or_equation_gt,
    extract_rhs_and_category,
)
from prkit.evaluation.utils.compare_same_type import (
    compare_formula,
    compare_number,
    compare_option,
    compare_physical_quantity,
    compare_plain_text,
    parse_physical_quantity,
)
from prkit.evaluation.utils.normalization import normalize_answer, normalize_text

from .base import BaseComparator
from .smart_pipeline import run_smart_pipeline

# Matches LaTeX-delimited math expressions in free text.
# Alternation order matters: $$...$$ must precede $...$ to avoid partial matches.
_LATEX_DELIMITED_RE = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)",
    re.DOTALL,
)


def _extract_latex_equations(text: str) -> list[str]:
    """Extract LaTeX-delimited math expressions from free text.

    Returns the full matched strings (including delimiters) so that
    ``normalize_answer`` receives proper LaTeX context.
    """
    return [m.group(0) for m in _LATEX_DELIMITED_RE.finditer(text)]


def _typed_category_and_value(
    answer: str | Answer,
) -> tuple[AnswerCategory | None, float | str]:
    """Get typed category and normalized value for an answer."""
    if isinstance(answer, Answer):
        return answer.answer_category, str(answer.value)
    try:
        category, normalized = normalize_answer(answer)
        return category, normalized
    except (ValueError, TypeError, RuntimeError):
        return None, str(answer).strip()


class SmartMatchComparator(BaseComparator):
    """
    Comparator that combines same-type comparison, equation RHS extraction,
    and deterministic cross-type matching.

    ``answer1`` is treated as the model prediction and ``answer2`` as ground
    truth.
    """

    DEFAULT_COMPARATORS: dict[AnswerCategory, SameCategoryCompareFn] = {
        AnswerCategory.NUMBER: compare_number,
        AnswerCategory.PHYSICAL_QUANTITY: compare_physical_quantity,
        AnswerCategory.FORMULA: compare_formula,
        AnswerCategory.TEXT: compare_plain_text,
        AnswerCategory.OPTION: compare_option,
    }

    def __init__(self) -> None:
        """Initialize with default category comparators."""
        self._comparators = dict(self.DEFAULT_COMPARATORS)
        self.logger = PRKitLogger.get_logger(__name__)

    # ------------------------------------------------------------------
    # Cross-type matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_equation_rhs_raw(raw_text: str) -> str | None:
        """Extract RHS from a raw equation string (e.g. ``$T_B = 355\\,K$``).

        For multi-line text only the first line is considered, so subsidiary
        definitions (``where omega^2 = ...``) do not contaminate the RHS.
        """
        s = raw_text.strip()
        for opening, closing in [("$$", "$$"), ("\\(", "\\)"), ("\\[", "\\]")]:
            if s.startswith(opening) and s.endswith(closing):
                s = s[len(opening) : -len(closing)].strip()
                break
        s = s.strip("$").strip()

        if "\n" in s:
            s = s.split("\n", 1)[0].strip()

        if "=" not in s:
            return None
        rhs = s.rsplit("=", 1)[1].strip()
        return rhs if rhs else None

    @staticmethod
    def _compare_numeric_with_renormalized(
        numeric_cat: AnswerCategory,
        numeric_value: float | str,
        rhs_raw: str,
    ) -> bool | None:
        """Compare a NUMBER or PHYSICAL_QUANTITY value against a re-normalized
        raw RHS string extracted from an equation.

        When *numeric_cat* is NUMBER, matching against a PQ-typed RHS is
        intentionally skipped: a bare number missing its unit should not be
        accepted when the ground truth carries units.
        """
        try:
            rhs_cat, rhs_norm = normalize_answer(rhs_raw)
        except (ValueError, TypeError, RuntimeError):
            return None

        if numeric_cat == AnswerCategory.PHYSICAL_QUANTITY:
            if rhs_cat == AnswerCategory.PHYSICAL_QUANTITY:
                try:
                    return compare_physical_quantity(str(numeric_value), str(rhs_norm))
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass
            elif rhs_cat == AnswerCategory.NUMBER:
                try:
                    num, _, num_str = parse_physical_quantity(str(numeric_value))
                    if num is not None:
                        return compare_number(num_str, str(rhs_norm))
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass
        elif numeric_cat == AnswerCategory.NUMBER:
            if rhs_cat == AnswerCategory.NUMBER:
                try:
                    return compare_number(str(numeric_value), str(rhs_norm))
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass
            # NUMBER vs PQ intentionally omitted: missing unit = wrong.
        return None

    @staticmethod
    def _compare_formula_with_renormalized(
        formula_value: float | str,
        rhs_raw: str,
    ) -> bool | None:
        """Compare a FORMULA value against a re-normalized raw RHS string
        extracted from an equation on the other side."""
        try:
            rhs_cat, rhs_norm = normalize_answer(rhs_raw)
        except (ValueError, TypeError, RuntimeError):
            return None

        if rhs_cat not in (
            AnswerCategory.FORMULA,
            AnswerCategory.EQUATION,
            AnswerCategory.NUMBER,
        ):
            return None

        try:
            if compare_formula(str(formula_value), str(rhs_norm)):
                return True
        except (ValueError, TypeError, ZeroDivisionError, AttributeError):
            pass
        try:
            if compare_plain_text(str(formula_value), str(rhs_norm)):
                return True
        except (ValueError, TypeError):
            pass
        return None

    # ------------------------------------------------------------------
    # Equation-from-text extraction
    # ------------------------------------------------------------------

    def _try_equation_from_text(
        self,
        text_raw: str,
        gt_norm: float | str,
        gt_raw: str,
    ) -> bool:
        """Try matching by extracting an embedded LaTeX equation from text.

        1. Extract LaTeX-delimited equations from *text_raw*.
        2. Normalize each and compare against *gt_norm* (direct, RHS, formula).
        3. If nothing matched, check whether *gt_norm* or *gt_raw* appears as a
           substring of the prediction text.
        """
        equations = _extract_latex_equations(text_raw)
        for eq_str in equations:
            try:
                eq_cat, eq_norm = normalize_answer(eq_str)
            except (ValueError, TypeError, RuntimeError):
                continue
            # Direct same-type comparison
            if same_comparison_category(eq_cat, AnswerCategory.EQUATION):
                if compare_by_category(
                    AnswerCategory.EQUATION,
                    eq_norm,
                    gt_norm,
                    self._comparators,
                    self.logger,
                ):
                    return True
            # RHS extraction and comparison
            eq_rhs, eq_rhs_cat = extract_rhs_and_category(eq_norm, eq_cat)
            gt_rhs, gt_rhs_cat = extract_rhs_and_category(
                gt_norm, AnswerCategory.EQUATION
            )
            if same_comparison_category(eq_rhs_cat, gt_rhs_cat):
                if compare_by_category(
                    eq_rhs_cat, eq_rhs, gt_rhs, self._comparators, self.logger
                ):
                    return True
            # Formula-level comparison
            try:
                if compare_formula(str(eq_norm), str(gt_norm)):
                    return True
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass

        # Substring fallback: check if gt_norm or gt_raw appears in pred text
        pred_text = normalize_text(text_raw)
        gt_norm_text = normalize_text(str(gt_norm))
        if gt_norm_text and gt_norm_text in pred_text:
            return True
        gt_raw_text = normalize_text(gt_raw)
        if gt_raw_text and gt_raw_text in pred_text:
            return True
        return False

    # ------------------------------------------------------------------
    # Main cross-type dispatch
    # ------------------------------------------------------------------

    def _cross_type_match(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
    ) -> bool | None:
        """Positive-only cross-type matching for pairs that same-type
        comparison could not resolve.

        Returns ``True`` when a confident deterministic match is found,
        ``None`` otherwise.  Never returns ``False``, so callers can safely
        fall back.
        """
        pred_cat, pred_value = _typed_category_and_value(answer1)
        gt_cat, gt_value = _typed_category_and_value(answer2)

        if pred_cat is None or gt_cat is None:
            return None

        # --- PHYSICAL_QUANTITY (pred) vs NUMBER (gt) ---
        # Model gives value+unit while GT is just a number -> model has more
        # information; compare the numeric parts.
        # The reverse direction (NUMBER pred vs PQ gt) is intentionally
        # excluded: a bare number missing its unit is wrong for physics.
        if (
            pred_cat == AnswerCategory.PHYSICAL_QUANTITY
            and gt_cat == AnswerCategory.NUMBER
        ):
            try:
                pred_num, _, pred_num_str = parse_physical_quantity(str(pred_value))
                if pred_num is not None and compare_number(pred_num_str, gt_value):
                    return True
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass
        # --- NUMBER (pred) vs PHYSICAL_QUANTITY (gt) ---
        if (
            pred_cat == AnswerCategory.NUMBER
            and gt_cat == AnswerCategory.PHYSICAL_QUANTITY
        ):
            try:
                gt_num, _, gt_num_str = parse_physical_quantity(str(gt_value))
                if gt_num is None or not compare_number(pred_value, gt_num_str):
                    return False
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass

        # --- TEXT(pred) vs FORMULA / EQUATION (gt) ---
        if pred_cat == AnswerCategory.TEXT and gt_cat in (
            AnswerCategory.FORMULA,
            AnswerCategory.EQUATION,
        ):
            pred_str = str(pred_value)
            gt_str = str(gt_value)
            try:
                if compare_text_against_formula_or_equation_gt(pred_str, gt_value):
                    return True
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass
            try:
                if compare_plain_text(pred_str, gt_str):
                    return True
            except (ValueError, TypeError):
                pass
            try:
                if compare_formula(pred_str, gt_str):
                    return True
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                pass

        # --- EQUATION (GT) vs TEXT (pred) ---
        if gt_cat == AnswerCategory.EQUATION and pred_cat == AnswerCategory.TEXT:
            pred_raw = str(answer1).strip()
            gt_raw = str(answer2).strip()
            if self._try_equation_from_text(pred_raw, gt_value, gt_raw):
                return True

        # --- EQUATION (GT) vs PHYSICAL_QUANTITY/NUMBER (pred) ---
        if gt_cat == AnswerCategory.EQUATION and pred_cat in (
            AnswerCategory.PHYSICAL_QUANTITY,
            AnswerCategory.NUMBER,
        ):
            gt_raw = str(answer2).strip()
            rhs = self._extract_equation_rhs_raw(gt_raw)
            if rhs is not None:
                match = self._compare_numeric_with_renormalized(
                    pred_cat, pred_value, rhs
                )
                if match is True:
                    return True

        # --- EQUATION (GT) vs FORMULA (pred) ---
        if gt_cat == AnswerCategory.EQUATION and pred_cat == AnswerCategory.FORMULA:
            gt_raw = str(answer2).strip()
            rhs = self._extract_equation_rhs_raw(gt_raw)
            if rhs is not None:
                match = self._compare_formula_with_renormalized(pred_value, rhs)
                if match is True:
                    return True

        # --- EQUATION (pred) vs PHYSICAL_QUANTITY/NUMBER (gt) ---
        if pred_cat == AnswerCategory.EQUATION and gt_cat in (
            AnswerCategory.PHYSICAL_QUANTITY,
            AnswerCategory.NUMBER,
        ):
            pred_raw = str(answer1).strip()
            rhs = self._extract_equation_rhs_raw(pred_raw)
            if rhs is not None:
                match = self._compare_numeric_with_renormalized(gt_cat, gt_value, rhs)
                if match is True:
                    return True

        # --- EQUATION (pred) vs FORMULA (gt) ---
        if pred_cat == AnswerCategory.EQUATION and gt_cat == AnswerCategory.FORMULA:
            pred_raw = str(answer1).strip()
            rhs = self._extract_equation_rhs_raw(pred_raw)
            if rhs is not None:
                match = self._compare_formula_with_renormalized(gt_value, rhs)
                if match is True:
                    return True

        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compare(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> bool:
        """
        True when either same-type comparison, RHS-extracted same-type
        comparison, or cross-type matching resolves to True.
        """
        _ = kwargs
        outcome = run_smart_pipeline(self, answer1, answer2)
        if outcome == "match":
            return True
        if outcome == "no_match":
            return False
        # Cross-type inconclusive: pure SmartMatch still returns False.
        return False

    def accuracy_score(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> float:
        """1.0 if :meth:`compare` is True, else 0.0."""
        is_match = self.compare(answer1, answer2, **kwargs)
        return 1.0 if is_match else 0.0
