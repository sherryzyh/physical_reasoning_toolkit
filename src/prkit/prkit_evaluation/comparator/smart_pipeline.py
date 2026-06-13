"""
Shared deterministic pipeline for SmartMatch-style comparison.

Same-type comparison, equation RHS extraction, equation-from-text rescue, then
:class:`~prkit.prkit_evaluation.comparator.smart_match.SmartMatchComparator`
cross-type matching. Used by :class:`~prkit.prkit_evaluation.comparator.smart_match.SmartMatchComparator`
and :class:`~prkit.prkit_evaluation.comparator.smart_llm.SmartLLMComparator`.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Protocol, Union

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category
from prkit.prkit_evaluation.utils.category_dispatch import (
    SameCategoryCompareFn,
    compare_by_category,
)
from prkit.prkit_evaluation.utils.compare_cross_type import extract_rhs_and_category
from prkit.prkit_evaluation.utils.normalization import normalize_answer

SmartPipelineResult = Literal["match", "no_match", "inconclusive"]


class SmartMatchPipelineHost(Protocol):
    """Minimum interface required to run :func:`run_smart_pipeline`."""

    _comparators: Mapping[AnswerCategory, SameCategoryCompareFn]
    logger: object

    def _try_equation_from_text(
        self,
        text_raw: str,
        gt_norm: Union[float, str],
        gt_raw: str,
    ) -> bool: ...

    def _cross_type_match(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
    ) -> Optional[bool]: ...


def run_smart_pipeline(
    comparator: SmartMatchPipelineHost,
    answer1: Union[str, Answer],
    answer2: Union[str, Answer],
) -> SmartPipelineResult:
    """Same-type, RHS, equation-from-text rescue, then cross-type.

    Returns ``match`` / ``no_match`` / ``inconclusive`` (cross-type unresolved).

    Implemented against :class:`~prkit.prkit_evaluation.comparator.smart_match.SmartMatchComparator`
    internals (protected members); callers should pass that comparator or a compatible host.
    """
    # pylint: disable=protected-access
    if isinstance(answer1, Answer):
        pred_norm = str(answer1.value)
        pred_cat = answer1.answer_category
    else:
        pred_cat, pred_norm = normalize_answer(answer1)
    if isinstance(answer2, Answer):
        gt_norm = str(answer2.value)
        gt_cat = answer2.answer_category
    else:
        gt_cat, gt_norm = normalize_answer(answer2)

    # 1. Same-type comparison
    if same_comparison_category(gt_cat, pred_cat):
        if compare_by_category(
            gt_cat, pred_norm, gt_norm, comparator._comparators, comparator.logger
        ):
            return "match"
        # Non-EQUATION same-type failures are final.
        if gt_cat != AnswerCategory.EQUATION:
            return "no_match"
        # EQUATION same-type failure: the normalised pred may be garbage
        # (e.g. preamble text contaminating the SymPy parse).  Fall through
        # to RHS extraction and equation-from-text rescue below.

    # 2. Extract RHS from equation-like answers, then retry same-type
    pred_rhs_norm, pred_rhs_cat = extract_rhs_and_category(pred_norm, pred_cat)
    gt_rhs_norm, gt_rhs_cat = extract_rhs_and_category(gt_norm, gt_cat)
    if same_comparison_category(gt_rhs_cat, pred_rhs_cat):
        comparator.logger.debug(
            "Same RHS category - answer: %s and model answer: %s",
            gt_rhs_norm,
            pred_rhs_norm,
        )
        if compare_by_category(
            gt_rhs_cat,
            pred_rhs_norm,
            gt_rhs_norm,
            comparator._comparators,
            comparator.logger,
        ):
            return "match"

    # 2b. Same-type EQUATION rescue: extract embedded LaTeX equations
    # from the raw prediction string and compare against GT.
    if gt_cat == AnswerCategory.EQUATION and pred_cat == AnswerCategory.EQUATION:
        pred_raw = str(answer1).strip()
        gt_raw = str(answer2).strip()
        if comparator._try_equation_from_text(pred_raw, gt_norm, gt_raw):
            return "match"

    # 3. Cross-type matching
    cross = comparator._cross_type_match(answer1, answer2)
    if cross is True:
        return "match"
    if cross is False:
        return "no_match"
    return "inconclusive"
