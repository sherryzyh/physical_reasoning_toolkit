"""
Typed-path + LLM comparator using OpenAI Responses API.

Quick typed match, LLM judge fallback, payload schema, and overrides.
Reusable judge primitives live in :mod:`prkit.prkit_evaluation.llm_judge`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

from openai import OpenAI
from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.llm_judge import (
    DEFAULT_MODEL,
    LLMJudgeResult,
    OpenAIJudgeRunner,
    RESULT_SOURCE_LLM_JUDGE,
    RESULT_SOURCE_SKIPPED_LLM,
    RESULT_SOURCE_TYPED_MATCH,
    build_standard_answer_judge_payload,
)
from prkit.prkit_evaluation.llm_judge.payload import answer_to_text_and_category
from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category
from prkit.prkit_evaluation.utils.category_dispatch import compare_by_category
from prkit.prkit_evaluation.utils.normalization import normalize_answer
from prkit.prkit_evaluation.utils.compare_same_type import (
    compare_formula,
    compare_number,
    compare_option,
    compare_physical_quantity,
    compare_plain_text,
    parse_physical_quantity,
)
from sympy import Eq, sympify
from sympy.core.relational import Relational
from sympy.core.sympify import SympifyError

from .base import BaseComparator

_OPTION_ONLY_COMPARATORS = {AnswerCategory.OPTION: compare_option}


def _typed_category_and_value(
    answer: Union[str, Answer],
) -> tuple[Optional[AnswerCategory], Union[float, str]]:
    if isinstance(answer, Answer):
        return answer.answer_category, str(answer.value)
    try:
        category, normalized = normalize_answer(answer)
        return category, normalized
    except (ValueError, TypeError, RuntimeError):
        return None, str(answer).strip()


def _contains_latex_text_macro(s: str) -> bool:
    return bool(re.search(r"\\text\s*\{", s))


def infer_symbolic_answer_is_expression(question: Optional[str]) -> Optional[bool]:
    """
    Heuristic: whether the question asks for a *symbolic expression* (scalar /
    formula) vs a full *equation* as the graded object.

    When this returns ``True``, formula vs equation pairs may be compared by RHS
    (see :meth:`TypedLLMComparator._quick_typed_match`). When ``False``, compare
    full equations (no RHS-only shortcut). When ``None``, the comparator does not
    apply that shortcut (falls back to LLM judge).

    Tuned using seephys-style wording (English + Chinese). Callers can override
    with ``symbolic_answer_is_expression=`` in ``compare()``.
    """
    if not question or not question.strip():
        return None
    q_en = question.lower()

    if re.search(r"\bin terms of\b", q_en):
        return True
    if "的表达式" in question:
        return True
    if re.search(
        r"(?:写出|试求|试计算|求|计算)[^。\n;；]{0,55}表达式",
        question,
    ):
        return True

    _equation_false_en = (
        r"\bwrite (?:down )?the (?:time[- ])?wave equation\b",
        r"\bwrite (?:down )?the wave equation\b",
        r"\bwrite (?:down )?the equation\b",
        r"\bderive (?:an |the )?equations?\s+describing\b",
        r"\bderive (?:an |the )?equations?\s+of\s+motion\b",
        r"\bderive (?:an |the )?equations?\b",
        r"\bderive (?:an |the )?equation\s+for\s+the\s+distribution\b",
        r"\bderive (?:an |the )?equation\s+for\s+the\s+(?:phase|intensity)\b",
        r"\bderive an equation for\b",
        r"\bequation-involving\b",
        r"\bequations?\s+describing\b",
        r"\bequations?\s+of\s+motion\b",
        r"\bequation\s+describing\b",
        r"\bequation\s+which\s+describes\b",
        r"\bdifferential equation\b",
        r"\bstate lagrange",
        r"\blagrange'?s\s+equations?\b",
        r"\bhamilton'?s\s+equations?\b",
        r"\bwhat is the equation\b",
        r"\bfind the equation\b",
        r"\bstate the equation\b",
        r"\bshow that the equation\b",
        r"\bequation of motion\b",
        r"\bthe equation (?:of|relating)\b",
        r"\bequation for (?:the\s+)?(?:phase|distribution|field)\b",
    )
    for pat in _equation_false_en:
        if re.search(pat, q_en):
            return False

    _equation_false_cn = (
        r"(?:试求|写出|试分别写出|试写出|推导)[^。\n;；]{0,160}"
        r"(?:轨迹|轨道|迹线|包络线|谐振动|简谐振动|运动学|漏水|抛物线|彗星|椭圆|飞船|"
        r"薛定谔|麦克斯韦|哈密顿|拉格朗日|微分|振动|运动)方程(?!的表达式)"
        r"|(?:试求|写出|试分别写出|试写出|推导)[^。\n;；]{0,80}波动方程(?!的表达式)"
    )
    if re.search(_equation_false_cn, question):
        return False

    _expr_true_en = (
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bhow long\b",
        r"\bhow fast\b",
        r"\bfind the\b",
        r"\bcalculate\b",
        r"\bdetermine\b",
        r"\bobtain\b",
        r"\bcompute\b",
        r"\bgive (?:me )?the\b",
        r"\bevaluate\b",
        r"\bexpress\b",
        r"\bmagnitude of\b",
        r"\bvalue of\b",
        r"\bspeed of\b",
        r"\bvelocity of\b",
        r"\bacceleration of\b",
        r"\bangular velocity of\b",
        r"\bangular frequency of\b",
        r"\bforce on\b",
        r"\bforce between\b",
        r"\bfrequency of\b",
        r"\bperiod of\b",
        r"\bpotential (?:at|of)\b",
        r"\bcurrent through\b",
        r"\bcurrent in\b",
        r"\bresistance of\b",
        r"\bvoltage across\b",
        r"\belectric field (?:at|of)\b",
        r"\bmagnetic field (?:at|of)\b",
        r"\bcharge (?:on|of)\b",
        r"\bcapacitance of\b",
        r"\binductance of\b",
        r"\benergy (?:of|stored)\b",
        r"\bpower (?:dissipated|delivered|of)\b",
        r"\bmaximum\b.*\bof\b",
        r"\bminimum\b.*\bof\b",
    )
    for pat in _expr_true_en:
        if re.search(pat, q_en):
            return True

    if re.search(
        r"(?:^|[\s,，、。；;])(?:试求|试计算|试导出|计算|求|问|多大|多少|何值)",
        question,
    ):
        return True
    if re.search(r"用[^，。；\n]{0,40}表示", question):
        return True

    return None


def _normalized_equation_rhs_string(normalized_value: str) -> Optional[str]:
    try:
        e = sympify(str(normalized_value))
    except (SympifyError, TypeError, ValueError, AttributeError):
        return None
    if isinstance(e, Eq) or (isinstance(e, Relational) and getattr(e, "rhs", None) is not None):
        try:
            return str(e.rhs)
        except (ValueError, TypeError, AttributeError):
            return None
    return None


def _symbolic_operand_for_expression_compare(
    category: AnswerCategory,
    normalized_value: Union[float, str],
) -> Optional[str]:
    if category == AnswerCategory.FORMULA:
        return str(normalized_value)
    if category == AnswerCategory.EQUATION:
        return _normalized_equation_rhs_string(str(normalized_value))
    return None


def _compare_formula_or_equation_as_expressions(
    pred_cat: AnswerCategory,
    pred_value: Union[float, str],
    gt_cat: AnswerCategory,
    gt_value: Union[float, str],
    pred_raw: str,
    gt_raw: str,
) -> Optional[bool]:
    if pred_cat not in (AnswerCategory.FORMULA, AnswerCategory.EQUATION):
        return None
    if gt_cat not in (AnswerCategory.FORMULA, AnswerCategory.EQUATION):
        return None
    if _contains_latex_text_macro(pred_raw) or _contains_latex_text_macro(gt_raw):
        return None
    pe = _symbolic_operand_for_expression_compare(pred_cat, pred_value)
    ge = _symbolic_operand_for_expression_compare(gt_cat, gt_value)
    if pe is None or ge is None:
        return None
    try:
        return compare_formula(pe, ge)
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        return None


def _plain_text_true_else_llm(
    predicted_norm: Union[float, str],
    ground_truth_norm: Union[float, str],
) -> Optional[bool]:
    try:
        return True if compare_plain_text(predicted_norm, ground_truth_norm) else None
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        return None


def _compare_physical_quantity_same_unit_pool_placeholder(
    pred_num_str: str,
    pred_unit: str,
    gt_num_str: str,
    gt_unit: str,
) -> tuple[bool, bool]:
    _ = (pred_num_str, pred_unit, gt_num_str, gt_unit)
    return False, False


class TypedLLMComparator(BaseComparator):
    """OpenAI Responses API comparator with question-aware judging."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        instructions: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.logger = PRKitLogger.get_logger(__name__)
        self._runner = OpenAIJudgeRunner(
            model=model,
            instructions=instructions,
            client=client,
            logger=self.logger,
        )
        self._last_result: Optional[LLMJudgeResult] = None

    def _judge(self, payload: Dict[str, Any]) -> LLMJudgeResult:
        return self._runner.judge(payload)

    @staticmethod
    def _quick_typed_match(
        predicted: Union[str, Answer],
        ground_truth: Union[str, Answer],
        *,
        question: Optional[str] = None,
        symbolic_answer_is_expression: Optional[bool] = None,
    ) -> Optional[bool]:
        pred_raw, _ = answer_to_text_and_category(predicted)
        gt_raw, _ = answer_to_text_and_category(ground_truth)
        pred_cat, pred_value = _typed_category_and_value(predicted)
        gt_cat, gt_value = _typed_category_and_value(ground_truth)
        if pred_cat is None or gt_cat is None:
            return None

        if same_comparison_category(gt_cat, pred_cat):
            if pred_cat == AnswerCategory.OPTION:
                return compare_by_category(
                    pred_cat,
                    pred_value,
                    gt_value,
                    _OPTION_ONLY_COMPARATORS,
                    None,
                )

            if pred_cat == AnswerCategory.NUMBER:
                try:
                    return compare_number(pred_value, gt_value)
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    return None

            if pred_cat == AnswerCategory.PHYSICAL_QUANTITY:
                try:
                    _pred_num, pred_unit, pred_num_str = parse_physical_quantity(str(pred_value))
                    _gt_num, gt_unit, gt_num_str = parse_physical_quantity(str(gt_value))
                    if pred_unit == gt_unit:
                        return compare_physical_quantity(pred_value, gt_value)
                    is_same_unit_pool, is_equal_value = _compare_physical_quantity_same_unit_pool_placeholder(
                        pred_num_str, pred_unit, gt_num_str, gt_unit
                    )
                    if is_same_unit_pool and is_equal_value:
                        return True
                    return _plain_text_true_else_llm(str(pred_value), str(gt_value))

                except (ValueError, TypeError, ZeroDivisionError, RuntimeError, AttributeError):
                    return None

            if pred_cat == AnswerCategory.FORMULA:
                if _contains_latex_text_macro(pred_raw) or _contains_latex_text_macro(gt_raw):
                    return _plain_text_true_else_llm(pred_value, gt_value)
                try:
                    if compare_formula(pred_value, gt_value):
                        return True
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass
                return _plain_text_true_else_llm(pred_value, gt_value)

            if pred_cat == AnswerCategory.EQUATION:
                pred_rhs = _normalized_equation_rhs_string(str(pred_value))
                gt_rhs = _normalized_equation_rhs_string(str(gt_value))
                if _contains_latex_text_macro(pred_raw) or _contains_latex_text_macro(gt_raw):
                    return _plain_text_true_else_llm(pred_value, gt_value)
                if pred_rhs is not None and gt_rhs is not None:
                    try:
                        if compare_formula(pred_rhs, gt_rhs):
                            return True
                    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                        pass
                return _plain_text_true_else_llm(pred_value, gt_value)

            if pred_cat == AnswerCategory.TEXT:
                return _plain_text_true_else_llm(pred_value, gt_value)

            return None

        formula_equation_pair = (
            pred_cat in (AnswerCategory.FORMULA, AnswerCategory.EQUATION)
            and gt_cat in (AnswerCategory.FORMULA, AnswerCategory.EQUATION)
        )
        if formula_equation_pair:
            resolved_expr: Optional[bool] = symbolic_answer_is_expression
            if resolved_expr is None and question:
                resolved_expr = infer_symbolic_answer_is_expression(question)
            if resolved_expr is True:
                qm = _compare_formula_or_equation_as_expressions(
                    pred_cat, pred_value, gt_cat, gt_value, pred_raw, gt_raw
                )
                if qm is True:
                    return True
            return _plain_text_true_else_llm(pred_value, gt_value)

        return None

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        *,
        skip_llm: bool = False,
        **kwargs: Any,
    ) -> bool:
        quick = self._quick_typed_match(
            answer1,
            answer2,
            question=kwargs.get("question"),
            symbolic_answer_is_expression=kwargs.get("symbolic_answer_is_expression"),
        )
        if quick is not None:
            self._last_result = LLMJudgeResult(
                verdict="correct" if quick else "incorrect",
                confidence=1.0,
                expected_answer_type="other",
                reasoning="Quick typed match path.",
                raw_response="local_shortcut",
                verdict_type=RESULT_SOURCE_TYPED_MATCH,
            )
            return quick

        if skip_llm:
            self._last_result = LLMJudgeResult(
                verdict="incorrect",
                confidence=0.0,
                expected_answer_type="other",
                reasoning="LLM judge skipped (skip_llm=True); no API call was made.",
                raw_response="skipped_llm",
                verdict_type=RESULT_SOURCE_SKIPPED_LLM,
            )
            return False

        payload = build_standard_answer_judge_payload(
            answer1,
            answer2,
            kwargs.get("question"),
        )
        result = self._judge(payload)
        self._last_result = result
        return result.verdict == "correct"

    def accuracy_score(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        *,
        skip_llm: bool = False,
        **kwargs: Any,
    ) -> float:
        is_correct = self.compare(answer1, answer2, skip_llm=skip_llm, **kwargs)
        return 1.0 if is_correct else 0.0

    @property
    def last_result(self) -> Optional[LLMJudgeResult]:
        return self._last_result

    @property
    def model_name(self) -> str:
        return self._runner.model_name
