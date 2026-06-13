"""
Record-aware comparator for canonical typed final-answer records.

This comparator accepts canonical-answer records (mapping-like objects or
objects exposing the same attributes), maps them into
``prkit.core.domain.answer.Answer``, and then delegates to the
deterministic SmartMatch pipeline. This mirrors SmartLLM's local typed /
cross-type behavior without any LLM fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union

from prkit.core.domain.answer import Answer
from prkit.core.domain.answer_category import AnswerCategory

from .base import BaseComparator
from .smart_match import SmartMatchComparator

RecordLike = object

_RECORD_TYPE_TO_CATEGORY = {
    "number": AnswerCategory.NUMBER,
    "physical_quantity": AnswerCategory.PHYSICAL_QUANTITY,
    "formula": AnswerCategory.FORMULA,
    "equation": AnswerCategory.EQUATION,
    "short_text": AnswerCategory.TEXT,
    "option": AnswerCategory.OPTION,
    "invalid": AnswerCategory.TEXT,
}


def _is_record_like(answer: object) -> bool:
    if isinstance(answer, Mapping):
        return any(
            key in answer for key in ("status", "answer_type", "final_answer")
        )
    return any(hasattr(answer, key) for key in ("status", "answer_type", "final_answer"))


def _record_field(record: RecordLike, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _record_string(record: RecordLike, field_name: str) -> Optional[str]:
    value = _record_field(record, field_name)
    if value is None:
        return None
    return str(value).strip()


def _is_usable_record(record: RecordLike) -> bool:
    status = (_record_string(record, "status") or "").lower()
    answer_type = (_record_string(record, "answer_type") or "").lower()
    return status == "ok" and answer_type != "invalid"


def _record_to_answer(record: RecordLike) -> Answer:
    answer_type = (_record_string(record, "answer_type") or "short_text").lower()
    category = _RECORD_TYPE_TO_CATEGORY.get(answer_type, AnswerCategory.TEXT)

    final_answer = _record_string(record, "final_answer") or ""
    final_answer_latex = _record_string(record, "final_answer_latex")
    value = _record_string(record, "value")
    unit = _record_string(record, "unit")

    if category in (AnswerCategory.FORMULA, AnswerCategory.EQUATION):
        return Answer(
            value=final_answer_latex or final_answer,
            answer_category=category,
        )

    if category == AnswerCategory.NUMBER:
        return Answer(
            value=value or final_answer,
            answer_category=category,
        )

    if category == AnswerCategory.PHYSICAL_QUANTITY:
        quantity_value = value or final_answer
        if quantity_value and unit:
            quantity_value = f"{quantity_value} {unit}"
        return Answer(
            value=quantity_value,
            answer_category=category,
            unit=unit,
        )

    return Answer(
        value=final_answer if category != AnswerCategory.OPTION else (final_answer or (_record_string(record, "option_label") or "")),
        answer_category=category,
    )


class RecordMatchComparator(BaseComparator):
    """
    Comparator for typed final-answer records.

    Records are compared only when both sides are usable final answers
    (``status == "ok"`` and ``answer_type != "invalid"``). The mapped
    ``Answer`` objects are then delegated to :class:`SmartMatchComparator`,
    which runs same-type comparison, equation-RHS extraction, equation-from-text
    rescue, and deterministic cross-type matching, but never calls an LLM.
    """

    def __init__(self) -> None:
        self._delegate = SmartMatchComparator()

    def _coerce_answer(self, answer: Union[str, Answer, RecordLike]) -> Union[str, Answer]:
        if isinstance(answer, (str, Answer)):
            return answer
        if _is_record_like(answer):
            return _record_to_answer(answer)
        raise TypeError(
            "RecordMatchComparator expects an Answer, string, or record-like object "
            "with status/answer_type/final_answer fields."
        )

    def compare(
        self,
        answer1: Union[str, Answer, RecordLike],
        answer2: Union[str, Answer, RecordLike],
        **kwargs: Any,
    ) -> bool:
        if _is_record_like(answer1) and not _is_usable_record(answer1):
            return False
        if _is_record_like(answer2) and not _is_usable_record(answer2):
            return False

        pred = self._coerce_answer(answer1)
        gt = self._coerce_answer(answer2)
        _ = kwargs
        return self._delegate.compare(pred, gt)

    def accuracy_score(
        self,
        answer1: Union[str, Answer, RecordLike],
        answer2: Union[str, Answer, RecordLike],
        **kwargs: Any,
    ) -> float:
        return 1.0 if self.compare(answer1, answer2, **kwargs) else 0.0
