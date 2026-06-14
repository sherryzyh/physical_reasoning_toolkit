"""
Comparator: quick typed match from :class:`TypedLLMComparator` when it returns a
definite bool; otherwise word-level ROUGE-L F1 on plain text (no LLM).
"""

from __future__ import annotations

from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain.answer import Answer
from prkit.evaluation.similarities import rouge_l_f1

from .base import BaseComparator
from .typed_llm import TypedLLMComparator, _typed_category_and_value


def _text_for_rouge(answer: str | Answer) -> str:
    """Return the plain text representation of *answer* used as ROUGE-L input."""
    if isinstance(answer, Answer):
        return str(answer.value).strip()
    return str(answer).strip()


class SimilarityMatchComparator(BaseComparator):
    """
    If :meth:`TypedLLMComparator._quick_typed_match` returns a definite ``bool``,
    :meth:`compare` / :meth:`accuracy_score` use it. Otherwise similarity is
    word-level ROUGE-L F1 between the two answers as plain text.

    :meth:`compare` returns ``True`` if ROUGE-L ≥ ``rouge_threshold``;
    :meth:`accuracy_score` returns the ROUGE-L value in ``[0, 1]`` on that path.

    Optional ``kwargs``: ``question`` and ``symbolic_answer_is_expression`` are
    forwarded to :meth:`TypedLLMComparator._quick_typed_match` only.
    """

    def __init__(
        self,
        *,
        rouge_threshold: float = 0.5,
    ) -> None:
        self.logger = PRKitLogger.get_logger(__name__)
        self._rouge_threshold = rouge_threshold
        self._last_rouge_score: float | None = None

    @property
    def last_rouge_score(self) -> float | None:
        """ROUGE-L F1 from the last comparison that used the ROUGE fallback, if any."""
        return self._last_rouge_score

    def compare(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> bool:
        self._last_rouge_score = None
        quick = TypedLLMComparator._quick_typed_match(
            answer1,
            answer2,
            question=kwargs.get("question"),
            symbolic_answer_is_expression=kwargs.get("symbolic_answer_is_expression"),
        )
        if quick is not None:
            return bool(quick)

        pred_text = _text_for_rouge(answer1)
        gt_text = _text_for_rouge(answer2)
        score = rouge_l_f1(pred_text, gt_text)
        self._last_rouge_score = score
        pred_cat, _ = _typed_category_and_value(answer1)
        gt_cat, _ = _typed_category_and_value(answer2)
        self.logger.debug(
            "ROUGE-L fallback F1=%.4f (threshold=%.4f) pred_cat=%s gt_cat=%s",
            score,
            self._rouge_threshold,
            pred_cat,
            gt_cat,
        )
        return score >= self._rouge_threshold

    def accuracy_score(
        self,
        answer1: str | Answer,
        answer2: str | Answer,
        **kwargs: Any,
    ) -> float:
        self._last_rouge_score = None
        quick = TypedLLMComparator._quick_typed_match(
            answer1,
            answer2,
            question=kwargs.get("question"),
            symbolic_answer_is_expression=kwargs.get("symbolic_answer_is_expression"),
        )
        note = (
            "falling back to rouge-l"
            if quick is None
            else "returning typed match score"
        )
        self.logger.debug("quick=%s, %s", quick, note)
        if quick is not None:
            return 1.0 if quick else 0.0

        pred_text = _text_for_rouge(answer1)
        gt_text = _text_for_rouge(answer2)
        score = rouge_l_f1(pred_text, gt_text)
        self._last_rouge_score = score
        return score
