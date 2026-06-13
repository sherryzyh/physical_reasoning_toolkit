"""Tests for SimilarityMatchComparator (quick path, ROUGE-L fallback)."""

import pytest

from prkit.core.domain import Answer, AnswerCategory
from prkit.evaluation.comparator.similarity_match import SimilarityMatchComparator


class TestSimilarityMatchComparator:
    def test_same_type_number_uses_quick_path(self):
        comp = SimilarityMatchComparator()
        a1 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        a2 = Answer(value=42.0, answer_category=AnswerCategory.NUMBER)
        assert comp.compare(a1, a2) is True
        assert comp.accuracy_score(a1, a2) == 1.0
        assert comp.last_rouge_score is None

    def test_cross_type_high_rouge_matches(self):
        comp = SimilarityMatchComparator(rouge_threshold=0.5)
        pred = Answer(value="42", answer_category=AnswerCategory.NUMBER)
        gt = Answer(value="42", answer_category=AnswerCategory.TEXT)
        assert comp.compare(pred, gt) is True
        assert comp.accuracy_score(pred, gt) == pytest.approx(1.0)
        assert comp.last_rouge_score == pytest.approx(1.0)

    def test_cross_type_low_rouge_no_match(self):
        comp = SimilarityMatchComparator(rouge_threshold=0.99)
        pred = Answer(value="1", answer_category=AnswerCategory.NUMBER)
        gt = Answer(
            value="unrelated explanation with many different words",
            answer_category=AnswerCategory.TEXT,
        )
        assert comp.compare(pred, gt) is False
        assert 0.0 <= (comp.last_rouge_score or 0.0) < 0.99

    def test_same_type_text_uses_rouge_when_quick_none(self):
        comp = SimilarityMatchComparator(rouge_threshold=0.5)
        pred = Answer(value="aaa", answer_category=AnswerCategory.TEXT)
        gt = Answer(value="bbb", answer_category=AnswerCategory.TEXT)
        assert comp.compare(pred, gt) is False
        assert 0.0 <= comp.accuracy_score(pred, gt) < 0.5
        assert comp.last_rouge_score is not None
