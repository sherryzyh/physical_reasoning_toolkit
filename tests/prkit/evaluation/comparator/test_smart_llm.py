"""Tests for SmartLLMComparator: deterministic SmartMatch path + LLM fallback metadata."""

from unittest.mock import patch

import pytest

from prkit.evaluation.comparator.smart_llm import SmartLLMComparator
from prkit.evaluation.llm_judge import (
    RESULT_SOURCE_SKIPPED_LLM,
    RESULT_SOURCE_SMART_MATCH,
)


class TestSmartLLMComparator:
    @pytest.fixture(autouse=True)
    def _mock_openai(self):
        with patch("prkit.evaluation.llm_judge.runner.OpenAI"):
            yield

    def test_match_records_smart_match_source(self):
        comp = SmartLLMComparator()
        assert comp.compare("42", "42") is True
        assert comp.last_result is not None
        assert comp.last_result.verdict_type == RESULT_SOURCE_SMART_MATCH
        assert comp.last_result.verdict == "correct"

    def test_no_match_records_smart_match_source(self):
        comp = SmartLLMComparator()
        assert comp.compare("42", "43") is False
        assert comp.last_result is not None
        assert comp.last_result.verdict_type == RESULT_SOURCE_SMART_MATCH
        assert comp.last_result.verdict == "incorrect"

    @pytest.mark.parametrize(
        ("pred", "gt"),
        [
            ("unrelated text", "3.14"),
            ("3.14", "unrelated text"),
        ],
    )
    def test_cross_inconclusive_skip_llm(self, pred, gt):
        """Pairs that reach cross-type with no deterministic verdict defer to LLM."""
        comp = SmartLLMComparator()
        assert comp.compare(pred, gt, skip_llm=True) is False
        assert comp.last_result is not None
        assert comp.last_result.verdict_type == RESULT_SOURCE_SKIPPED_LLM
