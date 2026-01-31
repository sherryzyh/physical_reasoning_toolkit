"""
Tests for TheoremDetector.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_annotation.annotations.theorem import TheoremAnnotation
from prkit.prkit_annotation.workers.theorem_detector import (
    TheoremDetector,
    TheoremDetail,
    TheoremResponse,
)


class TestTheoremDetector:
    """Test cases for TheoremDetector."""

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_theorem_detector_initialization(self, mock_llm_class):
        """Test theorem detector initialization."""
        mock_client = Mock()
        mock_llm_class.from_model.return_value = mock_client

        detector = TheoremDetector(model="gpt-4o")

        assert detector.model == "gpt-4o"
        assert detector.llm_client == mock_client

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_theorem_detector_work_success(self, mock_llm_class):
        """Test theorem detection with successful structured response."""
        mock_client = Mock()
        theorem_detail = TheoremDetail(
            name="Newton's Second Law",
            description="F = ma",
            equations=["F = ma"],
            domain="mechanics",
            conditions=["constant mass"],
        )
        mock_response = TheoremResponse(theorems=[theorem_detail])
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        detector = TheoremDetector(model="gpt-4o")
        result = detector.work("What is F=ma?")

        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 1
        assert result.theorems[0]["name"] == "Newton's Second Law"
        assert result.theorems[0]["equations"] == ["F = ma"]

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_theorem_detector_work_fallback(self, mock_llm_class):
        """Test theorem detection falls back to regular LLM call."""
        mock_client = Mock()
        # Structured call fails
        mock_client.chat_structured.side_effect = Exception("Structured failed")
        # Regular call succeeds
        fallback_response = json.dumps(
            {
                "theorems": [
                    {
                        "name": "Newton's Second Law",
                        "description": "F = ma",
                        "equations": ["F = ma"],
                        "domain": "mechanics",
                    }
                ]
            }
        )
        mock_client.chat.return_value = fallback_response
        mock_llm_class.from_model.return_value = mock_client

        detector = TheoremDetector(model="gpt-4o")
        result = detector.work("What is F=ma?")

        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) >= 1

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_theorem_detector_work_both_fail(self, mock_llm_class):
        """Test theorem detection when both methods fail."""
        mock_client = Mock()
        mock_client.chat_structured.side_effect = Exception("Structured failed")
        mock_client.chat.side_effect = Exception("Regular failed")
        mock_llm_class.from_model.return_value = mock_client

        detector = TheoremDetector(model="gpt-4o")
        result = detector.work("What is F=ma?")

        # Should return empty annotation
        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 0

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_theorem_detector_work_invalid_json(self, mock_llm_class):
        """Test theorem detection with invalid JSON in fallback."""
        mock_client = Mock()
        mock_client.chat_structured.return_value = None
        mock_client.chat.return_value = "Invalid JSON {"
        mock_llm_class.from_model.return_value = mock_client

        detector = TheoremDetector(model="gpt-4o")
        result = detector.work("What is F=ma?")

        # Should return empty annotation on JSON parse error
        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 0
