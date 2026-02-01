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

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_theorem_detector_initialization(self, mock_create):
        """Test theorem detector initialization."""
        mock_client = Mock()
        mock_create.return_value = mock_client

        detector = TheoremDetector(model="gpt-5.1")

        assert detector.model == "gpt-5.1"
        assert detector.llm_client == mock_client

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_theorem_detector_work_success(self, mock_create):
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
        # _call_llm_structured now parses JSON and returns Pydantic model
        import json
        mock_client.chat.return_value = json.dumps(mock_response.model_dump())
        mock_create.return_value = mock_client

        detector = TheoremDetector(model="gpt-5.1")
        result = detector.work("What is F=ma?")

        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 1
        assert result.theorems[0]["name"] == "Newton's Second Law"
        assert result.theorems[0]["equations"] == ["F = ma"]

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_theorem_detector_work_fallback(self, mock_create):
        """Test theorem detection falls back to regular LLM call."""
        mock_client = Mock()
        # Structured call fails (now just chat with combined prompt)
        mock_client.chat.side_effect = [
            Exception("Structured failed"),  # First call fails
            json.dumps(  # Fallback call succeeds
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
            ),
        ]
        mock_create.return_value = mock_client

        detector = TheoremDetector(model="gpt-5.1")
        result = detector.work("What is F=ma?")

        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) >= 1

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_theorem_detector_work_both_fail(self, mock_create):
        """Test theorem detection when both methods fail."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("All calls failed")
        mock_create.return_value = mock_client

        detector = TheoremDetector(model="gpt-5.1")
        result = detector.work("What is F=ma?")

        # Should return empty annotation
        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 0

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_theorem_detector_work_invalid_json(self, mock_create):
        """Test theorem detection with invalid JSON in fallback."""
        mock_client = Mock()
        mock_client.chat.side_effect = [
            None,  # First call returns None (treated as failure)
            "Invalid JSON {",  # Fallback call returns invalid JSON
        ]
        mock_create.return_value = mock_client

        detector = TheoremDetector(model="gpt-5.1")
        result = detector.work("What is F=ma?")

        # Should return empty annotation on JSON parse error
        assert isinstance(result, TheoremAnnotation)
        assert len(result.theorems) == 0
