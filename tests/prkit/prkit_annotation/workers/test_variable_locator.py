"""
Tests for VariableLocator.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_annotation.workers.variable_locator import (
    VariableAnnotation,
    VariableLocator,
    VariableMetadata,
    VariableResponse,
)


class TestVariableLocator:
    """Test cases for VariableLocator."""

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_variable_locator_initialization(self, mock_llm_class):
        """Test variable locator initialization."""
        mock_client = Mock()
        mock_llm_class.from_model.return_value = mock_client

        locator = VariableLocator(model="gpt-4o")

        assert locator.model == "gpt-4o"
        assert locator.llm_client == mock_client

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_variable_locator_work_success(self, mock_llm_class):
        """Test variable extraction with successful structured response."""
        mock_client = Mock()
        variables = [
            VariableMetadata(
                symbol="v",
                description="velocity",
                known=True,
                value=10.0,
                unit="m/s",
                type="scalar",
            ),
            VariableMetadata(
                symbol="t",
                description="time",
                known=False,
                value=None,
                unit="s",
                type="scalar",
            ),
        ]
        mock_response = VariableResponse(
            variables=variables, problem_summary="Find time given velocity"
        )
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        locator = VariableLocator(model="gpt-4o")
        result = locator.work("A car travels at 10 m/s. Find the time.")

        assert isinstance(result, VariableAnnotation)
        assert len(result.known_variables) >= 1
        assert len(result.unknown_variables) >= 1
        assert "v" in result.known_variables or any(
            "v" in str(k) for k in result.known_variables.keys()
        )

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_variable_locator_work_fallback(self, mock_llm_class):
        """Test variable extraction falls back to regular LLM call."""
        mock_client = Mock()
        mock_client.chat_structured.side_effect = Exception("Structured failed")
        fallback_response = json.dumps(
            {
                "variables": [
                    {
                        "symbol": "v",
                        "description": "velocity",
                        "known": True,
                        "value": 10.0,
                        "unit": "m/s",
                    }
                ],
                "problem_summary": "Test",
            }
        )
        mock_client.chat.return_value = fallback_response
        mock_llm_class.from_model.return_value = mock_client

        locator = VariableLocator(model="gpt-4o")
        result = locator.work("Test question")

        assert isinstance(result, VariableAnnotation)

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_variable_locator_work_both_fail(self, mock_llm_class):
        """Test variable extraction when both methods fail."""
        mock_client = Mock()
        mock_client.chat_structured.side_effect = Exception("Structured failed")
        mock_client.chat.side_effect = Exception("Regular failed")
        mock_llm_class.from_model.return_value = mock_client

        locator = VariableLocator(model="gpt-4o")
        result = locator.work("Test question")

        # Should return empty annotation
        assert isinstance(result, VariableAnnotation)
        assert len(result.known_variables) == 0
        assert len(result.unknown_variables) == 0

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_variable_locator_separates_known_unknown(self, mock_llm_class):
        """Test that known and unknown variables are properly separated."""
        mock_client = Mock()
        variables = [
            VariableMetadata(
                symbol="v", description="velocity", known=True, value=10.0, unit="m/s"
            ),
            VariableMetadata(
                symbol="t", description="time", known=False, value=None, unit="s"
            ),
        ]
        mock_response = VariableResponse(variables=variables, problem_summary="Test")
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        locator = VariableLocator(model="gpt-4o")
        result = locator.work("Test question")

        # Check that known and unknown are separated
        assert len(result.known_variables) >= 1
        assert len(result.unknown_variables) >= 1


class TestVariableAnnotation:
    """Test cases for VariableAnnotation dataclass."""

    def test_variable_annotation_to_dict(self):
        """Test converting VariableAnnotation to dictionary."""
        annotation = VariableAnnotation(
            known_variables={"v": {"value": 10.0, "unit": "m/s"}},
            unknown_variables={"t": {"unit": "s"}},
        )

        result = annotation.to_dict()

        assert "known_variables" in result
        assert "unknown_variables" in result
        assert result["known_variables"]["v"]["value"] == 10.0
