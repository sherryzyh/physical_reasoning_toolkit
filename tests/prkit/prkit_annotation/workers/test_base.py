"""
Tests for BaseAnnotator.
"""

from unittest.mock import Mock, patch

import pytest

from prkit.prkit_annotation.workers.base import BaseAnnotator


# Create a concrete subclass for testing abstract methods
class ConcreteAnnotator(BaseAnnotator):
    """Concrete implementation of BaseAnnotator for testing."""
    
    def work(self, question: str, **kwargs):
        """Concrete implementation of abstract method."""
        return {"result": "test"}


class TestBaseAnnotator:
    """Test cases for BaseAnnotator."""

    def test_base_annotator_initialization(self):
        """Test base annotator initialization."""
        with patch("prkit.prkit_annotation.workers.base.LLMClient") as mock_llm:
            mock_client_instance = Mock()
            mock_llm.from_model.return_value = mock_client_instance

            annotator = ConcreteAnnotator(model="gpt-4o")

            assert annotator.model == "gpt-4o"
            assert annotator.llm_client == mock_client_instance
            mock_llm.from_model.assert_called_once_with("gpt-4o")

    def test_base_annotator_work_abstract(self):
        """Test that work method is abstract."""
        with patch("prkit.prkit_annotation.workers.base.LLMClient") as mock_llm:
            mock_client_instance = Mock()
            mock_llm.from_model.return_value = mock_client_instance

            # Test that abstract class cannot be instantiated
            with pytest.raises(TypeError, match="abstract"):
                BaseAnnotator(model="gpt-4o")

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_base_annotator_call_llm_structured(self, mock_llm_class):
        """Test _call_llm_structured method."""
        mock_client = Mock()
        mock_client.chat_structured.return_value = {"result": "structured"}
        mock_llm_class.from_model.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm_structured("test prompt", "response_format")

        assert result == {"result": "structured"}
        mock_client.chat_structured.assert_called_once()

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_base_annotator_call_llm_structured_error(self, mock_llm_class):
        """Test _call_llm_structured with error handling."""
        mock_client = Mock()
        mock_client.chat_structured.side_effect = Exception("API error")
        mock_llm_class.from_model.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm_structured("test prompt", "response_format")

        assert result is None

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_base_annotator_call_llm(self, mock_llm_class):
        """Test _call_llm method."""
        mock_client = Mock()
        mock_client.chat.return_value = "LLM response"
        mock_llm_class.from_model.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm("test prompt")

        assert result == "LLM response"
        mock_client.chat.assert_called_once()

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_base_annotator_call_llm_error(self, mock_llm_class):
        """Test _call_llm with error handling."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("API error")
        mock_llm_class.from_model.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm("test prompt")

        assert result == "{}"
