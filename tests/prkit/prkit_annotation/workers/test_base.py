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
        with patch("prkit.prkit_annotation.workers.base.create_model_client") as mock_create:
            mock_client_instance = Mock()
            mock_create.return_value = mock_client_instance

            annotator = ConcreteAnnotator(model="gpt-5.1")

            assert annotator.model == "gpt-5.1"
            assert annotator.llm_client == mock_client_instance
            mock_create.assert_called_once_with("gpt-5.1")

    def test_base_annotator_work_abstract(self):
        """Test that work method is abstract."""
        with patch("prkit.prkit_annotation.workers.base.create_model_client") as mock_create:
            mock_client_instance = Mock()
            mock_create.return_value = mock_client_instance

            # Test that abstract class cannot be instantiated
            with pytest.raises(TypeError, match="abstract"):
                BaseAnnotator(model="gpt-5.1")

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_base_annotator_call_llm_structured(self, mock_create):
        """Test _call_llm_structured method."""
        from pydantic import BaseModel
        
        class TestResponse(BaseModel):
            result: str
        
        mock_client = Mock()
        mock_client.chat.return_value = '{"result": "structured"}'
        mock_create.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-5.1")
        result = annotator._call_llm_structured("test prompt", TestResponse)

        assert result is not None
        assert result.result == "structured"
        mock_client.chat.assert_called_once()

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_base_annotator_call_llm_structured_error(self, mock_create):
        """Test _call_llm_structured with error handling."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("API error")
        mock_create.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm_structured("test prompt", "response_format")

        assert result is None

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_base_annotator_call_llm(self, mock_create):
        """Test _call_llm method."""
        mock_client = Mock()
        mock_client.chat.return_value = "LLM response"
        mock_create.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm("test prompt")

        assert result == "LLM response"
        mock_client.chat.assert_called_once()

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_base_annotator_call_llm_error(self, mock_create):
        """Test _call_llm with error handling."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("API error")
        mock_create.return_value = mock_client

        annotator = ConcreteAnnotator(model="gpt-4o")
        result = annotator._call_llm("test prompt")

        assert result == "{}"
