"""
Tests for BaseModelClient abstract base class.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.base import BaseModelClient


class TestBaseModelClient:
    """Test cases for BaseModelClient."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseModelClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelClient("test-model")

    def test_base_class_has_abstract_chat_method(self):
        """Test that BaseModelClient defines abstract chat method."""
        # Check that chat is abstract
        assert hasattr(BaseModelClient, "chat")
        # Try to create a concrete implementation without chat
        with pytest.raises(TypeError):
            class IncompleteModel(BaseModelClient):
                pass
            IncompleteModel("test-model")

    def test_base_class_initialization_attributes(self):
        """Test that BaseModelClient initializes with correct attributes."""
        class ConcreteModel(BaseModelClient):
            def chat(self, user_prompt, image_paths=None):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            assert client.model == "test-model"
            assert client.client is None
            assert client.provider is None
            assert client.logger is not None

    def test_base_class_with_custom_logger(self):
        """Test BaseModelClient initialization with custom logger."""
        import logging
        custom_logger = logging.getLogger("custom")

        class ConcreteModel(BaseModelClient):
            def chat(self, user_prompt, image_paths=None):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model", logger=custom_logger)
            assert client.logger == custom_logger

    def test_base_class_loads_project_dotenv(self):
        """Test that BaseModelClient loads environment variables."""
        class ConcreteModel(BaseModelClient):
            def chat(self, user_prompt, image_paths=None):
                return "response"

        with patch(
            "prkit.core.model_clients.base.load_project_dotenv"
        ) as mock_load:
            ConcreteModel("test-model")
            mock_load.assert_called_once()

    def test_concrete_implementation_must_implement_chat(self):
        """Test that concrete implementations must implement chat method."""
        class ConcreteModel(BaseModelClient):
            def chat(self, user_prompt, image_paths=None):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            # Should be able to call chat
            result = client.chat("test prompt")
            assert result == "response"

    def test_chat_method_signature(self):
        """Test that chat method accepts correct parameters."""
        class ConcreteModel(BaseModelClient):
            def chat(self, user_prompt, image_paths=None):
                return f"Response to: {user_prompt}"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            # Test with text only
            result = client.chat("Hello")
            assert "Hello" in result

            # Test with images
            result = client.chat("Hello", image_paths=["image.jpg"])
            assert "Hello" in result

    def test_chat_structured_returns_parsed_model(self):
        class ResponseModel(BaseModel):
            answer: str

        class ConcreteModel(BaseModelClient):
            supports_response_format_json_schema = True

            def chat(self, user_prompt, image_paths=None, response_format=None, **kwargs: Any):
                assert response_format["type"] == "json_schema"
                assert response_format["name"] == "ResponseModel"
                return '{"answer":"ok"}'

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            client.provider = "dummy"
            result = client.chat_structured(
                "Hello",
                response_model=ResponseModel,
                structured_policy="best_effort",
            )

        assert result.parsed == ResponseModel(answer="ok")
        assert result.structured_output_mode == "json_schema"
        assert result.structured_output_strategy == "dummy_json_schema"

    def test_chat_structured_native_required_rejects_non_native_provider(self):
        class ResponseModel(BaseModel):
            answer: str

        class ConcreteModel(BaseModelClient):
            supports_response_format_json_schema = False
            supports_response_format_json_object = True

            def chat(self, user_prompt, image_paths=None, response_format=None, **kwargs: Any):
                return '{"answer":"ok"}'

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            client.provider = "dummy"
            with pytest.raises(ValueError, match="native provider-enforced schema support"):
                client.chat_structured(
                    "Hello",
                    response_model=ResponseModel,
                    structured_policy="native_required",
                )
