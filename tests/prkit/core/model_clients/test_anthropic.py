"""
Tests for Anthropic model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.anthropic import (
    AnthropicModel,
    _detect_image_media_type,
    _extract_tool_use_json,
    _parse_data_url,
)
from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS

ANTHROPIC_TEST_MODEL = "claude-sonnet-4-6"


class TestAnthropicModel:
    """Test cases for AnthropicModel class."""

    @patch("prkit.core.model_clients.base.load_project_dotenv")
    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_init(self, mock_anthropic_class, _mock_load_project_dotenv):
        """Test initializing Anthropic model."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = AnthropicModel(ANTHROPIC_TEST_MODEL)

        assert client.model == ANTHROPIC_TEST_MODEL
        assert client.provider == "anthropic"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_text_only(self, mock_anthropic_class):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Test response"
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == ANTHROPIC_TEST_MODEL
        assert call_kwargs["max_tokens"] == 1024
        # Default system prompt is sent as Anthropic's top-level `system` param.
        assert call_kwargs["system"] == DEFAULT_INSTRUCTIONS
        assert call_kwargs["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}],
            }
        ]

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_response_system_prompt_explicit_and_suppressed(self, mock_anthropic_class):
        """Explicit instructions override the default; an empty string suppresses it."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "ok"
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)

        client.response("Hi", instructions="Custom system.")
        assert mock_client.messages.create.call_args[1]["system"] == "Custom system."

        client.response("Hi", instructions="")
        assert "system" not in mock_client.messages.create.call_args[1]

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_with_data_url_image(self, mock_anthropic_class):
        """Test chat with base64 data URL image input."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Image response"
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        data_url = "data:image/png;base64,ZmFrZS1kYXRh"
        response = client.response("Describe image", image_paths=[data_url])

        assert response == "Image response"
        call_kwargs = mock_client.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe image"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
        assert content[1]["source"]["data"] == "ZmFrZS1kYXRh"

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_with_http_image_warns_and_ignores(self, mock_anthropic_class):
        """Test chat warns and ignores HTTP image URL inputs."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        with patch.object(client.logger, "warning") as mock_warning:
            response = client.response(
                "Hello",
                image_paths=["https://example.com/image.jpg"],
            )

        assert response == "Response"
        mock_warning.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == [
            {"type": "text", "text": "Hello"}
        ]

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_response_format_uses_output_config(self, mock_anthropic_class):
        """Test response_format uses Anthropic json-schema structured output."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        class ExampleResponse(BaseModel):
            answer: str

        text_block = Mock()
        text_block.type = "text"
        text_block.text = '{"answer":"Response"}'
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        response = client.response(
            "Hello",
            response_format=ExampleResponse,
        )

        assert response == '{"answer":"Response"}'
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        # Anthropic forbids a ``name`` key in output_config.format (400 otherwise).
        assert "name" not in call_kwargs["output_config"]["format"]
        assert call_kwargs["output_config"]["format"]["schema"]["type"] == "object"
        assert "tools" not in call_kwargs

    def test_extract_tool_use_json_requires_exactly_one_tool_block(self):
        with pytest.raises(ValueError, match="exactly one tool_use block"):
            _extract_tool_use_json([{"type": "text", "text": "not structured"}])

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_invalid_data_url_raises_value_error(self, mock_anthropic_class):
        """Test malformed data URL raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        with pytest.raises(ValueError, match="base64"):
            client.response("Hello", image_paths=["data:image/png,not-base64"])

    @patch("prkit.core.model_clients.anthropic.encode_image_to_base64")
    @patch("prkit.core.model_clients.anthropic.os.path.exists")
    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_chat_with_local_image_and_extra_kwargs(
        self,
        mock_anthropic_class,
        mock_exists,
        mock_encode_image,
    ):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_exists.return_value = True
        mock_encode_image.return_value = "ZmFrZQ=="

        text_block = Mock(type="text", text="Image response")
        non_text_block = Mock(type="tool_result")
        mock_client.messages.create.return_value = Mock(
            content=[non_text_block, text_block]
        )

        client = AnthropicModel(ANTHROPIC_TEST_MODEL)
        response = client.response(
            "Describe",
            image_paths=["/tmp/example.png"],
            temperature=0.2,
        )

        assert response == "Image response"
        content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert content[1]["source"]["media_type"] == "image/png"
        assert content[1]["source"]["data"] == "ZmFrZQ=="
        assert mock_client.messages.create.call_args.kwargs["temperature"] == 0.2

    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_raises_when_anthropic_package_missing(
        self, _mock_load_project_dotenv
    ):
        with patch("prkit.core.model_clients.anthropic.Anthropic", None):
            with pytest.raises(ImportError, match="anthropic package not installed"):
                AnthropicModel(ANTHROPIC_TEST_MODEL)

    def test_anthropic_helper_functions_cover_default_paths(self):
        assert _detect_image_media_type("image.unknown") == "image/jpeg"
        assert _parse_data_url("data:;base64,ZmFrZQ==") == {
            "media_type": "image/jpeg",
            "data": "ZmFrZQ==",
        }
