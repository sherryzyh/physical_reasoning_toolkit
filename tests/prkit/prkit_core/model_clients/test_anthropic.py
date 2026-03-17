"""
Tests for Anthropic model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from prkit.prkit_core.model_clients.anthropic import AnthropicModel


class TestAnthropicModel:
    """Test cases for AnthropicModel class."""

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
    def test_init(self, mock_anthropic_class):
        """Test initializing Anthropic model."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = AnthropicModel("claude-sonnet-4-6")

        assert client.model == "claude-sonnet-4-6"
        assert client.provider == "anthropic"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
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

        client = AnthropicModel("claude-sonnet-4-6")
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}],
            }
        ]

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
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

        client = AnthropicModel("claude-sonnet-4-6")
        data_url = "data:image/png;base64,ZmFrZS1kYXRh"
        response = client.chat("Describe image", image_paths=[data_url])

        assert response == "Image response"
        call_kwargs = mock_client.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe image"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
        assert content[1]["source"]["data"] == "ZmFrZS1kYXRh"

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
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

        client = AnthropicModel("claude-sonnet-4-6")
        with patch.object(client.logger, "warning") as mock_warning:
            response = client.chat(
                "Hello",
                image_paths=["https://example.com/image.jpg"],
            )

        assert response == "Response"
        mock_warning.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == [
            {"type": "text", "text": "Hello"}
        ]

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
    def test_chat_response_format_warns(self, mock_anthropic_class):
        """Test response_format logs warning and is ignored."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response = Mock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicModel("claude-sonnet-4-6")
        with patch.object(client.logger, "warning") as mock_warning:
            response = client.chat(
                "Hello",
                response_format={"type": "json_schema", "name": "x", "schema": {}},
            )

        assert response == "Response"
        mock_warning.assert_called_once()

    @patch("prkit.prkit_core.model_clients.anthropic.Anthropic")
    def test_chat_invalid_data_url_raises_value_error(self, mock_anthropic_class):
        """Test malformed data URL raises ValueError."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client = AnthropicModel("claude-sonnet-4-6")
        with pytest.raises(ValueError, match="base64"):
            client.chat("Hello", image_paths=["data:image/png,not-base64"])

