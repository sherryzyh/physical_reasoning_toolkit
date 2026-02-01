"""
Tests for DeepSeek model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from prkit.prkit_core.model_clients.deepseek import DeepseekModel


class TestDeepseekModel:
    """Test cases for DeepseekModel class."""

    @patch("prkit.prkit_core.model_clients.deepseek.OpenAI")
    def test_init(self, mock_openai_class):
        """Test initializing DeepSeek model."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            client = DeepseekModel("deepseek-chat")

        assert client.model == "deepseek-chat"
        assert client.provider == "deepseek"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com"
        )

    @patch("prkit.prkit_core.model_clients.deepseek.OpenAI")
    def test_chat_text_only(self, mock_openai_class):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DeepseekModel("deepseek-chat")
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello, world!"}]
        )

    @patch("prkit.prkit_core.model_clients.deepseek.OpenAI")
    def test_chat_with_images_warning(self, mock_openai_class):
        """Test chat with images logs warning."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DeepseekModel("deepseek-chat")
        with patch.object(client.logger, "warning") as mock_warning:
            response = client.chat("Hello", image_paths=["image.jpg"])

        assert response == "Response"
        mock_warning.assert_called_once()
        assert "does not support image inputs" in mock_warning.call_args[0][0].lower()

    @patch("prkit.prkit_core.model_clients.deepseek.OpenAI")
    def test_chat_ignores_images(self, mock_openai_class):
        """Test that images are ignored in the API call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DeepseekModel("deepseek-chat")
        client.chat("Hello", image_paths=["image1.jpg", "image2.png"])

        # Verify API call only includes text, not images
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["content"] == "Hello"
