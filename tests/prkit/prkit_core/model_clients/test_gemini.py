"""
Tests for Gemini model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from prkit.prkit_core.model_clients.gemini import GeminiModel


class TestGeminiModel:
    """Test cases for GeminiModel class."""

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    def test_init_with_api_key(self, mock_genai):
        """Test initializing with API key from environment."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            client = GeminiModel("gemini-pro")

        assert client.model == "gemini-pro"
        assert client.provider == "google"
        mock_genai.Client.assert_called_once_with(api_key="test-key")

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    @patch("prkit.prkit_core.model_clients.gemini.load_dotenv")
    @patch("prkit.prkit_core.model_clients.base.load_dotenv")
    def test_init_with_google_api_key_fallback(self, mock_base_load_dotenv, mock_load_dotenv, mock_genai):
        """Test initializing with GOOGLE_API_KEY as fallback."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            client = GeminiModel("gemini-pro")

        mock_genai.Client.assert_called_once_with(api_key="google-key")

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    @patch("prkit.prkit_core.model_clients.gemini.load_dotenv")
    @patch("prkit.prkit_core.model_clients.base.load_dotenv")
    def test_init_without_api_key(self, mock_base_load_dotenv, mock_load_dotenv, mock_genai):
        """Test initializing without API key (uses default)."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            client = GeminiModel("gemini-pro")

        mock_genai.Client.assert_called_once_with()

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    def test_chat_text_only(self, mock_genai):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel("gemini-pro")
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-pro"
        # contents is a list of parts (text strings and/or PIL Images)
        assert len(call_kwargs["contents"]) == 1
        assert call_kwargs["contents"][0] == "Hello, world!"
        # Verify config is None when no kwargs provided
        assert call_kwargs.get("config") is None

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    def test_chat_with_kwargs(self, mock_genai):
        """Test chat with additional kwargs for config."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel("gemini-pro")
        client.chat("Hello", temperature=0.7)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "config" in call_kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0.7

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    def test_chat_without_config_kwargs(self, mock_genai):
        """Test chat without config kwargs passes None."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel("gemini-pro")
        client.chat("Hello")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs.get("config") is None

    @patch("prkit.prkit_core.model_clients.gemini.genai")
    def test_chat_with_images_error(self, mock_genai):
        """Test chat with non-existent image path logs error."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel("gemini-pro")
        with patch.object(client.logger, "error") as mock_error:
            response = client.chat("Hello", image_paths=["image.jpg"])

        assert response == "Response"
        mock_error.assert_called_once()
        assert "Image path not found" in mock_error.call_args[0][0]
        assert "image.jpg" in mock_error.call_args[0][0]

