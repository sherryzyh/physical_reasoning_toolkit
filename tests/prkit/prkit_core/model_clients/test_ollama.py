"""
Tests for Ollama model client.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from prkit.prkit_core.model_clients.ollama import OllamaModel


class TestOllamaModel:
    """Test cases for OllamaModel class."""

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_check_ollama_running_success(self, mock_ollama_module):
        """Test checking if Ollama is running successfully."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running()
        assert result is True
        mock_client.list.assert_called_once()

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_check_ollama_running_with_base_url(self, mock_ollama_module):
        """Test checking Ollama with custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running("http://custom:11434")
        assert result is True
        mock_ollama_module.Client.assert_called_once_with(host="http://custom:11434")

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_check_ollama_running_failure(self, mock_ollama_module):
        """Test checking Ollama when it's not running."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running()
        assert result is False

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_init_success(self, mock_ollama_module):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel("qwen3-vl")
        assert client.model == "qwen3-vl"
        assert client.provider == "ollama"
        assert client.base_url is None

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_init_with_base_url(self, mock_ollama_module):
        """Test initialization with custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel("qwen3-vl", base_url="http://custom:11434")
        assert client.model == "qwen3-vl"
        assert client.base_url == "http://custom:11434"

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_init_connection_error(self, mock_ollama_module):
        """Test initialization when Ollama is not running."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama_module.Client.return_value = mock_client

        with pytest.raises(ConnectionError, match="Ollama service is not running"):
            OllamaModel("qwen3-vl")

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_text_only(self, mock_ollama_module):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Test response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel("qwen3-vl")
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_ollama_module.chat.assert_called_once()
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["model"] == "qwen3-vl"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Hello, world!"

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_with_images(self, mock_ollama_module, tmp_path):
        """Test chat with image inputs."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        # Create temporary image files
        image1 = tmp_path / "test1.jpg"
        image2 = tmp_path / "test2.png"
        image1.write_bytes(b"fake image data 1")
        image2.write_bytes(b"fake image data 2")

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Image description"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel("qwen3-vl")
        response = client.chat("Describe these images", image_paths=[str(image1), str(image2)])

        assert response == "Image description"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert "images" in call_kwargs["messages"][0]
        assert len(call_kwargs["messages"][0]["images"]) == 2

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_with_nonexistent_image(self, mock_ollama_module):
        """Test chat with non-existent image raises FileNotFoundError."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = OllamaModel("qwen3-vl")
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            client.chat("Describe this", image_paths=["/nonexistent/image.jpg"])

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_with_base_url(self, mock_ollama_module):
        """Test chat using custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_client.chat.return_value = Mock(message=Mock(content="Response"))
        mock_ollama_module.Client.return_value = mock_client

        client = OllamaModel("qwen3-vl", base_url="http://custom:11434")
        response = client.chat("Hello")

        assert response == "Response"
        mock_client.chat.assert_called_once()

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_model_not_found_error(self, mock_ollama_module):
        """Test chat when model is not found."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        error = Exception("model 'unknown-model' not found")
        error.status_code = 404
        mock_ollama_module.chat.side_effect = error

        client = OllamaModel("unknown-model")
        with pytest.raises(ValueError, match="Model 'unknown-model' not found"):
            client.chat("Hello")

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_connection_error(self, mock_ollama_module):
        """Test chat when connection fails."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        error = Exception("Connection refused")
        mock_ollama_module.chat.side_effect = error

        client = OllamaModel("qwen3-vl")
        with pytest.raises(ConnectionError, match="Ollama service is not running"):
            client.chat("Hello")

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_response_dict_format(self, mock_ollama_module):
        """Test chat when response is in dict format."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = {"message": {"content": "Dict response"}}
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel("qwen3-vl")
        response = client.chat("Hello")

        assert response == "Dict response"

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_with_empty_image_list(self, mock_ollama_module):
        """Test chat with empty image list."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel("qwen3-vl")
        response = client.chat("Hello", image_paths=[])

        assert response == "Response"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert "images" not in call_kwargs["messages"][0]

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_temperature_option(self, mock_ollama_module):
        """Test that temperature option is set correctly."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel("qwen3-vl")
        client.chat("Hello")

        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["temperature"] == 0.7

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_chat_with_custom_logger(self, mock_ollama_module):
        """Test initialization with custom logger."""
        import logging
        logger = logging.getLogger("test_ollama")
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel("qwen3-vl", logger=logger)
        assert client.logger == logger
