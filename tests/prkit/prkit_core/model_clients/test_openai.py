"""
Tests for OpenAI model client.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from prkit.prkit_core.model_clients.openai import (
    OpenAIModel,
    _is_o_family_model,
    _is_supported_openai_model,
    prepare_image_url_from_image_path,
)


class TestOpenAIModelValidation:
    """Test cases for OpenAI model validation functions."""

    def test_is_supported_openai_model_gpt_4_1(self):
        """Test gpt-4.1 is supported."""
        assert _is_supported_openai_model("gpt-4.1") is True
        assert _is_supported_openai_model("gpt-4.1-mini") is True
        assert _is_supported_openai_model("gpt-4.1-nano") is True

    def test_is_supported_openai_model_gpt_5(self):
        """Test gpt-5xxxx models are supported."""
        assert _is_supported_openai_model("gpt-5") is True
        assert _is_supported_openai_model("gpt-5.1") is True
        assert _is_supported_openai_model("gpt-5.2") is True
        assert _is_supported_openai_model("gpt-5.1-mini") is True

    def test_is_supported_openai_model_o_family(self):
        """Test o-family models are supported."""
        assert _is_supported_openai_model("o3") is True
        assert _is_supported_openai_model("o4") is True
        assert _is_supported_openai_model("o4-mini") is True

    def test_is_supported_openai_model_unsupported(self):
        """Test unsupported models return False."""
        assert _is_supported_openai_model("gpt-3.5-turbo") is False
        assert _is_supported_openai_model("gpt-4") is False
        assert _is_supported_openai_model("unknown") is False

    def test_is_o_family_model(self):
        """Test o-family model detection."""
        assert _is_o_family_model("o3") is True
        assert _is_o_family_model("o4") is True
        assert _is_o_family_model("o4-mini") is True
        assert _is_o_family_model("o1") is True

    def test_is_o_family_model_false(self):
        """Test non-o-family models return False."""
        assert _is_o_family_model("gpt-5.1") is False
        assert _is_o_family_model("gpt-4.1") is False
        assert _is_o_family_model("gemini-pro") is False
        assert _is_o_family_model("openai") is False  # starts with 'o' but not followed by digit


class TestOpenAIModel:
    """Test cases for OpenAIModel class."""

    def test_init_supported_model(self):
        """Test initializing with supported model."""
        client = OpenAIModel("gpt-5.1")
        assert client.model == "gpt-5.1"
        assert client.provider == "openai"
        assert client.is_o_family is False

    def test_init_o_family_model(self):
        """Test initializing with o-family model."""
        client = OpenAIModel("o3")
        assert client.model == "o3"
        assert client.is_o_family is True

    def test_init_unsupported_model(self):
        """Test initializing with unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            OpenAIModel("gpt-3.5-turbo")

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_text_only(self, mock_openai_class):
        """Test chat with text-only input."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("gpt-5.1")
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.responses.create.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.1"
        assert len(call_kwargs["input"]) == 1
        assert call_kwargs["input"][0]["type"] == "input_text"
        assert call_kwargs["input"][0]["text"] == "Hello, world!"

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_with_images(self, mock_openai_class, tmp_path):
        """Test chat with image inputs."""
        # Create a temporary image file
        image_file = tmp_path / "test_image.png"
        image_file.write_bytes(b"fake image data")

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("gpt-5.1")
        response = client.chat("Describe this image", image_paths=[str(image_file)])

        assert response == "Image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert len(call_kwargs["input"]) == 2  # text + image
        assert call_kwargs["input"][0]["type"] == "input_text"
        assert call_kwargs["input"][1]["type"] == "input_image"
        assert "image_url" in call_kwargs["input"][1]
        assert call_kwargs["input"][1]["image_url"].startswith("data:image/png;base64,")

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_with_http_url(self, mock_openai_class):
        """Test chat with HTTP URL image."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "URL image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("gpt-5.1")
        response = client.chat(
            "Describe this image",
            image_paths=["https://example.com/image.jpg"]
        )

        assert response == "URL image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"][1]["image_url"] == "https://example.com/image.jpg"

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_with_base64_data_url(self, mock_openai_class):
        """Test chat with base64 data URL."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Base64 image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("gpt-5.1")
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        response = client.chat("Describe this image", image_paths=[data_url])

        assert response == "Base64 image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"][1]["image_url"] == data_url

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_o_family_with_reasoning(self, mock_openai_class):
        """Test o-family models include reasoning parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Reasoned response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("o3")
        client.chat("Solve this problem")

        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"] == {"effort": "medium"}

    @patch("prkit.prkit_core.model_clients.openai.OpenAI")
    def test_chat_non_o_family_no_reasoning(self, mock_openai_class):
        """Test non-o-family models don't include reasoning parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Regular response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("gpt-5.1")
        client.chat("Hello")

        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" not in call_kwargs


class TestPrepareImageURL:
    """Test cases for prepare_image_url_from_image_path function."""

    def test_prepare_image_url_from_file_path(self, tmp_path):
        """Test preparing image URL from file path."""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        image_url = prepare_image_url_from_image_path(str(image_file))

        assert image_url.startswith("data:image/jpeg;base64,")
        assert len(image_url) > len("data:image/jpeg;base64,")

    def test_prepare_image_url_from_png(self, tmp_path):
        """Test preparing PNG image URL."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png data")

        image_url = prepare_image_url_from_image_path(str(image_file))

        assert image_url.startswith("data:image/png;base64,")

    def test_prepare_image_url_from_http_url(self):
        """Test HTTP URL is returned as-is."""
        url = "http://example.com/image.jpg"
        result = prepare_image_url_from_image_path(url)
        assert result == url

    def test_prepare_image_url_from_https_url(self):
        """Test HTTPS URL is returned as-is."""
        url = "https://example.com/image.jpg"
        result = prepare_image_url_from_image_path(url)
        assert result == url

    def test_prepare_image_url_from_data_url(self):
        """Test base64 data URL is returned as-is."""
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        result = prepare_image_url_from_image_path(data_url)
        assert result == data_url

    def test_prepare_image_url_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            prepare_image_url_from_image_path("/nonexistent/image.jpg")
