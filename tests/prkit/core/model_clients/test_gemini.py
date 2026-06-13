"""
Tests for Gemini model client.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.gemini import (
    GeminiModel,
    _extract_gemini_error_details,
)

GEMINI_TEST_MODEL = "gemini-2.5-pro"


class TestGeminiModel:
    """Test cases for GeminiModel class."""

    @patch("prkit.core.model_clients.gemini.genai")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_with_api_key(self, _mock_load_project_dotenv, mock_genai):
        """Test initializing with API key from environment."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            client = GeminiModel(GEMINI_TEST_MODEL)

        assert client.model == GEMINI_TEST_MODEL
        assert client.provider == "google"
        mock_genai.Client.assert_called_once_with(api_key="test-key")

    @patch("prkit.core.model_clients.gemini.genai")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_with_google_api_key_fallback(self, _mock_load_project_dotenv, mock_genai):
        """Test initializing with GOOGLE_API_KEY as fallback."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            GeminiModel(GEMINI_TEST_MODEL)

        mock_genai.Client.assert_called_once_with(api_key="google-key")

    @patch("prkit.core.model_clients.gemini.genai")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_without_api_key(self, _mock_load_project_dotenv, mock_genai):
        """Test initializing without API key (uses default)."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            GeminiModel(GEMINI_TEST_MODEL)

        mock_genai.Client.assert_called_once_with()

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_text_only(self, mock_genai):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == GEMINI_TEST_MODEL
        # contents is a list of parts (text strings and/or PIL Images)
        assert len(call_kwargs["contents"]) == 1
        assert call_kwargs["contents"][0] == "Hello, world!"
        config = call_kwargs["config"]
        assert config is not None
        assert config.max_output_tokens == 65535

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_with_kwargs(self, mock_genai):
        """Test chat with additional kwargs for config."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        client.chat("Hello", temperature=0.7)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "config" in call_kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0.7
        assert config.max_output_tokens == 65535

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_without_config_kwargs(self, mock_genai):
        """Test chat without explicit config kwargs still passes default config."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        client.chat("Hello")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config is not None
        assert config.max_output_tokens == 65535

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_with_images_error(self, mock_genai):
        """Test chat with non-existent image path logs error."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        with patch.object(client.logger, "error") as mock_error:
            response = client.chat("Hello", image_paths=["image.jpg"])

        assert response == "Response"
        mock_error.assert_called_once()
        assert "Image path not found" in mock_error.call_args[0][0]
        assert "image.jpg" in mock_error.call_args[0][0]

    @patch("prkit.core.model_clients.gemini.normalize_response_format")
    @patch("prkit.core.model_clients.gemini.extract_schema_for_gemini")
    @patch("prkit.core.model_clients.gemini.PIL.Image.open")
    @patch("prkit.core.model_clients.gemini.os.path.exists")
    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_with_structured_output_and_loaded_image(
        self,
        mock_genai,
        mock_exists,
        mock_image_open,
        mock_extract_schema,
        mock_normalize_response_format,
    ):
        """Structured-output config and valid images should be forwarded to Gemini."""

        class ExampleResponse(BaseModel):
            answer: str

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_exists.return_value = True
        mock_image_open.return_value = "image-object"
        mock_normalize_response_format.return_value = {"type": "json_schema"}
        mock_extract_schema.return_value = {"type": "object"}

        mock_response = Mock()
        mock_response.text = '{"answer":"ok"}'
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        response = client.chat(
            "Return JSON",
            image_paths=["/tmp/example.png"],
            response_format=ExampleResponse,
            temperature=0.1,
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["contents"] == ["Return JSON", "image-object"]
        assert call_kwargs["config"].response_mime_type == "application/json"
        assert call_kwargs["config"].response_json_schema == {"type": "object"}
        assert call_kwargs["config"].temperature == 0.1

    @patch("prkit.core.model_clients.gemini.PIL.Image.open")
    @patch("prkit.core.model_clients.gemini.os.path.exists")
    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_logs_failed_image_open(self, mock_genai, mock_exists, mock_image_open):
        """Unreadable images should be logged and ignored."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_exists.return_value = True
        mock_image_open.side_effect = OSError("bad image")
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        with patch.object(client.logger, "error") as mock_error:
            response = client.chat("Hello", image_paths=["/tmp/bad.png"])

        assert response == "Response"
        mock_error.assert_called_once()
        assert "Failed to load image" in mock_error.call_args[0][0]

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_empty_response_raises_runtime_error_with_details(self, mock_genai):
        """Empty Gemini responses should raise with extracted block details."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        response = Mock()
        response.text = None
        response.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
        response.candidates = [SimpleNamespace(finish_reason="RECITATION")]
        mock_client.models.generate_content.return_value = response

        client = GeminiModel(GEMINI_TEST_MODEL)
        with pytest.raises(
            RuntimeError,
            match="prompt_block_reason=SAFETY; finish_reason=RECITATION",
        ):
            client.chat("Hello")

    def test_extract_gemini_error_details_handles_empty_and_prompt_blocks(self):
        empty = SimpleNamespace(prompt_feedback=None, candidates=None)
        blocked = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="BLOCKED"),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

        assert _extract_gemini_error_details(empty) == "empty_response"
        assert _extract_gemini_error_details(blocked) == "prompt_block_reason=BLOCKED"
