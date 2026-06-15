"""
Tests for Ollama model client.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.ollama import OllamaModel, normalize_ollama_model_name

OLLAMA_QWEN_TEST_MODEL = "ollama/qwen3.5:397b-cloud"
OLLAMA_MISTRAL_TEST_MODEL = "ollama/mistral-large-3:675b-cloud"
SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_INSTRUCTIONS}


class TestOllamaModel:
    def test_normalize_ollama_model_name_prefixed(self):
        """Test prefixed Ollama identifiers are normalized."""
        assert (
            normalize_ollama_model_name(OLLAMA_QWEN_TEST_MODEL) == "qwen3.5:397b-cloud"
        )

    def test_normalize_ollama_model_name_plain(self):
        """Test plain Ollama identifiers are preserved."""
        assert normalize_ollama_model_name("llava:latest") == "llava:latest"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_check_ollama_running_success(self, mock_ollama_module):
        """Test checking if Ollama is running successfully."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running()
        assert result is True
        mock_client.list.assert_called_once()

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_check_ollama_running_with_base_url(self, mock_ollama_module):
        """Test checking Ollama with custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running("http://custom:11434")
        assert result is True
        mock_ollama_module.Client.assert_called_once_with(host="http://custom:11434")

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_check_ollama_running_failure(self, mock_ollama_module):
        """Test checking Ollama when it's not running."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama_module.Client.return_value = mock_client

        result = OllamaModel.check_ollama_running()
        assert result is False

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_init_success(self, mock_ollama_module):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        assert client.model == "qwen3.5:397b-cloud"
        assert client.provider == "ollama"
        assert client.base_url is None

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_init_with_base_url(self, mock_ollama_module):
        """Test initialization with custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL, base_url="http://custom:11434")
        assert client.model == "qwen3.5:397b-cloud"
        assert client.base_url == "http://custom:11434"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_init_with_prefixed_model_name(self, mock_ollama_module):
        """Test initialization normalizes provider-prefixed model names."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        assert client.model == "qwen3.5:397b-cloud"
        assert client.provider == "ollama"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_init_connection_error(self, mock_ollama_module):
        """Test initialization when Ollama is not running."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama_module.Client.return_value = mock_client

        with pytest.raises(ConnectionError, match="Ollama service is not running"):
            OllamaModel(OLLAMA_QWEN_TEST_MODEL)

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_text_only(self, mock_ollama_module):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Test response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_ollama_module.chat.assert_called_once()
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["model"] == "qwen3.5:397b-cloud"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == SYSTEM_MESSAGE
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "Hello, world!"
        assert "format" not in call_kwargs

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_pydantic_response_format_uses_json_schema(
        self, mock_ollama_module
    ):
        """Structured output should pass a JSON Schema to Ollama's format parameter."""

        class ExampleResponse(BaseModel):
            answer: str
            notes: str | None = None

        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = '{"answer":"ok","notes":null}'
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response(
            "Return JSON only.",
            response_format=ExampleResponse,
            max_output_tokens=256,
        )

        assert response == '{"answer":"ok","notes":null}'
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["format"]["type"] == "object"
        assert set(call_kwargs["format"]["properties"]) == {"answer", "notes"}
        assert call_kwargs["format"]["required"] == ["answer"]
        assert call_kwargs["options"]["num_predict"] == 256

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_json_object_response_format_uses_json_mode(
        self, mock_ollama_module
    ):
        """Generic JSON mode should map to Ollama's format='json'."""

        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = '{"answer":"ok"}'
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response(
            "Return JSON only.",
            response_format={"type": "json_object"},
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["format"] == "json"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_uses_normalized_prefixed_model_name(self, mock_ollama_module):
        """Test chat strips ollama/ prefix before API call."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response("Hello")

        assert response == "Response"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["model"] == "qwen3.5:397b-cloud"

    @patch("prkit.core.model_clients.ollama.ollama")
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

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response(
            "Describe these images", image_paths=[str(image1), str(image2)]
        )

        assert response == "Image description"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["messages"][0] == SYSTEM_MESSAGE
        assert "images" in call_kwargs["messages"][1]
        assert len(call_kwargs["messages"][1]["images"]) == 2

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_nonexistent_image(self, mock_ollama_module):
        """Test chat with non-existent image raises FileNotFoundError."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = OllamaModel(OLLAMA_MISTRAL_TEST_MODEL)
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            client.response("Describe this", image_paths=["/nonexistent/image.jpg"])

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_base_url(self, mock_ollama_module):
        """Test chat using custom base URL."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_client.chat.return_value = Mock(message=Mock(content="Response"))
        mock_ollama_module.Client.return_value = mock_client

        client = OllamaModel(OLLAMA_MISTRAL_TEST_MODEL, base_url="http://custom:11434")
        response = client.response("Hello")

        assert response == "Response"
        mock_client.chat.assert_called_once()

    @patch("prkit.core.model_clients.ollama.ollama")
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
            client.response("Hello")

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_connection_error(self, mock_ollama_module):
        """Test chat when connection fails."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        error = Exception("Connection refused")
        mock_ollama_module.chat.side_effect = error

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        with pytest.raises(ConnectionError, match="Ollama service is not running"):
            client.response("Hello")

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_response_dict_format(self, mock_ollama_module):
        """Test chat when response is in dict format."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = {"message": {"content": "Dict response"}}
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_MISTRAL_TEST_MODEL)
        response = client.response("Hello")

        assert response == "Dict response"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_empty_image_list(self, mock_ollama_module):
        """Test chat with empty image list."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        response = client.response("Hello", image_paths=[])

        assert response == "Response"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert "images" not in call_kwargs["messages"][0]

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_temperature_option(self, mock_ollama_module):
        """Test that temperature option is set correctly."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        client.response("Hello")

        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["temperature"] == 0

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_chat_with_custom_logger(self, mock_ollama_module):
        """Test initialization with custom logger."""
        import logging

        logger = logging.getLogger("test_ollama")
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client
        mock_ollama_module.chat = MagicMock()

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL, logger=logger)
        assert client.logger == logger


class TestOllamaModelCloudAuth:
    """Tests for explicit api_key / api_key_env params and remote-safe preflight."""

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_explicit_api_key_forwarded_as_auth_header(self, mock_ollama_module):
        """Explicit api_key is sent as a lowercase authorization header to ollama.Client."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        OllamaModel(
            OLLAMA_QWEN_TEST_MODEL,
            base_url="https://ollama.com",
            api_key="test-cloud-key",
        )

        _, kwargs = mock_ollama_module.Client.call_args
        assert kwargs["host"] == "https://ollama.com"
        assert kwargs["headers"] == {"authorization": "Bearer test-cloud-key"}

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_api_key_env_resolves_named_var(self, mock_ollama_module):
        """api_key_env reads the key from the named environment variable."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        with patch.dict(os.environ, {"MY_OLLAMA_KEY": "env-ollama-key"}):
            OllamaModel(
                OLLAMA_QWEN_TEST_MODEL,
                base_url="https://ollama.com",
                api_key_env="MY_OLLAMA_KEY",
            )

        _, kwargs = mock_ollama_module.Client.call_args
        assert kwargs["headers"] == {"authorization": "Bearer env-ollama-key"}

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_no_api_key_falls_back_to_module_level_chat(self, mock_ollama_module):
        """Without base_url or api_key, chat falls back to module-level ollama.chat."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Response"
        mock_ollama_module.chat.return_value = mock_response

        client = OllamaModel(OLLAMA_QWEN_TEST_MODEL)
        client.response("Hello")

        mock_ollama_module.chat.assert_called_once()

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_remote_host_preflight_failure_warns_not_raises(self, mock_ollama_module):
        """A failed preflight against a remote host emits a warning but does not raise."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Network unreachable")
        mock_ollama_module.Client.return_value = mock_client

        # Should not raise — remote host gets a warning instead
        client = OllamaModel(
            OLLAMA_QWEN_TEST_MODEL,
            base_url="https://ollama.com",
            api_key="k",
        )
        assert client.base_url == "https://ollama.com"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_local_host_preflight_failure_still_raises(self, mock_ollama_module):
        """A failed preflight against a local host still raises ConnectionError."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama_module.Client.return_value = mock_client

        with pytest.raises(ConnectionError, match="Ollama service is not running"):
            OllamaModel(OLLAMA_QWEN_TEST_MODEL)
