"""
Tests for DeepSeek model client.
"""

from unittest.mock import MagicMock, Mock, patch

from pydantic import BaseModel

from prkit.core.model_clients.deepseek import DeepseekModel

DEEPSEEK_CHAT_TEST_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_TEST_MODEL = "deepseek-reasoner"


class TestDeepseekModel:
    """Test cases for DeepseekModel class."""

    @patch("prkit.core.model_clients.base.load_project_dotenv")
    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    def test_init(self, mock_openai_class, _mock_load_project_dotenv):
        """Test initializing DeepSeek model."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            client = DeepseekModel(DEEPSEEK_CHAT_TEST_MODEL)

        assert client.model == DEEPSEEK_CHAT_TEST_MODEL
        assert client.provider == "deepseek"
        assert client.supports_native_structured_output is True
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    def test_init_normalizes_provider_prefixed_model(self, mock_openai_class):
        """Test provider-prefixed model names are normalized before request use."""
        mock_openai_class.return_value = MagicMock()

        client = DeepseekModel(f"deepseek/{DEEPSEEK_CHAT_TEST_MODEL}")

        assert client.model == DEEPSEEK_CHAT_TEST_MODEL

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
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

        client = DeepseekModel(DEEPSEEK_REASONER_TEST_MODEL)
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=DEEPSEEK_REASONER_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello, world!"}],
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
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

        client = DeepseekModel(DEEPSEEK_CHAT_TEST_MODEL)
        with patch.object(client.logger, "warning") as mock_warning:
            response = client.chat("Hello", image_paths=["image.jpg"])

        assert response == "Response"
        mock_warning.assert_called_once()
        assert "does not support image inputs" in mock_warning.call_args[0][0].lower()

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
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

        client = DeepseekModel(DEEPSEEK_REASONER_TEST_MODEL)
        client.chat("Hello", image_paths=["image1.jpg", "image2.png"])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["content"] == "Hello"

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    def test_chat_response_format_uses_json_object(self, mock_openai_class):
        """Test DeepSeek structured output falls back to json_object mode."""

        class ExampleResponse(BaseModel):
            answer: str

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = '{"answer":"ok"}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DeepseekModel(DEEPSEEK_CHAT_TEST_MODEL)
        response = client.chat(
            "Return JSON only.",
            response_format=ExampleResponse,
            max_output_tokens=512,
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == DEEPSEEK_CHAT_TEST_MODEL
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["max_tokens"] == 512
        assert "Return JSON only." in call_kwargs["messages"][0]["content"]
        assert "Return ONLY JSON" in call_kwargs["messages"][0]["content"]
