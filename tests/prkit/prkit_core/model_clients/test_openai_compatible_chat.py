from unittest.mock import MagicMock, Mock, patch

from pydantic import BaseModel

from prkit.prkit_core.model_clients.openai_compatible_chat import (
    OpenAICompatibleChatModel,
)


class DummyChatProvider(OpenAICompatibleChatModel):
    provider_name = "dummy"
    provider_prefix = "dummy"
    api_key_env_var = "DUMMY_API_KEY"
    base_url_env_var = "DUMMY_BASE_URL"
    default_base_url = "https://dummy.example/v1"

    def get_client_kwargs(self):
        return {"timeout": 10}


class TestOpenAICompatibleChatModel:
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    def test_init_normalizes_prefixed_model_and_uses_env_base_url(
        self, mock_openai_class, _mock_load_project_dotenv
    ):
        mock_openai_class.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {"DUMMY_API_KEY": "test-key", "DUMMY_BASE_URL": "https://override/v1"},
            clear=True,
        ):
            client = DummyChatProvider("dummy/model-a")

        assert client.model == "model-a"
        assert client.provider == "dummy"
        assert client.base_url == "https://override/v1"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://override/v1",
            timeout=10,
        )

    def test_build_message_content_without_images_returns_plain_prompt(self):
        client = object.__new__(DummyChatProvider)
        assert client._build_message_content("hello") == "hello"

    @patch(
        "prkit.prkit_core.model_clients.openai_compatible_chat.prepare_image_url_from_image_path"
    )
    def test_build_message_content_with_images_uses_image_urls(self, mock_prepare):
        mock_prepare.side_effect = ["data:image/png;base64,abc", "https://example/image.jpg"]
        client = object.__new__(DummyChatProvider)

        content = client._build_message_content("describe", ["a.png", "b.jpg"])

        assert content == [
            {"type": "text", "text": "describe"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example/image.jpg"},
            },
        ]

    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    def test_chat_forwards_structured_output_and_max_tokens(
        self, mock_openai_class, _mock_load_project_dotenv
    ):
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

        with patch.dict("os.environ", {"DUMMY_API_KEY": "test-key"}, clear=True):
            client = DummyChatProvider("model-a")

        response = client.chat(
            "Return JSON only.",
            response_format=ExampleResponse,
            max_output_tokens=123,
            temperature=0.2,
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "model-a"
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Return JSON only."}
        ]
        assert call_kwargs["max_tokens"] == 123
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "ExampleResponse"
