"""
Tests for DashScope model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.prkit_core.model_clients.dashscope import (
    DEFAULT_DASHSCOPE_TIMEOUT_SECONDS,
    DashscopeModel,
    _parse_bool_env,
    resolve_dashscope_base_url,
    resolve_dashscope_max_retries,
    resolve_dashscope_timeout_seconds,
)

DASHSCOPE_TEST_MODEL = "qwen3.6-plus"


class TestDashscopeModel:
    """Test cases for DashscopeModel."""

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_init_uses_us_default_base_url(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """Test initializing DashScope model with the default US endpoint."""
        mock_openai_class.return_value = MagicMock()

        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            client = DashscopeModel(DASHSCOPE_TEST_MODEL)

        assert client.model == DASHSCOPE_TEST_MODEL
        assert client.provider == "dashscope"
        assert (
            client.base_url
            == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
        )
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            timeout=DEFAULT_DASHSCOPE_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def test_resolve_dashscope_base_url_respects_region(self):
        """Test region-based DashScope endpoint selection."""
        with patch.dict("os.environ", {"DASHSCOPE_REGION": "singapore"}, clear=True):
            assert (
                resolve_dashscope_base_url()
                == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

    def test_dashscope_env_helpers(self):
        with patch.dict(
            "os.environ",
            {
                "FLAG_TRUE": " yes ",
                "FLAG_FALSE": "0",
                "DASHSCOPE_TIMEOUT_SECONDS": "12.5",
                "DASHSCOPE_MAX_RETRIES": "3",
                "DASHSCOPE_BASE_URL": "https://example.invalid/v1",
            },
            clear=True,
        ):
            assert _parse_bool_env("FLAG_TRUE") is True
            assert _parse_bool_env("FLAG_FALSE") is False
            assert resolve_dashscope_timeout_seconds() == 12.5
            assert resolve_dashscope_max_retries() == 3
            assert resolve_dashscope_base_url() == "https://example.invalid/v1"

    def test_dashscope_bool_env_rejects_invalid_values(self):
        with patch.dict("os.environ", {"FLAG": "maybe"}, clear=True):
            with pytest.raises(ValueError, match="must be one of"):
                _parse_bool_env("FLAG")

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_chat_text_only(self, _mock_load_project_dotenv, mock_openai_class):
        """Test DashScope chat with text-only input."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DashscopeModel(DASHSCOPE_TEST_MODEL)
        response = client.chat("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=DASHSCOPE_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello, world!"}],
            extra_body={"enable_thinking": False},
        )

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_chat_structured_output_uses_json_schema(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """DashScope structured output should use native json_schema with thinking disabled."""

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

        client = DashscopeModel(DASHSCOPE_TEST_MODEL)
        response = client.chat(
            "Return JSON only.",
            response_format=ExampleResponse,
            max_output_tokens=256,
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == DASHSCOPE_TEST_MODEL
        assert call_kwargs["messages"] == [{"role": "user", "content": "Return JSON only."}]
        assert call_kwargs["extra_body"] == {"enable_thinking": False}
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "ExampleResponse"

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_chat_respects_explicit_extra_body(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """Explicit caller settings should win over DashScope defaults."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = DashscopeModel(DASHSCOPE_TEST_MODEL)
        response = client.chat(
            "Hello, world!",
            extra_body={"enable_thinking": True, "thinking_budget": 2048},
        )

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=DASHSCOPE_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello, world!"}],
            extra_body={"enable_thinking": True, "thinking_budget": 2048},
        )

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_chat_rejects_structured_output_in_thinking_mode(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        class ExampleResponse(BaseModel):
            answer: str

        mock_openai_class.return_value = MagicMock()
        client = DashscopeModel(DASHSCOPE_TEST_MODEL)

        with pytest.raises(ValueError, match="not supported in thinking mode"):
            client.chat(
                "Return JSON only.",
                response_format=ExampleResponse,
                extra_body={"enable_thinking": True},
            )

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_chat_rejects_non_dict_extra_body(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        mock_openai_class.return_value = MagicMock()
        client = DashscopeModel("custom-model")

        with pytest.raises(TypeError, match="must be a dict"):
            client.chat("Hello", extra_body="bad")

    @patch("prkit.prkit_core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.prkit_core.model_clients.base.load_project_dotenv")
    def test_default_enable_thinking_respects_env_override(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "ok"
        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"DASHSCOPE_API_KEY": "test-key", "DASHSCOPE_ENABLE_THINKING": "true"},
            clear=True,
        ):
            client = DashscopeModel("custom-model")
            response = client.chat("Hello")

        assert response == "ok"
        mock_client.chat.completions.create.assert_called_once_with(
            model="custom-model",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"enable_thinking": True},
        )
