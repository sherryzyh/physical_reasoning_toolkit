"""
Tests for DashScope model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.dashscope import (
    DEFAULT_DASHSCOPE_TIMEOUT_SECONDS,
    DashscopeModel,
    _parse_bool_env,
    resolve_dashscope_base_url,
    resolve_dashscope_max_retries,
    resolve_dashscope_timeout_seconds,
)
from prkit.core.model_clients.retry import DEFAULT_MAX_RETRIES

DASHSCOPE_TEST_MODEL = "qwen3.6-plus"
SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_INSTRUCTIONS}


class TestDashscopeModel:
    """Test cases for DashscopeModel."""

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_uses_us_default_base_url(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """Test initializing DashScope model with the default US endpoint."""
        mock_openai_class.return_value = MagicMock()

        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            client = DashscopeModel(DASHSCOPE_TEST_MODEL)

        assert client.model == DASHSCOPE_TEST_MODEL
        assert client.provider == "dashscope"
        assert client.base_url == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            timeout=DEFAULT_DASHSCOPE_TIMEOUT_SECONDS,
            max_retries=DEFAULT_MAX_RETRIES,
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

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
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
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=DASHSCOPE_TEST_MODEL,
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": "Hello, world!"}],
            extra_body={"enable_thinking": False},
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_chat_structured_output_uses_json_object_and_a_schema_prompt(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """DashScope gets json_object plus the schema as prose, with thinking disabled.

        Verified against the live API: a json_schema response format is accepted
        and then ignored — the same request returned different key names on
        consecutive attempts for a two-field schema. DashScope also rejects any
        response_format unless the word "json" appears in the messages, which is
        why the schema has to reach it through the prompt.
        """

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
        response = client.response(
            "Return JSON only.",
            response_format=ExampleResponse,
            max_output_tokens=256,
        )

        assert response == '{"answer":"ok"}'
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == DASHSCOPE_TEST_MODEL
        assert call_kwargs["messages"][0] == SYSTEM_MESSAGE
        user_content = call_kwargs["messages"][1]["content"]
        assert user_content.startswith("Return JSON only.")
        assert "JSON Schema" in user_content
        assert "json" in user_content.lower(), "DashScope 400s without it"
        assert call_kwargs["extra_body"] == {"enable_thinking": False}
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
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
        response = client.response(
            "Hello, world!",
            extra_body={"enable_thinking": True, "thinking_budget": 2048},
        )

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=DASHSCOPE_TEST_MODEL,
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": "Hello, world!"}],
            extra_body={"enable_thinking": True, "thinking_budget": 2048},
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_chat_rejects_structured_output_in_thinking_mode(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        class ExampleResponse(BaseModel):
            answer: str

        mock_openai_class.return_value = MagicMock()
        client = DashscopeModel(DASHSCOPE_TEST_MODEL)

        with pytest.raises(ValueError, match="not supported in thinking mode"):
            client.response(
                "Return JSON only.",
                response_format=ExampleResponse,
                extra_body={"enable_thinking": True},
            )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_chat_rejects_non_dict_extra_body(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        mock_openai_class.return_value = MagicMock()
        client = DashscopeModel("custom-model")

        with pytest.raises(TypeError, match="must be a dict"):
            client.response("Hello", extra_body="bad")

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
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
            response = client.response("Hello")

        assert response == "ok"
        mock_client.chat.completions.create.assert_called_once_with(
            model="custom-model",
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": "Hello"}],
            extra_body={"enable_thinking": True},
        )


class TestDashscopeStructuredOutputPlan:
    @staticmethod
    def _client():
        client = object.__new__(DashscopeModel)
        client.model = DASHSCOPE_TEST_MODEL
        client.provider = "dashscope"
        client.logger = MagicMock()
        return client

    def test_plan_does_not_claim_enforcement_dashscope_does_not_provide(self):
        class ExampleResponse(BaseModel):
            answer: str

        plan = self._client().resolve_structured_output_plan(ExampleResponse)

        assert plan.mode == "json_object"
        assert plan.native_schema_enforced is False
        assert plan.response_format == {"type": "json_object"}
        assert plan.prompt_suffix and "JSON Schema" in plan.prompt_suffix

    def test_native_required_raises(self):
        class ExampleResponse(BaseModel):
            answer: str

        with pytest.raises(ValueError, match="does not.*enforce"):
            self._client().resolve_structured_output_plan(
                ExampleResponse, structured_policy="native_required"
            )
