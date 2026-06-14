"""
Tests for xAI model client.
"""

from unittest.mock import MagicMock, Mock, patch

from pydantic import BaseModel, Field

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.structured_output import coerce_structured_output_spec
from prkit.core.model_clients.xai import XAIModel

XAI_TEST_MODEL = "grok-4-1-fast-reasoning"
SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_INSTRUCTIONS}


class TestXAIModel:
    """Test cases for XAIModel."""

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init(self, _mock_load_project_dotenv, mock_openai_class):
        """Test initializing xAI model."""
        mock_openai_class.return_value = MagicMock()

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=True):
            client = XAIModel(XAI_TEST_MODEL)

        assert client.model == XAI_TEST_MODEL
        assert client.provider == "xai"
        assert client.base_url == "https://api.x.ai/v1"
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.x.ai/v1",
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_chat_text_only(self, _mock_load_project_dotenv, mock_openai_class):
        """Test xAI chat with text-only input."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        client = XAIModel(XAI_TEST_MODEL)
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=XAI_TEST_MODEL,
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": "Hello, world!"}],
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_resolve_structured_output_plan_strips_unsupported_constraint_keywords(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        class ExampleResponse(BaseModel):
            answer: str = Field(min_length=1, max_length=8)
            tags: list[str] = Field(min_length=1, max_length=3)

        mock_openai_class.return_value = MagicMock()
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=True):
            client = XAIModel(XAI_TEST_MODEL)

        plan = client.resolve_structured_output_plan(
            ExampleResponse,
            structured_policy="native_required",
        )
        schema = plan.response_format["schema"]

        assert plan.mode == "json_schema"
        assert "minLength" not in str(schema)
        assert "maxLength" not in str(schema)
        assert "minItems" not in str(schema)
        assert "maxItems" not in str(schema)

    def test_resolve_structured_output_plan_falls_back_when_schema_uses_allof(self):
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()
        spec = coerce_structured_output_spec(
            {
                "type": "json_schema",
                "name": "AllOfResponse",
                "schema": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        }
                    ]
                },
            }
        )

        plan = client._resolve_structured_output_plan(
            spec,
            structured_policy="best_effort",
        )

        assert plan.mode == "prompt_only"
        assert plan.strategy == "xai_prompt_only_unsupported_schema"
