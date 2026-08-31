"""
Tests for xAI model client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.structured_output import coerce_structured_output_spec
from prkit.core.model_clients.xai import XAIModel

XAI_TEST_MODEL = "grok-4.6"
SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_INSTRUCTIONS}


class TestXAIModel:
    """Test cases for XAIModel."""

    @staticmethod
    def _stub_client():
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()
        return client

    @staticmethod
    def _closed(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
            "additionalProperties": False,
        }

    @staticmethod
    def _spec(schema):
        return coerce_structured_output_spec(
            {"type": "json_schema", "name": "Example", "schema": schema}
        )

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

    @patch(
        "prkit.core.model_clients.openai_compatible_chat.prepare_image_url_from_image_path"
    )
    def test_images_use_the_chat_completions_block_shape(self, mock_prepare):
        """xAI's input_image block is Responses-API only; chat completions 422s on it."""
        mock_prepare.side_effect = ["data:image/png;base64,abc"]
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()

        content = client._build_message_content("describe", ["a.png"])

        assert content == [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]

    def test_unsupported_image_media_type_logs_a_warning(self):
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()

        block = client._build_image_content_block("data:image/gif;base64,abc")

        assert block == {
            "type": "image_url",
            "image_url": {"url": "data:image/gif;base64,abc"},
        }
        client.logger.warning.assert_called_once()
        assert "image/gif" in client.logger.warning.call_args[0]

    def test_supported_image_media_type_does_not_warn(self):
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()

        client._build_image_content_block("data:image/jpeg;base64,abc")

        client.logger.warning.assert_not_called()

    def test_remote_image_url_is_not_media_type_checked(self):
        """The media type behind an http(s) URL is unknowable, so do not guess."""
        client = object.__new__(XAIModel)
        client.model = XAI_TEST_MODEL
        client.provider = "xai"
        client.logger = MagicMock()

        block = client._build_image_content_block("https://example.com/a.gif")

        assert block == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.gif"},
        }
        client.logger.warning.assert_not_called()

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_length_and_item_constraints_survive(
        self, _mock_load_project_dotenv, mock_openai_class
    ):
        """xAI enforces these up to documented thresholds, so keep them."""

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
        schema = str(plan.response_format["schema"])

        assert plan.mode == "json_schema"
        assert "minLength" in schema
        assert "maxLength" in schema
        assert "minItems" in schema
        assert "maxItems" in schema

    def test_contains_bounds_are_stripped(self):
        client = self._stub_client()
        spec = self._spec(
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "contains": {"type": "string"},
                        "minContains": 1,
                        "maxContains": 3,
                    }
                },
                "required": ["tags"],
                "additionalProperties": False,
            }
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"
        assert "minContains" not in str(plan.response_format["schema"])
        assert "maxContains" not in str(plan.response_format["schema"])

    def test_single_subschema_allof_is_native(self):
        """xAI supports allOf in its single-subschema form."""
        client = self._stub_client()
        spec = self._spec({"allOf": [self._closed({"answer": {"type": "string"}})]})

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"
        assert plan.strategy == "xai_chat_json_schema"

    def test_multi_subschema_allof_stays_native(self):
        """xAI enforces this best-effort rather than rejecting it.

        Demoting would forfeit native enforcement of the whole schema to avoid
        a construct that does not actually fail.
        """
        client = self._stub_client()
        spec = self._spec(
            {
                "allOf": [
                    self._closed({"answer": {"type": "string"}}),
                    self._closed({"score": {"type": "number"}}),
                ]
            }
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_instance_data_is_not_read_as_schema(self):
        """A ``default`` holding a key named ``items`` is a value, not a keyword."""
        client = self._stub_client()
        spec = self._spec(
            self._closed({"cfg": {"type": "object", "default": {"items": ["a", "b"]}}})
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_enum_members_are_not_walked_as_schema(self):
        client = self._stub_client()
        spec = self._spec(self._closed({"k": {"enum": [{"enum": []}]}}))

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_circular_ref_schema_falls_back(self):
        client = self._stub_client()
        spec = self._spec(
            {
                "$defs": {"Node": self._closed({"child": {"$ref": "#/$defs/Node"}})},
                **self._closed({"root": {"$ref": "#/$defs/Node"}}),
            }
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "prompt_only"

    def test_reused_ref_stays_native(self):
        """Reuse is not recursion; only a genuine cycle should demote."""
        client = self._stub_client()
        spec = self._spec(
            {
                "$defs": {"Item": self._closed({"name": {"type": "string"}})},
                **self._closed(
                    {
                        "a": {"$ref": "#/$defs/Item"},
                        "b": {"$ref": "#/$defs/Item"},
                    }
                ),
            }
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_array_items_falls_back(self):
        client = self._stub_client()
        spec = self._spec(
            self._closed(
                {
                    "pair": {
                        "type": "array",
                        "items": [{"type": "string"}, {"type": "integer"}],
                    }
                }
            )
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "prompt_only"

    def test_empty_enum_falls_back(self):
        client = self._stub_client()
        spec = self._spec(self._closed({"choice": {"enum": []}}))

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "prompt_only"

    def test_boolean_property_schema_falls_back(self):
        client = self._stub_client()
        spec = self._spec(
            {
                "type": "object",
                "properties": {"anything": True},
                "required": ["anything"],
                "additionalProperties": False,
            }
        )

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "prompt_only"

    def test_additional_properties_false_is_not_a_boolean_subschema(self):
        """Every provider transform emits this; it must not trip the gate."""
        client = self._stub_client()
        spec = self._spec(self._closed({"answer": {"type": "string"}}))

        plan = client._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_native_required_raises_on_circular_schema(self):
        client = self._stub_client()
        spec = self._spec(
            {
                "$defs": {"Node": self._closed({"child": {"$ref": "#/$defs/Node"}})},
                **self._closed({"root": {"$ref": "#/$defs/Node"}}),
            }
        )

        with pytest.raises(ValueError, match="circular"):
            client._resolve_structured_output_plan(
                spec, structured_policy="native_required"
            )

    def test_reasoning_params_are_dropped_but_temperature_is_kept(self):
        client = self._stub_client()

        result = client._apply_param_omissions(
            {
                "model": XAI_TEST_MODEL,
                "presence_penalty": 1,
                "frequency_penalty": 1,
                "stop": ["x"],
                "temperature": 0.2,
            }
        )

        assert result == {"model": XAI_TEST_MODEL, "temperature": 0.2}
