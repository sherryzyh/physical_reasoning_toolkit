"""
Tests for Gemini model client.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.gemini import (
    _GEMINI_IGNORABLE_SCHEMA_KEYWORDS,
    _GEMINI_SUPPORTED_SCHEMA_KEYWORDS,
    GeminiModel,
    _extract_gemini_error_details,
)
from prkit.core.model_clients.retry import DEFAULT_MAX_RETRIES
from prkit.core.model_clients.structured_output import coerce_structured_output_spec

GEMINI_TEST_MODEL = "gemini-2.5-pro"


def _assert_client_call(mock_genai, *, api_key: str | None) -> None:
    """Assert how genai.Client was constructed, including the retry options.

    Retry options are asserted by value rather than by comparing an opaque
    HttpOptions object, so a change to ``attempts`` is legible in the failure.
    """
    mock_genai.Client.assert_called_once()
    kwargs = mock_genai.Client.call_args.kwargs
    assert kwargs.get("api_key") == api_key
    retry = kwargs["http_options"].retry_options
    assert retry.attempts == DEFAULT_MAX_RETRIES + 1


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
        _assert_client_call(mock_genai, api_key="test-key")

    @patch("prkit.core.model_clients.gemini.genai")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_with_google_api_key_fallback(
        self, _mock_load_project_dotenv, mock_genai
    ):
        """Test initializing with GOOGLE_API_KEY as fallback."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            GeminiModel(GEMINI_TEST_MODEL)

        _assert_client_call(mock_genai, api_key="google-key")

    @patch("prkit.core.model_clients.gemini.genai")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_init_without_api_key(self, _mock_load_project_dotenv, mock_genai):
        """Test initializing without API key (uses default)."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            GeminiModel(GEMINI_TEST_MODEL)

        _assert_client_call(mock_genai, api_key=None)

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_text_only(self, mock_genai):
        """Test chat with text-only input."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        response = client.response("Hello, world!")

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
        # Default system prompt is sent via Gemini's system_instruction config.
        assert config.system_instruction == DEFAULT_INSTRUCTIONS

    @patch("prkit.core.model_clients.gemini.genai")
    def test_response_system_instruction_explicit_and_suppressed(self, mock_genai):
        """Explicit instructions override the default; empty string suppresses it."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "ok"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)

        client.response("Hi", instructions="Custom system.")
        config = mock_client.models.generate_content.call_args[1]["config"]
        assert config.system_instruction == "Custom system."

        client.response("Hi", instructions="")
        config = mock_client.models.generate_content.call_args[1]["config"]
        assert config.system_instruction is None

    @patch("prkit.core.model_clients.gemini.genai")
    def test_chat_with_kwargs(self, mock_genai):
        """Test chat with additional kwargs for config."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = Mock()
        mock_response.text = "Response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiModel(GEMINI_TEST_MODEL)
        client.response("Hello", temperature=0.7)

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
        client.response("Hello")

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
            response = client.response("Hello", image_paths=["image.jpg"])

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
        response = client.response(
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
    def test_chat_logs_failed_image_open(
        self, mock_genai, mock_exists, mock_image_open
    ):
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
            response = client.response("Hello", image_paths=["/tmp/bad.png"])

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
            client.response("Hello")

    def test_extract_gemini_error_details_handles_empty_and_prompt_blocks(self):
        empty = SimpleNamespace(prompt_feedback=None, candidates=None)
        blocked = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="BLOCKED"),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

        assert _extract_gemini_error_details(empty) == "empty_response"
        assert _extract_gemini_error_details(blocked) == "prompt_block_reason=BLOCKED"


class TestGeminiStructuredOutputGate:
    """Gemini rejects some schema constructs; demote rather than send a 400."""

    @staticmethod
    def _client():
        client = object.__new__(GeminiModel)
        client.model = "gemini-3.5-flash"
        client.provider = "google"
        client.logger = MagicMock()
        return client

    @staticmethod
    def _spec(schema):
        return coerce_structured_output_spec(
            {"type": "json_schema", "name": "Example", "schema": schema}
        )

    @staticmethod
    def _closed(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
            "additionalProperties": False,
        }

    def test_plain_schema_stays_native(self):
        plan = self._client()._resolve_structured_output_plan(
            self._spec(self._closed({"answer": {"type": "string"}})),
            structured_policy="best_effort",
        )

        assert plan.mode == "json_schema"
        assert plan.native_schema_enforced is True

    @pytest.mark.parametrize(
        "schema",
        [
            {"allOf": [{"type": "object"}, {"type": "object"}]},
            {"type": "object", "properties": {"a": {"not": {"type": "null"}}}},
            {"type": "object", "properties": {"a": {"const": 3}}},
            {"type": "object", "propertyNames": {"type": "string"}},
            {
                "type": "object",
                "properties": {"a": {"type": "number", "multipleOf": 2}},
            },
            {"type": "object", "properties": {"a": {"exclusiveMinimum": 0}}},
            {
                "type": "object",
                "properties": {"a": {"type": "string", "pattern": "^x"}},
            },
            {
                "type": "object",
                "properties": {"a": {"type": "string", "minLength": 1}},
            },
            {"type": "object", "properties": {"a": {"uniqueItems": True}}},
        ],
    )
    def test_unsupported_keywords_demote(self, schema):
        plan = self._client()._resolve_structured_output_plan(
            self._spec(schema), structured_policy="best_effort"
        )

        assert plan.mode == "prompt_only"
        assert plan.strategy == "gemini_prompt_only_unsupported_schema"
        assert plan.prompt_suffix and "JSON Schema" in plan.prompt_suffix

    def test_native_required_raises_on_an_unsupported_schema(self):
        with pytest.raises(ValueError, match="unsupported schema keyword"):
            self._client()._resolve_structured_output_plan(
                self._spec({"allOf": [{"type": "object"}, {"type": "object"}]}),
                structured_policy="native_required",
            )

    def test_prefix_items_stays_native(self):
        """``prefixItems`` is on Google's allowlist; gating it was a false negative."""
        plan = self._client()._resolve_structured_output_plan(
            self._spec(
                {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "array",
                            "prefixItems": [{"type": "string"}, {"type": "integer"}],
                        }
                    },
                }
            ),
            structured_policy="best_effort",
        )

        assert plan.mode == "json_schema"

    def test_the_allowlist_is_the_documented_one(self):
        """Pins the allowlist against Google's published set, verbatim.

        Transcribed from the v1beta REST discovery document, revision 20260830,
        and the ``generative_service.proto`` comment it is generated from. If
        someone widens or trims this set, they should have re-read the source.
        """
        assert _GEMINI_SUPPORTED_SCHEMA_KEYWORDS == {
            "$id",
            "$defs",
            "$ref",
            "$anchor",
            "type",
            "format",
            "title",
            "description",
            "enum",
            "items",
            "prefixItems",
            "minItems",
            "maxItems",
            "minimum",
            "maximum",
            "anyOf",
            "oneOf",
            "properties",
            "additionalProperties",
            "required",
            "propertyOrdering",
        }

    def test_annotations_are_never_treated_as_constraints(self):
        """These are off the allowlist but validate nothing, so losing them is free."""
        for keyword in _GEMINI_IGNORABLE_SCHEMA_KEYWORDS:
            plan = self._client()._resolve_structured_output_plan(
                self._spec(
                    {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        keyword: 1,
                    }
                ),
                structured_policy="best_effort",
            )
            assert plan.mode == "json_schema", keyword

    def test_default_values_do_not_demote(self):
        """`default` is outside Google's allowlist but 32 of 44 prkit models use it.

        Those are sent natively today and work, so the allowlist is not
        exhaustive in practice and annotation keywords must not gate.
        """
        plan = self._client()._resolve_structured_output_plan(
            self._spec(self._closed({"a": {"type": "string", "default": "x"}})),
            structured_policy="best_effort",
        )

        assert plan.mode == "json_schema"

    def test_ref_siblings_and_cycles_do_not_demote(self):
        """Documented as unsupported, deliberately not gated — see the module note."""
        schema = {
            "$defs": {
                "Node": self._closed({"child": {"$ref": "#/$defs/Node"}}),
            },
            **self._closed(
                {"root": {"$ref": "#/$defs/Node", "description": "a sibling key"}}
            ),
        }

        plan = self._client()._resolve_structured_output_plan(
            self._spec(schema), structured_policy="best_effort"
        )

        assert plan.mode == "json_schema"

    def test_a_field_named_like_a_keyword_does_not_demote(self):
        plan = self._client()._resolve_structured_output_plan(
            self._spec(self._closed({"pattern": {"type": "string"}})),
            structured_policy="best_effort",
        )

        assert plan.mode == "json_schema"
