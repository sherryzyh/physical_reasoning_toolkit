from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.anthropic import AnthropicModel
from prkit.core.model_clients.dashscope import DashscopeModel
from prkit.core.model_clients.deepseek import DeepseekModel
from prkit.core.model_clients.gemini import GeminiModel
from prkit.core.model_clients.ollama import OllamaModel
from prkit.core.model_clients.openai import (
    OpenAIModel,
    ensure_openai_strict_json_schema,
)
from prkit.core.model_clients.structured_output import (
    StructuredCallResult,
    coerce_structured_output_spec,
    extract_json_object,
    extract_json_payload,
    extract_schema_for_gemini,
    normalize_response_format,
    strip_schema_keywords,
)
from prkit.core.model_clients.xai import XAIModel
from prkit.semantics.build.strict_models import (
    StrictPredictionFinalAnswerResponse,
    StrictPredictionSemanticsResponse,
)


class TestStructuredOutputUtilities:
    def test_openai_strict_schema_marks_objects_required(self):
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "meta": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                },
            },
        }

        strict_schema = ensure_openai_strict_json_schema(schema)

        assert strict_schema["additionalProperties"] is False
        assert strict_schema["required"] == ["answer", "meta"]
        assert strict_schema["properties"]["meta"]["additionalProperties"] is False
        assert strict_schema["properties"]["meta"]["required"] == ["score"]

    def test_normalize_response_format_from_dict(self):
        normalized = normalize_response_format(
            {
                "type": "json_schema",
                "name": "Example",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "description": "example schema",
            }
        )

        assert normalized["type"] == "json_schema"
        assert normalized["name"] == "Example"
        assert normalized["strict"] is True
        assert normalized["description"] == "example schema"
        assert "additionalProperties" not in normalized["schema"]

    def test_normalize_response_format_from_pydantic_model(self):
        class ResponseSchema(BaseModel):
            answer: str
            notes: str | None = None

        normalized = normalize_response_format(ResponseSchema)

        assert normalized["type"] == "json_schema"
        assert normalized["name"] == "ResponseSchema"
        assert normalized["strict"] is True
        assert normalized["schema"]["type"] == "object"

    def test_openai_strict_schema_strips_ref_siblings(self):
        schema = {
            "$defs": {
                "Choice": {
                    "type": "string",
                    "enum": ["a", "b"],
                }
            },
            "type": "object",
            "properties": {
                "choice": {
                    "$ref": "#/$defs/Choice",
                    "description": "user choice",
                }
            },
        }

        strict_schema = ensure_openai_strict_json_schema(schema)

        assert strict_schema["properties"]["choice"] == {"$ref": "#/$defs/Choice"}

    def test_normalize_response_format_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="type='json_schema'"):
            normalize_response_format({"type": "text", "name": "bad", "schema": {}})

        with pytest.raises(ValueError, match="contain 'schema'"):
            normalize_response_format({"type": "json_schema", "name": "bad"})

        with pytest.raises(ValueError, match="Pydantic BaseModel"):
            normalize_response_format(object())

    def test_extract_schema_for_gemini_returns_schema_only(self):
        normalized = {
            "type": "json_schema",
            "name": "Example",
            "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            "strict": True,
        }

        assert extract_schema_for_gemini(normalized) == normalized["schema"]

    def test_coerce_structured_output_spec_collects_schema_features(self):
        class ResponseSchema(BaseModel):
            answer: str
            notes: str | None = None

        spec = coerce_structured_output_spec(ResponseSchema)

        assert spec.name == "ResponseSchema"
        assert spec.source_model is ResponseSchema
        assert spec.schema_features is not None
        assert spec.schema_features.optional_field_count >= 1

    def test_extract_json_payload_recovers_from_fenced_response(self):
        payload = extract_json_payload('```json\n{"answer": "x"}\n```')

        assert payload == {"answer": "x"}
        assert extract_json_object('prefix\n{"answer": "x"}\nsuffix') == {"answer": "x"}

    def test_strip_schema_keywords_removes_nested_entries(self):
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
        }

        stripped = strip_schema_keywords(schema, keywords={"minLength", "minItems"})

        assert "minLength" not in str(stripped)
        assert "minItems" not in str(stripped)

    def test_strict_prediction_schema_provider_plan_matrix(self):
        def stub_client(cls, *, model: str, provider: str):
            client = object.__new__(cls)
            client.model = model
            client.provider = provider
            client.logger = MagicMock()
            return client

        plans = {
            "openai": stub_client(OpenAIModel, model="gpt-5.4-mini", provider="openai"),
            "gemini": stub_client(
                GeminiModel, model="gemini-2.5-pro", provider="google"
            ),
            "xai": stub_client(XAIModel, model="grok-4.20-reasoning", provider="xai"),
            "dashscope": stub_client(
                DashscopeModel,
                model="qwen3.6-plus",
                provider="dashscope",
            ),
            "ollama": stub_client(OllamaModel, model="gpt-oss", provider="ollama"),
            "deepseek": stub_client(
                DeepseekModel,
                model="deepseek-chat",
                provider="deepseek",
            ),
            "anthropic": stub_client(
                AnthropicModel,
                model="claude-sonnet-4-6",
                provider="anthropic",
            ),
        }

        resolved = {
            name: client.resolve_structured_output_plan(
                StrictPredictionSemanticsResponse,
                structured_policy="best_effort",
            )
            for name, client in plans.items()
        }

        assert resolved["openai"].mode == "json_schema"
        assert resolved["gemini"].mode == "json_schema"
        assert resolved["xai"].mode == "json_schema"
        assert resolved["dashscope"].mode == "json_schema"
        assert resolved["ollama"].mode == "json_schema"
        assert resolved["deepseek"].mode == "json_object"
        assert resolved["anthropic"].mode == "prompt_only"

        anthropic_compact = plans["anthropic"].resolve_structured_output_plan(
            StrictPredictionFinalAnswerResponse,
            structured_policy="best_effort",
        )
        assert anthropic_compact.mode == "json_schema"

    def test_structured_call_result_require_parsed_raises(self):
        result = StructuredCallResult(
            parsed=None,
            raw_text="{}",
            raw_payload={},
            validation_error="boom",
            structured_output_mode="json_schema",
            structured_output_strategy="openai_responses_json_schema",
            native_schema_enforced=True,
            provider="openai",
            model_name="gpt-5.4-mini",
        )

        with pytest.raises(ValueError, match="boom"):
            result.require_parsed()
