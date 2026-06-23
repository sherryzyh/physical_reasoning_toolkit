"""
xAI API client implementation.
"""

from __future__ import annotations

from typing import Any

from .openai_compatible_chat import OpenAICompatibleChatModel
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    normalize_response_format,
    schema_contains_keyword,
    strip_schema_keywords,
)

_XAI_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
    }
)


def _xai_native_schema_incompatibility(spec: StructuredOutputSpec) -> str | None:
    """Return a reason string when *spec* is incompatible with xAI native structured output, or ``None``."""
    if schema_contains_keyword(spec.schema, "allOf"):
        return "allOf is not supported by xAI structured outputs"
    return None


def _xai_transformed_response_format(spec: StructuredOutputSpec) -> dict[str, Any]:
    """Build the response format dict after stripping xAI-unsupported schema keywords."""
    transformed_schema = strip_schema_keywords(
        spec.schema,
        keywords=_XAI_UNSUPPORTED_SCHEMA_KEYWORDS,
    )
    return normalize_response_format(
        {
            "type": "json_schema",
            "name": spec.name,
            "schema": transformed_schema,
            "strict": spec.strict,
            "description": spec.description,
        }
    )


class XAIModel(OpenAICompatibleChatModel):
    """xAI Grok client via the OpenAI-compatible Chat Completions API."""

    provider_name = "xai"
    provider_prefix = "xai"
    api_key_env_var = "XAI_API_KEY"
    base_url_env_var = "XAI_BASE_URL"
    default_base_url = "https://api.x.ai/v1"

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        incompatibility = _xai_native_schema_incompatibility(spec)
        if incompatibility is not None:
            if structured_policy == "native_required":
                raise ValueError(
                    "xAI structured outputs do not accept this schema "
                    f"for model={self.model!r}: {incompatibility}."
                )
            return StructuredOutputPlan(
                mode="prompt_only",
                strategy="xai_prompt_only_unsupported_schema",
                native_schema_enforced=False,
                accepted_artifact_modes=("prompt_only",),
                accepted_artifact_strategies=("xai_prompt_only_unsupported_schema",),
                response_format=None,
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        return StructuredOutputPlan(
            mode="json_schema",
            strategy="xai_chat_json_schema",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema",),
            accepted_artifact_strategies=("xai_chat_json_schema",),
            response_format=_xai_transformed_response_format(spec),
        )
