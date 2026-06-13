"""
DeepSeek API client implementation.
"""

from __future__ import annotations

import json
from typing import Any

from .openai_compatible_chat import OpenAICompatibleChatModel
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
)


class DeepseekModel(OpenAICompatibleChatModel):
    """DeepSeek API client implementation via the OpenAI-compatible Chat API."""

    provider_name = "deepseek"
    provider_prefix = "deepseek"
    api_key_env_var = "DEEPSEEK_API_KEY"
    base_url_env_var = "DEEPSEEK_BASE_URL"
    default_base_url = "https://api.deepseek.com"
    supports_response_format_json_schema = False
    supports_response_format_json_object = True

    def _build_message_content(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str | list[dict[str, Any]]:
        if image_paths:
            self.logger.warning(
                "DeepSeek model %s does not support image inputs. "
                "Received %s image(s) which will be ignored.",
                self.model,
                len(image_paths),
            )
        return user_prompt

    def _structured_prompt_for_chat(
        self,
        user_prompt: str,
        response_format: dict | type | None,
    ) -> str:
        if response_format is None:
            return user_prompt
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            return user_prompt

        spec = StructuredOutputSpec(
            name="response",
            schema={},
            schema_features=None,
        )
        try:
            from .structured_output import coerce_structured_output_spec

            spec = coerce_structured_output_spec(response_format)
        except Exception:
            return user_prompt

        schema_text = json.dumps(spec.schema, indent=2, ensure_ascii=False)
        example_keys = list((spec.schema.get("properties") or {}).keys())[:3]
        example = {key: None for key in example_keys}
        return (
            user_prompt
            + "\n\nReturn ONLY JSON. The response must be valid JSON and match this JSON Schema exactly.\n"
            + schema_text
            + "\nExample shape:\n"
            + json.dumps(example, ensure_ascii=False)
        )

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        if structured_policy == "native_required":
            raise ValueError(
                "DeepSeek does not provide native schema-enforced structured output. "
                f"Got model={self.model!r}."
            )
        return StructuredOutputPlan(
            mode="json_object",
            strategy="deepseek_json_object",
            native_schema_enforced=False,
            accepted_artifact_modes=("json_object", "prompt_only"),
            accepted_artifact_strategies=("deepseek_json_object",),
            response_format={"type": "json_object"},
            prompt_suffix=(
                "\n\nReturn ONLY JSON that matches this JSON Schema exactly.\n"
                "Do not include markdown fences or commentary.\n"
                + json.dumps(spec.schema, indent=2, ensure_ascii=False)
            ),
        )

    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict | type | None = None,
        **kwargs: Any,
    ) -> str:
        return super().chat(
            user_prompt=user_prompt,
            image_paths=image_paths,
            response_format=response_format,
            **kwargs,
        )
