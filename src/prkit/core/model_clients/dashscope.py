"""
DashScope API client implementation.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible_chat import OpenAICompatibleChatModel
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    normalize_response_format,
)

_DASHSCOPE_BASE_URLS = {
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "hk": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    "hongkong": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    "hong-kong": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "sg": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}

DEFAULT_DASHSCOPE_TIMEOUT_SECONDS = 60.0
DEFAULT_DASHSCOPE_MAX_RETRIES = 0


def _parse_bool_env(name: str) -> bool | None:
    """Parse a boolean environment variable, returning ``None`` when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of "
        "'true', 'false', '1', '0', 'yes', or 'no'."
    )


def resolve_dashscope_timeout_seconds() -> float:
    """Resolve the request timeout for DashScope OpenAI-compatible requests."""
    raw = os.environ.get("DASHSCOPE_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_DASHSCOPE_TIMEOUT_SECONDS
    return float(raw)


def resolve_dashscope_max_retries() -> int:
    """Resolve the retry count for DashScope OpenAI-compatible requests."""
    raw = os.environ.get("DASHSCOPE_MAX_RETRIES")
    if raw is None or not raw.strip():
        return DEFAULT_DASHSCOPE_MAX_RETRIES
    return int(raw)


def resolve_dashscope_base_url() -> str:
    """Resolve the DashScope OpenAI-compatible endpoint from env configuration."""
    explicit_base_url = os.environ.get("DASHSCOPE_BASE_URL")
    if explicit_base_url:
        return explicit_base_url

    region = os.environ.get("DASHSCOPE_REGION", "us").strip().lower()
    return _DASHSCOPE_BASE_URLS.get(
        region,
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    )


class DashscopeModel(OpenAICompatibleChatModel):
    """DashScope Qwen client via the OpenAI-compatible Chat Completions API."""

    supports_response_format_json_schema = True
    supports_response_format_json_object = True
    provider_name = "dashscope"
    provider_prefix = "dashscope"
    api_key_env_var = "DASHSCOPE_API_KEY"
    base_url_env_var = "DASHSCOPE_BASE_URL"
    default_base_url = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"

    def resolve_base_url(self) -> str:
        return resolve_dashscope_base_url()

    def get_client_kwargs(self) -> dict[str, Any]:
        return {
            "timeout": resolve_dashscope_timeout_seconds(),
            "max_retries": resolve_dashscope_max_retries(),
        }

    def _default_enable_thinking(self) -> bool | None:
        env_override = _parse_bool_env("DASHSCOPE_ENABLE_THINKING")
        if env_override is not None:
            return env_override

        # qwen3.6-plus timed out on some image-backed reasoning requests under
        # DashScope's default thinking mode in this toolkit workflow. Keep it
        # off by default unless the caller explicitly opts in.
        if self.model.lower().startswith("qwen3.6"):
            return False

        return None

    def chat(
        self, user_prompt: str, image_paths=None, response_format=None, **kwargs: Any
    ) -> str:
        extra_body = kwargs.get("extra_body")
        if extra_body is None:
            extra_body = {}
        elif not isinstance(extra_body, dict):
            raise TypeError("DashScope extra_body must be a dict when provided.")
        else:
            extra_body = dict(extra_body)

        enable_thinking = self._default_enable_thinking()
        if response_format is not None:
            if bool(extra_body.get("enable_thinking")):
                raise ValueError(
                    "DashScope structured output is not supported in thinking mode. "
                    "Set extra_body['enable_thinking']=False or omit it."
                )
            if "enable_thinking" not in extra_body:
                enable_thinking = False
        if enable_thinking is not None and "enable_thinking" not in extra_body:
            extra_body["enable_thinking"] = enable_thinking

        if response_format is not None:
            kwargs.pop("max_output_tokens", None)

        if extra_body:
            kwargs["extra_body"] = extra_body

        return super().chat(
            user_prompt=user_prompt,
            image_paths=image_paths,
            response_format=response_format,
            **kwargs,
        )

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        del structured_policy
        return StructuredOutputPlan(
            mode="json_schema",
            strategy="dashscope_chat_json_schema",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema", "json_object"),
            accepted_artifact_strategies=(
                "dashscope_chat_json_schema",
                "dashscope_json_object",
            ),
            response_format=spec.source_model
            or normalize_response_format(
                {
                    "type": "json_schema",
                    "name": spec.name,
                    "schema": spec.schema,
                    "strict": spec.strict,
                    "description": spec.description,
                }
            ),
        )
