"""
DashScope API client implementation.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible_chat import OpenAICompatibleChatModel
from .retry import resolve_max_retries
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    coerce_structured_output_spec,
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
    """Resolve the retry count for DashScope OpenAI-compatible requests.

    Previously defaulted to 0, which arrived incidentally in a bulk rename
    rather than as a considered choice and left DashScope the one provider that
    never retried a transient failure.
    """
    return resolve_max_retries("DASHSCOPE_MAX_RETRIES")


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

    # DashScope accepts a json_schema response format but does not enforce it
    # (verified live), so it is not advertised as a schema-enforcing provider.
    supports_response_format_json_schema = False
    supports_response_format_json_object = True
    provider_name = "dashscope"
    provider_prefix = "dashscope"
    api_key_env_var = "DASHSCOPE_API_KEY"
    base_url_env_var = "DASHSCOPE_BASE_URL"
    default_base_url = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"

    def resolve_base_url(self) -> str:
        """Return the DashScope endpoint URL from env configuration."""
        return resolve_dashscope_base_url()

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return timeout and retry settings for the DashScope OpenAI client."""
        return {
            "timeout": resolve_dashscope_timeout_seconds(),
            "max_retries": resolve_dashscope_max_retries(),
        }

    def _default_enable_thinking(self) -> bool | None:
        """Return the default thinking-mode flag, honouring the env override and model-specific defaults."""
        env_override = _parse_bool_env("DASHSCOPE_ENABLE_THINKING")
        if env_override is not None:
            return env_override

        # qwen3.6-plus timed out on some image-backed reasoning requests under
        # DashScope's default thinking mode in this toolkit workflow. Keep it
        # off by default unless the caller explicitly opts in.
        if self.model.lower().startswith("qwen3.6"):
            return False

        return None

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        *,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a request to DashScope, automatically managing the thinking-mode flag."""
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

        return super().response(
            input=input,
            image_paths=image_paths,
            response_format=response_format,
            instructions=instructions,
            **kwargs,
        )

    def _structured_prompt_for_chat(
        self,
        user_prompt: str,
        response_format: dict[str, Any] | type | None,
    ) -> str:
        """Append the schema as prose when a caller reaches ``response()`` directly.

        DashScope rejects any ``response_format`` unless the word "json" appears
        in the messages, and it does not enforce a schema, so the schema has to
        travel in the prompt either way. :meth:`parse` already appends the plan's
        suffix and passes ``{"type": "json_object"}`` down, which is why that
        shape is left alone here rather than being suffixed twice.
        """
        if response_format is None:
            return user_prompt
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            return user_prompt

        try:
            spec = coerce_structured_output_spec(response_format)
        except ValueError:
            return user_prompt
        return user_prompt + build_json_schema_prompt_suffix(spec.schema)

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        """Resolve to ``json_object`` plus a schema prompt; DashScope enforces neither.

        Verified against the live API: a ``json_schema`` response format is
        accepted and then ignored — the same request returned
        ``{"meters_in_kilometre": ...}`` and ``{"meters_per_kilometre": ...}``
        on consecutive attempts for a schema declaring ``number`` and ``unit``.
        Claiming ``native_schema_enforced`` for that would be exactly the false
        promise the flag exists to prevent, so the schema travels as prose.

        DashScope additionally rejects any ``response_format`` unless the word
        "json" appears in the messages, which the prompt suffix satisfies.
        """
        if structured_policy == "native_required":
            raise ValueError(
                "DashScope accepts a json_schema response format but does not "
                f"enforce it (model={self.model!r}); pass "
                "structured_policy='best_effort' to fall back to a "
                "prompt-enforced schema."
            )
        return StructuredOutputPlan(
            mode="json_object",
            strategy="dashscope_json_object",
            native_schema_enforced=False,
            accepted_artifact_modes=("json_object", "prompt_only"),
            accepted_artifact_strategies=(
                "dashscope_json_object",
                "dashscope_chat_json_schema",
            ),
            response_format={"type": "json_object"},
            prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
        )
