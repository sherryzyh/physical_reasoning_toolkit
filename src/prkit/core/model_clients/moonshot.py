"""
Moonshot AI (Kimi) API client implementation.

Synchronous only. Moonshot operates a batch API, but it accepts only the
``kimi-k2.5``, ``kimi-k2.6`` and ``kimi-k2.7-code`` models — ``kimi-k3``, the
flagship, is excluded — so this client deliberately has no batch surface and
``prkit.batch`` refuses a submit for provider ``"moonshot"`` up front.

Two behaviours are reachable through ``response(**kwargs)`` and so are not
modelled here: ``reasoning_effort`` (``low``/``high``/``max``) on K3, whose
reasoning cannot be disabled, and the ``thinking`` object K2.x uses instead.

Moonshot's rate-limit errors split three ways and want different handling:
``engine_overloaded_error`` is transient and should be retried with backoff,
``rate_limit_reached_error`` means reduce concurrency, and
``exceeded_current_quota_error`` means the account is out of credit, where a
retry can never succeed. prkit has no retry layer on the synchronous chat path
to act on that distinction; the OpenAI SDK's own default retries apply.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible_chat import OpenAICompatibleChatModel
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    normalize_response_format,
)

_MOONSHOT_GLOBAL_BASE_URL = "https://api.moonshot.ai/v1"

# Moonshot serves a global and a China endpoint with distinct hosts.
_MOONSHOT_BASE_URLS = {
    "global": _MOONSHOT_GLOBAL_BASE_URL,
    "intl": _MOONSHOT_GLOBAL_BASE_URL,
    "cn": "https://api.moonshot.cn/v1",
    "china": "https://api.moonshot.cn/v1",
}

# Models whose json_schema enforcement Moonshot documents as dependable. The
# rest carry published caveats — unstable on complex schemas, $ref sometimes
# returned inside markdown fences, oneOf sometimes ignored — so this client
# does not claim native enforcement for them.
_MOONSHOT_RELIABLE_JSON_SCHEMA_PREFIXES = ("kimi-k3", "kimi-k2.7")

# K3 pins these server-side and its docs say to omit them from requests.
_MOONSHOT_K3_FIXED_PARAMS = frozenset(
    {"temperature", "top_p", "n", "presence_penalty", "frequency_penalty"}
)


def resolve_moonshot_base_url() -> str:
    """Resolve the Moonshot endpoint from the environment.

    An explicit ``MOONSHOT_BASE_URL`` wins. Otherwise ``MOONSHOT_REGION``
    selects between the global and China hosts, defaulting to global; an
    unrecognised region falls back to global rather than raising.
    """
    explicit = os.environ.get("MOONSHOT_BASE_URL")
    if explicit:
        return explicit
    region = os.environ.get("MOONSHOT_REGION", "global").strip().lower()
    return _MOONSHOT_BASE_URLS.get(region, _MOONSHOT_GLOBAL_BASE_URL)


class MoonshotModel(OpenAICompatibleChatModel):
    """Moonshot AI (Kimi) client via the OpenAI-compatible Chat Completions API."""

    provider_name = "moonshot"
    provider_prefix = "moonshot"
    api_key_env_var = "MOONSHOT_API_KEY"
    base_url_env_var = "MOONSHOT_BASE_URL"
    default_base_url = _MOONSHOT_GLOBAL_BASE_URL
    supports_response_format_json_schema = True
    supports_response_format_json_object = True

    # Moonshot deprecates ``max_tokens``; its docs give both a default and a
    # maximum for ``max_completion_tokens``, so that is the live parameter.
    max_output_tokens_param = "max_completion_tokens"

    def resolve_base_url(self) -> str:
        """Return the regional endpoint, honouring an explicit base URL first."""
        return resolve_moonshot_base_url()

    def _omitted_request_params(self) -> frozenset[str]:
        """Drop the sampling parameters K3 pins server-side.

        K2.x accepts them, so the rule is per model rather than per provider.
        """
        if self.model.lower().startswith("kimi-k3"):
            return _MOONSHOT_K3_FIXED_PARAMS
        return super()._omitted_request_params()

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        if self.model.lower().startswith(_MOONSHOT_RELIABLE_JSON_SCHEMA_PREFIXES):
            return StructuredOutputPlan(
                mode="json_schema",
                strategy="moonshot_chat_json_schema",
                native_schema_enforced=True,
                accepted_artifact_modes=("json_schema",),
                accepted_artifact_strategies=("moonshot_chat_json_schema",),
                response_format=normalize_response_format(
                    {
                        "type": "json_schema",
                        "name": spec.name,
                        "schema": spec.schema,
                        "strict": spec.strict,
                        "description": spec.description,
                    }
                ),
            )

        if structured_policy == "native_required":
            raise ValueError(
                "Moonshot documents json_schema output as unreliable for "
                f"model={self.model!r}; pass structured_policy='best_effort' or "
                "use a kimi-k3 / kimi-k2.7 model."
            )
        return StructuredOutputPlan(
            mode="json_object",
            strategy="moonshot_json_object",
            native_schema_enforced=False,
            accepted_artifact_modes=("json_object", "prompt_only"),
            accepted_artifact_strategies=("moonshot_json_object",),
            response_format={"type": "json_object"},
            prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
        )

    def _structured_prompt_for_chat(
        self,
        user_prompt: str,
        response_format: dict[str, Any] | type | None,
    ) -> str:
        """Leave the prompt unchanged; the schema travels in ``response_format``."""
        del response_format
        return user_prompt
