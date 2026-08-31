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
    iter_schema_nodes,
    normalize_response_format,
    schema_has_circular_refs,
    strip_schema_keywords,
)
from .utils import parse_data_url

# xAI accepts only JPEG and PNG image payloads.
_XAI_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png"})

# The only two keywords xAI rejects outright. Its length and item bounds are
# enforced up to documented thresholds (2048 characters, 256 items), so
# stripping those would discard validation xAI would otherwise apply.
_XAI_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"minContains", "maxContains"})


def _has_array_items(schema: Any) -> bool:
    """Return ``True`` when an ``items`` is a list; xAI wants ``prefixItems`` instead."""
    return any(
        isinstance(node.get("items"), list) for node in iter_schema_nodes(schema)
    )


def _has_empty_variant_list(schema: Any) -> bool:
    """Return ``True`` when an ``enum``, ``anyOf`` or ``oneOf`` is present but empty."""
    return any(
        isinstance(node.get(key), list) and not node[key]
        for node in iter_schema_nodes(schema)
        for key in ("enum", "anyOf", "oneOf")
    )


def _has_boolean_subschema(schema: Any) -> bool:
    """Return ``True`` when a declared property or variant is the literal ``true``/``false``.

    ``additionalProperties: false`` is deliberately not counted: it is a
    constraint rather than a property schema, and it is what every provider
    transform in this codebase emits.
    """
    for node in iter_schema_nodes(schema):
        properties = node.get("properties")
        if isinstance(properties, dict) and any(
            isinstance(value, bool) for value in properties.values()
        ):
            return True
        for key in ("anyOf", "oneOf", "allOf"):
            variants = node.get(key)
            if isinstance(variants, list) and any(
                isinstance(variant, bool) for variant in variants
            ):
                return True
    return False


def _xai_native_schema_incompatibility(spec: StructuredOutputSpec) -> str | None:
    """Return the xAI-documented reason native structured output is incompatible, or ``None``.

    Each branch mirrors a rule xAI publishes as a hard 400, so that a schema it
    would reject is demoted here instead of failing at the API. Constructs xAI
    accepts but enforces only best-effort — multi-subschema ``allOf``, ``not``,
    ``if``/``then``/``else``, and bounds past its documented thresholds — are
    deliberately not listed: demoting those would forfeit native enforcement of
    the rest of the schema for no gain.
    """
    features = spec.schema_features
    has_circular = (
        features.has_circular_refs
        if features is not None
        else schema_has_circular_refs(spec.schema)
    )
    if has_circular:
        return "circular $ref definitions are not supported"
    if _has_array_items(spec.schema):
        return "'items' as an array is not supported (use prefixItems)"
    if _has_empty_variant_list(spec.schema):
        return "an empty enum/anyOf/oneOf is not supported"
    if _has_boolean_subschema(spec.schema):
        return "boolean (true/false) property schemas are not supported"
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

    # xAI's reasoning models reject these outright, and reasoning cannot be
    # disabled on the current Grok models, so they are dropped unconditionally.
    # Over-dropping costs a caller an explicit stop sequence and logs why;
    # under-dropping costs a hard 400.
    omitted_request_params = frozenset(
        {"presence_penalty", "frequency_penalty", "stop"}
    )

    def _build_image_content_block(self, image_url: str) -> dict[str, Any]:
        """Warn about media types xAI rejects, then use the chat-completions shape.

        xAI's documented ``{"type": "input_image", "image_url": "<url>"}`` block
        belongs to its Responses API. This client posts to
        ``/v1/chat/completions``, which takes OpenAI's nested form; sending the
        ``input_image`` block there fails to deserialize with a 422 before the
        request ever reaches a model.
        """
        self._warn_on_unsupported_image_media_type(image_url)
        return super()._build_image_content_block(image_url)

    def _warn_on_unsupported_image_media_type(self, image_url: str) -> None:
        """Warn when an inline image carries a media type xAI rejects.

        Only data URLs are checked; the media type behind an ``http(s)`` URL is
        not knowable here. Warns rather than raises, so a request prkit has
        misjudged still reaches the API and fails there with the real reason.
        """
        if not image_url.startswith("data:"):
            return
        try:
            media_type = parse_data_url(image_url)["media_type"]
        except ValueError:
            return
        if media_type not in _XAI_SUPPORTED_IMAGE_MEDIA_TYPES:
            self.logger.warning(
                "xAI accepts only %s images; model %s was given %s, which the "
                "API is expected to reject.",
                ", ".join(sorted(_XAI_SUPPORTED_IMAGE_MEDIA_TYPES)),
                self.model,
                media_type,
            )

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
