"""Anthropic API client implementation."""

import json
import logging
import os
import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .base import BaseModelClient
from .batch_types import BatchItemStatus, BatchResult, BatchState, BatchStatus
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    normalize_response_format,
)
from .utils import detect_image_mime_type, encode_image_to_base64, parse_data_url

Anthropic: Any | None = None
anthropic_transform_schema: (
    Callable[[type[BaseModel] | dict[str, Any]], dict[str, Any]] | None
) = None

try:
    from anthropic import Anthropic as _AnthropicClient
    from anthropic import transform_schema as _anthropic_transform_schema

    Anthropic = _AnthropicClient
    anthropic_transform_schema = _anthropic_transform_schema
except ImportError:  # pragma: no cover - tested via runtime error path
    pass

if TYPE_CHECKING:
    from anthropic.types import OutputConfigParam

TOOL_NAME = "emit_structured_output"
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
ANTHROPIC_OPTIONAL_PARAMETER_LIMIT = 24
ANTHROPIC_UNION_PARAMETER_LIMIT = 16

# Map the Anthropic batch ``processing_status`` onto a provider-agnostic ``BatchState``.
_ANTHROPIC_BATCH_STATE_MAP = {
    "in_progress": BatchState.IN_PROGRESS,
    "canceling": BatchState.IN_PROGRESS,
    "ended": BatchState.COMPLETED,
}

# Map a per-request batch ``result.type`` onto a provider-agnostic ``BatchItemStatus``.
_ANTHROPIC_ITEM_STATUS_MAP = {
    "succeeded": BatchItemStatus.SUCCEEDED,
    "errored": BatchItemStatus.ERRORED,
    "canceled": BatchItemStatus.CANCELED,
    "expired": BatchItemStatus.EXPIRED,
}


def _detect_image_media_type(image_path: str) -> str:
    """Detect media type for an image file path, from its bytes or extension."""
    return detect_image_mime_type(image_path)


def _parse_data_url(data_url: str) -> dict[str, str]:
    """
    Parse a data URL into media_type and base64 payload for Anthropic image blocks.

    Expects format: data:<media_type>;base64,<payload>
    """
    try:
        return parse_data_url(data_url)
    except ValueError as exc:
        detail = str(exc).replace("Image data URL", "Anthropic image data URL")
        raise ValueError(detail) from exc


def _tool_name(raw_name: str | None) -> str:
    """Sanitize a schema name for Anthropic tool use."""
    candidate = _TOOL_NAME_RE.sub("_", (raw_name or TOOL_NAME)).strip("_")
    return candidate[:64] or TOOL_NAME


def _block_attr(block: Any, name: str) -> Any:
    """Read a content-block attribute from dict or SDK object shapes."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _extract_tool_use_json(content_blocks: list[Any]) -> str:
    """Return the emitted Anthropic tool input as a JSON string."""
    tool_blocks = [
        block for block in content_blocks if _block_attr(block, "type") == "tool_use"
    ]
    if len(tool_blocks) != 1:
        text_chunks = [
            str(text)
            for block in content_blocks
            if _block_attr(block, "type") == "text"
            for text in [_block_attr(block, "text")]
            if text
        ]
        detail = (
            "Anthropic structured-output response must contain exactly one tool_use "
            f"block. Got {len(tool_blocks)}."
        )
        if text_chunks:
            detail += " Raw text: " + "\n".join(text_chunks)
        raise ValueError(detail)

    tool_input = _block_attr(tool_blocks[0], "input")
    if tool_input is None:
        raise ValueError("Anthropic structured-output tool_use block is missing input.")
    return json.dumps(tool_input, ensure_ascii=False)


def _extract_text_json(content_blocks: list[Any]) -> str:
    """Return plain text from content blocks, falling back to tool-use JSON extraction."""
    text_chunks = [
        str(text)
        for block in content_blocks
        if _block_attr(block, "type") == "text"
        for text in [_block_attr(block, "text")]
        if text
    ]
    text = "\n".join(text_chunks).strip()
    if text:
        return text
    return _extract_tool_use_json(content_blocks)


def _anthropic_native_schema_incompatibility(spec: StructuredOutputSpec) -> str | None:
    """Return the Anthropic-docs reason native structured output is incompatible."""

    features = spec.schema_features
    if features is None:
        return None

    issues: list[str] = []
    if features.has_circular_refs:
        issues.append("recursive definitions are not supported")
    if features.optional_field_count > ANTHROPIC_OPTIONAL_PARAMETER_LIMIT:
        issues.append(
            "optional parameter count "
            f"{features.optional_field_count} exceeds documented limit "
            f"{ANTHROPIC_OPTIONAL_PARAMETER_LIMIT}"
        )
    if features.union_field_count > ANTHROPIC_UNION_PARAMETER_LIMIT:
        issues.append(
            "union-typed parameter count "
            f"{features.union_field_count} exceeds documented limit "
            f"{ANTHROPIC_UNION_PARAMETER_LIMIT}"
        )
    if not issues:
        return None
    return "; ".join(issues)


def _anthropic_transformed_response_format(
    spec: StructuredOutputSpec,
) -> dict[str, Any]:
    """Build the internal response_format contract while honoring Anthropic SDK transforms."""

    transformed_schema = (
        anthropic_transform_schema(spec.source_model or spec.schema)
        if anthropic_transform_schema is not None
        else spec.schema
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


class AnthropicModel(BaseModelClient):
    """Anthropic Messages API client implementation."""

    # Anthropic does not use ``response_format`` directly, but its forced tool-use
    # path accepts a JSON Schema input contract that satisfies the same guarantee.
    supports_response_format_json_schema = True

    def __init__(self, model: str, logger: logging.Logger | None = None) -> None:
        """
        Initialize Anthropic model client.

        Args:
            model: Anthropic model name (e.g., 'claude-sonnet-4-6')
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        if Anthropic is None:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        self.client: Any = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.provider = "anthropic"

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        max_output_tokens: int = 1024,
        *args: Any,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from Anthropic Messages API.

        Args:
            input: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings). Supports:
                       - File paths: encoded to base64
                       - Base64 data URLs: passed as-is after parsing
                       HTTP/HTTPS URLs are currently ignored with a warning.
            response_format: Optional structured output schema. When provided, the
                           request is translated into a forced Anthropic tool call.
            max_output_tokens: Maximum output tokens for Anthropic API.
            *args: Additional positional arguments (ignored, kept for compatibility)
            instructions: Optional system prompt. Sent as the top-level Anthropic
                        ``system`` parameter. Defaults to ``DEFAULT_INSTRUCTIONS``
                        when omitted (see ``_resolve_instructions``).
            **kwargs: Additional keyword arguments for request parameters
                     (e.g., temperature, top_p, etc.)

        Returns:
            Response text from Anthropic model

        Raises:
            FileNotFoundError: If any image_path is a file path that doesn't exist
            ValueError: If a data URL is malformed
        """
        params = self._build_messages_params(
            input=input,
            instructions=self._resolve_instructions(instructions),
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            extra=kwargs,
        )

        response = self.client.messages.create(**params)

        if response_format is not None:
            text = _extract_text_json(response.content)
            self.logger.info(f"Response: {text}")
            return text

        text_chunks = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)

        text = "\n".join(text_chunks).strip()
        self.logger.info(f"Response: {text}")
        return text

    def _build_content_blocks(
        self, input: str, image_paths: Sequence[str] | None
    ) -> list[dict[str, Any]]:
        """Build Anthropic user-message content blocks (text + base64 images)."""
        content: list[dict[str, Any]] = [{"type": "text", "text": input}]
        if not image_paths:
            return content
        for image_path in image_paths:
            if image_path.startswith("data:"):
                parsed = _parse_data_url(image_path)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": parsed["media_type"],
                            "data": parsed["data"],
                        },
                    }
                )
            elif image_path.startswith("http://") or image_path.startswith("https://"):
                self.logger.warning(
                    "Anthropic image URL inputs are not enabled in this client yet. "
                    f"Ignoring URL image input: {image_path}"
                )
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _detect_image_media_type(image_path),
                            "data": encode_image_to_base64(image_path),
                        },
                    }
                )
        return content

    def _build_messages_params(
        self,
        *,
        input: str,
        instructions: str | None,
        image_paths: Sequence[str] | None,
        max_output_tokens: int,
        response_format: dict[str, Any] | type | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Messages API params shared by ``response()`` and the batch builders.

        *instructions* must already be resolved (an empty string omits ``system``).
        *extra* carries any remaining params (e.g. ``temperature``) and is merged last.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_content_blocks(input, image_paths),
                }
            ],
            "max_tokens": max_output_tokens,
        }
        if instructions:
            params["system"] = instructions
        if response_format is not None:
            normalized = normalize_response_format(response_format)
            # Anthropic's output_config.format accepts only ``type`` and ``schema``;
            # a ``name`` key (an OpenAI-ism) makes the API reject the request with a
            # 400. The typed annotation makes any future stray key a mypy error.
            output_config: OutputConfigParam = {
                "format": {
                    "type": "json_schema",
                    "schema": normalized["schema"],
                }
            }
            params["output_config"] = output_config
        if extra:
            params.update(extra)
        return params

    def _build_batch_request(
        self,
        *,
        request_id: str,
        input: str,
        instructions: str | None,
        image_paths: tuple[str, ...],
        max_output_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a free-text Messages batch request ({custom_id, params})."""
        extra: dict[str, Any] = dict(kwargs)
        if temperature is not None:
            extra["temperature"] = temperature
        params = self._build_messages_params(
            input=input,
            instructions=instructions,
            image_paths=image_paths,
            max_output_tokens=(
                max_output_tokens if max_output_tokens is not None else 1024
            ),
            response_format=None,
            extra=extra,
        )
        return {"custom_id": request_id, "params": params}

    def submit_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        del metadata  # Anthropic batch creation takes no metadata argument.
        batch = self.client.messages.batches.create(requests=list(requests))
        return str(batch.id)

    def poll_batch(self, batch_id: str) -> BatchStatus:
        batch = self.client.messages.batches.retrieve(batch_id)
        raw_status = str(getattr(batch, "processing_status", "") or "")
        counts: dict[str, int] = {}
        request_counts = getattr(batch, "request_counts", None)
        if request_counts is not None:
            for key in ("processing", "succeeded", "errored", "canceled", "expired"):
                counts[key] = int(getattr(request_counts, key, 0) or 0)
        return BatchStatus(
            batch_id=batch_id,
            state=_ANTHROPIC_BATCH_STATE_MAP.get(raw_status, BatchState.UNKNOWN),
            provider=self._provider_name(),
            raw_status=raw_status,
            counts=counts,
            output_ref=getattr(batch, "results_url", None),
        )

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        for entry in self.client.messages.batches.results(batch_id):
            yield _parse_anthropic_result_entry(entry)

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        del structured_policy
        incompatibility = _anthropic_native_schema_incompatibility(spec)
        if incompatibility is not None:
            prompt_schema = (
                anthropic_transform_schema(spec.source_model or spec.schema)
                if anthropic_transform_schema is not None
                else spec.schema
            )
            return StructuredOutputPlan(
                mode="prompt_only",
                strategy="anthropic_prompt_only_schema_limits",
                native_schema_enforced=False,
                accepted_artifact_modes=("prompt_only", "json_schema", "tool_schema"),
                accepted_artifact_strategies=(
                    "anthropic_prompt_only_schema_limits",
                    "anthropic_output_config",
                    "anthropic_tool_schema",
                ),
                response_format=None,
                prompt_suffix=build_json_schema_prompt_suffix(prompt_schema),
            )

        return StructuredOutputPlan(
            mode="json_schema",
            strategy="anthropic_output_config",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema", "tool_schema"),
            accepted_artifact_strategies=(
                "anthropic_output_config",
                "anthropic_tool_schema",
            ),
            response_format=_anthropic_transformed_response_format(spec),
        )


def _parse_anthropic_result_entry(entry: Any) -> BatchResult:
    """Parse one streamed Message Batch result entry into a ``BatchResult``."""
    custom_id = str(getattr(entry, "custom_id", "") or "")
    result = getattr(entry, "result", None)
    result_type = str(getattr(result, "type", "") or "")
    if result_type == "succeeded":
        message = getattr(result, "message", None)
        blocks = getattr(message, "content", None) or []
        text = "\n".join(
            str(_block_attr(block, "text"))
            for block in blocks
            if _block_attr(block, "type") == "text" and _block_attr(block, "text")
        ).strip()
        return BatchResult(custom_id, BatchItemStatus.SUCCEEDED, text=text)
    status = _ANTHROPIC_ITEM_STATUS_MAP.get(result_type, BatchItemStatus.ERRORED)
    error = getattr(result, "error", None)
    return BatchResult(
        custom_id,
        status,
        error=str(error) if error is not None else result_type or "unknown",
    )
