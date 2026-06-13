"""Anthropic API client implementation."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

try:
    from anthropic import Anthropic, transform_schema as anthropic_transform_schema
except ImportError:  # pragma: no cover - tested via runtime error path
    Anthropic = None
    anthropic_transform_schema = None

from .base import BaseModelClient
from .structured_output import (
    build_json_schema_prompt_suffix,
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    normalize_response_format,
)
from .utils import encode_image_to_base64

TOOL_NAME = "emit_structured_output"
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
ANTHROPIC_OPTIONAL_PARAMETER_LIMIT = 24
ANTHROPIC_UNION_PARAMETER_LIMIT = 16


def _detect_image_media_type(image_path: str) -> str:
    """Detect media type for image file path."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def _parse_data_url(data_url: str) -> Dict[str, str]:
    """
    Parse a data URL into media_type and base64 payload for Anthropic image blocks.

    Expects format: data:<media_type>;base64,<payload>
    """
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL to start with 'data:'")
    if ";base64," not in data_url:
        raise ValueError("Anthropic image data URL must include ';base64,'")

    header, payload = data_url.split(",", 1)
    media_type = header[5:].split(";")[0]
    if not media_type:
        media_type = "image/jpeg"

    return {"media_type": media_type, "data": payload}


def _tool_name(raw_name: str | None) -> str:
    """Sanitize a schema name for Anthropic tool use."""
    candidate = _TOOL_NAME_RE.sub("_", (raw_name or TOOL_NAME)).strip("_")
    return candidate[:64] or TOOL_NAME


def _block_attr(block: Any, name: str) -> Any:
    """Read a content-block attribute from dict or SDK object shapes."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _extract_tool_use_json(content_blocks: List[Any]) -> str:
    """Return the emitted Anthropic tool input as a JSON string."""
    tool_blocks = [block for block in content_blocks if _block_attr(block, "type") == "tool_use"]
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


def _extract_text_json(content_blocks: List[Any]) -> str:
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
    if features.has_recursive_refs:
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


def _anthropic_transformed_response_format(spec: StructuredOutputSpec) -> dict[str, Any]:
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

    def __init__(self, model: str, logger=None):
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
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.provider = "anthropic"

    def chat(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Union[dict, type]] = None,
        max_output_tokens: int = 1024,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from Anthropic Messages API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings). Supports:
                       - File paths: encoded to base64
                       - Base64 data URLs: passed as-is after parsing
                       HTTP/HTTPS URLs are currently ignored with a warning.
            response_format: Optional structured output schema. When provided, the
                           request is translated into a forced Anthropic tool call.
            max_output_tokens: Maximum output tokens for Anthropic API.
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for request parameters
                     (e.g., temperature, top_p, etc.)

        Returns:
            Response text from Anthropic model

        Raises:
            FileNotFoundError: If any image_path is a file path that doesn't exist
            ValueError: If a data URL is malformed
        """
        normalized_response_format = None
        if response_format is not None:
            normalized_response_format = normalize_response_format(response_format)

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_paths:
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
                elif image_path.startswith("http://") or image_path.startswith(
                    "https://"
                ):
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

        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_output_tokens,
        }
        if normalized_response_format is not None:
            request_params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": normalized_response_format["schema"],
                }
            }
        if kwargs:
            request_params.update(kwargs)

        response = self.client.messages.create(**request_params)

        if normalized_response_format is not None:
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

    def _build_batch_structured_request(
        self,
        *,
        request_id: str,
        user_prompt: str,
        response_model: type[BaseModel],
        image_paths: tuple[str, ...],
        max_output_tokens: int | None,
        plan: StructuredOutputPlan,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del response_model, kwargs
        if plan.mode != "json_schema":
            raise ValueError(
                "Anthropic batch structured requests require json_schema mode. "
                f"Got {plan.mode!r}."
            )

        normalized = normalize_response_format(plan.response_format or {})
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
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
            else:
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

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens or 4096,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": normalized["schema"],
                }
            },
        }

        return {
            "custom_id": request_id,
            "params": params,
        }
