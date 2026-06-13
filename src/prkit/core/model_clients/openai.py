"""
OpenAI API client implementations.

This module provides client implementations for OpenAI's Responses API.

Supported OpenAI models:
- gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
- gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
- o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
"""

import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..project_env import ensure_openai_api_key
from .base import BaseModelClient
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    normalize_response_format,
)
from .utils import encode_image_to_base64


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()
    if result.get("type") == "object":
        if "additionalProperties" not in result:
            result["additionalProperties"] = False
        if "properties" in result and isinstance(result["properties"], dict):
            result["properties"] = {
                key: _ensure_additional_properties_false(value)
                for key, value in result["properties"].items()
            }
    if "items" in result:
        result["items"] = _ensure_additional_properties_false(result["items"])
    if "$defs" in result and isinstance(result["$defs"], dict):
        result["$defs"] = {
            key: _ensure_additional_properties_false(value)
            for key, value in result["$defs"].items()
        }
    return result


def _strip_ref_siblings(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    if "$ref" not in schema:
        return schema
    return {"$ref": schema["$ref"]}


def ensure_openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    result = _ensure_additional_properties_false(schema)
    if result.get("type") == "object":
        properties = result.get("properties")
        if isinstance(properties, dict):
            result["properties"] = {
                key: ensure_openai_strict_json_schema(value)
                for key, value in properties.items()
            }
            result["required"] = list(result["properties"].keys())

    if "items" in result:
        result["items"] = ensure_openai_strict_json_schema(result["items"])

    for union_key in ("anyOf", "allOf", "oneOf"):
        union_value = result.get(union_key)
        if isinstance(union_value, list):
            result[union_key] = [
                ensure_openai_strict_json_schema(item) for item in union_value
            ]

    if "$defs" in result and isinstance(result["$defs"], dict):
        result["$defs"] = {
            key: ensure_openai_strict_json_schema(value)
            for key, value in result["$defs"].items()
        }

    return _strip_ref_siblings(result)


def _openai_native_schema_incompatibility(spec: StructuredOutputSpec) -> str | None:
    features = spec.schema_features
    if features is None:
        return None
    if features.has_allof:
        return "allOf is not supported by OpenAI structured outputs"
    return None


def _is_supported_openai_model(model: str) -> bool:
    """
    Check if the OpenAI model is supported.

    Supported OpenAI models:
    - gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
    - gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
    - o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)

    Args:
        model: Model name to check

    Returns:
        True if the model is supported, False otherwise
    """
    model_lower = model.lower()

    # Check for o-family (o3, o4, o4-mini, etc. - starts with 'o' followed by number)
    if len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit():
        return True

    # Check for gpt-4.1
    if model_lower.startswith("gpt-4.1"):
        return True

    # Check for gpt-5xxxx
    if model_lower.startswith("gpt-5"):
        return True

    return False


def _is_o_family_model(model: str) -> bool:
    """
    Check if the model is an o-family reasoning model.

    Args:
        model: Model name to check

    Returns:
        True if the model is an o-family model, False otherwise
    """
    model_lower = model.lower()
    return len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit()


def prepare_image_url_from_image_path(image_path: str) -> str:
    """
    Prepare an image URL from a file path, URL, or base64 data URL.

    Args:
        image_path: Can be:
                   - File path: "/path/to/image.jpg" - will be encoded to base64
                   - HTTP/HTTPS URL: "https://example.com/image.jpg" - used as-is
                   - Base64 data URL: "data:image/jpeg;base64,..." - used as-is

    Returns:
        Image URL in the appropriate format:
        - Base64 data URL for file paths (e.g., "data:image/jpeg;base64,...")
        - Original URL for HTTP/HTTPS URLs
        - Original string for base64 data URLs

    Raises:
        FileNotFoundError: If image_path is a file path that doesn't exist
        IOError: If there's an error reading the image file
    """
    # If it's already a data URL, return as-is
    if image_path.startswith("data:"):
        return image_path

    # If it's an HTTP/HTTPS URL, return as-is
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path

    # Otherwise, treat it as a file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Determine MIME type from file extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")  # Default to jpeg

    base64_image_string = encode_image_to_base64(image_path)
    image_url = f"data:{mime_type};base64,{base64_image_string}"

    return image_url


class OpenAIModel(BaseModelClient):
    """OpenAI model client implementation using Responses API."""

    supports_response_format_json_schema = True

    def __init__(self, model: str, logger=None):
        """
        Initialize OpenAI model client.

        Args:
            model: OpenAI model name. Supported models:
                  - gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
                  - gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
                  - o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
            logger: Optional logger instance

        Raises:
            ValueError: If the model is not supported
        """
        if not _is_supported_openai_model(model):
            raise ValueError(
                f"Unsupported OpenAI model: {model}. "
                "Supported models: gpt-4.1 (and variants), gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), "
                "and o-family (o3, o4, o4-mini, etc.)"
            )
        super().__init__(model, logger)
        self.client = OpenAI(api_key=ensure_openai_api_key(__file__, required=False))
        self.provider = "openai"
        self.is_o_family = _is_o_family_model(model)

    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from OpenAI Responses API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings). Can be:
                       - File paths: ["/path/to/image.jpg", ...] - will be encoded to base64
                       - HTTP/HTTPS URLs: ["https://example.com/image.jpg", ...] - used as-is
                       - Base64 data URLs: ["data:image/jpeg;base64,...", ...] - used as-is
            response_format: Optional structured output format (OpenAI-style dict or Pydantic
                           model). Ensures response adheres to JSON Schema.
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for request parameters
                     (e.g., max_tokens, etc.)

        Returns:
            Response text from OpenAI model (JSON string when response_format is used)

        Raises:
            FileNotFoundError: If any image_path is a file path that doesn't exist
            IOError: If there's an error reading any image file
        """
        # Build request parameters
        request_params = {"model": self.model}

        # Add structured output if requested
        if response_format is not None:
            normalized = normalize_response_format(response_format)
            strict_schema = ensure_openai_strict_json_schema(normalized["schema"])
            request_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": normalized["name"],
                    "schema": strict_schema,
                    "strict": normalized["strict"],
                }
            }
            if normalized.get("description") is not None:
                request_params["text"]["format"]["description"] = normalized[
                    "description"
                ]

        # Use role/content format for all models
        content = [{"type": "input_text", "text": user_prompt}]

        if image_paths:
            for image_path in image_paths:
                image_url = prepare_image_url_from_image_path(image_path)
                content.append({"type": "input_image", "image_url": image_url})

        request_params["input"] = [{"role": "user", "content": content}]

        # Add reasoning parameter for o-family models
        if self.is_o_family:
            request_params["reasoning"] = {"effort": "medium"}

        max_output_tokens = kwargs.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            request_params["max_output_tokens"] = max_output_tokens

        if kwargs:
            request_params.update(kwargs)

        response = self.client.responses.create(**request_params)
        self.logger.info(f"Response: {response.output_text}")
        return response.output_text

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        incompatibility = _openai_native_schema_incompatibility(spec)
        if incompatibility is not None:
            if structured_policy == "native_required":
                raise ValueError(
                    "OpenAI structured outputs do not accept this schema "
                    f"for model={self.model!r}: {incompatibility}."
                )
            return StructuredOutputPlan(
                mode="prompt_only",
                strategy="openai_prompt_only_unsupported_schema",
                native_schema_enforced=False,
                accepted_artifact_modes=("prompt_only",),
                accepted_artifact_strategies=("openai_prompt_only_unsupported_schema",),
                response_format=None,
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        if spec.schema_features and spec.schema_features.has_open_objects:
            if structured_policy == "native_required":
                raise ValueError(
                    "OpenAI structured outputs do not accept schemas with open objects "
                    f"for model={self.model!r}."
                )
            return StructuredOutputPlan(
                mode="prompt_only",
                strategy="openai_prompt_only",
                native_schema_enforced=False,
                accepted_artifact_modes=("prompt_only",),
                accepted_artifact_strategies=("openai_prompt_only",),
                response_format=None,
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        return StructuredOutputPlan(
            mode="json_schema",
            strategy="openai_responses_json_schema",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema",),
            accepted_artifact_strategies=("openai_responses_json_schema",),
            response_format=spec.source_model
            or normalize_response_format(
                spec.source_model
                or {
                    "type": "json_schema",
                    "name": spec.name,
                    "schema": spec.schema,
                    "strict": spec.strict,
                    "description": spec.description,
                }
            ),
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
                f"OpenAI batch structured requests require json_schema mode. Got {plan.mode!r}."
            )

        normalized = normalize_response_format(plan.response_format or {})
        strict_schema = ensure_openai_strict_json_schema(normalized["schema"])
        text_format: dict[str, Any] = {
            "type": "json_schema",
            "name": normalized["name"],
            "schema": strict_schema,
            "strict": normalized.get("strict", True),
        }
        if normalized.get("description") is not None:
            text_format["description"] = normalized["description"]

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": prepare_image_url_from_image_path(image_path),
                }
            )

        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "text": {"format": text_format},
        }
        if self.is_o_family:
            body["reasoning"] = {"effort": "medium"}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
