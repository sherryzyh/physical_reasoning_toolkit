"""
OpenAI API client implementations.

This module provides client implementations for OpenAI's Responses API.

Supported OpenAI models:
- gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
- gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
- o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
"""

import io
import json
import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..project_env import ensure_openai_api_key
from .base import BaseModelClient
from .batch_types import BatchItemStatus, BatchResult, BatchState, BatchStatus
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    normalize_response_format,
)
from .utils import prepare_image_url_from_path

# Map the OpenAI batch ``status`` string onto a provider-agnostic ``BatchState``.
_OPENAI_BATCH_STATE_MAP = {
    "validating": BatchState.PENDING,
    "in_progress": BatchState.IN_PROGRESS,
    "finalizing": BatchState.IN_PROGRESS,
    "completed": BatchState.COMPLETED,
    "failed": BatchState.FAILED,
    "expired": BatchState.EXPIRED,
    "cancelling": BatchState.IN_PROGRESS,
    "cancelled": BatchState.CANCELLED,
}


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on all object nodes."""
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
    """Drop sibling keys alongside ``$ref`` nodes, which OpenAI strict mode disallows."""
    if not isinstance(schema, dict):
        return schema
    if "$ref" not in schema:
        return schema
    return {"$ref": schema["$ref"]}


def ensure_openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform *schema* into a form that satisfies OpenAI strict structured-output requirements."""
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
    """Return a human-readable reason string when *spec* cannot use OpenAI native structured output, or ``None`` when compatible."""
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
    return prepare_image_url_from_path(image_path)


class OpenAIModel(BaseModelClient):
    """OpenAI model client implementation using Responses API."""

    supports_response_format_json_schema = True

    def __init__(
        self,
        model: str,
        logger: logging.Logger | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        """
        Initialize OpenAI model client.

        Args:
            model: OpenAI model name. Supported models:
                  - gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
                  - gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
                  - o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
            logger: Optional logger instance
            base_url: Optional custom Responses-API endpoint (e.g. a proxy at
                      ``https://gw.example/v1``). When omitted the OpenAI SDK default
                      is used (honouring its own ``OPENAI_BASE_URL`` env if set).
            api_key: Explicit API key. Takes precedence over ``api_key_env`` and the
                     default ``OPENAI_API_KEY`` environment variable.
            api_key_env: Name of an environment variable to read the API key from.
                         Used when ``api_key`` is not supplied.

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

        resolved_api_key: str | None
        if api_key is not None:
            resolved_api_key = api_key
        elif api_key_env is not None:
            resolved_api_key = os.environ.get(api_key_env)
        else:
            resolved_api_key = ensure_openai_api_key(__file__, required=False)

        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.provider = "openai"
        self.base_url = base_url
        self.is_o_family = _is_o_family_model(model)

    def _resolve_instructions(self, instructions: str | None) -> str | None:
        """OpenAI works with ``input`` alone — no default system prompt.

        Only the caller's explicit *instructions* (if any) are sent; ``None`` stays
        ``None`` so the Responses API receives no ``instructions`` parameter.
        """
        return instructions

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        *args: Any,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from OpenAI Responses API.

        Args:
            input: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings). Can be:
                       - File paths: ["/path/to/image.jpg", ...] - will be encoded to base64
                       - HTTP/HTTPS URLs: ["https://example.com/image.jpg", ...] - used as-is
                       - Base64 data URLs: ["data:image/jpeg;base64,...", ...] - used as-is
            response_format: Optional structured output format (OpenAI-style dict or Pydantic
                           model). Ensures response adheres to JSON Schema.
            *args: Additional positional arguments (ignored, kept for compatibility)
            instructions: Optional system prompt, sent as the Responses API
                        ``instructions`` parameter. OpenAI does not apply a default,
                        so it is omitted entirely when not provided.
            **kwargs: Additional keyword arguments for request parameters
                     (e.g., max_tokens, etc.)

        Returns:
            Response text from OpenAI model (JSON string when response_format is used)

        Raises:
            FileNotFoundError: If any image_path is a file path that doesn't exist
            IOError: If there's an error reading any image file
        """
        max_output_tokens = kwargs.pop("max_output_tokens", None)
        request_params = self._build_responses_body(
            input=input,
            instructions=self._resolve_instructions(instructions),
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            extra=kwargs,
        )

        response = self.client.responses.create(**request_params)
        text = str(response.output_text)
        self.logger.info(f"Response: {text}")
        return text

    def _omit_temperature(self) -> bool:
        """o-family reasoning models reject an explicit ``temperature`` parameter."""
        return self.is_o_family

    def _build_responses_body(
        self,
        *,
        input: str,
        instructions: str | None,
        image_paths: Sequence[str] | None,
        max_output_tokens: int | None,
        response_format: dict[str, Any] | type | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Responses API body shared by ``response()`` and the batch builders.

        *instructions* must already be resolved (see ``_resolve_instructions``); an
        empty string is omitted. *extra* carries any remaining request params
        (e.g. ``temperature``) and is merged last, mirroring ``response()``'s
        historical ``request_params.update(kwargs)`` behavior.
        """
        body: dict[str, Any] = {"model": self.model}

        if response_format is not None:
            normalized = normalize_response_format(response_format)
            strict_schema = ensure_openai_strict_json_schema(normalized["schema"])
            text_format: dict[str, Any] = {
                "type": "json_schema",
                "name": normalized["name"],
                "schema": strict_schema,
                "strict": normalized.get("strict", True),
            }
            if normalized.get("description") is not None:
                text_format["description"] = normalized["description"]
            body["text"] = {"format": text_format}

        content: list[dict[str, Any]] = [{"type": "input_text", "text": input}]
        if image_paths:
            for image_path in image_paths:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": prepare_image_url_from_image_path(image_path),
                    }
                )
        body["input"] = [{"role": "user", "content": content}]

        if instructions:
            body["instructions"] = instructions
        if self.is_o_family:
            body["reasoning"] = {"effort": "medium"}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if extra:
            body.update(extra)
        return body

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
        """Build a free-text Responses batch line (one ``input.jsonl`` row)."""
        extra: dict[str, Any] = dict(kwargs)
        if temperature is not None and not self._omit_temperature():
            extra["temperature"] = temperature
        body = self._build_responses_body(
            input=input,
            instructions=instructions,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            response_format=None,
            extra=extra,
        )
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    def submit_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload the requests as a JSONL file and create a 24h batch job."""
        payload = "\n".join(json.dumps(request) for request in requests)
        upload = self.client.files.create(
            file=("batch_requests.jsonl", io.BytesIO(payload.encode("utf-8"))),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata=metadata or {},
        )
        return str(batch.id)

    def poll_batch(self, batch_id: str) -> BatchStatus:
        batch = self.client.batches.retrieve(batch_id)
        raw_status = str(getattr(batch, "status", "") or "")
        counts: dict[str, int] = {}
        request_counts = getattr(batch, "request_counts", None)
        if request_counts is not None:
            for key in ("total", "completed", "failed"):
                counts[key] = int(getattr(request_counts, key, 0) or 0)
        return BatchStatus(
            batch_id=batch_id,
            state=_OPENAI_BATCH_STATE_MAP.get(raw_status, BatchState.UNKNOWN),
            provider=self._provider_name(),
            raw_status=raw_status,
            counts=counts,
            output_ref=getattr(batch, "output_file_id", None),
            error_ref=getattr(batch, "error_file_id", None),
        )

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        batch = self.client.batches.retrieve(batch_id)
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            for line in _iter_jsonl_lines(self.client.files.content(output_file_id)):
                yield _parse_openai_result_line(line)
        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            for line in _iter_jsonl_lines(self.client.files.content(error_file_id)):
                yield _parse_openai_result_line(line, force_error=True)

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
        body = self._build_responses_body(
            input=user_prompt,
            instructions=None,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            response_format=plan.response_format or {},
        )
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }


def _iter_jsonl_lines(content: Any) -> Iterator[str]:
    """Yield non-empty lines from an OpenAI ``files.content`` payload.

    ``files.content`` returns an ``HttpxBinaryResponseContent``; prefer its
    ``.text``, falling back to ``.read()`` / raw bytes for stubbed clients.
    """
    text = getattr(content, "text", None)
    if text is None:
        raw = content.read() if hasattr(content, "read") else content
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _extract_output_text_from_responses_body(body: Any) -> str:
    """Pull the assistant text out of a Responses API object body (batch output line)."""
    if not isinstance(body, dict):
        return ""
    flat = body.get("output_text")
    if isinstance(flat, str) and flat:
        return flat
    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                piece = part.get("text")
                if piece:
                    chunks.append(str(piece))
    return "".join(chunks)


def _parse_openai_result_line(line: str, *, force_error: bool = False) -> BatchResult:
    """Parse one JSONL line from a batch output/error file into a ``BatchResult``."""
    obj = json.loads(line)
    custom_id = str(obj.get("custom_id", ""))
    error = obj.get("error")
    response = obj.get("response") if isinstance(obj.get("response"), dict) else None
    status_code = response.get("status_code") if response else None
    if (
        force_error
        or error is not None
        or (status_code is not None and status_code != 200)
    ):
        message = (
            json.dumps(error) if error is not None else f"status_code={status_code}"
        )
        return BatchResult(custom_id, BatchItemStatus.ERRORED, error=message)
    body = response.get("body") if response else None
    return BatchResult(
        custom_id,
        BatchItemStatus.SUCCEEDED,
        text=_extract_output_text_from_responses_body(body),
    )
