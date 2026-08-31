"""Google Gemini API model client using the ``google-genai`` SDK."""

import io
import json
import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

import PIL.Image
from google import genai
from google.genai import types

from .base import BaseModelClient
from .batch_types import BatchItemStatus, BatchResult, BatchState, BatchStatus
from .retry import attempts_from_retries, resolve_max_retries
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    extract_schema_for_gemini,
    iter_schema_nodes,
    normalize_response_format,
)
from .utils import detect_image_mime_type, encode_image_to_base64

# Map the Gemini batch job ``state.name`` onto a provider-agnostic ``BatchState``.
_GEMINI_BATCH_STATE_MAP = {
    "JOB_STATE_PENDING": BatchState.PENDING,
    "JOB_STATE_QUEUED": BatchState.PENDING,
    "JOB_STATE_RUNNING": BatchState.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchState.COMPLETED,
    "JOB_STATE_FAILED": BatchState.FAILED,
    "JOB_STATE_CANCELLED": BatchState.CANCELLED,
    "JOB_STATE_EXPIRED": BatchState.EXPIRED,
}


# Validation keywords absent from ``google.genai.types.JSONSchema``, the pinned
# SDK's own model of the schema subset Gemini accepts for ``response_json_schema``.
# That field set is the vendor's machine-readable statement of what the backend
# understands, and it is the evidence this gate rests on — a keyword it declares
# is not gated even when Google's prose allowlist omits it, and ``default`` is the
# clearest example: 32 of prkit's 44 response models use it, they are enforced
# natively today, and the SDK declares it.
#
# Deliberately NOT gated, despite being documented as unsupported: a ``$ref``
# carrying a non-``$`` sibling, and cycles reached through a required property.
# Pydantic emits a ``description`` beside a ``$ref`` as a matter of course, and
# gating those two rules would demote 23 and 13 of prkit's own models — schemas
# that work against the live API today. The prose is stricter than the behaviour.
#
# ``test_denylist_matches_the_pinned_sdk_schema_subset`` fails if this list and
# the SDK's field set ever disagree, so an SDK upgrade cannot silently widen or
# narrow the gate.
_GEMINI_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "propertyNames",
        "contains",
        "dependentSchemas",
        "prefixItems",
        "multipleOf",
        "const",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
)


def _gemini_native_schema_incompatibility(spec: StructuredOutputSpec) -> str | None:
    """Return why Gemini cannot enforce *spec* natively, or ``None``.

    Only schema-position nodes are inspected, so a ``default`` value or a field
    named after a keyword is never mistaken for the keyword itself.
    """
    found = sorted(
        {
            keyword
            for node in iter_schema_nodes(spec.schema)
            for keyword in _GEMINI_UNSUPPORTED_SCHEMA_KEYWORDS
            if keyword in node
        }
    )
    if not found:
        return None
    return f"unsupported schema keyword(s): {', '.join(found)}"


class GeminiModel(BaseModelClient):
    """Google Gemini API client implementation."""

    supports_response_format_json_schema = True

    def __init__(self, model: str, logger: logging.Logger | None = None) -> None:
        """
        Initialize Gemini model client.

        Args:
            model: Gemini model name (e.g., 'gemini-pro')
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        # The new SDK uses GEMINI_API_KEY, but we support GOOGLE_API_KEY for backward compatibility
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # Without explicit retry options google-genai does not retry at all
        # (``retry_args`` returns ``stop_after_attempt(1)`` for ``None``), unlike
        # every other provider SDK here. ``attempts`` counts the first request,
        # hence the conversion.
        client_kwargs: dict[str, Any] = {
            "http_options": types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=attempts_from_retries(
                        resolve_max_retries("GEMINI_MAX_RETRIES")
                    )
                )
            )
        }
        if api_key:
            self.genai_client = genai.Client(api_key=api_key, **client_kwargs)
        else:
            # Will try to pick up from GEMINI_API_KEY env var automatically
            self.genai_client = genai.Client(**client_kwargs)
        self.provider = "google"

    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        max_output_tokens: int = 65535,
        *args: Any,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from Gemini API.

        Args:
            input: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings).
                       Note: Gemini models support vision, but this implementation
                       currently only handles text. Images are ignored with a warning.
            response_format: Optional structured output format (OpenAI-style dict or
                           Pydantic model). Converted to Gemini's response_json_schema.
            *args: Additional positional arguments (ignored, kept for compatibility)
            instructions: Optional system prompt, sent as the Gemini
                        ``system_instruction`` config field. Defaults to
                        ``DEFAULT_INSTRUCTIONS`` when omitted.
            **kwargs: Additional keyword arguments for generate_content config
                     (e.g., temperature, max_tokens, etc.)

        Returns:
            Response text from Gemini model (JSON string when response_format is used)
        """
        # Prepare content parts
        # Start with the text prompt
        contents_parts: list[Any] = [input]

        # Process images if provided
        if image_paths:
            for path in image_paths:
                try:
                    if os.path.exists(path):
                        # Load image using PIL
                        img = PIL.Image.open(path)
                        contents_parts.append(img)
                    else:
                        if self.logger:
                            self.logger.error(f"Image path not found: {path}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to load image at {path}: {str(e)}")

        # Build config with any additional kwargs
        config_dict: dict[str, Any] = {"max_output_tokens": max_output_tokens}
        instr = self._resolve_instructions(instructions)
        if instr:
            config_dict["system_instruction"] = instr
        if kwargs:
            config_dict.update(kwargs)

        # Add structured output if requested (convert OpenAI format to Gemini format)
        # Gemini accepts raw JSON Schema via response_json_schema (same as Pydantic's model_json_schema)
        if response_format is not None:
            normalized = normalize_response_format(response_format)
            schema = extract_schema_for_gemini(normalized)
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_json_schema"] = schema

        config = types.GenerateContentConfig(**config_dict)

        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=contents_parts,
            config=config,
        )

        text = response.text if response.text is not None else ""
        if not text:
            # Extract block/error details for empty responses
            details = _extract_gemini_error_details(response)
            if details:
                raise RuntimeError(details)
        self.logger.info(f"Response: {text}")
        return text

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        incompatibility = _gemini_native_schema_incompatibility(spec)
        if incompatibility is not None:
            if structured_policy == "native_required":
                raise ValueError(
                    "Gemini structured outputs do not accept this schema "
                    f"for model={self.model!r}: {incompatibility}."
                )
            return StructuredOutputPlan(
                mode="prompt_only",
                strategy="gemini_prompt_only_unsupported_schema",
                native_schema_enforced=False,
                accepted_artifact_modes=("prompt_only",),
                accepted_artifact_strategies=("gemini_prompt_only_unsupported_schema",),
                response_format=None,
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        return StructuredOutputPlan(
            mode="json_schema",
            strategy="gemini_response_json_schema",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema",),
            accepted_artifact_strategies=("gemini_response_json_schema",),
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

    def _build_batch_request(
        self,
        *,
        request_id: str,
        input: str,
        instructions: str | None,
        image_paths: tuple[str, ...],
        max_output_tokens: int | None,
        temperature: float | None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Gemini batch request line ({key, request}).

        Mirrors a synchronous ``response(input=..., instructions=...)`` call: the
        folded prompt becomes the user text and *instructions* (when truthy)
        becomes the request-level ``system_instruction``.

        Unlike the other providers this cannot share a builder with
        ``response()``: that path assembles a typed ``GenerateContentConfig``,
        while a batch line is serialized straight to JSONL and so needs the
        snake_case wire keys. The structured-output block below therefore
        mirrors the sync one by hand and must be kept in step with it.
        """
        parts: list[dict[str, Any]] = [{"text": input}]
        for image_path in image_paths:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _guess_mime_type(image_path),
                        "data": _encode_image_file_base64(image_path),
                    }
                }
            )

        generation_config: dict[str, Any] = {}
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if response_format is not None:
            # Mirrors the sync branch in ``response()``, in snake_case wire form.
            normalized = normalize_response_format(response_format)
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_json_schema"] = extract_schema_for_gemini(
                normalized
            )
        generation_config.update(kwargs)

        request: dict[str, Any] = {"contents": [{"parts": parts, "role": "user"}]}
        if generation_config:
            request["generation_config"] = generation_config
        if instructions:
            request["system_instruction"] = {"parts": [{"text": instructions}]}
        return {"key": request_id, "request": request}

    def submit_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload the requests as keyed JSONL and create a file-based batch job.

        Each request is a ``{"key": ..., "request": {...}}`` line — the documented
        Gemini batch file format. Submitting via an uploaded file (rather than
        ``src=[inline requests]``) is what makes results come back as keyed JSONL
        (``{"key": ..., "response": {...}}``); inline responses carry no key and
        cannot be correlated back to their request.
        """
        payload = "\n".join(json.dumps(request) for request in requests)
        display_name = (metadata or {}).get("display_name", "prkit-batch")
        uploaded = self.genai_client.files.upload(
            file=io.BytesIO(payload.encode("utf-8")),
            config=types.UploadFileConfig(mime_type="jsonl", display_name=display_name),
        )
        uploaded_name = uploaded.name
        if uploaded_name is None:
            raise RuntimeError(
                "Gemini File API upload returned no file name; cannot submit batch"
            )
        job = self.genai_client.batches.create(
            model=self.model,
            src=uploaded_name,
            config={"display_name": display_name},
        )
        return str(job.name)

    def poll_batch(self, batch_id: str) -> BatchStatus:
        job = self.genai_client.batches.get(name=batch_id)
        state = getattr(job, "state", None)
        raw_status = getattr(state, "name", None) or str(state or "")
        dest = getattr(job, "dest", None)
        output_ref = getattr(dest, "file_name", None) if dest is not None else None
        return BatchStatus(
            batch_id=batch_id,
            state=_GEMINI_BATCH_STATE_MAP.get(raw_status, BatchState.UNKNOWN),
            provider=self._provider_name(),
            raw_status=raw_status,
            output_ref=output_ref,
        )

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        """Yield per-request results from the batch's output JSONL file.

        Submission always uses the file-based path (see ``submit_batch``), so a
        terminal job exposes its results as ``dest.file_name`` — a keyed JSONL
        file whose lines correlate back to each request by ``key``.
        """
        job = self.genai_client.batches.get(name=batch_id)
        dest = getattr(job, "dest", None)
        file_name = getattr(dest, "file_name", None) if dest is not None else None
        if not file_name:
            return
        content = self.genai_client.files.download(file=file_name)
        text = (
            content.decode("utf-8")
            if isinstance(content, (bytes, bytearray))
            else str(content)
        )
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                yield _parse_gemini_result_line(stripped)


def _extract_gemini_text(response: dict[str, Any]) -> str:
    """Extract the assistant text from a batch ``response`` dict (parsed from JSONL)."""
    for candidate in response.get("candidates") or []:
        content = (candidate or {}).get("content") or {}
        chunks = [
            str(part.get("text"))
            for part in content.get("parts") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if chunks:
            return "".join(chunks)
    return ""


def _parse_gemini_result_line(line: str) -> BatchResult:
    """Parse one JSONL line ({key, response|error}) from a batch output file."""
    obj = json.loads(line)
    key = str(obj.get("key", "") or "")
    error = obj.get("error")
    response = obj.get("response")
    if error is not None or response is None:
        message = json.dumps(error) if error is not None else "no response"
        return BatchResult(key, BatchItemStatus.ERRORED, error=message)
    return BatchResult(
        key, BatchItemStatus.SUCCEEDED, text=_extract_gemini_text(response)
    )


def _extract_gemini_error_details(response: object) -> str | None:
    """Extract block reason and error details from Gemini response when text is empty."""
    parts = []
    try:
        # Prompt-level block (content filter on input)
        pf = getattr(response, "prompt_feedback", None)
        if pf:
            block_reason = getattr(pf, "block_reason", None)
            if block_reason is not None and str(block_reason):
                parts.append(f"prompt_block_reason={block_reason}")
        # Candidate-level block (content filter on output)
        candidates = getattr(response, "candidates", None) or []
        for c in candidates[:1]:
            fr = getattr(c, "finish_reason", None)
            if fr is not None and str(fr) and "STOP" not in str(fr).upper():
                parts.append(f"finish_reason={fr}")
        if not parts:
            parts.append("empty_response")
    except Exception:
        parts.append("empty_response")
    return "; ".join(parts) if parts else None


def _guess_mime_type(path: str) -> str:
    """Return the MIME type for the image at *path*, from its bytes or extension."""
    return detect_image_mime_type(path)


def _encode_image_file_base64(path: str) -> str:
    """Return the base64-encoded contents of the image file at *path*."""
    return encode_image_to_base64(path)
