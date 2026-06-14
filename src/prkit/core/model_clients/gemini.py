"""Google Gemini API model client using the ``google-genai`` SDK."""

import logging
import os
from typing import Any

import PIL.Image
from google import genai
from google.genai import types
from pydantic import BaseModel

from .base import BaseModelClient
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    extract_schema_for_gemini,
    normalize_response_format,
)
from .utils import detect_image_mime_type, encode_image_to_base64


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
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            # Will try to pick up from GEMINI_API_KEY env var automatically
            self.genai_client = genai.Client()
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
        del structured_policy
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
                f"Gemini batch structured requests require json_schema mode. Got {plan.mode!r}."
            )

        parts: list[dict[str, Any]] = []
        for image_path in image_paths:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _guess_mime_type(image_path),
                        "data": _encode_image_file_base64(image_path),
                    }
                }
            )
        parts.append({"text": user_prompt})

        normalized = normalize_response_format(plan.response_format or {})
        generation_config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_json_schema": extract_schema_for_gemini(normalized),
        }
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens

        return {
            "key": request_id,
            "request": {
                "contents": [{"parts": parts, "role": "user"}],
                "generation_config": generation_config,
            },
        }


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
    """Return the MIME type for the image at *path* based on its extension."""
    return detect_image_mime_type(path)


def _encode_image_file_base64(path: str) -> str:
    """Return the base64-encoded contents of the image file at *path*."""
    return encode_image_to_base64(path)
