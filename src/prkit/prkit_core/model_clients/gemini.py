"""
Google Gemini API client implementation.
"""

import os
from typing import Any, Dict, List, Optional, Union

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseModelClient
from .structured_output import extract_schema_for_gemini, normalize_response_format


class GeminiModel(BaseModelClient):
    """Google Gemini API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize Gemini model client.

        Args:
            model: Gemini model name (e.g., 'gemini-pro')
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        load_dotenv()
        # The new SDK uses GEMINI_API_KEY, but we support GOOGLE_API_KEY for backward compatibility
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            # Will try to pick up from GEMINI_API_KEY env var automatically
            self.genai_client = genai.Client()
        self.provider = "google"
        self.client = None

    def chat(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Union[Dict[str, Any], type]] = None,
        max_output_tokens: int = 8192,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from Gemini API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings).
                       Note: Gemini models support vision, but this implementation
                       currently only handles text. Images are ignored with a warning.
            response_format: Optional structured output format (OpenAI-style dict or
                           Pydantic model). Converted to Gemini's response_json_schema.
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for generate_content config
                     (e.g., temperature, max_tokens, etc.)

        Returns:
            Response text from Gemini model (JSON string when response_format is used)
        """
        # Prepare content parts
        # Start with the text prompt
        contents_parts = [user_prompt]

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
        config_dict: Dict[str, Any] = {"max_output_tokens": max_output_tokens}
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


def _extract_gemini_error_details(response: object) -> Optional[str]:
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
