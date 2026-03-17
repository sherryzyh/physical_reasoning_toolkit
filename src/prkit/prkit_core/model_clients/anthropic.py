"""
Anthropic API client implementation.
"""

import os
from typing import Any, Dict, List, Optional, Union

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - tested via runtime error path
    Anthropic = None

from .base import BaseModelClient
from .utils import encode_image_to_base64


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


class AnthropicModel(BaseModelClient):
    """Anthropic Messages API client implementation."""

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
            response_format: Not supported. If provided, a warning is logged and it is ignored.
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
        if response_format is not None:
            self.logger.warning(
                f"Structured output (response_format) is not supported by Anthropic model {self.model}. "
                "Ignoring response_format and returning plain text. "
                "Use OpenAI or Gemini for structured output support."
            )

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
        if kwargs:
            request_params.update(kwargs)

        response = self.client.messages.create(**request_params)

        text_chunks = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)

        text = "\n".join(text_chunks).strip()
        self.logger.info(f"Response: {text}")
        return text
