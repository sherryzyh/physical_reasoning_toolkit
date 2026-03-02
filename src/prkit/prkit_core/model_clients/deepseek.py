"""
DeepSeek API client implementation.
"""

import os
from typing import Any, List, Optional, Union

from openai import OpenAI

from .base import BaseModelClient


class DeepseekModel(BaseModelClient):
    """DeepSeek API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize DeepSeek model client.

        Args:
            model: DeepSeek model name
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.provider = "deepseek"

    def chat(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Union[dict, type]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from DeepSeek API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings).
                        DeepSeek models do not support vision. Images are ignored with a warning.
            response_format: Not supported. If provided, a warning is logged and it is ignored.
            **kwargs: Additional keyword arguments (ignored, kept for compatibility)

        Returns:
            Response text from DeepSeek model
        """
        if response_format is not None:
            self.logger.warning(
                f"Structured output (response_format) is not supported by DeepSeek model {self.model}. "
                "Ignoring response_format and returning plain text. "
                "Use OpenAI or Gemini for structured output support."
            )
        if image_paths:
            self.logger.warning(
                f"DeepSeek model {self.model} does not support image inputs. "
                f"Received {len(image_paths)} image(s) which will be ignored."
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content
