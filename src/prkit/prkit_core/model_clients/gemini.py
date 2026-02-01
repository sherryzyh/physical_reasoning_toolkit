"""
Google Gemini API client implementation.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import BaseModelClient


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

    def chat(self, user_prompt: str, image_paths: Optional[List[str]] = None, *args, **kwargs):
        """
        Generate a response from Gemini API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings).
                       Note: Gemini models support vision, but this implementation
                       currently only handles text. Images are ignored with a warning.
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for generate_content config
                     (e.g., temperature, max_tokens, etc.)

        Returns:
            Response text from Gemini model
        """
        if image_paths:
            self.logger.warning(
                f"Gemini model {self.model} received {len(image_paths)} image(s), "
                "but image support is not yet implemented. Images will be ignored."
            )
        
        # Build config with any additional kwargs
        config_dict = {}
        if kwargs:
            config_dict.update(kwargs)

        config = types.GenerateContentConfig(**config_dict) if config_dict else None

        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=config,
        )

        return response.text
