"""
Base model client interface for multiple AI model providers.

This module provides the base class for interacting with various model providers.
All models (LLMs and VLMs) inherit from BaseModelClient and handle image inputs
according to their capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from ..logging_config import PRKitLogger


class BaseModelClient(ABC):
    """Abstract base class for all model client implementations."""

    def __init__(self, model: str, logger=None):
        """
        Initialize model client.

        Args:
            model: Model name/identifier
            logger: Optional logger instance
        """
        load_dotenv()
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger if logger else PRKitLogger.get_logger(__name__)

    @abstractmethod
    def chat(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Union[Dict[str, Any], type]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model given a user prompt and optional images.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths (strings)
            response_format: Optional structured output format. Use OpenAI-style format:
                {"type": "json_schema", "name": str, "schema": dict, "strict": bool}
                or a Pydantic BaseModel class.
                Only supported by OpenAI and Gemini; other providers will warn.
            **kwargs: Additional provider-specific arguments

        Returns:
            Response text from the model (JSON string when response_format is used)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement .chat()")
