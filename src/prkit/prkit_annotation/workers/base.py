"""
Base annotator class for physical problem annotation.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from prkit.prkit_core.model_clients import create_model_client


class BaseAnnotator(ABC):
    """Base class for all annotators."""

    def __init__(self, model: str = "gpt-5-mini"):
        """
        Initialize base annotator.

        Args:
            model: LLM model name to use for annotation
        """
        self.model = model
        self.llm_client = create_model_client(model)

    @abstractmethod
    def work(self, question: str, **kwargs) -> Any:
        """
        Perform annotation for this step.

        Args:
            question: Physics problem question text
            **kwargs: Additional arguments specific to the annotator

        Returns:
            Annotation result (type depends on subclass)
        """
        pass

    def _call_llm_structured(self, prompt: str, response_format: Any) -> Any:
        """
        Make a structured call to the LLM API.

        Args:
            prompt: Prompt text for the LLM
            response_format: Expected response format specification (Pydantic model)

        Returns:
            Parsed response object (instance of response_format), or None if call fails
        """
        try:
            full_prompt = (
                "You are a physics expert. Provide accurate, detailed analysis of physics problems. "
                "Always respond with valid JSON in the exact format requested.\n\n"
                f"{prompt}"
            )
            response_text = self.llm_client.chat(full_prompt)
            if response_text:
                # Parse JSON response and create response_format instance
                import json
                response_dict = json.loads(response_text.strip())
                return response_format(**response_dict)
            return None
        except Exception as e:
            print(f"Error calling LLM API with structured output: {e}")
            return None

    def _call_llm(self, prompt: str) -> str:
        """
        Make a regular call to the LLM API as fallback.

        Args:
            prompt: Prompt text for the LLM

        Returns:
            Response text from LLM, or empty JSON string if call fails
        """
        try:
            full_prompt = (
                "You are a physics expert. Provide accurate, detailed analysis of physics problems. "
                "Always respond with valid JSON in the exact format requested.\n\n"
                f"{prompt}"
            )
            return self.llm_client.chat(full_prompt).strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "{}"
