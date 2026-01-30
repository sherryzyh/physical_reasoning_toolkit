"""
Base annotator class for physical problem annotation.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from prkit.prkit_core.llm import LLMClient


class BaseAnnotator(ABC):
    """Base class for all annotators."""

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize base annotator.

        Args:
            model: LLM model name to use for annotation
        """
        self.model = model
        self.llm_client = LLMClient.from_model(model)

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
            response_format: Expected response format specification

        Returns:
            Structured response from LLM, or None if call fails
        """
        try:
            return self.llm_client.chat_structured(
                system_prompt="You are a physics expert. Provide accurate, detailed analysis of physics problems. Always respond with valid JSON in the exact format requested.",
                prompt=prompt,
                response_format=response_format,
            )
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
            messages = [
                {
                    "role": "system",
                    "content": "You are a physics expert. Provide accurate, detailed analysis of physics problems. Always respond with valid JSON in the exact format requested.",
                },
                {"role": "user", "content": prompt},
            ]
            return self.llm_client.chat(messages).strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "{}"
