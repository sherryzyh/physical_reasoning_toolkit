"""
Base annotator class for physical problem annotation.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from ...physkit_core.llm import LLMClient


class BaseAnnotator(ABC):
    """Base class for all annotators."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm_client = LLMClient.from_model(model)
    
    @abstractmethod
    def annotate(self, question: str, **kwargs) -> Any:
        """Perform annotation for this step."""
        pass
    
    def _call_llm_structured(self, prompt: str, response_format: Any) -> Any:
        """Make a structured call to the LLM API."""
        try:
            return self.llm_client.chat_structured(
                system_prompt="You are a physics expert. Provide accurate, detailed analysis of physics problems. Always respond with valid JSON in the exact format requested.",
                prompt=prompt,
                response_format=response_format
            )
        except Exception as e:
            print(f"Error calling LLM API with structured output: {e}")
            # Return None to indicate failure - let the calling code handle fallback
            return None
    
    def _call_llm(self, prompt: str) -> str:
        """Make a regular call to the LLM API as fallback."""
        try:
            messages = [
                {"role": "system", "content": "You are a physics expert. Provide accurate, detailed analysis of physics problems. Always respond with valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ]
            return self.llm_client.chat(messages).strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "{}"
