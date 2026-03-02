"""
Model Clients Package

This package provides a unified interface for interacting with various AI model providers.
All models inherit from BaseModelClient and handle image inputs according to their capabilities.

Architecture:
- BaseModelClient: Abstract base class for all model clients
  - All models support text input
  - Models that support vision handle image_paths parameter
  - Models that don't support vision ignore images with a warning
  - Structured output (response_format) supported by OpenAI and Gemini

Currently supports:
- OpenAI (Responses API) - supports vision, structured output (gpt-4.1, gpt-5xxxx, o-family)
- Google Gemini - vision support, structured output
- DeepSeek - text-only (structured output not supported, warns if used)
- Ollama - supports vision (structured output not supported, warns if used)

The package is designed to be extensible - you can add new providers by:
1. Creating a new module (e.g., `anthropic.py`) with a class inheriting from `BaseModelClient`
2. Implementing the `chat(user_prompt, image_paths=None, response_format=None)` method
3. Registering it in the factory function in `factory.py`
"""

from .base import BaseModelClient
from .deepseek import DeepseekModel
from .factory import create_model_client
from .gemini import GeminiModel
from .ollama import OllamaModel
from .openai import OpenAIModel


__all__ = [
    "BaseModelClient",
    "create_model_client",
    "DeepseekModel",
    "GeminiModel",
    "OllamaModel",
    "OpenAIModel",
]
