"""
Model Clients Package

This package provides a unified interface for interacting with various AI model providers.
All models inherit from BaseModelClient and handle image inputs according to their capabilities.

Architecture:
- BaseModelClient: Abstract base class for all model clients
  - All models support text input
  - Models that support vision handle image_paths parameter
  - Models that don't support vision ignore images with a warning

Currently supports:
- OpenAI (Responses API) - supports vision (gpt-4.1, gpt-5xxxx, o-family)
- Google Gemini - vision support planned (currently text-only)
- DeepSeek - text-only
- Ollama - supports vision (qwen3-vl, qwen*)

The package is designed to be extensible - you can add new providers by:
1. Creating a new module (e.g., `anthropic.py`) with a class inheriting from `BaseModelClient`
2. Implementing the `chat(user_prompt, image_paths=None)` method
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
