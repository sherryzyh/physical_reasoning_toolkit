"""
Model Clients Package

Provider modules are loaded lazily (see ``__getattr__``) so importing
``create_model_client`` does not require every optional SDK (OpenAI, Anthropic,
Gemini, Ollama, etc.) to be installed.
"""

from .base import BaseModelClient
from .factory import create_model_client

create_llm_client = create_model_client

__all__ = [
    "BaseModelClient",
    "create_model_client",
    "create_llm_client",
    "AnthropicModel",
    "DashscopeModel",
    "DeepseekModel",
    "GeminiModel",
    "OllamaModel",
    "OpenAIModel",
    "XAIModel",
]


def __getattr__(name: str):
    if name == "AnthropicModel":
        from .anthropic import AnthropicModel

        return AnthropicModel
    if name == "DeepseekModel":
        from .deepseek import DeepseekModel

        return DeepseekModel
    if name == "DashscopeModel":
        from .dashscope import DashscopeModel

        return DashscopeModel
    if name == "GeminiModel":
        from .gemini import GeminiModel

        return GeminiModel
    if name == "OllamaModel":
        from .ollama import OllamaModel

        return OllamaModel
    if name == "OpenAIModel":
        from .openai import OpenAIModel

        return OpenAIModel
    if name == "XAIModel":
        from .xai import XAIModel

        return XAIModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
