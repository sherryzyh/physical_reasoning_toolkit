"""
Model Clients Package

Provider modules are loaded lazily (see ``__getattr__``) so importing
``create_model_client`` does not require every optional SDK (OpenAI, Anthropic,
Gemini, Ollama, etc.) to be installed.
"""

from typing import Any

from .base import DEFAULT_INSTRUCTIONS, BaseModelClient
from .batch_types import (
    TERMINAL_STATES,
    BatchItemStatus,
    BatchResult,
    BatchState,
    BatchStatus,
)
from .factory import ProviderRule, create_model_client, register_model_client
from .modes import PhysicsOutputMode
from .prompts import build_plain_question_prompt, format_problem_context

create_llm_client = create_model_client

__all__ = [
    "BaseModelClient",
    "BatchItemStatus",
    "BatchResult",
    "BatchState",
    "BatchStatus",
    "DEFAULT_INSTRUCTIONS",
    "PhysicsOutputMode",
    "ProviderRule",
    "TERMINAL_STATES",
    "build_plain_question_prompt",
    "create_model_client",
    "create_llm_client",
    "format_problem_context",
    "register_model_client",
    "AnthropicModel",
    "DashscopeModel",
    "DeepseekModel",
    "GeminiModel",
    "OllamaModel",
    "OpenAIModel",
    "XAIModel",
]


def __getattr__(name: str) -> Any:
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
