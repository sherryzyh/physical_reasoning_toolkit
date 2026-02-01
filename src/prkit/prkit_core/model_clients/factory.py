"""
Factory for creating model client instances based on model names.

This module provides a factory function that automatically selects the appropriate
model client implementation based on the model name.
"""

from .base import BaseModelClient
from .deepseek import DeepseekModel
from .gemini import GeminiModel
from .ollama import OllamaModel
from .openai import OpenAIModel, _is_supported_openai_model


def create_model_client(model: str, logger=None) -> BaseModelClient:
    """
    Create appropriate model client instance based on model name.

    This factory function automatically selects the correct client implementation
    based on the model name pattern. It currently supports:
    - OpenAI Chat models: gpt-4.1, gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), o-family (o3, o4, etc.)
    - Google Gemini models (gemini-*)
    - DeepSeek models (deepseek-*)
    - Ollama models (qwen3-vl, qwen3-vl:*, etc.)

    Args:
        model: Model name (e.g., 'gpt-5.1', 'gpt-4.1-mini', 'o3-mini', 'gemini-pro', 'deepseek-chat', 'qwen3-vl')
        logger: Optional logger instance

    Returns:
        Appropriate BaseModelClient subclass instance

    Raises:
        ValueError: If model type is not recognized or OpenAI model is not supported

    Examples:
        >>> client = create_model_client("gpt-5.1")
        >>> isinstance(client, OpenAIModel)
        True

        >>> client = create_model_client("o3-mini")
        >>> isinstance(client, OpenAIModel)
        True

        >>> client = create_model_client("gemini-pro")
        >>> isinstance(client, GeminiModel)
        True

        >>> client = create_model_client("deepseek-chat")
        >>> isinstance(client, DeepseekModel)
        True

        >>> client = create_model_client("qwen3-vl")
        >>> isinstance(client, OllamaModel)
        True
    """
    model_lower = model.lower()
    
    if "deepseek" in model_lower:
        return DeepseekModel(model, logger)
    elif "qwen3-vl" in model_lower or model_lower.startswith("qwen"):
        # Ollama models (qwen3-vl, qwen3-vl:8b-instruct, etc.)
        return OllamaModel(model, logger)
    elif len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit():
        # o-family models (o3, o4, o4-mini, etc.)
        return OpenAIModel(model, logger)
    elif model_lower.startswith("gpt"):
        # Validate OpenAI GPT models
        if not _is_supported_openai_model(model):
            raise ValueError(
                f"Unsupported OpenAI model: {model}. "
                "Supported OpenAI models: gpt-4.1, gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), "
                "and o-family (o3, o4, o4-mini, etc.)"
            )
        return OpenAIModel(model, logger)
    elif model_lower.startswith("gemini"):
        return GeminiModel(model, logger)
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            "Supported models: OpenAI (gpt-4.1, gpt-5xxxx, o-family), "
            "Google (gemini-*), DeepSeek (deepseek-*), Ollama (qwen3-vl, qwen*)"
        )


# Alias for backward compatibility
create_llm_client = create_model_client
