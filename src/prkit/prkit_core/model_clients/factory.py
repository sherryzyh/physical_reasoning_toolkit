"""
Factory for creating model client instances based on model names.

Provider classes are imported lazily so callers only need dependencies for the
backend they use (e.g. Ollama scripts do not require OpenAI/Anthropic packages).
"""

from .base import BaseModelClient


def create_model_client(model: str, logger=None) -> BaseModelClient:
    """
    Create appropriate model client instance based on model name.

    Args:
        model: Model name (e.g. 'gpt-5.1', 'ollama/qwen3-vl:8b')
        logger: Optional logger instance

    Returns:
        Appropriate BaseModelClient subclass instance
    """
    model_lower = model.lower()

    if "deepseek" in model_lower:
        from .deepseek import DeepseekModel

        return DeepseekModel(model, logger)
    if "claude" in model_lower:
        from .anthropic import AnthropicModel

        return AnthropicModel(model, logger)
    if model_lower.startswith("ollama/"):
        from .ollama import OllamaModel

        return OllamaModel(model, logger)
    if len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit():
        from .openai import OpenAIModel

        return OpenAIModel(model, logger)
    if model_lower.startswith("gpt"):
        from .openai import OpenAIModel, _is_supported_openai_model

        if not _is_supported_openai_model(model):
            raise ValueError(
                f"Unsupported OpenAI model: {model}. "
                "Supported OpenAI models: gpt-4.1, gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), "
                "and o-family (o3, o4, o4-mini, etc.)"
            )
        return OpenAIModel(model, logger)
    if model_lower.startswith("gemini"):
        from .gemini import GeminiModel

        return GeminiModel(model, logger)
    raise ValueError(
        f"Unknown model: {model}. "
        "Supported models: OpenAI (gpt-4.1, gpt-5xxxx, o-family), "
        "Anthropic (claude-*), Google (gemini-*), DeepSeek (deepseek-*), "
        "Ollama (ollama/<model_name>, qwen*)"
    )


# Alias for backward compatibility
create_llm_client = create_model_client
