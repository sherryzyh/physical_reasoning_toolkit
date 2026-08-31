"""
Factory for creating model client instances based on model names.

Providers are resolved through an ordered registry of rules. Each rule pairs a
predicate over the (lower-cased) model name with a loader that lazily imports
and constructs the matching client — so callers only need the dependencies for
the backend they actually use (e.g. Ollama scripts do not require the OpenAI or
Anthropic SDKs).

To add a provider, append a rule with :func:`register_model_client`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..exceptions import UnknownModelError
from .base import BaseModelClient

ModelMatcher = Callable[[str], bool]
ModelLoader = Callable[[str, logging.Logger | None], BaseModelClient]


@dataclass(frozen=True)
class ProviderRule:
    """A single provider-resolution rule for the model-client factory."""

    name: str
    matches: ModelMatcher
    load: ModelLoader


def _is_o_family(model_lower: str) -> bool:
    """Match OpenAI o-family ids such as ``o3`` or ``o4-mini``."""
    return len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit()


def _load_xai(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return an ``XAIModel`` instance."""
    from .xai import XAIModel

    return XAIModel(model, logger)


def _load_dashscope(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return a ``DashscopeModel`` instance."""
    from .dashscope import DashscopeModel

    return DashscopeModel(model, logger)


def _load_deepseek(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return a ``DeepseekModel`` instance."""
    from .deepseek import DeepseekModel

    return DeepseekModel(model, logger)


def _load_moonshot(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return a ``MoonshotModel`` instance."""
    from .moonshot import MoonshotModel

    return MoonshotModel(model, logger)


def _load_anthropic(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return an ``AnthropicModel`` instance."""
    from .anthropic import AnthropicModel

    return AnthropicModel(model, logger)


def _load_ollama(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return an ``OllamaModel`` instance."""
    from .ollama import OllamaModel

    return OllamaModel(model, logger)


def _load_openai(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return an ``OpenAIModel`` instance (o-family path)."""
    from .openai import OpenAIModel

    return OpenAIModel(model, logger)


def _load_gpt(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return an ``OpenAIModel`` after validating the GPT model name."""
    from .openai import OpenAIModel, _is_supported_openai_model

    if not _is_supported_openai_model(model):
        raise UnknownModelError(
            f"Unsupported OpenAI model: {model}. "
            "Supported OpenAI models: gpt-4.1, gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), "
            "and o-family (o3, o4, o4-mini, etc.)"
        )
    return OpenAIModel(model, logger)


def _load_gemini(model: str, logger: logging.Logger | None) -> BaseModelClient:
    """Lazily import and return a ``GeminiModel`` instance."""
    from .gemini import GeminiModel

    return GeminiModel(model, logger)


# Order matters: the first matching rule wins.
_PROVIDER_RULES: list[ProviderRule] = [
    ProviderRule("xai", lambda m: m.startswith(("xai/", "grok-")), _load_xai),
    ProviderRule(
        "dashscope",
        lambda m: m.startswith("dashscope/") or m == "qwen3.6-plus",
        _load_dashscope,
    ),
    ProviderRule("deepseek", lambda m: "deepseek" in m, _load_deepseek),
    ProviderRule(
        "moonshot",
        lambda m: m.startswith(("moonshot/", "kimi-", "moonshot-v1")),
        _load_moonshot,
    ),
    ProviderRule("anthropic", lambda m: "claude" in m, _load_anthropic),
    ProviderRule(
        "ollama",
        lambda m: m.startswith("ollama/") or m.startswith("qwen"),
        _load_ollama,
    ),
    ProviderRule("openai-o-family", _is_o_family, _load_openai),
    ProviderRule("openai-gpt", lambda m: m.startswith("gpt"), _load_gpt),
    ProviderRule("gemini", lambda m: m.startswith("gemini"), _load_gemini),
]


def register_model_client(rule: ProviderRule, *, prepend: bool = False) -> None:
    """Register a provider rule.

    Args:
        rule: The :class:`ProviderRule` to add.
        prepend: If ``True``, insert ahead of the built-in rules so it takes
            precedence; otherwise append (lowest priority).
    """
    if prepend:
        _PROVIDER_RULES.insert(0, rule)
    else:
        _PROVIDER_RULES.append(rule)


def create_model_client(
    model: str, logger: logging.Logger | None = None
) -> BaseModelClient:
    """
    Create the appropriate model client instance based on the model name.

    Args:
        model: Model name (e.g. 'gpt-5.1', 'ollama/qwen3-vl:8b')
        logger: Optional logger instance

    Returns:
        Appropriate BaseModelClient subclass instance

    Raises:
        ValueError: If no registered provider matches the model name.
    """
    model_lower = model.lower()
    for rule in _PROVIDER_RULES:
        if rule.matches(model_lower):
            return rule.load(model, logger)
    raise UnknownModelError(
        f"Unknown model: {model}. "
        "Supported models: OpenAI (gpt-4.1, gpt-5xxxx, o-family), "
        "Anthropic (claude-*), Google (gemini-*), DeepSeek (deepseek-*), "
        "xAI (grok-*), Moonshot (kimi-*, moonshot-v1-*, moonshot/*), "
        "DashScope (dashscope/*, qwen3.6-plus), "
        "Ollama (ollama/<model_name>, qwen*)"
    )


# Alias for backward compatibility
create_llm_client = create_model_client
