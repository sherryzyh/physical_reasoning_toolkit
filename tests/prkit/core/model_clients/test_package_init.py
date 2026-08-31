import importlib
import sys

import pytest

import prkit
import prkit.core.model_clients as model_clients


def test_prkit_subpackages_importable_under_namespace():
    reloaded = importlib.reload(prkit)

    for sub in ("core", "datasets", "evaluation", "annotation", "semantics"):
        module = importlib.import_module(f"prkit.{sub}")
        assert module is not None
        assert f"prkit.{sub}" in sys.modules

    assert "PRKitLogger" in reloaded.__all__
    assert reloaded.__version__


def test_model_clients_getattr_returns_lazy_classes():
    assert model_clients.__getattr__("OpenAIModel").__name__ == "OpenAIModel"
    assert model_clients.__getattr__("GeminiModel").__name__ == "GeminiModel"
    assert model_clients.__getattr__("AnthropicModel").__name__ == "AnthropicModel"
    assert model_clients.__getattr__("OllamaModel").__name__ == "OllamaModel"
    assert model_clients.__getattr__("XAIModel").__name__ == "XAIModel"
    assert model_clients.__getattr__("MoonshotModel").__name__ == "MoonshotModel"
    assert model_clients.__getattr__("DashscopeModel").__name__ == "DashscopeModel"
    assert model_clients.__getattr__("DeepseekModel").__name__ == "DeepseekModel"


def test_model_clients_getattr_rejects_unknown_names():
    with pytest.raises(AttributeError):
        model_clients.__getattr__("UnknownModel")


def test_new_public_symbols_are_exported():
    """DEFAULT_INSTRUCTIONS, PhysicsOutputMode, and prompt helpers are public."""
    for name in (
        "DEFAULT_INSTRUCTIONS",
        "PhysicsOutputMode",
        "build_plain_question_prompt",
        "format_problem_context",
    ):
        assert name in model_clients.__all__
        assert getattr(model_clients, name) is not None

    assert isinstance(model_clients.DEFAULT_INSTRUCTIONS, str)
    assert model_clients.DEFAULT_INSTRUCTIONS.strip()
