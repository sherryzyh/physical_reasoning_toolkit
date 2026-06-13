import importlib
import sys

import pytest

import prkit
import prkit.prkit_core.model_clients as model_clients


def test_prkit_package_exposes_top_level_aliases():
    reloaded = importlib.reload(prkit)

    assert "prkit_core" in sys.modules
    assert "prkit_datasets" in sys.modules
    assert "prkit_annotation" in sys.modules
    assert "prkit_evaluation" in sys.modules
    assert "PRKitLogger" in reloaded.__all__


def test_model_clients_getattr_returns_lazy_classes():
    assert model_clients.__getattr__("OpenAIModel").__name__ == "OpenAIModel"
    assert model_clients.__getattr__("GeminiModel").__name__ == "GeminiModel"
    assert model_clients.__getattr__("AnthropicModel").__name__ == "AnthropicModel"
    assert model_clients.__getattr__("OllamaModel").__name__ == "OllamaModel"
    assert model_clients.__getattr__("XAIModel").__name__ == "XAIModel"
    assert model_clients.__getattr__("DashscopeModel").__name__ == "DashscopeModel"
    assert model_clients.__getattr__("DeepseekModel").__name__ == "DeepseekModel"


def test_model_clients_getattr_rejects_unknown_names():
    with pytest.raises(AttributeError):
        model_clients.__getattr__("UnknownModel")
