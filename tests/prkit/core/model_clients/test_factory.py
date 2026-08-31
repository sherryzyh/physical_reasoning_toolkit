"""
Tests for model client factory function.
"""

from unittest.mock import MagicMock, patch

import pytest

from prkit.core.model_clients import (
    ProviderRule,
    create_model_client,
    register_model_client,
)
from prkit.core.model_clients.anthropic import AnthropicModel
from prkit.core.model_clients.base import BaseModelClient
from prkit.core.model_clients.dashscope import DashscopeModel
from prkit.core.model_clients.deepseek import DeepseekModel
from prkit.core.model_clients.factory import _PROVIDER_RULES
from prkit.core.model_clients.gemini import GeminiModel
from prkit.core.model_clients.ollama import OllamaModel
from prkit.core.model_clients.openai import OpenAIModel
from prkit.core.model_clients.xai import XAIModel

OPENAI_TEST_MODEL = "gpt-5.4-mini"
GEMINI_TEST_MODEL = "gemini-2.5-pro"
ANTHROPIC_TEST_MODEL = "claude-sonnet-4-6"
OLLAMA_QWEN_TEST_MODEL = "ollama/qwen3.5:397b-cloud"
OLLAMA_MISTRAL_TEST_MODEL = "ollama/mistral-large-3:675b-cloud"
DEEPSEEK_CHAT_TEST_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_TEST_MODEL = "deepseek-reasoner"
XAI_TEST_MODEL = "grok-4.6"
DASHSCOPE_TEST_MODEL = "qwen3.6-plus"


class TestCreateModelClient:
    """Test cases for create_model_client factory function."""

    @pytest.fixture(autouse=True)
    def _mock_provider_sdks(self):
        """Prevent real SDK credential checks during factory routing tests."""
        with (
            patch("prkit.core.model_clients.openai.OpenAI"),
            patch("prkit.core.model_clients.gemini.genai"),
            patch("prkit.core.model_clients.openai_compatible_chat.OpenAI"),
        ):
            yield

    def test_create_openai_gpt_4_1(self):
        """Test creating OpenAI gpt-4.1 model."""
        client = create_model_client("gpt-4.1")
        assert isinstance(client, OpenAIModel)
        assert client.model == "gpt-4.1"
        assert client.provider == "openai"

    def test_create_openai_gpt_5_4_mini(self):
        """Test creating OpenAI gpt-5.4-mini model."""
        client = create_model_client(OPENAI_TEST_MODEL)
        assert isinstance(client, OpenAIModel)
        assert client.model == OPENAI_TEST_MODEL

    def test_create_openai_gpt_5_2_mini(self):
        """Test creating OpenAI gpt-5.2-mini model."""
        client = create_model_client("gpt-5.2-mini")
        assert isinstance(client, OpenAIModel)
        assert client.model == "gpt-5.2-mini"

    def test_create_openai_o3(self):
        """Test creating OpenAI o3 model."""
        client = create_model_client("o3")
        assert isinstance(client, OpenAIModel)
        assert client.model == "o3"
        assert client.is_o_family is True

    def test_create_openai_o4_mini(self):
        """Test creating OpenAI o4-mini model."""
        client = create_model_client("o4-mini")
        assert isinstance(client, OpenAIModel)
        assert client.model == "o4-mini"
        assert client.is_o_family is True

    def test_create_gemini_model(self):
        """Test creating Gemini model."""
        client = create_model_client(GEMINI_TEST_MODEL)
        assert isinstance(client, GeminiModel)
        assert client.model == GEMINI_TEST_MODEL
        assert client.provider == "google"

    def test_create_gemini_model_case_insensitive(self):
        """Test creating Gemini model with different case."""
        client = create_model_client(GEMINI_TEST_MODEL.upper())
        assert isinstance(client, GeminiModel)
        assert client.model == GEMINI_TEST_MODEL.upper()

    def test_create_deepseek_model(self):
        """Test creating DeepSeek model."""
        client = create_model_client(DEEPSEEK_CHAT_TEST_MODEL)
        assert isinstance(client, DeepseekModel)
        assert client.model == DEEPSEEK_CHAT_TEST_MODEL
        assert client.provider == "deepseek"

    def test_create_deepseek_model_case_insensitive(self):
        """Test creating DeepSeek model with different case."""
        client = create_model_client(DEEPSEEK_REASONER_TEST_MODEL.upper())
        assert isinstance(client, DeepseekModel)
        assert client.model == DEEPSEEK_REASONER_TEST_MODEL.upper()

    def test_create_deepseek_model_with_provider_prefix(self):
        """Test provider-prefixed DeepSeek identifiers normalize to raw model names."""
        client = create_model_client("deepseek/deepseek-reasoner")
        assert isinstance(client, DeepseekModel)
        assert client.model == "deepseek-reasoner"

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    def test_create_xai_model(self, mock_openai_class):
        """Test creating xAI Grok model."""
        mock_openai_class.return_value = MagicMock()

        client = create_model_client(XAI_TEST_MODEL)
        assert isinstance(client, XAIModel)
        assert client.model == XAI_TEST_MODEL
        assert client.provider == "xai"

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    def test_create_dashscope_model(self, mock_openai_class):
        """Test creating DashScope model."""
        mock_openai_class.return_value = MagicMock()

        client = create_model_client(DASHSCOPE_TEST_MODEL)
        assert isinstance(client, DashscopeModel)
        assert client.model == DASHSCOPE_TEST_MODEL
        assert client.provider == "dashscope"

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_create_anthropic_model(self, mock_anthropic_class):
        """Test creating Anthropic Claude model."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client = create_model_client(ANTHROPIC_TEST_MODEL)
        assert isinstance(client, AnthropicModel)
        assert client.model == ANTHROPIC_TEST_MODEL
        assert client.provider == "anthropic"

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_create_anthropic_model_case_insensitive(self, mock_anthropic_class):
        """Test creating Anthropic model with different case."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client = create_model_client(ANTHROPIC_TEST_MODEL.upper())
        assert isinstance(client, AnthropicModel)
        assert client.model == ANTHROPIC_TEST_MODEL.upper()

    def test_unsupported_openai_model(self):
        """Test that unsupported OpenAI models raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            create_model_client("gpt-3.5-turbo")

    def test_unknown_model(self):
        """Test that unknown models raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model_client("unknown-model")

    def test_create_with_logger(self):
        """Test creating model client with custom logger."""
        import logging

        logger = logging.getLogger("test")
        client = create_model_client(OPENAI_TEST_MODEL, logger=logger)
        assert client.logger == logger

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_create_ollama_model_with_provider_prefix(self, mock_ollama_module):
        """Test creating Ollama model from provider-prefixed identifier."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = create_model_client(OLLAMA_QWEN_TEST_MODEL)
        assert isinstance(client, OllamaModel)
        assert client.model == "qwen3.5:397b-cloud"
        assert client.provider == "ollama"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_create_second_ollama_model_with_provider_prefix(self, mock_ollama_module):
        """Test creating a second Ollama provider test model."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = create_model_client(OLLAMA_MISTRAL_TEST_MODEL)
        assert isinstance(client, OllamaModel)
        assert client.model == "mistral-large-3:675b-cloud"

    @patch("prkit.core.model_clients.ollama.ollama")
    def test_create_qwen_model_variants(self, mock_ollama_module):
        """Test creating different Qwen model variants."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        for model_name in ["qwen3-vl", "qwen2.5", "qwen"]:
            client = create_model_client(model_name)
            assert isinstance(client, OllamaModel)

    @patch("prkit.core.model_clients.anthropic.Anthropic")
    def test_all_clients_are_base_model_client(self, mock_anthropic_class):
        """Test that all created clients are instances of BaseModelClient."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        clients = [
            create_model_client(OPENAI_TEST_MODEL),
            create_model_client(GEMINI_TEST_MODEL),
            create_model_client(DEEPSEEK_CHAT_TEST_MODEL),
            create_model_client(ANTHROPIC_TEST_MODEL),
        ]
        for client in clients:
            assert isinstance(client, BaseModelClient)

    def test_register_model_client_appends_custom_rule(self):
        """Test that custom provider rules are supported without changing factory code."""
        sentinel_client = MagicMock(spec=BaseModelClient)
        rule = ProviderRule(
            name="custom",
            matches=lambda model: model == "custom-model",
            load=lambda model, logger: sentinel_client,
        )

        register_model_client(rule)
        try:
            assert create_model_client("custom-model") is sentinel_client
        finally:
            _PROVIDER_RULES.remove(rule)

    def test_register_model_client_can_prepend_rule(self):
        """Test that prepended rules take precedence over built-in providers."""
        sentinel_client = MagicMock(spec=BaseModelClient)
        rule = ProviderRule(
            name="custom-openai",
            matches=lambda model: model == "gpt-4.1",
            load=lambda model, logger: sentinel_client,
        )

        register_model_client(rule, prepend=True)
        try:
            assert create_model_client("gpt-4.1") is sentinel_client
        finally:
            _PROVIDER_RULES.remove(rule)
