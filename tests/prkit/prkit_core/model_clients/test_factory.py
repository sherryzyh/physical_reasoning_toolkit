"""
Tests for model client factory function.
"""

from unittest.mock import MagicMock, patch

import pytest

from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_core.model_clients.base import BaseModelClient
from prkit.prkit_core.model_clients.deepseek import DeepseekModel
from prkit.prkit_core.model_clients.gemini import GeminiModel
from prkit.prkit_core.model_clients.ollama import OllamaModel
from prkit.prkit_core.model_clients.openai import OpenAIModel


class TestCreateModelClient:
    """Test cases for create_model_client factory function."""

    def test_create_openai_gpt_4_1(self):
        """Test creating OpenAI gpt-4.1 model."""
        client = create_model_client("gpt-4.1")
        assert isinstance(client, OpenAIModel)
        assert client.model == "gpt-4.1"
        assert client.provider == "openai"

    def test_create_openai_gpt_5_1(self):
        """Test creating OpenAI gpt-5.1 model."""
        client = create_model_client("gpt-5.1")
        assert isinstance(client, OpenAIModel)
        assert client.model == "gpt-5.1"

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
        client = create_model_client("gemini-pro")
        assert isinstance(client, GeminiModel)
        assert client.model == "gemini-pro"
        assert client.provider == "google"

    def test_create_gemini_model_case_insensitive(self):
        """Test creating Gemini model with different case."""
        client = create_model_client("GEMINI-PRO")
        assert isinstance(client, GeminiModel)
        assert client.model == "GEMINI-PRO"

    def test_create_deepseek_model(self):
        """Test creating DeepSeek model."""
        client = create_model_client("deepseek-chat")
        assert isinstance(client, DeepseekModel)
        assert client.model == "deepseek-chat"
        assert client.provider == "deepseek"

    def test_create_deepseek_model_case_insensitive(self):
        """Test creating DeepSeek model with different case."""
        client = create_model_client("DEEPSEEK-CHAT")
        assert isinstance(client, DeepseekModel)
        assert client.model == "DEEPSEEK-CHAT"

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
        client = create_model_client("gpt-5.1", logger=logger)
        assert client.logger == logger

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_create_ollama_model(self, mock_ollama_module):
        """Test creating Ollama model."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = create_model_client("qwen3-vl")
        assert isinstance(client, OllamaModel)
        assert client.model == "qwen3-vl"
        assert client.provider == "ollama"

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_create_ollama_model_with_tag(self, mock_ollama_module):
        """Test creating Ollama model with tag."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        client = create_model_client("qwen3-vl:8b-instruct")
        assert isinstance(client, OllamaModel)
        assert client.model == "qwen3-vl:8b-instruct"

    @patch("prkit.prkit_core.model_clients.ollama.ollama")
    def test_create_qwen_model_variants(self, mock_ollama_module):
        """Test creating different Qwen model variants."""
        mock_client = MagicMock()
        mock_client.list.return_value = []
        mock_ollama_module.Client.return_value = mock_client

        for model_name in ["qwen3-vl", "qwen2.5", "qwen"]:
            client = create_model_client(model_name)
            assert isinstance(client, OllamaModel)

    def test_all_clients_are_base_model_client(self):
        """Test that all created clients are instances of BaseModelClient."""
        clients = [
            create_model_client("gpt-5.1"),
            create_model_client("gemini-pro"),
            create_model_client("deepseek-chat"),
        ]
        for client in clients:
            assert isinstance(client, BaseModelClient)
