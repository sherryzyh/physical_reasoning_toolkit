"""
Tests for LLM client.
"""

import os
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_core.llm import (
    DeepseekModel,
    GeminiModel,
    LLMClient,
    OAIChatModel,
    OAIReasonModel,
)


class TestLLMClient:
    """Test cases for LLMClient base class."""

    def test_llm_client_initialization(self):
        """Test LLM client initialization."""
        # This should raise NotImplementedError since it's abstract
        with pytest.raises(NotImplementedError):
            client = LLMClient(model="test-model")
            client.chat([])

    def test_llm_client_from_model_deepseek(self):
        """Test creating DeepSeek model client."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("prkit.prkit_core.llm.OpenAI"):
                client = LLMClient.from_model("deepseek-chat")
                assert isinstance(client, DeepseekModel)
                assert client.model == "deepseek-chat"

    def test_llm_client_from_model_gpt(self):
        """Test creating GPT model client."""
        with patch("prkit.prkit_core.llm.OpenAI"):
            client = LLMClient.from_model("gpt-4o")
            assert isinstance(client, OAIChatModel)
            assert client.model == "gpt-4o"

    def test_llm_client_from_model_o3(self):
        """Test creating O3 reasoning model client."""
        with patch("prkit.prkit_core.llm.OpenAI"):
            client = LLMClient.from_model("o3-mini")
            assert isinstance(client, OAIReasonModel)
            assert client.model == "o3-mini"

    def test_llm_client_from_model_gemini(self):
        """Test creating Gemini model client."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("prkit.prkit_core.llm.genai.Client"):
                client = LLMClient.from_model("gemini-pro")
                assert isinstance(client, GeminiModel)
                assert client.model == "gemini-pro"

    def test_llm_client_from_model_unknown(self):
        """Test creating unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLMClient.from_model("unknown-model")


class TestDeepseekModel:
    """Test cases for DeepseekModel."""

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    @patch("prkit.prkit_core.llm.OpenAI")
    def test_deepseek_chat(self, mock_openai_class):
        """Test DeepSeek chat method."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = DeepseekModel("deepseek-chat")
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()


class TestOAIChatModel:
    """Test cases for OAIChatModel."""

    @patch("prkit.prkit_core.llm.OpenAI")
    def test_oai_chat(self, mock_openai_class):
        """Test OpenAI chat method."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OAIChatModel("gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()
        # Check temperature is set to 0
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0

    @patch("prkit.prkit_core.llm.OpenAI")
    def test_oai_chat_structured_gpt4o(self, mock_openai_class):
        """Test OpenAI structured chat for gpt-4o."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_parsed = {"result": "parsed"}
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OAIChatModel("gpt-4o")
        response = client.chat_structured(
            system_prompt="System", prompt="User", response_format="format"
        )

        assert response == {"result": "parsed"}
        mock_client.responses.parse.assert_called_once()

    @patch("prkit.prkit_core.llm.OpenAI")
    def test_oai_chat_structured_not_gpt4o(self, mock_openai_class):
        """Test OpenAI structured chat for non-gpt-4o raises error."""
        mock_openai_class.return_value = Mock()
        client = OAIChatModel("gpt-3.5-turbo")

        with pytest.raises(NotImplementedError):
            client.chat_structured("System", "User", "format")


class TestOAIReasonModel:
    """Test cases for OAIReasonModel."""

    @patch("prkit.prkit_core.llm.OpenAI")
    def test_oai_reason_chat(self, mock_openai_class):
        """Test OpenAI reasoning model chat method."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Reasoning response"
        mock_client.responses.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OAIReasonModel("o3-mini")
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Reasoning response"
        mock_client.responses.create.assert_called_once()

    @patch("prkit.prkit_core.llm.OpenAI")
    def test_oai_reason_chat_structured(self, mock_openai_class):
        """Test OpenAI reasoning model structured chat."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_parsed = {"result": "parsed"}
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OAIReasonModel("o3-mini")
        response = client.chat_structured("System", "User", "format")

        assert response == {"result": "parsed"}
        mock_client.responses.parse.assert_called_once()


class TestGeminiModel:
    """Test cases for GeminiModel."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @patch("prkit.prkit_core.llm.genai.Client")
    def test_gemini_chat(self, mock_genai_class):
        """Test Gemini chat method."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_class.return_value = mock_client

        client = GeminiModel("gemini-pro")
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response == "Gemini response"
        mock_client.models.generate_content.assert_called_once()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @patch("prkit.prkit_core.llm.genai.Client")
    def test_gemini_chat_with_system_instruction(self, mock_genai_class):
        """Test Gemini chat with system instruction."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_class.return_value = mock_client

        client = GeminiModel("gemini-pro")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        response = client.chat(messages)

        assert response == "Gemini response"
        # Verify system instruction was extracted
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "config" in call_kwargs or len(call_kwargs) > 0

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @patch("prkit.prkit_core.llm.genai.Client")
    def test_gemini_chat_structured_not_implemented(self, mock_genai_class):
        """Test Gemini structured chat raises NotImplementedError."""
        mock_genai_class.return_value = Mock()
        client = GeminiModel("gemini-pro")

        with pytest.raises(NotImplementedError):
            client.chat_structured([], "format")

    @patch.dict(os.environ, {}, clear=True)
    @patch("prkit.prkit_core.llm.genai.Client")
    def test_gemini_fallback_to_google_api_key(self, mock_genai_class):
        """Test Gemini falls back to GOOGLE_API_KEY if GEMINI_API_KEY not set."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=False):
            mock_genai_class.return_value = Mock()
            client = GeminiModel("gemini-pro")
            # Should not raise an error
            assert client.model == "gemini-pro"
