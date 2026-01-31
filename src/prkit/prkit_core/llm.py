"""
LLM client interface for multiple language model providers.

This module provides a unified interface for interacting with various LLM providers
including OpenAI, Google Gemini, and DeepSeek.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from .logging_config import PRKitLogger


class LLMClient:
    """Base class for LLM client implementations."""

    def __init__(self, model: str, logger=None):
        """
        Initialize LLM client.

        Args:
            model: Model name/identifier
            logger: Optional logger instance
        """
        load_dotenv()
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger if logger else PRKitLogger.get_logger(__name__)

    def chat(self, messages):
        """
        Send chat messages to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Response text from the model

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement .chat()")

    def chat_structured(self, system_prompt, prompt, response_format=None):
        """
        Send structured chat request with format specification.

        Args:
            system_prompt: System instruction prompt
            prompt: User prompt
            response_format: Expected response format specification

        Returns:
            Parsed structured response

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement .chat_structured()")

    @staticmethod
    def from_model(model: str, logger=None):
        """
        Create appropriate LLM client instance based on model name.

        Args:
            model: Model name (e.g., 'gpt-4o', 'gemini-pro', 'deepseek-chat')
            logger: Optional logger instance

        Returns:
            Appropriate LLMClient subclass instance

        Raises:
            ValueError: If model type is not recognized
        """
        model_lower = model.lower()
        if "deepseek" in model_lower:
            return DeepseekModel(model, logger)
        elif model_lower.startswith("o"):
            return OAIReasonModel(model, logger)
        elif model_lower.startswith("gpt"):
            return OAIChatModel(model, logger)
        elif model_lower.startswith("gemini"):
            return GeminiModel(model, logger)
        else:
            raise ValueError(f"Unknown model: {model}")


class DeepseekModel(LLMClient):
    """DeepSeek API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize DeepSeek model client.

        Args:
            model: DeepSeek model name
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.provider = "deepseek"

    def chat(self, messages):
        """
        Send chat messages to DeepSeek API.

        Args:
            messages: List of message dictionaries

        Returns:
            Response text from DeepSeek model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


class OAIReasonModel(LLMClient):
    """OpenAI Reasoning API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize OpenAI Reasoning model client.

        Args:
            model: OpenAI reasoning model name
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        self.client = OpenAI()
        self.provider = "openai"

    def chat(self, messages):
        """
        Send chat messages to OpenAI Reasoning API.

        Args:
            messages: List of message dictionaries

        Returns:
            Response text from OpenAI reasoning model
        """
        response = self.client.responses.create(
            model=self.model, reasoning={"effort": "medium"}, input=messages
        )
        return response.output_text

    def chat_structured(self, system_prompt, prompt, response_format=None):
        """
        Send structured chat request to OpenAI Reasoning API.

        Args:
            system_prompt: System instruction prompt
            prompt: User prompt
            response_format: Expected response format specification

        Returns:
            Parsed structured response
        """
        response = self.client.responses.parse(
            model=self.model, input=prompt, text_format=response_format
        )
        return response.output_parsed


class OAIChatModel(LLMClient):
    """OpenAI Chat API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize OpenAI Chat model client.

        Args:
            model: OpenAI chat model name (e.g., 'gpt-4o')
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        self.client = OpenAI()
        self.provider = "openai"

    def chat(self, messages):
        """
        Send chat messages to OpenAI Chat API.

        Args:
            messages: List of message dictionaries

        Returns:
            Response text from OpenAI chat model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    def chat_structured(self, system_prompt, prompt, response_format=None):
        """
        Send structured chat request to OpenAI Chat API.

        Args:
            system_prompt: System instruction prompt
            prompt: User prompt
            response_format: Expected response format specification

        Returns:
            Parsed structured response

        Raises:
            NotImplementedError: If model doesn't support structured output
        """
        if self.model != "gpt-4o":
            raise NotImplementedError(
                f"Structured Output is not supported for {self.model}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.client.responses.parse(
            model=self.model, input=messages, text_format=response_format
        )
        return response.output_parsed


class GeminiModel(LLMClient):
    """Google Gemini API client implementation."""

    def __init__(self, model: str, logger=None):
        """
        Initialize Gemini model client.

        Args:
            model: Gemini model name (e.g., 'gemini-pro')
            logger: Optional logger instance
        """
        super().__init__(model, logger)
        load_dotenv()
        # The new SDK uses GEMINI_API_KEY, but we support GOOGLE_API_KEY for backward compatibility
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            # Will try to pick up from GEMINI_API_KEY env var automatically
            self.genai_client = genai.Client()
        self.provider = "google"
        self.client = None

    def _convert_openai_messages_to_gemini_contents(self, messages):
        """
        Convert OpenAI-style messages to Gemini-style contents.

        Args:
            messages: List of OpenAI message dictionaries

        Returns:
            Tuple of (system_instruction, contents) for Gemini API
        """
        system_instruction = None
        contents = []

        if messages and messages[0]["role"] == "system":
            system_instruction = messages[0]["content"]
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages

        for msg in conversation_messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        return system_instruction, contents

    def chat(self, messages, *args, **kwargs):
        """
        Send chat messages to Gemini API.

        Args:
            messages: List of message dictionaries
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for generate_content config

        Returns:
            Response text from Gemini model
        """
        system_instruction, contents = self._convert_openai_messages_to_gemini_contents(
            messages
        )

        # Build config with system instruction and any additional kwargs
        config_dict = {}
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        # Merge any additional kwargs into config
        if kwargs:
            config_dict.update(kwargs)

        config = types.GenerateContentConfig(**config_dict) if config_dict else None

        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text

    def chat_structured(self, messages, response_format=None):
        """
        Send structured chat request (not implemented for Gemini).

        Args:
            messages: List of message dictionaries
            response_format: Expected response format specification

        Raises:
            NotImplementedError: Structured output requires function calling
        """
        raise NotImplementedError(
            f"Structured chat for Gemini requires function calling, which is not implemented for {self.model}"
        )
