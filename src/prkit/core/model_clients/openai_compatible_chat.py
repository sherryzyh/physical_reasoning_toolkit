"""
Shared helpers for OpenAI-compatible chat-completions providers.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

from .base import BaseModelClient
from .openai import prepare_image_url_from_image_path
from .structured_output import normalize_response_format


class OpenAICompatibleChatModel(BaseModelClient):
    """Base class for providers that expose an OpenAI-compatible Chat Completions API."""

    supports_response_format_json_schema = True
    supports_response_format_json_object = False
    provider_name: str = ""
    provider_prefix: str = ""
    api_key_env_var: str = ""
    base_url_env_var: str = ""
    default_base_url: str = ""

    @classmethod
    def normalize_model_name(cls, model: str) -> str:
        """Strip the provider prefix from *model* (e.g. ``'deepseek/foo'`` → ``'foo'``)."""
        if cls.provider_prefix and model.lower().startswith(f"{cls.provider_prefix}/"):
            return model.split("/", 1)[1]
        return model

    def resolve_base_url(self) -> str:
        """Return the API base URL, consulting the env var before falling back to the default."""
        return os.environ.get(self.base_url_env_var, self.default_base_url)

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return extra keyword arguments forwarded to the OpenAI client constructor."""
        return {}

    def __init__(self, model: str, logger: logging.Logger | None = None) -> None:
        normalized_model = self.normalize_model_name(model)
        super().__init__(normalized_model, logger)
        base_url = self.resolve_base_url()
        self.client = OpenAI(
            api_key=os.environ.get(self.api_key_env_var),
            base_url=base_url,
            **self.get_client_kwargs(),
        )
        self.provider = self.provider_name
        self.base_url = base_url

    def _build_message_content(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Build the ``content`` field for an OpenAI-compatible chat message, attaching images when present."""
        if not image_paths:
            return user_prompt

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": prepare_image_url_from_image_path(image_path),
                    },
                }
            )
        return content

    def _build_chat_response_format(
        self,
        response_format: dict[str, Any] | type | None,
    ) -> dict[str, Any] | None:
        """Translate *response_format* to the ``response_format`` parameter for chat completions."""
        if response_format is None:
            return None

        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            return {"type": "json_object"}

        normalized = normalize_response_format(response_format)
        if self.supports_response_format_json_schema:
            json_schema: dict[str, Any] = {
                "name": normalized["name"],
                "schema": normalized["schema"],
                "strict": normalized["strict"],
            }
            if normalized.get("description") is not None:
                json_schema["description"] = normalized["description"]
            return {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        if self.supports_response_format_json_object:
            return {"type": "json_object"}

        self.logger.warning(
            "Structured output (response_format) is not supported by %s model %s. "
            "Ignoring response_format and returning plain text.",
            self.provider_name or self.provider or "unknown",
            self.model,
        )
        return None

    def _structured_prompt_for_chat(
        self,
        user_prompt: str,
        response_format: dict[str, Any] | type | None,
    ) -> str:
        """Optionally augment *user_prompt* with schema instructions; base implementation returns it unchanged."""
        del response_format
        return user_prompt

    def _extract_text_from_chat_completion(self, response: Any) -> str:
        """Extract the plain text content from a chat-completions response object."""
        message = response.choices[0].message
        content = message.content or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_chunks.append(text)
            return "\n".join(chunk for chunk in text_chunks if chunk).strip()
        return str(content)

    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat-completions request and return the model's text response."""
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_message_content(
                        self._structured_prompt_for_chat(user_prompt, response_format),
                        image_paths,
                    ),
                }
            ],
        }

        request_response_format = self._build_chat_response_format(response_format)
        if request_response_format is not None:
            request_params["response_format"] = request_response_format

        max_output_tokens = kwargs.pop("max_output_tokens", None)
        if max_output_tokens is not None and "max_tokens" not in kwargs:
            request_params["max_tokens"] = max_output_tokens

        if kwargs:
            request_params.update(kwargs)

        response = self.client.chat.completions.create(**request_params)
        text = self._extract_text_from_chat_completion(response)
        self.logger.info(f"Response: {text}")
        return text
